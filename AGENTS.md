# AGENTS.md — working notes for this repository

Amazon ML Challenge 2026: Business Entity Resolution. Read `RULES.md` (binding rules), `PROBLEM_STATEMENT.md` (full spec), and `DATA/student_resource/dataset/DATASET.md` (measured data facts) before changing code.

Keep documentation and logs current as you work:
- `docs/PROJECT_LOG.md` — chronological run log (commands, timings, numbers).
- `docs/DECISIONS.md` — architecture decisions and reasoning.
- `docs/FAILURES_AND_FIXES.md` — failures and resolutions; add an entry for every non-trivial bug.
- `docs/RESULTS.md` — metrics; update when a new believable evaluation is produced.

## Environment

- ML code runs on **Python 3.12** via the project venv: `.venv\Scripts\python.exe` (created with `uv`, deps in `code/business_entity_resolution/requirements.txt`).
- Graphify runs on system Python 3.14 (`python`, not the venv). Keep the environments separate.
- Colab (free **Tesla T4, 15.6 GB, fp16 only**, 2 vCPU / 13.6 GB RAM) is used via the Colab MCP tools for
  GPU stages only; CPU stages and all outputs stay local. See
  `docs/superpowers/plans/2026-09-26-precision-colab.md` and `RULES.md` §6.
- Harnesses: `opencode` and `mcode` are both agent harnesses used on this repo; keep instructions harness-agnostic.
- Tests: `.venv\Scripts\python.exe -m pytest -q` from the repo root (45 tests). `conftest.py` puts `src/` on `sys.path`.
- CLI: set `PYTHONPATH=code/business_entity_resolution/src` then run `python -m ber.cli <command> --config code/business_entity_resolution/config.json`. Commands: `prepare`, `block`, `audit`, `features`, `train`, `calibrate`, `predict`, `evaluate`, `validation`, `loo`, `diagnose`, `colab-export`, `colab-import`, `colab-merge`, `all`.

## Pipeline data contracts (all on disk, parquet/JSON)

- `DATA/processed/{split}_source{1,2,3}.parquet` — normalized records from `prepare`. Key columns: `entity_id`, `name_norm`, `name_fold`, `name_roman`, `name_tokens`, `name_idf_tokens`, `name_script`, `name_stripped`, `suffix_class`, `addr_norm`, `addr_raw_missing`, `house_no`, `street_tokens`, `postal`, `state_key`, `landmark_flag`.
- `DATA/processed/train_ground_truth.parquet` — official columns `source1_entity_id`, `matched_entity_ids`.
- `DATA/reports/{split}_token_idf.json` — `{"n", "min_idf", "idf": {token: idf}}` for every token (used for pass-3 rarity and top-4 rarest ranking).
- `DATA/keys/{split}_s1_keys.parquet`, `{split}_source{2,3}_keys.parquet` — blocking keys `(entity_id, pass_id, key, block_score)`.
- `DATA/candidates/{split}_candidates.parquet` — `(s1_id, cand_id, pass_id, block_score, is_s2)`, one row per final candidate pair after dedup and the per-S1 cap. This is exactly the set the matcher scores.

## Blocking contracts

- Passes: 1 exact normalized name; 3 rare-token exact; 4 `house_no|street[0][:5]`; 5 `postal|first name token`; 6 fallback `state|name[:3]`; 7 name-token prefix-5; 8 street-token prefix-5; 9 token pairs over the 4 rarest tokens; 10 token triples over the 3 rarest tokens.
- Per-pass candidate-block caps live in `config.json` under `pass_caps` (keys are strings). Per-S1 cap is `cap` (200). `max_block` is not used by the current join; `_valid_sql` uses the pass caps.
- Tuning loop: edit `pass_caps` → `ber.cli block --split <train|test>` (keys are cached; deleting `DATA/keys/` forces regeneration, ~20 min for train) → `ber.cli audit --split train` (~50 s) → read recall. Current operating point: train recall **0.8115** (US 0.868, India 0.727), 304.8M candidates, reduction 29.5.
- Never run `block` and `audit` concurrently with another heavy job (RAM/disk contention; DuckDB uses up to 12 GB and a 50-60 GiB temp directory).

## Audit contract

- Entry point: `ber.audit.run_audit(cfg, split)`, DuckDB end-to-end (never materializes candidate tuples). `audit_candidates()` is a small-scale pandas reference used only by tests; do not delete it and do not overwrite `run_audit`.
- Report path: `DATA/reports/{split}_blocking_audit.json`. Keys: `recall`, `recall_by_country`, `reduction_ratio`, `candidates_per_s1_mean`, `singleton_candidates_mean`, `per_pass_recall`, `truth_pairs`, `found_pairs`.
- `test_run_audit_matches_pandas_reference` asserts the DuckDB path equals the pandas reference field-by-field.

## Matcher + calibration contract

- `ber/features.py::FEATURE_ORDER` (36 features) is the single source of truth for model inputs; it
  ends with `name_char3_cos`, `name_roman_char3_cos`, `addr_char3_cos` (char-3 TF-IDF cosine).
  `models/feature_list.json` mirrors it and `predict` reads that file.
- Char vectorizer: fit once on a 299,997-doc train sample (seed 42), persisted at
  `DATA/processed/char3_vectorizer.pkl` (`char3_vocab.json` records n_vocab/sha1).
  `ensure_char_vectorizer` refits only if the pickle is missing; `min_df` drops to 1 for tiny corpora.
- `ber/calibration.py`: `tune_per_country` (per-country sweep, global fallback for sparse/single-class
  countries) + `singleton_decision` (`max(prob) < tau` → empty). `models/threshold.json` schema:
  `{"global", "by_country", "singleton_tau", "use_one_to_one"}`; `load_calibration` stays backward
  compatible with the old `{"global","use_one_to_one"}` file.
- Current operating point (v2.0.0): held-out full-candidate macro F0.5 **0.8577** (US 0.8983 / India
  0.7968); thresholds global 0.95, India 0.925 / US 0.95, `singleton_tau` 0.30, one-to-one.
- Calibrate on **full-candidate held-out predictions** (`DATA/tmp/valfull_pred`), never the 4:1 sample
  (D8). `ber.cli calibrate` requires `ber.cli validation` to have produced those predictions first.
- `predict` applies the calibration by default: floor = `min(global, country thresholds)`, then
  per-country threshold + singleton tau + one-to-one in `_write_tsv`.
- Oracle headroom diagnostic: `ber.cli diagnose` → `DATA/reports/eval_oracle.json` (0.9122).

## Colab GPU stages

- Optional GPU work (`notebooks/colab_embeddings.ipynb`) via the Colab MCP. Inputs are exported by
  `ber.cli colab-export` to `DATA/colab_in/`; per-pair cosines return to `DATA/colab_out/` and are
  merged with `ber.cli colab-merge`. Free-Colab `files.upload()` caps at ~100 MB (use Drive/Files
  panel); random disk I/O over the full embedding files is unusable (F15). Embeddings were tested and
  **not adopted** (D16); the test split was regenerated without them.

## Gotchas already learned

- `DATA` and `data` collide on Windows; intermediates land in the existing `DATA/` directory even though `config.json` says `data`.
- Parquet list columns come back from `iter_batches` as numpy arrays, not Python lists.
- Per-token Metaphone blocking explodes (75 GiB temp); it was removed. Do not reintroduce it.
- Commas are data: always `sep="\t"`, `keep_default_na=False`.
- Generated datasets/keys/candidates are git-ignored; never commit them. `DATA/colab_in/` and
  `DATA/colab_out/` are git-ignored too.
- The LightGBM text model (`models/lgbm.txt`) must stay **LF-only**; CRLF breaks its parser (F14).
  `.gitattributes` marks it `-text` so git never converts it.
- Free-Colab `files.upload()` truncates at ~100 MB; move large inputs via Drive or the Files panel (F17).
