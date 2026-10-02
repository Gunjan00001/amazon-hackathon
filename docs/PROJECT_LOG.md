# Project Log — Business Entity Resolution

Chronological record of what was run, on what data, with measured numbers and timings.
All commands run from the repository root with `PYTHONPATH=code/business_entity_resolution/src`
and the Python 3.12 venv (`.venv\Scripts\python.exe`).

## 2026-09-25 / 26 — environment and tooling

| Step | Command | Result |
|---|---|---|
| Create venv | `uv venv .venv --python 3.12` | Python 3.12.14 |
| Install deps | `uv pip install pandas pyarrow numpy scikit-learn lightgbm xgboost rapidfuzz duckdb indic-transliteration jellyfish pytest` | pinned in `requirements.txt` |
| Dataset EDA | `.venv\Scripts\python.exe tools\eda_dataset.py` | `tools/eda_stats.json`, `DATA/student_resource/dataset/DATASET.md` |
| GBDT benchmark | `.venv\Scripts\python.exe tools\bench_gbdt.py` | `tools/bench_results.json` |
| Graphify init | graphify skill on project docs | `graphify-out/` (graph.html, GRAPH_REPORT.md, graph.json) |
| Graphify refresh | graphify skill over docs + code (53 files) | `graphify-out/` 265 nodes / 563 edges, 40.5x token reduction |

## Pipeline runs (measured)

| Stage | Split | Command | Output | Notes |
|---|---|---|---|---|
| Prepare | both | `ber.cli prepare` | `DATA/processed/*.parquet` | 24.2M rows, ~40 min; counts match EDA exactly |
| Block | train | `ber.cli block --split train` | 304,759,423 candidates | keys cached in `DATA/keys/` |
| Block | test | `ber.cli block --split test` | 250,607,135 candidates | |
| Audit | train | `ber.cli audit --split train` | recall 0.8115 (US 0.868 / India 0.727), reduction 29.5 | ~50 s, DuckDB |
| Features | train | `ber.cli features --split train --workers 8 --combine` | 30,494,378 × 36 (`DATA/features/train.parquet`) | 4:1 sampled negatives |
| Train | train | `ber.cli train` | `models/lgbm.txt` (610 trees) | val macro F0.5 0.9803 (4:1, optimistic) |
| Features | test | `ber.cli features --split test --workers 8` | 250,607,135 rows in 16 parts | parallel, ~ minutes |
| Predict | test | `ber.cli predict --split test --one-to-one --threshold 0.925` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` | 1,732,544 rows each |
| Evaluate (4:1) | train | `ber.cli evaluate` | `DATA/reports/eval_marks.json` | threshold-only 0.9803 / one-to-one 0.9807 |
| Validation | train | `ber.cli validation --workers 8` | `DATA/reports/eval_full_candidates.json` | **0.8488** on full candidates, held-out S1 |
| LOO | train | `ber.cli loo` | `DATA/reports/eval_loo.json` | unseen-country proxy: 0.668 / 0.804 |
| Validate | test | `validate_submission.py ...` | `PASS` | format gate |

## Final submission

- `output/matching_results.tsv` — 1,732,544 rows (200,982 empty / 1,533,562 non-empty).
- `output/candidate_pairs.tsv` — 1,732,544 rows (554 empty).
- 5,071,867 candidate pairs above threshold 0.925.
- Package staged at `dist/AA.._submission/`; zip `dist/AA..__submission.zip`.
- **Portal submission 26 Sep 2026, 02:43 PM IST → public leaderboard macro F0.5 = 0.811 (Evaluated).**

## Corrections made during the run

- Threshold was first tuned on the 4:1 sampled split (0.675) and later correctly re-tuned on the
  full candidate distribution to **0.925**; all submitted outputs use 0.925.
- The 4:1 number (0.98) is optimistic and must not be quoted as the leaderboard score; see
  `RESULTS.md`.

## 2026-09-26 — Colab GPU probe and M2 plan

- Connected to Colab via the MCP tool (`colab_open_colab_browser_connection` → true) and ran a probe
  in a scratch cell: **Tesla T4, 15,360 MiB VRAM, compute capability 7.5, CUDA available,
  `torch 2.11.0+cu128` preinstalled, 2 vCPU, 13.6 GB RAM (free tier)**. fp16 supported; no bf16.
- Wrote the next-milestone design and plan:
  `docs/superpowers/specs/2026-09-26-precision-colab-design.md` and
  `docs/superpowers/plans/2026-09-26-precision-colab.md`.
- Split decision: phases 0–1 and 5 local (CPU/data-bound, faster locally); phases 2–3 on Colab T4
  (embeddings + optional cross-encoder); phase 4 (MinHash-LSH re-block) conditional.

## 2026-09-26 — M2 Task 1: oracle diagnostic

- `.venv\Scripts\python.exe -m ber.cli diagnose` → `DATA/reports/eval_oracle.json`.
- Oracle macro F0.5 **0.9122** (US 0.9468 / India 0.8602); blocking pair recall 0.8142; 4.28% of
  truth-bearing entities have zero found candidates. Headroom over the 0.8488 baseline is **+0.0634**
  (> 0.05) → matcher precision/recall is the lever; re-blocking stays conditional.

## 2026-09-26 — M2 Task 2: char n-gram TF-IDF cosine features

- Added `name_char3_cos`, `name_roman_char3_cos`, `addr_char3_cos` (char-3 TF-IDF cosine) and appended
  them to `FEATURE_ORDER` (33 → 36 features).
- Vectorizer fit once on a 299,997-doc train sample (seed 42), persisted at
  `DATA/processed/char3_vectorizer.pkl`; `DATA/processed/char3_vocab.json` records n_vocab 21,015.
- `ber.cli features --split train --workers 8 --combine` → `DATA/features/train.parquet`
  30,494,378 × 39 (`{'positives': 6,198,915, 'negatives': 24,295,463}`).
- Regenerated held-out parts: `DATA/tmp/valfull_features` 60,912,676 rows × 38, 16 parts.
- Verified the three new columns have 0 nulls and range [0, 1].
- Test-split features deliberately deferred to Task 10 (embeddings added in Task 7 would invalidate them).

## 2026-09-26 — M2 Task 3: per-country thresholds + singleton calibration

- Added `ber/calibration.py`: `tune_per_country` (per-country sweep with global fallback for sparse or
  single-class countries), `singleton_decision` (suppress a Source 1 group when `max(prob) < tau`),
  `tune_singleton_tau`, `apply_calibration`, `save_calibration`/`load_calibration`.
- `models/threshold.json` schema gains `by_country`, `singleton_tau`; `load_calibration` stays
  backward compatible with the old `{"global", "use_one_to_one"}` file.
- Added `ber.cli calibrate` (`run_calibration`) which tunes on the held-out 20% and reports to
  `DATA/reports/calibration.json`. Not run yet — baseline `models/threshold.json` is preserved until
  the Task 4 gate.
- Tests: `tests/test_calibration.py` (6 tests); full suite 41 passed.

## 2026-09-26 — M2 Task 4: retrain + held-out gate (adopted, v1.4.1)

- `ber.cli train` → `models/lgbm.txt` best_iteration 523; 4:1 val macro F0.5 0.9824 @ 0.7 (+589 s).
  `threshold.json` now written with the `by_country`/`singleton_tau` schema (provisional, empty).
- `ber.cli validation --workers 8` → `DATA/reports/eval_full_candidates.json`: one-to-one
  **0.85742** @ 0.925 (US 0.8978 / India 0.7968), ceiling 0.8142 (char-3 alone beat baseline 0.8488).
- `ber.cli calibrate` (tuned on 25% of held-out full-candidate preds, scored on all held-out S1) →
  `models/threshold.json` global 0.95, by country India 0.925 / US 0.95, `singleton_tau` 0.30;
  `DATA/reports/eval_calibrated.json` chosen_macro_f05 **0.85772** (US 0.8983 / India 0.7968,
  CI 0.8570–0.8585).
- Gate: 0.85772 > 0.8488 and India 0.7968 > 0.788 → **adopted**; `git tag -a 1.4.1`
  (`1.4.0` was already used for the M2 plan commit).

## 2026-09-26 — M2 Task 5: Colab export/import (local)

- Added `ber/colab_io.py`: `export_for_colab` → `DATA/colab_in/entities.parquet` (unique entity texts
  from all processed sources) and `pairs_{split}.parquet`; `import_cosine` reads
  `DATA/colab_out/cosine_{split}.parquet` and returns `(s1_id, cand_id, emb_name_cos, emb_addr_cos)`.
- Added `ber.cli colab-export` / `colab-import`.
- Export: entities 24,229,173 rows / 1,066.7 MB; pairs train 212.3 MB, valfull 354.8 MB,
  test 1,452.1 MB (250.6M rows).
- `tests/test_colab_io.py` round-trip (export → cosine → import → key merge); full suite 42 passed.
- `.gitignore` now excludes `DATA/colab_in/` and `DATA/colab_out/` (never commit data).

## 2026-09-26 — M2 Tasks 6–7: Colab embeddings (tried, NOT adopted)

- Task 6, Colab free T4, `notebooks/colab_embeddings.ipynb`: encoded all 24,229,173 entities ×
  {name, addr} with `intfloat/multilingual-e5-small` (fp16) in ~110 min; produced
  `cosine_train.parquet` (30,494,378 rows, 273 MB) and `cosine_valfull.parquet` (60,912,676 rows,
  487 MB), no NaNs.
- Environment constraint: free-Colab random disk reads over the 18 GB fp16 embedding files were
  ~37× amplified (~60 MB/s, ~28 GB read for one 250k-pair batch) and unusable. Worked around with a
  single sequential pass applying a seeded random projection 384 → 64 dims (cosine-preserving) into
  RAM. This is a deviation from the plan's 384-dim cosine and is the likely cause of the small loss.
- Task 7: added `ber.cli colab-merge` to attach `emb_name_cos`/`emb_addr_cos` to train + valfull
  features; retrained (469 trees, 4:1 val macro F0.5 0.9823); `ber.cli validation --workers 8`
  (one-to-one 0.8568) and `ber.cli calibrate` (calibrated 0.8570).
- Gate FAILED: 0.8570 < Phase-1 0.8577 (US 0.8977 vs 0.8983, India 0.7961 vs 0.7968). Not adopted;
  `models/` reverted to the 1.4.1 baseline, `FEATURE_ORDER` restored to the 36 char-3 feature set.
- The export/import/merge tooling and the notebook are kept for future runs (e.g. full 384-dim on a
  host with more RAM/local disk).

## 2026-09-26 — M2 Task 10: rebuild outputs with the adopted 1.4.1 model (v2.0.0)

- `predict.py` now loads `models/threshold.json` calibration and applies per-country thresholds +
  `singleton_tau` + one-to-one for the test split (predict floor = min(global, country thresholds)).
  New tests `tests/test_predict.py` (per-country drop, singleton tau); full suite 45 passed.
- `ber.cli features --split test --workers 8` → `DATA/tmp/test_features` 250,607,135 rows, 16 parts.
- `ber.cli predict --split test --one-to-one` → `output/matching_results.tsv` (1,732,544 rows;
  212,118 singletons) + `output/candidate_pairs.tsv` (554 empty); 5,063,955 pairs ≥ 0.925 floor.
- Validator (from repo root): **PASS** — `DATA/reports/validate_char3.log`.
- Package refreshed: synced `src/ber/*.py` into `dist/AA.._submission/`, updated outputs, rebuilt
  `dist/AA..__submission.zip` (1,379.6 MB) with Python `zipfile` (PowerShell 2 GB limit, F18),
  staged `dist/leaderboard_upload/matching_results.tsv` and `submission/matching_results.tsv`.
- Fixed an LF/CRLF model-load failure while reverting `models/` (F14; added `.gitattributes`).
- Portal upload intentionally **not** performed by the agent.

## 2026-09-27 — E0 baseline freeze (execution plan started)

- Plan of record: `C:\Users\Gunjan\.opencode\plan\2026-09-27-ber-execution-plan.md` (E0–E11).
- Confirmed decisions: GPU = Colab T4 first / Kaggle T4x2 fallback; raw text via export-time TSV join;
  single shared e5 prefix; clean `DATA/tmp` before E4.
- `.venv\Scripts\python.exe -m pytest -q` → **45 passed**.
- Snapshot `code/business_entity_resolution/models/` → `models_baseline_2.0.0/`; `lgbm.txt` SHA256
  `808643FB08FE26BF30982ABB6F33E9993AAFFF4C8D1375D7231217F2E4B70DC8` identical to source.
- Environment: git tags 0.1.0 … **2.0.0**, working tree clean, D: ~96.3 GB free, Python 3.12.14.
- Colab MCP browser connection established (result `true`); live session is fresh/empty.
- **Stale-artifact note:** `DATA/reports/eval_calibrated.json` (0.8570, `singleton_tau` 0.93) is the
  rejected-embedding artifact (D16) and does **not** match the adopted `models/threshold.json`
  (global 0.95, India 0.925/US 0.95, `singleton_tau` 0.30, held-out 0.8577). Also `DATA/features/train.parquet`
  and `DATA/tmp/valfull_features` still carry the rejected `emb_name_cos`/`emb_addr_cos` columns, and
  `DATA/tmp/valfull_pred` was produced by the embedding model. Regenerating `valfull_pred` with the adopted
  1.4.1 booster to restore a trustworthy E2 baseline (does not modify `models/`).

## 2026-09-27 — E2 diagnostics (baseline "before" point)

- `ber.cli audit --split train` reproduced exactly: recall **0.81155** (US 0.8681 / India 0.7269),
  reduction 29.53, candidates/S1 138.10, 304,759,423 candidates, 6,198,915 / 7,638,365 truth pairs.
- `ber.cli diagnose` reproduced: oracle macro F0.5 **0.91216** (US 0.9468 / India 0.8602), held-out
  pair recall 0.81422, entity recall 0.81393, zero-candidate S1 17,785 / 415,317 (4.28%).
- `ber.cli validation --workers 8` regenerated `valfull_pred` with the adopted 1.4.1 booster (does not
  touch `models/`): uncalibrated one-to-one **0.85742** @ 0.925 (US 0.8978 / India 0.7968). Read-only
  `score_calibration` with the adopted `threshold.json` → **0.85772** (US 0.8983 / India 0.7968, tau 0.30).
  Gate PASSED. Written to `DATA/reports/exp_E2.json`.

## 2026-09-27 — E1 local prep (raw-text export + streaming Colab notebook)

- `ber/colab_io.py::export_entities` now joins raw `business_name`/`business_address` from the source
  TSVs on `entity_id` (decision D-2), keeping normalized columns; `ENTITY_COLUMNS` →
  `[entity_id, name_raw, addr_raw, name_norm, name_roman, addr_norm]`. `COSINE_COLUMNS`/`EMB_FEATURES`
  renamed to `name_e5_cos`, `addr_e5_cos`, `entity_e5_cos`.
- `tests/test_colab_io.py` updated for the new schema/columns (raw join + e5 cosine names); tests pass.
- `ber.cli colab-export` (splits train+valfull): `DATA/colab_in/entities.parquet` 24,229,173 rows,
  **1,949 MB** (raw text incl. non-Latin script), 32 s; `pairs_train.parquet` 30,494,378 rows,
  `pairs_valfull.parquet` 60,912,676 rows.
- Colab live probe: 113 GB disk / 66 GB free, **12 GB RAM**, T4 15 GB, 2 vCPU. Full 3-field fp16 =
  55.8 GB (barely fits disk) and cannot be gathered in 12 GB RAM — the D16 failure mode. Rewrote
  `notebooks/colab_embeddings.ipynb` to a field-at-a-time streaming design: encode one field to a
  18.6 GB memmap, hold all Source-1 embeddings in RAM (~3 GB), stream candidates sequentially
  (`ORDER BY e2.row`), then delete; peak disk ~20 GB. Shared `"passage: "` prefix (D-3).
- Blocked on: placing `entities.parquet` + `pairs_{train,valfull}.parquet` (~2.4 GB) into the Colab
  session (agent cannot upload >100 MB; needs Files panel / Drive by a human).

## 2026-09-27 — E1 on Kaggle, fully automated (no manual uploads)

- Discovered no Kaggle MCP tool, but an authenticated Kaggle CLI is present (`KAGGLE_API_TOKEN`,
  user `gunjanpal001`, auth_method ACCESS_TOKEN). Switched E1's GPU stage from manual Colab to
  **automated Kaggle** per the user's request.
- Reused the existing private dataset `gunjanpal001/amz-er-2026-raw` (the 7 raw TSVs) so entities
  (with raw `business_name`/`business_address`) are built **on Kaggle**, avoiding the 1.86 GB upload.
  Uploaded only a new private dataset `gunjanpal001/amz-er-2026-e1-pairs` (pairs_train 212 MB +
  pairs_valfull 355 MB).
- `notebooks/e1_runner.py` (env-agnostic, shared by Kaggle kernel + Colab notebook): auto-detects
  Kaggle (`/kaggle/input`) vs Colab, builds `entities.parquet` from raw TSVs, encodes three fields
  (name/addr/entity) with `intfloat/multilingual-e5-small` revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`, shared `"passage: "` prefix, and computes per-pair
  cosines via numpy argsort + sequential memmap reads.
- Bugs caught by a 200k-entity smoke run (kernel `amz-er-e1-smoke`):
  - dataset mount path is `/kaggle/input/datasets/<owner>/<slug>` (not `/kaggle/input/<slug>`) →
    input discovery now walks `/kaggle/input`; dataset IDs unchanged.
  - empty `business_address` → `None` → `TypeError` in text concat → coalesce to `""`.
  - merge assumed all three fields → made field-aware.
  - kaggle CLI log download broke on Windows charmap → installed `kaggle` into the venv and ran it
    with `PYTHONUTF8=1`.
- Smoke (200k entities, 3 fields): entities 24,229,173 built from TSVs; encodes name 24.5 s /
  addr 34.2 s / entity 51.9 s (~6-8k texts/s); merge + zip OK. Full-run estimate ~3-3.5 h.
- Full run launched: kernel **`gunjanpal001/amz-er-e1-embeddings`** (script, private, GPU T4x2,
  internet on). Outputs: `/kaggle/working/cosine_{train,valfull}.parquet` + `cosine_e5_out.zip`.
  Smoke kernel `amz-er-e1-smoke` kept as a reproduction/testing artifact.

## 2026-09-27 — second Kaggle account (`gunjanpal`) + RTX Pro 6000 for E3

- User requested use of `kaggle.com/code/gunjanpal/the-gpu-one`, which is a bare template notebook
  configured with `machine_shape: "NvidiaRtxPro6000"` (Blackwell, ~96 GB VRAM) — far better than the
  T4x2. It belongs to a second account, `gunjanpal`; the original token is `gunjanpal001` (403 on the
  other account). User supplied a `gunjanpal` API token; it is stored at
  `%TEMP%\opencode\kg_gunjanpal.txt` (not in the repo) and applied per-command via `KAGGLE_API_TOKEN`.
  `gunjanpal001` remains the token for the in-flight E1 kernel.
- `gunjanpal` had no datasets, and private datasets cannot be shared across accounts (and the challenge
  data must not be made public, RULES §2), so the inputs were re-created under `gunjanpal`:
  `gunjanpal/amz-er-2026-raw` (7 TSVs, flat) and `gunjanpal/amz-er-2026-e1-valfull`
  (`valfull_pairs.parquet`, `valfull_s1_ids.parquet`).
- Kaggle CLI skips subfolders on `datasets create` (needs `--dir-mode`); fixed by uploading the TSVs
  flat and making both runners locate TSVs individually via `_find_file` (layout-agnostic).
- E3 runner (`notebooks/e3_kaggle.py`) staged for `gunjanpal` with `machine_shape: NvidiaRtxPro6000`:
  train-only encode (12.5M entities) → IVF-PQ over Source-2/3 → held-out Source-1 query at K=2000 →
  per-truth-pair min ANN rank. Recall/oracle/cost curves are computed locally.






## 2026-09-27 — E1g + E3 on RTX Pro 6000 (Blackwell, interactive)

- Blackwell reality: machine_shape is ignored for API/CLI runs (always T4x2); the RTX Pro 6000
  (~96-102 GB) is only offered in interactive browser sessions with Internet OFF. Built offline
  dataset gunjanpal/amz-er-2026-offline (faiss wheel + e5 model) and consolidated inputs into
  gunjanpal/amz-er-2026-all; ran gunjanpal/blackwell-001 interactively (wall 7550 s).
- E1g: 24,229,173 entities x 3 fields encoded (~27k texts/s); per-pair cosine for train (30,494,378)
  and valfull (60,912,676).
- E1 gate: merged e5 cosines; retrained (436 trees); full-candidate held-out uncalibrated 0.85939
  (US 0.8989 / India 0.8001); calibrated **0.85900** (US 0.8994 / India 0.7984) vs baseline 0.85772
  -> **PROMOTED** (FEATURE_ORDER 39 features; commit 11b88bc).
- E3 ANN ceiling (DATA/reports/exp_E3_curves.json): e5-small full-384 IVF-PQ over train S2/S3
  (10.32M), held-out S1 query K=2000, union with lexical:
  K=50 recall 0.9496 / oracle 0.9810; K=2000 recall **0.9700** / oracle **0.9892** (vs lexical-only
  0.8142 / 0.9122). Channel attribution over 283,130 lexical-missed truth pairs: addr 200,724,
  entity 165,818, name 87,692, any 237,473, multiple 156,141, none 45,657.
- Conclusion: E1 feature gain is small (lexical ceiling binds); the E3 ANN union lifts the oracle
  0.9122 -> 0.9892. 0.99 not reachable at K<=2000 (recall 0.97). Next: E4 production multi-channel
  candidates + adaptive K, then matcher on the new candidate set.


## 2026-09-27 — Final submission (safe path) prepared

- The RTX Pro 6000 re-run (E4) was not available in time, so the final submission uses the adopted
  1.4.1 baseline model (36 features; held-out 0.8577) on the existing lexical candidates. The E1 model
  (39 features, held-out 0.8590) could not be applied to test (test features lack the e5 cosine columns
  and no GPU re-run was possible); it is preserved at models_e1/ and in git (11b88bc). models/ was
  restored from models_baseline_2.0.0/.
- Regenerated test submission with baseline calibration (global 0.95, India 0.925 / US 0.95,
  singleton_tau 0.30, one-to-one) via er.cli predict --split test --one-to-one --reuse-predictions:
  matching_results.tsv 1,732,544 rows, 212,118 empty (singletons), 1,520,426 non-empty.
- Validator: **PASS** (0 blocking issues; 554 empty candidate rows).
- Package refreshed: dist/AA.._submission/ (src synced), dist/AA..__submission.zip (1,379.6 MB,
  Python zipfile/ZIP64), submission/matching_results.tsv, dist/leaderboard_upload/matching_results.tsv.
- E4 Blackwell run reached ield ANN done entity then OOM'd on the ~2.6B-row ANN concat; a streaming
  rewrite was pushed (lackwell-001 v11) but not re-run for time. E3 measurement stands: the e5 ANN
  union lifts the oracle 0.9122 -> 0.9892 at K=2000.


## 2026-09-29 - Repository archival / handover

- Prepared the repo for the local working folder to be deleted. Confirmed `main` was already in
  sync with `origin/main` (93 tracked files; tags `0.1.0`-`2.1.0`, plus `submission-0.811`).
- Measured the full folder: **~80 GB** - `DATA/` 62.8 GB, `SUBMIT/` 7.6 GB, `dist/` 4.6 GB,
  `output/` 3.2 GB, `kaggle Downloaded/` 0.9 GB, `.venv/` 0.5 GB. GitHub cannot host this
  (100 MB plain-git cap, 2 GB Git-LFS per-file cap, ~1 GB free LFS quota), and `RULES.md` section 2.2
  forbids committing the raw `DATA/` TSVs.
- Added **`docs/HANDOVER.md`** (restore guide: clone+LFS, Python 3.12 env, raw-data sources,
  rebuild commands, results) and **`docs/ARCHIVE_MANIFEST.md`** (per-directory inventory with byte
  sizes, git status, regeneration paths, LFS object list).
- Tracked the large submission artifacts with **Git LFS**: `SUBMIT/AA.._submission.zip` (1.28 GiB)
  and the byte-exact split `SUBMIT/candidate_pairs.tsv.part_{00,01}` (1.51 GiB each); the 3.03 GiB
  `candidate_pairs.tsv` itself exceeds GitHub's 2 GB LFS cap, so only the split parts are stored.
- Updated the graphify knowledge graph to include this handover documentation.

## 2026-10-02 - Final submission package rebuild (release asset)

- Audited the released `AA._submission.zip` against PROBLEM_STATEMENT.md section 9 and found three
  packaging defects: the methodology header was still a placeholder (`AA..><`, `[members]`,
  `[Date]`); `code/business_entity_resolution/README.md` was the repo root README instead of the
  pipeline run instructions; and `models/` was missing. The packaged code README also omitted
  `calibrate`, so following it could not reproduce the adopted v2.0.0 output.
- Filled the methodology header (team `AA..`; members Gunjan Pal, Aman Gandotra, Anushka, Ibrahim;
  submission date 2026-09-26) and corrected the code README run order (`validation` then
  `calibrate`, no `--threshold` override) plus the adopted 0.8577 score.
- Rebuilt `AA.._submission.zip` from the current `main` code with spec-compliant root-level entries
  `output/`, `code/`, `Documentation_template.md` (no wrapper folder), including
  `code/business_entity_resolution/models/`. The new zip is 1,377,402,398 bytes (33 entries;
  `candidate_pairs.tsv` sha256 `a259aad1e02ea08453366f980fe2fcc92cebb2566285a1759739eb8fcf2ed0cf`).
  Re-uploaded to release `submission-0.811`; GitHub stores the asset as `AA._submission.zip`
  (the double dot is collapsed on upload) and the previous asset was overwritten in place.
- Verified `output/matching_results.tsv` is byte-identical to the scored release file
  (sha256 `2363074e279ea7f75489c44ebedffd006864730c37dc79c6a7c5886c68f51236`) and structurally valid:
  1,732,544 rows, 212,118 singletons, no duplicate rows or IDs, `S2-`/`S3-` only.
- Updated the Git-LFS object for `SUBMIT/AA.._submission.zip` to the corrected package
  (oid `ff391b6c7094f08b0e297c0d6d15fd974adf2aec97ca19d4ddffd644b971bf90`, 1,377,402,398 bytes),
  so `git lfs pull` now yields the corrected zip as well as the release asset.