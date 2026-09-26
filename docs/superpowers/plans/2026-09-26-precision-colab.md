# Precision Lift + Colab Embeddings — Implementation Plan (M2)

> **For the executing agent:** this plan is self-contained. Work from `D:\Amazon project`. Use the
> Python 3.12 venv `.venv\Scripts\python.exe`; export `PYTHONPATH=code/business_entity_resolution/src`
> for every local command. Use the Colab MCP tools for the GPU phases. Do NOT change the model
> scoring formula, do NOT commit `DATA/`/`output/`/`dist/`, and keep seed 42.

**Goal:** raise held-out full-candidate macro F0.5 above the current 0.8488 and the leaderboard above
0.811 by improving matcher precision (features + calibration + GPU rerank), conditionally raising the
blocking recall ceiling.

**Architecture:** char n-gram TF-IDF cosine features computed locally; multilingual embedding
cosine features computed on Colab T4 and returned as a small per-pair parquet; per-country thresholds
plus a singleton decision; optional cross-encoder rerank; optional MinHash-LSH re-block.

**Tech stack:** pandas/pyarrow, DuckDB, scikit-learn (TF-IDF), LightGBM, rapidfuzz; Colab:
`sentence-transformers`, `torch` (fp16).

## Global constraints

- Metric is macro F0.5 with singletons; precision weighted 2×.
- Do not modify `ber/pairs.py::grouped_split`, `ber/audit.py::run_audit`, or `threshold.entity_f05`
  semantics.
- Baseline artifacts to preserve until a gate passes: `models/lgbm.txt`, `models/threshold.json`,
  `output/*.tsv`, `submission/matching_results.tsv`.
- Local runtimes: Phase 0 ~30 min; Phase 1 ~2–3 h; Phase 5 ~1 h. Colab: Phase 2 ~2–4 h; Phase 3 ~3–9 h.
- Baseline gate: held-out full-candidate macro F0.5 = **0.8488** (see `DATA/reports/eval_full_candidates.json`).

---

### Task 1: Diagnostic — oracle ceiling and per-country headroom

**Files:** create `code/business_entity_resolution/src/ber/diagnostic.py`; test
`code/business_entity_resolution/tests/test_diagnostic.py`; modify `cli.py` (`diagnose`).

- [ ] **Step 1: failing test** — `macro_oracle` on a tiny frame equals hand-computed macro F0.5.

```python
def test_oracle_macro():
    from ber.diagnostic import oracle_macro
    truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": {"S2-9"}}
    cand = {"S1-1": {"S2-1", "S2-7"}, "S1-2": set()}
    # S1-1 pred = {S2-1} -> P=1,R=.5 -> f=1.25*.5/(.25+.5)=0.8333 ; S1-2 pred empty, truth non-empty -> 0
    assert round(oracle_macro(truth, cand), 4) == round((0.8333333333 + 0.0) / 2, 4)
```

- [ ] **Step 2: implement** `oracle_macro(truth, cand)` = macro_f05 over `pred[s1] = truth[s1] & cand[s1]`.
- [ ] **Step 3: production `run_diagnostic(cfg)`** — DuckDB over `DATA/candidates/train_candidates.parquet`
  and `train_ground_truth.parquet`, restricted to held-out S1 ids from
  `ber.validation.val_s1_ids(cfg)`; compute:
  - `oracle_macro_f05` (global and per country),
  - `pair_recall` (found/total truth pairs),
  - `entities_with_zero_found` and its share,
  - `mean_entities_recall`.
  Write `DATA/reports/eval_oracle.json`; print a one-line summary.
- [ ] **Step 4: verify** `.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests/test_diagnostic.py -q`
  then `... -m ber.cli diagnose`.
- [ ] **Step 5: commit** `feat: oracle diagnostic for matcher vs blocking headroom`.

**Decision rule:** if `oracle_macro_f05 - 0.8488 > 0.05`, matcher work (Tasks 2–3) is the lever; if
small, prioritize Task 9 (blocking). Record the number in `docs/RESULTS.md`.

---

### Task 2: Char n-gram TF-IDF cosine features

**Files:** modify `ber/features.py`; test `tests/test_features.py`.

- [ ] **Step 1: failing test** — for `("Best Bakery Inc","Best Bakery")` the char-3 cosine `>`
  the `("Best Bakery Inc","Pizza Palace")` value.
- [ ] **Step 2: implement**
  - `fit_char_vectorizer(texts_sample, ngram=(3,3), min_df=3)` → pickled `TfidfVectorizer`
    (sha1, sublinear_tf) persisted at `DATA/processed/char3_vocab.json` (vocabulary only, to keep
    the artifact small and reproducible).
  - In `_feature_block`, add `name_char3_cos`, `name_roman_char3_cos`, `addr_char3_cos`
    (cosine of L2-normalized sparse rows via `(A.multiply(B)).sum(axis=1)`).
  - Extend `FEATURE_ORDER` with the three new names; keep the order stable and append at the end.
- [ ] **Step 3: compute** new columns for `DATA/features/train.parquet`, the held-out valfull feature
  parts, and test parts by re-running `ber.cli features` for each split (the vectorizer must be fit
  only on the train sample and reused).
- [ ] **Step 4: verify** pytest green; feature parquet has the new columns and no NaNs.
- [ ] **Step 5: commit** `feat: char n-gram TF-IDF cosine features`.

---

### Task 3: Per-country thresholds + singleton calibration

**Files:** create `ber/calibration.py`; test `tests/test_calibration.py`; modify `cli.py`.

- [ ] **Step 1: failing test** — two countries with different optimal thresholds are both recovered
  by `tune_per_country(probs, groups, labels, countries, global_default)`.
- [ ] **Step 2: implement**
  - `tune_per_country(...)` sweeps 0.05–0.95 per country using `threshold.entity_f05`, falls back to
    the global best for countries with < 5k val entities or no labels (France).
  - `singleton_decision(probs, groups, tau)`: per S1, keep no match if `max(prob) < tau`; else keep
    thresholded pairs. `tau` tuned as a second scalar on the held-out set (grid 0.3–0.99).
  - Persist to `models/threshold.json`: `{"global": .., "by_country": {"us":..,"india":..},
    "singleton_tau": .., "use_one_to_one": true}`.
- [ ] **Step 3: verify** pytest green; single scalar behavior still matches the old path when all
  countries share the same threshold.
- [ ] **Step 4: commit** `feat: per-country thresholds and singleton calibration`.

---

### Task 4: Retrain + held-out gate (milestone 1.4.0)

**Files:** modify `ber/train.py` (call calibration), `ber/validation.py` (use calibrated predict).

- [ ] **Step 1** `ber.cli train` → new `models/lgbm.txt` + calibrated `threshold.json`.
- [ ] **Step 2** `ber.cli validation --workers 8` → `DATA/reports/eval_full_candidates.json`.
- [ ] **Step 3 gate** adopt only if `chosen_macro_f05 > 0.8488` and India not worse. Otherwise revert
  `models/` and keep the baseline; record the outcome in `docs/RESULTS.md`.
- [ ] **Step 4** update `docs/RESULTS.md`, `docs/DECISIONS.md`; `git tag -a 1.4.0`.

---

### Task 5: Colab export/import (local)

**Files:** create `ber/colab_io.py`; test `tests/test_colab_io.py`; modify `cli.py`.

- [ ] **Step 1: failing test** — round-trip a tiny `entities`/`pairs`/`cosine` set through
  `export_for_colab` → `import_cosine` and assert the feature column is attached by `(s1_id, cand_id)`.
- [ ] **Step 2: implement**
  - `export_for_colab(cfg, splits)` → writes `DATA/colab_in/entities.parquet` (unique entity texts from
    all processed sources: `name_norm`, `name_roman`, `addr_norm`) and `pairs_{split}.parquet`
    (`s1_id, cand_id`) for `train|valfull|test` from `DATA/pairs/*`.
  - `import_cosine(cfg, split)` → reads `DATA/colab_out/cosine_{split}.parquet` and returns columns
    `emb_name_cos, emb_addr_cos` keyed by `(s1_id, cand_id)` for merging into features.
- [ ] **Step 3: verify** pytest green; export sizes are as expected (entities ~1–2 GB; test pairs
  ~250M rows).
- [ ] **Step 4: commit** `feat: colab export/import for per-pair cosine features`.

---

### Task 6: Colab embeddings notebook → per-pair cosine (GPU)

**Files:** create `notebooks/colab_embeddings.ipynb`. Drive it with the Colab MCP tools
(`colab_open_colab_browser_connection`, `colab_add_code_cell`, `colab_run_code_cell`).

Cells (execute in order; shard and checkpoint):

1. Install: `!pip -q install sentence-transformers`
2. Input: unzip/upload `entities.parquet` and `pairs_{split}.parquet` (or mount a Kaggle Dataset).
   **If input is missing, stop and report — do not fabricate data.**
3. Encode:
   ```python
   import numpy as np, pandas as pd, torch, pyarrow.parquet as pq
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
   model.half()
   # encode name_norm and addr_norm in shards of 500_000, normalize_embeddings=True,
   # batch_size=256, convert_to_numpy=True, cast to fp16, write DATA/emb_{field}_shard_{i}.npy
   ```
4. Pair cosine: build an `entity_id -> row index` map; for each `pairs_{split}` batch, gather the
   normalized fp16 vectors for `s1_id` and `cand_id` (valfull/test pair lists), compute row dot
   products, write `DATA/colab_out/cosine_{split}.parquet`
   (`s1_id, cand_id, emb_name_cos, emb_addr_cos`).
5. Report: print device, rows encoded, pair rows scored, and output file sizes.

Acceptance: all three splits produce `cosine_*.parquet`; test file ≤ ~0.6 GB; reruns are idempotent
(skip existing shards).

---

### Task 7: Import embeddings + retrain + gate (milestone 1.5.0)

- [ ] Merge `emb_name_cos`, `emb_addr_cos` into train/valfull/test features (Task 5 import).
- [ ] `ber.cli train` → `ber.cli validation --workers 8`.
- [ ] Gate: adopt only if held-out > Phase-1 value; else keep Phase 1. Update docs; `git tag -a 1.5.0`.

---

### Task 8 (optional): Cross-encoder rerank (GPU)

**Files:** create `notebooks/colab_rerank.ipynb`; `ber/colab_io.py` rerank helpers.

- [ ] Local: export top-K (K=10) pairs per S1 with `s1_text` (name_norm + addr_norm) and `cand_text`
  for train/valfull/test → `DATA/colab_in/rerank_pairs_{split}.parquet`.
- [ ] Colab: `CrossEncoder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")`, fp16,
  score pairs in shards of 200k, checkpoint; write `DATA/colab_out/rerank_{split}.parquet`.
- [ ] Local: import `ce_score` as a feature and/or rerank rule; retrain; gate on held-out; `git tag -a 1.6.0`.

---

### Task 9 (conditional): MinHash-LSH blocking to lift the recall ceiling

Trigger only if Task 1 shows recall binding (or after Phases 1–3 if the ceiling caps F0.5).

- [ ] Implement token/char-gram MinHash banding with DuckDB (no external service): k=8 hashes,
  4 bands; emit band keys as a new blocking pass 11 in `ber/blocking.py` with a per-pass cap; re-run
  `ber.cli block --split both` then `audit` (expect recall > 0.814). Re-run features/predict/outputs.
- [ ] Gate on net held-out macro F0.5 (re-blocking can lower precision). `git tag -a 1.7.0`.

---

### Task 10: Rebuild outputs, validate, package, submit (milestone 2.0.0)

- [ ] `ber.cli features --split test --workers 8` (with all adopted features).
- [ ] `ber.cli predict --split test --one-to-one` with calibrated thresholds.
- [ ] Validator must print PASS:
  ```
  .venv\Scripts\python.exe DATA\student_resource\utils\validate_submission.py `
    --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv `
    --test-dir DATA\student_resource\dataset\test
  ```
- [ ] Refresh `docs/RESULTS.md` with the new held-out number and the official leaderboard score once
  submitted. Repackage `dist/AA.._submission.zip`; copy the leaderboard file to
  `dist/leaderboard_upload/matching_results.tsv` and `submission/matching_results.tsv`.
- [ ] Commit, `git tag -a 2.0.0`, push `main` + tag; attach the new zip to a GitHub Release; upload
  `matching_results.tsv` in the Portal.

---

## Self-review checklist

- Every spec section maps to a task (diagnostic→1, char n-gram→2, calibration→3, gate→4, interface→5,
  embeddings→6, gate→7, rerank→8, blocking→9, rebuild→10).
- No candidate-set change before Task 9, so `candidate_pairs.tsv` stays valid through Phases 1–3.
- All gates are explicit numeric comparisons against 0.8488.
- Interfaces use exact column names `(s1_id, cand_id)` and feature names listed in `FEATURE_ORDER`.
