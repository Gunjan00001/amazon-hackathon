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


