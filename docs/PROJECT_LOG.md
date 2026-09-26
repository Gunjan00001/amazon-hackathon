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
