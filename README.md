# Amazon ML Challenge 2026 — Business Entity Resolution

Solution workspace for the Amazon ML Challenge 2026 *Business Entity Resolution* problem: match noisy business records from **Source 2** and **Source 3** to the deduplicated reference **Source 1**, evaluated by macro **F_0.5** (precision-heavy).

Status: **0.1.0** — problem statement, dataset analysis, GBDT benchmark, and project knowledge graph are in place. The end-to-end matching pipeline is next.

## Documentation

| File | What it covers |
|---|---|
| [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md) | Full challenge spec: format, outputs, metric, constraints, fair play. |
| [`RULES.md`](RULES.md) | Binding project rules: fair play, data handling, engineering, validation, versioning. |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Believable metrics (full-candidate 0.8488), LOO proxy, test output stats. |
| [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) | Chronological run log with commands, timings, and measured numbers. |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Architecture decisions and reasoning (why blocking+GBDT, threshold 0.925, ...). |
| [`docs/FAILURES_AND_FIXES.md`](docs/FAILURES_AND_FIXES.md) | Every failure, root cause, and fix, for future reference. |
| [`DATA/student_resource/dataset/DATASET.md`](DATA/student_resource/dataset/DATASET.md) | Measured dataset facts: schemas, row counts, ground-truth analysis, noise and encoding notes. |
| [`DATA/student_resource/README.md`](DATA/student_resource/README.md) | Official challenge README. |
| `graphify-out/GRAPH_REPORT.md` | Knowledge graph report over the project docs and tools. |

## Repository layout

```
PROBLEM_STATEMENT.md          # full transcription of the official statement
RULES.md                      # project rules (read first)
README.md                     # this file
DATA/                         # challenge drop (datasets git-ignored)
  student_resource/
    dataset/DATASET.md        # measured dataset documentation
    utils/validate_submission.py
    Documentation_template.md # required methodology write-up template
tools/
  eda_dataset.py              # dataset profiling -> tools/eda_stats.json
  bench_gbdt.py               # LightGBM vs XGBoost benchmark -> tools/bench_results.json
graphify-out/                 # knowledge graph (graph.html, graph.json, report)
.graphifyignore               # excludes .venv, data TSVs, caches from the graph
```

Not committed: `DATA/**/*.tsv`, `DATA/**/*.zip`, `.venv/` (see `.gitignore`).

## Quickstart

```powershell
# 1. Python 3.12 environment (uv)
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe `
    pandas pyarrow numpy scikit-learn lightgbm xgboost rapidfuzz

# 2. Profile the dataset
.venv\Scripts\python.exe tools\eda_dataset.py

# 3. Benchmark the matchers on sampled labeled pairs
.venv\Scripts\python.exe tools\bench_gbdt.py

# 4. Validate submission files (run from DATA/student_resource/)
python utils/validate_submission.py `
    --matching output/matching_results.tsv `
    --candidate output/candidate_pairs.tsv `
    --test-dir dataset/test
```

## Key dataset facts

- 6 record files + 1 ground-truth file, ~2.35 GiB: 2.21M / 5.03M / 5.29M train rows (S1/S2/S3) and 1.73M / 4.89M / 5.08M test rows.
- Ground truth is **injective**: each S2/S3 ID is matched to at most one S1 (all 7,638,365 matched IDs unique).
- **5.59% of Source 1 entities are singletons**; matched entities average 3.46 matches (max 11).
- Country is open-set: train is `US`/`India`, test adds `France` (zero-shot).
- ~15% of S2 train names are non-Latin scripts; Source 1 train names are 100% ASCII.
- Data is UTF-8; empty `business_address` in ~3.3% of S2/S3 rows.

## Matcher benchmark (0.1.0)

100k sampled S1 entities, identical 15 features, grouped 80/20 split:

| Scenario | Model | AUC | macro F0.5 | Train time |
|---|---|---|---|---|
| ~1.4:1 negatives:positives | LightGBM | 0.99973 | 0.99539 | 5.7s |
| ~1.4:1 | XGBoost | 0.99973 | 0.99561 | 25.4s |
| ~4.3:1 (harder) | LightGBM | 0.99980 | 0.98604 | 33.1s |
| ~4.3:1 | XGBoost | 0.99980 | 0.98574 | 96.2s |

Verdict: quality is tied; LightGBM trains 3–4.5× faster. Full results in `tools/bench_results.json`.

## Roadmap

1. Normalize + country-aware address parsing, cached to parquet.
2. Blocking passes (name/phonetic/rare-token/street/PIN) → `candidate_pairs.tsv`.
3. Pairwise features → LightGBM matcher, threshold tuned for macro F_0.5.
4. Predict test matches → `matching_results.tsv`, validate, package the submission zip.

## Fair play

No external databases, APIs, geocoding, or internet data augmentation. Models must be MIT/Apache-2.0 and ≤8B parameters. See `RULES.md`.
