# Amazon ML Challenge 2026 — Business Entity Resolution

Solution workspace for the Amazon ML Challenge 2026 *Business Entity Resolution* problem: match noisy business records from **Source 2** and **Source 3** to the deduplicated reference **Source 1**, evaluated by macro **F_0.5** (precision-heavy).

Status: **1.0.0** — submission-ready. The classical cascade (IDF-weighted blocking + LightGBM)
reaches **validation macro F_0.5 = 0.9740** at **recall ceiling 1.0**, and the two required TSVs
pass the official validator with `--check-ids`. All compute runs on Kaggle; the local machine
only edits code, runs unit tests, and validates the final TSVs.

## Submission outputs

| File | Description |
|---|---|
| `output/leaderboard_matching_results.tsv` | matches to upload to the Portal (threshold 0.70) |
| `output/matching_results.tsv` | canonical final matches (same as the leaderboard file) |
| `output/matching_results_t650.tsv` | alternate threshold 0.65 (validated) |
| `output/candidate_pairs.tsv` | the 16,167,646 candidate pairs the model scored |
| `submission/team_submission.zip` | final package: `output/` + `code/` + `Documentation_template.md` |

Interactive/measured metrics: pair AUC 0.99952, average precision 0.99665, blocking
recall ceiling 1.0, test rows 1,732,544, matched entities 1,310,189 (threshold 0.70).

## Documentation

Full documentation index: [`docs/README.md`](docs/README.md).

| File | What it covers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Pipeline stages, module map, design rationale, results. |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | End-to-end reproduction: environment, stages, Kaggle, validation. |
| [`docs/REPOSITORY_GUIDE.md`](docs/REPOSITORY_GUIDE.md) | Repo/GitHub layout, Git LFS, what is excluded and how to restore it. |
| [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md) | Full challenge spec: format, outputs, metric, constraints, fair play. |
| [`RULES.md`](RULES.md) | Binding project rules: fair play, data handling, engineering, validation, versioning. |
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

Not committed: `DATA/**/*.tsv`, `DATA/**/*.zip`, `.venv/` (see `.gitignore` and
[`docs/REPOSITORY_GUIDE.md`](docs/REPOSITORY_GUIDE.md)).

Large artifacts (`output/*.tsv`, `submission/team_submission.zip`) are stored with **Git LFS**.
After cloning run `git lfs install && git lfs pull` to fetch their real contents.

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

## Automated Kaggle search

`ber.search` runs entirely on Kaggle: it searches blocking parameters (recall ceiling)
and matcher/threshold parameters, scoring every trial by held-out macro F_0.5
(S1-grouped, singletons included), then finalizes the best config into the two TSVs.
Trials are checkpointed to `search/trials.json`.

```powershell
# from the repo root
.venv\Scripts\python.exe kaggle\auto.py --plan auto --update-code   # push + wait + fetch + validate
.venv\Scripts\python.exe kaggle\auto.py --validate-only             # local format gate
```

## Version log

- `1.0.0` — validated submission: recall ceiling 1.0, validation macro F_0.5 0.9740, cached
  blocking keys, split search/finalize kernels, official validator PASS, packaged zip.
- `0.3.0` — autonomous validation-driven search (`ber.search`, `ber.evaluate`), IDF-weighted
  blocking, robust GBDT validation fallback, and the Kaggle `auto` driver.
- `0.2.0` — Kaggle-only compute rules and cascade C+B design.
- `0.1.0` — docs, dataset analysis, GBDT benchmark, knowledge graph.

## Roadmap

1. ~~Normalize + country-aware address parsing, cached to parquet.~~ (done)
2. ~~Blocking passes (name/phonetic/rare-token/street/PIN) → candidates.~~ (done; IDF-ranked)
3. ~~Pairwise features → LightGBM matcher, threshold tuned for macro F_0.5.~~ (done)
4. Multi-seed / held-out-country validation; embedding + cross-encoder cascade (v2/v3).
5. Predict test matches → `matching_results.tsv`, validate, package the submission zip.

## Fair play

No external databases, APIs, geocoding, or internet data augmentation. Models must be MIT/Apache-2.0 and ≤8B parameters. See `RULES.md`.
