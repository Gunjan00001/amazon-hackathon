# Amazon ML Challenge 2026 — Business Entity Resolution

Solution workspace for the Amazon ML Challenge 2026 *Business Entity Resolution* problem: match noisy business records from **Source 2** and **Source 3** to the deduplicated reference **Source 1**, evaluated by macro **F_0.5** (precision-heavy).

Status: **2.0.0** — end-to-end pipeline shipped and submitted. Public leaderboard macro F0.5 = **0.811**; the adopted local matcher (char-3 TF-IDF features + per-country/singleton calibration) reaches held-out full-candidate **0.8577** and a calibrated test submission (validator PASS) is staged for upload. Multilingual-embedding features were tried on a Colab T4 but rejected by their gate.

## Documentation

| File | What it covers |
|---|---|
| [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md) | Full challenge spec: format, outputs, metric, constraints, fair play. |
| [`RULES.md`](RULES.md) | Binding project rules: fair play, data handling, engineering, validation, versioning. |
| [`AGENTS.md`](AGENTS.md) | Working notes for agents: environments, data contracts, blocking/audit contracts, gotchas. |
| [`docs/HANDOVER.md`](docs/HANDOVER.md) | **Start here after cloning**: what is on GitHub vs. not, how to restore a working state, where the data lives. |
| [`docs/ARCHIVE_MANIFEST.md`](docs/ARCHIVE_MANIFEST.md) | Full inventory of the original ~80 GB working folder with sizes, git status, and regeneration steps. |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Metrics: official leaderboard **0.811**, baseline 0.8488, adopted **0.8577**, LOO proxy, output stats. |
| [`docs/superpowers/plans/2026-09-26-precision-colab.md`](docs/superpowers/plans/2026-09-26-precision-colab.md) | M2 plan (precision lift + Colab embeddings) and its outcome. |
| [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) | Chronological run log with commands, timings, and measured numbers. |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Architecture decisions and reasoning (blocking+GBDT, char-3 features, calibration, embeddings rejection, ...). |
| [`docs/FAILURES_AND_FIXES.md`](docs/FAILURES_AND_FIXES.md) | Every failure, root cause, and fix, for future reference. |
| [`notebooks/colab_embeddings.ipynb`](notebooks/colab_embeddings.ipynb) | Colab T4 embedding-cosine notebook (run; features not adopted). |
| [`DATA/student_resource/dataset/DATASET.md`](DATA/student_resource/dataset/DATASET.md) | Measured dataset facts: schemas, row counts, ground-truth analysis, noise and encoding notes. |
| [`DATA/student_resource/README.md`](DATA/student_resource/README.md) | Official challenge README. |
| `graphify-out/GRAPH_REPORT.md` | Knowledge graph report over the project docs and code. |

## Repository layout

```
PROBLEM_STATEMENT.md          # full transcription of the official statement
RULES.md                      # project rules (read first)
README.md                     # this file
AGENTS.md                     # agent working notes
code/business_entity_resolution/
  src/ber/                    # pipeline package (prepare, blocking, features, train, predict, calibration, ...)
  config.json                 # pipeline config (pass caps, LightGBM params)
  models/                     # LightGBM model + calibrated threshold.json (tracked)
  tests/                      # pytest suite (conftest puts src/ on sys.path)
notebooks/
  colab_embeddings.ipynb      # Colab T4 embedding-cosine notebook (run; not adopted)
DATA/                         # challenge drop (datasets git-ignored)
  student_resource/
    dataset/DATASET.md        # measured dataset documentation
    utils/validate_submission.py
    Documentation_template.md # required methodology write-up template
tools/
  eda_dataset.py              # dataset profiling -> tools/eda_stats.json
  bench_gbdt.py               # LightGBM vs XGBoost benchmark -> tools/bench_results.json
output/                       # matching_results.tsv + candidate_pairs.tsv (TSVs git-ignored)
submission/                   # staged leaderboard file
dist/                         # submission package + zip (git-ignored)
graphify-out/                 # knowledge graph (graph.html, graph.json, report)
.gitattributes                # keeps the LightGBM text model LF-only
```

Not committed: `DATA/**/*.tsv`, `DATA/**/*.zip`, `.venv/` (see `.gitignore`).

Large deliverables are stored with **Git LFS**: `output/matching_results.tsv`, `SUBMIT/AA.._submission.zip`,
and `SUBMIT/candidate_pairs.tsv.part_{00,01}` (a byte-exact split of the 3.03 GiB `candidate_pairs.tsv`,
which exceeds GitHub's 2 GB LFS limit). Run `git lfs pull` after cloning. What is and isn't on GitHub,
plus restore steps, is documented in [`docs/HANDOVER.md`](docs/HANDOVER.md) and
[`docs/ARCHIVE_MANIFEST.md`](docs/ARCHIVE_MANIFEST.md).

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

## Results

| Evaluation | macro F0.5 |
|---|---|
| **Official leaderboard (public Portal, 26 Sep 2026)** | **0.811** |
| **Adopted M2 local model (char-3 + calibration)** | **0.8577** |
| Full-candidate held-out baseline (local, test-like) | 0.8488 |
| 4:1 sampled (local, optimistic) | 0.9807 |
| Unseen-country proxy (train US -> India / India -> US) | 0.668 / 0.804 |

The official 0.811 fell inside the predicted 0.80–0.85 band, ~0.04 below the baseline held-out
estimate (unseen France + public/private split). The adopted M2 model improves the held-out estimate
to **0.8577** (US 0.8983 / India 0.7968) with an unchanged candidate set; the calibrated test
submission is regenerated and validator-PASS, awaiting upload. Candidate recall ceiling 0.814 remains
the main limiter.

## Roadmap

1. ~~Normalize + country-aware address parsing, cached to parquet.~~
2. ~~Blocking passes (name/rare-token/prefix/pair/triple) → `candidate_pairs.tsv`.~~
3. ~~Pairwise features → LightGBM matcher, threshold tuned for macro F_0.5.~~
4. ~~Predict test matches → `matching_results.tsv`, validate, package the submission zip.~~
5. ~~M2 precision lift: char n-gram TF-IDF features + per-country/singleton calibration (adopted);~~
   ~~Colab T4 embeddings (tested; rejected by the gate).~~
6. **Next:** re-test the full 384-dim embedding cosine on Kaggle T4x2 (the 64-dim projection was
   noise-limited), try the cross-encoder rerank, and/or raise blocking recall above 0.814 with
   MinHash-LSH fuzzy blocking.


## Fair play

No external databases, APIs, geocoding, or internet data augmentation. Models must be MIT/Apache-2.0 and ≤8B parameters. See `RULES.md`.
