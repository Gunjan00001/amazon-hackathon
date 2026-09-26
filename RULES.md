# RULES.md — Amazon ML Challenge 2026: Business Entity Resolution

Project rules for every human and agent working in this repository. These are binding.

## 1. Challenge rules (from the official problem statement)

1. **No external data lookup. STRICTLY PROHIBITED.** No commercial ER APIs, government registries, geocoding APIs, or any internet data augmentation. Violation = immediate disqualification. Every model/algorithm must run on the provided data only.
2. **Model license:** the final model must be MIT or Apache-2.0 licensed, and at most 8B parameters.
3. **Format is law.** All inputs and outputs are UTF-8, **tab-separated** `.tsv`. Commas are data (inside addresses and ID lists) and must never be treated as delimiters.
4. **Output invariants** (a violation rejects the submission):
   - `matching_results.tsv` has exactly one row per test Source 1 entity (1,732,544 rows plus header).
   - `matched_entity_ids` is empty for singletons.
   - No duplicate IDs within any ID list; no duplicate `source1_entity_id` rows.
   - ID lists contain only existing `S2-`/`S3-` test IDs (no `S1-` self-matches).
   - Every matched ID must also appear in `candidate_pairs.tsv` (matches ⊆ candidates).
5. **Country is an open set.** Train has `US`/`India`; test adds `France`. Never hard-code, filter, or one-hot to seen countries. Every test entity must appear in the output.
6. **Validate before every submission** from the `student_resource/` directory (see §4).
7. **Evaluation** is macro F_0.5 (β = 0.5), per Source 1 entity, singletons included. Optimize precision first.

## 2. Data rules

1. `DATA/` is read-only challenge input. Never edit the source TSVs.
2. Do not commit data: `.tsv`, `.zip`, and `__MACOSX/` under `DATA/` are git-ignored (>100 MB and non-redistributable).
3. Read with an explicit tab separator and preserve empties:
   - pandas: `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`
   - Empty `business_address` values are real signal (~3.3% of S2/S3), not NaN.
4. Process in chunks/streams; assume the ~2.4 GB dataset does not fit comfortably alongside feature matrices.
5. Use fixed seeds for sampling and splits; never split pairs randomly across the same Source 1 entity (group by `source1_entity_id`).
6. Hold one country out when validating generalization, because France is zero-shot.

## 3. Engineering rules

1. **Python 3.12 only for ML work.** The system Python 3.14 lacks reliable ML wheels. Use the project venv:
   ```powershell
   uv venv .venv --python 3.12
   uv pip install --python .venv\Scripts\python.exe pandas pyarrow numpy scikit-learn lightgbm xgboost rapidfuzz
   ```
   Run project scripts with `.venv\Scripts\python.exe`.
2. **Python 3.14 is fine for graphify** (`graphify` is installed system-wide). Keep the two environments separate.
3. Tools live in `tools/`; the deliverable pipeline will live in `code/business_entity_resolution/src/`.
4. No comments in code unless they explain non-obvious constraints.
5. Keep everything on the `D:` drive; clean up large intermediates after use.
6. New behavior must be verifiable: unit-test normalizers, blocking keys, feature calculators, the F_0.5 scorer, and output writing. Run `utils/validate_submission.py` against fixtures before shipping.
7. LightGBM is the default matcher (3–4.5× faster to train at equal quality); XGBoost stays available as a drop-in alternative/ensemble member. Re-benchmark on the real candidate set before locking this in.

## 4. Verification commands

```powershell
# 1) Dataset EDA (writes tools/eda_stats.json, caches in tools/eda_cache/)
.venv\Scripts\python.exe tools\eda_dataset.py

# 2) LightGBM vs XGBoost benchmark (writes tools/bench_results.json)
.venv\Scripts\python.exe tools\bench_gbdt.py

# 3) Submission validation (run from DATA/student_resource/)
python utils/validate_submission.py `
    --matching output/matching_results.tsv `
    --candidate output/candidate_pairs.tsv `
    --test-dir dataset/test

# 4) Kaggle notebook chain (manual web-UI runs, in this order):
#    N1 clean -> N2 block -> N3 embed (GPU) -> N4 GBDT -> N5 rerank (GPU) -> N6 decide+validate
#    Per-notebook inputs/outputs and paste-back metrics: see the design doc
#    (docs/superpowers/specs/2026-09-25-kaggle-cascade-design.md).
```

The validator is a gate: exit 0 (`PASS`) required before any submission.

## 5. Versioning and git rules

1. **Tags use `major.minor.bugs`** (e.g. `0.1.0`). Bump `bugs` for fixes/docs, `minor` for new pipeline capability, `major` for a submission-ready milestone.
2. Never commit secrets, tokens, or credentials.
3. Never hand-edit `graphify-out/` outputs; regenerate them with the graphify pipeline.
4. Keep `main` in a state where the documented commands work.
5. Current version: **1.1.0** (leaderboard submission documented; graphify knowledge graph refreshed). Earlier: 1.0.0 = submission-ready package (validator PASS, held-out full-candidate macro F0.5 0.8488); 0.2.0 = Kaggle-only compute rules and cascade C+B design; 0.1.0 = initial docs, EDA and benchmark tooling, knowledge graph.

## 6. Compute rules — Kaggle-only

1. All training and GPU inference runs on Kaggle: Kaggle Notebooks + Kaggle Datasets only. The local machine is for editing, CPU-only prep/validation, and preparing notebook runs.
2. Kaggle accelerators only; baseline target is T4 x2 (fp16), L4 x4 as optional speedup. Free tier gives ~30 GPU-h/week (9-12 h session cap, ~20 GB notebook outputs): every GPU stage must cache its artifacts as a versioned Kaggle Dataset output so a re-run never repeats GPU work.
3. The 7 challenge TSVs are mirrored to a **private** Kaggle Dataset `amz-er-2026-raw` (never public, never in git). Canonical local data stays at `D:\Amazon project\DATA\`; code reads `DATA_DIR` from env/config, no hard-coded paths.
4. Runtime model downloads on internet-enabled notebooks are allowed, but every model must be MIT/Apache-2.0 and <=8B params; pin exact model IDs + revisions in requirements/metadata.
5. Each pipeline stage is one notebook (N1 cleaning, N2 blocking, N3 embed, N4 GBDT, N5 rerank, N6 decision+outputs) with pinned dependency versions; artifacts flow dataset -> notebook -> dataset so runs are reproducible.
6. `kaggle.json`, notebook secrets, and tokens are never committed.
7. Validate outputs locally with `utils/validate_submission.py` before any upload; record notebook + dataset versions in the version log (see §5).
8. Data cleaning and feature engineering are first-class pipeline code: normalization, transliteration, address parsing, and every pair feature are versioned, cached Kaggle Dataset artifacts, and the SAME cleaning + feature code runs for train, validation and test. No label-derived features; validation splits stay grouped by Source 1 entity with a held-out country (France proxy).
