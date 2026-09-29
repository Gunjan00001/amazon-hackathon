# HANDOVER.md — Amazon ML Challenge 2026: Business Entity Resolution

**Team:** AA.. · **Repo:** https://github.com/Gunjan00001/amazon-hackathon · **Branch:** `main`
**Version:** 2.1.0 · **Official public leaderboard:** macro F0.5 = **0.811** (2026-09-26, Evaluated)

This document is the single entry point for anyone who receives this repository after the
original working folder (`D:\Amazon project`, ~80 GB) is deleted. It explains **what is on
GitHub, what is not, and how to get back to a fully working state.**

> Read order: `README.md` → `RULES.md` → `AGENTS.md` → this file → `docs/ARCHIVE_MANIFEST.md`.

---

## 1. TL;DR

- The **entire source of truth for the solution** — pipeline code, tests, configs, trained
  model, calibration thresholds, documentation, plans/specs, and the knowledge graph — is on
  GitHub `main` (93 tracked files).
- The **raw challenge data (~2.4 GiB of TSVs)** and **generated intermediates (~62.8 GiB)** are
  **intentionally not in git**: they are non-redistributable and exceed GitHub's limits.
- The raw challenge TSVs are mirrored to a **private** Kaggle dataset `amz-er-2026-raw`
  (see §4). Everything else is reproducible from the raw data with the documented commands.
- The **leaderboard submission** (`matching_results.tsv`, 75.6 MiB) is tracked in the repo.
- The **large submission package artifacts** (zip + split `candidate_pairs.tsv`) are stored via
  **Git LFS** on `main` — see `docs/ARCHIVE_MANIFEST.md` §"Git LFS objects" for the exact list.

---

## 2. What is on GitHub vs what is not

| Content | On GitHub `main`? | Notes |
|---|---|---|
| `code/business_entity_resolution/` (pipeline `ber` package, tests, config, models) | ✅ tracked | ~10.6 MB, the real deliverable |
| `docs/` (all docs, plans, specs) | ✅ tracked | includes this file |
| `notebooks/` (Colab/Kaggle runners) | ✅ tracked | e1/e3/e4 runners + embedding notebook |
| `graphify-out/` (knowledge graph) | ✅ tracked | `graph.json`, `graph.html`, `GRAPH_REPORT.md` (cache ignored) |
| `tools/` (EDA + benchmark) | ✅ tracked | `eda_cache/` ignored (regenerable) |
| `submission/matching_results.tsv` | ✅ tracked | 75.6 MiB leaderboard file |
| `output/matching_results.tsv` | ✅ **LFS** | same file, LFS-tracked |
| `SUBMIT/*.zip` and `SUBMIT/*.part_*` | ✅ **LFS** | large submission artifacts (see manifest) |
| `DATA/**/*.tsv`, `*.zip` | ❌ | non-redistributable; private Kaggle dataset |
| `DATA/{keys,candidates,processed,pairs,features,reports,tmp,kaggle,colab_*}/` | ❌ | regenerable intermediates (62.8 GB) |
| `DATA/student_resource/` docs (`README.md`, `DATASET.md`, `utils/`, `Documentation_template.md`) | ✅ tracked | only the docs/validator, not the data |
| `dist/` (submission zip + package) | ❌ | generated; see manifest |
| `output/candidate_pairs.tsv` | ❌ | 3.25 GB, > GitHub's 2 GB LFS limit; split parts are in LFS |
| `kaggle Downloaded/` | ❌ | Kaggle artifacts; regenerate on Kaggle |
| `.venv/` | ❌ | recreate with `uv` (see §3) |
| `.git/` | n/a | local only |

Full per-directory inventory and byte sizes: **`docs/ARCHIVE_MANIFEST.md`**.

---

## 3. Restore to a working state (fresh clone)

### 3.1 Clone with Git LFS

```powershell
git lfs install
git clone https://github.com/Gunjan00001/amazon-hackathon.git
cd amazon-hackathon
git lfs pull        # downloads matching_results.tsv and the LFS submission artifacts
```

> Large-file LFS bandwidth is metered by GitHub. If the account is over its LFS quota, use
> `git lfs pull --include=...` to fetch only what you need.

### 3.2 Python environment (ML + CPU pipeline)

The ML pipeline needs **Python 3.12** (3.14 lacks reliable ML wheels):

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r code/business_entity_resolution/requirements.txt
```

Graphify runs on the **system Python 3.14** (installed via `pip install graphifyy`); keep the
two environments separate.

### 3.3 Get the raw challenge data

The 7 challenge TSVs are **not** in git. Recover them from one of:

1. **Private Kaggle dataset `amz-er-2026-raw`** (canonical mirror, `RULES.md` §6.3) —
   download and place under `DATA/student_resource/dataset/`:
   ```
   train/source1.csv  train/source2.csv  train/source3.csv
   test/source1.csv   test/source2.csv   test/source3.csv
   train_ground_truth.csv
   ```
2. The original challenge drop, if still available: `DATA/student_resource/` (it was also
   shipped as `DATA/6ab10eb3b23ba_student_resource.zip`).

Once present, run the EDA to regenerate `tools/eda_stats.json`:

```powershell
.venv\Scripts\python.exe tools\eda_dataset.py
```

### 3.4 Rebuild pipeline outputs

From the repo root with `PYTHONPATH` set:

```powershell
$env:PYTHONPATH = "code/business_entity_resolution/src"
$cfg = "code/business_entity_resolution/config.json"

.venv\Scripts\python.exe -m ber.cli prepare  --config $cfg
.venv\Scripts\python.exe -m ber.cli block    --split both --config $cfg   # keys cached; ~20 min train
.venv\Scripts\python.exe -m ber.cli audit    --split train --config $cfg
.venv\Scripts\python.exe -m ber.cli features --split train --workers 8 --combine --config $cfg
.venv\Scripts\python.exe -m ber.cli train    --config $cfg
.venv\Scripts\python.exe -m ber.cli calibrate --config $cfg               # needs `validation` first
.venv\Scripts\python.exe -m ber.cli features --split test --workers 8 --config $cfg
.venv\Scripts\python.exe -m ber.cli predict  --split test --one-to-one --config $cfg
```

The trained `models/lgbm.txt`, `models/feature_list.json`, and `models/threshold.json` are
tracked, so **`predict` works immediately without retraining** (as long as the blocked
candidate set exists).

### 3.5 Verify

```powershell
.venv\Scripts\python.exe -m pytest -q                                   # 45 tests
# Validator (from DATA/student_resource/):
python utils/validate_submission.py --matching ../../submission/matching_results.tsv `
    --candidate ../../output/candidate_pairs.tsv --test-dir dataset/test
```

---

## 4. Where the big data lives

| Data | Location | How to restore |
|---|---|---|
| Raw challenge TSVs (~2.4 GiB) | Private Kaggle dataset `amz-er-2026-raw` | Kaggle API / web download |
| Processed/keys/candidates/features/reports (62.8 GB) | **Nowhere persistent** — regenerate | §3.4 |
| `candidate_pairs.tsv` (3.25 GB) | Rebuild via `ber.cli block`; split copies in LFS | §3.4 |
| Submission zip | Git LFS (`SUBMIT/AA.._submission.zip`) | §3.1 |
| Kaggle parquet artifacts | Re-run the Kaggle/Colab notebooks | `notebooks/` |

**Nobody should rely on the deleted `D:\Amazon project` folder for anything except perhaps the
Kaggle credentials file.** The data mirror is the private Kaggle dataset; everything else is
code + reproducible artifacts.

---

## 5. Key results (for context)

| Evaluation | macro F0.5 |
|---|---|
| Official leaderboard (public, Portal) | **0.811** |
| Held-out full-candidate (adopted model, test-like) | 0.8577 (US 0.8983 / India 0.7968) |
| Oracle (candidate-recall ceiling) | 0.9122 |
| E3 lexical ∪ e5-ANN oracle (not shipped) | 0.9892 (K=2000) |
| Unseen-country proxy (leave-one-country-out) | 0.668–0.804 |

Held-out candidate recall ceiling 0.8142 was the main limiter. The E1 e5-feature model
(held-out 0.8590) and E3 ANN ceiling are documented but not shipped — see
`docs/SUBMISSION.md` §6 and `docs/DECISIONS.md` D16.

---

## 6. Gotchas (carried over from `AGENTS.md`)

- `DATA` and `data` collide on Windows; intermediates land in `DATA/`.
- Parquet list columns come back from `iter_batches` as numpy arrays.
- Per-token Metaphone blocking explodes temp space — removed; do not reintroduce.
- Commas are data: always `sep="\t"`, `keep_default_na=False`.
- The LightGBM text model (`models/lgbm.txt`) must stay **LF-only** (`.gitattributes` marks it `-text`).
- Generated datasets/keys/candidates are git-ignored; never commit them.
- Free-Colab `files.upload()` truncates at ~100 MB; use Drive or the Files panel.

---

## 7. If something is missing

1. Check `docs/ARCHIVE_MANIFEST.md` — it lists every directory that existed locally, its size,
   its git status, and how to regenerate it.
2. Check `docs/FAILURES_AND_FIXES.md` — every non-trivial bug already hit.
3. The knowledge graph (`graphify-out/GRAPH_REPORT.md`, `graph.html`) maps concepts to files.
