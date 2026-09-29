# Reproduction guide

How to go from a fresh clone to a validated submission. All heavy compute runs on Kaggle;
the local machine only edits code, runs unit tests, and validates the final TSVs.

## 1. Prerequisites

| Tool | Notes |
|---|---|
| Git + [Git LFS](https://git-lfs.com/) | `output/*.tsv` and `submission/team_submission.zip` are LFS objects. |
| [uv](https://docs.astral.sh/uv/) | Creates the Python 3.12 environment. |
| Python 3.12 | 3.14 lacks reliable ML wheels for this stack. |
| Kaggle account + `kaggle.json` API token | Required to push datasets/kernels. Never commit it. |

```powershell
git clone https://github.com/Gunjan00001/amazon-hackathon.git
cd amazon-hackathon
git lfs install          # one-time per machine
git lfs pull             # materialize the LFS outputs
```

## 2. Environment

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r code\business_entity_resolution\requirements.txt
```

Run project scripts with `.venv\Scripts\python.exe`.

## 3. Restore the dataset and configuration

The raw challenge TSVs are **not** in the repo (non-redistributable, >100 MB each). Restore
them from the official challenge download to `DATA/student_resource/dataset/{train,test}/`
(see [`REPOSITORY_GUIDE.md`](REPOSITORY_GUIDE.md)).

Point the pipeline at the data and an artifact directory:

```powershell
$env:BER_DATA_DIR     = "D:\Amazon Proj Approach 3\amazon-hackathon\DATA\student_resource\dataset"
$env:BER_ARTIFACT_DIR = "D:\Amazon Proj Approach 3\amazon-hackathon\artifacts"
```

## 4. Unit tests (local, CPU)

```powershell
.venv\Scripts\python.exe -m pytest code\business_entity_resolution\tests -q
```

## 5. Local pipeline (CPU, optional)

The same stages run locally (on a sample) or on Kaggle (full scale):

```powershell
# clean -> cached parquet
.venv\Scripts\python.exe -m ber.stages.clean --sample 0

# autonomous search + finalize (CPU); --smoke-first does a quick sanity pass
.venv\Scripts\python.exe -m ber.stages.search --smoke-first
```

Two-step variant: `ber.stages.search --search-only`, then
`ber.stages.search --finalize-only --search-dir <search-output>`.

## 6. Kaggle flow (full scale)

The current automated driver is `kaggle/auto.py`:

```powershell
# refresh the code dataset, push the A1 search kernel, wait, fetch, validate
.venv\Scripts\python.exe kaggle\auto.py --plan auto --update-code

# local format gate only
.venv\Scripts\python.exe kaggle\auto.py --validate-only
```

Under the hood (manual equivalent):

```powershell
python kaggle\push_all.py datasets --only code          # publish src as a private dataset
python kaggle\push_all.py push --plan auto --only A1     # search: writes best.json + cached keys
python kaggle\push_all.py push --plan auto --only A1f    # finalize: writes the TSVs
python kaggle\push_all.py fetch --collect               # -> output/
```

Notebooks:

- **Current:** `kaggle/notebooks/A1_search.ipynb`, `A1f_finalize.ipynb`.
- **Legacy cascade:** `N1_clean` → `N2_block` → `N3_embed` → `N4_train_gbdt` →
  `N5_rerank` → `N6_decide` (see the design spec for inputs/outputs per stage).

Progress is visible at `https://www.kaggle.com/code/gunjanpal001/amz-er-a1-search`.

## 7. Validate the submission (required gate)

Run from `DATA/student_resource/`:

```powershell
python utils/validate_submission.py `
    --matching  ..\..\..\output\matching_results.tsv `
    --candidate ..\..\..\output\candidate_pairs.tsv `
    --test-dir  dataset\test --check-ids
```

`PASS` (exit 0) is required before uploading to the portal. Invariants checked: one row per
test Source 1 entity, empty IDs for singletons, no duplicates, only existing S2/S3 IDs, and
matches ⊆ candidates.

## 8. Package the submission

`submission/make_zip.py` packages `output/` + the pipeline source + the filled
`Documentation_template.md` into `submission/team_submission.zip`.
