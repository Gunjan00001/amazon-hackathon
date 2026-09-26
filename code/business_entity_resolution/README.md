# Business Entity Resolution — Code and Run Instructions

Self-contained pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task.
It matches Source 2 / Source 3 records to the deduplicated reference Source 1 and writes the two
submission files under `output/`.

## Environment

```
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Run every command from the repository root with the source path exported:

```powershell
$env:PYTHONPATH="code/business_entity_resolution/src"
```

`--config` defaults to `code/business_entity_resolution/config.json` when present, otherwise
`code/business_entity_resolution/src/config.json`, so the packaged copy runs without extra flags.
Adjust `dataset_dir` in the config to where the challenge data is extracted.

## Run order

```powershell
# 1. Clean + normalize all six record files (cached to DATA/processed)
.venv\Scripts\python.exe -m ber.cli prepare

# 2. Blocking: build keys and candidates for both splits (cached in DATA/keys)
.venv\Scripts\python.exe -m ber.cli block --split both

# 3. Blocking recall audit on train (DuckDB, ~50 s)
.venv\Scripts\python.exe -m ber.cli audit --split train

# 4a. Training pairs + pairwise features for train (parallel, 8 workers)
.venv\Scripts\python.exe -m ber.cli features --split train --workers 8 --combine

# 4b. Train the LightGBM matcher and tune the F0.5 threshold
.venv\Scripts\python.exe -m ber.cli train

# 5. Inference pairs + parallel features for test
.venv\Scripts\python.exe -m ber.cli features --split test --workers 8

# 6. Predict, apply one-to-one, write output/*.tsv
.venv\Scripts\python.exe -m ber.cli predict --split test --one-to-one --threshold 0.925

# 7. Validate (must print PASS)
.venv\Scripts\python.exe DATA\student_resource\utils\validate_submission.py `
    --matching output\matching_results.tsv `
    --candidate output\candidate_pairs.tsv `
    --test-dir DATA\student_resource\dataset\test

# 8. Local (4:1, optimistic) validation marks
.venv\Scripts\python.exe -m ber.cli evaluate

# 9. Believable held-out estimate: full candidates for the 20% held-out S1
.venv\Scripts\python.exe -m ber.cli validation --workers 8

# 10. Leave-one-country-out (unseen-country / France proxy)
.venv\Scripts\python.exe -m ber.cli loo
```

Prebuilt artifacts already in the repo: `models/lgbm.txt`, `models/feature_list.json`,
`models/threshold.json`. To reuse an existing prediction run, add `--reuse-predictions` to
`predict`; the intermediate prediction parts live in `DATA/tmp/test_pred/`.

## Package layout

```
src/ber/
  config.py       configuration dataclass
  prepare.py      stage 0: normalization, address parsing, parquet cache
  blocking.py     DuckDB blocking passes and candidate caps
  audit.py        DuckDB recall/reduction audit (run_audit)
  features.py     two-phase parallel pairwise feature generation
  pairs.py        training/inference pair construction and grouped split
  train.py        LightGBM training + threshold tuning
  threshold.py    vectorized macro-F0.5 entity scorer and sweep
  postprocess.py  one-to-one assignment filter
  predict.py      streaming inference and submission TSV writers
  evaluate.py     macro-F0.5 metric and local validation marks
  cli.py          command line entry point
tests/            pytest suite (run: .venv\Scripts\python.exe -m pytest -q)
```

## Outputs

- `output/matching_results.tsv` — scored file: `source1_entity_id`, `matched_entity_ids`.
- `output/candidate_pairs.tsv` — `source1_entity_id`, `candidate_entity_ids` (the exact candidate
  set scored by the model; matches are a subset of candidates).

Both files contain exactly 1,732,544 rows (one per test Source 1 entity); unmatched entities get an
empty second column. The official validator reports `PASS`.

## Tuning notes

- Per-pass block caps are in `config.json` (`pass_caps`); per-S1 cap is `cap`.
- **Threshold must be tuned on the full candidate distribution, not the 4:1 sample.** On the full
  candidates for the held-out S1 groups, the optimum moved from 0.675 (4:1 sample) to **0.925**;
  all submission outputs use 0.925.
- Reported scores:
  - 4:1 sampled split (grouped), one-to-one: macro F0.5 **0.9807** — *optimistic, not comparable to
    the leaderboard* (see `DATA/reports/eval_marks.json`).
  - Full candidates, held-out S1 groups (test-like): macro F0.5 **0.8488**, 95% CI 0.848–0.850,
    candidate recall ceiling 0.814, India 0.788 / US 0.889
    (see `DATA/reports/eval_full_candidates.json`).
  - Leave-one-country-out (unseen-country proxy for France): train-US→India **0.668**,
    train-India→US **0.804**, versus full-model US 0.891 / India 0.788
    (see `DATA/reports/eval_loo.json`). France is ~15% of test and has no labels, so the realistic
    leaderboard expectation is **~0.80–0.85**, below the in-domain 0.8488.
- The ground truth is injective (each S2/S3 ID matches at most one S1), which the `--one-to-one`
  post-process exploits for precision; it was verified to beat threshold-only on the full-candidate split.
- The true score is only available via the portal upload of `output/matching_results.tsv`.
