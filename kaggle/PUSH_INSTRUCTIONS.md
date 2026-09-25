# Kaggle push and run instructions (manual web-UI mode)

Everything heavy runs on Kaggle. This machine only edits code and validates outputs.
Read `RULES.md` §6 first. Order: prepare datasets once, then run N1 → N2 → N4 → N6 (v1),
then N3 → N4 → N6 (v2), then N5 → N6 (v3).

## 0. One-time datasets (private)

1. `amz-er-2026-raw` — Kaggle → Create → Dataset → name `amz-er-2026-raw`, visibility
   **Private**. Upload the 7 TSVs from `D:\Amazon project\DATA\student_resource\dataset`:
   open the folder and drag its `train` and `test` subfolders into the uploader so the
   dataset root contains `train/` and `test/`. Do **not** upload `.DS_Store`.
2. `amz-er-2026-code` — Create → Dataset → name `amz-er-2026-code`, **Private**. Upload the
   `src` folder from `code/business_entity_resolution` (drag the folder itself) so the
   dataset root contains `src/ber/...`. Later code changes: open the dataset → **New Version**
   → delete the old `src` and upload the new one.

## 1. Notebooks

For each of `kaggle/notebooks/N1_clean.ipynb` … `N6_decide.ipynb`:

1. Kaggle → Create → Notebook → File → Import Notebook → upload the `.ipynb`.
2. Settings → Internet: **ON**. Accelerator: **GPU T4 x2** only for N3 and N5, **None** for
   N1/N2/N4/N6 (saves GPU quota).
3. Add Input → `amz-er-2026-raw` + `amz-er-2026-code`, plus previous stage outputs
   (Add Input → Your Work → Notebooks → pick the previous notebook → its output).
   Dependency map: N2←N1, N3←N1, N4←N1+N2(+N3), N5←N4, N6←N2+N4(+N5).
4. Run All. Then **Save Version → Save & Run All** so the output (under
   `/kaggle/working/artifacts/...`) is stored and can be added as input downstream.
5. Copy the printed `metrics.json` / `stats.json` block back to the repo owner (paste in chat).

N1–N2 take ~10–40 min each on CPU; N4 is the longest CPU stage; N3 is ~2–4 GPU h;
N5 is ~1–3 GPU h. Stop idle sessions — they burn GPU quota (30 h/week total).

## 2. Pulling results

After N6 passes, download `matching_results.tsv` and `candidate_pairs.tsv` from the
notebook Output panel into `D:\Amazon Proj Approach 3\amazon-hackathon\output\`, then run
the official validator locally from `DATA/student_resource`:

```powershell
python utils/validate_submission.py `
    --matching D:\Amazon Proj Approach 3\amazon-hackathon\output\matching_results.tsv `
    --candidate D:\Amazon Proj Approach 3\amazon-hackathon\output\candidate_pairs.tsv `
    --test-dir dataset/test
```

`PASS` (exit 0) is required before uploading to the portal.

## 3. Quota and failure notes

- If a GPU stage fails mid-run, fix code, upload a code dataset New Version, re-run the
  notebook. Cached earlier artifacts stay valid.
- If GPU quota is exhausted: skip N3/N5 and ship the last validated v1/v2 output.
- Never commit `kaggle.json`, tokens, or `/kaggle/working` outputs to git.
