# BER Execution Plan v2 — Blackwell-interactive + local CPU

**Status:** PLAN ONLY. The agent that produced this document does **not** execute it.
**Audience:** a *different* execution agent (or human) who will run the code.
**Supersedes:** `2026-09-27-ber-execution-plan.md` (v1) and `2026-09-27-ber-execution-plan-rtx6000.md`.
**Author context:** Amazon ML Challenge 2026 — Business Entity Resolution.

> Read `RULES.md`, `AGENTS.md`, `PROBLEM_STATEMENT.md`, and `docs/RESULTS.md` before touching anything.

---

## 0. How to use this document

This is the single source of truth for execution. It encodes hard-won environment facts so the
executor does not rediscover them. Follow the phase order. Every phase has an explicit **owner**
(agent-local vs human-interactive), **gate**, and **fallback**. Never destroy the baseline.

---

## 1. Reality constraints (do not fight these)

### 1.1 Compute matrix

| Environment | GPU | VRAM | RAM | Disk | Internet | Who drives it |
|---|---|---|---|---|---|---|
| **Kaggle interactive** (`gunjanpal/the-gpu-one`) | **RTX Pro 6000 Blackwell** | **~96 GB (97,887 MiB)** | (large) | ~20 GB working + big scratch | must be ON | **Human in browser** |
| Kaggle API/CLI run (any account) | 2× Tesla T4 | 2× 15.6 GB | **31 GB** | `/kaggle/working` 20 GB, overlay **1.1 TB** | yes | Agent via CLI |
| Local CPU (`.venv`, Py 3.12) | — | — | ~23 GB | D: ~96 GB free | yes | Agent |
| Colab free (MCP) | 1× Tesla T4 | 15.6 GB | 12 GB | 66 GB | yes | Agent (fallback) |

### 1.2 Critical, verified facts

1. **The RTX Pro 6000 is only available in an interactive browser session.** API/CLI-triggered runs
   (including `kaggle kernels push` on `the-gpu-one` itself) allocate **T4x2**. `machine_shape` in
   `kernel-metadata.json` is **silently ignored** (a bogus value is accepted with no error).
   → Therefore the heavy GPU stages must be **run interactively by a human**, and the agent writes
   the notebook code.
2. **The human runs the notebook; outputs must be persisted or downloaded manually.** Interactive
   runs are not saved as versions unless the human clicks **Save Version → Quick Save**.
3. **Kaggle CLI is the automation channel** (no Kaggle MCP exists). It works fully for datasets,
   kernels, status, and output downloads.
4. **Two accounts:**
   - `gunjanpal001` — token in the OS env `KAGGLE_API_TOKEN` (default).
   - `gunjanpal` — token stored at `%TEMP%\opencode\kg_gunjanpal.txt`
     (`C:\Users\Gunjan\AppData\Local\Temp\opencode\kg_gunjanpal.txt`). Never commit it. Rotate after use.
5. **Private datasets cannot be shared across accounts**, and challenge data must not be public
   (RULES §2) → datasets are mirrored under each account (see §4).
6. **Kaggle CLI gotchas already hit (do not repeat):**
   - `datasets create` **skips subfolders** unless `--dir-mode` is given → upload **flat** files.
   - Windows console **charmap** breaks progress output → run the CLI from the **venv**
     (`.venv\Scripts\python.exe -m kaggle`) with `$env:PYTHONUTF8='1'`.
   - A competition source (`arc-prize-2026-arc-agi-3`) cannot be attached via CLI → **400 Bad Request**.
   - A column named `row` breaks DuckDB joins (reserved) → use `row_idx` (F16-A).
   - DuckDB `ORDER BY` external sort on Colab OOMed (F16-B) → prefer numpy argsort / GPU gather.
7. **Kaggle T4x2 runtime** has `sentence-transformers 5.4.1` preinstalled, `torch 2.10.0+cu128`,
   CUDA available, **no faiss** (pip-installable, internet works), 31 GB RAM, 1.1 TB scratch disk.

### 1.3 Human-in-the-loop protocol (GPU stages)

> **Blackwell requires Internet OFF.** On this account the RTX Pro 6000 is only offered when the
> notebook has **Internet OFF**; with Internet ON the accelerator falls back to T4x2. Therefore every
> dependency needed at runtime (Python packages, model weights) must be **pre-staged in a Kaggle
> dataset** and attached — no `pip`/Hugging Face at run time.
> Offline dataset: **`gunjanpal/amz-er-2026-offline`** = a `faiss-cpu` cp312 manylinux wheel (flat)
> + `multilingual-e5-small.zip` (the pinned e5 model). The notebook installs faiss with
> `pip install --no-index --find-links <wheels> faiss-cpu` and unzips the model, then loads it locally.

For any stage that needs the Blackwell:

1. Agent writes/updates the notebook and **pushes it** to `gunjanpal/the-gpu-one` via CLI
   (this triggers a throwaway T4 run; include a guard that exits fast if the GPU is not the RTX Pro 6000).
2. Agent tells the human: *open `the-gpu-one` → Accelerator = GPU RTX Pro 6000 → Internet ON → Run All*.
3. Human runs it, then **Save Version → Quick Save** (persists outputs) or downloads files manually.
4. Agent pulls outputs with `kaggle kernels output gunjanpal/the-gpu-one -p <dir>` and continues locally.

**Notebook guard pattern (must be present):**
```python
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
if "RTX PRO 6000" not in gpu.upper():
    raise SystemExit("needs interactive RTX Pro 6000, got %s" % gpu)
```

---

## 2. Baseline, objective, and honesty about 0.99

- **Frozen baseline:** held-out full-candidate macro F0.5 = **0.8577** (tag `2.0.0`; adopted 1.4.1 model).
- **Candidate recall:** full train 0.8115 (US 0.868 / India 0.727); held-out pair recall 0.8142.
- **Oracle** (perfect matcher on current candidates) = **0.9122** (US 0.9468 / India 0.8602).
- **Scale:** 304,759,423 train candidates; 250,607,135 test candidates; ~138 cand/S1; cap 200.
- **Official public leaderboard (submitted 2026-09-26): 0.811.**
- **Objective:** measure the attainable candidate-recall ceiling, then maximize macro F0.5 at that ceiling.

**On 0.99:** it is **not** a credible target on current evidence. Even a perfect matcher caps at
0.9122 (candidate recall 0.8142). Reaching 0.99 needs candidate recall ≈0.99+ *and* precision ≈0.99
*and* generalization to unseen France (~15% of test, LOO penalty 0.086–0.121). Report **measured
values only**; never claim 0.99. Realistic aim: lift the ceiling via E3/E4 and the leaderboard from
0.811 toward roughly 0.85–0.88, confirmed by measurement.

---

## 3. Global rules (binding)

1. **Never destroy the 0.8577 baseline.** Snapshot exists at
   `code/business_entity_resolution/models_baseline_2.0.0/` (git-ignored). Git tag `2.0.0` must not move.
2. **Experiment isolation:** experiments write to `models/exp_<ID>/` and `DATA/reports/exp_<ID>_*.json`.
   `models/` is updated only by an explicit **PROMOTE** after a gate passes.
3. **Seed 42 everywhere; deterministic.** No external labeled data/APIs (RULES §1). Models MIT/Apache-2.0
   and ≤8B params. Outputs UTF-8 TSV. Validator **PASS** before any submission.
4. **Candidate set may change ONLY in E3/E4.** After the E4 freeze, E5+ must use the identical frozen
   candidate parquet.
5. **Never commit** `DATA/`, `output/`, `dist/`, `*.npy`, embeddings, `DATA/colab_in|out/`, `DATA/kaggle/`.
6. **Phase separation:** retrieval (E3–E4) → matcher (E1 feature merge, E5) → decision (E7–E9) → calibration (E11).
7. **Log every run:** append to `docs/PROJECT_LOG.md` and write `DATA/reports/exp_<ID>.json` with fields:
   `exp_id, date, commit, split, candidate_set_id, K, adaptive, pair_recall, entity_recall,
   recall_by_country, zero_candidate_s1, candidates_per_s1, total_candidates, oracle_macro_f05,
   heldout_f05, precision, recall, wall_s, peak_ram_gb, peak_vram_gb, adopted(bool), notes`.
8. **Tags:** `2.1.0` = retrieval/features adopted, `2.2.0` = matcher adopted, `3.0.0` = final submission.

---

## 4. Data assets and where they live

| Asset | Local path | `gunjanpal001` slug | `gunjanpal` slug |
|---|---|---|---|
| Raw TSVs (7) | `DATA/student_resource/dataset/{train,test}/*.tsv` | `amz-er-2026-raw` | `amz-er-2026-raw` (flat) |
| E1 pairs | `DATA/colab_in/pairs_{train,valfull}.parquet` | `amz-er-2026-e1-pairs` | `amz-er-2026-e1-pairs` |
| Held-out inputs | `DATA/pairs/valfull_pairs.parquet`, `DATA/reports/valfull_s1_ids.parquet` | `amz-er-2026-e1-valfull` | `amz-er-2026-e1-valfull` |
| Candidates | `DATA/candidates/{train,test}_candidates.parquet` | — (local only) | — |
| Ground truth | `DATA/processed/train_ground_truth.parquet` | in raw | in raw |
| Feature vectorizer | `DATA/processed/char3_vectorizer.pkl` | — | — |

All Kaggle datasets are **private**. `DATA/kaggle/` holds staging folders (git-ignored).

---

## 5. Existing code/artifacts inventory (built, tests passing: 52)

- `code/business_entity_resolution/src/ber/`
  - `colab_io.py` — `export_entities` joins **raw** `business_name`/`business_address` from TSVs (D-2);
    `COSINE_COLUMNS = ("name_e5_cos","addr_e5_cos","entity_e5_cos")`; `merge_cosine`.
  - `features.py` — `FEATURE_ORDER` currently **36 features + 3 e5 cosines appended (UNCOMMITTED)**.
  - `diagnostic.py` — `run_diagnostic(cfg, candidates_path=None, out_path=None, label=None)` emits the
    full tuple incl. `recall_by_country`, `candidates_per_s1`, `total_candidates`, `heldout_candidates`.
  - `embedding_block.py` — `AnnIndex`/`build_ann_index`/`retrieve`/`union_with_lexical`/`channel_attribution`
    (faiss ivf_pq/ivf_flat or exact numpy fallback) + `tests/test_embedding_block.py`.
- `notebooks/`
  - `e1_runner.py` — env-agnostic T4/Colab streaming E1 (entities from raw TSVs, field-at-a-time memmap,
    numpy-argsort sequential cosine). Proven on T4 smoke + full run.
  - `e1_blackwell.py` — interactive Blackwell E1 (resident GPU tensors, batched gather+dot, guard).
  - `e3_kaggle.py` — E3 ANN ceiling runner (train-only encode → IVF-PQ → per-truth-pair min rank).
  - `colab_embeddings.ipynb` — Colab wrapper (embeds `e1_runner.py`).
- `docs/` — `PROJECT_LOG.md`, `DECISIONS.md`, `FAILURES_AND_FIXES.md`, `RESULTS.md`.
- `DATA/reports/exp_E2.json` — E2 baseline tuple (recorded).
- `code/business_entity_resolution/models_baseline_2.0.0/` — frozen baseline snapshot.

**Uncommitted change to be aware of:** `features.py::FEATURE_ORDER` has `name_e5_cos`, `addr_e5_cos`,
`entity_e5_cos` appended. If the E1 gate fails, **revert** `FEATURE_ORDER` (and `models/`).

---

## 6. Phase plan

### E0 — Baseline freeze ✅ DONE
Snapshot `models_baseline_2.0.0/`; 45→52 tests pass; tags/disk/Colab verified.

### E2 — Diagnostics on current candidates ✅ DONE
`DATA/reports/exp_E2.json`: recall 0.81155, oracle 0.91216, held-out 0.85772.

### E1 — Full-dim e5 cosine features + gate

**E1g (canonical, Blackwell interactive).**
- Owner: **human runs**, agent prepares.
- Notebook: `notebooks/e1_blackwell.py`, pushed to `gunjanpal/the-gpu-one` (datasets: `amz-er-2026-raw`,
  `amz-er-2026-e1-pairs`; Internet ON; RTX Pro 6000).
- Outputs (in `/kaggle/working`): `cosine_train.parquet`, `cosine_valfull.parquet`, `cosine_e5_out.zip`.
- Human: Run All → Save Version → Quick Save. Agent: `kaggle kernels output gunjanpal/the-gpu-one -p DATA/kaggle/gpuone_out`.

**E1a (T4x2 API, fallback/cross-check).** `gunjanpal001/amz-er-e1-embeddings` (already ran once; outputs in
its `/kaggle/working`). Use only if E1g fails.

**E1 gate (agent, local):**
1. Copy cosine files to `DATA/colab_out/cosine_{train,valfull}.parquet`.
2. `ber.cli colab-merge` → attaches `name_e5_cos/addr_e5_cos/entity_e5_cos` to `DATA/features/train.parquet`
   and `DATA/tmp/valfull_features`.
3. Ensure `FEATURE_ORDER` has the 3 e5 features (already appended, uncommitted).
4. `ber.cli train` (writes `models/lgbm.txt` + `feature_list.json`; overwrites `models/threshold.json` provisional).
5. `ber.cli validation --workers 8` (regenerates `DATA/tmp/valfull_pred` + `eval_full_candidates.json`).
6. Read-only calibration score: `score_calibration(cfg, load_calibration(models/threshold.json))`.
7. **Gate:** held-out > **0.8577** → **PROMOTE** (commit FEATURE_ORDER + models; tag `2.1.0`).
   Else **revert** `FEATURE_ORDER` and restore `models/` from `models_baseline_2.0.0/`.
8. Record `DATA/reports/exp_E1.json`.

**E1b (optional, Blackwell) — larger encoder sweep.** Try `intfloat/multilingual-e5-large` (1024-dim) or
`BAAI/bge-m3` for the same 3 cosines. Gate: held-out > E1. Keep e5-small as control. License-check first.

### E3 — ANN K-sweep (measure the ceiling)

- Owner: **human runs** on Blackwell (fast) — or agent on T4x2 (slower) as fallback.
- Notebook: `notebooks/e3_kaggle.py` (train-only encode of 12.5M entities → IVF-PQ over S2/S3 →
  held-out S1 query at K=2000 → per-truth-pair min ANN rank). Datasets: `amz-er-2026-raw`, `amz-er-2026-e1-valfull`.
- Outputs (small, in `/kaggle/working`): `e3_minrank_{name,addr,entity}.parquet`, `e3_summary.json`.
- **Local post-processing (agent):** combine the min-rank tables with the local lexical candidate pairs
  (`DATA/pairs/valfull_pairs.parquet`) and GT to compute:
  - pair recall, entity recall, per-country recall, zero-candidate S1, candidates/S1, total candidates, oracle F0.5;
  - **recall-vs-candidate-count** and **oracle-F0.5-vs-cost** curves; interpolate the candidate cost to reach
    0.90/0.95/0.98/0.99/0.995 (record "not reached at K=2000" if applicable — never fabricate);
  - channel attribution for truth pairs missed by the lexical blocker (e5-name/addr/entity, char-ngram, phonetic,
    multiple, none).
- Write `DATA/reports/exp_E3-K{K}.json` per K and `DATA/reports/exp_E3_curves.json`.
- **Gate: none — STOP and present results before E4.**

### E4 — Multi-channel retrieval + adaptive K

- Channels: e5-name, e5-addr, e5-entity, char 3/4/5-gram TF-IDF (+SVD ANN), whole-name phonetic
  (**never per-token Metaphone** — D4/F1).
- `ber/embedding_block.py` production channels; `ber/blocking.py` `run_block_hybrid` unioning lexical +
  ANN; new pass ids **11–15**; adaptive `K(s1)`; global candidate budget (target ≤600M train).
- `config.json`: `ann{channels,k_max,adaptive,budget}`, `pass_caps` 11–15, raised `cap`.
- Rebuild `DATA/candidates/{train,test}_candidates.parquet`.
- **Freeze rule:** write `DATA/reports/candidate_set_frozen.json` (candidate_set_id, sha256 of both parquets,
  recall, oracle, counts). E5+ must use exactly these parquets.
- Gate rules: <0.90 keep improving; 0.90–0.95 continue; 0.95–0.98 continue if oracle rising; ≥0.98 weigh cost;
  ≥0.995 freeze immediately. Do not freeze merely at >0.95.

### E5 — Matcher improvements (LightGBM, local CPU)
- Features: existing RapidFuzz + address + char-3, plus char-4/5 cosine, e5 cosines, `ret_rank_{name,addr,entity}`,
  `ret_score_*`, `ret_channel_count`, phonetic similarity.
- Files: `ber/features.py`, `ber/colab_io.py::merge_cosine`, `ber/train.py`, `ber/pairs.py`.
- Gate: held-out > previous adopted. Revert on failure. Candidate set frozen.
- **Fallback:** if E5 doesn't beat E4's matcher, revert and try E6 (CE may capture what LGBM didn't).

### E6 — Cross-encoder rerank (Blackwell interactive)
- `ber/rerank.py` + notebook; `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0) first; larger ≤8B only if it helps.
- top-K=10/S1 + boundary band; `ce_score` as a feature; never a hard override.
- Gate: held-out > E5 and precision not worse.

### E7 — Global assignment (local CPU)
- `ber/assignment.py`: max-weight bipartite / min-cost flow with explicit NO-MATCH dummy nodes;
  `scipy.optimize.linear_sum_assignment` or networkx; never force a match. Replace greedy in `ber/predict.py::_write_tsv`.
- Gate: held-out > E6. Revert to greedy one-to-one if it hurts.

### E8 — Transductive consistency / pseudo-labeling (local CPU)
- `ber/pseudo.py`: deterministic high-confidence pseudo-labels only; alias/address cluster consistency;
  no val/test labels into training. Gate: held-out > E7 and stable across ≥2 seeds.

### E9 — DeepSeek selective reranking (Blackwell interactive, LAST)
- `ber/deepseek_rerank.py`: `DeepSeek-R1-Distill-Qwen-1.5B` (verify MIT, pin revision), single forward pass
  yes/no logits on a capped boundary subset; feature/rerank signal only. Gate: held-out > E8, else discard.

### E10 — Ensemble / meta-model (local CPU)
- `ber/meta.py`: calibrated stack of LightGBM + CE + retrieval (+ optional DeepSeek); seed ensemble; grouped CV.
- Gate: held-out > E9. Revert: keep best single model.

### E11 — Calibration + test inference + submission (local CPU)
- `ber/calibration.py` (unchanged semantics), `ber/predict.py` (thresholds + assignment), `ber/cli.py`.
- Country thresholds (US/India) + global fallback for France + singleton rule + assignment.
- `ber.cli features --split test`, `ber.cli predict --split test`; validator PASS; package.
- Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `dist/AA.._submission/`,
  `dist/AA..__submission.zip`, `submission/matching_results.tsv`, `dist/leaderboard_upload/matching_results.tsv`.
- Gate: validator PASS and invariants (one row/S1; matches ⊆ candidates; empty for singletons). Tag `3.0.0`.
- Human uploads to the Portal (agent must not).

---

## 7. Detailed interactive-Blackwell workflow (E1g and later E6/E9)

### 7.1 Preparing the notebook (agent)

1. Edit `notebooks/e1_blackwell.py` (or the relevant runner).
2. Generate the notebook + metadata into `DATA/kaggle/gpu_probe_nb/`:
   - `the-gpu-one.ipynb` (markdown intro + one code cell containing the script).
   - `kernel-metadata.json`:
     ```json
     {
       "id": "gunjanpal/the-gpu-one",
       "title": "the gpu one",
       "code_file": "the-gpu-one.ipynb",
       "language": "python",
       "kernel_type": "notebook",
       "is_private": true,
       "enable_gpu": true,
       "enable_internet": false,
       "enable_tpu": false,
       "machine_shape": "NvidiaRtxPro6000",
       "dataset_sources": ["gunjanpal/amz-er-2026-raw", "gunjanpal/amz-er-2026-e1-pairs",
                            "gunjanpal/amz-er-2026-e1-valfull", "gunjanpal/amz-er-2026-offline"],
       "competition_sources": [], "kernel_sources": [], "model_sources": []
     }
     ```
3. Push with the `gunjanpal` token:
   ```powershell
   $env:PYTHONUTF8='1'
   $env:KAGGLE_API_TOKEN=(Get-Content "$env:TEMP\opencode\kg_gunjanpal.txt" -Raw)
   .venv\Scripts\python.exe -m kaggle kernels push -p "DATA\kaggle\gpu_probe_nb"
   ```
   The API run will ERROR fast (guard) — that is expected and costs ~1 min of T4.
4. Tell the human the exact steps.

### 7.2 Running (human)

1. Open `https://www.kaggle.com/code/gunjanpal/the-gpu-one/edit`.
2. Session options → **Accelerator = GPU RTX Pro 6000**; **Internet = ON**.
3. **Run All**. Watch the `[e1b]`/`[e3]` progress logs.
4. On completion: **Save Version → Quick Save** (persist outputs) — do **not** use "Save & Run All" (re-runs, likely T4).
   If Quick Save is unavailable, download the output files from the notebook's output file browser.

### 7.3 Retrieving outputs (agent)

```powershell
$env:PYTHONUTF8='1'
$env:KAGGLE_API_TOKEN=(Get-Content "$env:TEMP\opencode\kg_gunjanpal.txt" -Raw)
.venv\Scripts\python.exe -m kaggle kernels output gunjanpal/the-gpu-one -p "DATA\kaggle\gpuone_out"
```
If outputs are not present (Quick Save skipped), ask the human to place the downloaded files in
`DATA/colab_out/` (E1) or `DATA/reports/` (E3) manually.

---

## 8. E3 local curve computation (agent)

Inputs:
- `e3_minrank_{field}.parquet` → `(s1_id, cand_id, min_rank)` for held-out truth pairs recovered by ANN.
- `DATA/pairs/valfull_pairs.parquet` → lexical candidate pairs (held-out).
- `DATA/processed/train_ground_truth.parquet` + `DATA/reports/valfull_s1_ids.parquet` → held-out truth.

Algorithm per field and K:
1. `recovered(K) = min_rank < K` (per field) OR lexical-found.
2. Per-S1: `ntrue`, `nfound(K) = |truth ∩ (lexical ∪ ANN_K)|`.
3. Metrics: pair recall = Σnfound/Σntrue; entity recall = mean(nfound/ntrue) over truth-bearing S1;
   zero-candidate S1 = count(nfound==0); candidates/S1 ≈ K per field (report ANN rows per K from `e3_summary.json`);
   oracle F0.5 via `_oracle_scores(ntrue, nfound)` (reuse `ber/diagnostic.py`).
4. Union across channels: `min_rank_union = min over fields`; recompute.
5. Curves + interpolation to 0.90/0.95/0.98/0.99/0.995; channel attribution for lexical-missed truth pairs.

Write `DATA/reports/exp_E3_curves.json` and per-K `exp_E3-K{K}.json`. **Stop and present.**

---

## 9. Execution checklist (sequential, with owner)

| # | Step | Owner | Command / action | Gate |
|---|---|---|---|---|
| 1 | E0 verify | agent | `pytest -q` (52 tests); confirm `models_baseline_2.0.0/` | tests pass |
| 2 | E1g notebook ready | agent | push `e1_blackwell.py` to `the-gpu-one` | pushed |
| 3 | E1g run | **human** | Run All on RTX Pro 6000, Internet ON, Quick Save | completes |
| 4 | E1g retrieve | agent | `kaggle kernels output` | cosine files present |
| 5 | E1 merge | agent | `ber.cli colab-merge` | e5 cols in features |
| 6 | E1 train+validate | agent | `ber.cli train`; `ber.cli validation --workers 8` | runs |
| 7 | E1 gate | agent | read-only `score_calibration` | **>0.8577 → promote (tag 2.1.0) else revert** |
| 8 | (opt) E1b | human+agent | larger encoder, same protocol | >E1 |
| 9 | E3 notebook ready | agent | push `e3_kaggle.py` (datasets raw+valfull) | pushed |
| 10 | E3 run | **human** | Run All on RTX Pro 6000 | completes |
| 11 | E3 retrieve + curves | agent | download + local computation | **STOP for review** |
| 12 | E4 build+audit | agent (+human GPU if needed) | multi-channel + adaptive K; `ber.cli audit` | freeze `candidate_set_frozen.json` |
| 13 | E5 matcher | agent | features + `ber.cli train`/`validation` | held-out > prior |
| 14 | E6 CE | human+agent | rerank notebook | held-out > E5 |
| 15 | E7 assignment | agent | `ber/assignment.py` | held-out > E6 |
| 16 | E8 pseudo | agent | `ber/pseudo.py` | >E7, 2-seed stable |
| 17 | E9 DeepSeek | human+agent | rerank notebook | >E8 else discard |
| 18 | E10 meta | agent | `ber/meta.py` | >E9 |
| 19 | E11 submit | agent (+human upload) | calibrate/predict/validate/package | validator PASS; tag 3.0.0 |
| 20 | Docs + graphify | agent | update `docs/*`, `AGENTS.md`, `README`; refresh graphify | done |

---

## 10. Fallbacks

1. **E1g fails/blocked** → run E1a on T4x2 (automated) or Colab free.
2. **E5 doesn't beat E4 matcher** → revert; go to E6 (CE).
3. **ANN recall saturates <0.95** → add channels (SVD/char variants, phonetic) + adaptive K; document the ceiling.
4. **0.95–0.98 reached but 0.99 too costly** → freeze at the knee; record "0.99/0.995 not economical at K≤2000".
5. **Recall ≥0.995** → freeze immediately; matcher-first.
6. **CE hurts precision** → drop as feature, keep as boundary reranker; else discard.
7. **Global assignment hurts** → keep greedy one-to-one.
8. **Transductive instability** → require determinism + 2-seed agreement; else discard.
9. **DeepSeek no gain** → discard.
10. **GPU limits** → record failing stage/resource; stop at the highest completed K; never fabricate.

---

## 11. Reproduce commands (local, for reference)

```powershell
# tests
.venv\Scripts\python.exe -m pytest -q

# pipeline
$env:PYTHONPATH="code/business_entity_resolution/src"
.venv\Scripts\python.exe -m ber.cli prepare   --config code/business_entity_resolution/config.json
.venv\Scripts\python.exe -m ber.cli block     --split both --config code/business_entity_resolution/config.json
.venv\Scripts\python.exe -m ber.cli audit     --split train --config code/business_entity_resolution/config.json
.venv\Scripts\python.exe -m ber.cli features  --split train --workers 8 --combine --config code/business_entity_resolution/config.json
.venv\Scripts\python.exe -m ber.cli train     --config code/business_entity_resolution/config.json
.venv\Scripts\python.exe -m ber.cli validation --workers 8 --config code/business_entity_resolution/config.json
.venv\Scripts\python.exe -m ber.cli calibrate --config code/business_entity_resolution/config.json
.venv\Scripts\python.exe -m ber.cli features  --split test --workers 8 --config code/business_entity_resolution/config.json
.venv\Scripts\python.exe -m ber.cli predict   --split test --one-to-one --config code/business_entity_resolution/config.json
```

Kaggle CLI (account-specific):
```powershell
# gunjanpal001 (default OS env token)
.venv\Scripts\python.exe -m kaggle kernels status gunjanpal001/amz-er-e1-embeddings

# gunjanpal
$env:KAGGLE_API_TOKEN=(Get-Content "$env:TEMP\opencode\kg_gunjanpal.txt" -Raw)
.venv\Scripts\python.exe -m kaggle kernels status gunjanpal/the-gpu-one
```

---

## 12. Key paths & slugs reference

- Baseline snapshot: `code/business_entity_resolution/models_baseline_2.0.0/`
- Adopted model: `code/business_entity_resolution/models/{lgbm.txt,feature_list.json,threshold.json,training_metrics.json}`
- Candidates: `DATA/candidates/{train,test}_candidates.parquet`
- Held-out preds: `DATA/tmp/valfull_pred/`; features `DATA/tmp/valfull_features/`
- Reports: `DATA/reports/exp_*.json`, `eval_oracle.json`, `eval_full_candidates.json`
- Kaggle: `gunjanpal/the-gpu-one` (interactive Blackwell), `gunjanpal001/amz-er-e1-embeddings` (T4),
  datasets `gunjanpal/amz-er-2026-{raw,e1-pairs,e1-valfull}`, `gunjanpal001/amz-er-2026-{raw,e1-pairs,e1-valfull}`
- Token: `%TEMP%\opencode\kg_gunjanpal.txt` (gunjanpal); OS env `KAGGLE_API_TOKEN` (gunjanpal001)

---

## 13. Documentation duties (per AGENTS.md)

After each phase: append to `docs/PROJECT_LOG.md`; update `docs/RESULTS.md` on new believable metrics;
add a `docs/DECISIONS.md` entry for adopted changes; add a `docs/FAILURES_AND_FIXES.md` entry for every
non-trivial bug (F16-style: symptom, cause, fix). Refresh `graphify-out/` at milestones.


---

## 14. Remaining steps — ordered checklist, estimates, and E4 spec

### 14.1 Order (dependencies are strict)
1. **E4 decide** operating point from E3 curves (local).
2. **E4 build** production candidates + cosines for train/valfull/test (Blackwell interactive).
3. **E4 audit + FREEZE** (candidate_set_frozen.json, sha256) (local).
4. **E5** matcher on the frozen candidates (local) -> gate > 0.85900.
5. **E6** cross-encoder (Blackwell, optional) -> gate > E5.
6. **E7** global assignment (local) -> gate > E6/E5.
7. **E8** pseudo-labeling (local, optional) -> gate > E7, 2-seed stable.
8. **E9** DeepSeek rerank (Blackwell, optional, last) -> gate > E8.
9. **E10** ensemble/meta (local, optional) -> gate > E9.
10. **E11** calibrate -> test features -> predict -> validator PASS -> package TSV (local).

Never tune thresholds while candidates change; freeze before E5; calibrate last.

### 14.2 Time estimates (rough; measured where noted)
| Step | Where | Estimate |
|---|---|---|
| E4 build (re-encode + ANN index train+test + cosines) | Blackwell | 2.5-3.5 h |
| E4 audit + freeze | local | 5-15 min |
| E5 features / train / validation | local | 40-90 min / ~10 min / 10-15 min |
| E6 cross-encoder | Blackwell | 2-4 h |
| E7 assignment | local | 10-30 min |
| E8 pseudo | local | 1-2 h |
| E9 DeepSeek | Blackwell | 1-2 h |
| E10 meta | local | 20-40 min |
| E11 calibrate / test features / predict+validate+package | local | 5 min / 30-60 min / 15-30 min |
Minimum viable path (E4->freeze->E5->E11): ~5-7 h compute + one ~3 h Blackwell run.

### 14.3 E4 operating point (from measured E3 curves)
The E3 nn_pairs_per_s1_upper = K * 3 (3 channels, pre-dedup). To respect a train budget of
<=600M candidates (~270 candidates/S1 over 2.2M S1), K per channel must be small:
- K=50/channel -> ~150/S1 upper, pair recall 0.9496, oracle 0.9810
- K=100/channel -> ~300/S1 upper, pair recall 0.9544, oracle 0.9830
K=2000 (recall 0.9700 / oracle 0.9892) is **not** budget-feasible (would be billions of candidates).
**Chosen starting point: K=100/channel with per-S1 cap 250 and a <=600M train budget, adaptive K up
for India / short / non-Latin names.** Audit, then adjust K/cap to the best recall/cost knee.

### 14.4 E4 candidate contract
{split}_candidates.parquet = (s1_id, cand_id, pass_id, block_score, is_s2) plus per-channel

ank_<ch> and score_<ch> and 
et_channel_count; pass ids 11=e5-name, 12=e5-addr, 13=e5-entity,
14=char-ngram, 15=phonetic (whole-name, never per-token). Union with lexical passes 1-10.
Also emit cosine_{train,valfull,test}.parquet (
ame_e5_cos,ddr_e5_cos,entity_e5_cos) for the
**union** pairs (E1 cosines covered only the old lexical pairs).

### 14.5 E4 Blackwell run outline (interactive)
1. Encode name/addr/entity for all 24.2M entities (as in E1g).
2. Build IVF-PQ indexes over train S2/S3 **and** test S2/S3 per field.
3. Query train held-out S1 + test S1 at K; keep per-channel rank/score.
4. Union with the existing lexical candidates; apply per-S1 cap / adaptive K / budget.
5. Compute e5 cosines for the union pairs (train/valfull/test).
6. Write candidate + cosine parquets to /kaggle/working; download.
