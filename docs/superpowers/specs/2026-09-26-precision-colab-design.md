# Design Spec — Precision Lift + Colab Embeddings (Milestone M2)

Date: 2026-09-26
Status: approved for implementation by a separate agent
Baseline: official leaderboard macro F0.5 = **0.811**; local held-out full-candidate = **0.8488**;
candidate (blocking) recall ceiling = **0.814**.
Target: in-domain held-out > 0.90, leaderboard > 0.85.

## 1. Goal

Increase macro F0.5 (β = 0.5, per Source 1 entity, singletons included) beyond the current shipped
pipeline. The metric is precision-heavy (precision weighted 2×), so the priority is reducing false
merges while preserving recall, then lifting the blocking recall ceiling if it proves binding.

## 2. Why this plan (bottleneck analysis)

- The candidate ceiling (0.814 pair-level) translates to roughly **0.95** in macro-F0.5 terms,
  because ~3.67 true matches per matched entity means most entities keep partial credit, and the
  23k singletons score 1.0 when correctly left empty.
- The current held-out score is 0.8488, i.e. **~0.10 below that ceiling** — so most of the loss is
  in the matcher (precision and candidate-level recall), not blocking.
- India is the weakest cell (0.788) and has both lower blocking recall (0.73) and the most matcher
  headroom.
- Therefore: **Phase 1–3 improve the matcher** (feature information + calibration + GPU rerank);
  **Phase 4 (blocking) is conditional** on the Phase-0 diagnostic showing recall is binding.

## 3. Environments

Local (measured): AMD Ryzen 7 350, 8C/16T, 23.3 GB RAM, no CUDA, D: ~118 GB free, Python 3.12 venv
`.venv`. Data and all intermediates are local.

Colab (verified by probe on 2026-09-26): **Tesla T4, 15.6 GB VRAM, compute capability 7.5,
CUDA available, `torch 2.11.0+cu128` preinstalled, 2 vCPU, 13.6 GB RAM, free tier.** Connection via
the Colab MCP tools works (`colab_open_colab_browser_connection` → true). Consequences: use **fp16**
(T4 has no bf16), shard/checkpoint long jobs, keep only GPU work on Colab.

## 4. Interfaces (small transfers)

Export (local → Colab), under `DATA/colab_in/`:

| File | Columns | Approx size |
|---|---|---|
| `entities.parquet` | `entity_id, name_norm, name_roman, addr_norm` (unique across all 6 sources) | ~1–2 GB |
| `pairs_train.parquet` | `s1_id, cand_id` | ~small (sampled 4:1 pairs) |
| `pairs_valfull.parquet` | `s1_id, cand_id` | ~60M rows |
| `pairs_test.parquet` | `s1_id, cand_id` | ~250M rows |
| `rerank_pairs_{split}.parquet` (Phase 3) | `s1_id, cand_id, s1_text, cand_text` (top-K per S1) | ~1.5 GB |

Import (Colab → local), under `DATA/colab_out/`:

| File | Columns | Notes |
|---|---|---|
| `cosine_{split}.parquet` | `s1_id, cand_id, emb_name_cos, emb_addr_cos` | fp16, test ≈ 0.5 GB |
| `rerank_{split}.parquet` (Phase 3) | `s1_id, cand_id, ce_score` | fp16 |

Models (license-compliant, no external data): `intfloat/multilingual-e5-small` (MIT);
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0).

## 5. Phases and adoption gates

| Phase | Where | Deliverable | Gate to adopt |
|---|---|---|---|
| 0 | local | `DATA/reports/eval_oracle.json`: oracle macro F0.5 (pred = GT ∩ candidates), per-country P/R, entity-recall distribution | decision input only |
| 1 | local | char 3–5-gram TF-IDF cosine features; per-country thresholds; singleton calibration | held-out full-candidate > 0.8488, India not worse |
| 2 | Colab T4 | embeddings per-pair cosine | held-out > Phase 1 |
| 3 (opt) | Colab T4 | cross-encoder rerank score | held-out > Phase 2 |
| 4 (cond) | local | MinHash-LSH blocking | ceiling > 0.814 and net F0.5 > Phase 3 |
| 5 | local | rebuilt `output/*.tsv`, validator PASS, zip, leaderboard file | validator PASS |

## 6. Validation protocol (reused, unchanged)

Held-out 20% of Source 1 groups (deterministic `grouped_split(..., 0.2, 42)`), scored on the **full
candidate set** for those groups, predicted exactly as inference (per-country threshold + one-to-one),
macro F0.5 with singleton rule; report precision/recall, per-country, and leave-one-country-out as
the France proxy. Artifacts under `DATA/reports/`.

## 7. Risks

- Colab free tier disconnects / ~12 h sessions → shard 500k rows, write idempotent shards, checkpoint.
- T4 fp16-only → cast to fp16; no bf16 assumptions.
- Unlabeled France → per-country thresholds only for US/India with a global fallback; use LOO to
  guard against overfitting.
- Transfer size → return per-pair cosine (~0.5 GB), not full embeddings (~15 GB).
- Candidate set must stay unchanged in Phases 1–3 so `candidate_pairs.tsv` remains valid.
- Fixed seed 42 everywhere; no external data; models MIT/Apache only.

## 8. Out of scope

Re-blocking with embeddings (unless Phase 4 triggers), transformer fine-tuning, changes to the
scoring formula, and any external data/API.
