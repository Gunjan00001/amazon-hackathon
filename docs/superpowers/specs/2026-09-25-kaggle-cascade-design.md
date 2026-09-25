# Design — Kaggle-only cascade (C+B) for Business Entity Resolution

- Date: 2026-09-25
- Status: draft for user review (Phase 0, repo version 0.2.0)
- Scope: compute rules (RULES.md §6, already appended) + cascade `C+B` pipeline + 24-hour execution schedule. The step-by-step implementation plan is a separate document produced with the writing-plans skill.

## 1. Context and constraints

Challenge: match noisy business records from Source 2 (5.03M train / 4.89M test rows) and Source 3 (5.29M / 5.08M) to a deduplicated Source 1 reference (2.21M / 1.73M). Metric: macro F_0.5 per Source 1 entity, singletons included, precision-heavy.

Measured facts that shape the design (see `DATA/student_resource/dataset/DATASET.md`):

- Ground truth is injective on the S2/S3 side: no matched ID is claimed by two S1 entities — a one-to-one assignment post-process is legitimate.
- 5.585% of train S1 entities are singletons; correct empty predictions score 1.0 and any false merge scores 0.0 for that entity.
- Test adds France (zero-shot, ~15% of rows); train is US/India only. Country is open-set — never filter or one-hot on it.
- ~15% of train S2 names are non-Latin scripts while Source 1 train names are ASCII — cross-script matching is required.
- ~3.3% of S2/S3 addresses are empty (real signal, not NaN).
- `tools/bench_gbdt.py` measured, on sampled candidate pairs: LightGBM pair AUC 0.9997, macro F_0.5 0.986–0.995 with 15 classical features. The matcher concept is already validated; the missing pieces are full-scale blocking, full-scale inference, and the GPU stages.

Hard constraints:

- **24-hour deadline.**
- **Kaggle only** (RULES.md §6): training and GPU inference on Kaggle Notebooks; T4 x2 baseline (~30 GPU-h/week). Runs are launched manually in the Kaggle web UI by the user; artifacts and small metrics come back to the local machine for inspection.
- Final models must be MIT/Apache-2.0 and <=8B parameters. No external data lookup, ever.

## 2. Approach — cascade C+B

A precision-first cascade: a cheap recall net produces candidates, GBDT prunes them, and a multilingual cross-encoder reranks the survivors. Every GPU stage is optional to the critical path — the classical baseline can ship alone.

```
S0 Kaggle private dataset `amz-er-2026-raw` (7 TSVs from D:\Amazon project\DATA)
   |
S1 DATA CLEANING (CPU)      UTF-8/NFKC, casefold, punctuation, legal-suffix expansion,
   |                        anyascii translit (Indic/French), address parse (tokens, PIN/ZIP,
   |                        state, landmark flags), dedup -> clean parquet + id maps
S2 BLOCKING (CPU)           multi-pass: IDF rare-name tokens + phonetic + address/PIN,
   |                        country gate -> candidate_pairs_raw, cap ~200/S1 (tuned), recall
   |                        ceiling measured against train ground truth
S3 FEATURE ENGINEERING      ONE versioned feature spec, identical at train/inference:
   |  (a) classical        Jaro/Jaccard/ratios/token-set/len/country/source (the 15 existing)
   |  (b) encoder (GPU)    multilingual-e5-small (MIT, 384-d) name+address vectors -> cosine,
   |                       abs-diff, component product
   |  (c) structural       translit-match, phonetic, postal/PIN match, component matches,
   |                       missing-address flags
S4 GBDT (CPU)               LightGBM over S3 features -> prune to top-K/S1 (K tuned so the
   |                        measured F0.5 recall ceiling is retained)
S5 CROSS-ENCODER (GPU)      mmarco-mMiniLMv2-L12-H384-v1 (Apache-2.0, 118M) reranks top-K:
   |                        zero-shot first; fine-tune on GT positives + S4 hard negatives only
   |                        if the schedule allows
S6 DECISION (CPU)           macro-F0.5 threshold tuned on entity-grouped split + country holdout
   |                        (France proxy); optional injective one-to-one assignment if it lifts
   |                        validation macro-F0.5
S7 OUTPUTS                  candidate_pairs.tsv = exact set S5 scored; matching_results.tsv =
                            IDs above threshold; validator gate -> submission zip
```

Why this shape:

- F_0.5 rewards precision 2x over recall, so the final decision is conservative and the expensive models only touch a small candidate set.
- The classical path already scores 0.986–0.995 macro F_0.5 on sampled pairs; it is the v1 submission and the fallback at every later step.
- Multilingual embeddings and the cross-encoder are what buy the cross-script and France-generalization headroom; the graph is implicit in the candidate set + optional one-to-one assignment rather than a separate clustering system, which keeps the 24-hour risk sane.

Model choices (license/parameter limits verified at implementation preflight, pinned by revision):

| Role | Model | Params | License | Notes |
|---|---|---|---|---|
| Encoder (S3) | `intfloat/multilingual-e5-small` | 118M | MIT | 384-d keeps the vector cache ~9 GB fp16 for ~11.7M test records |
| Reranker (S5) | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 118M | Apache-2.0 | Fits T4 x2 inference budget |
| Reranker fallback | `BAAI/bge-reranker-base` | 278M | MIT | Only if quota and time allow |
| Matcher | LightGBM | n/a | MIT | Default per RULES §3.7 |

## 3. Kaggle execution model

Datasets (both private, user-uploaded):

- `amz-er-2026-raw` — the 7 challenge TSVs (2.35 GiB, upload from `D:\Amazon project\DATA\student_resource\dataset`, skip `.DS_Store`).
- `amz-er-2026-code` — pinned snapshot of `code/business_entity_resolution/` so every notebook imports the same source; versioned with each repo tag.

Notebook chain (run manually, in order; each promotes outputs to a versioned dataset):

| Notebook | Accelerator | Input | Output (heavy) | Paste back |
|---|---|---|---|---|
| N1 clean | CPU | raw | clean parquet, id maps, stats | `metrics.json`, 10 sample rows |
| N2 block | CPU | N1 parquet | candidate pairs (parquet shards) | `metrics.json` incl. recall ceiling |
| N3 embed | T4 x2 | N1 parquet | fp16 vectors (sharded, <20 GB/session) | `metrics.json` |
| N4 gbdt | CPU | N2 + N3 (+ labels) | trained model, pruned top-K candidates | `metrics.json` (AUC, macro F0.5, threshold) |
| N5 rerank | T4 x2 | N4 top-K | CE scores | `metrics.json` (AP, macro F0.5, lift) |
| N6 decide | CPU | N5 scores | `candidate_pairs.tsv`, `matching_results.tsv` | `metrics.json`, validator output |

Quota budget: N3 ~2–4 h + N5 ~2–4 h per full pass ≈ 4–8 GPU-h of the ~30 h/week. Re-runs must reuse cached dataset outputs. Sessions capped at 9–12 h and outputs at ~20 GB, so all caches are sharded parquet/npy.

Local machine: Python 3.12 venv, CPU only — editing, unit tests, macro-F0.5 scorer, `utils/validate_submission.py`. Repository code resolves inputs via `DATA_DIR`/stage input dirs; no hard-coded `D:\Amazon project` paths survive (current `tools/bench_gbdt.py` hard-codes them).

## 4. Stage details

**S1 cleaning.** NFKC + casefold; punctuation/whitespace normalization; legal-suffix expansion (`Pvt`→`Private`, `Ltd`→`Limited`, ...); `anyascii` transliteration for cross-script keys; address parsing into tokens, PIN/ZIP, state, landmark flags; output parquet keyed by `entity_id` plus stable id maps. Deterministic and cached.

**S2 blocking.** Union of passes, then a country gate (S1 country must equal S2/S3 country — test France included automatically since it never filters on a fixed set): (a) inverted index on IDF-filtered rare name tokens, (b) phonetic key on transliterated names, (c) address token overlap and PIN/ZIP exact keys. Per-S1 cap around 200 candidates (exact cap tuned in N2 against measured ground-truth recall). Output documents the recall ceiling — the hard upper bound for everything downstream.

**S3 feature engineering.** One feature module used unchanged for train, validation, and test: (a) 15 classical pair features already benchmarked, (b) encoder features (cosine plus abs-diff/product of name and address vectors; vectors computed only on unique records), (c) structural features (translit match, phonetic score, postal match, address-component matches, missing-address flags). No label-derived features; validation splits grouped by `source1_entity_id` with a country holdout.

**S4 GBDT.** LightGBM on S3 over the full candidate set; entity-grouped split; threshold search for macro F_0.5 with singleton rule; prune to top-K per S1 where K is chosen from the measured F_0.5-vs-K curve so the recall ceiling is retained.

**S5 cross-encoder.** Input serialization `name1 [SEP] addr1 [SEP] country1` vs the same for the candidate. Zero-shot scores first; fine-tuning (GT positives + S4 hard negatives, subsampled for quota) only if the schedule still has >3 h of buffer. CE score becomes the primary decision signal; GBDT score is retained as a fallback and as a stacking input if time allows.

**S6 decision.** Threshold tuned for macro F_0.5 on the grouped validation split; country holdout (hold out one train country, or an India/US slice, as the France proxy). Optional injective one-to-one assignment (greedy by score, since ground truth is injective) is enabled only if it lifts validation macro F_0.5; otherwise it stays off.

**S7 outputs.** `candidate_pairs.tsv` must be the exact set the final model scored (S5's input set when the reranker decides, otherwise S4's top-K); `matching_results.tsv` ⊆ candidates. Local validator gate (exit 0) before every upload.

## 5. 24-hour schedule

| Hours | Work | Milestone |
|---|---|---|
| 0–1 | Rules (done), this design, implementation plan; user uploads `amz-er-2026-raw` | rules + design + plan |
| 1–3 | N1 clean, N2 block on Kaggle CPU | recall ceiling measured |
| 3–8 | N4 features + LightGBM + threshold tuning | validation macro F_0.5 |
| 8–12 | N6 test inference + validator | **SUBMISSION v1 (classical)** |
| 12–16 | N3 embeddings + GBDT retrain | **SUBMISSION v2** |
| 16–20 | N5 cross-encoder zero-shot rerank | **SUBMISSION v3** |
| 20–24 | Final validation, submission zip + `Documentation_template.md`, README | package ready |

Fallbacks: if behind at hour 12, ship v1 and decide on v3 later; if behind at hour 20, skip the CE fine-tune; v1 needs zero GPU quota.

## 6. Verification and gates

- Unit tests (local, CPU): normalizers, blocking keys, feature calculators, macro-F0.5 scorer (singleton rule), TSV writer. Extend the existing `entity_macro_f05` implementation from `tools/bench_gbdt.py`.
- Stage gates: N2 recall ceiling (measured on grouped sample), N4 entity-level macro F_0.5 + chosen K, N5 rerank lift over S4, N6 final holdout macro F_0.5 with France proxy.
- Format gate: `utils/validate_submission.py` exit 0; matches ⊆ candidates; one row per test S1.
- Versioning: `0.2.0` = rules + design (this doc); `0.3.0` = pipeline v1 capability; `1.0.0` = submission-ready package. Version log updated in `README.md`.

## 7. Deliverables

- Validated `output/matching_results.tsv` + `output/candidate_pairs.tsv`, uploaded to the portal.
- Submission zip per the official structure: `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, filled `Documentation_template.md`.
- Repo updates: RULES.md §6 (done at approval), design doc (this file), implementation plan, pipeline source + tests, Kaggle notebook chain, fixed path configuration.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Manual run loop latency | Notebooks pre-validated locally; small paste-back metrics designed for fast iteration |
| GPU quota exhaustion | Cache-first artifacts; v1 classical path needs no GPU; CE fine-tune optional |
| Indic-script matching | Multilingual encoder + transliteration keys + phonetic features |
| France zero-shot | Country-agnostic features, held-out-country threshold tuning, no country filtering |
| One-to-one assignment hurting recall | Off by default; enabled only on measured F0.5 improvement |
| Kaggle session/output limits | Sharded parquet/npy caches; stages split N1–N6 |
| Stale hardcoded data paths | Central `DATA_DIR`/input-dir resolution; fixed during implementation |

## 9. Open items for the implementation plan

- Exact blocking passes, caps, and shard layout; K-vs-F0.5 curve procedure.
- Exact feature serialization for the cross-encoder and batching/quota estimates.
- Notebook cell structure, package pins, and per-notebook paste-back checklists.
- Local test fixtures and CI-style commands for the validator.
