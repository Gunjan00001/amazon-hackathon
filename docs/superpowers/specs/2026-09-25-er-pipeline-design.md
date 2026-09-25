# Design Spec — Business Entity Resolution Pipeline

Date: 2026-09-25
Status: approved for implementation
Challenge: Amazon ML Challenge 2026 — Business Entity Resolution
Metric: macro F_0.5 (β = 0.5), precision-heavy, singletons included

## 1. Context and goal

Match business records from Source 2 and Source 3 to the deduplicated reference Source 1. The deliverable is a reproducible submission package whose scored artifact is `output/matching_results.tsv`, plus `output/candidate_pairs.tsv` (the final pre-model candidate set), runnable code, pinned dependencies, and a filled methodology document. All processing happens locally (Windows, Ryzen 8C/16T, 23 GB RAM, no CUDA) via a Python 3.12 venv; DuckDB provides out-of-core relational work.

Reference facts: `DATA/student_resource/dataset/DATASET.md` (measured), `PROBLEM_STATEMENT.md` (official), `RULES.md` (binding project rules).

## 2. Locked decisions

| Decision | Choice |
|---|---|
| Blocking engine | DuckDB + pandas/pyarrow |
| Blocking objective | Recall-first: target ≥97% train candidate recall, per-S1 cap tuned near 200 |
| Matcher | Single LightGBM binary classifier; XGBoost drop-in retained behind same feature matrix |
| Transliteration | Rule-based Romanization (indic-transliteration) as features; no external data |
| Split policy | Group by `source1_entity_id`; leave-one-country-out for France generalization |
| Post-processing | Optional one-to-one assignment, adopted only if held-out macro F_0.5 improves |
| Raw data | Read-only, UTF-8, `sep="\t"`, `keep_default_na=False` |

Measured data facts that shape the design:

- Train rows: S1 2,206,821 / S2 5,034,616 / S3 5,285,603; test rows: 1,732,544 / 4,887,273 / 5,082,316.
- Ground truth: 7,638,365 pairs, injective (each S2/S3 ID in exactly one S1 list), 5.585% singletons, mean 3.461 matches, max 11.
- Countries: train {US, India}; test adds France (~15%).
- ~15% of train S2 names are non-Latin; Source 1 train names are 100% ASCII.
- Empty `business_address` in 3.36% (S2) and 3.33% (S3) of train rows; 2.65%/2.68% in test.
- No duplicate `entity_id` anywhere. All GT IDs exist in their source files.

## 3. Architecture and data flow

```
dataset/*.tsv
  │ Stage 0  ingest + clean (chunked)
  ▼
data/processed/{split}_{source}.parquet      (normalized + parsed, raw preserved)
  │ Stage 1  blocking passes (DuckDB) + cap
  ▼
data/candidates/{split}_candidates.parquet   (s1_id, cand_id, is_s2, pass_id, block_score)
  │ Stage 2  recall audit vs train GT
  │ Stage 3  pairwise features (chunked, float32) + pair sampling
  ▼
data/features/train_pairs.parquet
  │ Stage 4  LightGBM train (grouped split, early stopping)
  ▼
models/lgbm.txt · models/feature_list.json
  │ Stage 5  threshold sweep (+ one-to-one), macro F_0.5 protocol
  ▼
models/threshold.json
  │ Stage 6  chunked scoring of test candidates
  ▼
output/candidate_pairs.tsv + output/matching_results.tsv
  │ Stage 7  invariant asserts + validate_submission.py
  ▼
submission zip (output/, code/, Documentation_template.md filled)
```

Every stage is restartable and writes parquet/JSON atoms on `D:`. No stage requires the full pair space in RAM.

## 4. Stage 0 — Cleaning and normalization

Inputs: the six record TSVs. Output: one parquet per file plus a schema manifest.

Per record, preserve raw `business_name`, `business_address`, `country`; derive:

- `name_norm`: NFKC, casefold, control chars stripped, punctuation → space, whitespace collapsed.
- `name_fold`: `name_norm` under NFKD (for fuzzy scorers).
- `name_stripped`: `name_norm` minus legal suffixes; `suffix_class` id from a fixed dictionary (`pvt/private`, `ltd/limited`, `inc`, `llc`, `corp/corporation`, `llp`, `sarl`, `sas`, `sa`, `gmbh`, others).
- `name_script`: dominant script tag (latin/indic/cyrillic/arabic/hebrew/other) from codepoint ranges.
- `name_roman`: rule-based transliteration from Indic scripts to Latin (identity for Latin/Cyrillic; best-effort otherwise).
- Address parts: `house_no` (leading numeric token), `street_tokens`, `postal` (US `\d{5}(-\d{4})?`, India `\d{6}`, France `\d{5}`), `state_key` (US two-letter code expanded to name; India/France state/region normalized), `city_tokens`, `landmark_flag`, `addr_missing`.
- Token sets for blocking and Jaccard: `name_tokens`, `name_rare_tokens` (idf computed on the full corpus), `addr_tokens`.

Cleaning rules: no row dropping; empties stay empty; no country filtering; deterministic output ordering by `entity_id`.

Acceptance: per-file row counts match `tools/eda_stats.json`; every derived field non-null; unit tests cover the normalizer, script detector, address parser, and suffix stripper.

## 5. Stage 1 — Blocking / candidate generation

Six passes, each emitting `(s1_id, cand_id, is_s2, pass_id, block_score)`:

| pass_id | key | notes |
|---|---|---|
| 1 | `name_norm` | exact |
| 2 | phonetic code of `name_fold` (Double Metaphone, token-wise) | spelling variants |
| 3 | each `name_rare_token` (idf > threshold) | word-order/DBA; skip blocks larger than `MAX_BLOCK` |
| 4 | `house_no` + first street token | address-anchored |
| 5 | `postal` + one non-stopword `name_token` | fallback when pass 3 empty |
| 6 | `country` + `state_key` + first 3 chars of `name_norm` | all-common-name fallback |

Deduplicate to one row per (s1, cand) keeping min pass_id and max block_score. Apply per-S1 cap `CAP=200` by `(pass_id asc, block_score desc, cand_id)`. Ties resolved deterministically.

DuckDB implementation: each pass is a SQL join between a keys table of S1 and a keys table of S2∪S3, executed per key-class with block-size guards. Candidate tables are written to parquet in chunks.

Audit (Stage 2): join train candidates to GT to compute overall and per-country candidate recall, reduction ratio (candidates ÷ 10.3M S2/S3 records), singleton candidate distributions, and per-pass contribution. Tune `CAP`, idf threshold, and `MAX_BLOCK` until train recall ≥97% and singleton S1 receive ≤ ~10 candidates on average.

The same pipeline runs on test to produce `output/candidate_pairs.tsv`; that file is the exact set the matcher scores.

## 6. Stage 3 — Training pairs and features

Pair construction:

- Positives: all GT pairs (7,638,365).
- Negatives: sample ~4:1 over positives; half are hard negatives drawn from the same blocking pass/key classes, half are random country-matched candidates; failed-blocking positives never enter negatives.
- Split: grouped by `source1_entity_id` (80/20); a second protocol holds out one country entirely.

Features (~40, float32), computed with vectorized kernels (`rapidfuzz.process.cpdist`, sparse matrices) and stored chunked:

- Name fuzzy: ratio, partial, token_sort, token_set, WRatio, QRatio, Jaro-Winkler (on `name_fold`/`name_roman`).
- Name token: exact, Jaccard, containment, sorted-token equality, acronym match, token-count and length deltas.
- Name lexical: char 3-gram TF-IDF cosine (vocabulary fit on train), stripped-name equality, suffix-class match, numeric-token overlap/mismatch, script-match flag, romanization ratio.
- Address: fuzzy ratios, token Jaccard/containment, house-number match, street-token overlap, postal exact + prefix match, state match, landmark flag, missing/length flags.
- Country: same-country flag, seen-in-train flag.
- Structural: source flag, pass_id, block_score, S1 candidate degree, candidate sharing degree, shared rare-token count, idf-weighted name overlap.

Acceptance: golden-value unit tests per feature family; deterministic given seed; feature matrix builds within RAM budget (≤8 GB peak).

## 7. Stage 4 — Model training

- LightGBM binary: `n_estimators=2000`, `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=50`, `subsample=0.8/freq=1`, `colsample_bytree=0.8`, early stopping 100 on grouped validation AUC, `n_jobs=8`, fixed seed.
- Model selection by entity-level macro F_0.5 (singleton rule), not AUC alone.
- Final model retrained on all train pairs; artifacts: `models/lgbm.txt`, `models/feature_list.json`, training metrics JSON.
- XGBoost comparison is re-run on the real candidate-derived sample; any switch is documented in `RULES.md`.

Acceptance: macro F_0.5 reported for grouped and leave-country-out protocols; reloaded model reproduces predictions exactly.

## 8. Stage 5 — Threshold and post-processing

- Sweep pair thresholds 0.05–0.95 step 0.025; choose the value maximizing held-out macro F_0.5.
- Per-country thresholds for US/India when they improve the corresponding holdout; France uses the global threshold.
- One-to-one post-process: if a candidate ID is above threshold for multiple S1, keep only the max-probability assignment (ties by entity_id). Adopt only on held-out improvement.
- Persist `models/threshold.json` with global, per-country, and post-process flags.

## 9. Stages 6–7 — Inference, outputs, validation

- Score test candidates in ~5M-pair chunks; for each S1 keep the final match list after threshold + optional one-to-one.
- Write `output/candidate_pairs.tsv` from the candidate table (header `source1_entity_id`, `candidate_entity_ids`) and `output/matching_results.tsv` (header `source1_entity_id`, `matched_entity_ids`) with exactly one row per test S1 (1,732,544 rows), empty for singletons, sorted by `source1_entity_id`, IDs comma-joined without quoting.
- Assert invariants: row count, no duplicate S1 rows, no duplicate IDs in lists, S2/S3-only, matches ⊆ candidates.
- Run `utils/validate_submission.py` from `DATA/student_resource/` with `--matching`, `--candidate`, `--test-dir`; require `PASS` (exit 0).

Acceptance: validator PASS; invariant assertions green; output written in ≤3 h.

## 10. Package layout and CLI

```
code/business_entity_resolution/
  src/ber/{__init__,config,io_utils,normalize,address,translit,blocking,audit,features,pairs,train,threshold,postprocess,predict,evaluate,cli}.py
  tests/{test_normalize,test_address,test_blocking,test_features,test_score,test_output,...}.py
  models/  README.md  requirements.txt
```

CLI (`python -m ber.cli <command> --config config.json`): `prepare`, `block`, `audit`, `features`, `train`, `tune`, `predict`, `outputs`, `all`. Config holds paths, seed, cap, thresholds, sample ratios.

`requirements.txt` pins: pandas, pyarrow, numpy, scikit-learn, lightgbm, xgboost, rapidfuzz, duckdb, indic-transliteration, jellyfish (phonetics: metaphone/soundex/jaro-winkler), pytest.

## 11. Testing and verification

- Unit: normalizer, script detection, transliteration fallbacks, address parser, suffix stripper, blocking keys, each feature calculator, macro F_0.5 scorer (verified against the 0.714 worked example), one-to-one post-process, output writer.
- Integration: 100-S1 fixture through every stage; validator PASS on fixture outputs.
- Determinism: rerun feature build and training with fixed seed, compare hashes.
- Gates: pytest green; audit recall ≥97%; validator PASS before commit of `1.0.0`.

## 12. Milestones, risks, non-goals

Milestones/tags: `0.2.0` cleaning + blocking audit; `0.3.0` features + model + threshold; `0.4.0` end-to-end fixture smoke; `1.0.0` test outputs + validated package.

Risks: zero-shot France (leave-country-out validation, global fallback); cross-script names (romanization + script features, dropped if no lift); candidate volume (cap + measured recall); RAM (streaming, sampled training negatives); leakage (group splits).

Non-goals: embeddings/transformers (deferred; local CPU only), external data (prohibited), Spark/distributed infra, UI.
