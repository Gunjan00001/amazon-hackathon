# Decisions and Reasoning

Architecture Decision Records (lightweight) for the Business Entity Resolution pipeline.
Each entry: decision, context, alternatives, and consequence.

## D1 — Blocking plus classifier, not end-to-end
- **Context:** 2.21M train / 1.73M test Source 1 entities must be matched against 10.3M / 10.0M
  candidate records. The naive cross-product is ~10^13 pairs.
- **Decision:** A hard blocking stage produces a bounded candidate set; a pairwise LightGBM
  classifier scores it; a threshold (plus injective assignment) produces final matches.
- **Alternatives:** end-to-end transformer/page-level model (infeasible on local CPU at 10M
  records); embedding ANN blocking (higher ceiling, but CPU-only machine and no GPU locally).
- **Consequence:** recall is bounded by blocking (0.814), which is now the main ceiling.

## D2 — DuckDB for blocking joins, pandas/pyarrow for the rest
- **Context:** joins over 10M+ rows and 250–300M pairs must run in 23 GB RAM.
- **Decision:** DuckDB does out-of-core joins, dedup, ranking, and CSV writing; pandas/pyarrow
  handle normalization, feature blocks, and I/O.
- **Consequence:** stages are restartable via parquet caches; memory stays bounded.

## D3 — Ten blocking passes with per-pass block caps
- **Context:** exact-name blocking alone recalls only ~22% of true pairs (names are noisy);
  per-token Metaphone exploded; large blocks for common tokens must not dominate.
- **Decision:** union passes 1,3,4,5,6,7,8,9,10 (exact name, rare token, address prefix,
  postal+name, fallback, name/street prefix-5, rare-token pairs and triples) and cap candidate
  block sizes per pass (`pass_caps` in `config.json`), then cap per Source 1 entity (`cap`).
- **Reasoning:** the measured coverage of a shared name-token prefix-5 is ~89% and of >=2 shared
  tokens ~78%, so prefix and token-pair keys are the highest-yield signals.
- **Consequence:** train recall 0.8115; India (0.727) trails US (0.868) because Indic names and
  sparse addresses yield fewer shared rare tokens.

## D4 — Do not reintroduce per-token Metaphone
- **Context:** per-token phonetic codes collide massively (generic codes shared by hundreds of
  thousands of records).
- **Decision:** removed entirely; kept only as an offline experiment.
- **Consequence:** avoided the 75 GiB temp explosion; recall recovered via prefix/pair passes.

## D5 — LightGBM as the matcher
- **Context:** benchmarked LightGBM vs XGBoost on 100k sampled Source 1 with hard negatives.
- **Decision:** LightGBM primary. Quality was tied (AUC identical, F0.5 within ±0.0003);
  LightGBM trained 3–4.5x faster, which matters for local iteration. XGBoost remains a drop-in.
- **Consequence:** 610-tree model, `models/lgbm.txt`.

## D6 — 4:1 negative sampling for training
- **Context:** positives are 7.64M GT pairs; scoring every negative pair during training is wasteful.
- **Decision:** train on all found positives plus negatives sampled 4:1, half hard (same pass) and
  half random country-matched.
- **Consequence:** fast training, but thresholds learned on this sample are miscalibrated for the
  full candidate distribution — see D8.

## D7 — Injective one-to-one post-process
- **Context:** the ground truth is injective (each S2/S3 ID appears in at most one Source 1 list).
- **Decision:** when several Source 1 entities claim the same candidate above threshold, keep only
  the highest-probability assignment.
- **Consequence:** verified to improve macro F0.5 on the full-candidate validation (0.8488 vs 0.8475).

## D8 — Threshold must be tuned on full candidates (0.925, not 0.675)
- **Context:** the 4:1 sampled split suggested 0.675–0.70, but it contains ~4 negatives per
  positive versus ~135 candidates per entity at inference.
- **Decision:** re-tune the threshold on the full candidate set for the held-out Source 1 groups.
  The optimum moves to **0.925**; all submission outputs use it.
- **Consequence:** 200,982 singletons instead of 151,576, and much better precision on the metric.

## D9 — Full-candidate held-out validation is the reported number
- **Context:** the 4:1 mark (0.9807) is optimistic because it undersamples confusable negatives.
- **Decision:** report macro F0.5 on the full candidate set of the 20% held-out Source 1 groups,
  predicted exactly as inference.
- **Consequence:** **0.8488** (95% CI 0.8481–0.8496) is the believable in-domain estimate.

## D10 — Leave-one-country-out as the France proxy
- **Context:** France is ~15% of test and has no labels.
- **Decision:** retrain on US only -> validate India, and India only -> validate US.
- **Consequence:** unseen-country drop of 0.09–0.12 F0.5; realistic leaderboard expectation
  ~0.80–0.85 rather than 0.85+.

## D11 — Ship exactly the required submission tree
- **Context:** the problem statement prescribes `output/`, `code/business_entity_resolution/`
  (`src/`, `README.md`, `requirements.txt`), and `Documentation_template.md`.
- **Decision:** ship only that; no `models/` in the package (a reproducer retrains from data).
  A minimal `src/config.json` is included so the packaged pipeline runs as-is.
- **Consequence:** package is spec-compliant and self-contained; model is reproducible but not shipped.

## D12 — Local CPU for stages 0–1 and 5, Colab/Kaggle GPU for transformer stages
- **Context:** the matcher/precision work is CPU- and data-bound; embeddings and cross-encoders are GPU-bound.
- **Decision:** keep cleaning, blocking, tabular features, LightGBM, calibration, validation, outputs and packaging local; run embeddings/rerank on Colab (free T4) or Kaggle.
- **Reasoning:** Colab free has 2 vCPU / 13.6 GB RAM vs local 8C/16T / 23 GB and the data is local, so phases 0–1 are faster locally; the T4 is ~20–50× faster than local CPU for transformers.
- **Consequence:** avoids large data transfers for CPU work; `RULES.md` §6 amended to allow both.

## D13 — Colab returns per-pair cosine features, not full embeddings
- **Context:** full embeddings for ~20M unique entities are ~15 GB fp16; free Drive is 15 GB.
- **Decision:** Colab computes embeddings and returns only `(s1_id, cand_id, emb_name_cos, emb_addr_cos)` (~0.5 GB for test).
- **Consequence:** tiny transfers and no extra storage account needed; full vectors are not reused, so embedding changes require a ~2–4 h re-encode.

## D14 — Add char n-gram TF-IDF cosine features
- **Context:** the shipped matcher used rapidfuzz/token features only; char n-grams capture typos and morphological variants that token features miss.
- **Decision:** add char 3-gram TF-IDF cosine for `name_norm`, `name_roman`, `addr_norm` (vectorizer fit on the train sample, reused for val/test).
- **Consequence:** more precision/recall information locally at low cost; candidate set unchanged.

## D15 — Per-country thresholds + singleton calibration
- **Context:** F0.5 is precision-heavy and per-entity; US and India have different score distributions, and true singletons are worth 1.0 only if predicted empty.
- **Decision:** tune thresholds per country (US/India) with a global fallback for unseen France, and add a singleton decision (`max prob < tau` → empty).
- **Consequence:** targets the India cell (0.788) and false merges on singletons; guarded by LOO to avoid overfitting the unlabeled France slice.

