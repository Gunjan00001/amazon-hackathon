# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** AA..><  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
We solve business entity resolution as a blocking-plus-classifier pipeline: DuckDB blocking passes
reduce 10.3M candidate records against each Source 1 entity to a high-recall candidate set, a
LightGBM pairwise classifier scores each candidate (fuzzy string, char-3 TF-IDF cosine, and
address/token features), and per-country thresholds with a singleton decision and an injective
one-to-one assignment produce the final matches. Source 1 entities are matched to Source 2/3 only;
all signals come from the provided data. A multilingual sentence-embedding feature set was also
built on a free Colab T4, but it did not beat the local matcher and was not adopted.

---

## 2. Methodology

### 2.1 Problem Analysis
- 3 sources, no shared identifiers; Source 1 (2.21M train rows) is the deduplicated reference and
  must be matched to Source 2 (5.03M) and Source 3 (5.29M).
- Ground truth is **injective**: each Source 2/3 ID appears in at most one Source 1 list
  (7,638,365 matched IDs, all unique).
- **5.585%** of Source 1 entities are singletons; matched entities average 3.46 matches (max 11).
- Country is open-set: train covers US and India; test adds France (~15%) with no training labels.
- Noise: abbreviations, legal-suffix variants, typos, word-order changes, transliterations,
  missing address components, landmark references. ~15% of Source 2 train names are non-Latin
  scripts while Source 1 train names are ASCII; ~3.3% of Source 2/3 addresses are empty.

### 2.2 Solution Strategy
A staged, restartable parquet pipeline driven by `ber.cli`: normalize → block → audit → features →
train → predict. DuckDB performs the out-of-core blocking joins; pandas/pyarrow and rapidfuzz
compute pairwise features in parallel; LightGBM trains the matcher.

**Approach Type:** Blocking + Classifier (with injective assignment post-processing)

**Core Innovation:** A recall-first suite of ten blocking passes (exact name, rare token, address
prefix, name/street prefix, and rare-token pairs/triples) with per-pass block caps, followed by a
learned pairwise matcher and an injective one-to-one filter that exploits the ground-truth
structure to raise precision.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. exact normalized name; 3. rare-token exact (IDF-filtered); 4. `house_no | street[0][:5]`;
  5. `postal | first-name-token`; 6. fallback `state | name[:3]`; 7. name-token prefix-5;
  8. street-token prefix-5; 9. token pairs over the 4 rarest tokens; 10. token triples over the
  3 rarest tokens. All passes are unioned, deduplicated, and capped per Source 1.
- **Candidate pairs generated:** train 304,759,423; test 250,607,135 (about 138 per S1 on train).
- **How true matches were preserved:** recall was measured directly against the ground truth with a
  DuckDB audit (no candidate tuples materialized). Train candidate recall = **0.8115** overall
  (US 0.868, India 0.727), reduction ratio 29.5. Per-pass caps were tuned by rebuilding candidates
  and re-auditing (each audit ~50 s).

---

## 4. Matching Model

**Features used (36):**
- Name features: rapidfuzz ratio, partial, token-sort, token-set, WRatio, QRatio, Jaro-Winkler;
  normalized exact match; token Jaccard and containment; sorted-token equality; length and token-count
  deltas; romanized-name ratio; script-match; char-3 TF-IDF cosine on `name_norm` and `name_roman`.
- Address features: fuzzy ratio, token-sort ratio, Jaccard, containment, house-number match,
  street-token Jaccard, postal exact and 3-prefix, state match, landmark flag, missing flag, length
  delta; char-3 TF-IDF cosine on `addr_norm`.
- Other: same-country flag, source indicator (S2/S3), blocking pass id, blocking score, S1 candidate
  degree, candidate degree.

The char-3 TF-IDF vectorizer is fit once on a 299,997-document train sample (seed 42) and reused for
validation/test.

**Model type:** LightGBM binary classifier (523 trees at the adopted operating point), early stopping
on a grouped validation AUC.

**Threshold selection method:** Grouped 80/20 split by `source1_entity_id`. Because a 4:1 negative
sample understates the false-merge risk, thresholds were tuned on the **full candidate set** for the
held-out groups: a per-country sweep (`tune_per_country`) with a global fallback for unseen France,
plus a singleton decision (`max(prob) < tau` → empty), then the injective one-to-one assignment. The
adopted calibration is global 0.95, India 0.925 / US 0.95, `singleton_tau` 0.30, one-to-one; inference
keeps every candidate at/above `min(global, country thresholds)` and filters by the country threshold
and singleton rule.

---

## 5. Results & Error Analysis

- **Official leaderboard (public Portal, 26 Sep 2026): macro F_0.5 = 0.811** (Evaluated) for the
  baseline pipeline. The adopted v2.0.0 pipeline regenerates `output/matching_results.tsv` with
  char-3 features and per-country/singleton calibration (validator PASS); its upload result is
  pending.
- **Full-candidate held-out estimate (test-like), baseline:** macro F_0.5 **0.8488**, 95% CI
  [0.8481, 0.8496], on the 20% held-out Source 1 groups with their complete candidate sets
  (438,499 entities; 60.9M candidate pairs). Candidate recall ceiling 0.814.
- **Full-candidate held-out estimate (test-like), adopted v2.0.0:** macro F_0.5 **0.8577**, 95% CI
  [0.8570, 0.8585]; per-country US 0.8983, India 0.7968. Adds char-3 TF-IDF cosine features
  (0.8488 → 0.8574) and per-country/singleton calibration (→ 0.8577), candidate set unchanged.
- **Diagnostic ceiling:** intersecting ground-truth pairs with the candidate set gives an oracle
  macro F_0.5 of **0.9122**, so ~0.055 of headroom remains in the matcher at the current blocking
  ceiling.
- **Embeddings (tested, not adopted):** multilingual `intfloat/multilingual-e5-small` per-pair cosine
  was produced on a free Colab T4. Free-tier disk I/O forced a 384 → 64-dim random projection; the
  resulting calibrated held-out was 0.8570 < 0.8577, so embeddings were rejected.
- **Leave-one-country-out (unseen-country / France proxy):** train-US → validate-India 0.668;
  train-India → validate-US 0.804. France is ~15% of the test set with no labels, so the realistic
  leaderboard expectation is ~0.81–0.83.
- **4:1 sampled split (optimistic, not comparable):** macro F_0.5 0.9823, precision 0.988,
  recall 0.977. This over-states the leaderboard because negatives are sampled rather than the full
  candidate distribution.
- **Test output (v2.0.0):** 1,732,544 rows in both files; 212,118 Source 1 entities predicted as
  singletons (baseline 200,982); 5,063,955 candidate pairs at/above the 0.925 prediction floor.
- **Common false positives (wrong merges):** generic names and popular street tokens sharing a
  common blocking key; mitigated by per-country thresholds, the singleton rule, and one-to-one.
- **Common false negatives (missed matches):** true pairs whose only shared signal is a common token
  dropped by a block cap, and India records with sparse addresses (candidate recall 0.73).

---

## 6. Conclusion
A recall-audited blocking stage plus a compact LightGBM matcher (fuzzy, char-3 TF-IDF cosine, and
address features) with per-country/singleton calibration and an injective assignment filter produces a
validated, reproducible submission under the local CPU budget. Char-3 features and calibration lift the
held-out estimate from 0.8488 to **0.8577** with no candidate-set change. Multilingual embeddings were
built on a GPU but did not beat the local matcher and were rejected. The main residual risk remains the
0.81 candidate-recall ceiling (especially on India and the unseen France split); future work would
re-run the full-dim embedding cosine on a host with more RAM/local disk, try a cross-encoder rerank, or
add embedding/LSH fuzzy blocking to raise that ceiling.

---

## Appendix

### A. Code Artefacts
The runnable code ships under `code/business_entity_resolution/` (all source in `src/ber/`, with
`README.md` and `requirements.txt`). Entry points: `ber.cli prepare`, `block`, `audit`, `features`,
`train`, `calibrate`, `diagnose`, `predict`, `evaluate`, `validation`, `loo`. `predict` reproduces
`output/matching_results.tsv` and `output/candidate_pairs.tsv` from the cached intermediates, the
prebuilt model, and the calibrated `threshold.json`.

### B. Additional Results
- Blocking audit: `DATA/reports/train_blocking_audit.json`.
- Local validation marks: `DATA/reports/eval_marks.json`.
- Training metrics: `code/business_entity_resolution/models/training_metrics.json`.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
