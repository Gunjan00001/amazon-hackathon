# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** AA..><  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
We solve business entity resolution as a blocking-plus-classifier pipeline: DuckDB blocking passes
reduce 10.3M candidate records against each Source 1 entity to a high-recall candidate set, a
LightGBM pairwise classifier scores each candidate, and a threshold tuned for macro F_0.5 (with an
injective one-to-one assignment post-process) produces the final matches. Source 1 entities are
matched to Source 2/3 only; all signals come from the provided data.

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

**Features used (33):**
- Name features: rapidfuzz ratio, partial, token-sort, token-set, WRatio, QRatio, Jaro-Winkler;
  normalized exact match; token Jaccard and containment; sorted-token equality; length and token-count
  deltas; romanized-name ratio; script-match.
- Address features: fuzzy ratio, token-sort ratio, Jaccard, containment, house-number match,
  street-token Jaccard, postal exact and 3-prefix, state match, landmark flag, missing flag, length delta.
- Other: same-country flag, source indicator (S2/S3), blocking pass id, blocking score, S1 candidate
  degree, candidate degree.

**Model type:** LightGBM binary classifier (610 trees), early stopping on a grouped validation AUC.

**Threshold selection method:** Grouped 80/20 split by `source1_entity_id`; threshold swept from
0.05 to 0.95 to maximize entity-level macro F_0.5. Because a 4:1 negative sample understates the
false-merge risk, the operating point was re-tuned on the **full candidate set** for the held-out
groups, where the optimum moves from 0.675 to **0.925**. The injective one-to-one assignment was
also compared against the raw threshold and selected because it scored higher (0.8488 vs 0.8475).
Final inference uses threshold 0.925.

---

## 5. Results & Error Analysis

- **Official leaderboard (public Portal, 26 Sep 2026): macro F_0.5 = 0.811** (Evaluated). This is the
  real score on the public test split, computed from `output/matching_results.tsv` (one-to-one,
  threshold 0.925). It falls inside the predicted ~0.80–0.85 band.
- **Full-candidate held-out estimate (test-like):** macro F_0.5 **0.8488**, 95% CI
  [0.8481, 0.8496], computed on the 20% held-out Source 1 groups using their complete candidate
  sets (438,499 entities; 60.9M candidate pairs). Candidate recall ceiling 0.814.
- **Per-country (full model):** US 0.889, India 0.788.
- **Leave-one-country-out (unseen-country / France proxy):** train-US → validate-India 0.668;
  train-India → validate-US 0.804. France is ~15% of the test set with no labels, so the realistic
  leaderboard expectation is ~0.80–0.85.
- **4:1 sampled split (optimistic, not comparable):** macro F_0.5 0.9807, precision 0.988,
  recall 0.977. This over-states the leaderboard because negatives are sampled rather than the full
  candidate distribution.
- **Test output:** 1,732,544 rows in both files at threshold 0.925; 200,982 Source 1 entities
  predicted as singletons; 5,071,867 candidate pairs above threshold.
- **Common false positives (wrong merges):** generic names and popular street tokens sharing a
  common blocking key; mitigated by the one-to-one filter.
- **Common false negatives (missed matches):** true pairs whose only shared signal is a common token
  dropped by a block cap, and India records with sparse addresses (candidate recall 0.73).

---

## 6. Conclusion
A recall-audited blocking stage plus a compact LightGBM matcher with an injective assignment filter
produces a validated, reproducible submission under the local CPU budget. The main residual risk is
the 0.81 candidate-recall ceiling (especially on India and the unseen France split), which bounds the
maximum achievable score; future work would add embedding-based fuzzy blocking to raise that ceiling.

---

## Appendix

### A. Code Artefacts
The runnable code ships under `code/business_entity_resolution/` (all source in `src/ber/`, with
`README.md` and `requirements.txt`). Entry points: `ber.cli prepare`, `block`, `audit`, `features`,
`train`, `predict`, `evaluate`. `predict` reproduces `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the cached intermediates and the prebuilt model.

### B. Additional Results
- Blocking audit: `DATA/reports/train_blocking_audit.json`.
- Local validation marks: `DATA/reports/eval_marks.json`.
- Training metrics: `code/business_entity_resolution/models/training_metrics.json`.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
