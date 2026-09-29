# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We frame entity resolution as a precision-first cascade: a single IDF-weighted blocking pass
guarantees that virtually all true matches survive as candidates (measured recall ceiling 1.0),
then a LightGBM classifier over 20 name/address features scores each candidate and a
macro-F_0.5-optimal threshold selects the final matches. The whole pipeline (cleaning,
blocking-parameter search, model and threshold search, inference) runs on Kaggle; the local
machine only edits code, runs unit tests, and validates the submission.

---

## 2. Methodology

### 2.1 Problem Analysis

Source 2 / Source 3 records are noisy restatements of Source 1 businesses: legal-suffix and
abbreviation variation, word-order transpositions, typos, missing address components, and
~15% of S2 names in Indic scripts while Source 1 names are ASCII. The ground truth is
injective on the S2/S3 side (no ID is claimed twice), and 5.59% of Source 1 entities are
singletons that must be predicted empty (worth a full 1.0 each).

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier cascade (precision-first)
**Core Innovation:** IDF-weighted candidate ranking with ASCII-folded tokens, plus an
autonomous on-Kaggle search over blocking/matcher parameters scored by held-out macro F_0.5.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** country-gated inverted indexes on (a) significant name tokens,
  (b) address tokens, (c) phonetic keys (Metaphone of folded tokens), and (d) exact postal /
  PIN codes. All keys are computed from ASCII-folded, suffix-expanded normalized text.
- **Scoring:** each matched key contributes its inverse document frequency; exact postal is
  weighted highest. The top `max_candidates` (40) per Source 1 are kept.
- **Candidate pairs generated:** 16,167,646 for the 1,732,544 test Source 1 entities.
- **How you ensured true matches were not lost:** the recall ceiling was measured directly
  against the full train ground truth during the search; the selected configuration achieves
  **recall_ceiling = 1.0** at 40 candidates per entity (~9.3 candidates per entity on average
  after pruning to top-10 for the final model).

---

## 4. Matching Model

**Features used:**
- Name features: rapidfuzz ratio / partial / token-sort / token-set, Jaro-Winkler, exact match,
  Jaccard, length difference (on folded, normalized names).
- Address features: rapidfuzz ratio / token-sort, Jaccard, missing-address flag, length
  difference.
- Other: same-country, is-Source-2 flags; structural postal match, phonetic match, address
  token overlap, name token overlap, first-token match.

**Model type:** LightGBM (binary), grouped 80/20 split by Source 1 entity.
**Threshold selection method:** F_0.5 optimization on the grouped validation split
(singletons included); candidates pruned to top-10 per Source 1 before thresholding.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9740 (validation, grouped by Source 1 entity). Pair AUC 0.99952,
  average precision 0.99665.
- **Common false positives (wrong merges):** near-duplicate names in dense urban areas with
  the same PIN and overlapping address tokens.
- **Common false negatives (missed matches):** cross-script (Indic) names and cases where the
  true partner shares only very common tokens; these are the target of the planned
  multilingual-embedding and cross-encoder stages.

---

## 6. Conclusion

A single well-engineered IDF-weighted blocking pass plus a LightGBM matcher already reaches
~0.974 macro F_0.5 on held-out data with 100% candidate recall, and the entire search,
training and inference run reproducibly on Kaggle. The remaining headroom is cross-script and
zero-shot-country generalization, which the embedding/cross-encoder stages target next.
