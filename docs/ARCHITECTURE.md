# Architecture

Business Entity Resolution: match noisy business records from **Source 2** and **Source 3**
to the deduplicated reference **Source 1**. Scored by **macro F_0.5** per Source 1 entity,
singletons included (precision is worth 2× recall).

## 1. Pipeline at a glance

```
raw challenge TSVs (S1/S2/S3 train+test, ground truth)
        │
        ▼
[1] CLEAN     normalize text, expand legal suffixes + address abbreviations,
        │     ASCII-fold (anyascii) for cross-script keys, extract PIN/ZIP/state
        ▼
[2] BLOCK     union of inverted-index passes, IDF-scored, country gate,
        │     top `max_candidates` per Source 1  ──► candidate pairs
        ▼
[3] FEATURES  15 classical pair features + structural features
        │     (+ optional multilingual embedding features / cross-encoder)
        ▼
[4] CLASSIFY  LightGBM binary matcher, split grouped by Source 1 entity
        │
        ▼
[5] DECIDE    top-K pruning per Source 1, macro-F_0.5 threshold, TSV output
        │
        ▼
output/candidate_pairs.tsv   (exact set the model scored)
output/matching_results.tsv  (IDs above threshold; subset of candidates)
```

The optional GPU stages (`embed`, `rerank`) are additive: the classical path ships on its
own and is the fallback at every step. See
[`superpowers/specs/2026-09-25-kaggle-cascade-design.md`](superpowers/specs/2026-09-25-kaggle-cascade-design.md).

## 2. Package layout

All pipeline code lives in `code/business_entity_resolution/`.

```
src/ber/
  config.py           paths / env resolution (BER_DATA_DIR, BER_ARTIFACT_DIR)
  text.py             normalization: NFKC, casefold, punctuation, legal suffixes,
                      address abbreviations, anyascii transliteration, PIN/ZIP parsing
  io_tsv.py           explicit-tab TSV read/write; preserves empty strings as signal
  labels.py           ground-truth loading / injective label handling
  blocking.py         blocking keys + IDF-scored candidate generation, country gate
  features.py         classical pairwise features (rapidfuzz ratios, Jaccard, lengths,
                      country/source flags) + structural features
  vector_features.py  optional multilingual embedding features (cosine, abs-diff)
  gbdt.py             LightGBM matcher (train / predict / persistence)
  rerank.py           optional cross-encoder reranking
  score.py            metric helpers (macro F_0.5, entity-level aggregation)
  evaluate.py         held-out evaluation utilities (grouped by Source 1 entity)
  decision.py         top-K pruning, threshold search, TSV assembly
  search.py           autonomous parameter search + finalize driver
  stages/
    clean.py          stage [1]
    block.py          stage [2]
    embed.py          optional embedding pass (GPU on Kaggle)
    train_gbdt.py     stage [4]
    rerank.py         optional rerank stage (GPU on Kaggle)
    decide.py         stage [5]
    search.py         search + finalize entry point (--smoke-first, --search-only, --finalize-only)
tests/                unit tests for every module above (pytest)
```

## 3. Design decisions and rationale

| Decision | Why |
|---|---|
| **Precision-first cascade** — cheap recall net, then a classifier | F_0.5 rewards precision 2×; the expensive model only sees a small candidate set. |
| **IDF-weighted blocking with a country gate** | Raises candidate recall while keeping the set small. Country is treated as an open set, never filtered against a fixed list. |
| **Single versioned feature spec** reused for train/val/test | Prevents train/serve skew; no label-derived features. |
| **Entity-grouped splits** (`source1_entity_id`) | Pairs from the same Source 1 entity must not straddle the split. |
| **LightGBM default** | Measured equal quality to XGBoost, 3–4.5× faster to train (see `tools/bench_results.json`). |
| **Kaggle-only heavy compute** | Rules §6: training/GPU inference on Kaggle; local machine edits, tests, validates only. |
| **Autonomous search** (`ber.search`) | Searches blocking + matcher/threshold params, scoring every trial by held-out macro F_0.5, checkpointing trials. |

## 4. Data flow / artifacts

- `BER_DATA_DIR` → the challenge dataset directory (defaults resolved in `config.py`).
- `BER_ARTIFACT_DIR` → intermediate caches (clean parquet, candidate pairs, model).
- `kaggle/staging/` → the Kaggle push staging area (datasets, kernels, fetched outputs);
  regenerable and **not** committed.
- `output/` → the final validated submission TSVs, committed via Git LFS.

## 5. Results

| Metric | Value |
|---|---|
| Validation macro F_0.5 | **0.9740** |
| Blocking recall ceiling | 1.0 |
| Candidate pairs scored | 16,167,646 |
| Test rows (Source 1) | 1,732,544 |
| Matched entities (threshold 0.70) | 1,310,189 |
