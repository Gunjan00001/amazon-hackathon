# Dataset Description — Business Entity Resolution Challenge

> Companion to `student_resource/README.md` (later transcribed to `PROBLEM_STATEMENT.md` at the repo root).
> Every statistic below was measured directly from the files, not copied from the statement.
> Reproduce with `tools/eda_dataset.py` (writes `tools/eda_stats.json`). Stats cache: `tools/eda_cache/`.

## 1. TL;DR

- 7 UTF-8, tab-separated files: 6 record files + 1 ground-truth file. ~2.35 GiB total.
- 3 sources: **Source 1** (deduplicated reference, 2.21M train / 1.73M test rows) is matched against **Source 2** (5.03M / 4.89M) and **Source 3** (5.29M / 5.08M).
- All inputs are source-only — there are **no cross-source identifiers** other than the ground truth.
- Ground truth is **injective**: each matched S2/S3 ID appears in exactly one S1 row's list.
- **5.59% of S1 entities are singletons** (no matches); the rest match a mean of 3.46 records (max 11).
- Missing `business_address` in ~3.3% of S2/S3 records; names/addresses in Source 1 are never empty.
- ~15% of train S2 names are non-Latin scripts (Devanagari, Tamil, Telugu, Gujarati, Gurmukhi, Malayalam) while Source 1 train names are 100% ASCII — cross-script matching is required.

## 2. File inventory

| File | Rows (excl. header) | Size |
|---|---|---|
| `train/train_source1.tsv` | 2,206,821 | 200.3 MiB |
| `train/train_source2.tsv` | 5,034,616 | 466.6 MiB |
| `train/train_source3.tsv` | 5,285,603 | 480.4 MiB |
| `train/train_ground_truth.tsv` | 2,206,821 | 121.1 MiB |
| `test/test_source1.tsv` | 1,732,544 | 166.9 MiB |
| `test/test_source2.tsv` | 4,887,273 | 485.9 MiB |
| `test/test_source3.tsv` | 5,082,316 | 482.6 MiB |

All files have exactly the header `entity_id\tbusiness_name\tbusiness_address\tcountry` (record files) or `source1_entity_id\tmatched_entity_ids` (ground truth). Row counts are exact newline counts; the last line of each file is newline-terminated.

## 3. Schema

### Record files (`*_source{1,2,3}.tsv`)

| Column | Type | Notes |
|---|---|---|
| `entity_id` | string | Unique within its file. Prefix fixes the source: `S1-`, `S2-`, `S3-`. |
| `business_name` | string | Never empty in any file. Can be non-Latin script, accented Latin, or contain typos/abbreviations/legal suffixes. |
| `business_address` | string | Empty in ~3% of S2/S3 rows (see §6). Comma-heavy, so the delimiter is a tab. |
| `country` | string | Label, not an ISO code. Observed values: `US`, `India` (train), plus `France` (test only). |

### Ground truth (`train/train_ground_truth.tsv`)

| Column | Type | Notes |
|---|---|---|
| `source1_entity_id` | string | One row per Source 1 entity — exactly 2,206,821 rows, matching `train_source1.tsv` 1:1. |
| `matched_entity_ids` | string | Comma-separated `S2-`/`S3-` IDs, or empty for singletons. Never contains `S1-` IDs. |

## 4. Country distribution (measured)

| File | US | India | France |
|---|---|---|---|
| `train_source1` | 1,323,633 (59.98%) | 883,188 (40.02%) | — |
| `train_source2` | 3,016,817 (59.92%) | 2,017,799 (40.08%) | — |
| `train_source3` | 3,170,056 (59.97%) | 2,115,547 (40.03%) | — |
| `test_source1` | 663,106 (38.27%) | 809,986 (46.75%) | 259,452 (14.98%) |
| `test_source2` | 1,871,330 (38.29%) | 2,312,565 (47.32%) | 703,378 (14.39%) |
| `test_source3` | 1,945,701 (38.29%) | 2,405,000 (47.33%) | 731,615 (14.40%) |

Train is US/India only; test is ~38% US, ~47% India, ~15% France. France is a **zero-shot country** — no training labels exist for it, so any country-conditioned model/threshold must degrade gracefully to a country-agnostic fallback.

## 5. Ground-truth analysis (measured)

| Metric | Value |
|---|---|
| Rows | 2,206,821 |
| Singletons (empty match list) | 123,247 (**5.585%**) |
| Total matched IDs | 7,638,365 |
| Mean matches per S1 | 3.461 |
| Max matches per S1 | 11 |
| Unique matched IDs | 7,638,365 |
| Matched IDs used by **multiple** S1 entities | **0** |

**Match-count distribution:**

| # matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 entities | 123,247 | 119,157 | 375,212 | 530,841 | 484,115 | 321,957 | 164,868 | 63,968 | 18,680 | 4,205 | 534 | 37 |
| % | 5.59 | 5.40 | 17.00 | 24.06 | 21.94 | 14.59 | 7.47 | 2.90 | 0.85 | 0.19 | 0.02 | 0.002 |

**Structural facts:**

- The mapping S1 → {S2/S3} is **injective on the S2/S3 side**: no S2/S3 ID is claimed by two S1 entities. This is a strong prior — an assignment/one-to-one consistency post-process can legitimately boost precision without external data.
- Every ID referenced by the ground truth exists in its corresponding source file (0 missing across 7,638,365 IDs).
- Singletons are only 5.59% of entities, but under macro F_0.5 each correctly-empty prediction is worth 1.0 and each false merge on them is worth 0.0 — so they carry ~5.6% of the score by themselves.

## 6. Data quality and encoding notes

| Quality metric | train S1 | train S2 | train S3 | test S1 | test S2 | test S3 |
|---|---|---|---|---|---|---|
| Empty `business_name` | 0 | 0 | 0 | 0 | 0 | 0 |
| Empty `business_address` | 0 | 168,967 (3.36%) | 175,916 (3.33%) | 0 | 129,408 (2.65%) | 136,098 (2.68%) |
| Names with non-ASCII chars | 0 | 764,608 (15.19%) | 606,737 (11.48%) | 40,789 (2.35%) | 928,158 (18.99%) | 737,515 (14.51%) |
| Addresses with non-ASCII chars | 554 (0.03%) | 478,453 (9.50%) | 476,588 (9.02%) | 73,800 (4.26%) | 720,665 (14.75%) | 729,222 (14.35%) |
| Addresses starting with a digit | 1,365,839 (61.90%) | 2,567,674 (51.00%) | 2,705,751 (51.19%) | 1,001,839 (57.83%) | 2,290,504 (46.87%) | 2,385,124 (46.93%) |
| Null/missing cells | 0 | 0 | 0 | 0 | 0 | 0 |

**Encoding:** all files are valid UTF-8. There are **no literal `???` placeholder values** — the `??????`-looking names seen in a Windows console are real non-Latin script rendered through a cp1252 console; the bytes are proper Devanagari/Tamil/etc. Likewise `L�arning` in a console is the correctly encoded `Léarning` (U+00E9). Read the files as UTF-8 and this disappears.

**Real non-Latin name examples (verbatim):**

- `राम मार्केटिंग प्राइवेट लिमिटेड` (train_source2, Hindi)
- `குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்` (train_source2, Tamil)
- `Sun पावर Provision` (train_source3, code-mixed)
- `École primaire Sainte Pierre` (test_source1, French accent)
- `Engages Àrt Pharmacie SCI` (test_source3, `À` + typo likely)
- `Béque` (train_source3, truncated/noisy name)

**Other observed noise classes:** legal suffixes with inconsistent abbreviation (`Pvt`/`Private`, `Ltd`/`Limited`), ampersands, punctuation, `#98825`-style trailing tokens, person/brand names with embedded typos, addresses with landmark references and municipal formats.

**Format gotchas:**
- Commas are **inside** `business_address` and `matched_entity_ids`, so any comma-delimited reading of these files is wrong. Always `sep="\t"`.
- With pandas, read with `keep_default_na=False` (or `dtype=str`) so empty addresses stay `""` instead of becoming `NaN`/`None`.
- `country` is a label with free-form casing in the source statement (`US`, `India`, `France`); normalize with `strip().lower()` before use, but never filter on it.

## 7. Implications for the pipeline

1. **Candidate cap math:** 2.21M train S1 × ~3.5 true matches = 7.64M positives; test has 1.73M S1 needing any matches at all. Blocking must produce a candidate set that is a superset of matches, but the model only ever sees the final candidate set.
2. **One-to-one prior:** because no S2/S3 record is shared, a stable-matching / greedy assignment filter (each S2/S3 assigned to its highest-scoring S1) can be applied after thresholding — but only if it does not lower validation macro F_0.5.
3. **France handling:** hold out one train country (or a name-similarity slice) to approximate the unseen-country generalization before trusting test results.
4. **Missing addresses:** ~3.3% of candidates only have a name signal; features must support `address_missing` flags and avoid dividing by zero.
5. **Script mismatch:** Source 1 train names are ASCII while many S2/S3 names are Indic scripts. Pure edit distance is weak here; phonetic/transliteration-insensitive name features (or native↔Latin transliteration handling) matter.
6. **Singletons:** 123,247 entities (train) require predicting an empty list; threshold tuning must optimize macro F_0.5 with singletons included, not pair-level AUC alone.

## 8. Reproduction

```bash
# from the repo root, using the project venv (Python 3.12)
.venv\Scripts\python.exe tools\eda_dataset.py
```

Raw measured output: `tools/eda_stats.json`. Per-file caches: `tools/eda_cache/*.json` (delete the cache to force a rescan).
