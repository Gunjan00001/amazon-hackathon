# Failures and Fixes

Every non-trivial failure encountered while building the pipeline, with root cause and resolution.
Kept for future reference so the same dead ends are not re-entered.

## F1 — Per-token Metaphone blocking exploded the temp directory
- **Symptom:** `OutOfMemoryException ... 75.6 GiB/75.6 GiB used`; the run never completed.
- **Cause:** per-token phonetic codes collide at scale (a single generic code was shared by >1.2M
  candidate records), so the pass-2 join produced billions of rows.
- **Fix:** removed the Metaphone blocking pass entirely (`RULES`/`DECISIONS` note: do not
  reintroduce). Recall was recovered with name-token prefix-5 and rare-token pair/triple passes.

## F2 — 64-bucket join loop took >3 hours
- **Symptom:** a per-bucket loop over 64 hash buckets ran for the full 3 h tool timeout.
- **Cause:** each bucket re-scanned the entire key tables (64 full scans) instead of one join.
- **Fix:** single DuckDB join with pass-specific block caps (`_valid_sql`); join size fell from
  819M to 38.5M rows in the first tuning step and candidates were produced in minutes.

## F3 — Pandas audit could not handle 98.8M candidate rows
- **Symptom:** `ber.cli audit` hung/OOMed after blocking grew to ~100M candidates.
- **Cause:** the audit materialized candidate tuples into Python sets/dicts.
- **Fix:** rewrote `run_audit` to be DuckDB end-to-end (read_parquet + joins/group-bys) with
  streaming; ~50 s for the full train audit. `audit_candidates` kept only as a pandas test reference.

## F4 — Parquet list columns came back as numpy arrays
- **Symptom:** `ValueError: The truth value of an array with more than one element is ambiguous`
  in `block_keys`.
- **Cause:** `ParquetFile.iter_batches(...).to_pandas()` returns list columns as `numpy.ndarray`,
  not Python lists; `if house and street:` then fails.
- **Fix:** coerce list columns with `list(x)` before use in `block_keys`.

## F5 — Feature pipeline needed a different metadata split
- **Symptom:** validation pairs lived in `pairs/valfull_pairs.parquet` but metadata/processed
  parquet are named for `train`.
- **Fix:** added an optional `meta_split` parameter to `run_features`/`_phase1_merged`.

## F6 — Prediction TSV aggregation OOM
- **Symptom:** `_duckdb.OutOfMemoryException (7.4 GiB/7.4 GiB used)` while writing
  `candidate_pairs.tsv`.
- **Cause:** a single `string_agg` over 250.6M pairs grouped into 1.73M rows exceeded the memory
  limit.
- **Fix:** bucket the candidate pairs by `hash(s1_id) % 64` once, aggregate each bucket separately
  (bounded memory), then byte-concatenate the parts with a single header.

## F7 — Empty match lists written as `""`, validator FAIL
- **Symptom:** `matched_entity_ids contains IDs without an S2-/S3- prefix: ""` (validator exit 1).
- **Cause:** DuckDB's CSV writer quotes empty strings as `""`; the validator then parses a literal
  empty ID.
- **Fix:** add `QUOTE ''` to both CSV `COPY` options so empty lists are written as true empty fields.

## F8 — `data` vs `DATA` path collision on Windows
- **Symptom:** intermediates landed under `DATA/` even though `config.json` said `data`.
- **Cause:** NTFS is case-insensitive and the challenge directory `DATA/` already existed.
- **Fix:** documented as expected behaviour in `AGENTS.md`; the packaged `src/config.json` uses
  `DATA` explicitly.

## F9 — Normalization stripped Indic combining marks
- **Symptom:** test `normalize_name("राम मार्केटिंग")` returned only consonants.
- **Cause:** the regex `[^\w\s]` does not treat combining marks (category Mn/Mc) as word characters.
- **Fix:** replaced regex cleanup with a `unicodedata`-based `_clean_chars` that keeps letters,
  digits, and marks.

## F10 — Suffix class overwritten by stacked suffixes
- **Symptom:** `"Private Ltd"` yielded suffix class `pvt` instead of `ltd`.
- **Cause:** the pop loop kept overwriting the class.
- **Fix:** keep the first (outermost) suffix class while still stripping all suffix tokens;
  `strip_legal_suffix` normalizes internally (uses `fold_name`).

## F11 — `grouped_split` was O(n·m) (43 h)
- **Symptom:** validation split took ~43 h on the full pair set.
- **Cause:** `np.isin(groups, val_groups)` over every group id.
- **Fix (by prior session):** `pd.factorize` + boolean group mask; ~12 s.

## F12 — Threshold tuned on the wrong distribution
- **Symptom:** deployed threshold 0.675, chosen on the 4:1 sample, produced many false merges when
  scored against the full candidate distribution.
- **Cause:** sampled negatives under-represent the confusable candidate mass present at inference.
- **Fix:** re-tuned on full candidates for held-out Source 1 groups -> **0.925**; outputs regenerated.

## F13 — Blocking recall below target
- **Symptom:** initial blocking recall 0.52, then 0.61, then 0.70, then 0.80.
- **Cause:** tight pass caps dropped common-token blocks; only exact/rare-token passes existed.
- **Fix:** added name-token prefix-5, street-token prefix-5, and rare-token pair/triple passes and
  widened caps; final train recall **0.8115**. India (0.727) remains the weak spot; raising recall
  further (embedding/LSH fuzzy blocking) is the top future work item.
