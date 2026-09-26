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

## F14 — LightGBM model failed to load after `git checkout` (CRLF)
- **Symptom:** `[LightGBM] [Fatal] Model format error, expect a tree here. met 65786 43005 ...` on
  `ber.cli predict`, after reverting `models/` with `git checkout`.
- **Cause:** git `core.autocrlf` rewrote `models/lgbm.txt` with CRLF on checkout; LightGBM's text
  parser does not strip the trailing `\r`, so the header/tree keywords are mis-parsed.
- **Fix:** normalized the file to LF in place and added `.gitattributes` with
  `code/business_entity_resolution/models/lgbm.txt -text` so git never converts it again.

## F15 — Free-Colab random disk I/O made embedding cosine infeasible
- **Symptom:** the pair-cosine job read ~28 GB and spent >6 min per 250k-pair batch (process in `D`
  state, ~60 MB/s) over the 18 GB fp16 embedding `memmap` — ~37× read amplification.
- **Cause:** random row gather across a file far larger than RAM/page-cache on Colab's overlay disk.
- **Fix:** one sequential pass applying a seeded random projection 384 → 64 dims (cosine-preserving)
  into RAM, then gather from the compact arrays (~500k pairs/s). Correctness note: the reduced-dim
  cosine was not good enough to pass the gate (see D16) — plan a full-dim re-run on a host with
  enough RAM/local disk rather than judging embeddings from the 64-dim result.

## F16 — DuckDB quirks during the Colab pair-cosine (reserved word, temp OOM)
- **Symptom A:** `COPY (...) ORDER BY r1, r2` to parquet produced a 0-byte file; a later
  `ParquetFile` raised `Parquet file size is 0 bytes`.
- **Cause A:** the index column was named `row`, a reserved SQL word, so the join/select silently
  misbehaved.
- **Fix A:** renamed to `row_idx`.
- **Symptom B:** `OutOfMemoryException: failed to offload data block ... max_temp_directory_size`
  when externally sorting 60.9M rows.
- **Fix B:** abandoned the DuckDB external sort for this job in favour of the in-RAM approach (F15).

## F17 — Colab `files.upload()` truncated files at ~100 MB
- **Symptom:** uploaded `entities.parquet` etc. arrived ~99/89/78 MB and were unreadable
  (`Parquet magic bytes not found in footer`).
- **Cause:** the `google.colab.files.upload()` widget caps per-file size (~100 MB).
- **Fix:** transfer large inputs via Google Drive / the Colab Files panel; keep `files.download`
  for outputs and verify row counts after transfer.

## F18 — `Compress-Archive` could not zip the submission (2 GB stream limit)
- **Symptom:** `Exception calling "Write" ... "Stream was too long."` building the submission zip
  (3.1 GB `candidate_pairs.tsv`).
- **Cause:** Windows PowerShell 5.1 `Compress-Archive` uses a .NET stream limited to 2 GB per entry.
- **Fix:** build the zip with Python `zipfile` (ZIP64, `ZIP_DEFLATED`) — 3.1 GB → ~1.38 GB.
