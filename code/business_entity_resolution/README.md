# business_entity_resolution

Business Entity Resolution pipeline for the Amazon ML Challenge 2026. Matches noisy
Source 2 / Source 3 business records to the deduplicated Source 1 reference, evaluated by
macro F_0.5 (precision-heavy, per Source 1 entity, singletons included).

## Approach

A precision-first candidate-generation + classifier cascade:

1. **Clean** — NFKC/casefold, legal-suffix expansion, address abbreviation expansion, ASCII
   folding (`anyascii`) for cross-script keys, PIN/ZIP extraction.
2. **Block** — a single inverted-index pass with a country gate. Name tokens, address tokens,
   phonetic keys (metaphone) and postal codes are scored by **IDF**; the top `max_candidates`
   per Source 1 survive. Raises candidate recall while keeping the candidate set small.
   Measured recall ceiling on the full train ground truth is **1.0**.
3. **Features** — 15 classical pair features (rapidfuzz ratios/Jaccard on folded name and
   address, lengths, country/source flags) plus 5 structural features (postal/phonetic match,
   token overlaps, first-token match). No label-derived features.
4. **Classify** — LightGBM (grouped by Source 1 entity); threshold tuned for macro F_0.5.
5. **Decide** — top-K pruning per Source 1, thresholding, and writing the two submission TSVs.

An autonomous search (`ber.search`) runs the blocking-parameter and matcher/hyperparameter
search entirely on Kaggle, scoring every trial by held-out macro F_0.5, checkpointing
`trials.json`, then finalizing the best config. Candidate keys are cached to parquet so the
search and finalize stages never re-tokenize the 10M-row source sets.

## Environment

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Run (Kaggle)

All heavy compute runs on Kaggle (`kaggle/`). From the repo root:

```powershell
python kaggle\push_all.py datasets --only code          # publish src as a private dataset
python kaggle\push_all.py push --plan auto --only A1    # search (writes best.json + keys)
python kaggle\push_all.py push --plan auto --only A1f   # finalize (writes the TSVs)
python kaggle\push_all.py fetch --collect               # -> output/
```

Validate the outputs from `student_resource/`:

```powershell
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

## Reproduce from raw TSVs

Set `BER_DATA_DIR` to the dataset directory and `BER_ARTIFACT_DIR` to an output directory,
then run the stages in order (locally on CPU, or as the Kaggle notebooks):

```
python -m ber.stages.clean --sample 0
python -m ber.stages.search --smoke-first        # search + finalize (CPU)
```

For a two-step run: `ber.stages.search --search-only` then
`ber.stages.search --finalize-only --search-dir <search-output>`.

## Tests

```powershell
python -m pytest tests -q
```

## Model / license

LightGBM (MIT). The optional encoder stages use `intfloat/multilingual-e5-small` (MIT) and
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0), both <= 8B parameters and pinned by
revision. Only the provided competition data is used.
