# Submission

## Leaderboard upload (what to submit)

**`matching_results.tsv`** in this folder is the file to upload to the challenge Portal. Per the
problem statement it is the **only file scored on the leaderboard**.

- Tab-separated, header exactly: `source1_entity_id<TAB>matched_entity_ids`
- 1,732,544 data rows (one per test Source 1 entity)
- `matched_entity_ids` empty for entities with no matches (singletons): 200,982 rows
- Matches reference Source 2 / Source 3 IDs only; no duplicates; matches are a subset of candidates
- Produced by `ber.cli predict --split test --one-to-one --threshold 0.925`
- Official validator: **PASS** (format/rules check; it does not compute the score)

Download: https://github.com/Gunjan00001/amazon-hackathon/raw/main/submission/matching_results.tsv

## Final submission package (not needed for the leaderboard)

The full `<team>_submission.zip` (output/ + code/ + Documentation_template.md) is ~1.29 GB and
exceeds GitHub's per-file limit for normal git; it is published as a GitHub Release asset instead:

https://github.com/Gunjan00001/amazon-hackathon/releases/tag/1.1.0

## How to submit to the Portal

1. Download `matching_results.tsv` (link above) or use the byte-identical local copy
   `dist/leaderboard_upload/matching_results.tsv`.
2. Log in to the challenge Portal and upload it as the leaderboard submission.
3. The Portal returns the public-leaderboard macro F0.5 (private split decides final rankings).

Local validation (held-out, full candidates): macro F0.5 **0.8488** (95% CI 0.8481-0.8496);
unseen-country proxy 0.668/0.804. See `docs/RESULTS.md`.
