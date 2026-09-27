# SUBMIT — Amazon ML Challenge 2026 (Business Entity Resolution)

**Team:** AA..
**Official public leaderboard (Portal):** macro F0.5 = **0.811** (submitted 2026-09-26, status Evaluated)
**Held-out full-candidate estimate (train, test-like):** macro F0.5 = **0.8577**

---

## Files in this folder

| File | Size | What it is | Where it goes |
|---|---|---|---|
| **`matching_results.tsv`** | 75.6 MB | Final entity matches. **The only file scored on the leaderboard.** | Upload to the challenge **Portal**. |
| `candidate_pairs.tsv` | 3.10 GB | Blocking / candidate-generation set actually fed to the matcher. | Goes inside the final package `output/` (not scored on the leaderboard). |
| `AA.._submission.zip` | 1.32 GB | Full final submission package (see structure below). | Submit as the team's final package. |

All three are byte-identical copies (hardlinks) of the working outputs in `output/` and `dist/`.

---

## `matching_results.tsv` format

- Tab-separated, UTF-8, with header: `source1_entity_id<TAB>matched_entity_ids`
- Exactly one row per test Source 1 entity: **1,732,544 rows** (+ header)
- `matched_entity_ids` is a comma-separated list of `S2-`/`S3-` test IDs, **empty for singletons**
- No duplicate IDs in a list; no duplicate `source1_entity_id`; no `S1-` self-matches
- 212,118 empty (singletons); 1,520,426 non-empty

## `candidate_pairs.tsv` format

- Same rules, header `source1_entity_id<TAB>candidate_entity_ids`
- 1,732,544 rows; 554 empty; every ID in `matching_results.tsv` is a subset of these candidates

## Validator

Run from `DATA/student_resource/`:
```bash
python utils/validate_submission.py \
  --matching SUBMIT/matching_results.tsv \
  --candidate SUBMIT/candidate_pairs.tsv \
  --test-dir dataset/test
```
Result: **PASS** — no blocking issues; safe to submit.

---

## `AA.._submission.zip` structure (per Problem Statement §9)

```
AA.._submission/
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/            # runnable pipeline (ber package + config.json)
│   ├── README.md       # reproduction steps
│   └── requirements.txt# pinned dependencies
└── Documentation_template.md   # methodology write-up
```

## Method (one paragraph)

Blocking-plus-classifier pipeline. DuckDB blocking passes (exact name, rare-token, address
prefix, postal+name, name/street prefix-5, rare-token pairs/triples) reduce each Source 1
entity to a bounded candidate set. A LightGBM pairwise classifier (RapidFuzz string metrics,
address/structured features, char-3 TF-IDF cosine) scores every candidate. A threshold tuned
for macro F0.5 on the **full candidate distribution** (global 0.95; India 0.925 / US 0.95),
a singleton rule (`max prob < tau` → empty), and an injective one-to-one assignment produce
the final matches. All signal comes from the provided challenge data only.

## Reproduction (from `code/business_entity_resolution/`)

```bash
export PYTHONPATH=src
python -m ber.cli prepare   --config config.json
python -m ber.cli block --split both --config config.json
python -m ber.cli audit     --split train --config config.json
python -m ber.cli features  --split train --workers 8 --combine --config config.json
python -m ber.cli train     --config config.json
python -m ber.cli calibrate --config config.json
python -m ber.cli features  --split test --workers 8 --config config.json
python -m ber.cli predict   --split test --one-to-one --config config.json
```

## Repo

https://github.com/Gunjan00001/amazon-hackathon (tag `2.1.0`)
