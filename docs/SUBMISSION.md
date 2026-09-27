# SUBMISSION.md — Final submission documentation

**Team:** AA..
**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution
**Repo:** https://github.com/Gunjan00001/amazon-hackathon (git tag `2.1.0`)
**Official public leaderboard:** macro F0.5 = **0.811** (submitted 2026-09-26, status Evaluated)
**Held-out full-candidate estimate:** macro F0.5 = **0.8577** (train held-out S1 groups, test-like)

---

## 1. Deliverables (ready in `SUBMIT/`)

| File | Size | Purpose |
|---|---|---|
| `SUBMIT/matching_results.tsv` | 75.6 MB | **Leaderboard upload** — the only scored file |
| `SUBMIT/candidate_pairs.tsv` | 3.10 GB | Blocking candidate set (final package `output/`) |
| `SUBMIT/AA.._submission.zip` | 1.32 GB | Full final submission package (code + outputs + methodology) |
| `SUBMIT/README.md` | — | Folder-level manifest and instructions |

Equivalent locations: `output/matching_results.tsv`, `output/candidate_pairs.tsv`,
`dist/AA..__submission.zip`, `submission/matching_results.tsv` (tracked in git).

## 2. Output contract

- **`matching_results.tsv`** — TSV, header `source1_entity_id<TAB>matched_entity_ids`;
  1,732,544 rows (one per test Source 1); `S2-`/`S3-` IDs only; empty lists for singletons;
  212,118 empty / 1,520,426 non-empty.
- **`candidate_pairs.tsv`** — TSV, header `source1_entity_id<TAB>candidate_entity_ids`;
  1,732,544 rows; 554 empty; `matches ⊆ candidates`.
- **Validator:** `PASS` (run from `DATA/student_resource`), 0 blocking issues.

## 3. Method

Blocking-plus-classifier:

1. **Prepare** — normalize names (fold, legal-suffix strip, script detect), rule-based
   transliteration, country-aware address parsing; cache to parquet.
2. **Block** — DuckDB blocking passes: (1) exact normalized name, (3) rare-token exact,
   (4) `house_no|street[:5]`, (5) `postal|first-name-token`, (6) fallback `state|name[:3]`,
   (7) name-token prefix-5, (8) street-token prefix-5, (9) rare-token pairs, (10) rare-token
   triples; per-pass block caps + a per-Source-1 cap of 200.
3. **Features** — LightGBM pairwise features: RapidFuzz ratios (name/roman), token-set
   metrics, address/structured matches, char-3 TF-IDF cosine (`name`, `roman`, `addr`).
4. **Train** — LightGBM binary classifier on all found positives plus 4:1 sampled negatives,
   grouped 80/20 split by Source 1 entity.
5. **Calibrate** — threshold tuned on the **full candidate distribution** of the held-out S1
   groups (not the 4:1 sample): global 0.95; India 0.925 / US 0.95 (France falls back to the
   global 0.95); singleton rule `max(prob) < 0.30` → empty list; injective one-to-one.
6. **Predict** — apply the above to the test candidate set and write the output TSVs.

All signals come from the provided challenge data only; no external data or APIs (RULES §1).

## 4. Results

| Evaluation | macro F0.5 |
|---|---|
| **Official leaderboard (public, Portal)** | **0.811** |
| Held-out full-candidate (test-like) | 0.8577 (US 0.8983 / India 0.7968) |
| Oracle (candidate-recall ceiling) | 0.9122 |
| Unseen-country proxy (leave-one-country-out) | 0.668–0.804 |

Held-out candidate recall ceiling 0.8142; 4.28% of truth-bearing entities have zero candidates.

## 5. Reproduction

From `code/business_entity_resolution/` (see `src/`, `README.md`, `requirements.txt`):

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

## 6. Honest scope note

A larger gain was measured but **not adopted** for lack of compute time:

- **E1 (e5 multilingual features)** — held-out **0.8590** (US 0.8994 / India 0.7984); the code and
  the trained model exist (`models_e1/`, git `11b88bc`), but the model **could not be applied to
  test** because the test candidate pairs lacked the e5 cosine columns and no GPU re-run was possible.
- **E3 (e5 ANN retrieval ceiling)** — lexical ∪ ANN raises the **oracle to 0.9892** (pair recall
  0.9700 at K=2000). The production E4 step that turns this into candidates was not completed
  (interactive GPU OOM/time). Artifacts: `DATA/reports/exp_E3_curves.json`.

The submitted result therefore remains the adopted 1.4.1 baseline (held-out 0.8577 → leaderboard 0.811).
