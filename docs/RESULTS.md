# Results

Authoritative metric: macro F_0.5 (beta = 0.5), per Source 1 entity, singletons included.
The only true score is produced by the challenge portal from `output/matching_results.tsv`.

## Headline

| Evaluation | macro F0.5 | Notes |
|---|---|---|
| 4:1 sampled split (train, grouped) | 0.9807 | **optimistic, not leaderboard-comparable** |
| **Full candidates, held-out S1 (test-like)** | **0.8488** | 95% CI 0.8481–0.8496 |
| Unseen-country proxy (train US -> India) | 0.6684 | France proxy (lower bound) |
| Unseen-country proxy (train India -> US) | 0.8041 | France proxy |
| Full model, US val entities | 0.8905 | in-domain |
| Full model, India val entities | 0.7885 | in-domain |

Realistic leaderboard expectation: **~0.80–0.85**. France is ~15% of the test set, has no labels,
and an unseen country costs 0.09–0.12 F0.5 (LOO). Candidate recall ceiling on the held-out set is
**0.8142**, which bounds the maximum achievable score.

## Full-candidate held-out details (`DATA/reports/eval_full_candidates.json`)

- 438,499 held-out Source 1 entities; 60,912,676 candidate pairs; 1,524,017 truth pairs of which
  1,240,887 found (ceiling 0.8142).
- Threshold-only: 0.8475 @ 0.925. One-to-one: **0.8488 @ 0.925** (chosen).
- Per country: US 0.889, India 0.788. Singletons in val: 23,182.

## 4:1 sampled details (`DATA/reports/eval_marks.json`)

- Threshold-only 0.98033 @ 0.700; one-to-one 0.98075 @ 0.675.
- Precision 0.988, recall 0.977, 40,967 singleton entities, 1,352 false merges on singletons.
- Included only to show the optimism gap versus the full-candidate number.

## Leave-one-country-out (`DATA/reports/eval_loo.json`)

- train US -> validate India: 0.6684 (ceiling 0.7322).
- train India -> validate US: 0.8041 (ceiling 0.8690).
- Full model: US 0.8905, India 0.7885.
- Interpretation: cross-country transfer loses 0.086–0.121 F0.5.

## Test submission (`output/`)

- `matching_results.tsv`: 1,732,544 rows total, 200,982 empty (singletons), 1,533,562 non-empty.
- `candidate_pairs.tsv`: 1,732,544 rows, 554 empty, 1,731,990 non-empty.
- 5,071,867 candidate pairs scored above threshold 0.925.
- Method: one-to-one post-process, threshold 0.925.
- Official validator: **PASS**.

## Reproduce

```
ber.cli prepare
ber.cli block --split both
ber.cli audit --split train
ber.cli features --split train --workers 8 --combine
ber.cli train
ber.cli features --split test --workers 8
ber.cli predict --split test --one-to-one --threshold 0.925
ber.cli evaluate        # 4:1 marks
ber.cli validation --workers 8   # believable full-candidate mark
ber.cli loo             # unseen-country proxy
```

## Leaderboard upload

Per PROBLEM_STATEMENT.md §13, the leaderboard submission is **only** `output/matching_results.tsv`:

- Tab-separated, header exactly `source1_entity_id<TAB>matched_entity_ids`.
- One row per test Source 1 entity (1,732,544 rows), empty `matched_entity_ids` for singletons.
- This is the file uploaded in the Portal; public and private leaderboards are computed from it
  (public = a subset, private = the remainder; final rankings use the private split).
- A byte-identical upload copy is staged at `dist/leaderboard_upload/matching_results.tsv`.
- `candidate_pairs.tsv` is not scored on the leaderboard; it belongs to the final submission zip.

