# Results

Authoritative metric: macro F_0.5 (beta = 0.5), per Source 1 entity, singletons included.
The only true score is produced by the challenge portal from `output/matching_results.tsv`.

## Official leaderboard result

- **Public leaderboard macro F0.5 = 0.811** (submitted 26 Sep 2026, 02:43 PM IST; status: Evaluated).
- This is the real Portal score on the public test split, computed from `output/matching_results.tsv`
  (one-to-one, threshold 0.925).
- It lands inside the predicted **0.80–0.85** band and slightly below the held-out full-candidate
  estimate (0.8488). The gap is consistent with the ~15% unseen-France slice and public/private
  split differences.

## Headline

| Evaluation | macro F0.5 | Notes |
|---|---|---|
| **Official leaderboard (public, Portal)** | **0.811** | real score, 26 Sep 2026 |
| 4:1 sampled split (train, grouped) | 0.9807 | **optimistic, not leaderboard-comparable** |
| **Full candidates, held-out S1 (test-like)** | **0.8488** | baseline v1.0.0; sweep 0.925 + one-to-one |
| **M2: char-3 TF-IDF + per-country/singleton calibration** | **0.8577** | **v1.4.1 adopted** (US 0.8983 / India 0.7968) |
| Unseen-country proxy (train US -> India) | 0.6684 | France proxy (lower bound) |
| Unseen-country proxy (train India -> US) | 0.8041 | France proxy |
| Full model, US val entities | 0.8905 | in-domain |
| Full model, India val entities | 0.7885 | in-domain |

Realistic leaderboard expectation (now confirmed): **~0.80–0.85**. France is ~15% of the test set,
has no labels, and an unseen country costs 0.09–0.12 F0.5 (LOO). Candidate recall ceiling on the
held-out set is **0.8142**, which bounds the maximum achievable score.


## Full-candidate held-out details (`DATA/reports/eval_full_candidates.json`)

- 438,499 held-out Source 1 entities; 60,912,676 candidate pairs; 1,524,017 truth pairs of which
  1,240,887 found (ceiling 0.8142).
- Threshold-only: 0.8475 @ 0.925. One-to-one: **0.8488 @ 0.925** (chosen; v1.0.0 baseline,
  95% CI 0.8481–0.8496).
- Per country: US 0.889, India 0.788. Singletons in val: 23,182.

## M2 (v1.4.1) — char-3 features + per-country/singleton calibration

Two changes, gated on the same full-candidate held-out protocol (candidate set unchanged):

1. **Char-3 TF-IDF cosine features** (`name_char3_cos`, `name_roman_char3_cos`, `addr_char3_cos`;
   vectorizer fit on a 299,997-doc train sample, vocab 21,015). Retrained LightGBM (523 trees,
   4:1 val macro F0.5 0.9824). Uncalibrated full-candidate held-out:
   **0.8574** (US 0.8978 / India 0.7968).
2. **Per-country thresholds + singleton tau**, calibrated on held-out full-candidate predictions
   (25% tuning sample) and scored on all held-out S1 (`DATA/reports/eval_calibrated.json`):
   global 0.95, by country India 0.925 / US 0.95, `singleton_tau` 0.30, one-to-one.
   **chosen_macro_f05 = 0.8577** (95% CI 0.8570–0.8585; US 0.8983 / India 0.7968).

Gate: 0.8577 > 0.8488 and India 0.7968 > 0.788 → **adopted** (tag `1.4.1`; `1.4.0` already marked the M2 plan). Open country France
falls back to the 0.95 global threshold. Remaining M2 headroom is in the matcher (oracle 0.9122) and,
beyond that, the 0.8142 blocking ceiling.

## M2 (Tasks 6–7) — multilingual embeddings: tried, rejected

`intfloat/multilingual-e5-small` (MIT) on the free Colab T4 encoded all 24,229,173 entities; per-pair
`emb_name_cos` / `emb_addr_cos` were merged into train + valfull and the matcher was retrained
(469 trees). Because free-Colab random disk reads over the 18 GB fp16 embedding files were ~37×
amplified, the embeddings were projected 384 → 64 dims (seeded random projection, cosine-preserving)
to fit in RAM — a deviation from the planned 384-dim cosine.

- Uncalibrated held-out: 0.8568 (US 0.8972 / India 0.7962).
- Calibrated held-out (`DATA/reports/eval_calibrated.json`): **0.8570** (US 0.8977 / India 0.7961).
- Gate: 0.8570 < Phase-1 0.8577 → **not adopted**; `models/` and `FEATURE_ORDER` reverted to 1.4.1.

Conclusion: at 64 dims the embedding signal does not beat the char-3-only matcher. A future attempt
should use the full 384-dim cosine on a host with enough RAM/local disk (e.g. Kaggle T4x2) before
concluding embeddings are unhelpful.


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
- **Submitted 26 Sep 2026, 02:43 PM IST — result: macro F0.5 = 0.811 (Evaluated).**

## Matcher vs blocking headroom (`DATA/reports/eval_oracle.json`)

Oracle = ground-truth pairs intersected with the candidate set (prediction precision fixed at 1.0,
i.e. the best a perfect matcher could do on the current candidates), scored macro F0.5 over the
held-out S1 groups with the singleton rule.

- **Oracle macro F0.5 = 0.9122** vs current baseline 0.8488 → matcher headroom **+0.0634**.
- Pair recall (= accepted blocking ceiling) 0.8142; 1,240,887 / 1,524,017 truth pairs found.
- 17,785 / 415,317 entities (4.28%) have truth but zero found candidates; mean per-entity recall 0.8139.
- Per country: US 0.9468, India 0.8602 — India carries the most matcher headroom.
- Decision: headroom > 0.05 → **matcher work (Tasks 2–3) is the lever**; blocking (Task 9) stays
  conditional.

## Next milestone (M2) — status

Plan: `docs/superpowers/plans/2026-09-26-precision-colab.md`. Local phases 0–1 are done and adopted
at **v1.4.1**: diagnostic (oracle 0.9122, matcher is the lever), char-3 TF-IDF features, and
per-country/singleton calibration → held-out **0.8577**. Remaining: Colab T4 multilingual embedding
cosine features (Tasks 5–7, gate > 0.8577), optional cross-encoder rerank (Task 8), conditional
MinHash-LSH re-blocking (Task 9, only if the 0.8142 ceiling binds), then rebuild/validate/submit
(Task 10). Target: held-out > 0.90, leaderboard > 0.85.



