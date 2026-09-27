# PLAN.md — BER Execution Plan (index)

The authoritative, detailed execution plan is:

**`docs/superpowers/plans/2026-09-27-ber-execution-plan-v2.md`**

Read that document in full before executing. It is written to be self-contained for an execution agent.

## TL;DR

- **Baseline (frozen):** held-out full-candidate macro F0.5 = **0.8577** (tag `2.0.0`); oracle 0.9122;
  candidate recall 0.8142; official leaderboard 0.811. **0.99 is not a credible target** — see §2 of the plan.
- **Compute reality:** the **RTX Pro 6000 (96 GB) is interactive-only** (browser "Run All"); API/CLI Kaggle runs
  always get **T4x2**. So the **human runs GPU notebooks interactively**, the agent writes code and does all
  local CPU work + Kaggle CLI automation.
- **Phases:** E0 ✅ → E2 ✅ → E1 (e5 cosine features + gate) → E3 (ANN ceiling sweep; STOP for review) →
  E4 (multi-channel + adaptive K; freeze candidates) → E5 matcher → E6 cross-encoder → E7 global assignment →
  E8 pseudo-labeling → E9 DeepSeek → E10 ensemble → E11 calibrate/submit.
- **Global rules:** never destroy the baseline; candidate set changes only in E3/E4; seed 42; no external data;
  validator PASS before submission; report measured values only.

## Where things live

- Detailed plan: `docs/superpowers/plans/2026-09-27-ber-execution-plan-v2.md`
- Earlier plans: `docs/superpowers/plans/2026-09-27-ber-execution-plan.md`,
  `docs/superpowers/plans/2026-09-27-ber-execution-plan-rtx6000.md`
- Runners: `notebooks/e1_blackwell.py` (interactive Blackwell E1), `notebooks/e1_runner.py` (T4/Colab E1),
  `notebooks/e3_kaggle.py` (ANN ceiling)
- Local modules: `code/business_entity_resolution/src/ber/embedding_block.py`, `diagnostic.py`, `colab_io.py`
- Logs/metrics: `docs/PROJECT_LOG.md`, `docs/RESULTS.md`, `DATA/reports/exp_*.json`
