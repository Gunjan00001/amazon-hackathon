# Graph Report - Amazon project  (2026-09-29)

## Corpus Check
- 76 files · ~65,967 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1211 nodes · 2009 edges · 99 communities (86 shown, 13 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 228 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Project Docs and Plans
- Prepare and Normalization
- Blocking Audit
- Pipeline Configuration
- Kaggle and Colab Runners
- Blocking and Candidates
- Normalization and Transliteration
- Problem Statement Format
- Config Pass Caps
- Operations Log and Handover
- Feature Engineering
- Feature Computation
- Parallel Features and Output
- Candidate Pair Output
- EDA and Benchmark Tooling
- CLI and Evaluation
- Failures and Fixes Log
- Pair Construction
- Threshold Sweeping
- Architecture Decisions
- M2 Plan Tasks
- Scoring and One-to-One
- Validation and Evaluation
- Results and Reproduction
- Pipeline Design Spec
- Blocking Keys
- Colab IO
- Calibration
- Calibration Scoring
- Methodology Template
- Colab Export and Merge
- Full Candidate Evaluation
- Dataset Documentation
- Submission Template Docs
- Embedding Model Choice
- Calibration Decisions
- Repository Overview
- Per-Country Tuning
- Submission Validator
- Kaggle Cascade Design
- Macro F0.5 Scoring
- Code Run Instructions
- Prediction Pipeline
- Recall Ceiling Diagnostics
- Project Rules
- M2 Design Spec
- Agent Working Notes
- Scaffold Tasks
- Diagnostic Oracle
- Blocking Design Decisions
- DuckDB Pipeline Decisions
- Local Environment Setup
- Diagnostic Runner
- Char Features and Thresholds
- Pipeline Implementation Plan
- Prepare Stage Task
- Predict and Booster Loading
- LightGBM Matcher Decisions
- Project File Structure
- Blocking Audit Task
- Submission Packaging
- One-to-One Assignment
- Cleaning Module Task
- TSV IO Task
- Blocking Module Task
- Pair Features Task
- GBDT Wrapper Task
- Decision Logic Task
- Embedding Stage Task
- Name Normalization Task
- Transliteration Task
- Address Parsing Task
- Production Run Task
- Compute Placement Decisions
- ER Pipeline Plan
- Kaggle Notebook Chain
- Kaggle Notebooks Task
- Cross-Encoder Task
- Packaging Task
- Final Validation Task
- DuckDB Blocking Task
- Pairwise Features Task
- Metric and Training Task
- Test Prediction Task
- End-to-End Test Task
- Calibration Test
- Normalize Test
- Candidate Pairs Artifact
- DATA Intermediate Store
- Regenerable Artifacts
- Submission Tree Decision
- Grouped Split Performance Fix
- Colab Cosine OOM Fix
- Zip Size Limit Fix
- Parquet List Columns Fix
- Prediction Aggregation OOM Fix
- Empty Match List Fix

## God Nodes (most connected - your core abstractions)
1. `AGENTS.md Working Notes` - 21 edges
2. `main (ber CLI)` - 20 edges
3. `Failures and Fixes` - 19 edges
4. `Decisions and Reasoning (DECISIONS.md)` - 18 edges
5. `M2 Precision + Colab Plan (10 Tasks)` - 18 edges
6. `main()` - 18 edges
7. `Config` - 17 edges
8. `prepare_frame` - 17 edges
9. `Decisions and Reasoning` - 17 edges
10. `Business Entity Resolution Challenge` - 17 edges

## Surprising Connections (you probably didn't know these)
- `bench_gbdt.compute_features` --semantically_similar_to--> `compute_features`  [INFERRED] [semantically similar]
  tools/bench_gbdt.py → code/business_entity_resolution/src/ber/features.py
- `bench_gbdt.entity_macro_f05` --semantically_similar_to--> `macro_f05`  [INFERRED] [semantically similar]
  tools/bench_gbdt.py → code/business_entity_resolution/src/ber/evaluate.py
- `bench_gbdt.entity_macro_f05` --semantically_similar_to--> `entity_f05`  [INFERRED] [semantically similar]
  tools/bench_gbdt.py → code/business_entity_resolution/src/ber/threshold.py
- `Submission Folder README` --semantically_similar_to--> `SUBMIT Folder README`  [INFERRED] [semantically similar]
  submission/README.md → SUBMIT/README.md
- `Problem Statement (Business Entity Resolution)` --semantically_similar_to--> `Official Challenge README (student_resource)`  [INFERRED] [semantically similar]
  PROBLEM_STATEMENT.md → DATA/student_resource/README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Blocking-plus-Classifier Pipeline** — docs_decisions_d1_blocking_classifier, docs_decisions_d3_ten_blocking_passes, docs_decisions_d5_lightgbm, er_pipeline_design_stage1_blocking, er_pipeline_design_stage4_training [EXTRACTED 1.00]
- **Blocking-to-Submission Pipeline** — ber_cli_pipeline, blocking_candidate_generation, lightgbm_pairwise_matcher, per_country_calibration, one_to_one_assignment, matching_results_tsv [EXTRACTED 1.00]
- **M2 Precision Lift Milestone** — precision_colab_plan_task2_char, precision_colab_plan_task3_calibration, docs_decisions_d14_char3_tfidf, docs_decisions_d15_per_country_thresholds, docs_results_m2_char3 [EXTRACTED 1.00]
- **Documents To Read Before Changing Code** — rules_project_rules, problem_statement_spec, data_student_resource_dataset_dataset_measured_facts, agents_working_notes [EXTRACTED 1.00]
- **Repository Archival and Handover** — docs_decisions_d17_archive_strategy, docs_handover_restore_guide, docs_handover_github_tracking, docs_archive_manifest_inventory, docs_archive_manifest_git_lfs_objects [EXTRACTED 1.00]
- **Final Submission Package Contents** — matching_results_tsv, candidate_pairs_tsv, ber_code_readme, code_business_entity_resolution_requirements_pinned, documentation_template_solution [EXTRACTED 1.00]
- **Per-country threshold tuning, sweep and application flow** — code_business_entity_resolution_src_ber_calibration_tune_per_country, code_business_entity_resolution_src_ber_threshold_sweep_threshold, code_business_entity_resolution_src_ber_calibration_apply_calibration, code_business_entity_resolution_tests_test_calibration_test_tune_per_country_recovers_distinct_thresholds [INFERRED 0.80]
- **GPU e5 embedding + ANN candidate generation notebooks (E1/E3/E4)** — notebooks_e1_runner, notebooks_e1_blackwell, notebooks_e1e3_blackwell, notebooks_e3_kaggle, notebooks_e4_blackwell [INFERRED 0.80]
- **Name and address normalization flow** — code_business_entity_resolution_src_ber_normalize_normalize_name, code_business_entity_resolution_src_ber_address_parse_address, code_business_entity_resolution_src_ber_translit_romanize, code_business_entity_resolution_src_ber_prepare_prepare_frame, code_business_entity_resolution_src_ber_features_feature_block [INFERRED 0.80]
- **Candidate generation to feature flow** — code_business_entity_resolution_src_ber_blocking_run_block, code_business_entity_resolution_src_ber_blocking_block_keys, code_business_entity_resolution_src_ber_audit_run_audit, code_business_entity_resolution_src_ber_pairs_write_training_pairs, code_business_entity_resolution_src_ber_features_run_features [INFERRED 0.85]
- **DuckDB blocking audit verified field-by-field against pandas reference** — code_business_entity_resolution_src_ber_audit_run_audit, code_business_entity_resolution_src_ber_audit_audit_candidates, code_business_entity_resolution_tests_test_audit_test_run_audit_matches_pandas_reference [INFERRED 0.85]
- **Matcher training and calibration flow** — code_business_entity_resolution_src_ber_train_train_model, code_business_entity_resolution_src_ber_validation_evaluate_full_candidates, code_business_entity_resolution_src_ber_calibration_run_calibration, code_business_entity_resolution_src_ber_threshold_sweep_threshold, code_business_entity_resolution_src_ber_predict_run_predict [INFERRED 0.85]

## Communities (99 total, 13 thin omitted)

### Community 0 - "Project Docs and Plans"
Cohesion: 0.06
Nodes (92): M2 Adoption Gates (numeric comparisons vs 0.8488), AGENTS.md Working Notes, ber.cli Staged Pipeline, Pipeline Code README, Blocking / Candidate Generation, Per-S1 Candidate Cap 200 + Per-Pass pass_caps, Blocking + Classifier Approach (D1), Blocking Recall Ceiling 0.814 (+84 more)

### Community 1 - "Prepare and Normalization"
Cohesion: 0.07
Nodes (45): prepare_frame(), run_prepare(), _write_stream(), AddressParts, parse_address, _postal_re(), _clean_chars, detect_script (+37 more)

### Community 2 - "Blocking Audit"
Cohesion: 0.07
Nodes (39): audit_candidates(), _connect(), _explode_truth(), load_country_map(), _Progress, Reference (in-memory pandas) implementation. Kept for tests/cross-checks., run_audit(), _sql_path() (+31 more)

### Community 3 - "Pipeline Configuration"
Cohesion: 0.06
Nodes (38): config.json (root), Config, ber package __init__, test_config_load_roundtrip, test_parallel_features_match_single_process, _cfg(), _matches(), _stage() (+30 more)

### Community 4 - "Kaggle and Colab Runners"
Cohesion: 0.08
Nodes (40): E1 Blackwell runner (full-resident GPU e5 cosine), build_entities(), cosine_split(), encode_field(), field_texts(), _find_file(), log(), _pip() (+32 more)

### Community 5 - "Blocking and Candidates"
Cohesion: 0.10
Nodes (32): block_keys, compute_token_idf, DEFAULT_PASS_CAPS, generate_candidates, _join_sql, load_token_idf, normalize_keys(), _pass_caps (+24 more)

### Community 6 - "Normalization and Transliteration"
Cohesion: 0.10
Nodes (31): AddressParts, parse_address(), _postal_re(), _clean_chars(), detect_script(), fold_name(), name_tokens(), normalize_name() (+23 more)

### Community 7 - "Problem Statement Format"
Cohesion: 0.07
Nodes (27): 10. Constraints, 11. Evaluation Criteria, 12. Leaderboard Information, 13. Submission Requirements, 14. Academic Integrity and Fair Play, 15. Tips for Success, 1. Overview, 2. File Format (+19 more)

### Community 8 - "Config Pass Caps"
Cohesion: 0.07
Nodes (27): cap, data_dir, dataset_dir, idf_min, lgbm_params, colsample_bytree, learning_rate, min_child_samples (+19 more)

### Community 9 - "Operations Log and Handover"
Cohesion: 0.08
Nodes (26): Git LFS Objects on main, Archive Inventory (80 GB local folder), D17 Archive Strategy: Code+Docs in Git, Data Out of Git, What Is on GitHub vs What Is Not, Private Kaggle Dataset amz-er-2026-raw, Python 3.12 venv Restore, HANDOVER Restore Guide, 2026-09-25 / 26 — environment and tooling (+18 more)

### Community 10 - "Feature Engineering"
Cohesion: 0.15
Nodes (23): attach_country(), char3_cosine(), _combine_feature_parts(), compute_features(), _country_map(), ensure_char_vectorizer(), _feature_block(), fit_char_vectorizer() (+15 more)

### Community 11 - "Feature Computation"
Cohesion: 0.16
Nodes (23): attach_country, char3_cosine, _combine_feature_parts(), compute_features, ensure_char_vectorizer, _feature_block, fit_char_vectorizer, load_char_vectorizer (+15 more)

### Community 12 - "Parallel Features and Output"
Cohesion: 0.13
Nodes (17): Config, _write_tsv(), _config(), _make_dataset(), _prepared(), Path, test_parallel_features_match_single_process(), cfg() (+9 more)

### Community 13 - "Candidate Pair Output"
Cohesion: 0.09
Nodes (22): **Academic Integrity and Fair Play:**, Business Entity Resolution Challenge, candidate_pairs.tsv, code:python (import pandas as pd), code:block2 (source1_entity_id	matched_entity_ids), code:block3 (source1_entity_id	candidate_entity_ids), code:bash (python3 utils/validate_submission.py \), code:block5 (<team_name>_submission.zip) (+14 more)

### Community 14 - "EDA and Benchmark Tooling"
Cohesion: 0.22
Nodes (21): tools/bench_gbdt.py (LightGBM vs XGBoost), best_entity_f05(), build_pairs(), bench_gbdt.compute_features, bench_gbdt.entity_macro_f05, load_positives(), load_s1_sample(), main() (+13 more)

### Community 15 - "CLI and Evaluation"
Cohesion: 0.18
Nodes (17): _default_config(), main (ber CLI), evaluate_marks, _one_to_one_mask, FEATURE_ORDER, build_training_pairs, _explode_truth (pairs), grouped_split (+9 more)

### Community 16 - "Failures and Fixes Log"
Cohesion: 0.10
Nodes (19): F10 — Suffix class overwritten by stacked suffixes, F11 — `grouped_split` was O(n·m) (43 h), F12 — Threshold tuned on the wrong distribution, F13 — Blocking recall below target, F14 — LightGBM model failed to load after `git checkout` (CRLF), F15 — Free-Colab random disk I/O made embedding cosine infeasible, F16 — DuckDB quirks during the Colab pair-cosine (reserved word, temp OOM), F17 — Colab `files.upload()` truncated files at ~100 MB (+11 more)

### Community 17 - "Pair Construction"
Cohesion: 0.16
Nodes (15): _default_config(), main(), build_training_pairs(), _explode_truth(), grouped_split(), write_inference_pairs(), write_training_pairs(), train_model() (+7 more)

### Community 18 - "Threshold Sweeping"
Cohesion: 0.18
Nodes (16): ber.calibration.apply_calibration, singleton_decision, tune_per_country, _entity_f05_from_codes, sweep_threshold, _country_data (calibration fixture), test_apply_calibration_without_tau_is_plain_threshold, test_apply_calibration_without_tau_is_plain_threshold() (+8 more)

### Community 19 - "Architecture Decisions"
Cohesion: 0.11
Nodes (17): D10 — Leave-one-country-out as the France proxy, D11 — Ship exactly the required submission tree, D12 — Local CPU for stages 0–1 and 5, Colab/Kaggle GPU for transformer stages, D13 — Colab returns per-pair cosine features, not full embeddings, D14 — Add char n-gram TF-IDF cosine features, D15 — Per-country thresholds + singleton calibration, D16 — Multilingual embedding cosine: tried at 64 dims, not adopted, D1 — Blocking plus classifier, not end-to-end (+9 more)

### Community 20 - "M2 Plan Tasks"
Cohesion: 0.12
Nodes (16): code:python (def test_oracle_macro():), code:python (import numpy as np, pandas as pd, torch, pyarrow.parquet as ), code:block3 (.venv\Scripts\python.exe DATA\student_resource\utils\validat), Global constraints, Precision Lift + Colab Embeddings — Implementation Plan (M2), Self-review checklist, Task 10: Rebuild outputs, validate, package, submit (milestone 2.0.0), Task 1: Diagnostic — oracle ceiling and per-country headroom (+8 more)

### Community 21 - "Scoring and One-to-One"
Cohesion: 0.15
Nodes (13): one_to_one(), macro_f05, one_to_one, DataFrame, test_entity_f05_singleton_rule(), test_macro_f05_worked_example, test_macro_f05_singleton_rules(), test_macro_f05_worked_example() (+5 more)

### Community 22 - "Validation and Evaluation"
Cohesion: 0.28
Nodes (14): _bin_counts(), build_valfull_pairs(), _country_of_source1(), evaluate_full_candidates(), evaluate_loo(), _macro_at_bins(), _macro_from_counts(), _pivot() (+6 more)

### Community 23 - "Results and Reproduction"
Cohesion: 0.12
Nodes (15): 4:1 sampled details (`DATA/reports/eval_marks.json`), code:block1 (ber.cli prepare), Full-candidate held-out details (`DATA/reports/eval_full_candidates.json`), Headline, Leaderboard upload, Leave-one-country-out (`DATA/reports/eval_loo.json`), M2 (Tasks 6–7) — multilingual embeddings: tried, rejected, M2 (v1.4.1) — char-3 features + per-country/singleton calibration (+7 more)

### Community 24 - "Pipeline Design Spec"
Cohesion: 0.12
Nodes (15): 10. Package layout and CLI, 11. Testing and verification, 12. Milestones, risks, non-goals, 1. Context and goal, 2. Locked decisions, 3. Architecture and data flow, 4. Stage 0 — Cleaning and normalization, 5. Stage 1 — Blocking / candidate generation (+7 more)

### Community 25 - "Blocking Keys"
Cohesion: 0.27
Nodes (13): block_keys(), compute_token_idf(), generate_candidates(), _join_sql(), load_token_idf(), normalize_keys(), _pass_caps(), run_block() (+5 more)

### Community 26 - "Colab IO"
Cohesion: 0.27
Nodes (13): _configure (colab_io), export_entities, export_for_colab, export_pairs, _feature_targets, import_cosine, merge_cosine, _build_cfg (colab_io fixture) (+5 more)

### Community 27 - "Calibration"
Cohesion: 0.30
Nodes (13): apply_calibration(), _calibration_device(), _configure(), load_calibration(), _pred_glob(), run_calibration(), save_calibration(), score_calibration() (+5 more)

### Community 28 - "Calibration Scoring"
Cohesion: 0.30
Nodes (12): _calibration_device(), _configure (calibration), _pred_glob(), run_calibration, score_calibration, _truth_table_sql(), tune_calibration, tune_singleton_tau (+4 more)

### Community 29 - "Methodology Template"
Cohesion: 0.15
Nodes (12): 1. Executive Summary, 2.1 Problem Analysis, 2.2 Solution Strategy, 2. Methodology, 3. Candidate Generation (Blocking), 4. Matching Model, 5. Results & Error Analysis, 6. Conclusion (+4 more)

### Community 30 - "Colab Export and Merge"
Cohesion: 0.29
Nodes (11): _configure(), export_entities(), export_for_colab(), export_pairs(), _feature_targets(), import_cosine(), merge_cosine(), _build_cfg() (+3 more)

### Community 31 - "Full Candidate Evaluation"
Cohesion: 0.38
Nodes (12): _load_booster, _bin_counts, build_valfull_pairs, evaluate_full_candidates, evaluate_loo, _macro_at_bins, _pivot, _reports() (+4 more)

### Community 32 - "Dataset Documentation"
Cohesion: 0.15
Nodes (12): 1. TL;DR, 2. File inventory, 3. Schema, 4. Country distribution (measured), 5. Ground-truth analysis (measured), 6. Data quality and encoding notes, 7. Implications for the pipeline, 8. Reproduction (+4 more)

### Community 33 - "Submission Template Docs"
Cohesion: 0.15
Nodes (12): 1. Executive Summary, 2.1 Problem Analysis, 2.2 Solution Strategy, 2. Methodology, 3. Candidate Generation (Blocking), 4. Matching Model, 5. Results & Error Analysis, 6. Conclusion (+4 more)

### Community 34 - "Embedding Model Choice"
Cohesion: 0.17
Nodes (13): D13 Colab Returns Per-Pair Cosine, not Full Embeddings, D16 Multilingual Embedding Cosine Tried at 64 dims, Not Adopted, E1 e5 Embeddings on Kaggle, M2 Embeddings Rejected (calibrated 0.8570), Blackwell-interactive + Local CPU Execution Plan v2, E1 Full-dim e5 Cosine Features + Gate, E6 Cross-Encoder Rerank, E9 DeepSeek Selective Reranking (+5 more)

### Community 35 - "Calibration Decisions"
Cohesion: 0.17
Nodes (13): D6 4:1 Negative Sampling for Training, D8 Threshold Must Be Tuned on Full Candidates (0.925), D9 Full-Candidate Held-Out Validation Is the Reported Number, Measured Pipeline Runs, Held-Out Full-Candidate macro F0.5 = 0.8577, Macro F0.5 Metric, Stage 0 Cleaning and Normalization, Stage 0 Prepare Pipeline (+5 more)

### Community 36 - "Repository Overview"
Cohesion: 0.17
Nodes (11): Amazon ML Challenge 2026 — Business Entity Resolution, code:block1 (PROBLEM_STATEMENT.md          # full transcription of the of), code:powershell (# 1. Python 3.12 environment (uv)), Documentation, Fair play, Key dataset facts, Matcher benchmark (0.1.0), Quickstart (+3 more)

### Community 37 - "Per-Country Tuning"
Cohesion: 0.29
Nodes (10): tune_per_country(), _entity_f05_from_codes(), sweep_threshold(), _country_data(), test_apply_calibration_without_tau_is_plain_threshold(), test_singleton_decision_suppresses_low_max_groups(), test_tune_per_country_falls_back_for_sparse_country(), test_tune_per_country_recovers_distinct_thresholds() (+2 more)

### Community 38 - "Submission Validator"
Cohesion: 0.27
Nodes (11): examples(), load_match_targets(), main(), Validate the submission output(s); return ``(errors, warnings)`` lists.      ``c, Return the set of first-column entity IDs from a source TSV.      The header row, Return a short, human-readable sample of ``items`` for an error message., Return the set of valid S2/S3 match IDs, or ``None`` if unavailable.      Only c, Validate one results-style TSV (matching or candidate).      Applies the shared (+3 more)

### Community 39 - "Kaggle Cascade Design"
Cohesion: 0.17
Nodes (11): 1. Context and constraints, 2. Approach — cascade C+B, 3. Kaggle execution model, 4. Stage details, 5. 24-hour schedule, 6. Verification and gates, 7. Deliverables, 8. Risks and mitigations (+3 more)

### Community 40 - "Macro F0.5 Scoring"
Cohesion: 0.25
Nodes (9): evaluate_marks(), macro_f05(), _one_to_one_mask(), entity_f05(), float, test_entity_f05_singleton_rule(), test_macro_f05_singleton_rules(), test_macro_f05_worked_example() (+1 more)

### Community 41 - "Code Run Instructions"
Cohesion: 0.18
Nodes (10): Business Entity Resolution — Code and Run Instructions, code:block1 (uv venv .venv --python 3.12), code:powershell ($env:PYTHONPATH="code/business_entity_resolution/src"), code:powershell (# 1. Clean + normalize all six record files (cached to DATA/), code:block4 (src/ber/), Environment, Outputs, Package layout (+2 more)

### Community 42 - "Prediction Pipeline"
Cohesion: 0.31
Nodes (10): load_calibration, save_calibration, _country_map, _feature_sources, _load_calibration, predict_parts, run_predict, ber.predict._write_tsv (+2 more)

### Community 43 - "Recall Ceiling Diagnostics"
Cohesion: 0.22
Nodes (11): E3 ANN Ceiling on RTX Pro 6000, Blocking Recall Ceiling 0.8142, Oracle macro F0.5 = 0.9122, Honest Scope Note (E1/E3 not shipped), E3 ANN K-sweep (Measure Candidate Ceiling), E4 Multi-Channel Retrieval + Adaptive K, F13 Blocking Recall Below Target, Bottleneck Analysis (+3 more)

### Community 44 - "Project Rules"
Cohesion: 0.20
Nodes (9): 1. Challenge rules (from the official problem statement), 2. Data rules, 3. Engineering rules, 4. Verification commands, 5. Versioning and git rules, 6. Compute rules — local CPU + Kaggle/Colab GPU, code:powershell (uv venv .venv --python 3.12), code:powershell (# 1) Dataset EDA (writes tools/eda_stats.json, caches in too) (+1 more)

### Community 45 - "M2 Design Spec"
Cohesion: 0.20
Nodes (9): 1. Goal, 2. Why this plan (bottleneck analysis), 3. Environments, 4. Interfaces (small transfers), 5. Phases and adoption gates, 6. Validation protocol (reused, unchanged), 7. Risks, 8. Out of scope (+1 more)

### Community 46 - "Agent Working Notes"
Cohesion: 0.22
Nodes (8): AGENTS.md — working notes for this repository, Audit contract, Blocking contracts, Colab GPU stages, Environment, Gotchas already learned, Matcher + calibration contract, Pipeline data contracts (all on disk, parquet/JSON)

### Community 47 - "Scaffold Tasks"
Cohesion: 0.22
Nodes (9): code:python (# code/business_entity_resolution/tests/conftest.py), code:python (# code/business_entity_resolution/src/ber/__init__.py), code:python (# code/business_entity_resolution/src/ber/config.py), code:json (// code/business_entity_resolution/config.json), code:text (# code/business_entity_resolution/requirements.txt), code:ini (# pytest.ini), code:powershell (uv pip install --python .venv\Scripts\python.exe duckdb indi), code:bash (git add code/business_entity_resolution pytest.ini) (+1 more)

### Community 48 - "Diagnostic Oracle"
Cohesion: 0.39
Nodes (6): oracle_macro, _oracle_scores, run_diagnostic, _country_of_source1, test_oracle_macro, test_oracle_macro_singleton_scores_one()

### Community 49 - "Blocking Design Decisions"
Cohesion: 0.32
Nodes (8): D1 Blocking plus Classifier, not End-to-End, D3 Ten Blocking Passes with Per-Pass Caps, D4 Do Not Reintroduce Per-Token Metaphone, Stage 1 Blocking / Candidate Generation, Blocking Passes with DuckDB (Task 6), F1 Per-Token Metaphone Blocking Exploded Temp, Cascade C+B Design, Kaggle-only Cascade (C+B) Implementation Plan

### Community 50 - "DuckDB Pipeline Decisions"
Cohesion: 0.25
Nodes (8): D2 DuckDB for Blocking Joins, Environment and Tooling Setup (2026-09-25), Locked Design Decisions, ber.cli Command Surface, Reproducible Local Staged Parquet Pipeline, F2 64-Bucket Join Loop Took >3 Hours, F3 Pandas Audit Could Not Handle 98.8M Rows, ber Core Module File Structure

### Community 51 - "Local Environment Setup"
Cohesion: 0.25
Nodes (8): code:powershell (uv venv .venv --python 3.12), code:python (import numpy as np), code:python (import os), code:python (import numpy as np), code:python (import sys), code:block8 (pandas), code:bash (git add .gitignore code/business_entity_resolution), Task 0: Local env, package scaffold, macro-F0.5 scorer

### Community 52 - "Diagnostic Runner"
Cohesion: 0.38
Nodes (5): oracle_macro(), _oracle_scores(), run_diagnostic(), test_oracle_macro(), test_oracle_macro_singleton_scores_one()

### Community 53 - "Char Features and Thresholds"
Cohesion: 0.33
Nodes (7): D10 Leave-One-Country-Out as the France Proxy, D14 Char n-gram TF-IDF Cosine Features, D15 Per-Country Thresholds + Singleton Calibration, M2 Char-3 TF-IDF + Per-Country/Singleton Calibration (v1.4.1), Stage 3 Training Pairs and Features, Task 2 Char n-gram TF-IDF Cosine Features, Task 3 Per-Country Thresholds + Singleton Calibration

### Community 54 - "Pipeline Implementation Plan"
Cohesion: 0.29
Nodes (6): Business Entity Resolution Pipeline — Implementation Plan, code:python (import numpy as np), code:bash (git add code/business_entity_resolution/src/ber/pairs.py cod), Global Constraints, Self-Review, Task 9: Training pair construction and split

### Community 55 - "Prepare Stage Task"
Cohesion: 0.29
Nodes (7): code:python (import pandas as pd), code:python (# code/business_entity_resolution/src/ber/io_utils.py), code:python (# code/business_entity_resolution/src/ber/prepare.py), code:python (# code/business_entity_resolution/src/ber/cli.py), code:powershell (.venv\Scripts\python.exe -m pytest code/business_entity_reso), code:bash (git add code/business_entity_resolution/src/ber/io_utils.py ), Task 5: Stage 0 prepare pipeline (parquet cache)

### Community 56 - "Predict and Booster Loading"
Cohesion: 0.60
Nodes (5): _feature_sources(), _load_booster(), _load_calibration(), predict_parts(), run_predict()

### Community 57 - "LightGBM Matcher Decisions"
Cohesion: 0.33
Nodes (6): D5 LightGBM as the Matcher, Submission Method: Blocking-plus-Classifier, Stage 4 LightGBM Model Training, E11 Calibration + Test Inference + Submission, E5 Matcher Improvements (LightGBM, Local CPU), F14 LightGBM Model Failed to Load After git checkout (CRLF)

### Community 58 - "Project File Structure"
Cohesion: 0.33
Nodes (6): code:block1 (code/business_entity_resolution/), code:block2 (artifacts/clean/{s1.parquet,s2.parquet,s3.parquet,labels.par), code:powershell ($env:BER_ARTIFACT_DIR="artifacts"), code:bash (git add code/business_entity_resolution kaggle), File Structure, Task 8: Train/predict stages + N4/N6 → submission v1 (classical)

### Community 59 - "Blocking Audit Task"
Cohesion: 0.40
Nodes (5): code:python (import pandas as pd), code:powershell (.venv\Scripts\python.exe -m pytest code/business_entity_reso), code:bash (git add code/business_entity_resolution/src/ber/audit.py cod), code:bash (git tag -a 0.2.0 -m "0.2.0: cleaning and recall-audited bloc), Task 7: Blocking audit and cap tuning

### Community 60 - "Submission Packaging"
Cohesion: 0.40
Nodes (4): Final submission package (not needed for the leaderboard), How to submit to the Portal, Leaderboard upload (what to submit), Submission

### Community 61 - "One-to-One Assignment"
Cohesion: 0.50
Nodes (4): D7 Injective One-to-One Post-Process, Stage 5 Threshold and Post-Processing, E7 Global Assignment, E8 Transductive Consistency / Pseudo-Labeling

### Community 62 - "Cleaning Module Task"
Cohesion: 0.50
Nodes (4): code:python (from ber.text import fold_ascii, normalize_address, normaliz), code:python (import re), code:bash (git add code/business_entity_resolution), Task 1: Cleaning module `text.py`

### Community 63 - "TSV IO Task"
Cohesion: 0.50
Nodes (4): code:python (import pandas as pd), code:python (import re), code:bash (git add code/business_entity_resolution), Task 2: TSV I/O module `io_tsv.py`

### Community 64 - "Blocking Module Task"
Cohesion: 0.50
Nodes (4): code:python (import pandas as pd), code:python (import numpy as np), code:bash (git add code/business_entity_resolution), Task 3: Blocking module `blocking.py`

### Community 65 - "Pair Features Task"
Cohesion: 0.50
Nodes (4): code:python (import numpy as np), code:python (CLASSICAL_FEATURES = [), code:bash (git add code/business_entity_resolution), Task 4: Pair features `features.py`

### Community 66 - "GBDT Wrapper Task"
Cohesion: 0.50
Nodes (4): code:python (import numpy as np), code:python (import numpy as np), code:bash (git add code/business_entity_resolution), Task 5: GBDT wrapper `gbdt.py`

### Community 67 - "Decision Logic Task"
Cohesion: 0.50
Nodes (4): code:python (import numpy as np), code:python (import pandas as pd), code:bash (git add code/business_entity_resolution), Task 6: Decision logic `decision.py`

### Community 68 - "Embedding Stage Task"
Cohesion: 0.50
Nodes (4): code:python (import numpy as np), code:powershell (.venv\Scripts\python.exe -m pytest code\business_entity_reso), code:bash (git add code/business_entity_resolution kaggle), Task 9: Embedding stage N3 + vector features → submission v2

### Community 69 - "Name Normalization Task"
Cohesion: 0.50
Nodes (4): code:python (import re), code:bash (git add code/business_entity_resolution/src/ber/normalize.py), code:python (from ber.normalize import detect_script, fold_name, name_tok), Task 2: Name normalization and script detection

### Community 70 - "Transliteration Task"
Cohesion: 0.50
Nodes (4): code:python (from ber.translit import romanize), code:python (import functools), code:bash (git add code/business_entity_resolution/src/ber/translit.py ), Task 3: Rule-based transliteration

### Community 71 - "Address Parsing Task"
Cohesion: 0.50
Nodes (4): code:python (from ber.address import parse_address), code:python (import re), code:bash (git add code/business_entity_resolution/src/ber/address.py c), Task 4: Country-aware address parsing

### Community 72 - "Production Run Task"
Cohesion: 0.50
Nodes (4): code:powershell ($env:PYTHONPATH="D:\Amazon project\code\business_entity_reso), code:powershell (cd DATA\student_resource), code:bash (git add -A), Task 13: Production run, outputs, and submission package

### Community 73 - "Compute Placement Decisions"
Cohesion: 0.67
Nodes (3): D12 Local CPU for Stages 0-1 and 5, Colab/Kaggle GPU for Transformer Stages, Colab T4 GPU Probe, F17 Colab files.upload() Truncated Files at ~100 MB

### Community 75 - "Kaggle Notebook Chain"
Cohesion: 0.67
Nodes (3): Kaggle Notebook Chain N1-N6 (Design), 24-Hour Execution Schedule, Kaggle Notebook Chain N1-N6

### Community 76 - "Kaggle Notebooks Task"
Cohesion: 0.67
Nodes (3): code:powershell ($env:BER_ARTIFACT_DIR="artifacts"), code:bash (git add code/business_entity_resolution kaggle), Task 7: Stages N1/N2 + Kaggle notebooks + push instructions (user can start running)

### Community 77 - "Cross-Encoder Task"
Cohesion: 0.67
Nodes (3): code:python (import numpy as np), code:bash (git add code/business_entity_resolution kaggle), Task 10: Cross-encoder stage N5 → submission v3

### Community 78 - "Packaging Task"
Cohesion: 0.67
Nodes (3): code:python (import os), code:bash (git add code/business_entity_resolution submission), Task 11: Submission packaging and documentation

### Community 79 - "Final Validation Task"
Cohesion: 0.67
Nodes (3): code:powershell (python utils/validate_submission.py --matching D:\Amazon Pro), code:bash (git add README.md RULES.md), Task 12: Final validation, versioning, push

### Community 80 - "DuckDB Blocking Task"
Cohesion: 0.67
Nodes (3): code:python (import pandas as pd), code:bash (git add code/business_entity_resolution/src/ber/blocking.py ), Task 6: Blocking passes with DuckDB

### Community 81 - "Pairwise Features Task"
Cohesion: 0.67
Nodes (3): code:python (import pandas as pd), code:bash (git add code/business_entity_resolution/src/ber/features.py ), Task 8: Pairwise features

### Community 82 - "Metric and Training Task"
Cohesion: 0.67
Nodes (3): code:python (from ber.evaluate import macro_f05), code:bash (git add code/business_entity_resolution/src/ber/evaluate.py ), Task 10: Metric, training, threshold tuning, and one-to-one

### Community 83 - "Test Prediction Task"
Cohesion: 0.67
Nodes (3): code:python (import pandas as pd), code:bash (git add code/business_entity_resolution/src/ber/predict.py c), Task 11: Test prediction and output writers

### Community 84 - "End-to-End Test Task"
Cohesion: 0.67
Nodes (3): code:python (import subprocess), code:bash (git add code/business_entity_resolution/tests code/business_), Task 12: End-to-end fixture and validator gate

## Knowledge Gaps
- **396 isolated node(s):** `Per-S1 Candidate Cap 200 + Per-Pass pass_caps`, `Full-Candidate Held-Out Validation (D9)`, `4:1 Negative Sampling (D6)`, `Challenge Documentation Template`, `test_landmark_and_missing` (+391 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Pair Construction` to `Prepare and Normalization`, `Blocking Audit`, `Macro F0.5 Scoring`, `Feature Engineering`, `Diagnostic Runner`, `Validation and Evaluation`, `Predict and Booster Loading`, `Blocking Keys`, `Calibration`, `Colab Export and Merge`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Why does `run_prepare()` connect `Prepare and Normalization` to `Pair Construction`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `run_prepare` connect `Prepare and Normalization` to `Feature Computation`, `Blocking and Candidates`, `CLI and Evaluation`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **What connects `Per-S1 Candidate Cap 200 + Per-Pass pass_caps`, `Full-Candidate Held-Out Validation (D9)`, `4:1 Negative Sampling (D6)` to the rest of the system?**
  _396 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Project Docs and Plans` be split into smaller, more focused modules?**
  _Cohesion score 0.060917343526039176 - nodes in this community are weakly interconnected._
- **Should `Prepare and Normalization` be split into smaller, more focused modules?**
  _Cohesion score 0.07138047138047138 - nodes in this community are weakly interconnected._
- **Should `Blocking Audit` be split into smaller, more focused modules?**
  _Cohesion score 0.07346938775510205 - nodes in this community are weakly interconnected._