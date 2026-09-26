# Graph Report - Amazon project  (2026-09-27)

## Corpus Check
- 62 files · ~48,897 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 931 nodes · 1465 edges · 67 communities (66 shown, 1 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 146 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- config.json
- config.json
- parse_address()
- run_audit()
- blocking.py
- calibration.py
- main()
- features.py
- validation.py
- features.py
- grouped_split()
- prepare.py
- romanize()
- validate()
- bench_gbdt.py
- eda_dataset.py
- AGENTS.md — working notes for this repository
- ML Challenge 2026: Business Entity Resolution Solution Template
- Business Entity Resolution Challenge
- Amazon ML Challenge 2026 — Business Entity Resolution
- RULES.md — Amazon ML Challenge 2026: Business Entity Resolution
- Business Entity Resolution — Code and Run Instructions
- ML Challenge 2026: Business Entity Resolution Solution Template
- Business Entity Resolution Challenge
- Dataset Description — Business Entity Resolution Challenge
- Decisions and Reasoning
- Failures and Fixes
- Project Log — Business Entity Resolution
- Results
- Business Entity Resolution Pipeline Implementation Plan
- File Structure
- Task 0: Local env, package scaffold, macro-F0.5 scorer
- Task 1: Cleaning module `text.py`
- Task 2: TSV I/O module `io_tsv.py`
- Task 3: Blocking module `blocking.py`
- Task 4: Pair features `features.py`
- Task 5: GBDT wrapper `gbdt.py`
- Task 6: Decision logic `decision.py`
- Task 7: Stages N1/N2 + Kaggle notebooks + push instructions (user can start running)
- Task 9: Embedding stage N3 + vector features → submission v2
- Task 10: Cross-encoder stage N5 → submission v3
- Task 11: Submission packaging and documentation
- Task 12: Final validation, versioning, push
- Global Constraints
- Task 1: Scaffold, config, and dependency install
- Task 2: Name normalization and script detection
- Task 3: Rule-based transliteration
- Task 4: Country-aware address parsing
- Task 5: Stage 0 prepare pipeline (parquet cache)
- Task 6: Blocking passes with DuckDB
- Task 7: Blocking audit and cap tuning
- Task 8: Pairwise features
- Task 10: Metric, training, threshold tuning, and one-to-one
- Task 11: Test prediction and output writers
- Task 12: End-to-end fixture and validator gate
- Task 13: Production run, outputs, and submission package
- Global constraints
- Design Spec — Business Entity Resolution Pipeline
- Design — Kaggle-only cascade (C+B) for Business Entity Resolution
- Design Spec — Precision Lift + Colab Embeddings (Milestone M2)
- Submission
- validation.py
- validate()
- blocking.py
- Decisions and Reasoning (DECISIONS.md)

## God Nodes (most connected - your core abstractions)
1. `Failures and Fixes` - 19 edges
2. `main()` - 18 edges
3. `Decisions and Reasoning (DECISIONS.md)` - 18 edges
4. `M2 Precision + Colab Plan (10 Tasks)` - 18 edges
5. `Business Entity Resolution Challenge` - 17 edges
6. `Decisions and Reasoning` - 17 edges
7. `File Structure` - 16 edges
8. `main()` - 15 edges
9. `Results` - 14 edges
10. `Global Constraints` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Results Metrics` --references--> `Milestone Version Tags (1.4.0-2.0.0)`  [INFERRED]
  docs/RESULTS.md → RULES.md
- `main()` --calls--> `write_inference_pairs()`  [INFERRED]
  code/business_entity_resolution/src/ber/cli.py → code/business_entity_resolution/src/ber/pairs.py
- `main()` --calls--> `write_training_pairs()`  [INFERRED]
  code/business_entity_resolution/src/ber/cli.py → code/business_entity_resolution/src/ber/pairs.py
- `Pinned ML Stack (pandas 3.0.6, lightgbm 4.7.0, duckdb 1.5.5)` --conceptually_related_to--> `LightGBM Matcher (D5)`  [INFERRED]
  code/business_entity_resolution/requirements.txt → docs/DECISIONS.md
- `Blocking + Classifier Approach (D1)` --shares_data_with--> `candidate_pairs.tsv Contract`  [INFERRED]
  docs/DECISIONS.md → PROBLEM_STATEMENT.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Blocking to Recall Ceiling Flow** — ten_blocking_passes, blocking_cap, train_recall_08115, blocking_recall_ceiling_0814, candidate_pairs_contract [EXTRACTED 0.80]
- **M2 Precision + Colab Flow** — char_ngram_features_d14, per_country_thresholds_d15, singleton_calibration, per_pair_cosine_interface_d13, adoption_gates [EXTRACTED 0.80]
- **Submission Output and Validation Flow** — matching_results_contract, candidate_pairs_contract, submission_validator, submission_tree, official_leaderboard_0811 [EXTRACTED 0.85]

## Communities (67 total, 1 thin omitted)

### Community 14 - "config.json"
Cohesion: 0.07
Nodes (27): dataset_dir, data_dir, models_dir, output_dir, seed, cap, idf_min, max_block (+19 more)

### Community 15 - "config.json"
Cohesion: 0.07
Nodes (27): dataset_dir, data_dir, models_dir, output_dir, seed, cap, idf_min, max_block (+19 more)

### Community 16 - "parse_address()"
Cohesion: 0.14
Nodes (23): AddressParts, _postal_re(), str, Pattern, parse_address(), _clean_chars(), str, normalize_name() (+15 more)

### Community 4 - "run_audit()"
Cohesion: 0.09
Nodes (31): load_country_map(), _explode_truth(), DataFrame, audit_candidates(), int, _Progress, _sql_path(), str (+23 more)

### Community 25 - "blocking.py"
Cohesion: 0.27
Nodes (13): compute_token_idf(), load_token_idf(), normalize_keys(), block_keys(), _pass_caps(), _valid_sql(), _join_sql(), generate_candidates() (+5 more)

### Community 7 - "calibration.py"
Cohesion: 0.11
Nodes (32): tune_per_country(), singleton_decision(), tune_singleton_tau(), apply_calibration(), save_calibration(), load_calibration(), _configure(), _pred_glob() (+24 more)

### Community 20 - "main()"
Cohesion: 0.20
Nodes (14): _default_config(), main(), _configure(), export_entities(), export_pairs(), export_for_colab(), import_cosine(), _feature_targets() (+6 more)

### Community 3 - "features.py"
Cohesion: 0.10
Nodes (25): Config, cfg(), _prepared(), _make_dataset(), Path, _config(), test_parallel_features_match_single_process(), attach_country() (+17 more)

### Community 12 - "validation.py"
Cohesion: 0.11
Nodes (30): oracle_macro(), _oracle_scores(), run_diagnostic(), _load_booster(), _load_calibration(), _feature_sources(), predict_parts(), _write_tsv() (+22 more)

### Community 17 - "features.py"
Cohesion: 0.15
Nodes (23): fit_char_vectorizer(), save_char_vectorizer(), load_char_vectorizer(), char3_cosine(), _train_text_sample(), ensure_char_vectorizer(), _country_map(), attach_country() (+15 more)

### Community 5 - "grouped_split()"
Cohesion: 0.07
Nodes (36): _explode_truth(), DataFrame, build_training_pairs(), grouped_split(), float, int, write_training_pairs(), write_inference_pairs() (+28 more)

### Community 1 - "prepare.py"
Cohesion: 0.09
Nodes (39): prepare_frame(), DataFrame, _write_stream(), Path, int, run_prepare(), test_prepare_frame_columns_and_values(), AddressParts (+31 more)

### Community 33 - "romanize()"
Cohesion: 0.35
Nodes (9): _scheme_for(), str, _token_romanize(), _romanize_cached(), romanize(), test_romanize_identity_for_latin(), test_romanize_indic_is_latin(), test_romanize_mixed_keeps_latin() (+1 more)

### Community 31 - "validate()"
Cohesion: 0.27
Nodes (11): read_ids(), examples(), load_match_targets(), validate_id_list_file(), validate(), main(), Return the set of first-column entity IDs from a source TSV.      The header row, Return a short, human-readable sample of ``items`` for an error message. (+3 more)

### Community 8 - "bench_gbdt.py"
Cohesion: 0.38
Nodes (12): norm_series(), norm_one(), load_s1_sample(), load_positives(), stream_negatives_and_texts(), build_pairs(), token_set(), compute_features() (+4 more)

### Community 9 - "eda_dataset.py"
Cohesion: 0.53
Nodes (7): hist(), pct(), scan_source(), scan_ground_truth(), check_membership(), cached(), main()

### Community 37 - "AGENTS.md — working notes for this repository"
Cohesion: 0.22
Nodes (8): AGENTS.md — working notes for this repository, Environment, Pipeline data contracts (all on disk, parquet/JSON), Blocking contracts, Audit contract, Matcher + calibration contract, Colab GPU stages, Gotchas already learned

### Community 27 - "ML Challenge 2026: Business Entity Resolution Solution Template"
Cohesion: 0.15
Nodes (12): ML Challenge 2026: Business Entity Resolution Solution Template, 1. Executive Summary, 2. Methodology, 2.1 Problem Analysis, 2.2 Solution Strategy, 3. Candidate Generation (Blocking), 4. Matching Model, 5. Results & Error Analysis (+4 more)

### Community 13 - "Business Entity Resolution Challenge"
Cohesion: 0.07
Nodes (27): Business Entity Resolution Challenge, Table of Contents, 1. Overview, 2. File Format, code:python (import pandas as pd), 3. Data Description, 4. Noise Patterns to Expect, 5. Dataset Details (+19 more)

### Community 30 - "Amazon ML Challenge 2026 — Business Entity Resolution"
Cohesion: 0.17
Nodes (11): Amazon ML Challenge 2026 — Business Entity Resolution, Documentation, Repository layout, code:block1 (PROBLEM_STATEMENT.md          # full transcription of the of), Quickstart, code:powershell (# 1. Python 3.12 environment (uv)), Key dataset facts, Matcher benchmark (0.1.0) (+3 more)

### Community 35 - "RULES.md — Amazon ML Challenge 2026: Business Entity Resolution"
Cohesion: 0.20
Nodes (9): RULES.md — Amazon ML Challenge 2026: Business Entity Resolution, 1. Challenge rules (from the official problem statement), 2. Data rules, 3. Engineering rules, code:powershell (uv venv .venv --python 3.12), 4. Verification commands, code:powershell (# 1) Dataset EDA (writes tools/eda_stats.json, caches in too), 5. Versioning and git rules (+1 more)

### Community 34 - "Business Entity Resolution — Code and Run Instructions"
Cohesion: 0.18
Nodes (10): Business Entity Resolution — Code and Run Instructions, Environment, code:block1 (uv venv .venv --python 3.12), code:powershell ($env:PYTHONPATH="code/business_entity_resolution/src"), Run order, code:powershell (# 1. Clean + normalize all six record files (cached to DATA/), Package layout, code:block4 (src/ber/) (+2 more)

### Community 29 - "ML Challenge 2026: Business Entity Resolution Solution Template"
Cohesion: 0.15
Nodes (12): ML Challenge 2026: Business Entity Resolution Solution Template, 1. Executive Summary, 2. Methodology, 2.1 Problem Analysis, 2.2 Solution Strategy, 3. Candidate Generation (Blocking), 4. Matching Model, 5. Results & Error Analysis (+4 more)

### Community 18 - "Business Entity Resolution Challenge"
Cohesion: 0.09
Nodes (22): ML Challenge 2026 Problem Statement, Business Entity Resolution Challenge, File Format, code:python (import pandas as pd), Data Description:, Dataset Details:, File Descriptions:, Output Format: (+14 more)

### Community 28 - "Dataset Description — Business Entity Resolution Challenge"
Cohesion: 0.15
Nodes (12): Dataset Description — Business Entity Resolution Challenge, 1. TL;DR, 2. File inventory, 3. Schema, Record files (`*_source{1,2,3}.tsv`), Ground truth (`train/train_ground_truth.tsv`), 4. Country distribution (measured), 5. Ground-truth analysis (measured) (+4 more)

### Community 21 - "Decisions and Reasoning"
Cohesion: 0.11
Nodes (17): Decisions and Reasoning, D1 — Blocking plus classifier, not end-to-end, D2 — DuckDB for blocking joins, pandas/pyarrow for the rest, D3 — Ten blocking passes with per-pass block caps, D4 — Do not reintroduce per-token Metaphone, D5 — LightGBM as the matcher, D6 — 4:1 negative sampling for training, D7 — Injective one-to-one post-process (+9 more)

### Community 19 - "Failures and Fixes"
Cohesion: 0.10
Nodes (19): Failures and Fixes, F1 — Per-token Metaphone blocking exploded the temp directory, F2 — 64-bucket join loop took >3 hours, F3 — Pandas audit could not handle 98.8M candidate rows, F4 — Parquet list columns came back as numpy arrays, F5 — Feature pipeline needed a different metadata split, F6 — Prediction TSV aggregation OOM, F7 — Empty match lists written as `""`, validator FAIL (+11 more)

### Community 26 - "Project Log — Business Entity Resolution"
Cohesion: 0.14
Nodes (13): Project Log — Business Entity Resolution, 2026-09-25 / 26 — environment and tooling, Pipeline runs (measured), Final submission, Corrections made during the run, 2026-09-26 — Colab GPU probe and M2 plan, 2026-09-26 — M2 Task 1: oracle diagnostic, 2026-09-26 — M2 Task 2: char n-gram TF-IDF cosine features (+5 more)

### Community 23 - "Results"
Cohesion: 0.12
Nodes (15): Results, Official leaderboard result, Headline, Full-candidate held-out details (`DATA/reports/eval_full_candidates.json`), M2 (v1.4.1) — char-3 features + per-country/singleton calibration, M2 (Tasks 6–7) — multilingual embeddings: tried, rejected, 4:1 sampled details (`DATA/reports/eval_marks.json`), Leave-one-country-out (`DATA/reports/eval_loo.json`) (+7 more)

### Community 42 - "File Structure"
Cohesion: 0.33
Nodes (6): File Structure, code:block1 (code/business_entity_resolution/), code:block2 (artifacts/clean/{s1.parquet,s2.parquet,s3.parquet,labels.par), Task 8: Train/predict stages + N4/N6 → submission v1 (classical), code:powershell ($env:BER_ARTIFACT_DIR="artifacts"), code:bash (git add code/business_entity_resolution kaggle)

### Community 39 - "Task 0: Local env, package scaffold, macro-F0.5 scorer"
Cohesion: 0.25
Nodes (8): Task 0: Local env, package scaffold, macro-F0.5 scorer, code:powershell (uv venv .venv --python 3.12), code:python (import numpy as np), code:python (import os), code:python (import numpy as np), code:python (import sys), code:block8 (pandas), code:bash (git add .gitignore code/business_entity_resolution)

### Community 45 - "Task 1: Cleaning module `text.py`"
Cohesion: 0.50
Nodes (4): Task 1: Cleaning module `text.py`, code:python (from ber.text import fold_ascii, normalize_address, normaliz), code:python (import re), code:bash (git add code/business_entity_resolution)

### Community 46 - "Task 2: TSV I/O module `io_tsv.py`"
Cohesion: 0.50
Nodes (4): Task 2: TSV I/O module `io_tsv.py`, code:python (import pandas as pd), code:python (import re), code:bash (git add code/business_entity_resolution)

### Community 47 - "Task 3: Blocking module `blocking.py`"
Cohesion: 0.50
Nodes (4): Task 3: Blocking module `blocking.py`, code:python (import pandas as pd), code:python (import numpy as np), code:bash (git add code/business_entity_resolution)

### Community 48 - "Task 4: Pair features `features.py`"
Cohesion: 0.50
Nodes (4): Task 4: Pair features `features.py`, code:python (import numpy as np), code:python (CLASSICAL_FEATURES = [), code:bash (git add code/business_entity_resolution)

### Community 49 - "Task 5: GBDT wrapper `gbdt.py`"
Cohesion: 0.50
Nodes (4): Task 5: GBDT wrapper `gbdt.py`, code:python (import numpy as np), code:python (import numpy as np), code:bash (git add code/business_entity_resolution)

### Community 50 - "Task 6: Decision logic `decision.py`"
Cohesion: 0.50
Nodes (4): Task 6: Decision logic `decision.py`, code:python (import numpy as np), code:python (import pandas as pd), code:bash (git add code/business_entity_resolution)

### Community 57 - "Task 7: Stages N1/N2 + Kaggle notebooks + push instructions (user can start running)"
Cohesion: 0.67
Nodes (3): Task 7: Stages N1/N2 + Kaggle notebooks + push instructions (user can start running), code:powershell ($env:BER_ARTIFACT_DIR="artifacts"), code:bash (git add code/business_entity_resolution kaggle)

### Community 51 - "Task 9: Embedding stage N3 + vector features → submission v2"
Cohesion: 0.50
Nodes (4): Task 9: Embedding stage N3 + vector features → submission v2, code:python (import numpy as np), code:powershell (.venv\Scripts\python.exe -m pytest code\business_entity_reso), code:bash (git add code/business_entity_resolution kaggle)

### Community 58 - "Task 10: Cross-encoder stage N5 → submission v3"
Cohesion: 0.67
Nodes (3): Task 10: Cross-encoder stage N5 → submission v3, code:python (import numpy as np), code:bash (git add code/business_entity_resolution kaggle)

### Community 59 - "Task 11: Submission packaging and documentation"
Cohesion: 0.67
Nodes (3): Task 11: Submission packaging and documentation, code:python (import os), code:bash (git add code/business_entity_resolution submission)

### Community 60 - "Task 12: Final validation, versioning, push"
Cohesion: 0.67
Nodes (3): Task 12: Final validation, versioning, push, code:powershell (python utils/validate_submission.py --matching D:\Amazon Pro), code:bash (git add README.md RULES.md)

### Community 40 - "Global Constraints"
Cohesion: 0.29
Nodes (6): Business Entity Resolution Pipeline — Implementation Plan, Global Constraints, Task 9: Training pair construction and split, code:python (import numpy as np), code:bash (git add code/business_entity_resolution/src/ber/pairs.py cod), Self-Review

### Community 38 - "Task 1: Scaffold, config, and dependency install"
Cohesion: 0.22
Nodes (9): Task 1: Scaffold, config, and dependency install, code:python (# code/business_entity_resolution/tests/conftest.py), code:python (# code/business_entity_resolution/src/ber/__init__.py), code:python (# code/business_entity_resolution/src/ber/config.py), code:json (// code/business_entity_resolution/config.json), code:text (# code/business_entity_resolution/requirements.txt), code:ini (# pytest.ini), code:powershell (uv pip install --python .venv\Scripts\python.exe duckdb indi) (+1 more)

### Community 52 - "Task 2: Name normalization and script detection"
Cohesion: 0.50
Nodes (4): Task 2: Name normalization and script detection, code:python (from ber.normalize import detect_script, fold_name, name_tok), code:python (import re), code:bash (git add code/business_entity_resolution/src/ber/normalize.py)

### Community 53 - "Task 3: Rule-based transliteration"
Cohesion: 0.50
Nodes (4): Task 3: Rule-based transliteration, code:python (from ber.translit import romanize), code:python (import functools), code:bash (git add code/business_entity_resolution/src/ber/translit.py )

### Community 54 - "Task 4: Country-aware address parsing"
Cohesion: 0.50
Nodes (4): Task 4: Country-aware address parsing, code:python (from ber.address import parse_address), code:python (import re), code:bash (git add code/business_entity_resolution/src/ber/address.py c)

### Community 41 - "Task 5: Stage 0 prepare pipeline (parquet cache)"
Cohesion: 0.29
Nodes (7): Task 5: Stage 0 prepare pipeline (parquet cache), code:python (import pandas as pd), code:python (# code/business_entity_resolution/src/ber/io_utils.py), code:python (# code/business_entity_resolution/src/ber/prepare.py), code:python (# code/business_entity_resolution/src/ber/cli.py), code:powershell (.venv\Scripts\python.exe -m pytest code/business_entity_reso), code:bash (git add code/business_entity_resolution/src/ber/io_utils.py )

### Community 61 - "Task 6: Blocking passes with DuckDB"
Cohesion: 0.67
Nodes (3): Task 6: Blocking passes with DuckDB, code:python (import pandas as pd), code:bash (git add code/business_entity_resolution/src/ber/blocking.py )

### Community 43 - "Task 7: Blocking audit and cap tuning"
Cohesion: 0.40
Nodes (5): Task 7: Blocking audit and cap tuning, code:python (import pandas as pd), code:powershell (.venv\Scripts\python.exe -m pytest code/business_entity_reso), code:bash (git add code/business_entity_resolution/src/ber/audit.py cod), code:bash (git tag -a 0.2.0 -m "0.2.0: cleaning and recall-audited bloc)

### Community 62 - "Task 8: Pairwise features"
Cohesion: 0.67
Nodes (3): Task 8: Pairwise features, code:python (import pandas as pd), code:bash (git add code/business_entity_resolution/src/ber/features.py )

### Community 63 - "Task 10: Metric, training, threshold tuning, and one-to-one"
Cohesion: 0.67
Nodes (3): Task 10: Metric, training, threshold tuning, and one-to-one, code:python (from ber.evaluate import macro_f05), code:bash (git add code/business_entity_resolution/src/ber/evaluate.py )

### Community 64 - "Task 11: Test prediction and output writers"
Cohesion: 0.67
Nodes (3): Task 11: Test prediction and output writers, code:python (import pandas as pd), code:bash (git add code/business_entity_resolution/src/ber/predict.py c)

### Community 65 - "Task 12: End-to-end fixture and validator gate"
Cohesion: 0.67
Nodes (3): Task 12: End-to-end fixture and validator gate, code:python (import subprocess), code:bash (git add code/business_entity_resolution/tests code/business_)

### Community 55 - "Task 13: Production run, outputs, and submission package"
Cohesion: 0.50
Nodes (4): Task 13: Production run, outputs, and submission package, code:powershell ($env:PYTHONPATH="D:\Amazon project\code\business_entity_reso), code:powershell (cd DATA\student_resource), code:bash (git add -A)

### Community 22 - "Global constraints"
Cohesion: 0.12
Nodes (16): Precision Lift + Colab Embeddings — Implementation Plan (M2), Global constraints, Task 1: Diagnostic — oracle ceiling and per-country headroom, code:python (def test_oracle_macro():), Task 2: Char n-gram TF-IDF cosine features, Task 3: Per-country thresholds + singleton calibration, Task 4: Retrain + held-out gate (milestone 1.4.0), Task 5: Colab export/import (local) (+8 more)

### Community 24 - "Design Spec — Business Entity Resolution Pipeline"
Cohesion: 0.12
Nodes (15): Design Spec — Business Entity Resolution Pipeline, 1. Context and goal, 2. Locked decisions, 3. Architecture and data flow, code:block1 (dataset/*.tsv), 4. Stage 0 — Cleaning and normalization, 5. Stage 1 — Blocking / candidate generation, 6. Stage 3 — Training pairs and features (+7 more)

### Community 32 - "Design — Kaggle-only cascade (C+B) for Business Entity Resolution"
Cohesion: 0.17
Nodes (11): Design — Kaggle-only cascade (C+B) for Business Entity Resolution, 1. Context and constraints, 2. Approach — cascade C+B, code:block1 (S0 Kaggle private dataset `amz-er-2026-raw` (7 TSVs from D:\), 3. Kaggle execution model, 4. Stage details, 5. 24-hour schedule, 6. Verification and gates (+3 more)

### Community 36 - "Design Spec — Precision Lift + Colab Embeddings (Milestone M2)"
Cohesion: 0.20
Nodes (9): Design Spec — Precision Lift + Colab Embeddings (Milestone M2), 1. Goal, 2. Why this plan (bottleneck analysis), 3. Environments, 4. Interfaces (small transfers), 5. Phases and adoption gates, 6. Validation protocol (reused, unchanged), 7. Risks (+1 more)

### Community 44 - "Submission"
Cohesion: 0.40
Nodes (4): Submission, Leaderboard upload (what to submit), Final submission package (not needed for the leaderboard), How to submit to the Portal

### Community 2 - "validation.py"
Cohesion: 0.13
Nodes (27): Config, _feature_sources(), _load_booster(), predict_parts(), run_predict(), _write_tsv(), _bin_counts(), build_valfull_pairs() (+19 more)

### Community 10 - "validate()"
Cohesion: 0.62
Nodes (6): examples(), load_match_targets(), main(), read_ids(), validate(), validate_id_list_file()

### Community 6 - "blocking.py"
Cohesion: 0.28
Nodes (13): block_keys(), compute_token_idf(), generate_candidates(), _join_sql(), load_token_idf(), normalize_keys(), _pass_caps(), run_block() (+5 more)

### Community 0 - "Decisions and Reasoning (DECISIONS.md)"
Cohesion: 0.09
Nodes (64): Pinned Requirements, Pinned ML Stack (pandas 3.0.6, lightgbm 4.7.0, duckdb 1.5.5), M2 Adoption Gates (numeric comparisons vs 0.8488), AGENTS.md Working Notes, Per-S1 Candidate Cap 200 + Per-Pass pass_caps, Blocking + Classifier Approach (D1), Blocking Recall Ceiling 0.814, candidate_pairs.tsv Contract (+56 more)

## Knowledge Gaps
- **332 isolated node(s):** `dataset_dir`, `data_dir`, `models_dir`, `output_dir`, `seed` (+327 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `main()` to `prepare.py`, `run_audit()`, `grouped_split()`, `calibration.py`, `validation.py`, `features.py`, `blocking.py`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Why does `run_prepare()` connect `prepare.py` to `main()`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Why does `prepare_frame()` connect `prepare.py` to `parse_address()`, `features.py`, `features.py`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `main()` (e.g. with `run_audit()` and `run_block()`) actually correct?**
  _`main()` has 16 INFERRED edges - model-reasoned connections that need verification._
- **What connects `dataset_dir`, `data_dir`, `models_dir` to the rest of the system?**
  _332 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `config.json` be split into smaller, more focused modules?**
  _Cohesion score 0.07142857142857142 - nodes in this community are weakly interconnected._
- **Should `config.json` be split into smaller, more focused modules?**
  _Cohesion score 0.07142857142857142 - nodes in this community are weakly interconnected._