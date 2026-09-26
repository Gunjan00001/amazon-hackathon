# Graph Report - Amazon project  (2026-09-26)

## Corpus Check
- Corpus is ~39,382 words - fits in a single context window. You may not need a graph.

## Summary
- 240 nodes · 503 edges · 14 communities
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Docs, Metrics & Leaderboard
- Normalization, Address & CLI
- Prediction & Validation
- Blocking Audit
- Scoring & Thresholding
- Feature Engineering
- Blocking Engine
- Config & Test Fixtures
- GBDT Benchmark Tool
- Transliteration
- Pair Construction
- Dataset EDA Tool
- Submission Validator

## God Nodes (most connected - your core abstractions)
1. `main()` - 15 edges
2. `prepare_frame()` - 13 edges
3. `normalize_name()` - 11 edges
4. `evaluate_loo()` - 11 edges
5. `parse_address()` - 10 edges
6. `run_features()` - 10 edges
7. `grouped_split()` - 10 edges
8. `evaluate_full_candidates()` - 10 edges
9. `run_audit()` - 9 edges
10. `romanize()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Filled-in Methodology Write-up` --semantically_similar_to--> `Student Documentation Template`  [EXTRACTED] [semantically similar]
  Documentation_template.md → DATA/student_resource/Documentation_template.md
- `Problem Statement` --references--> `Official Challenge README`  [INFERRED]
  PROBLEM_STATEMENT.md → DATA/student_resource/README.md
- `Official Challenge README` --references--> `validate_submission.py Gate`  [EXTRACTED]
  DATA/student_resource/README.md → PROBLEM_STATEMENT.md
- `RULES.md Binding Project Rules` --references--> `validate_submission.py Gate`  [EXTRACTED]
  RULES.md → PROBLEM_STATEMENT.md
- `Version Tags 0.1.0-1.2.0` --references--> `GitHub Release 1.1.0 Assets`  [INFERRED]
  RULES.md → submission/README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Blocking + Classifier Pipeline Flow** — blocking_classifier_pipeline, duckdb_blocking, ten_blocking_passes, lightgbm_matcher, one_to_one_assignment, threshold_0925 [EXTRACTED 1.00]
- **Kaggle Cascade C+B Flow** — kaggle_cascade_cb, multilingual_embeddings, cross_encoder_rerank, lightgbm_matcher [EXTRACTED 1.00]
- **Submission Artifacts and Scoring** — matching_results_tsv, candidate_pairs_tsv, validate_submission, official_leaderboard_0811 [EXTRACTED 1.00]

## Communities (14 total, 0 thin omitted)

### Community 0 - "Docs, Metrics & Leaderboard"
Cohesion: 0.10
Nodes (46): AGENTS.md Working Notes, Business Entity Resolution Code README, Blocking + Classifier Pipeline, Business Entity Resolution Task, candidate_pairs.tsv Blocking Set, Candidate Recall Ceiling 0.814, Pinned Requirements, Cross-encoder Rerank Stage (+38 more)

### Community 1 - "Normalization, Address & CLI"
Cohesion: 0.11
Nodes (31): AddressParts, parse_address(), _postal_re(), _default_config(), main(), _clean_chars(), detect_script(), fold_name() (+23 more)

### Community 2 - "Prediction & Validation"
Cohesion: 0.21
Nodes (20): grouped_split(), _feature_sources(), _load_booster(), predict_parts(), run_predict(), _write_tsv(), _bin_counts(), build_valfull_pairs() (+12 more)

### Community 3 - "Blocking Audit"
Cohesion: 0.18
Nodes (15): audit_candidates(), _connect(), _explode_truth(), load_country_map(), _Progress, DataFrame, run_audit(), _sql_path() (+7 more)

### Community 4 - "Scoring & Thresholding"
Cohesion: 0.16
Nodes (15): evaluate_marks(), macro_f05(), _one_to_one_mask(), one_to_one(), DataFrame, entity_f05(), _entity_f05_from_codes(), sweep_threshold() (+7 more)

### Community 5 - "Feature Engineering"
Cohesion: 0.22
Nodes (14): attach_country(), _combine_feature_parts(), compute_features(), _country_map(), _feature_block(), _pair_set_metrics(), _pairs_has_label(), _phase1_merged() (+6 more)

### Community 6 - "Blocking Engine"
Cohesion: 0.28
Nodes (13): block_keys(), compute_token_idf(), generate_candidates(), _join_sql(), load_token_idf(), normalize_keys(), _pass_caps(), run_block() (+5 more)

### Community 7 - "Config & Test Fixtures"
Cohesion: 0.23
Nodes (9): Config, cfg(), test_config_load_roundtrip(), _config(), _make_dataset(), _prepared(), Path, test_parallel_features_match_single_process() (+1 more)

### Community 8 - "GBDT Benchmark Tool"
Cohesion: 0.35
Nodes (11): best_entity_f05(), build_pairs(), compute_features(), entity_macro_f05(), load_positives(), load_s1_sample(), main(), norm_one() (+3 more)

### Community 9 - "Transliteration"
Cohesion: 0.36
Nodes (8): romanize(), _romanize_cached(), _scheme_for(), _token_romanize(), test_romanize_empty(), test_romanize_identity_for_latin(), test_romanize_indic_is_latin(), test_romanize_mixed_keeps_latin()

### Community 10 - "Pair Construction"
Cohesion: 0.36
Nodes (7): build_training_pairs(), _explode_truth(), DataFrame, _cands(), _gt(), test_build_training_pairs_labels_and_ratio(), test_grouped_split_has_no_group_leakage()

### Community 11 - "Dataset EDA Tool"
Cohesion: 0.43
Nodes (6): cached(), check_membership(), main(), pct(), scan_ground_truth(), scan_source()

### Community 12 - "Submission Validator"
Cohesion: 0.62
Nodes (6): examples(), load_match_targets(), main(), read_ids(), validate(), validate_id_list_file()

## Knowledge Gaps
- **3 isolated node(s):** `Student Documentation Template`, `Business Entity Resolution Task`, `33 Pairwise Features`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_audit()` connect `Blocking Audit` to `Normalization, Address & CLI`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Why does `evaluate_marks()` connect `Scoring & Thresholding` to `Normalization, Address & CLI`, `Prediction & Validation`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Why does `main()` connect `Normalization, Address & CLI` to `Prediction & Validation`, `Blocking Audit`, `Scoring & Thresholding`, `Feature Engineering`, `Blocking Engine`, `Config & Test Fixtures`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `prepare_frame()` (e.g. with `detect_script()` and `fold_name()`) actually correct?**
  _`prepare_frame()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Student Documentation Template`, `Business Entity Resolution Task`, `33 Pairwise Features` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Docs, Metrics & Leaderboard` be split into smaller, more focused modules?**
  _Cohesion score 0.0966183574879227 - nodes in this community are weakly interconnected._
- **Should `Normalization, Address & CLI` be split into smaller, more focused modules?**
  _Cohesion score 0.11095305832147938 - nodes in this community are weakly interconnected._