# Graph Report - Amazon project  (2026-09-26)

## Corpus Check
- Corpus is ~38,780 words - fits in a single context window. You may not need a graph.

## Summary
- 265 nodes · 563 edges · 19 communities (12 shown, 7 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 23 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Docs, Decisions & Metrics
- Normalization & Address Parsing
- Pairs & Prediction
- Feature Engineering
- Blocking Audit
- Scoring & Thresholding
- Blocking Engine
- CLI, Config & Tests
- GBDT Benchmark Tool
- Dataset EDA Tool
- Submission Validator
- Failure F10 Suffix Class
- Failure F11 Grouped Split
- Failure F2 Bucket Join
- Failure F4 Parquet Arrays
- Failure F5 Meta Split
- Failure F8 DATA Collision
- Failure F9 Indic Marks

## God Nodes (most connected - your core abstractions)
1. `Pipeline Code README` - 17 edges
2. `Architecture Decisions D1-D11` - 16 edges
3. `main()` - 15 edges
4. `prepare_frame()` - 13 edges
5. `AGENTS.md Working Notes` - 12 edges
6. `normalize_name()` - 11 edges
7. `evaluate_loo()` - 11 edges
8. `Project Run Log` - 11 edges
9. `parse_address()` - 10 edges
10. `run_features()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `_config()` --uses--> `Config`  [INFERRED]
  code/business_entity_resolution/tests/test_parallel_features.py → code/business_entity_resolution/src/ber/config.py
- `F3 Pandas audit could not handle 98.8M rows` --conceptually_related_to--> `Pipeline Stage: audit`  [INFERRED]
  D:/Amazon project/docs/FAILURES_AND_FIXES.md → D:/Amazon project/code/business_entity_resolution/README.md
- `F6 Prediction TSV aggregation OOM` --conceptually_related_to--> `Pipeline Stage: predict`  [INFERRED]
  D:/Amazon project/docs/FAILURES_AND_FIXES.md → D:/Amazon project/code/business_entity_resolution/README.md
- `F7 Empty match lists written as quoted empty string` --conceptually_related_to--> `output/matching_results.tsv`  [INFERRED]
  D:/Amazon project/docs/FAILURES_AND_FIXES.md → D:/Amazon project/PROBLEM_STATEMENT.md
- `test_landmark_and_missing()` --calls--> `parse_address()`  [EXTRACTED]
  code/business_entity_resolution/tests/test_address.py → code/business_entity_resolution/src/ber/address.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Entity Resolution Pipeline Flow** — stage_prepare, stage_block, stage_audit, stage_features, stage_train, stage_predict [EXTRACTED 1.00]
- **Validation and France-proxy Evaluation Protocol** — stage_validation, stage_loo, metric_f05_08488, metric_loo_0668_0804, metric_threshold_0925, metric_recall_ceiling_0814 [EXTRACTED 1.00]
- **Submission Package Artifact Set** — output_matching_results, output_candidate_pairs, submission_package, student_doc_template, requirements_txt [EXTRACTED 1.00]

## Communities (19 total, 7 thin omitted)

### Community 0 - "Docs, Decisions & Metrics"
Cohesion: 0.08
Nodes (64): AGENTS.md Working Notes, Ten Blocking Passes, Candidates Parquet Contract, ber.cli Command Line, Pipeline Code README, Measured Dataset Facts (DATASET.md), D1 Blocking plus classifier, not end-to-end, D10 Leave-one-country-out as France proxy (+56 more)

### Community 1 - "Normalization & Address Parsing"
Cohesion: 0.10
Nodes (35): AddressParts, parse_address(), _postal_re(), _clean_chars(), detect_script(), fold_name(), name_tokens(), normalize_name() (+27 more)

### Community 2 - "Pairs & Prediction"
Cohesion: 0.14
Nodes (27): build_training_pairs(), _explode_truth(), grouped_split(), DataFrame, _feature_sources(), _load_booster(), predict_parts(), run_predict() (+19 more)

### Community 3 - "Feature Engineering"
Cohesion: 0.16
Nodes (19): attach_country(), _combine_feature_parts(), compute_features(), _country_map(), _feature_block(), _pair_set_metrics(), _pairs_has_label(), _phase1_merged() (+11 more)

### Community 4 - "Blocking Audit"
Cohesion: 0.18
Nodes (15): audit_candidates(), _connect(), _explode_truth(), load_country_map(), _Progress, DataFrame, run_audit(), _sql_path() (+7 more)

### Community 5 - "Scoring & Thresholding"
Cohesion: 0.16
Nodes (15): evaluate_marks(), macro_f05(), _one_to_one_mask(), one_to_one(), DataFrame, entity_f05(), _entity_f05_from_codes(), sweep_threshold() (+7 more)

### Community 6 - "Blocking Engine"
Cohesion: 0.28
Nodes (13): block_keys(), compute_token_idf(), generate_candidates(), _join_sql(), load_token_idf(), normalize_keys(), _pass_caps(), run_block() (+5 more)

### Community 7 - "CLI, Config & Tests"
Cohesion: 0.27
Nodes (8): _default_config(), main(), Config, write_inference_pairs(), write_training_pairs(), cfg(), test_config_load_roundtrip(), fixture

### Community 8 - "GBDT Benchmark Tool"
Cohesion: 0.35
Nodes (11): best_entity_f05(), build_pairs(), compute_features(), entity_macro_f05(), load_positives(), load_s1_sample(), main(), norm_one() (+3 more)

### Community 9 - "Dataset EDA Tool"
Cohesion: 0.43
Nodes (6): cached(), check_membership(), main(), pct(), scan_ground_truth(), scan_source()

### Community 10 - "Submission Validator"
Cohesion: 0.62
Nodes (6): examples(), load_match_targets(), main(), read_ids(), validate(), validate_id_list_file()

## Knowledge Gaps
- **12 isolated node(s):** `D1 Blocking plus classifier, not end-to-end`, `D2 DuckDB for joins, pandas/pyarrow for the rest`, `F2 64-bucket join loop took >3 hours`, `F3 Pandas audit could not handle 98.8M rows`, `F4 Parquet list columns returned as numpy arrays` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_audit()` connect `Blocking Audit` to `CLI, Config & Tests`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `evaluate_marks()` connect `Scoring & Thresholding` to `Pairs & Prediction`, `CLI, Config & Tests`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Why does `main()` connect `CLI, Config & Tests` to `Normalization & Address Parsing`, `Pairs & Prediction`, `Feature Engineering`, `Blocking Audit`, `Scoring & Thresholding`, `Blocking Engine`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `prepare_frame()` (e.g. with `detect_script()` and `fold_name()`) actually correct?**
  _`prepare_frame()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `D1 Blocking plus classifier, not end-to-end`, `D2 DuckDB for joins, pandas/pyarrow for the rest`, `F2 64-bucket join loop took >3 hours` to the rest of the system?**
  _12 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Docs, Decisions & Metrics` be split into smaller, more focused modules?**
  _Cohesion score 0.07936507936507936 - nodes in this community are weakly interconnected._
- **Should `Normalization & Address Parsing` be split into smaller, more focused modules?**
  _Cohesion score 0.09966777408637874 - nodes in this community are weakly interconnected._