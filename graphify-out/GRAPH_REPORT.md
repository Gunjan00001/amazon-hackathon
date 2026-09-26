# Graph Report - Amazon project  (2026-09-26)

## Corpus Check
- Corpus is ~42,197 words - fits in a single context window. You may not need a graph.

## Summary
- 258 nodes · 576 edges · 12 communities
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.87)
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

## God Nodes (most connected - your core abstractions)
1. `Decisions and Reasoning (DECISIONS.md)` - 18 edges
2. `M2 Precision + Colab Plan (10 Tasks)` - 18 edges
3. `main()` - 15 edges
4. `prepare_frame()` - 13 edges
5. `AGENTS.md Working Notes` - 12 edges
6. `M2 Precision + Colab Design Spec` - 12 edges
7. `normalize_name()` - 11 edges
8. `evaluate_loo()` - 11 edges
9. `Repo README` - 11 edges
10. `parse_address()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Results Metrics` --references--> `Milestone Version Tags (1.4.0-2.0.0)`  [INFERRED]
  docs/RESULTS.md → RULES.md
- `Challenge Documentation Template` --conceptually_related_to--> `Filled Methodology Write-up`  [EXTRACTED]
  DATA/student_resource/Documentation_template.md → Documentation_template.md
- `Official Challenge README` --semantically_similar_to--> `Problem Statement (PROBLEM_STATEMENT.md)`  [EXTRACTED] [semantically similar]
  DATA/student_resource/README.md → PROBLEM_STATEMENT.md
- `Results Metrics` --references--> `Submission README`  [INFERRED]
  docs/RESULTS.md → submission/README.md
- `Local ER Pipeline Implementation Plan` --references--> `33-36 Pairwise Feature Set`  [EXTRACTED]
  docs/superpowers/plans/2026-09-25-er-pipeline.md → Documentation_template.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **M2 Precision + Colab Flow** — char_ngram_features_d14, per_country_thresholds_d15, singleton_calibration, per_pair_cosine_interface_d13, adoption_gates [EXTRACTED 0.80]
- **Blocking to Recall Ceiling Flow** — ten_blocking_passes, blocking_cap, train_recall_08115, blocking_recall_ceiling_0814, candidate_pairs_contract [EXTRACTED 0.80]
- **Submission Output and Validation Flow** — matching_results_contract, candidate_pairs_contract, submission_validator, submission_tree, official_leaderboard_0811 [EXTRACTED 0.85]

## Communities (12 total, 0 thin omitted)

### Community 0 - "Docs, Metrics & Leaderboard"
Cohesion: 0.09
Nodes (64): M2 Adoption Gates (numeric comparisons vs 0.8488), AGENTS.md Working Notes, Per-S1 Candidate Cap 200 + Per-Pass pass_caps, Blocking + Classifier Approach (D1), Blocking Recall Ceiling 0.814, candidate_pairs.tsv Contract, Char N-Gram TF-IDF Cosine Features (D14), Pinned Requirements (+56 more)

### Community 1 - "Normalization, Address & CLI"
Cohesion: 0.10
Nodes (35): AddressParts, parse_address(), _postal_re(), _clean_chars(), detect_script(), fold_name(), name_tokens(), normalize_name() (+27 more)

### Community 2 - "Prediction & Validation"
Cohesion: 0.14
Nodes (27): build_training_pairs(), _explode_truth(), grouped_split(), DataFrame, _feature_sources(), _load_booster(), predict_parts(), run_predict() (+19 more)

### Community 3 - "Blocking Audit"
Cohesion: 0.16
Nodes (19): attach_country(), _combine_feature_parts(), compute_features(), _country_map(), _feature_block(), _pair_set_metrics(), _pairs_has_label(), _phase1_merged() (+11 more)

### Community 4 - "Scoring & Thresholding"
Cohesion: 0.18
Nodes (15): audit_candidates(), _connect(), _explode_truth(), load_country_map(), _Progress, DataFrame, run_audit(), _sql_path() (+7 more)

### Community 5 - "Feature Engineering"
Cohesion: 0.16
Nodes (15): evaluate_marks(), macro_f05(), _one_to_one_mask(), one_to_one(), DataFrame, entity_f05(), _entity_f05_from_codes(), sweep_threshold() (+7 more)

### Community 6 - "Blocking Engine"
Cohesion: 0.28
Nodes (13): block_keys(), compute_token_idf(), generate_candidates(), _join_sql(), load_token_idf(), normalize_keys(), _pass_caps(), run_block() (+5 more)

### Community 7 - "Config & Test Fixtures"
Cohesion: 0.27
Nodes (8): _default_config(), main(), Config, write_inference_pairs(), write_training_pairs(), cfg(), test_config_load_roundtrip(), fixture

### Community 8 - "GBDT Benchmark Tool"
Cohesion: 0.35
Nodes (11): best_entity_f05(), build_pairs(), compute_features(), entity_macro_f05(), load_positives(), load_s1_sample(), main(), norm_one() (+3 more)

### Community 9 - "Transliteration"
Cohesion: 0.43
Nodes (6): cached(), check_membership(), main(), pct(), scan_ground_truth(), scan_source()

### Community 10 - "Pair Construction"
Cohesion: 0.62
Nodes (6): examples(), load_match_targets(), main(), read_ids(), validate(), validate_id_list_file()

## Knowledge Gaps
- **4 isolated node(s):** `Challenge Documentation Template`, `4:1 Negative Sampling (D6)`, `Full-Candidate Held-Out Validation (D9)`, `Per-S1 Candidate Cap 200 + Per-Pass pass_caps`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_audit()` connect `Scoring & Thresholding` to `Config & Test Fixtures`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Why does `evaluate_marks()` connect `Feature Engineering` to `Prediction & Validation`, `Config & Test Fixtures`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `main()` connect `Config & Test Fixtures` to `Normalization, Address & CLI`, `Prediction & Validation`, `Blocking Audit`, `Scoring & Thresholding`, `Feature Engineering`, `Blocking Engine`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `prepare_frame()` (e.g. with `detect_script()` and `fold_name()`) actually correct?**
  _`prepare_frame()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Challenge Documentation Template`, `4:1 Negative Sampling (D6)`, `Full-Candidate Held-Out Validation (D9)` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Docs, Metrics & Leaderboard` be split into smaller, more focused modules?**
  _Cohesion score 0.08581349206349206 - nodes in this community are weakly interconnected._
- **Should `Normalization, Address & CLI` be split into smaller, more focused modules?**
  _Cohesion score 0.09966777408637874 - nodes in this community are weakly interconnected._