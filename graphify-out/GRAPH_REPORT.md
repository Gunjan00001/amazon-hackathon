# Graph Report - amazon-hackathon  (2026-09-26)

## Corpus Check
- Corpus is ~29,029 words - fits in a single context window. You may not need a graph.

## Summary
- 271 nodes · 636 edges · 21 communities (20 shown, 1 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 19 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- ber/search.py
- blocking.py
- train_gbdt.py
- stages/rerank.py
- best_threshold()
- Business Entity Resolution Challenge
- vector_features.py
- push_all.py
- io_tsv.py
- generate_candidates()
- bench_gbdt.py
- eda_dataset.py
- validate_submission.py
- make_notebooks.py
- auto.py
- Injective ground truth

## God Nodes (most connected - your core abstractions)
1. `finalize()` - 25 edges
2. `run_search()` - 22 edges
3. `prepare_records()` - 12 edges
4. `generate_candidates()` - 11 edges
5. `classical_features()` - 11 edges
6. `train_model()` - 11 edges
7. `fold_ascii()` - 11 edges
8. `normalize_name()` - 11 edges
9. `normalize_address()` - 11 edges
10. `prepare_keys()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `generate_candidates()`  [EXTRACTED]
  tools/probe_blocking.py → code/business_entity_resolution/src/ber/blocking.py
- `main()` --calls--> `split_id_list()`  [EXTRACTED]
  tools/probe_blocking.py → code/business_entity_resolution/src/ber/io_tsv.py
- `Open-set country (France zero-shot)` --conceptually_related_to--> `Business Entity Resolution Challenge`  [EXTRACTED]
  DATA/student_resource/dataset/DATASET.md → PROBLEM_STATEMENT.md
- `Three sources (S1 reference, S2/S3 noisy)` --conceptually_related_to--> `Business Entity Resolution Challenge`  [EXTRACTED]
  DATA/student_resource/dataset/DATASET.md → PROBLEM_STATEMENT.md
- `No external data lookup` --references--> `Business Entity Resolution Challenge`  [EXTRACTED]
  RULES.md → PROBLEM_STATEMENT.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Candidate-generation + matcher cascade stages** — design_idf_blocking, design_lightgbm_matcher, design_cross_encoder, design_embedding_stage [INFERRED 0.80]

## Communities (21 total, 1 thin omitted)

### Community 0 - "ber/search.py"
Cohesion: 0.11
Nodes (36): load_keys(), save_keys(), matches_from_scores(), one_to_one_assign(), prune_top_k(), to_entity_ids(), grouped_split(), _best_decision() (+28 more)

### Community 1 - "blocking.py"
Cohesion: 0.13
Nodes (30): _addr_tokens(), _idf(), mid_blocking_tokens(), _name_tokens(), _row_keys(), s1_blocking_tokens(), score_keys(), _significant() (+22 more)

### Community 2 - "train_gbdt.py"
Cohesion: 0.15
Nodes (25): build_pair_frame(), build_pair_frame_prepared(), classical_features(), _cpdist(), _jaccard_sets(), _lens(), prepare_records(), DataFrame (+17 more)

### Community 3 - "stages/rerank.py"
Cohesion: 0.17
Nodes (17): blocking_recall(), split_id_list(), positive_pairs(), _part(), ndarray, score_texts(), serialize_pair(), load_mid() (+9 more)

### Community 4 - "best_threshold()"
Cohesion: 0.20
Nodes (15): evaluate_solution(), holdout_mask(), best_threshold(), entity_f05(), macro_f05(), ndarray, _macro_from_pred(), _case() (+7 more)

### Community 5 - "Business Entity Resolution Challenge"
Cohesion: 0.11
Nodes (19): Pipeline reproduce README, Open-set country (France zero-shot), Cross-script Indic names, Three sources (S1 reference, S2/S3 noisy), Autonomous validation-driven search, Classical + Boost cascade (C+B), Cross-encoder rerank stage, Multilingual e5 embedding stage (+11 more)

### Community 6 - "vector_features.py"
Cohesion: 0.23
Nodes (13): encode_source(), load_model(), main(), cosine_matrix_rows(), dequantize(), load_vector_store(), pair_vector_features(), ndarray (+5 more)

### Community 7 - "push_all.py"
Cohesion: 0.33
Nodes (13): cmd_datasets(), cmd_fetch(), cmd_push(), cmd_status(), dataset_exists(), kernel_dir(), kernel_status(), main() (+5 more)

### Community 8 - "io_tsv.py"
Cohesion: 0.28
Nodes (10): format_id_list(), _id_sort_key(), s1_sort_key(), _write(), write_candidate_pairs(), write_matching_results(), test_id_list_roundtrip_and_empty(), test_write_matching_results_exact_format() (+2 more)

### Community 9 - "generate_candidates()"
Cohesion: 0.39
Nodes (11): build_index(), generate_candidates(), generate_candidates_from_keys(), prepare_keys(), _mid(), _s1(), test_cap_prefers_rare_shared_token(), test_generates_expected_pairs_and_country_gate() (+3 more)

### Community 10 - "bench_gbdt.py"
Cohesion: 0.35
Nodes (11): best_entity_f05(), build_pairs(), compute_features(), entity_macro_f05(), load_positives(), load_s1_sample(), main(), norm_one() (+3 more)

### Community 11 - "eda_dataset.py"
Cohesion: 0.43
Nodes (6): cached(), check_membership(), main(), pct(), scan_ground_truth(), scan_source()

### Community 12 - "validate_submission.py"
Cohesion: 0.62
Nodes (6): examples(), load_match_targets(), main(), read_ids(), validate(), validate_id_list_file()

### Community 13 - "make_notebooks.py"
Cohesion: 0.60
Nodes (5): code_cell(), main(), make_notebook(), md_cell(), run_expression()

### Community 14 - "auto.py"
Cohesion: 0.80
Nodes (4): find_student_resource(), main(), run(), validate()

## Knowledge Gaps
- **9 isolated node(s):** `Three sources (S1 reference, S2/S3 noisy)`, `Injective ground truth`, `Cross-script Indic names`, `No external data lookup`, `Kaggle-only compute` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `prepare_records()` connect `train_gbdt.py` to `ber/search.py`, `blocking.py`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Why does `pair_vector_features()` connect `vector_features.py` to `train_gbdt.py`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Why does `positive_pairs()` connect `stages/rerank.py` to `ber/search.py`, `train_gbdt.py`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `prepare_records()` (e.g. with `normalize_address()` and `normalize_name()`) actually correct?**
  _`prepare_records()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Three sources (S1 reference, S2/S3 noisy)`, `Injective ground truth`, `Cross-script Indic names` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `ber/search.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10707070707070707 - nodes in this community are weakly interconnected._
- **Should `blocking.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13445378151260504 - nodes in this community are weakly interconnected._