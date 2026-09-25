# Graph Report - Amazon project  (2026-09-25)

## Corpus Check
- Corpus is ~9,356 words - fits in a single context window. You may not need a graph.

## Summary
- 64 nodes · 106 edges · 8 communities
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 11 edges (avg confidence: 0.81)
- Token cost: 9,500 input · 4,200 output

## Community Hubs (Navigation)
- Challenge Docs & Country Constraints
- GBDT Benchmark Tool
- Sources, Ground Truth & Constraints
- Dataset EDA Tool
- Submission Validator
- Submission Package & Outputs
- Scoring & Singletons
- Data Noise & Matching Features

## God Nodes (most connected - your core abstractions)
1. `Business Entity Resolution Challenge` - 9 edges
2. `ML Challenge 2026 Problem Statement (README)` - 8 edges
3. `main()` - 7 edges
4. `matching_results.tsv` - 7 edges
5. `F_0.5 Score (β = 0.5)` - 7 edges
6. `validate()` - 6 edges
7. `norm_series()` - 5 edges
8. `main()` - 5 edges
9. `train_ground_truth.tsv` - 5 edges
10. `Solution Documentation Template` - 5 edges

## Surprising Connections (you probably didn't know these)
- `ML Challenge 2026 Problem Statement (README)` --semantically_similar_to--> `Business Entity Resolution Challenge`  [INFERRED] [semantically similar]
  D:/Amazon project/DATA/student_resource/README.md → D:/Amazon project/PROBLEM_STATEMENT.md
- `TSV / Comma / Encoding Reading Gotchas` --conceptually_related_to--> `matching_results.tsv`  [INFERRED]
  D:/Amazon project/DATA/student_resource/dataset/DATASET.md → D:/Amazon project/PROBLEM_STATEMENT.md
- `Singletons: 5.585% of Train S1 Entities` --semantically_similar_to--> `Singleton Scoring (no true matches)`  [INFERRED] [semantically similar]
  D:/Amazon project/DATA/student_resource/dataset/DATASET.md → D:/Amazon project/PROBLEM_STATEMENT.md
- `Template §4: Matching Model` --conceptually_related_to--> `String Similarity Features (Jaccard, Levenshtein, TF-IDF cosine)`  [INFERRED]
  D:/Amazon project/DATA/student_resource/Documentation_template.md → D:/Amazon project/PROBLEM_STATEMENT.md
- `Dataset Description (Measured Stats)` --references--> `Business Entity Resolution Challenge`  [EXTRACTED]
  D:/Amazon project/DATA/student_resource/dataset/DATASET.md → D:/Amazon project/PROBLEM_STATEMENT.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Entity Resolution Pipeline Flow (Sources -> Blocking -> Candidates -> Matches)** — problem_statement_source1, problem_statement_source2, problem_statement_source3, problem_statement_blocking, problem_statement_candidate_pairs, problem_statement_matching_results [EXTRACTED 1.00]
- **Final Submission Package Components** — problem_statement_submission_package, problem_statement_matching_results, problem_statement_candidate_pairs, documentation_template_solution_template [EXTRACTED 1.00]
- **Measured Dataset Constraints Shaping Evaluation** — dataset_injective_ground_truth, dataset_singletons, dataset_france_zero_shot, dataset_nonlatin_names, problem_statement_f05 [INFERRED 0.80]

## Communities (8 total, 0 thin omitted)

### Community 0 - "Challenge Docs & Country Constraints"
Cohesion: 0.19
Nodes (13): Measured Country Distribution (US/India/France), Dataset Description (Measured Stats), tools/eda_stats.json (EDA cache), tools/eda_dataset.py, France as Zero-Shot Country, Template §3: Candidate Generation (Blocking), Template §4: Matching Model, Solution Documentation Template (+5 more)

### Community 1 - "GBDT Benchmark Tool"
Cohesion: 0.35
Nodes (11): best_entity_f05(), build_pairs(), compute_features(), entity_macro_f05(), load_positives(), load_s1_sample(), main(), norm_one() (+3 more)

### Community 2 - "Sources, Ground Truth & Constraints"
Cohesion: 0.39
Nodes (8): Injective Ground-Truth Mapping, One-to-One Assignment Prior, Business Entity Resolution Challenge, train_ground_truth.tsv, Model Constraints (MIT/Apache 2.0, <=8B params), Source 1 (Deduplicated Reference Source), Source 2, Source 3

### Community 3 - "Dataset EDA Tool"
Cohesion: 0.43
Nodes (6): cached(), check_membership(), main(), pct(), scan_ground_truth(), scan_source()

### Community 4 - "Submission Validator"
Cohesion: 0.62
Nodes (6): examples(), load_match_targets(), main(), read_ids(), validate(), validate_id_list_file()

### Community 5 - "Submission Package & Outputs"
Cohesion: 0.47
Nodes (6): TSV / Comma / Encoding Reading Gotchas, Template Appendix A: Code Artefacts, candidate_pairs.tsv, matching_results.tsv, Final Submission Package (zip), utils/validate_submission.py

### Community 6 - "Scoring & Singletons"
Cohesion: 0.50
Nodes (5): Match-Count Distribution (mean 3.461, max 11), Singletons: 5.585% of Train S1 Entities, Entity Resolution (ER), F_0.5 Score (β = 0.5), Singleton Scoring (no true matches)

### Community 7 - "Data Noise & Matching Features"
Cohesion: 0.50
Nodes (5): Missing business_address (~3.3% of S2/S3), Non-Latin Script Names (~15% of Train S2), UTF-8 Files / cp1252 Console Artifact, Noise Patterns (Name & Address Variations), String Similarity Features (Jaccard, Levenshtein, TF-IDF cosine)

## Knowledge Gaps
- **9 isolated node(s):** `Model Constraints (MIT/Apache 2.0, <=8B params)`, `Template Appendix A: Code Artefacts`, `tools/eda_stats.json (EDA cache)`, `Match-Count Distribution (mean 3.461, max 11)`, `Missing business_address (~3.3% of S2/S3)` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `F_0.5 Score (β = 0.5)` connect `Scoring & Singletons` to `Challenge Docs & Country Constraints`, `Sources, Ground Truth & Constraints`, `Submission Package & Outputs`?**
  _High betweenness centrality (0.114) - this node is a cross-community bridge._
- **Why does `ML Challenge 2026 Problem Statement (README)` connect `Challenge Docs & Country Constraints` to `Sources, Ground Truth & Constraints`, `Submission Package & Outputs`, `Scoring & Singletons`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `Template §4: Matching Model` connect `Challenge Docs & Country Constraints` to `Scoring & Singletons`, `Data Noise & Matching Features`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **What connects `Model Constraints (MIT/Apache 2.0, <=8B params)`, `Template Appendix A: Code Artefacts`, `tools/eda_stats.json (EDA cache)` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._