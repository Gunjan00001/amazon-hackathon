# Documentation index

Start here. This folder collects the reference material for the Amazon ML Challenge 2026
**Business Entity Resolution** solution. The root [`README.md`](../README.md) is the
project landing page; the deep-dive docs live here.

| Document | What it covers |
|---|---|
| [`../README.md`](../README.md) | Project overview, headline results, quickstart, version log. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System design, pipeline stages, module map, data flow. |
| [`REPRODUCE.md`](REPRODUCE.md) | End-to-end reproduction: environment, local stages, Kaggle runs, validation. |
| [`REPOSITORY_GUIDE.md`](REPOSITORY_GUIDE.md) | Repository/GitHub layout, Git LFS, what is excluded and how to restore it. |
| [`../PROBLEM_STATEMENT.md`](../PROBLEM_STATEMENT.md) | Full transcription of the official challenge statement. |
| [`../RULES.md`](../RULES.md) | Binding project rules (fair play, data handling, engineering, compute). |
| [`../code/business_entity_resolution/README.md`](../code/business_entity_resolution/README.md) | Package-level usage notes for the pipeline code. |
| [`../kaggle/PUSH_INSTRUCTIONS.md`](../kaggle/PUSH_INSTRUCTIONS.md) | Kaggle push/run instructions (automated and manual). |
| [`../DATA/student_resource/dataset/DATASET.md`](../DATA/student_resource/dataset/DATASET.md) | Measured dataset facts: schemas, row counts, noise. (Present only when `DATA/` is restored.) |
| [`superpowers/specs/2026-09-25-kaggle-cascade-design.md`](superpowers/specs/2026-09-25-kaggle-cascade-design.md) | Original cascade `C+B` design specification. |
| [`superpowers/plans/2026-09-25-entity-resolution-pipeline.md`](superpowers/plans/2026-09-25-entity-resolution-pipeline.md) | Step-by-step implementation plan. |
| [`reference/approach-options.png`](reference/approach-options.png) | One-page comparison of Approaches A/B/C (screenshot). |
| [`../graphify-out/GRAPH_REPORT.md`](../graphify-out/GRAPH_REPORT.md) | Knowledge-graph report over the project docs and tools. |

> The `DATA/` dataset files (`.tsv`, `.zip`) are **not** in this repository — see
> [`REPOSITORY_GUIDE.md`](REPOSITORY_GUIDE.md) for how to restore them.
