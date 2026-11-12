# Reference results

These files retain the original computational assessments and superseded experimental comparisons. The current experimental assessment is selected in `inputs/experimental_assessment.json`; current figures use that revision. Current experimental sampling is described in [`docs/DATA.md`](../../docs/DATA.md); earlier metadata are retained as historical records.

- `low_complete_pool/`, `high_complete_pool/`: complete 644-context assessments of fixed deployed tables.
- `coverage_size_experiment/`: isolated initialization, bounded fitting over three font domains and complete-pool assessment.
- `benchmark_*.json`, `benchmark_summary.csv`, `workstation_environment.json`: synchronized performance measurements and hardware/runtime records.
- `kernel_parity_checks.json`, `native_case_parity.json`, `time_step_probe/`, `public_synthetic/`: numerical verification with distinct test scopes.
- `experimental_*`: paired-coordinate metrics, per-glyph summaries and version relationships. Measurements represent ten-print mean positions per raster.
- `context_pool_audit.json`: characterized contexts, populated lookup keys and binary table capacity.
- `structure_search/`: bounded search candidates, evaluations and selected refinements.

The controlled force-ablation and Euler/RK4 studies are stored separately in [`results/numerical/`](../evidence_v2/).
