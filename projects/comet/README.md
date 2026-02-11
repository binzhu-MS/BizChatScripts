# CoMet (CopilotMetrics)

This folder contains code, documents, test data, and `.vscode` configuration for the [CopilotMetrics (CoMet)](https://o365exchange.visualstudio.com/DefaultCollection/O365%20Core/_git/CopilotMetrics) repository.

## Important Notes

- Some code in this folder (e.g., `batch_run_llm_ndcg.py`) **must be run from the CoMet repository**, as it depends on CoMet internal modules.

## Path Mapping

| This folder | CoMet repository |
|---|---|
| `comet/` (except `.vscode/`) | `CopilotMetrics\sources\dev\MetricDefinition\local` |
| `.vscode/` | `CopilotMetrics\.vscode` |