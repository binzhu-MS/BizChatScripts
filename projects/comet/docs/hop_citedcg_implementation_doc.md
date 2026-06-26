# Hop CiteDCG Metric - Implementation Document

> **⚠️ RENAMED (December 2025):** This metric was renamed from `hop_dcg` to `hop_citedcg` to clarify that it uses **CiteDCGLLMLabel** (citation relevance to original utterance), not **LLMLabel** (retrieval relevance to search query). For retrieval quality metrics (CG, DCG, NDCG), see `llm_ndcg` or `retrieved_good_gain`.

## Overview

**Metric Name:** `hop_citedcg`  
**Version:** 1.0.0  
**Created:** December 10, 2025  
**Status:** Implemented ✅

---

## 1. Purpose

The **Hop DCG** (Hop Discounted Cumulative Gain) metric calculates per-hop average DCG scores for search results, enabling comparison of retrieval quality at different stages of the orchestration process between control and treatment arms.

### Key Use Cases
- Understand retrieval quality at each orchestration iteration
- Compare how control vs treatment performs at different hops
- Identify if quality degrades or improves across hops
- Enable @K analysis (top 3 vs top 5 vs all results)

### Relationship to CiteDCG
- **Input:** Uses `CiteDCGLLMLabel` scores from citeDCG's debug output
- **Dependency:** Requires citeDCG to run first with `debug=True`
- **Difference:** CiteDCG focuses on *cited* entities; hop_dcg analyzes *all* search results per iteration

---

## 2. File Structure

### 2.1 Production Files (To be committed to repository)

```
cometdefinition/metrics/hop_dcg/
├── __init__.py                    # Package initialization
├── hop_dcg.proto                  # Protobuf definitions
├── hop_dcg_pb2.py                 # Generated protobuf classes
├── hop_dcg_pb2.pyi                # Generated type stubs
├── README.md                      # User-facing documentation
├── config/
│   └── metric_spec.yaml           # Metric configuration
└── logic/
    ├── __init__.py                # Logic package initialization
    └── metric_logic.py            # Main implementation
```

**Registry Updates (existing files modified):**
```
cometdefinition/metrics/__init__.py        # Added import for hop_dcg metric
cometdefinition/__init__.py                # Added import for hop_dcg protobuf classes
```

### 2.2 E2E Test Files (Production)

```
cometdefinition/e2e_tests/deterministic/
├── deterministic_e2e_testing.py           # E2E test runner (TCase entry added)
└── data/
    ├── input/hop_dcg/
    │   └── evaluation_data.jsonl          # E2E test input data (~8.6 MB)
    └── output/hop_dcg/
        └── evaluation_data.jsonl          # E2E expected output (11 test cases)
```

### 2.3 Testing/Development Files (Local only)

```
sources/dev/MetricDefinition/temp/
├── hop_dcg_input.json                     # Sample test input file
├── hop_dcg_output.json                    # Generated test output (after running test)
├── test_hop_dcg_integration.py            # Integration test with citeDCG debug format
├── hop_dcg_integration_output.json        # Generated integration test output
├── generate_hop_dcg_e2e_data.py           # Script to generate E2E test data from CiteDCG
├── hop_dcg_implementation_plan.md         # Original implementation plan
└── hop_dcg_implementation_doc.md          # This implementation document
```

---

## 3. Implementation Details

### 3.1 Protobuf Definitions (`hop_dcg.proto`)

```protobuf
// Input message
message HopDCGInput {
    string cite_dcg_debug_control = 1;     // JSON from citeDCG debug (control)
    string cite_dcg_debug_treatment = 2;   // JSON from citeDCG debug (treatment)
    repeated int32 top_k_values = 3;       // Optional: custom @K values
    bool include_all_results = 4;          // Optional: include avg over all
}

// Output message
message HopDCGOutput {
    string Id = 1;
    repeated HopMetrics control_hops = 2;
    repeated HopMetrics treatment_hops = 3;
    HopSummary control_summary = 4;
    HopSummary treatment_summary = 5;
}

// Per-hop metrics
message HopMetrics {
    int32 hop = 1;              // Hop number (0, 1, 2, ...)
    int32 num_results = 2;      // Total results in hop
    int32 num_labeled = 3;      // Results with valid labels
    float avg_all = 4;          // Average over all results
    float avg_at_3 = 5;         // Average over top 3
    float avg_at_5 = 6;         // Average over top 5
    repeated TopKScore additional_top_k = 7;
}

// Summary statistics
message HopSummary {
    int32 total_hops = 1;
    int32 total_results = 2;
    int32 total_labeled = 3;
    float overall_avg = 4;      // Weighted average across all hops
}
```

### 3.2 Configuration (`metric_spec.yaml`)

```yaml
name: "hop_dcg"
description: "Per-hop average DCG scores for search results"
version: 1.0.0
author: comet@microsoft.com
lifecycle: development
icm_team: "MSAI User Understanding/BizChatExperimentationDRI"
ado_template: "https://aka.ms/comet-metric-bug"
# No LLM needed - pure computation from citeDCG output
```

### 3.3 Metric Logic (`metric_logic.py`)

#### Class: `HopDCGMetric`

Inherits from `CopilotMetricBase[HopDCGInput, HopDCGOutput]`

#### Methods Implemented:

| Method | Description |
|--------|-------------|
| `__init__` | Initialize metric with input parameters |
| `from_raw` | Class method to create from raw dictionary input |
| `validate` | Validate inputs (at least one arm required, positive K values) |
| `preprocess` | Parse JSON and extract AllSearchResults from citeDCG debug |
| `execute` | Calculate per-hop metrics for both arms |
| `validate_output` | Validate output structure and value ranges |

#### Key Private Methods:

| Method | Description |
|--------|-------------|
| `_parse_cite_dcg_debug` | Parse citeDCG debug JSON, extract AllSearchResults |
| `_calculate_hop_metrics` | Calculate metrics for all hops in one arm |
| `_extract_labels_from_hop` | Extract CiteDCGLLMLabel values from hop data |
| `_calculate_top_k_avg` | Calculate average over top K results |
| `_calculate_summary` | Calculate summary statistics across all hops |

---

## 4. Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Input: CiteDCG Debug Output (JSON)                              │
│                                                                 │
│  {                                                              │
│    "AllSearchResults": {                                        │
│      "0": { plugin: [{ Results: [...] }] },                     │
│      "1": { plugin: [{ Results: [...] }] }                      │
│    }                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ preprocess()                                                    │
│   - Parse JSON strings                                          │
│   - Extract AllSearchResults for control and treatment          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ execute()                                                       │
│   - For each hop:                                               │
│     - Extract CiteDCGLLMLabel from all entities                 │
│     - Calculate avg_all, avg_at_3, avg_at_5                     │
│   - Calculate summary statistics                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Output: HopDCGOutput                                            │
│                                                                 │
│  {                                                              │
│    "Id": "...",                                                 │
│    "control_hops": [ { hop: 0, avg_all: 2.4, ... }, ... ],      │
│    "treatment_hops": [ { hop: 0, avg_all: 3.5, ... }, ... ],    │
│    "control_summary": { total_hops: 2, overall_avg: 2.5 },      │
│    "treatment_summary": { total_hops: 1, overall_avg: 3.5 }     │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Calculation Logic

### Per-Hop Calculation

```python
def _calculate_hop_metrics(all_search_results):
    for hop_id in sorted(all_search_results.keys()):
        hop_data = all_search_results[hop_id]
        
        # Extract all labels from this hop
        labels = []
        for plugin_name, plugin_results in hop_data.items():
            for result_set in plugin_results:
                for entity in result_set.get('Results', []):
                    label = entity.get('CiteDCGLLMLabel')
                    if label is not None:
                        labels.append(label)
        
        # Calculate averages
        avg_all = mean(labels) if labels else 0.0
        avg_at_3 = mean(labels[:3]) if labels else 0.0
        avg_at_5 = mean(labels[:5]) if labels else 0.0
```

### Summary Calculation

```python
def _calculate_summary(hop_metrics_list):
    total_hops = len(hop_metrics_list)
    total_results = sum(hop["num_results"] for hop in hop_metrics_list)
    total_labeled = sum(hop["num_labeled"] for hop in hop_metrics_list)
    
    # Weighted average by number of labeled results
    weighted_sum = sum(
        hop["avg_all"] * hop["num_labeled"]
        for hop in hop_metrics_list
        if hop["num_labeled"] > 0
    )
    overall_avg = weighted_sum / total_labeled if total_labeled > 0 else 0.0
```

---

## 6. Edge Cases Handled

| Case | Handling |
|------|----------|
| No results in a hop | Output `avg = 0.0`, `num_results = 0` |
| Fewer than K results for @K | Average over available results |
| Missing CiteDCGLLMLabel | Skip entity, count as unlabeled |
| Empty citeDCG debug | Return empty hop list, log warning |
| Invalid JSON input | Return empty dict, log warning |
| Both arms empty | Raise validation error in `validate_output` |
| Non-numeric hop IDs | Treated as hop 0 |

---

## 7. Sample Test Input File

A sample input file is provided at `sources/dev/MetricDefinition/hop_dcg_input.json`:

```json
{
    "id": "test_utterance_001",
    "cite_dcg_debug_control": {
        "AllSearchResults": {
            "0": {
                "search_enterprise_files": [
                    {
                        "Results": [
                            {"Id": "file1", "Rank": 1, "CiteDCGLLMLabel": 3, "Type": "File"},
                            {"Id": "file2", "Rank": 2, "CiteDCGLLMLabel": 2, "Type": "File"},
                            {"Id": "file3", "Rank": 3, "CiteDCGLLMLabel": 4, "Type": "File"},
                            {"Id": "file4", "Rank": 4, "CiteDCGLLMLabel": 1, "Type": "File"},
                            {"Id": "file5", "Rank": 5, "CiteDCGLLMLabel": 2, "Type": "File"}
                        ]
                    }
                ],
                "search_enterprise_emails": [
                    {
                        "Results": [
                            {"Id": "email1", "Rank": 1, "CiteDCGLLMLabel": 2, "Type": "Message"},
                            {"Id": "email2", "Rank": 2, "CiteDCGLLMLabel": 1, "Type": "Message"}
                        ]
                    }
                ]
            },
            "1": {
                "search_enterprise_files": [
                    {
                        "Results": [
                            {"Id": "file6", "Rank": 1, "CiteDCGLLMLabel": 3, "Type": "File"},
                            {"Id": "file7", "Rank": 2, "CiteDCGLLMLabel": 4, "Type": "File"}
                        ]
                    }
                ]
            }
        }
    },
    "cite_dcg_debug_treatment": {
        "AllSearchResults": {
            "0": {
                "search_enterprise_files": [
                    {
                        "Results": [
                            {"Id": "file1", "Rank": 1, "CiteDCGLLMLabel": 4, "Type": "File"},
                            {"Id": "file5", "Rank": 2, "CiteDCGLLMLabel": 3, "Type": "File"},
                            {"Id": "file3", "Rank": 3, "CiteDCGLLMLabel": 4, "Type": "File"}
                        ]
                    }
                ],
                "search_enterprise_emails": [
                    {
                        "Results": [
                            {"Id": "email3", "Rank": 1, "CiteDCGLLMLabel": 3, "Type": "Message"}
                        ]
                    }
                ]
            }
        }
    }
}
```

### Input Structure Explained

| Field | Description |
|-------|-------------|
| `id` | Unique identifier for the utterance/evaluation |
| `cite_dcg_debug_control` | CiteDCG debug output for control arm |
| `cite_dcg_debug_treatment` | CiteDCG debug output for treatment arm |
| `AllSearchResults` | Dict keyed by hop number ("0", "1", ...) |
| `search_enterprise_*` | Plugin name containing result sets |
| `Results` | Array of entities with `CiteDCGLLMLabel` scores |

### Where Does Input Come From in Production?

In production, the input comes from **citeDCG's debug output**. When citeDCG runs with `debug=True`, it outputs a JSON object containing `AllSearchResults` with `CiteDCGLLMLabel` on each entity. That debug output gets passed to hop_dcg as input.

---

## 8. Running the Test

### Command

```bash
cd sources/dev/MetricDefinition

# Using the comet Python environment:
python scripts/testing_metric.py \
    --metric hop_dcg \
    --input temp/hop_dcg_input.json \
    --output temp/hop_dcg_output.json
```

### Sample Output

```json
{
  "Id": "test_utterance_001",
  "control_hops": [
    {
      "hop": 0,
      "num_results": 7,
      "num_labeled": 7,
      "avg_all": 2.142857,
      "avg_at_3": 3.0,
      "avg_at_5": 2.4
    },
    {
      "hop": 1,
      "num_results": 2,
      "num_labeled": 2,
      "avg_all": 3.5,
      "avg_at_3": 3.5,
      "avg_at_5": 3.5
    }
  ],
  "treatment_hops": [
    {
      "hop": 0,
      "num_results": 4,
      "num_labeled": 4,
      "avg_all": 3.5,
      "avg_at_3": 3.667,
      "avg_at_5": 3.5
    }
  ],
  "control_summary": {
    "total_hops": 2,
    "total_results": 9,
    "total_labeled": 9,
    "overall_avg": 2.444
  },
  "treatment_summary": {
    "total_hops": 1,
    "total_results": 4,
    "total_labeled": 4,
    "overall_avg": 3.5
  }
}
```

---

## 8.1 Integration Testing with CiteDCG

An integration test file (`temp/test_hop_dcg_integration.py`) demonstrates how hop_dcg works with citeDCG's debug output.

### Running the Integration Test

```bash
cd sources/dev/MetricDefinition
python temp/test_hop_dcg_integration.py
```

### What the Integration Test Does

1. **Simulates citeDCG debug output** - Creates realistic `AllSearchResults` data with `CiteDCGLLMLabel` on each result
2. **Transforms to hop_dcg input format** - Extracts control/treatment data
3. **Runs hop_dcg metric** - Validates, preprocesses, and executes
4. **Displays formatted results** - Shows per-hop metrics and summary

### Sample Integration Test Output

```
============================================================
HopDCG Integration Test with CiteDCG Debug Output
============================================================

[Step 1] Creating simulated citeDCG debug output...
  - Control has 2 hop(s)
  - Treatment has 1 hop(s)
  - Control has 6 total results with labels
  - Treatment has 2 total results with labels

[Step 2] Creating hop_dcg input from citeDCG output...
  - Include all results: True
  - Top K values: [3, 5]

[Step 3] Running HopDCG metric...
  - Input validation passed
  - Preprocessing completed
  - Execution completed
  - Output validation passed

[Step 4] Results:
----------------------------------------

  CONTROL ARM:
    Hop 1: 4 results (4 labeled)
           Avg@All=2.0000, Avg@3=2.0000, Avg@5=2.0000
    Hop 2: 2 results (2 labeled)
           Avg@All=1.5000, Avg@3=1.5000, Avg@5=1.5000
    Summary: 2 hops, 6 total results
             Overall avg label: 1.8333

  TREATMENT ARM:
    Hop 1: 2 results (2 labeled)
           Avg@All=2.5000, Avg@3=2.5000, Avg@5=2.5000
    Summary: 1 hops, 2 total results
             Overall avg label: 2.5000

============================================================
Integration Test PASSED!
============================================================
```

### How citeDCG Produces Debug Output

In production, citeDCG produces debug output when `debug=True`:

```python
# Inside citeDCG's execute() method
if self.debug:
    metrics["debug"] = json.dumps({
        "control": {
            "AllSearchResults": self.all_search_results.control,  # With CiteDCGLLMLabel added
            ...
        },
        "treatment": {
            "AllSearchResults": self.all_search_results.treatment,  # With CiteDCGLLMLabel added
            ...
        }
    })
```

The `CiteDCGLLMLabel` is added via `add_labels()`:

```python
def add_labels(self, all_search_results, cited_entities):
    for result, _ in utils.yield_all_webwork_entities(all_search_results):
        result_id = self.get_result_id(result)
        if result_id in cited_entities:
            result["CiteDCGLLMLabel"] = cited_entities[result_id].label
    return all_search_results
```

---

## 8.2 E2E Test Pipeline Integration

The hop_dcg metric is integrated into the deterministic E2E test pipeline, allowing automated testing alongside CiteDCG and other metrics.

### E2E Test Registration

Added to `deterministic_e2e_testing.py`:

```python
TCase(
    name="HopDCG E2E Test Group",
    metric="hop_dcg",
    input="data/input/hop_dcg/evaluation_data.jsonl",
    output="data/output/hop_dcg/evaluation_data.jsonl",
),
```

### Running E2E Tests

```bash
cd sources/dev/MetricDefinition

# Run only hop_dcg E2E test
python -m pytest cometdefinition/e2e_tests/deterministic/deterministic_e2e_testing.py -k "HopDCG" -v

# Run all E2E tests including hop_dcg
python -m pytest cometdefinition/e2e_tests/deterministic/deterministic_e2e_testing.py -v
```

### Generating E2E Test Data

A script (`temp/generate_hop_dcg_e2e_data.py`) generates hop_dcg E2E test data from CiteDCG's test data:

```bash
cd sources/dev/MetricDefinition
python temp/generate_hop_dcg_e2e_data.py
```

**What the script does:**
1. Runs CiteDCG metric with `debug=True` and `llm_mock` on existing CiteDCG test data
2. Extracts the debug output containing `AllSearchResults` with `CiteDCGLLMLabel`
3. Passes the debug output through hop_dcg metric
4. Saves input/output pairs as E2E test data

**Generated files:**
- Input: `cometdefinition/e2e_tests/deterministic/data/input/hop_dcg/evaluation_data.jsonl` (11 test cases)
- Output: `cometdefinition/e2e_tests/deterministic/data/output/hop_dcg/evaluation_data.jsonl`

### E2E Test Verification

```
$ pytest ... -k "HopDCG" -v
test_e2e[HopDCG E2E Test Group] PASSED
```

---

## 9. Registrations

### `cometdefinition/metrics/__init__.py`

```python
from cometdefinition.metrics.hop_dcg.logic.metric_logic import *
```

### `cometdefinition/__init__.py`

```python
from cometdefinition.metrics.hop_dcg.hop_dcg_pb2 import *
```

---

## 10. Usage Examples

### Basic Usage

```python
from cometdefinition.metrics import METRICS_MAP
from cometdefinition.metrics.hop_dcg.hop_dcg_pb2 import HopDCGInput

# Create input
metric_input = HopDCGInput(
    cite_dcg_debug_control='{"AllSearchResults": {"0": {...}}}',
    cite_dcg_debug_treatment='{"AllSearchResults": {"0": {...}}}',
)

# Get metric class and create instance
HopDCGMetric = METRICS_MAP["hop_dcg"]
metric = HopDCGMetric(id="test_id", metric_input=metric_input)

# Calculate
result = metric.calculate_metric()
```

### Using from_raw

```python
metric = HopDCGMetric.from_raw(
    id="test_id",
    cite_dcg_debug_control={"AllSearchResults": {"0": {...}}},
    cite_dcg_debug_treatment={"AllSearchResults": {"0": {...}}},
    top_k_values=[3, 5, 10],  # Custom @K values
)
result = metric.calculate_metric()
```

### Testing with Script

```bash
# At sources/dev/MetricDefinition
python scripts/testing_metric.py \
    --metric hop_dcg \
    --input "hop_dcg_input.json" \
    --output "hop_dcg_result.json"
```

---

## 11. SEVAL Integration

### Expected SEVAL Output

| Metric | Control | Treatment | Diff | P-value |
|--------|---------|-----------|------|---------|
| hop_0_avg_all | 2.45 | 2.67 | +0.22 | 0.03 |
| hop_0_avg_at_3 | 2.89 | 3.12 | +0.23 | 0.02 |
| hop_0_avg_at_5 | 2.71 | 2.95 | +0.24 | 0.01 |
| hop_1_avg_all | 2.12 | 2.34 | +0.22 | 0.05 |

---

## 12. Testing Recommendations

### Unit Tests to Create

1. **test_validate_missing_inputs** - Verify validation error when both arms empty
2. **test_validate_invalid_top_k** - Verify validation error for non-positive K values
3. **test_preprocess_valid_json** - Test JSON parsing
4. **test_preprocess_invalid_json** - Test graceful handling of malformed JSON
5. **test_execute_single_hop** - Basic single hop calculation
6. **test_execute_multiple_hops** - Multiple hops with varying results
7. **test_execute_empty_hop** - Hop with no results
8. **test_top_k_fewer_results** - K greater than available results
9. **test_summary_weighted_average** - Verify weighted average calculation
10. **test_validate_output_empty_arms** - Output validation with no data

### Test File Location

```
cometdefinition/tests/test_hop_dcg.py
```

---

## 13. Future Enhancements

1. **DCG Score Calculation** - Add actual DCG/NDCG calculation (not just average)
2. **Per-Plugin Breakdown** - Add metrics broken down by plugin
3. **Per-Entity-Type Breakdown** - Add metrics broken down by entity type
4. **Async Support** - Add support for batch execution via BatchMetricOrchestrator
5. **Caching** - Add caching support for repeated calculations

---

## 14. Verification

```bash
# Verify import works
python -c "from cometdefinition.metrics.hop_dcg.logic.metric_logic import HopDCGMetric; print('OK')"

# Verify registration
python -c "from cometdefinition.metrics import METRICS_MAP; print('hop_dcg' in METRICS_MAP)"

# Run build to compile proto
python dev_setup.py build

# Run E2E test
python -m pytest cometdefinition/e2e_tests/deterministic/deterministic_e2e_testing.py -k "HopDCG" -v
```

All verifications passed ✅

---

## 15. Implementation Changelog

| Date | Change |
|------|--------|
| Dec 10, 2025 | Initial implementation - proto, config, metric logic, README |
| Dec 10, 2025 | Bug fix: `HasField` on boolean field (use `include_all_results` directly) |
| Dec 11, 2025 | E2E test integration - added TCase, generated test data from CiteDCG |
