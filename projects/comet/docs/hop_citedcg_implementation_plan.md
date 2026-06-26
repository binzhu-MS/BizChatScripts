# Hop CiteDCG Metric Implementation Plan

> **⚠️ RENAMED (December 2025):** This metric was renamed from `hop_dcg` to `hop_citedcg` to clarify that it uses **CiteDCGLLMLabel** (citation relevance to original utterance), not **LLMLabel** (retrieval relevance to search query). For retrieval quality metrics (CG, DCG, NDCG), see `llm_ndcg` or `retrieved_good_gain`.

## Overview

**Metric Name:** `hop_citedcg`

**Purpose:** Calculate average CiteDCG scores per hop (orchestration iteration) for search results, enabling comparison of citation quality at different stages of the search process between control and treatment.

**Author:** [Your Name]  
**Date:** December 9, 2025  
**Status:** ✅ Implemented

---

## 1. Metric Description

### What It Measures
For each utterance, this metric:
1. Groups all search results by **hop** (iteration/orchestration step)
2. For each hop, calculates:
   - **Average CiteDCG label over ALL results** in that hop
   - **Average CiteDCG label over top-K results** (default: @3 and @5)
3. Outputs per-hop scores for both control and treatment arms

### Important: Label Types

| Label Type | Measures | Used By |
|------------|----------|----------|
| **CiteDCGLLMLabel** | Relevance to user's **original utterance** | cite_dcg, **hop_citedcg** |
| **LLMLabel** | Relevance to **search query** | llm_ndcg, retrieved_good_gain |

### Why It's Needed
- Understand citation quality at each orchestration iteration
- Compare how control vs treatment performs at different hops
- Identify if quality degrades or improves across hops
- Enable @K analysis (top 3 vs top 5 vs all results)

### Relationship to CiteDCG
- **Input:** Uses `CiteDCGLLMLabel` scores from citeDCG's debug output
- **Dependency:** Requires citeDCG to run first with `debug=True`
- **Difference:** CiteDCG focuses on *cited* entities; hop_citedcg analyzes *all* search results per iteration

---

## 2. Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ SEVAL Pipeline                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each utterance:                                           │
│                                                                 │
│    ┌─────────────┐     ┌─────────────┐     ┌──────────────┐   │
│    │   Input     │────▶│  cite_dcg   │────▶│ hop_citedcg  │   │
│    │ (EvalData)  │     │ (debug=True)│     │              │   │
│    └─────────────┘     └─────────────┘     └──────────────┘   │
│                              │                    │            │
│                              ▼                    ▼            │
│                        CiteDCGOutput       HopCiteDCGOutput    │
│                        (with debug)         (per-hop avgs)     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Aggregation (by SEVAL):                                       │
│    - Average hop_citedcg scores across all utterances          │
│    - Compare control vs treatment                              │
│    - Statistical significance (t-test, p-value)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Input/Output Contract

### Input (`HopCiteDCGInput`)

```protobuf
message HopCiteDCGInput {
    // CiteDCG debug output containing AllSearchResults with labels
    string cite_dcg_debug_control = 1;    // JSON: {control: {AllSearchResults: {...}}}
    string cite_dcg_debug_treatment = 2;  // JSON: {treatment: {AllSearchResults: {...}}}
    
    // Optional: Custom @K values (default: [3, 5])
    repeated int32 top_k_values = 3;
    
    // Optional: Include all results average (default: true)
    bool include_all_results = 4;
}
```

### Output (`HopCiteDCGOutput`)

```protobuf
message HopCiteDCGOutput {
    string Id = 1;
    
    // Per-hop metrics for control and treatment
    repeated HopMetrics control_hops = 2;
    repeated HopMetrics treatment_hops = 3;
    
    // Summary statistics across all hops
    HopSummary control_summary = 4;
    HopSummary treatment_summary = 5;
}

message HopMetrics {
    int32 hop = 1;                      // Hop/iteration number (0, 1, 2, ...)
    int32 num_results = 2;              // Total results in this hop
    int32 num_labeled = 3;              // Results with valid labels
    
    float avg_all = 4;                  // Average label over ALL results
    float avg_at_3 = 5;                 // Average label over top 3 results
    float avg_at_5 = 6;                 // Average label over top 5 results
    
    // Optional: additional @K values if configured
    repeated TopKScore additional_top_k = 7;
}

message TopKScore {
    int32 k = 1;
    float avg_at_k = 2;
}

message HopSummary {
    int32 total_hops = 1;               // Number of hops
    int32 total_results = 2;            // Total results across all hops
    float overall_avg = 3;              // Average across all hops and results
}
```

---

## 4. Calculation Logic

### Per-Hop Calculation

```python
def calculate_hop_metrics(all_search_results: dict, top_k_values: list[int] = [3, 5]) -> list[HopMetrics]:
    """
    Calculate CiteDCG averages per hop.
    
    Args:
        all_search_results: Dict keyed by hop number ("0", "1", ...) 
                           with CiteDCGLLMLabel on each entity
        top_k_values: List of K values for top-K averaging
    
    Returns:
        List of HopMetrics, one per hop
    """
    hop_metrics = []
    
    for hop_id in sorted(all_search_results.keys(), key=int):
        hop_data = all_search_results[hop_id]
        
        # Collect all labels from this hop
        labels = []
        for plugin_name, plugin_results in hop_data.items():
            for result_set in plugin_results:
                for entity in result_set.get('Results', []):
                    label = entity.get('CiteDCGLLMLabel')
                    if label is not None:
                        labels.append(label)
        
        # Calculate averages
        num_results = len(labels)
        avg_all = sum(labels) / num_results if num_results > 0 else 0.0
        
        # Top-K averages (assumes results are in rank order)
        top_k_avgs = {}
        for k in top_k_values:
            top_k_labels = labels[:k]
            top_k_avgs[k] = sum(top_k_labels) / len(top_k_labels) if top_k_labels else 0.0
        
        hop_metrics.append({
            'hop': int(hop_id),
            'num_results': num_results,
            'num_labeled': num_results,  # All collected have labels
            'avg_all': avg_all,
            'avg_at_3': top_k_avgs.get(3, 0.0),
            'avg_at_5': top_k_avgs.get(5, 0.0),
        })
    
    return hop_metrics
```
---

## 5. File Structure

```
cometdefinition/metrics/
    └── hop_citedcg/
        ├── __init__.py
        ├── hop_citedcg.proto              # Protobuf definitions
        ├── hop_citedcg_pb2.py             # Generated (after protoc)
        ├── config/
        │   └── metric_spec.yaml           # Metric configuration
        ├── logic/
        │   └── metric_logic.py            # Main implementation
        ├── README.md                      # Documentation
        └── CHANGELOG.md                   # Version history
```

---

## 6. Configuration

### metric_spec.yaml

```yaml
name: "hop_citedcg"
description: "Per-hop average CiteDCG scores for search results"
version: 1.0.0
author: comet@microsoft.com
lifecycle: development
icm_team: "MSAI User Understanding/BizChatExperimentationDRI"
ado_template: "https://aka.ms/comet-metric-bug"

# No LLM needed - pure computation from citeDCG output
# llm: (not needed)
```

---

## 7. SEVAL Integration

### How SEVAL Will Use This Metric

1. **Configuration:** SEVAL pipeline configured to run `hop_citedcg` after `cite_dcg`
2. **Input:** SEVAL passes citeDCG's debug output to hop_citedcg
3. **Output:** hop_citedcg returns per-hop metrics for each utterance
4. **Aggregation:** SEVAL averages metrics across utterances automatically

### Expected SEVAL Output (Example)

| Metric | Control | Treatment | Diff | P-value |
|--------|---------|-----------|------|---------|
| hop_0_avg_all | 2.45 | 2.67 | +0.22 | 0.03 |
| hop_0_avg_at_3 | 2.89 | 3.12 | +0.23 | 0.02 |
| hop_0_avg_at_5 | 2.71 | 2.95 | +0.24 | 0.01 |
| hop_1_avg_all | 2.12 | 2.34 | +0.22 | 0.05 |
| hop_1_avg_at_3 | 2.45 | 2.78 | +0.33 | 0.01 |
| hop_1_avg_at_5 | 2.33 | 2.61 | +0.28 | 0.02 |

---

## 8. Edge Cases

| Case | Handling |
|------|----------|
| No results in a hop | Output `avg = 0.0`, `num_results = 0` |
| Fewer than K results for @K | Average over available results |
| Missing CiteDCGLLMLabel | Skip entity, count as unlabeled |
| Empty citeDCG debug | Return empty hop list, log warning |
| Invalid JSON input | Raise validation error |


---

## Appendix A: Sample Data Structure

### Input: CiteDCG Debug Output

```json
{
  "control": {
    "AllSearchResults": {
      "0": {
        "search_enterprise_files": [
          {
            "Results": [
              {"Id": "file1", "Rank": 1, "CiteDCGLLMLabel": 3, "Type": "File"},
              {"Id": "file2", "Rank": 2, "CiteDCGLLMLabel": 2, "Type": "File"},
              {"Id": "file3", "Rank": 3, "CiteDCGLLMLabel": 4, "Type": "File"}
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
              {"Id": "file4", "Rank": 1, "CiteDCGLLMLabel": 3, "Type": "File"}
            ]
          }
        ]
      }
    }
  },
  "treatment": {
    "AllSearchResults": {
      "0": {
        "search_enterprise_files": [
          {
            "Results": [
              {"Id": "file1", "Rank": 1, "CiteDCGLLMLabel": 4, "Type": "File"},
              {"Id": "file5", "Rank": 2, "CiteDCGLLMLabel": 3, "Type": "File"}
            ]
          }
        ]
      }
    }
  }
}
```

### Output: HopCiteDCGOutput

```json
{
  "Id": "utterance_123",
  "control_hops": [
    {
      "hop": 0,
      "num_results": 5,
      "num_labeled": 5,
      "avg_all": 2.4,
      "avg_at_3": 3.0,
      "avg_at_5": 2.4
    },
    {
      "hop": 1,
      "num_results": 1,
      "num_labeled": 1,
      "avg_all": 3.0,
      "avg_at_3": 3.0,
      "avg_at_5": 3.0
    }
  ],
  "treatment_hops": [
    {
      "hop": 0,
      "num_results": 2,
      "num_labeled": 2,
      "avg_all": 3.5,
      "avg_at_3": 3.5,
      "avg_at_5": 3.5
    }
  ],
  "control_summary": {
    "total_hops": 2,
    "total_results": 6,
    "overall_avg": 2.5
  },
  "treatment_summary": {
    "total_hops": 1,
    "total_results": 2,
    "overall_avg": 3.5
  }
}
```

---

## Appendix B: Code Style Requirements

Per project guidelines (from `copilot-instructions.md`):

```python
# hop_citedcg/logic/metric_logic.py
#
# Copyright (c) Microsoft Corporation. All rights reserved.

"""
Hop CiteDCG metric implementation.

Author: [Your Name]
Created: December 2025
Description: Calculates per-hop average CiteDCG scores for search results
             using CiteDCGLLMLabel from citeDCG debug output.
"""
```

- Use type hints for all parameters and return values
- Follow PEP 8 style guide
- Include docstrings for all classes and methods
- Use snake_case for functions and variables
