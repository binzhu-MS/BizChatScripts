# Transform SEVAL to LLM NDCG - Deduplication Implementation

## Overview

The `transform_seval_with_dedup.py` script transforms SEVAL raw data files into the llm_ndcg input format. This document describes the deduplication logic that prevents duplicate results from being extracted when the same data appears in multiple formats within the SEVAL data structure.

## Problem Statement

SEVAL raw data files store search results in two formats within each tool invocation:

1. **Direct format**: `toolInvocation.processedResult` - A single JSON string containing all results
2. **Batched format**: `toolInvocation.batchedQueries[].processedResult` - Results split by query/domain

When both formats exist in the same tool invocation, they often contain **overlapping results**. The original transformer extracted from both formats, causing duplicates in the output.

### Analysis Findings

Analysis of SEVAL raw data revealed:
- **24.5%** of files have both direct and batched formats
- When both exist, direct results are typically a **subset** of batched results
- Overlapping results share the same `reference_id` (e.g., "turn1search1")
- **100%** of direct `reference_id`s overlap with batched `reference_id`s

## Deduplication Scope

The deduplication must preserve results that are legitimately different while eliminating true duplicates:

| Scenario | Should Deduplicate? |
|----------|---------------------|
| Same result in different **iterations** | ❌ NO - Keep both |
| Same result in different **tool invocations** | ❌ NO - Keep both |
| Same result from same tool, different **queries** | ❌ NO - Keep both |
| Same result in both direct AND batched formats (same invocation) | ✅ YES - Keep one |

## Implementation Strategy

### Step 1: Analyze reference_id Usage

Before processing, the code analyzes `reference_id` patterns in both formats:

```python
# Collect ref_ids from direct format
direct_ref_ids: Dict[str, int] = defaultdict(int)  # ref_id -> count
for result in direct_results:
    ref_id = result.get("reference_id", "")
    if ref_id:
        direct_ref_ids[ref_id] += 1

# Collect ref_ids from batched format
batched_ref_ids: Dict[str, int] = defaultdict(int)  # ref_id -> count
for batch_query in batched_queries:
    for result in batch_results:
        ref_id = result.get("reference_id", "")
        if ref_id:
            batched_ref_ids[ref_id] += 1
```

### Step 2: Check for reference_id Reuse

The code checks if any `reference_id` appears multiple times within each format:

```python
direct_has_reuse = any(count > 1 for count in direct_ref_ids.values())
batched_has_reuse = any(count > 1 for count in batched_ref_ids.values())
```

### Step 3: Apply Appropriate Dedup Strategy

#### Case A: No Reuse (Simple Case)

When `reference_id` is unique within each format:
- Use a single `seen_ref_ids` set for the entire invocation
- Process direct format first, add all ref_ids to the set
- Process batched format, skip results whose ref_id is already seen

```python
seen_ref_ids: Set[str] = set(direct_ref_ids.keys())

# Process direct format - all results included
results = transform_results_list(direct_results)

# Process batched format - skip duplicates
for batch_query in batched_queries:
    for result in batch_results:
        ref_id = result.get("reference_id", "")
        if ref_id and ref_id in seen_ref_ids:
            continue  # Skip duplicate
        seen_ref_ids.add(ref_id)
        filtered_results.append(transform(result))
```

#### Case B: Reuse Exists (Complex Case)

When `reference_id` is reused within a format (e.g., same ref_id in multiple batched queries for different domains):
- Find overlapping ref_ids between direct and batched formats
- Only deduplicate results whose ref_id exists in BOTH formats

```python
all_direct_ref_ids = set(direct_ref_ids.keys())
all_batched_ref_ids = set(batched_ref_ids.keys())
overlapping_ref_ids = all_direct_ref_ids & all_batched_ref_ids

# Process direct format - all results included
results = transform_results_list(direct_results)

# Process batched format - skip only overlapping ref_ids
for result in batch_results:
    ref_id = result.get("reference_id", "")
    if ref_id and ref_id in overlapping_ref_ids:
        continue  # Skip - this is a duplicate from direct
    filtered_results.append(transform(result))
```

## Data Flow Diagram

```
SEVAL Raw Data File
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Tool Invocation                                  │
│  ┌─────────────────────┐  ┌────────────────────┐  │
│  │ processedResult     │  │ batchedQueries[]   │  │
│  │ (Direct Format)     │  │ (Batched Format)   │  │
│  │                     │  │                    │  │
│  │ results: [          │  │ [0]: domain=emails │  │
│  │   {ref_id: "A"},    │  │      results: [A,B]│  │
│  │   {ref_id: "B"},    │  │ [1]: domain=files  │  │
│  │   {ref_id: "C"}     │  │      results: [C,D]│  │
│  │ ]                   │  │ [2]: domain=chats  │  │
│  │                     │  │      results: [E,F]│  │
│  └─────────────────────┘  └────────────────────┘  │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Step 1: Collect reference_ids                    │
│                                                   │
│  direct_ref_ids = {A, B, C}                       │
│  batched_ref_ids = {A, B, C, D, E, F}             │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Step 2: Check for reuse                          │
│                                                   │
│  direct_has_reuse = False (each appears once)     │
│  batched_has_reuse = False (each appears once)    │
└───────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Step 3: Apply dedup (Simple Case)                │
│                                                   │
│  seen_ref_ids = {A, B, C} (from direct)           │
│                                                   │
│  Direct results: [A, B, C] → all included         │
│  Batched results:                                 │
│    [0]: A, B → SKIPPED (in seen)                  │
│    [1]: C → SKIPPED, D → INCLUDED                 │
│    [2]: E, F → INCLUDED                           │
│                                                   │
│  Final: [A, B, C, D, E, F] (no duplicates)        │
└───────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Process Direct Format First

Direct results are processed before batched results. This ensures:
- Direct results are always included
- Batched results are checked against what was already extracted

### 2. reference_id as Dedup Key

The `reference_id` field (e.g., "turn1search1") uniquely identifies a result within a tool invocation. It's used as the dedup key because:
- It's consistent across both formats for the same result
- It's assigned at search time, not post-processing
- Our analysis confirmed 100% overlap correlation

### 3. Conservative Dedup in Complex Case

When `reference_id` is reused within a format, we use intersection-based dedup:
- Only skip results that appear in BOTH formats
- This prevents accidentally removing legitimate results

### 4. Per-Invocation Scope

The `seen_ref_ids` set is reset for each tool invocation:
- Different tool invocations may reuse the same `reference_id`
- Different iterations are processed independently
- This preserves all legitimate results across invocations

## Statistics Tracking

The implementation tracks deduplication statistics:

```python
diagnostic_info = {
    "results_before_dedup": 0,    # Total results encountered
    "results_after_dedup": 0,     # Results after dedup
    "duplicates_avoided": 0,      # Duplicates removed
}
```

These stats are aggregated and reported:

```
DEDUPLICATION STATISTICS
========================
Results before dedup: 1000
Results after dedup:  750
Duplicates avoided:   250
Reduction percentage: 25.0%
```

## Usage

```bash
python transform_seval_with_dedup.py transform \
    --input_dir "path/to/seval_raw_data" \
    --output_file "llm_ndcg_input.jsonl" \
    --max_pairs 100 \
    --threads 8
```

## Testing

To verify the deduplication works correctly:

1. **Check reference_id uniqueness**:
   ```bash
   python check_ref_id_reuse_simple.py
   ```

2. **Analyze cross-array duplicates**:
   ```bash
   python verify_cross_array_duplicates.py
   ```

3. **Run transformer and check stats**:
   ```bash
   python transform_seval_with_dedup.py transform \
       --input_dir "path/to/data" \
       --output_file "output.jsonl"
   ```
   
   Review the "DEDUPLICATION STATISTICS" section in output.

## Future Considerations

1. **New Result Formats**: If SEVAL introduces new result formats, the dedup logic may need updates to handle them.

2. **reference_id Changes**: If the `reference_id` assignment logic changes, the dedup assumptions should be re-validated.

3. **Performance**: For very large files, the two-pass approach (analyze then process) could be optimized to single-pass if needed.

## Related Files

- `transform_seval_with_dedup.py` - Main transformer with dedup
- `check_ref_id_reuse_simple.py` - Verify reference_id uniqueness
- `verify_cross_array_duplicates.py` - Analyze duplicate patterns
- `verify_reference_id_dedup.py` - Detailed reference_id analysis
- `duplicate_analysis.py` - Analyze duplicates in transformed output
