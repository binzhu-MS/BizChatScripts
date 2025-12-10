# SEVAL Implementation Guide: Extraction and Processing Methods

**Last Updated**: December 10, 2025  
**Purpose**: Technical implementation details for extracting CiteDCG scores and conversation details from SEVAL data

> **Note**: For data structure specifications and matching principles, see [SEVAL_Complete_Guide.md](SEVAL_Complete_Guide.md).

---

## Table of Contents

1. [Overview](#overview)
2. [Approach 1: Legacy Separate Extraction](#approach-1-legacy-separate-extraction)
3. [Approach 2: Unified DCG Extraction (Recommended)](#approach-2-unified-dcg-extraction-recommended)
4. [Key Modules and Functions](#key-modules-and-functions)
5. [CLI Commands](#cli-commands)
6. [Data Flow Diagrams](#data-flow-diagrams)
7. [Implementation Details](#implementation-details)
8. [Statistics and Plotting](#statistics-and-plotting)

---

## Overview

### Two Extraction Approaches

| Aspect           | Legacy (Separate)              | Unified (Recommended)              |
| ---------------- | ------------------------------ | ---------------------------------- |
| **Data Sources** | Conversation files + DCG files | DCG files only                     |
| **Steps**        | Extract → Extract → Merge      | Extract only                       |
| **Complexity**   | High (matching logic)          | Low                                |
| **Speed**        | Slower                         | Faster                             |
| **Key Insight**  | N/A                            | DCG files contain `EvaluationData` |

### Key Discovery

Raw DCG files (`results.json`) contain the **full `EvaluationData` structure** including:
- `turnData[]` with all conversation turns
- `orchestrationIterations` with search details
- `userInput` for each turn
- All hop and search result information

This means we can extract **both** conversation details AND CiteDCG scores from a single source.

---

## Approach 1: Legacy Separate Extraction

### Overview

The legacy approach uses separate extraction from two data sources, then merges them.

### Workflow

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│  Raw Conversation Files         │      │  Raw DCG Files                  │
│  (*_sydney_response_*.json)     │      │  (results.json)                 │
└───────────────┬─────────────────┘      └───────────────┬─────────────────┘
                │                                        │
                ▼                                        ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│  Step 1: Extract Conversations  │      │  Step 2: Extract CiteDCG        │
│  seval_batch_processor.py       │      │  get_seval_metrics.py           │
│  extract_conversations()        │      │  extract_per_result_citedcg()   │
└───────────────┬─────────────────┘      └───────────────┬─────────────────┘
                │                                        │
                ▼                                        ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│  {job_id}_conversation_details/ │      │  {job_id}_citedcg/              │
│  Per-conversation JSON files    │      │  CiteDCG scores JSONL           │
└───────────────┬─────────────────┘      └───────────────┬─────────────────┘
                │                                        │
                └────────────────┬───────────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────┐
                │  Step 3: Merge Data             │
                │  merge_seval_results.py         │
                │  merge_conversation_citedcg()   │
                └───────────────┬─────────────────┘
                                │
                                ▼
                ┌─────────────────────────────────┐
                │  {job_id}_conversation_w_       │
                │  citedcg_details/               │
                │  Merged conversation + scores   │
                └───────────────┬─────────────────┘
                                │
                                ▼
                ┌─────────────────────────────────┐
                │  Step 4: Build Utterance Details│
                │  build_utterance_details_with_  │
                │  top_k()                        │
                └───────────────┬─────────────────┘
                                │
                                ▼
                ┌─────────────────────────────────┐
                │  Step 5: Calculate Statistics   │
                │  & Generate Plots               │
                └─────────────────────────────────┘
```

### Key Functions

**Step 1: Conversation Extraction**
```python
# seval_batch_processor.py
extract_conversations(
    job_id="133560",
    experiment="both",
    raw_data_dir="seval_data/{job_id}_scraping_raw_data_output",
    output_dir="results/{job_id}_conversation_details"
)
```

**Step 2: CiteDCG Extraction**
```python
# get_seval_metrics.py
extract_per_result_citedcg(
    job_id="133560",
    experiment="control",
    input_file="seval_data/{job_id}_metrics/.../results.json",
    output_file="results/{job_id}_citedcg/{job_id}_citedcg_scores_control.json"
)
```

**Step 3: Merge**
```python
# merge_seval_results.py
merge_conversation_citedcg(
    conversation_file="..._conv_details.json",
    citedcg_file="..._citedcg_scores.json",
    output_file="..._merged.json"
)
```

**Step 4: Build Utterance Details**
```python
# merge_seval_results.py
build_utterance_details_with_top_k(
    merged_dir="results/{job_id}_conversation_w_citedcg_details/control",
    top_k_list=[1, 3, 5],
    output_file="results/{job_id}_utterance_details/control_utterance_details.json",
    experiment="control"
)
```

### Pros and Cons

**Pros:**
- Works with any data source combination
- Can use conversation files without DCG data
- Provides detailed conversation structure

**Cons:**
- Complex merging logic
- Multiple extraction steps
- Slower processing
- Matching issues possible

---

## Approach 2: Unified DCG Extraction (Recommended)

### Overview

The unified approach extracts everything from raw DCG files, leveraging the embedded `EvaluationData`.

### Workflow

```
┌─────────────────────────────────────────────────────────┐
│  Raw DCG Files (results.json)                           │
│  Contains: AllSearchResults + EvaluationData + Utterance │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Unified Extraction                             │
│  get_seval_metrics.py                                   │
│  extract_conv_details_and_dcg_from_raw_dcgfiles()       │
│                                                         │
│  Extracts from single source:                           │
│  - Conversation details from EvaluationData.turnData    │
│  - CiteDCG scores from AllSearchResults                 │
│  - Utterance, hop info, query strings                   │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  {job_id}_unified_hop_citedcg_scores/                   │
│  {job_id}_{experiment}_unified_citedcg.json             │
│                                                         │
│  Contains per-utterance:                                │
│  - conversation_id, utterance text                      │
│  - has_cite_dcg_scores flag                             │
│  - searches[] with hop, query, results[]                │
│  - Each result has CiteDCGLLMLabel                      │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: Build Utterance Details                        │
│  _build_utterance_details_from_unified()                │
│                                                         │
│  Produces same format as legacy approach:               │
│  - Per-utterance hop-level score averages               │
│  - Compatible with existing plotting functions          │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: Find Paired Utterances (if both experiments)   │
│  find_paired_utterances_with_scores()                   │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4: Calculate Statistics & Generate Plots          │
│  generate_plot_statistics_from_utterance_details()      │
└─────────────────────────────────────────────────────────┘
```

### Key Function

```python
# seval_batch_processor.py
process_unified_dcg_with_statistics_plots(
    job_id="133560",
    experiment="both",      # "control", "treatment", or "both"
    top_k_list="1,3,5",     # Comma-separated k values
    num_threads=8,
    output_base_dir="results",
    verbose=False
)
```

### What the Unified Extraction Outputs

**Per-record structure in unified output:**
```json
{
  "conversation_id": "uuid-string",
  "utterance": "user query text",
  "has_cite_dcg_scores": true,
  "num_turns": 1,
  "searches": [
    {
      "hop": "1",
      "plugin_name": "office365_search_files",
      "query_string": "search query",
      "result_count": 10,
      "results": [
        {
          "reference_id": "Turn1Search1",
          "CiteDCGLLMLabel": 2.4,
          "ResultType": "File",
          "Title": "Document Title"
        }
      ]
    }
  ]
}
```

### Pros and Cons

**Pros:**
- Single data source
- No merging complexity
- Faster processing
- Data already aligned (no matching issues)
- Simpler code path

**Cons:**
- Requires DCG files with EvaluationData
- Won't work with conversation-only scenarios

---

## Key Modules and Functions

### get_seval_metrics.py

| Function                                           | Purpose                                                                                           |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `extract_conv_details_and_dcg_from_raw_dcgfiles()` | **Unified extraction** - extracts both conversation details and CiteDCG scores from raw DCG files |
| `extract_per_result_citedcg()`                     | Legacy CiteDCG extraction                                                                         |
| `extract_conv_details_and_dcg_from_raw()`          | Batch wrapper for unified extraction                                                              |

### seval_batch_processor.py

| Function                                      | Purpose                                                      |
| --------------------------------------------- | ------------------------------------------------------------ |
| `process_unified_dcg_with_statistics_plots()` | **Main workflow** - end-to-end unified processing with plots |
| `extract_unified_dcg_batch()`                 | Batch extraction using unified approach                      |
| `process_seval_job_with_statistics_plots()`   | Legacy workflow using separate extraction                    |
| `extract_conversations()`                     | Legacy conversation extraction                               |
| `_build_utterance_details_from_unified()`     | Convert unified output to utterance details format           |

### merge_seval_results.py

| Function                               | Purpose                                     |
| -------------------------------------- | ------------------------------------------- |
| `merge_conversation_citedcg()`         | Legacy - merge conversation with CiteDCG    |
| `build_utterance_details_with_top_k()` | Build utterance details from merged data    |
| `find_paired_utterances_with_scores()` | Find utterances present in both experiments |

### seval_plotting.py

| Function                                            | Purpose                                     |
| --------------------------------------------------- | ------------------------------------------- |
| `generate_plot_statistics_from_utterance_details()` | Calculate statistics from utterance details |
| `generate_control_treatment_comparison_plots()`     | Generate comparison plots                   |

---

## CLI Commands

### Unified Approach (Recommended)

```bash
# End-to-end processing with statistics and plots
python seval_batch_processor.py process_unified_dcg_with_statistics_plots \
    --job_id=133560 \
    --experiment=both \
    --top_k_list=1,3,5 \
    --num_threads=8

# Just extract unified DCG data
python seval_batch_processor.py extract_unified_dcg_batch \
    --job_id=133560 \
    --experiment=both \
    --output_dir=results/133560_unified_hop_citedcg_scores
```

### Legacy Approach

```bash
# Full legacy workflow
python seval_batch_processor.py process_seval_job_with_statistics_plots \
    --job_id=133560 \
    --experiment=both \
    --top_k_list=1,3,5 \
    --threads=8
```

---

## Data Flow Diagrams

### Unified Extraction Data Flow

```
Raw DCG File (results.json)
    │
    ├── ConversationId ──────────────────────► conversation_id
    ├── Utterance ───────────────────────────► utterance
    │
    ├── AllSearchResults
    │   └── [hop_number]
    │       └── [plugin_name]
    │           └── [search_index]
    │               ├── PluginInvocation ────► query_string (parsed)
    │               └── Results[]
    │                   ├── CiteDCGLLMLabel ─► CiteDCGLLMLabel
    │                   ├── ReferenceId ─────► reference_id
    │                   └── Type ────────────► ResultType
    │
    └── EvaluationData
        └── turnData[]
            ├── userInput ───────────────────► (validation)
            └── orchestrationIterations[] ───► hop structure
```

### Score Aggregation Flow

```
Per-result scores                    Per-hop averages              Per-utterance
┌──────────────┐                    ┌─────────────┐               ┌─────────────┐
│ result 1: 2.4│                    │ hop 1:      │               │             │
│ result 2: 1.8│ ─► avg_all ──────► │  avg_all    │               │ utterance   │
│ result 3: 2.1│                    │  avg_top_k  │ ─► group ───► │ details     │
│ result 4: 2.6│ ─► avg_top_k ────► │  count: 4   │   by hop      │ with hops   │
└──────────────┘                    └─────────────┘               └─────────────┘
```

---

## Implementation Details

### How Unified Extraction Works

The `extract_conv_details_and_dcg_from_raw_dcgfiles()` function:

1. **Reads raw DCG file** (JSONL format, one conversation per line)

2. **For each conversation**, extracts:
   - `conversation_id` from `ConversationId`
   - `utterance` from `Utterance`
   - Hop and search data from `AllSearchResults`
   - Query strings using the **query retrieval priority** (see below)

3. **Groups results by hop**, preserving:
   - Hop number (from AllSearchResults key)
   - Plugin name
   - Query string
   - All results with their CiteDCGLLMLabel scores

4. **Tracks score availability**:
   - `has_cite_dcg_scores = True` if any result has CiteDCGLLMLabel

### Query String Retrieval (Critical for Web Search)

**Problem**: `search_web` results in `AllSearchResults` often have **empty `PluginInvocation`** strings, making it impossible to extract the query directly.

**Solution**: Use a multi-level fallback approach to retrieve queries from `EvaluationData`:

```
Query Retrieval Priority:
1. PluginInvocation (from AllSearchResults) ─────► Direct parsing if available
                          │
                          ▼ (if empty or missing)
2. Block-to-Query Map ───────────────────────────► Match by (turn, plugin, block_index)
                          │
                          ▼ (if no match)
3. First-Ref-to-Query Map ───────────────────────► Match by first reference_id
```

**How Each Fallback Works:**

1. **PluginInvocation** (Primary):
   - Parse the `PluginInvocation` string to extract query
   - Works for most `office365_search_*` plugins
   - Often empty for `search_web`

2. **Block-to-Query Map** (`_build_block_to_query_map`):
   - Built from `EvaluationData.turnData[].orchestrationIterations[].modelActions[].toolInvocations[].batchedQueries[]`
   - Maps `(turn_key, plugin_name, block_index)` → `query_string`
   - Uses position-based matching within each (turn, plugin) combination

3. **First-Ref-to-Query Map** (`_build_first_ref_to_query_map`):
   - Built from `EvaluationData.turnData[].orchestrationIterations[].modelActions[].toolInvocations[].batchedQueries[]`
   - For each `batchedQuery`, extracts:
     - Query from `arguments` field
     - First `reference_id` from `processedResult` (WebPages, News, etc.)
   - Maps `first_reference_id` → `query_string`
   - Used as final fallback when block-index matching fails

**Code Example:**
```python
# Priority 1: Try PluginInvocation
query_from_invocation = parse_plugin_invocation(plugin_invocation)

# Priority 2: Try block-index alignment
if not query_from_invocation:
    block_query_key = (turn_key, normalized_plugin, block_index)
    block_query = block_to_query.get(block_query_key, '')

# Priority 3: Try first reference_id lookup
if not query_from_invocation and not block_query:
    first_ref = results_list[0].get('reference_id', '')
    first_ref_query = first_ref_to_query.get(first_ref, '')

# Use best available query
query_string = query_from_invocation or block_query or first_ref_query
```

### Building Utterance Details from Unified Data

The `_build_utterance_details_from_unified()` function:

1. **For each record**, iterates through searches
2. **Groups scores by hop number**
3. **For utterances WITH scores**:
   - Only creates hop entries for hops that have actual CiteDCGLLMLabel values
   - Calculates avg_all_scores and avg_topk_scores
4. **For utterances WITHOUT scores**:
   - Creates hop entries for all hops seen in searches (for proper counting)
   - Marks as `is_empty: True`

### Statistics Calculation

The plotting function `generate_plot_statistics_from_utterance_details()`:

- **Per-hop averages**: Average over utterances with scores **at that specific hop**
- **Utterances with scores**: Count of utterances that have at least one score at this hop
- **Utterances without scores**: Count of utterances that have this hop but no scores

**Important**: Averages are calculated ONLY over utterances with scores. Utterances without scores do not contribute to the average.

---

## Statistics and Plotting

### What Gets Plotted

1. **Hop Index Plot** (`*_comparison_by_hop_index.png`)
   - Row 1: Average all scores by hop (control vs treatment)
   - Row 2: Average top-k scores by hop
   - Row 3: Utterance counts (three curves):
     - 🟢 **With Scores at This Hop**: Utterances that have CiteDCG scores at this specific hop
     - 🟡 **Scores Elsewhere (Not Here)**: Utterances that have scores at some hop, but not at this hop. Calculated as: `total_with_any_scores - with_scores_at_this_hop`
     - 🔴 **No Scores Anywhere**: Utterances with no scores at any hop (constant horizontal line)

2. **Hop Sequence Plot** (`*_comparison_by_hop_sequence.png`)
   - Similar to hop index, but using hop sequence (non-empty hop order)

3. **Score Distribution** (`*_score_distribution.png`)
   - Histograms of score distributions

4. **Summary Statistics** (`*_statistics_summary.json`)
   - JSON file with all calculated statistics

### Output Files

```
results/{job_id}_unified_statistics_plots/
├── {job_id}_comparison_by_hop_index.png
├── {job_id}_comparison_by_hop_sequence.png
├── {job_id}_score_distribution_control.png
├── {job_id}_score_distribution_treatment.png
├── {job_id}_statistics_summary.json
└── ...
```

---

## Troubleshooting

### Common Issues

**Issue**: "Without scores" count doesn't match summary  
**Cause**: Hop entries only created for hops with actual scores  
**Solution**: Use unified extraction which properly tracks all hops

**Issue**: Missing CiteDCGLLMLabel in results  
**Cause**: Result was not evaluated by DCG system  
**Solution**: This is expected - not all results get scored

**Issue**: Empty query string for web search results  
**Cause**: `search_web` entries in `AllSearchResults` often have empty `PluginInvocation`  
**Solution**: The extraction uses a 3-level fallback:
1. Parse `PluginInvocation` (if available)
2. Match by block-index from `EvaluationData.turnData` batchedQueries
3. Match by first `reference_id` from batchedQueries processedResult

**Issue**: Query string extraction still fails after fallbacks  
**Cause**: Misalignment between AllSearchResults and EvaluationData structures  
**Solution**: Check if the conversation has unusual structure; may need manual inspection

**Issue**: Result count mismatch between conversation and CiteDCG data  
**Cause**: SEVAL extraction bug where result counts don't match between sources  
**Behavior**: When counts don't match, the merge logic applies a workaround:

1. Sort CiteDCG results by score (descending) - so top scores are first
2. Assign scores by position up to the available CiteDCG count
3. **When conversation has MORE results than CiteDCG**:
   - First N conversation results get the top N CiteDCG scores
   - Remaining conversation results get **no score** (counted as "without scores")
4. **When CiteDCG has MORE results than conversation**:
   - Only the top N CiteDCG scores are used (where N = conversation count)
   - Extra CiteDCG scores are ignored

**Code location**: `merge_seval_results.py` lines 637-663

```python
# WORKAROUND: Handle result count mismatch
if len(search_results) != len(query_results):
    # Sort CiteDCG results by score (descending) to get top scores
    sorted_search_results = sorted(
        search_results,
        key=lambda r: r.get('CiteDCGLLMLabel', 0),
        reverse=True
    )
    # Use top N scores where N = conversation result count
    search_results = sorted_search_results[:len(query_results)]

# Assign scores by position
for idx, result in enumerate(query_results):
    if idx < len(search_results):
        score = search_results[idx].get('CiteDCGLLMLabel')
        if score is not None:
            result['citedcg_score'] = score
    else:
        # No score available for this result
        total_results_without_scores += 1
```

**Note**: This workaround is tracked in the summary report as "Result count mismatch (workaround applied)"

---

## Migration from Legacy to Unified

If you have existing code using the legacy approach:

1. **Replace** `process_seval_job_with_statistics_plots()` with `process_unified_dcg_with_statistics_plots()`

2. **Update output paths**:
   - Old: `{job_id}_conversation_w_citedcg_details/`
   - New: `{job_id}_unified_hop_citedcg_scores/`

3. **Remove** separate conversation extraction steps

4. **Keep** the same plotting functions - output format is compatible

---

**End of Document**
