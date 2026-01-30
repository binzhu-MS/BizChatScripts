# Retrieved Good Gain Metrics — Implementation

## Overview

This document describes the implementation of **Retrieved Good Gain metrics** integrated directly into the `llm_ndcg` metric, controlled by the flight flag `NDCG_RETRIEVED_GOOD_GAIN`.

### Metrics Implemented

| # | Metric | Formula | Description |
|---|--------|---------|-------------|
| 1 | CG@i | $\sum_{k=1}^{i} G_k$ | Cumulative Good Gain |
| 2 | RG@i | $\frac{CG@i}{i}$ | Rate of Good Gain |
| 3 | DCG@i | $\sum_{k=1}^{i} w(k) \cdot G_k$ | Discounted Cumulative Good Gain |
| 4 | DRG@i | $\frac{DCG@i}{i}$ | Discounted Rate of Good Gain |
| 5 | AvgGain_i | $\frac{G_i}{R_i}$ | Average Good Gain per Result at iteration i |
| 6 | RAG@i | $\frac{1}{i} \sum_{k=1}^{i} AvgGain_k$ | Rate of Average Good Gain |
| 7 | DRAG@i | $\frac{1}{i} \sum_{k=1}^{i} w(k) \cdot AvgGain_k$ | Discounted Rate of Average Good Gain |
| 8 | SRE@i | $\frac{GR@i}{R@i}$ | Search Result Efficiency |
| 9 | SRR@i | $\frac{DupR@i}{R@i}$ | Search Result Redundancy |
| 10 | IterationsForAllGoodResults | $\min(k, 100)$ | Iterations needed to find all good results |

Where:
- $G_i$ = total **good** gain at iteration $i$ (sum of gains for good unique results where gain ≥ threshold)
- $R_i$ = **all** results at iteration $i$ (**including duplicates**)
- $GR_i$ = **good unique** results at iteration $i$ (gain ≥ threshold, first occurrence only)
- $Dup_i$ = duplicate results at iteration $i$
- $R@i$, $GR@i$, $DupR@i$ = cumulative counts through iteration $i$
- $w(k) = \frac{1}{\log_2(k+1)}$ = discount weight

**Note:** $R@i = UR@i + DupR@i$ (total results = unique + duplicates)

---

## Deduplication Strategy

### Challenge

Identifying duplicate search results across iterations is complex because:

1. **Generic ID Instability**: Generic entity IDs (fallback when domain ID unavailable) may change across re-queries for the same document. Domain-specific IDs are more reliable.
2. **URL Variability**: Same document may have different query parameters
3. **Key Collisions**: Different documents may share the same generic ID
4. **Partial Information**: Some entities lack URLs or complete domain IDs

### Solution: Two-Phase Candidate Generation + Verification

We use a **two-phase approach** that provides both high recall (finding potential duplicates) and high precision (avoiding false positives):

```
Phase 1: Candidate Generation (O(1) hash lookup)
    └── Find potential matches via entity ID key OR URL key

Phase 2: Verification (multi-step)
    └── Confirm candidates are truly the same document
```

---

### Phase 1: Candidate Generation

Each entity is indexed by up to **two keys** for hash-based O(1) lookup:

#### 1.1 Entity ID Key

**Domain-Specific ID** (preferred) or **Generic ID** (fallback) — mutually exclusive.

| Entity Type | Domain-Specific ID Format | Fields Required |
|-------------|--------------------------|-----------------|
| File | `id:File:fileid\|uniqueid\|webid\|siteid` | All 4 fields |
| Event | `id:Event:eventid.id\|originalid` | Both fields |
| Email | `id:Email:source.emailmessage.url` | URL field |
| TeamsMessage | `id:TeamsMessage:itemid.id\|url` | Both fields |
| People | `id:People:emailaddresses[0].address` | Email address |

If domain-specific ID is not extractable (missing fields), fall back to:
- **Generic ID**: `id:generic:{entity_type}:{entity.Id}` or `id:generic:{entity.Id}`

#### 1.2 URL Key

**Aggressive normalization** — strip scheme (http/https), query parameters, and fragments:

| Original URL | Normalized Key |
|--------------|----------------|
| `https://sharepoint.com/doc.docx?version=1` | `url:sharepoint.com\|/doc.docx` |
| `http://sharepoint.com/doc.docx?web=1&preview=true` | `url:sharepoint.com\|/doc.docx` |
| `https://sharepoint.com/doc.docx#section1` | `url:sharepoint.com\|/doc.docx` |

**URL extraction priority**: `entity.Url` → `entity.Source.Url` → `entity.Source.EmailMessage.Url`

#### 1.3 Key Registration

For each entity processed:
1. Compute `(entity_id_key, url_key)` — either may be `None`
2. Look up candidates in `seen_ids[entity_id_key]` and `seen_ids[url_key]`
3. After verification, register both keys (if not duplicate) for future lookups

**Data structure**: `seen_ids: dict[str, list[tuple[entity, verification_data]]]`

---

### Phase 2: Verification

When candidates are found, apply **multi-step verification** to confirm they represent the same document:

#### Step 1: High-Confidence Comparison

**1a. Complete Domain ID Comparison**

If both entities have ALL required fields for their type (per citation_utils.py pattern):
- All fields match → **Same document**
- Any field differs → **Different document**
- Incomplete fields → Proceed to next step

**1b. URL Comparison with Safe Param Filtering**

Compare URLs after removing known safe parameters (view modifiers, not content identifiers):

| Safe Parameters (ignored) |
|---------------------------|
| `web`, `preview`, `d`, `e`, `action`, `cid`, `wdorigin`, `wdprevioussession`, `wdprevioussessionid` |

| Comparison Result | Outcome |
|-------------------|---------|
| Same domain + path + filtered params | **Same document** |
| Different domain or path | **Different document** |
| Differ only in unknown params | Proceed to Step 2 |

#### Step 2: Type-Specific Verification

Apply entity-type-specific rules that leverage domain knowledge about unique identifiers:

##### Email Verification (`verify_email_duplicate`)

| Condition | Result |
|-----------|--------|
| DateTimeReceived matches + From email matches | **Duplicate** |
| DateTimeReceived matches + To list identical | **Duplicate** |
| DateTimeReceived differs | Inconclusive → Step 3 |
| Fields missing | Inconclusive → Step 3 |

**Rationale**: Same datetime + same sender/recipients uniquely identifies an email.

##### Event Verification (`verify_event_duplicate`)

Fields: `Start`, `End`, `OrganizerEmail`, `Invitees`

| Condition | Result |
|-----------|--------|
| Invitees have zero overlap | **Different** (strong rejection) |
| Any of Start/End/OrganizerEmail mismatch | **Different** |
| 3 of 3 fields match (Start, End, Organizer) | **Duplicate** |
| 2 of 3 fields match + strong invitees (≥50% overlap) | **Duplicate** |
| 2 of 3 fields match + weak invitees (<50% overlap) | **Different** |
| < 2 fields available | Inconclusive → Step 3 |

**Rationale**: Zero invitee overlap is a strong signal that meetings are different even if times match. The 3-of-4 logic balances precision with recall.

##### People Verification (`verify_people_duplicate`)

| Condition | Result |
|-----------|--------|
| Alias matches | **Duplicate** (unique per org) |
| Alias differs | **Different** |
| UserPrincipalName matches | **Duplicate** (Azure AD unique) |
| UserPrincipalName differs | **Different** |
| Any EmailAddresses overlap | **Duplicate** (email is unique) |
| EmailAddresses have no overlap | **Different** |
| No unique IDs available | Inconclusive → Step 3 |

**Rationale**: People entities have strong unique identifiers (Alias, UPN, Email) that definitively identify individuals.

#### Step 3: Field-Based Verification (Partial Match + Metadata)

Triggered when: type-specific verification is inconclusive, OR shared domain field exists, OR URLs differ only in unknown params.

**Field Classification:**

| Level | Fields | Description |
|-------|--------|-------------|
| **Strong** | Long snippet (>50 chars) | High confidence content match |
| **Intermediate** | Title, Filename | Medium confidence identifiers |
| **Weak** | Entity type, File type, Author, Org, Project | Low confidence metadata |

**Matching Rules:**
- Any field **mismatch** → **Different document** (immediate rejection)
- With partial ID context (shared domain field or URL param diff):
  - 1+ strong OR 1+ intermediate OR 2+ weak → **Duplicate**
- Without partial ID context:
  - 1+ strong OR 2+ intermediate OR 4+ weak → **Duplicate**

#### Step 4: Minimal Entity Fallback

For entities with **no verifiable information** (no URL, no domain fields, no type-specific fields, no title, no snippet):

- Trust the key match — they wouldn't be candidates without matching a key
- Common for test data or minimal backend responses

---

### Verification Data

For each entity, extract and store verification-relevant fields:

| Field | Source | Purpose |
|-------|--------|---------|
| `source` | `entity.Source` (parsed) | Domain ID field access |
| `url` | Multiple locations | URL comparison |
| `entity_type` | Resolved via `entity_type_utils` | Type matching |
| `file_type` | `source.filetype` or `entity.type` | Subtype matching |
| `title` | `entity.title` | Content comparison |
| `file_name` | `source.filename` | Content comparison |
| `author` | `source.modifiedby` / `label_lastmodifiedby` / `label_authors[0]` | Metadata comparison |
| `snippet` | `entity.snippet` (normalized) | Content comparison |
| `organization`, `project` | `source.organization`, `source.project` | DevOps context |

**Type-Specific Fields** (extracted based on entity_type):

| Entity Type | Field | Source | Purpose |
|-------------|-------|--------|----------|
| Email | `date_time_received` | `entity.DateTimeReceived` | Timestamp matching |
| Email | `from` | `entity.From` | Sender matching |
| Email | `to` | `entity.To` | Recipient list matching |
| Event | `start` | `entity.Start` | Meeting start time |
| Event | `end` | `entity.End` | Meeting end time |
| Event | `organizer_email` | `entity.OrganizerEmail` | Organizer matching |
| Event | `invitees` | `entity.Invitees` | Attendee list matching |
| People | `alias` | `entity.Alias` | Org-unique identifier |
| People | `user_principal_name` | `entity.UserPrincipalName` | Azure AD unique ID |
| People | `email_addresses` | `entity.EmailAddresses` | Email addresses list |
---

### Domain ID Fields

Fields used for domain-specific ID construction and partial matching:

```
DOMAIN_ID_FIELDS = ["fileid", "uniqueid", "webid", "siteid", "listid"]
```

---

## Files Modified

### 1. Flight Constant

**File:** `cometdefinition/common/flights_constants.py`

```python
# Flight to enable Retrieved Good Gain metrics (CG@i, RG@i, DCG@i, etc.) in llm_ndcg
NDCG_RETRIEVED_GOOD_GAIN = "ndcg-retrieved-good-gain"
```

### 2. Proto Schema

**File:** `cometdefinition/metrics/llm_ndcg/llm_ndcg.proto`

Added new message types:

```protobuf
// Retrieved Good Gain metrics per iteration
message IterationGainMetrics {
    int32 iteration = 1;           // Iteration index (1-based)
    
    // Per-iteration values
    float G_i = 2;                 // Total good gain at this iteration
    int32 R_i = 3;                 // Number of ALL results (including duplicates)
    int32 UR_i = 4;                // Number of unique (new) results
    int32 GR_i = 5;                // Number of good unique results (gain >= threshold)
    int32 Dup_i = 6;               // Number of duplicate results
    float AvgGain_i = 7;           // G_i / R_i at this iteration
    
    // Cumulative values (through iteration i)
    float CG_at_i = 10;            // Cumulative Good Gain
    float RG_at_i = 11;            // Rate of Good Gain
    float DCG_at_i = 12;           // Discounted Cumulative Good Gain
    float DRG_at_i = 13;           // Discounted Rate of Good Gain
    float RAG_at_i = 14;           // Rate of Average Good Gain
    float DRAG_at_i = 15;          // Discounted Rate of Average Good Gain
    float SRE_at_i = 16;           // Search Result Efficiency = GR@i / R@i
    float SRR_at_i = 17;           // Search Result Redundancy = DupR@i / R@i
    
    // Cumulative counts
    int32 R_at_i = 20;             // Total results including duplicates
    int32 UR_at_i = 21;            // Cumulative unique results
    int32 GR_at_i = 22;            // Cumulative unique good results
    int32 DupR_at_i = 23;          // Cumulative duplicates
}

// Summary of Retrieved Good Gain metrics
message RetrievedGoodGainSummary {
    int32 search_iteration_count = 1;  // N = number of iterations where search was invoked
    int32 good_gain_threshold = 2;
    
    // Final cumulative values
    float final_CG = 10;           // CG@N
    float final_RG = 11;           // RG@N
    float final_DCG = 12;          // DCG@N
    float final_DRG = 13;          // DRG@N
    float final_RAG = 14;          // RAG@N
    float final_DRAG = 15;         // DRAG@N
    float final_SRE = 16;          // SRE@N
    float final_SRR = 17;          // SRR@N
    
    // Totals
    int32 total_results = 20;      // R@N (includes duplicates)
    int32 total_unique_results = 21; // UR@N
    int32 total_good_results = 22; // GR@N
    int32 total_duplicates = 23;   // DupR@N
    int32 last_iteration_with_good_results = 24;  // Capped at 100
}

// Container for Retrieved Good Gain metrics (one per arm)
message RetrievedGoodGainMetrics {
    repeated IterationGainMetrics iterations = 1;
    RetrievedGoodGainSummary summary = 2;
}
```

Updated `LLMNDCGOutput`:

```protobuf
message LLMNDCGOutput {
    // ... existing fields ...
    
    // Retrieved Good Gain metrics (only populated when flight is enabled)
    optional RetrievedGoodGainMetrics control_retrieved_good_gain = 10;
    optional RetrievedGoodGainMetrics treatment_retrieved_good_gain = 11;
}
```

### 3. Calculation Logic

**File:** `cometdefinition/metrics/llm_ndcg/logic/metric_logic.py`

**Constants:**

- `DEFAULT_GOOD_GAIN_THRESHOLD = 2` — Gain threshold for "good" results
- `DOMAIN_ID_FIELDS = ["fileid", "uniqueid", "webid", "siteid", "listid"]` — Fields for domain ID matching
- `SAFE_URL_QUERY_PARAMS = {"web", "preview", "d", "e", "action", "cid", "wdorigin", "wdprevioussession", "wdprevioussessionid"}` — View modifiers (safe to ignore)

**Methods (Deduplication):**

| Method | Purpose |
|--------|---------|
| `extract_verification_data(entity)` | Extract verification-relevant fields from entity (includes type-specific fields) |
| `get_dedup_keys(entity, verification_data)` | Generate (entity_id_key, url_key) for candidate lookup |
| `normalize_url_aggressive(url)` | Strip scheme, query params, fragment → `url:domain\|path` |
| `compare_complete_domain_id(src1, src2, entity_type)` | Step 1a: Compare domain ID fields |
| `compare_urls_with_safe_params(url1, url2)` | Step 1b: Compare URLs filtering safe params |
| `verify_email_duplicate(vdata1, vdata2)` | Step 2: Email-specific verification (DateTimeReceived + From/To) |
| `verify_event_duplicate(vdata1, vdata2)` | Step 2: Event-specific verification (Start/End/Organizer/Invitees) |
| `verify_people_duplicate(vdata1, vdata2)` | Step 2: People-specific verification (Alias/UPN/EmailAddresses) |
| `has_any_shared_domain_field(src1, src2)` | Check partial domain field overlap |
| `verify_by_available_fields(vd1, vd2)` | Step 3: Field-based verification with thresholds |
| `verify_same_entity(vd1, vd2, entity1, entity2)` | Orchestrate multi-step verification |

**Methods (Metric Calculation):**

| Method | Purpose |
|--------|---------|
| `group_results_by_iteration(results)` | Group results, apply two-phase deduplication |
| `calculate_discount_weight(k)` | $w(k) = \frac{1}{\log_2(k+1)}$ |
| `calculate_retrieved_good_gain_metrics(results, labels)` | Main calculation for all 10 metrics |

**Modified `execute()`:**

- Checks for `NDCG_RETRIEVED_GOOD_GAIN` flight flag
- Calls `calculate_retrieved_good_gain_metrics()` for control and treatment arms

### 4. Generated Files

**File:** `cometdefinition/metrics/llm_ndcg/llm_ndcg_pb2.py`

Regenerated from proto with new message types.

---

## Files Created

### Unit Tests

**File:** `cometdefinition/tests/test_llm_ndcg_retrieved_good_gain.py`

Comprehensive test suite covering:

- `TestGroupResultsByIteration` - Tests for result grouping and deduplication
- `TestCalculateDiscountWeight` - Tests for discount weight formula
- `TestCalculateRetrievedGoodGainMetrics` - Tests for metric calculations
- `TestFlightIntegration` - Tests for flight flag behavior
- `TestEdgeCases` - Tests for edge cases (empty results, None labels, etc.)

---

## How to Enable

Add the flight flag when creating the metric:

```python
from cometdefinition.metrics import LLMNDCGMetric
import cometdefinition.common.flights_constants as flights_constants

metric = LLMNDCGMetric.from_raw(
    id="my_metric",
    utterance="What is the weather?",
    user_profile={"Name": "User"},
    timestamp="2024-01-01T00:00:00Z",
    all_search_results_control={...},
    all_search_results_treatment={...},
    flights=[flights_constants.NDCG_RETRIEVED_GOOD_GAIN],  # Enable the flight
)

# Execute and get results
result = metric.execute()

# Access Retrieved Good Gain metrics
control_rgg = result["control_retrieved_good_gain"]
treatment_rgg = result["treatment_retrieved_good_gain"]

# Access per-iteration metrics
for iteration in control_rgg["iterations"]:
    print(f"Iteration {iteration['iteration']}:")
    print(f"  CG@i = {iteration['CG_at_i']}")
    print(f"  RG@i = {iteration['RG_at_i']}")
    print(f"  DCG@i = {iteration['DCG_at_i']}")

# Access summary
summary = control_rgg["summary"]
print(f"Final CG@N = {summary['final_CG']}")
print(f"Total good results = {summary['total_good_results']}")
```

---

## How to Run Tests

### Prerequisites

1. **Activate the conda environment:**
   ```powershell
   conda activate comet
   ```

2. **Ensure dependencies are installed:**
   ```powershell
   pip install -e .
   ```

3. **Internal dependencies:** Some tests require internal Microsoft packages (`sydneyevaluation`, etc.). Make sure these are available in your environment.

### Running the Tests

**Run all Retrieved Good Gain tests:**

```powershell
cd c:\working\CopilotMetrics\sources\dev\MetricDefinition
python -m pytest cometdefinition/tests/test_llm_ndcg_retrieved_good_gain.py -v
```

**Run specific test class:**

```powershell
python -m pytest cometdefinition/tests/test_llm_ndcg_retrieved_good_gain.py::TestCalculateRetrievedGoodGainMetrics -v
```

**Run a single test:**

```powershell
python -m pytest cometdefinition/tests/test_llm_ndcg_retrieved_good_gain.py::TestCalculateRetrievedGoodGainMetrics::test_single_iteration_metrics -v
```

**Run with coverage:**

```powershell
python -m pytest cometdefinition/tests/test_llm_ndcg_retrieved_good_gain.py -v --cov=cometdefinition.metrics.llm_ndcg
```

### Regenerating Proto Files

If you modify the proto schema:

```powershell
cd c:\working\CopilotMetrics\sources\dev\MetricDefinition
python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. cometdefinition/metrics/llm_ndcg/llm_ndcg.proto
```

---

## Multi-Turn Handling

Following the existing `llm_ndcg` pattern, **only the last turn is processed** for multi-turn conversations:

```python
if len(eval_data.turnData) > 1:
    logger.log_info("Multi-turn input detected, last turn will be processed.")

# Only processes the LAST turn
return get_turn_search_results(eval_data.turnData[-1])
```

The Retrieved Good Gain metrics operate on **iterations within the last turn**, not across multiple turns. This is handled automatically by `all_search_results_utils.py`.

---

## Output Schema

### Per-Iteration Output

Each iteration in `control_retrieved_good_gain.iterations` contains:

| Field | Type | Description |
|-------|------|-------------|
| `iteration` | int | 1-based iteration index |
| `G_i` | float | Total good gain at this iteration (only gains ≥ threshold, deduplicated) |
| `R_i` | int | Number of ALL results at iteration i (**including duplicates**) |
| `UR_i` | int | Number of unique (new) results |
| `GR_i` | int | Number of good unique results (gain ≥ threshold) |
| `Dup_i` | int | Number of duplicates |
| `AvgGain_i` | float | G_i / R_i (good gain / total results at iteration i, penalizes duplicates) |
| `CG_at_i` | float | Cumulative Good Gain through iteration i |
| `RG_at_i` | float | Rate of Good Gain |
| `DCG_at_i` | float | Discounted Cumulative Good Gain |
| `DRG_at_i` | float | Discounted Rate of Good Gain |
| `RAG_at_i` | float | Rate of Average Good Gain |
| `DRAG_at_i` | float | Discounted Rate of Average Good Gain |
| `SRE_at_i` | float | Search Result Efficiency (GR@i / R@i) |
| `SRR_at_i` | float | Search Result Redundancy (DupR@i / R@i) |
| `R_at_i` | int | Cumulative total results (**including duplicates**) |
| `UR_at_i` | int | Cumulative unique results through iteration i |
| `GR_at_i` | int | Cumulative good unique results through iteration i |
| `DupR_at_i` | int | Cumulative duplicates through iteration i |

### Summary Output

The `control_retrieved_good_gain.summary` contains:

| Field | Type | Description |
|-------|------|-------------|
| `search_iteration_count` | int | N = number of iterations where search was invoked |
| `good_gain_threshold` | int | Threshold used (default: 2) |
| `final_CG` | float | CG@N (final cumulative) |
| `final_RG` | float | RG@N |
| `final_DCG` | float | DCG@N |
| `final_DRG` | float | DRG@N |
| `final_RAG` | float | RAG@N |
| `final_DRAG` | float | DRAG@N |
| `final_SRE` | float | SRE@N (Search Result Efficiency) |
| `final_SRR` | float | SRR@N (Search Result Redundancy) |
| `total_results` | int | Total results (**including duplicates**) = R@N |
| `total_unique_results` | int | Total unique results = UR@N |
| `total_good_results` | int | Total good unique results = GR@N |
| `total_duplicates` | int | Total duplicates = DupR@N |
| `last_iteration_with_good_results` | int | Last iteration with new good results (capped at 100) |

---

## Example Calculation

Given 3 iterations with the following results:

| Iteration | Unique Results | Gains | Duplicates |
|-----------|----------------|-------|------------|
| 1 | 2 | [3, 2] | 0 |
| 2 | 2 | [4, 1] | 1 |
| 3 | 1 | [2] | 2 |

With threshold = 2:

**Iteration 1:**
- $G_1 = 3 + 2 = 5$ (both gains ≥ 2, so both count)
- $R_1 = 2 + 0 = 2$ (all results including dups), $UR_1 = 2$, $GR_1 = 2$, $Dup_1 = 0$
- $R@1 = 2$, $GR@1 = 2$, $DupR@1 = 0$
- $AvgGain_1 = G_1 / R_1 = 5/2 = 2.5$
- $SRE@1 = GR@1 / R@1 = 2/2 = 1.0$ (100% good results)
- $SRR@1 = DupR@1 / R@1 = 0/2 = 0$ (no duplicates)

**Iteration 2:**
- $G_2 = 4$ (only gain 4 counts; gain 1 is below threshold)
- $R_2 = 2 + 1 = 3$ (all results including dups), $UR_2 = 2$, $GR_2 = 1$, $Dup_2 = 1$
- $R@2 = 2 + 3 = 5$, $GR@2 = 3$, $DupR@2 = 1$
- $CG@2 = 5 + 4 = 9$
- $AvgGain_2 = G_2 / R_2 = 4/3 ≈ 1.33$
- $SRE@2 = GR@2 / R@2 = 3/5 = 0.6$ (60% good results)
- $SRR@2 = DupR@2 / R@2 = 1/5 = 0.2$ (20% duplicates)

**Iteration 3:**
- $G_3 = 2$
- $R_3 = 1 + 2 = 3$ (all results including dups), $UR_3 = 1$, $GR_3 = 1$, $Dup_3 = 2$
- $R@3 = 5 + 3 = 8$, $GR@3 = 4$, $DupR@3 = 3$
- $CG@3 = 9 + 2 = 11$
- $AvgGain_3 = G_3 / R_3 = 2/3 ≈ 0.67$
- $SRE@3 = GR@3 / R@3 = 4/8 = 0.5$ (50% good results)
- $SRR@3 = DupR@3 / R@3 = 3/8 = 0.375$ (37.5% duplicates)

**Summary:**
- `total_results = 8` (R@N, includes duplicates)
- `total_unique_results = 5` (UR@N)
- `total_good_results = 4` (GR@N)
- `total_duplicates = 3` (DupR@N)
- `last_iteration_with_good_results = 3`
