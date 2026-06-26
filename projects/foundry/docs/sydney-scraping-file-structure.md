# Sydney Scraping File Structure

This document describes the structure of Sydney scraping files (the raw JSON captured from Sydney/Copilot conversations), covering all three conversation types: **echo-based multi-turn**, **log-based multi-turn**, and **single-turn**. It explains where to extract reasoning results (chain-of-thought, tool calls, and search results).

---

## 1. Overview: Three Conversation Types

| Type | Description | `requests` count | `contentOrigin` prefix |
|---|---|---|---|
| **Echo multi-turn** | Sydney actually processes each turn; bot responses are generated live. Follow-up turns build on real responses. | N (one per turn) | `..._multiturn_echo_control` |
| **Log-based multi-turn** | Prior turns are replayed from a pre-recorded conversation log. Only the **last turn** where Sydney actually generates a new response is "live"; earlier turns use scripted bot replies. | N (one per turn) | `..._multiturn_logbased_control` |
| **Single-turn** | One user utterance, one bot response. No conversation history. | 1 | `..._default_control` |

### How to identify the type

1. **From `contentOrigin`** of the user message: contains `multiturn_echo`, `multiturn_logbased`, or `default`
2. **From `requests` length**: 1 = single-turn; >1 = multi-turn
3. **From the `query.id` field**: contains all turns as JSON -- for log-based, this includes scripted bot replies; for echo, only user messages

---

## 2. Scraping File Structure (Common to All Types)

**Filename pattern:** `{variant}_sydney_response_{conversationId}.json`

### 2.1 Top-Level Structure

```jsonc
{
  "conversation_id": "ebae8320-...",
  "query": {
    "id": "[{\"text\": \"...\", \"author\": \"user\"}, ...]",  // JSON string of all turns
    "segment": "Location based queries",                         // Evaluation segment name
    "query_hash": "e650f45d..."                                  // Hash for dedup
  },
  "requests": [ ... ]          // One entry per turn
}
```

**`query.id` differences by type:**
- **Echo**: Only user messages: `[{"text": "...", "author": "user"}, {"text": "...", "author": "user"}]`
- **Log-based**: User AND bot messages interleaved: `[{"text": "...", "author": "user"}, {"text": "...", "author": "bot"}, {"text": "...", "author": "user"}, ...]`
- **Single-turn**: Single user message: `[{"text": "...", "author": "user"}]`

### 2.2 Request Structure (Each Turn)

```jsonc
{
  "exp_name": "control",                    // Variant name
  "sydney_status_code": 200,                // HTTP status
  "request_id": "8a660151-...",             // UUID for this turn
  "utterance": "What's the location...",    // User input text
  "request_body": {                         // Outgoing request config
    "message": { "author": "user", "text": "...", "timestamp": "...", ... },
    "conversationId": "ebae8320-...",
    "isStartOfSession": true,               // true for first turn only
    "optionsSets": [ ... ],                 // Feature flags
    "plugins": [ ... ],
    // ... other config
  },
  "response_body": { ... },                // Full Sydney response (see Section 3)
  "request_headers": {}                     // Usually empty
}
```

**Note:** `request_body.isStartOfSession` is `true` only for the first turn. Subsequent turns use `false` and rely on the server-side conversation state (echo) or the accumulated messages (log-based).

### 2.3 Response Body Structure

The `response_body` (same structure whether accessed from the scraping file or from a standalone turn file):

```jsonc
{
  "messages": [ ... ],                      // All messages for this turn
  "firstNewMessageIndex": 1,
  "turnState": "Completed",
  "conversationId": "ebae8320-...",
  "requestId": "8a660151-...",
  "conversationExpiryTime": "2026-...",
  "telemetry": {
    "metrics": [ ... ],                     // Per-plugin telemetry
    "startTime": "2026-...",
    "optionsSets": [ ... ]                  // Active flight flags
  },
  "throttling": { ... },                    // Rate limit info
  "result": {
    "value": "Success",                     // "Success" or error
    "message": "...",                       // Final bot response text (quick access)
    "serviceVersion": "1.0.03365.21046",
    "traceId": "..."
  },
  "status_code": 200
}
```

---

## 3. Messages Array

The `messages` array contains all messages for a single turn in chronological order. The composition varies by conversation type and whether tools were invoked.

### 3.1 Common Message Types

These message types appear across all conversation types:

| messageType | contentOrigin | Author | Description |
|---|---|---|---|
| *(none)* | `eval_*_{type}_{variant}` | `user` | User utterance |
| `Internal` | `ModelSelector` | `system` | Model routing decision (e.g. `"meeting_insights_1_default"`) |
| `InvokeAction` | `DeepLeo` | `bot` | Tool invocation signal (text is always `"No content returned"`) |
| *(none)* | `DeepLeo` | `bot` | **Final bot response text** |
| `InternalStorageMetaData` | `internal-storage-metaData-...` | `bot` | Storage metadata |
| `Internal` | `annotation-provider` | `bot` | Entity annotation data |
| `Internal` | `past-enterprise-search-results-metadata` | `system` | Prior-turn search context carried forward |
| **`EvaluationData`** | *(none)* | *(none)* | **Full orchestration trace** (always present, always last or near-last) |

### 3.2 Type-Specific Message Types

| messageType | contentOrigin | Seen In | Description |
|---|---|---|---|
| `Progress` | `ChainOfThoughtSummary` | **Echo** | Reasoning summaries from the search sub-agent (user-visible chain-of-thought) |
| `Progress` | *(empty)* | **Log-based** | Short tool-call progress notes (e.g. `"OK, I'll search for '...'..."`) |
| `Internal` | `DeepLeo` | **Echo** | Condensed search results JSON (`ReferencedSources` + `Sources`) |
| `Internal` | `citation-formatter` | **Echo, Log-based** | Citation index mapping |
| `InternalSearchResult` | `DeepLeo` | **Log-based** | Search results (different format from Echo) |
| `InternalPaginationData` | `EnterpriseSearchExtension` | **Log-based** | Pagination metadata for searches |
| `HintInvocation` | `ExtensibleHintGenerator` | **Log-based, Single-turn** | Extension hint invocation |
| `Internal` | `fetch-location-info` | **Log-based** | Location query results |
| `InternalLoaderMessage` | *(empty)* | **Single-turn** | Loading indicator (`"Working on generating the response."`) |
| *(none)* | `TurnFinalizer` | **Single-turn** | Fallback/error response |

### 3.3 Example Message Sequences

**Echo multi-turn (with search agent):**
```
[0] user input
[1] ModelSelector
[2] InvokeAction (tool call signal)
[3-6] Progress/ChainOfThoughtSummary (reasoning summaries, one per search agent iteration)
[7] Internal/DeepLeo (condensed search results JSON)
[8] bot response
[9] InternalStorageMetaData
[10] citation-formatter
[11] annotation-provider
[12] past-enterprise-search-results-metadata
[13] EvaluationData                        <-- ALWAYS LAST
```

**Log-based multi-turn (with tool calls):**
```
[0] user input
[1] ModelSelector
[2] Progress ("OK, I'll search for '...'...")
[3] HintInvocation
[4-5] Progress + InvokeAction (tool call)
[6] InternalPaginationData
[7] InternalSearchResult
[8-10] more InvokeAction + InternalPaginationData + InternalSearchResult
[11] bot response
[12] InternalStorageMetaData
[13-16] citation-formatter, annotation-provider, past-enterprise-search-results-metadata
[17] EvaluationData                        <-- ALWAYS LAST
```

**Log-based or single-turn (no tool calls):**
```
[0] user input
[1] past-enterprise-search-results-metadata (if multi-turn)
[2] ModelSelector
[3] bot response
[4] InternalStorageMetaData
[5] annotation-provider
[6] past-enterprise-search-results-metadata
[7] EvaluationData                         <-- ALWAYS LAST
```

### 3.4 Quick Extraction from Messages

| What you need | How to find it |
|---|---|
| **User input** | First message with `author == "user"` -> `.text` |
| **Final bot response** | `result.message` at response_body level, OR message with `author == "bot"` and no `messageType` and `contentOrigin == "DeepLeo"` |
| **Reasoning summaries** | Messages with `messageType == "Progress"` |
| **Condensed search results** | (Echo) `messageType == "Internal" && contentOrigin == "DeepLeo"` -> JSON with `ReferencedSources`; (Log-based) `messageType == "InternalSearchResult"` |

---

## 4. EvaluationData -- The Detailed Orchestration Record

The `EvaluationData` message is always present and always the last (or near-last) message in each turn. There is exactly **one** per turn. It contains the richest data about what happened during orchestration.

### 4.1 Top-Level Fields

```jsonc
{
  "evaluationData": {
    "version": "2.0",
    "conversationId": "ebae8320-...",
    "requestId": "8a660151-...",           // This turn's request ID
    "market": "en-us",
    "experimentName": "",
    "declarativeAgentId": "",
    "userProfile": {                        // User profile for personalization
      "Name": "Adele Vance",
      "Department": "Contoso - SunTech",
      "JobTitle": "Retail Manager",
      "Manager": "Miriam Graham",
      // ...
    },
    "turnData": [ ... ],                   // THE MAIN DATA -- see 4.2
    "totalLatency": 0,
    "latencyOfFirstToken": 13033,          // ms until first token
    "optionsSets": [ ... ],                // 394-446 flight flags (repeated per turn)
    "requestOptions": {}
  }
}
```

The `userProfile` and `optionsSets` are **repeated identically** in every turn's EvaluationData. The unique data per turn is: `requestId`, `latencyOfFirstToken`, and the last entry in `turnData`.

### 4.2 turnData: Cumulative but Sparse

`turnData` is an array that **grows with each turn** but only the **last entry** has populated orchestration. All prior entries are empty shells (userInput only, no orchestration):

| Turn | turnData length | Which entry has orchestration |
|---|---|---|
| Turn 0 (Request 0) | 1 | `turnData[0]` -- populated |
| Turn 1 (Request 1) | 2 | `turnData[1]` -- populated; `[0]` empty |
| Turn 2 (Request 2) | 3 | `turnData[2]` -- populated; `[0]`, `[1]` empty |
| Turn 3 (Request 3) | 4 | `turnData[3]` -- populated; `[0]`-`[2]` empty |

**To get orchestration for all turns, you need each turn's EvaluationData.** A single turn file cannot reconstruct prior turns' orchestration.

Each `turnData` entry:
```jsonc
{
  "userInput": "What's the location for the 7 pm Project Sync call today?",
  "orchestrationIterations": [ ... ]   // Empty array [] for reference-only entries
}
```

### 4.3 orchestrationIterations: The Reasoning Pipeline

The `orchestrationIterations` array contains the multi-step reasoning pipeline. Structure differs between echo and log-based:

#### Echo Multi-Turn (with search sub-agent)

```
orchestrationIterations[0]       <-- Tool-calling iteration
  |-- modelActions[0]            <-- Orchestrator (dev-gpt-52-chat) decides to call search_office365
  |     +-- toolInvocations[0]   <-- Condensed search results (39K+ chars)
  |
  +-- nestedOrchestrations[]     <-- Search agent sub-steps (dev-sonicberry)
        |-- [0] iter=1: office365_search (first attempt, may return 0 results)
        |-- [1] iter=2: office365_search (refined query, gets results)
        |-- [2] iter=3: office365_open (fetches full content of top hits)
        +-- [3] iter=4: final ranking table (no tool call, just response)

orchestrationIterations[1]       <-- Response-generation iteration
  +-- modelActions[0]            <-- Orchestrator generates final response
        +-- response             <-- The bot's answer text
```

#### Log-Based Multi-Turn (direct tool calls, no sub-agent)

```
orchestrationIterations[0]       <-- First tool-calling iteration
  +-- modelActions[0]            <-- Orchestrator (dev-gpt-53-chat) calls office365_search
        +-- toolInvocations[0]   <-- Search results
  (NO nestedOrchestrations)

orchestrationIterations[1]       <-- Second tool-calling iteration (refined search)
  +-- modelActions[0]            <-- Another office365_search call
        +-- toolInvocations[0]   <-- More results, with batchedQueries

orchestrationIterations[2]       <-- Response-generation iteration
  +-- modelActions[0]            <-- Final response
```

**Key difference:** Echo uses `nestedOrchestrations` (a search sub-agent with its own model, `dev-sonicberry`). Log-based calls tools directly from the orchestrator with **no nested orchestrations**.

#### No-Tool-Call Turn

```
orchestrationIterations[0]       <-- Response-only iteration
  +-- modelActions[0]            <-- Orchestrator generates a direct response
        +-- response             <-- The answer text
  (NO nestedOrchestrations)
```

### 4.4 modelActions: Action Details

Every `modelAction` contains:

| Field | Description | Example Values |
|---|---|---|
| `tag` | Action type | `fluxv3:invokingfunction`, `fluxv3:responding`, `searchagent` |
| `model` | Model name | `dev-gpt-52-chat`, `dev-gpt-53-chat`, `dev-sonicberry` |
| `modelApi` | API endpoint | `/chat/completions` |
| `prompt` | **Full prompt** (system+user+developer) | 30K-100K+ chars |
| `modelOutput` | Raw model output with metadata | JSON string with token counts |
| `response` | Extracted text response | Bot answer text |
| `chainOfThought` | Reasoning/thinking text | Search agent CoT (prefixed with `<\|im_sep\|>`) |
| `toolInvocations` | Tool call array | See 4.6 |
| `startTime` / `endTime` | Timestamps | ISO 8601 |
| `latencyMilliseconds` | Wall-clock latency | Integer |
| `additionalMetrics` | Token usage | `{PromptToken, CompletionToken, CachedToken, ReasoningToken}` |
| `availableTools` | Tool definitions with schemas | Full function signatures |

### 4.5 Prompt Structure: Full Conversation History

The `prompt` field in each `modelAction` is a **JSON-serialized array** of OpenAI-style chat messages. For later turns, it includes the **complete conversation history** — all prior turns' user messages, tool calls, tool results, and bot responses:

**Example: Turn 2 (third turn) prompt in echo multi-turn — 12 messages, 103K chars:**

| Index | Role | Content |
|---|---|---|
| [0] | `system` | System prompt (42K) — persona, safety rules, tool definitions |
| [1] | `system` | Runtime context — current time, user location |
| [2] | `user` | Turn 0 user message |
| [3] | `assistant` | Turn 0 tool call (`search_office365`, content empty, `tool_calls` array) |
| [4] | `tool` | Turn 0 search results (39K) — `tool_call_id` + `name` fields |
| [5] | `assistant` | Turn 0 bot response |
| [6] | `user` | Turn 1 user message |
| [7] | `assistant` | Turn 1 tool call |
| [8] | `tool` | Turn 1 search results (3.6K) |
| [9] | `assistant` | Turn 1 bot response |
| [10] | `user` | **Turn 2 user message (current)** |
| [11] | `developer` | Tool invocation instructions (8.5K) |

**Key implications:**
- The LLM sees **all prior tool calls and their full results** when reasoning about the current turn
- Prompt size grows significantly with each turn (57K → 91K → 103K in the example above)
- Within a single turn with multiple `orchestrationIterations`, the response-generation iteration's prompt also includes the current turn's tool call and results (appended after the user message)

### 4.6 toolInvocations

```jsonc
{
  "function": "search_office365",
  "arguments": "{\"task\":\"...\"}",               // or: {"queries":[{"domain":"meetings",...}]}
  "processedResult": "{ ... }",                    // Processed results (up to 39K+ chars)
  "result": "",                                     // Raw result (sometimes populated in log-based)
  "callId": "call_JvVCnG1BMnMn8kOEUXmTatOp",
  "batchedQueries": [ ... ],                       // Per-domain sub-queries (when present)
  "sourceReferences": [ ... ]                       // Citation metadata
}
```

**`batchedQueries`** break down multi-domain searches:
```jsonc
{
  "arguments": "{\"domain\":\"meetings\", \"query\":\"past meetings Tuesday\", ...}",
  "processedResult": "{ ... }",            // Per-domain results
  "result": "{ ... }",                     // Raw API response
  "sourceReferences": [ ... ]
}
```

### 4.7 Tool Call Argument Format Differences

| Conversation Type | Tool Call Pattern | Argument Format |
|---|---|---|
| **Echo** (orchestrator) | `search_office365` | `{"task": "natural language description"}` |
| **Echo** (search sub-agent) | `office365_search`, `office365_open`, `office365_find` | `{"queries": [{"domain": "meetings", "query": "...", ...}]}` |
| **Log-based** (direct) | `office365_search` | `{"queries": [{"domain": "meetings", "query": "...", ...}]}` |
| **Log-based** (location) | `fetch_location_info` | `{"name": "Building 92", "type": "Building", ...}` |

---

## 5. Comparison: Echo vs Log-Based vs Single-Turn

| Aspect | Echo Multi-Turn | Log-Based Multi-Turn | Single-Turn |
|---|---|---|---|
| **Turns** | 2-5 (all live) | 2-8 (all live, but context from scripted log) | 1 |
| **`requests` count** | N (one per live turn) | N (one per live turn) | 1 |
| **Conversation context** | Built from actual prior responses | Carries forward prior bot responses (pre-scripted in `query.id`) | None |
| **Search architecture** | Orchestrator -> search sub-agent (`dev-sonicberry`) -> `office365_search/open/find` | Orchestrator calls `office365_search` directly | Same as log-based |
| **`nestedOrchestrations`** | Yes (search agent iterations) | **No** | **No** |
| **Chain-of-thought** | In `Progress/ChainOfThoughtSummary` messages + nested `chainOfThought` | In `Progress` messages (no `ChainOfThoughtSummary` origin) | May have `Progress` |
| **Search results in messages** | `Internal/DeepLeo` with `ReferencedSources` JSON | `InternalSearchResult/DeepLeo` with `groupedResults` JSON | Same as log-based |
| **Models observed** | `dev-gpt-52-chat` + `dev-sonicberry` | `dev-gpt-53-chat` (single model) | `dev-gpt-52-chat` |
| **EvaluationData `turnData`** | Cumulative, last populated | Cumulative, last populated | Single entry, populated |
| **`query.id` content** | User messages only | User + bot messages interleaved | Single user message |
| **Progress message type** | `InternalPaginationData`, `InternalSearchResult` absent | `InternalPaginationData`, `InternalSearchResult` present | May have `InternalLoaderMessage` |

---

## 6. Data Flow Diagrams

### 6.1 Echo Multi-Turn

```
Scraping File
|-- query.id = [user0, user1, user2]           <-- Only user messages
|-- requests[0] (Turn 0 -- live)
|   +-- response_body.messages[]
|       |-- user input
|       |-- ModelSelector
|       |-- InvokeAction
|       |-- Progress/ChainOfThoughtSummary xN  <-- Search agent reasoning
|       |-- Internal/DeepLeo                    <-- Condensed search results
|       |-- bot response
|       |-- citations, annotations
|       +-- EvaluationData
|           +-- turnData[0] <-- POPULATED
|               +-- orchestrationIterations
|                   |-- [0] orchestrator -> search_office365
|                   |   +-- nestedOrchestrations (search agent)
|                   |       |-- office365_search x2
|                   |       |-- office365_open
|                   |       +-- ranking
|                   +-- [1] orchestrator -> final response
|-- requests[1] (Turn 1 -- live)
|   +-- EvaluationData.turnData = [empty, POPULATED]
+-- requests[2] (Turn 2 -- live)
    +-- EvaluationData.turnData = [empty, empty, POPULATED]
```

### 6.2 Log-Based Multi-Turn

```
Scraping File
|-- query.id = [user0, bot0, user1, bot1, ...]  <-- User+bot interleaved (the "log")
|-- requests[0] (Turn 0 -- live)
|   +-- response_body.messages[]
|       |-- user input
|       |-- ModelSelector
|       |-- Progress ("OK, I'll search for '...'...")
|       |-- InvokeAction + InternalPaginationData + InternalSearchResult xN
|       |-- bot response
|       |-- citations, annotations
|       +-- EvaluationData
|           +-- turnData[0] <-- POPULATED
|               +-- orchestrationIterations (NO nested, direct tool calls)
|                   |-- [0] office365_search (attempt 1)
|                   |-- [1] office365_search (attempt 2)
|                   +-- [2] final response
|-- requests[1] (Turn 1 -- live)
|   +-- EvaluationData.turnData = [empty, POPULATED]
|-- requests[2] (Turn 2 -- live)
|   +-- EvaluationData.turnData = [empty, empty, POPULATED]
+-- requests[3] (Turn 3 -- live)
    +-- EvaluationData.turnData = [empty, empty, empty, POPULATED]
```

### 6.3 Single-Turn

```
Scraping File
|-- query.id = [user0]                          <-- Single user message
+-- requests[0] (only request)
    +-- response_body.messages[]
        |-- user input
        |-- InternalLoaderMessage                <-- "Working on generating..."
        |-- ModelSelector
        |-- Progress + HintInvocation
        |-- InvokeAction (if tools called)
        |-- bot response (or TurnFinalizer on error)
        +-- EvaluationData
            +-- turnData[0] <-- POPULATED (single entry)
```

---

## 7. Where to Extract Specific Information

### 7.1 Reasoning / Chain-of-Thought

| Source | Location | Detail Level | Available In |
|---|---|---|---|
| **Quick** | `messages[].messageType == "Progress"` | 1-2 sentence summaries | All types |
| **Full (Echo)** | `evaluationData -> turnData[N] -> orchestrationIterations -> nestedOrchestrations -> modelActions -> chainOfThought` | Full search agent reasoning (prefixed with `<\|im_sep\|>`) | Echo only |
| **Full prompt** | `evaluationData -> orchestrationIterations -> modelActions -> prompt` | Complete system/user/developer prompt (30K-100K+) | All types |

### 7.2 Search Results

| Source | Location | Detail Level | Available In |
|---|---|---|---|
| **Condensed (Echo)** | `messages[].messageType == "Internal" && contentOrigin == "DeepLeo"` | `ReferencedSources` (top hits) + `Sources` JSON | Echo |
| **Condensed (Log-based)** | `messages[].messageType == "InternalSearchResult"` | `groupedResults` JSON | Log-based |
| **Full per-domain** | `evaluationData -> toolInvocations -> batchedQueries -> processedResult` | Individual domain results, raw API responses | All (when tools called) |
| **Search agent level (Echo)** | `evaluationData -> nestedOrchestrations -> modelActions -> toolInvocations` | `office365_search`, `office365_open`, `office365_find` | Echo only |

### 7.3 Tool Calls

| Source | Location | Detail Level | Available In |
|---|---|---|---|
| **Signal only** | `messages[].messageType == "InvokeAction"` | Tool was invoked (no args/results) | All types |
| **Full (Echo)** | `evaluationData -> orchestrationIterations[0] -> modelActions[0] -> toolInvocations` | Top-level: `search_office365` with task | Echo |
| **Full sub-steps (Echo)** | `evaluationData -> nestedOrchestrations -> modelActions -> toolInvocations` | `office365_search`, `office365_open`, `office365_find` with args + results | Echo |
| **Full (Log-based)** | `evaluationData -> orchestrationIterations[N] -> modelActions[0] -> toolInvocations` | Direct `office365_search` / `fetch_location_info` calls | Log-based |

### 7.4 Token Usage and Latency

Only available in `evaluationData`:

- **Token counts**: `modelActions[].additionalMetrics.metrics` -> `{PromptToken, CompletionToken, CachedToken, ReasoningToken}`
- **Per-action latency**: `modelActions[].latencyMilliseconds`
- **First-token latency**: `evaluationData.latencyOfFirstToken`
- **Model info**: `modelActions[].model`

### 7.5 User Profile and Flight Flags

- **User profile**: `evaluationData.userProfile` -- Name, Department, JobTitle, Manager, etc.
- **Option sets (flights)**: `evaluationData.optionsSets` -- 394-446 feature flags
- These are **identical across all turns** in the same conversation

---

## 8. Extraction Recipes

### 8.1 Get all tool calls for a conversation (works for all types)

```python
import json

def extract_tool_calls(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    for ri, req in enumerate(data["requests"]):
        msgs = req["response_body"]["messages"]
        for msg in msgs:
            if msg.get("messageType") != "EvaluationData":
                continue
            for ti, td in enumerate(msg["evaluationData"]["turnData"]):
                for orch in td.get("orchestrationIterations", []):
                    # Direct tool calls (log-based and echo orchestrator level)
                    for act in orch.get("modelActions", []):
                        for tool in act.get("toolInvocations", []):
                            yield {
                                "request": ri, "turn": ti, "level": "orchestrator",
                                "function": tool["function"],
                                "arguments": tool.get("arguments", ""),
                                "result_len": len(tool.get("processedResult", "")),
                            }
                    # Nested tool calls (echo search agent)
                    for nest in orch.get("nestedOrchestrations", []):
                        for nact in nest.get("modelActions", []):
                            for tool in nact.get("toolInvocations", []):
                                yield {
                                    "request": ri, "turn": ti, "level": "search_agent",
                                    "function": tool["function"],
                                    "arguments": tool.get("arguments", ""),
                                    "result_len": len(tool.get("processedResult", "")),
                                }

for tc in extract_tool_calls("path/to/scraping_file.json"):
    print(f"  [{tc['level']}] {tc['function']} (result: {tc['result_len']} chars)")
```

### 8.2 Get chain-of-thought reasoning (all types)

```python
def extract_reasoning(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    for ri, req in enumerate(data["requests"]):
        msgs = req["response_body"]["messages"]
        
        # From Progress messages (all types)
        for msg in msgs:
            if msg.get("messageType") == "Progress":
                yield {"request": ri, "source": "progress_msg", "text": msg.get("text", "")}
        
        # From evaluationData nested orchestrations (echo only)
        for msg in msgs:
            if msg.get("messageType") != "EvaluationData":
                continue
            for td in msg["evaluationData"]["turnData"]:
                for orch in td.get("orchestrationIterations", []):
                    for nest in orch.get("nestedOrchestrations", []):
                        for act in nest.get("modelActions", []):
                            cot = act.get("chainOfThought", "")
                            if cot:
                                cot = cot.replace("<|im_sep|>", "").strip()
                                yield {"request": ri, "source": "evaldata_cot", "text": cot}
```

### 8.3 Get search results

```python
def extract_search_results(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    for ri, req in enumerate(data["requests"]):
        msgs = req["response_body"]["messages"]
        
        # From messages (quick, condensed)
        for msg in msgs:
            # Echo: Internal/DeepLeo contains ReferencedSources
            if msg.get("messageType") == "Internal" and msg.get("contentOrigin") == "DeepLeo":
                try:
                    results = json.loads(msg["text"])
                    yield {"request": ri, "source": "message", "data": results}
                except:
                    pass
            # Log-based: InternalSearchResult
            if msg.get("messageType") == "InternalSearchResult":
                try:
                    results = json.loads(msg["text"])
                    yield {"request": ri, "source": "message", "data": results}
                except:
                    pass
        
        # From evaluationData (full detail)
        for msg in msgs:
            if msg.get("messageType") != "EvaluationData":
                continue
            for td in msg["evaluationData"]["turnData"]:
                for orch in td.get("orchestrationIterations", []):
                    for act in orch.get("modelActions", []):
                        for tool in act.get("toolInvocations", []):
                            if tool.get("processedResult"):
                                yield {"request": ri, "source": "evaldata", "data": tool["processedResult"]}
                    for nest in orch.get("nestedOrchestrations", []):
                        for act in nest.get("modelActions", []):
                            for tool in act.get("toolInvocations", []):
                                if tool.get("processedResult"):
                                    yield {"request": ri, "source": "evaldata_nested", "data": tool["processedResult"]}
```

### 8.4 Get token usage per model call

```python
def extract_token_usage(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    for ri, req in enumerate(data["requests"]):
        msgs = req["response_body"]["messages"]
        for msg in msgs:
            if msg.get("messageType") != "EvaluationData":
                continue
            for td in msg["evaluationData"]["turnData"]:
                for orch in td.get("orchestrationIterations", []):
                    for act in orch.get("modelActions", []):
                        metrics = act.get("additionalMetrics", {}).get("metrics", {})
                        if metrics:
                            yield {"request": ri, "tag": act.get("tag"), "model": act.get("model"), **metrics}
                    for nest in orch.get("nestedOrchestrations", []):
                        for act in nest.get("modelActions", []):
                            metrics = act.get("additionalMetrics", {}).get("metrics", {})
                            if metrics:
                                yield {"request": ri, "tag": act.get("tag"), "model": act.get("model"), **metrics}
```
