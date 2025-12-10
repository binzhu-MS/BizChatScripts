# SEVAL Complete Data Structure and Processing Guide

**Last Updated**: December 10, 2025  
**Purpose**: Complete technical specification for SEVAL conversation data structures, CiteDCG data structures, and data matching principles

> **Note**: For implementation details on extracting and merging CiteDCG scores, see [SEVAL_Implementation_Guide.md](SEVAL_Implementation_Guide.md).

---

## Table of Contents

### Part 1: Conversation Data Structures
1. [Overview](#part-1-conversation-data-structures)
2. [Conversation File Structure](#conversation-file-structure)
3. [EvaluationData Structure](#evaluationdata-structure)
4. [Message Flow and Sequencing](#message-flow-and-sequencing)
5. [Message Types](#message-types)
6. [Search Result Patterns](#search-result-patterns)
7. [Content Access Detection](#content-access-detection)
8. [Error Patterns](#error-patterns)
9. [Model Information Extraction](#model-information-extraction)
10. [Conversation JSON Path Reference](#conversation-json-path-reference)
11. [Conversation Data Important Notes](#conversation-data-important-notes)
    - Multi-Turn Conversations and Pattern Analysis
    - Non-Empty Turn Definitions
12. [Current Implementation Notes](#current-implementation-notes)

### Part 2: CiteDCG Data Structures
13. [CiteDCG Data Organization](#part-2-citedcg-data-structures)
14. [CiteDCG File Structure](#citedcg-file-structure)
15. [CiteDCG Web Search Organization](#citedcg-web-search-organization)
16. [CiteDCG Score Interpretation](#citedcg-score-interpretation)
17. [Data Source Locations](#data-source-locations)

### Part 3: Data Matching and Merging
18. [Design Principles](#part-3-data-matching-and-merging)
19. [Reference ID Mapping](#reference-id-mapping)
20. [Type-Aware Matching Logic](#type-aware-matching-logic)
21. [Matching Algorithms](#matching-algorithms)
22. [Practical Examples](#practical-examples)
23. [Common Pitfalls](#common-pitfalls)
24. [Troubleshooting](#troubleshooting)
25. [Implementation Checklist](#implementation-checklist)

---

# Part 1: Conversation Data Structures

## Overview

### File Types
- **Control Files**: `control_sydney_response_*.json`
- **Experiment Files**: `experiment_sydney_response_*.json`
- **Size**: ~1-50MB per file (contains complete conversation data)
- **Format**: Single JSON object per file

### Purpose
SEVAL (Sydney Evaluation) conversation files contain complete M365 Copilot conversation traces including:
- User queries and bot responses
- Search operations and results
- File/content access patterns
- Model invocations and reasoning steps
- Error states and diagnostics
- Telemetry and metadata

---

## Conversation File Structure

### Top-Level Schema

```json
{
  "conversation_id": "uuid-string",
  "query": {
    "id": "user-query-text",
    "query_hash": "hash-string"
  },
  "exp_name": "control|experiment",
  "conversation_success": true|false,
  "requests": [
    {
      "exp_name": "control|experiment",
      "retry_attempt": 1,
      "utterance": "user-query-text",
      "eval_time_override": "2025-06-04T11:17:25.311599",
      "chat_correlation_id": "uuid",
      "chat_attempt_correlation_id": "uuid",
      "is_success": true|false,
      "failure_reason": null|"error-description",
      "request_url": "https://substrate.office.com/...",
      "request_params": {...},
      "request_headers": {...},
      "request_body": {...},
      "response_headers": {...},
      "response_body": {
        "messages": [...],
        "conversationId": "uuid",
        "requestId": "uuid",
        "result": {...}
      },
      "sydney_status_code": 200,
      "status_code": 200,
      "trace_id": "uuid",
      "request_id": "uuid",
      "conversation_context": {...},
      "timestamp": "timestamp"
    }
  ]
}
```

### Key Top-Level Fields

| Field                  | Type    | Description                                                 |
| ---------------------- | ------- | ----------------------------------------------------------- |
| `conversation_id`      | string  | Unique conversation identifier                              |
| `query.id`             | string  | User query text                                             |
| `exp_name`             | string  | "control" or "experiment"                                   |
| `conversation_success` | boolean | Overall conversation success indicator                      |
| `requests`             | array   | Array of request/response pairs (one per conversation turn) |

### Request Object Fields

| Field                          | Type        | Description                               |
| ------------------------------ | ----------- | ----------------------------------------- |
| `utterance`                    | string      | User query for this turn                  |
| `is_success`                   | boolean     | Request success indicator                 |
| `failure_reason`               | string/null | Specific failure description if failed    |
| `response_body.messages`       | array       | Sequence of messages in conversation flow |
| `response_body.conversationId` | string      | Conversation ID from bot                  |
| `response_body.result`         | object      | Result metadata                           |

---

## EvaluationData Structure

### Overview

SEVAL conversation files contain a special message with `messageType: "EvaluationData"` that provides structured conversation details and turn-by-turn breakdown. This is **the primary source** for analyzing multi-turn conversations and extracting detailed turn data.

### Location

```
requests[].response_body.messages[] 
  → Find message where messageType == "EvaluationData"
    → evaluationData object
```

### EvaluationData Top-Level Schema

```json
{
  "messageType": "EvaluationData",
  "evaluationData": {
    "version": "string",
    "conversationId": "uuid-string",
    "experimentName": "control|experiment",
    "requestId": "uuid-string",
    "market": "en-US",
    "declarativeAgentId": "string",
    "userProfile": {...},
    "turnData": [
      {
        "turnIndex": 1,
        "userInput": "user query text",
        "startTime": "timestamp",
        "endTime": "timestamp",
        "orchestrationIterations": [...],
        "botResponses": [...],
        "citedResponseAttributions": [...],
        "uncitedResponseAttributions": [...],
        "locationInfo": {...},
        "webPageContext": {...},
        "suggestedResponses": [...],
        "codeInterpreterOutputs": [...],
        "messageIds": [...],
        "telemetryIds": [...],
        "generatedImages": [...],
        "pptSlideImages": {...},
        "canvasPageContent": "",
        "implicitAppContexts": [...]
      }
    ],
    "totalLatency": 0,
    "latencyOfFirstToken": 21753,
    "optionsSets": [...]
  }
}
```

### EvaluationData Key Fields

| Field                 | Type   | Description                                           |
| --------------------- | ------ | ----------------------------------------------------- |
| `version`             | string | EvaluationData schema version                         |
| `conversationId`      | string | Conversation identifier                               |
| `experimentName`      | string | "control" or "experiment"                             |
| `requestId`           | string | Request tracking ID                                   |
| `turnData`            | array  | **Array of turn objects (one per conversation turn)** |
| `totalLatency`        | number | Total conversation latency (ms)                       |
| `latencyOfFirstToken` | number | Time to first token (ms)                              |

### TurnData Structure

Each element in the `turnData` array represents one conversation turn:

| Field                         | Type   | Description                      |
| ----------------------------- | ------ | -------------------------------- |
| `turnIndex`                   | number | Turn number (1-based)            |
| `userInput`                   | string | User's query for this turn       |
| `startTime`                   | string | Turn start timestamp             |
| `endTime`                     | string | Turn end timestamp               |
| `orchestrationIterations`     | array  | Reasoning/orchestration steps    |
| `botResponses`                | array  | Bot response messages            |
| `citedResponseAttributions`   | array  | Sources cited in response        |
| `uncitedResponseAttributions` | array  | Sources retrieved but not cited  |
| `messageIds`                  | array  | Message IDs for this turn        |
| `telemetryIds`                | array  | Telemetry tracking IDs           |
| `suggestedResponses`          | array  | Suggested follow-up queries      |
| `codeInterpreterOutputs`      | array  | Code execution outputs           |
| `generatedImages`             | array  | AI-generated images              |
| `pptSlideImages`              | object | PowerPoint slide images          |
| `implicitAppContexts`         | array  | Implicit app context information |

### Multi-Turn Conversation Detection

**Using EvaluationData** (RECOMMENDED):
```python
def count_turns_from_evaluation_data(seval_file):
    """
    Count turns using EvaluationData.turnData
    
    Returns:
        dict with turn_count and non_empty_turn_count
    """
    with open(seval_file, 'r') as f:
        data = json.load(f)
    
    # Find EvaluationData message
    eval_msg = None
    for request in data.get('requests', []):
        messages = request.get('response_body', {}).get('messages', [])
        for msg in messages:
            if msg.get('messageType') == 'EvaluationData':
                eval_msg = msg
                break
        if eval_msg:
            break
    
    if not eval_msg:
        return {'turn_count': 0, 'non_empty_turns': 0}
    
    eval_data = eval_msg.get('evaluationData', {})
    turn_data = eval_data.get('turnData', [])
    
    total_turns = len(turn_data)
    non_empty_turns = 0
    
    # Count non-empty turns (turns with bot responses or citations)
    for turn in turn_data:
        bot_responses = turn.get('botResponses', [])
        cited_attrs = turn.get('citedResponseAttributions', [])
        uncited_attrs = turn.get('uncitedResponseAttributions', [])
        
        # A turn is non-empty if it has bot responses or attributions
        if bot_responses or cited_attrs or uncited_attrs:
            non_empty_turns += 1
    
    return {
        'turn_count': total_turns,
        'non_empty_turns': non_empty_turns
    }
```

**Using requests[] array** (LEGACY):
```python
def count_turns_from_requests(seval_data):
    """Legacy method: Count turns by requests array length"""
    return len(seval_data.get('requests', []))
```

### Important Notes

**Multi-Turn vs Multi-Request**:
- **Multi-Turn**: Multiple entries in `evaluationData.turnData[]` (conversation retries/iterations)
- **Multi-Request**: Multiple entries in top-level `requests[]` (rare, usually just 1)
- For job 133560 control files:
  - 99 files have >1 turn in `turnData`
  - 95 files have >1 **non-empty** turn (with actual bot responses)
  - Most are 2-turn conversations (retries after first attempt)
  - 3 files have 3 turns

**When Retries Occur**:
- System errors requiring retry
- Content retrieval failures
- Timeout or connectivity issues
- Model invocation failures

**Non-Empty Turn Definition**:
A turn is considered non-empty if it contains:
- Bot responses (`botResponses` not empty), OR
- Cited attributions (`citedResponseAttributions` not empty), OR  
- Uncited attributions (`uncitedResponseAttributions` not empty)

---

## Message Flow and Sequencing

### Typical Message Sequence

SEVAL conversation files contain messages in a specific sequence that traces the complete interaction:

```
Message 0:  User query (author: "user")
Message 1:  Bot acknowledgment (author: "bot") - "Working on generating the response..."
Message 2:  System message (author: "system")
Message 3:  Search query execution (author: "bot") - Actual search query being executed
Messages 4-6: Search processing steps (may include "No content returned" errors)
Message 7-8: Search metadata (paginationSearchMetadata) - Search request details
Message 9:  Search results ({"results":...}) - Found documents and snippets
Message 10: Final formatted response (author: "bot") - User-facing response with citations
Message 11: Storage results ({"storageResults":...}) - File URLs and access information
Message 13: Search results mapping ({"searchResultsMap":...}) - Detailed file metadata
Message 14: Entity mappings ({"tokenToEntityMappings":...}) - File and person entity data
```

**Note**: Message indices may vary. Use `messageType` and `author` fields to identify messages, not fixed positions.

### Message Object Structure

```json
{
  "text": "message-content",
  "hiddenText": "internal-content (may contain additional data)",
  "author": "user|bot|system",
  "messageType": "InternalSearchResult|InternalPaginationData|Progress|etc",
  "messageId": "uuid",
  "requestId": "uuid",
  "createdAt": "timestamp",
  "timestamp": "timestamp",
  "contentOrigin": "DeepLeo|EnterpriseSearchExtension|etc",
  "adaptiveCards": [ /* UI card data for display */ ],
  "sourceAttributions": [ /* Citation/source data */ ],
  "invocation": [ /* Function call data */ ],
  "pluginInfo": { /* Plugin metadata */ },
  "telemetry": { /* Search telemetry */ }
}
```

---

## Message Types

### Message Type Catalog

| MessageType                | Author | Purpose                       | Content Structure                          |
| -------------------------- | ------ | ----------------------------- | ------------------------------------------ |
| **InternalSearchResult**   | bot    | Search results with file data | JSON: `{"results": [{"result": {...}}]}`   |
| **InternalPaginationData** | bot    | Search request metadata       | JSON with search parameters and pagination |
| **Progress**               | bot    | Search progress indicator     | Plain text: "Searching for..."             |
| **InvokeAction**           | bot    | Function invocation result    | JSON with function call results            |
| **Internal**               | system | System messages               | Text or JSON with system information       |
| **InternalLoaderMessage**  | bot    | Loading state indicator       | Plain text: "Working on..."                |
| **(no messageType)**       | user   | User query                    | Plain text with user's request             |
| **(no messageType)**       | bot    | Final response to user        | Formatted text with citations `[^1^][^2^]` |

### Key Message Types for Data Extraction

#### InternalSearchResult

Contains actual search results returned to the model:

```json
{
  "messageType": "InternalSearchResult",
  "author": "bot",
  "text": "{\"results\":[{\"result\":{\"reference_id\":\"turn1search7\",\"type\":\"File\",\"title\":\"<File>Grafana_Dashboard_Design_Specification</File>\",\"snippet\":\"Detailed content from the file...\",\"author\":\"<Person>Alex Kim</Person>\",\"fileType\":\"pdf\",\"fileName\":\"<File>Grafana_Dashboard_Design_Specification.pdf</File>\"}}]}"
}
```

**Key Fields in Results**:
- `reference_id`: Unique identifier (e.g., "turn1search7")
- `type`: File, EmailMessage, Chat, External, etc.
- `title`: Document title (may include XML tags like `<File>...</File>`)
- `snippet`: Content preview/excerpt
- `author`: Document author (may include XML tags like `<Person>...</Person>`)
- `fileName`: Original filename
- `fileType`: File extension (pdf, docx, xlsx, etc.)

#### InternalPaginationData

Contains search request metadata:

```json
{
  "messageType": "InternalPaginationData",
  "author": "bot",
  "text": "{\"paginationSearchMetadata\":{\"query\":\"machine learning best practices\",\"searchType\":\"Files\",\"pageSize\":10,\"offset\":0}}"
}
```

#### Final Response (no messageType)

The user-facing response with citations:

```json
{
  "author": "bot",
  "text": "Based on the documents, here are the best practices for machine learning: [^1^][^2^]\n\n1. **Data Quality**...",
  "sourceAttributions": [
    {
      "seeMoreUrl": "https://...",
      "searchQuery": "machine learning",
      "providerDisplayName": "Files"
    }
  ]
}
```

**Citation Format**: Citations appear as `[^1^][^2^]` in the text, referencing documents by position.

---

## Search Result Patterns

### Search Results Organization

Search results are embedded in message `text` field as JSON strings:

```json
{
  "results": [
    {
      "result": {
        "reference_id": "turn1search0",
        "type": "File",
        "title": "<File>Document Name</File>",
        "snippet": "Content preview...",
        "fileName": "<File>document.pdf</File>",
        "fileType": "pdf",
        "author": "<Person>John Doe</Person>",
        "lastModifiedDateTime": "2025-06-01T10:00:00Z"
      }
    }
  ]
}
```

### Storage Results (File Access URLs)

File access information appears in separate messages:

```json
{
  "storageResults": [
    {
      "id": "file-id",
      "url": "https://storage.../document.pdf",
      "type": "File",
      "fileName": "document.pdf"
    }
  ]
}
```

### Search Results Map (Detailed Metadata)

Extended file metadata:

```json
{
  "searchResultsMap": {
    "turn1search0": {
      "id": "file-id",
      "title": "Document Name",
      "url": "https://...",
      "fileType": "pdf",
      "lastModified": "2025-06-01",
      "author": "John Doe"
    }
  }
}
```

### Entity Mappings

Person and file entity references:

```json
{
  "tokenToEntityMappings": {
    "Document Name": {
      "type": "File",
      "id": "file-id",
      "name": "document.pdf"
    },
    "John Doe": {
      "type": "Person",
      "id": "person-id",
      "email": "john.doe@example.com"
    }
  }
}
```

---

## Content Access Detection

### Access Levels

**No Access**:
- Search performed but no files were accessed
- Only snippets/metadata shown to model
- Indicators: No `storageResults`, no content in `hiddenText`

**Partial Access**:
- Some files accessed but not all search results
- Mix of full content and snippets
- Indicators: `storageResults` present but fewer than search results

**Full Access**:
- All relevant files accessed and content provided to model
- Complete document content in conversation
- Indicators: `storageResults` count matches or exceeds search results, extensive content in `hiddenText`

### Detection Logic

```python
def determine_access_level(messages):
    """Determine content access level from message sequence"""
    
    search_results = []
    storage_results = []
    
    for message in messages:
        text = message.get('text', '')
        
        # Extract search results
        if message.get('messageType') == 'InternalSearchResult':
            try:
                data = json.loads(text)
                search_results.extend(data.get('results', []))
            except:
                pass
        
        # Extract storage/access results
        if '"storageResults"' in text:
            try:
                data = json.loads(text)
                storage_results.extend(data.get('storageResults', []))
            except:
                pass
    
    if not search_results:
        return 'no_search'
    
    if not storage_results:
        return 'no_access'
    
    if len(storage_results) >= len(search_results):
        return 'full_access'
    
    return 'partial_access'
```

---

## Error Patterns

### Error Detection Fields

| Field Path                                       | Type        | Error Relevance             |
| ------------------------------------------------ | ----------- | --------------------------- |
| `conversation_success`                           | boolean     | **HIGH** - Primary filter   |
| `requests[].is_success`                          | boolean     | **HIGH** - Secondary filter |
| `requests[].failure_reason`                      | string/null | **MEDIUM** - Error details  |
| `requests[].response_body.messages[].text`       | string      | **HIGH** - Error messages   |
| `requests[].response_body.messages[].hiddenText` | string      | **MEDIUM** - Diagnostics    |

### Common Error Patterns

#### Category 1: File Access Failures

**Pattern: "No content returned"**
```json
{
  "author": "bot",
  "text": "No content returned from the file access API."
}
```

**Indicators**:
- Phrase "No content returned" in message text
- Search succeeded but content retrieval failed
- May occur in messages 4-6 of sequence

**Pattern: FileNotFoundError**
```json
{
  "hiddenText": "FileNotFoundError: File not accessible or deleted"
}
```

#### Category 2: Python Execution Failures

**Pattern: Python error in code execution**
```json
{
  "text": "Error executing Python code: NameError: name 'variable' is not defined"
}
```

**Indicators**:
- "NameError", "TypeError", "ValueError" in text
- "Error executing" prefix
- Usually in `InvokeAction` message type

#### Category 3: Search and Retrieval Failures

**Pattern: Empty search results**
```json
{
  "messageType": "InternalSearchResult",
  "text": "{\"results\":[]}"
}
```

**Pattern: Search timeout**
```json
{
  "failure_reason": "Search operation timed out after 30s"
}
```

#### Category 4: Timeout and Connectivity Issues

**Pattern: API timeout**
```json
{
  "failure_reason": "Request timeout",
  "is_success": false
}
```

**Pattern: Connection failures**
```json
{
  "text": "Connection to service failed. Please retry."
}
```

### Error Detection Strategy

**Level 1: Structural Failures (Highest Priority)**
- `conversation_success: false`
- `requests[].is_success: false`
- `requests[].failure_reason` is not null

**Level 2: Content-Based Errors (High Priority)**
- "No content returned" in response text
- "FileNotFoundError" in any content
- Python execution failures

**Level 3: Pattern-Based Errors (Medium Priority)**
- Generic error patterns
- Timeout messages
- Empty result arrays

---

## Model Information Extraction

### Reasoning Model Identification

Model information appears in multiple locations within the conversation JSON. Models are used for:
- **Reasoning/Planning**: Determining which tools to invoke (`fluxv3:invokingfunction`)
- **Response Generation**: Creating the final user-facing response
- **Function Execution**: Running specific tools or plugins

### Model Information Locations

#### Location 1: orchestrationIterations Pattern

Found in `requests[].response_body.result.orchestrationIterations`:

```json
{
  "result": {
    "orchestrationIterations": [
      {
        "iteration": 1,
        "modelOutput": {
          "model": "dev-gpt-5-reasoning",
          "modelVersion": "2024-11-01",
          "eventName": "fluxv3:invokingfunction"
        }
      }
    ]
  }
}
```

**Key Fields**:
- `modelOutput.model`: Model name (e.g., "dev-gpt-5-reasoning")
- `modelOutput.eventName`: Phase where model was used
- `iteration`: Iteration number in orchestration loop

#### Location 2: Extended Data Pattern

Found in message-level extended data:

```json
{
  "extendedData": {
    "LoopCount": 1,
    "ModelInfo": {
      "model": "dev-gpt-5-chat-jj",
      "version": "2024-11-01"
    }
  }
}
```

#### Location 3: Model Tags in Invocation

Found in message invocation data:

```json
{
  "invocation": {
    "modelOutput": {
      "callTags": {
        "modelName": "dev-gpt-5-reasoning",
        "modelVersion": "2024-11-01"
      }
    }
  }
}
```

### Model Extraction Pattern

```python
def extract_reasoning_model(seval_data):
    """Extract reasoning model used in conversation"""
    
    models_found = set()
    
    # Check orchestrationIterations
    for request in seval_data.get('requests', []):
        result = request.get('response_body', {}).get('result', {})
        iterations = result.get('orchestrationIterations', [])
        
        for iteration in iterations:
            model_output = iteration.get('modelOutput', {})
            if model_output.get('eventName') == 'fluxv3:invokingfunction':
                model_name = model_output.get('model')
                if model_name:
                    models_found.add(model_name)
        
        # Check message-level extended data
        messages = request.get('response_body', {}).get('messages', [])
        for message in messages:
            extended_data = message.get('extendedData', {})
            model_info = extended_data.get('ModelInfo', {})
            model_name = model_info.get('model')
            if model_name:
                models_found.add(model_name)
    
    return list(models_found)
```

### Known Reasoning Models

| Model Name            | Description                    |
| --------------------- | ------------------------------ |
| `dev-gpt-5-reasoning` | GPT-5 reasoning/planning model |
| `dev-gpt-5-chat-jj`   | GPT-5 chat variant             |
| `gpt-4o`              | GPT-4 Omni model               |
| `gpt-4-turbo`         | GPT-4 Turbo model              |

---

## Conversation JSON Path Reference

### Essential Paths for Data Extraction

| Data Element             | JSON Path                                                                     |
| ------------------------ | ----------------------------------------------------------------------------- |
| **User Query**           | `requests[0].utterance`                                                       |
| **Conversation Success** | `conversation_success`                                                        |
| **Request Success**      | `requests[].is_success`                                                       |
| **Failure Reason**       | `requests[].failure_reason`                                                   |
| **Messages**             | `requests[].response_body.messages[]`                                         |
| **Search Results**       | Messages where `messageType == "InternalSearchResult"` → parse `text`         |
| **Final Response**       | Last message where `author == "bot"` and no `messageType`                     |
| **Storage Results**      | Messages where `text` contains `"storageResults"`                             |
| **Reasoning Model**      | `requests[].response_body.result.orchestrationIterations[].modelOutput.model` |
| **Citations**            | Final bot message → `sourceAttributions[]`                                    |
| **File Metadata**        | Messages where `text` contains `"searchResultsMap"`                           |

### Search Result Extraction

```python
def extract_search_results(messages):
    """Extract search results from message sequence"""
    
    results = []
    
    for message in messages:
        if message.get('messageType') == 'InternalSearchResult':
            try:
                data = json.loads(message.get('text', '{}'))
                for item in data.get('results', []):
                    result_data = item.get('result', {})
                    results.append({
                        'reference_id': result_data.get('reference_id'),
                        'type': result_data.get('type'),
                        'title': result_data.get('title'),
                        'snippet': result_data.get('snippet'),
                        'file_type': result_data.get('fileType'),
                        'author': result_data.get('author')
                    })
            except json.JSONDecodeError:
                continue
    
    return results
```

### Final Response Extraction

```python
def extract_final_response(messages):
    """Extract final user-facing response"""
    
    # Find last bot message without messageType (final response)
    for message in reversed(messages):
        if (message.get('author') == 'bot' and 
            not message.get('messageType') and
            len(message.get('text', '')) > 100):
            return {
                'text': message.get('text'),
                'citations': message.get('sourceAttributions', []),
                'message_id': message.get('messageId')
            }
    
    return None
```

---

## Conversation Data Important Notes

### Message Order Variability

- Message indices are **not fixed** across different conversations
- Always use `messageType` and `author` to identify messages
- Some message types may be missing if certain operations didn't occur
- Multiple messages of the same type may exist (e.g., multiple searches)

### JSON Parsing

- Many fields contain **JSON strings** (not JSON objects)
- Always parse `text` field content when it starts with `{`
- Handle JSON parsing errors gracefully (some text may contain partial JSON)

### XML Tags in Content

- File and person names often wrapped in XML tags: `<File>name</File>`, `<Person>name</Person>`
- Strip these tags when extracting for analysis
- Tags help identify entity types in unstructured text

### Multi-Turn Conversations

- Some conversations have multiple turns (multiple request/response pairs)
- Each turn has its own message sequence
- `requests[]` array contains one element per turn

#### Multi-Turn Pattern Analysis (Single-Turn Utterance SEVAL Jobs)

**Important Context**: This analysis applies to SEVAL jobs configured with **single-turn utterances** where multi-turn conversations represent automatic retry attempts, NOT user follow-up questions in a multi-turn dialogue.

**Observed Pattern (100% consistency across Job 133560)**:
- **All multi-turn conversations show "last turn only" pattern**
- Turn 1: No orchestration iterations (uses cached results from contentOrigin="past-enterprise-search-results-metadata")
- Turn 2+ (last turn): Has orchestration iterations with actual tool invocations

**Data Statistics**:
- Control: 99 multi-turn conversations (96 with 2 turns, 3 with 3 turns)
- Treatment: 12 multi-turn conversations (all with 2 turns)
- Pattern consistency: 100% - only the last turn contains orchestrationIterations

#### "Non-Empty Turn" Definitions

The term "non-empty turn" has **different meanings** in different extraction contexts:

**CiteDCG Extraction (`get_seval_metrics.py`)**:
- **Definition**: A turn is non-empty if it has `num_hops > 0`
- **Meaning**: The turn has search results with CiteDCG scores
- **Code location**: Line ~1030: `non_empty_turns = 1 if num_hops > 0 else 0`
- **For multi-turn**: Only the last turn has search results (other turns use cached data)

**Conversation Extraction (`seval_batch_processor.py`)**:
- **Definition**: A turn is non-empty if hop data structures exist in the array
- **Meaning**: The hop array has elements, even if they are empty placeholders without invocations
- **Code location**: Lines ~366-369: Counts all hop structures, regardless of content
- **Note**: Explicitly stated as "different from merge phase which requires invocations"
- **For multi-turn**: Only the last turn has hop data structures

**Analysis Results**:
Using either definition, the pattern is identical:
- Turn 1 (and earlier turns): 0 non-empty hops
- Turn 2+ (last turn): 1+ non-empty hops
- **100% of multi-turn conversations have exactly 1 non-empty turn (the last one)**

**Common Patterns Observed**:
```
2 turns: T1=0, T2=1  (81.8% of control multi-turn)
2 turns: T1=0, T2=2  (11.1% of control multi-turn)
2 turns: T1=0, T2=3  (2.0% of control multi-turn)
3 turns: T1=0, T2=0, T3=1  (3.0% of control multi-turn)
```

**Key Insight**: The first turn(s) in multi-turn conversations are retries that leverage cached search results from previous attempts, avoiding redundant tool invocations. Only the final successful turn performs new searches and gets scored.

**Caveat**: This analysis is specific to SEVAL jobs using single-turn utterances. Multi-turn utterance jobs (where users ask follow-up questions) may exhibit different patterns and have not been analyzed yet.

---

## Current Implementation Notes

> **See Also**: For detailed implementation code, extraction approaches, and workflow documentation, see [SEVAL_Implementation_Guide.md](SEVAL_Implementation_Guide.md).

### Two Extraction Approaches

There are two approaches for extracting and processing SEVAL data:

#### Approach 1: Separate Extraction (Legacy)
1. Extract conversation details from raw conversation files (`*_sydney_response_*.json`)
2. Extract CiteDCG scores from raw DCG files (`results.json`)
3. Merge the two data sources using reference ID matching

**Pros**: Works with any data source combination  
**Cons**: Requires separate extraction steps and complex merging logic

#### Approach 2: Unified Extraction (New)
1. Extract both conversation details AND CiteDCG scores from raw DCG files
2. DCG files contain `EvaluationData.turnData[]` with full conversation structure
3. No merging required - data is already aligned

**Pros**: Single source, no merging complexity, faster processing  
**Cons**: Only works when DCG files are available with EvaluationData

### Data File Organization

```
results/
  {job_id}_citedcg/                       # Raw CiteDCG extraction (legacy)
  {job_id}_conversation_details/          # Raw conversation extraction (legacy)
  {job_id}_conversation_w_citedcg_details/ # Merged data (legacy)
  {job_id}_unified_hop_citedcg_scores/    # Unified extraction output (new)
  {job_id}_unified_utterance_details/     # Utterance-level details (new)
  {job_id}_unified_statistics_plots/      # Statistics and plots (new)
```

### Key Modules

| Module                     | Purpose                                     |
| -------------------------- | ------------------------------------------- |
| `get_seval_metrics.py`     | Core extraction functions (both approaches) |
| `seval_batch_processor.py` | Batch processing and workflow orchestration |
| `merge_seval_results.py`   | Data merging (legacy approach)              |
| `seval_plotting.py`        | Statistics calculation and visualization    |



# Part 2: CiteDCG Data Structures

## CiteDCG Data Organization

CiteDCG (Citation Document Coverage Grading) data provides quality scores for search results. This section explains how CiteDCG data is structured and where to find it.

### Why CiteDCG Data Exists

- **Question**: Did the model cite high-quality results?
- **Question**: Are top-ranked results high quality?
- **Question**: Does the experimental treatment improve citation quality?

To answer these, we need **CiteDCG quality scores** for each search result.

---

## CiteDCG File Structure

### Raw DCG API Response Structure

The raw `results.json` files from the DCG API (before extraction) use a different structure than the extracted format. Understanding this is important when debugging extraction code.

**File Format**: JSONL (newline-delimited JSON) - one conversation object per line  
**File Location**: `seval_data/{job_id}_metrics/Consolidated NDCG and CiteDCG Labels {Experiment}/results.json`

**Key Insight**: Raw DCG files contain **both** search results with scores AND the full `EvaluationData` structure. This means conversation details (turn data, user inputs, orchestration iterations) can be extracted directly from DCG files without needing separate conversation files.

**Top-Level Structure**:
```json
{
  "ConversationId": "uuid-string",
  "AllSearchResults": {
    "hop_number": {
      "plugin_name": [
        {
          "PluginName": "plugin_name",
          "Results": [
            {
              "Type": "File",
              "Id": "original-item-id",
              "ReferenceId": "ce602c01-e9b1-13dc-0433-e9fbf11e5787.1000.3",
              "Rank": 0,  # 0-indexed position
              "HitHighlightedSummary": "content preview...",
              "Source": {...},
              "Content": "full document content...",
              "Metadata": {...},
              "CiteDCGLLMLabel": 2.4,  # Citation quality (0-3): Should this be cited?
              "LLMLabel": 2.2,  # Search quality (0-4): Is this a good search result?
              ...
            }
          ],
          "PluginInvocation": "office365_search({'domain': 'files', 'query': 'search query text', ...})"
        }
      ]
    }
  },
  "signals": {...},
  "QueryHash": "...",
  "EvaluationData": {...},
  "Utterance": "user query text",
  "SydneyReply": "bot response text",
  "SydneySuggestion": [...]
}
```

**Concrete Example**:
```json
{
  "ConversationId": "9bebdcb0-f8be-4de8-b3df-8296600af701",
  "AllSearchResults": {
    "1": {
      "office365_search_files": [
        {
          "PluginName": "office365_search_files",
          "Results": [
            {
              "ReferenceId": "Turn1Search1",
              "CiteDCGLLMLabel": 2.4,
              "Type": "File",
              "Rank": 1
            }
          ],
          "PluginInvocation": "office365_search({'domain': 'files', 'query': 'creatividad en programación OR creative programming', ...})"
        }
      ],
      "office365_search_people": [
        {
          "PluginName": "office365_search_people",
          "Results": [...]
        }
      ]
    },
    "2": {}
  }
}
```
  "Utterance": "user query text",
  "SydneyReply": "bot response text",
  "SydneySuggestion": [...]
}
```

**Key Structure Elements**:

1. **Top-level fields**:
   - `ConversationId`: Conversation UUID
   - `AllSearchResults`: Search results organized by hop and plugin (with CiteDCG scores)
   - `signals`: Metadata about search result types and user profile
   - `EvaluationData`: **Full evaluation data including `turnData[]`** - same structure as in conversation files, enabling unified extraction
   - `Utterance`: User's query text
   - `SydneyReply`: Bot's response text
   - `SydneySuggestion`: Suggested follow-up queries

> **Important**: The `EvaluationData` field in raw DCG files contains the complete `turnData[]` structure with `orchestrationIterations`, `userInput`, and all conversation details. This enables a **unified extraction approach** where both conversation details and CiteDCG scores can be extracted from a single source.

2. **`AllSearchResults`**: Root object containing search results
   - Key: `hop_number` (e.g., `"1"`, `"2"`) - **numeric string**, not `"hop0"`
   - Each hop represents a conversation turn

3. **`hop_number`** (e.g., `"1"`, `"2"`):
   - Key: `plugin_name` (e.g., `"office365_search_files"`, `"office365_search_people"`)
   - Value: **Array** of search objects (not a nested object)

4. **Search object** (element in plugin array):
   - `PluginName`: Plugin identifier (redundant with parent key)
   - `Results`: Array of search result objects
   - `PluginInvocation`: The full plugin call string with parameters (**query is embedded here**)

5. **`Results`** array contains result objects with:
   - `Type`: Content type (e.g., `"File"`, `"External_Connectors"`)
   - `Id`: Original item ID from the source system
   - `ReferenceId`: UUID format identifier (e.g., `"ce602c01-e9b1-13dc-0433-e9fbf11e5787.1000.3"`)
   - `Rank`: Position in search results (**0-indexed**)
   - `HitHighlightedSummary`: Content preview/snippet
   - `Source`: Object containing URL, Title, FileType, ModifiedBy, etc.
   - `Content`: Full document content
   - `Metadata`: Detailed metadata object
   - `CiteDCGLLMLabel`: **Citation quality score** (0-3 scale) - measures if this should be cited in the response
   - `LLMLabel`: **Search result quality score** (0-4 scale) - measures if this is a good search result
   - Additional fields: Prompts, Responses, and caching flags for both scoring systems
   
   **Note**: `CiteDCGLLMLabel` and `LLMLabel` are different scores - see [Two Scoring Systems](#two-scoring-systems) for details on the distinction between citation quality vs. search result quality.

**Important Differences from Extracted Format**:
- **PascalCase vs lowercase**: Raw uses `ReferenceId`, extracted uses `reference_id`
- **Hierarchy**: Raw uses `AllSearchResults[hop][plugin][array_index]`, extracted uses flat `turnData[].searches[]`
- **Field naming**: Raw uses `ConversationId`, extracted uses `conversation_id`
- **Plugin structure**: Raw uses array of search objects per plugin, extracted flattens to searches array
- **Score location**: `CiteDCGLLMLabel` is at the **result level**, not search level
- **Additional content**: Raw includes full `Content`, `Metadata`, `EvaluationData`, `Utterance`, `SydneyReply`

**Example Access Path**:
```python
# Raw structure access:
hop_num = "1"
plugin = "office365_search_files"
search_index = 0  # First search in the array
result_index = 0   # First result

score = data["AllSearchResults"][hop_num][plugin][search_index]["Results"][result_index]["CiteDCGLLMLabel"]
query = data["AllSearchResults"][hop_num][plugin][search_index]["Query"]

# Extracted structure access:
turn_index = 0     # turnData is 0-indexed
search_index = 0
result_index = 0

score = data["turnData"][turn_index]["searches"][search_index]["results"][result_index]["CiteDCGLLMLabel"]
query = data["turnData"][turn_index]["searches"][search_index]["query"]
```

---

### Extracted CiteDCG File Structure

After extraction by `extract_per_result_citedcg()`, the data is reorganized into a flatter, more accessible structure.

**File Format**: JSONL (JSON Lines / Newline-Delimited JSON)
- Each line is a complete, valid JSON object
- One conversation per line
- **File location**: `results/{job_id}_citedcg/{job_id}_citedcg_scores_{experiment}.json`

**Top-Level Structure**:

```json
{
  "conversation_id": "uuid-string",
  "turnData": [
    {
      "turn_index": 1,
      "searches": [
        {
          "plugin_name": "search_web",
          "search_domain": "search_web",
          "search_type": "search_web",
          "query": "full query text",
          "results": [
            {
              "reference_id": "turn1search1",
              "Type": "search_web_webpages",
              "CiteDCGLLMLabel": 2.5,
              "hop": 1,
              "Id": "original-item-id",
              "ReferenceId": "uuid.domain.rank",
              "result_id": "result_id_turn1search1",
              "Source": {...},
              "HitHighlightedSummary": "..."
            }
          ]
        }
      ]
    }
  ]
}
```

### Key Fields Explained

**Search-Level Fields**:
- **`plugin_name`**: Plugin used for search (e.g., `"search_web"`, `"office365_search_files"`)
- **`search_domain`**: Domain of search (e.g., `"search_web"`, `"office365"`)
- **`search_type`**: Type-specific identifier (for web: `"search_web_webpages"`, `"search_web_news"`)
- **`query`**: The actual search query text

**Result-Level Fields**:
- **`reference_id`**: Unique identifier for each search result (e.g., `"turn1search1"`, `"turn2search5"`)
  - **CRITICAL**: This is lowercase `reference_id`, not `ReferenceId`
  - Used to match with conversation data
- **`Type`**: Result type (e.g., `"search_web_webpages"`, `"search_web_news"`, `"File"`, `"Message"`)
- **`CiteDCGLLMLabel`**: Quality score (0-3 scale, higher = better)
- **`hop`**: Conversation hop/turn number where this result appeared
- **`Id`**: Original item ID (for emails/files)
- **`ReferenceId`**: UUID-format unique ID (e.g., `"c4faaa45-6d9d-cb5a-8f5c-dfcd004ea0e9.1000.8"`)
- **`result_id`**: Prefixed ID (e.g., `"result_id_turn1search1"`)

---

## CiteDCG Web Search Organization

**Important**: Web search results are organized by **Type** in CiteDCG, creating separate search entries for each type:

```json
{
  "searches": [
    {
      "plugin_name": "search_web",
      "search_type": "search_web_webpages",
      "query": "what is machine learning",
      "results": [
        {"reference_id": "turn1search0", "Type": "search_web_webpages"},
        {"reference_id": "turn1search2", "Type": "search_web_webpages"}
      ]
    },
    {
      "plugin_name": "search_web",
      "search_type": "search_web_news",
      "query": "what is machine learning",
      "results": [
        {"reference_id": "turn1search1", "Type": "search_web_news"}
      ]
    },
    {
      "plugin_name": "search_web",
      "search_type": "search_web_questionsandanswers",
      "query": "what is machine learning",
      "results": [
        {"reference_id": "turn1search3", "Type": "search_web_questionsandanswers"}
      ]
    }
  ]
}
```

**Key Points**:
- Single web search query → Multiple CiteDCG search entries (one per Type)
- Each entry has the same `query` text but different `search_type`
- Each entry contains results of only that specific Type

### Known Web Search Types

| Type                             | Description                      |
| -------------------------------- | -------------------------------- |
| `search_web_webpages`            | General web pages and articles   |
| `search_web_questionsandanswers` | Q&A sites (Stack Overflow, etc.) |
| `search_web_news`                | News articles and current events |

---

## CiteDCG Score Interpretation

### Two Scoring Systems

The raw DCG data contains **two different quality scores** for each search result:

#### 1. CiteDCG Score (`CiteDCGLLMLabel`)
- **Purpose**: Citation quality score
- **Question**: "Should this result be **cited** in the response?"
- **Scale**: 0-3 (can be decimal like 2.4)
- **Context**: Evaluates relevance for being cited in the bot's answer
- **Prompt type**: "You are a **citation quality rater**"
- **Fields**: `CiteDCGLLMLabel`, `CiteDCGLLMPrompt`, `CiteDCGLLMResponse`, `CiteDCGLLMCached`
- **Use case**: Measuring citation quality in responses

#### 2. Search Result Score (`LLMLabel`)
- **Purpose**: Search result quality score  
- **Question**: "Is this a good **search result** for the query?"
- **Scale**: 0-4 (can be decimal like 2.2)
- **Context**: Evaluates relevance to the search query
- **Prompt type**: "You are an **enterprise search engine quality rater**"
- **Fields**: `LLMLabel`, `LLMPrompt`, `LLMResponse`, `LLMCached`
- **Use case**: Measuring search engine quality

#### Why Two Scores?

A result can be:
- **Good search result but not cited** (relevant but not used in final answer)
- **Cited with high quality** (relevant and actually used effectively)
- **Good search result with poor citation** (found correctly but cited poorly)

The scores may differ because:
- Citation quality considers if the model **actually used** the information effectively
- Search quality evaluates if the result **matches the query** semantically

### Score Scale

**CiteDCG scores** (`CiteDCGLLMLabel` field):

| Score Range | Quality Level | Description                        |
| ----------- | ------------- | ---------------------------------- |
| 2.5 - 3.0   | Excellent     | Highly relevant, accurate, useful  |
| 2.0 - 2.4   | Good          | Relevant and accurate              |
| 1.5 - 1.9   | Fair          | Somewhat relevant, may have issues |
| 1.0 - 1.4   | Poor          | Low relevance or accuracy          |
| 0.0 - 0.9   | Very Poor     | Not relevant or incorrect          |

### Score Breakdown

The `CiteDCGLLMResponse` field contains detailed scoring dimensions:
- **T (Timeliness)**: Is information up-to-date?
- **F (Factual Accuracy)**: Is information correct?
- **P (Pertinence)**: Is information relevant to query?
- **O (Overall)**: Overall quality judgment

---

## Data Source Locations

### File Organization

| Data Type             | Location                                        | Format | Description                                     |
| --------------------- | ----------------------------------------------- | ------ | ----------------------------------------------- |
| **Raw SEVAL**         | `seval_data/{job_id}_scraping_raw_data_output/` | JSON   | Conversation files (control + experiment)       |
| **Raw DCG API**       | Raw DCG API responses                           | JSON   | Raw CiteDCG API responses                       |
| **Extracted CiteDCG** | `results/{job_id}_citedcg/`                     | JSONL  | Extracted quality scores (control + experiment) |
| **Merged Results**    | `results/{job_id}_merged/`                      | JSONL  | Conversation + CiteDCG joined                   |

### Control vs Experiment Files

**Control**:
- Conversation: `control_sydney_response_{conversation_id}.json`
- CiteDCG: `{job_id}_citedcg_scores_control.json`

**Experiment** (Treatment):
- Conversation: `experiment_sydney_response_{conversation_id}.json`
- CiteDCG: `{job_id}_citedcg_scores_experiment.json`

Each experiment has **both** control and experiment versions for comparison.

---

# Part 3: Data Matching and Merging

## Design Principles

### Data Preservation During Extraction

**CRITICAL PRINCIPLE**: All extraction processes must preserve the **exact original content and format** from raw input files.

#### Reference ID Preservation

When extracting reference IDs from raw data sources:

1. **Preserve Original Format**: The content/value of `reference_id` fields must match the raw input **exactly**
   - If raw CiteDCG API returns `"ReferenceId": "Turn1Search1"` → extract as `"reference_id": "Turn1Search1"`
   - If raw conversation data has `"turn1search1"` → preserve as `"reference_id": "turn1search1"`
   - If raw Office365 data has `"53c65d99-d953-fd1b-aafa-30d731024ab1.1000.1"` → preserve exactly

2. **Variable Names vs. Content**:
   - **Variable names** in code can use any meaningful style (e.g., `reference_id`, `ReferenceId`, `ref_id`)
   - **Content/values** must be preserved exactly as they appear in the raw input
   - Example: `reference_id = result.get("ReferenceId", "")` ✓ (variable name doesn't affect content)

3. **Field Name Mapping**:
   - Raw CiteDCG API uses PascalCase: `"ReferenceId"`
   - Extracted structure uses lowercase: `"reference_id"`
   - **But the VALUE inside must be preserved**: `"Turn1Search1"` stays `"Turn1Search1"`

#### Normalization During Matching

**When**: Only during the matching/merging phase (in `merge_seval_results.py`)

**How**: Convert reference IDs to lowercase for case-insensitive comparison:
```python
# Example matching logic
conv_ref_id = result.get("reference_id", "").lower()  # "turn1search1"
dcg_ref_id = dcg_result.get("reference_id", "").lower()  # "turn1search1"
if conv_ref_id == dcg_ref_id:
    # Match found - merge the data
```

**Why**: Different data sources may use different casing conventions (Turn1Search1 vs turn1search1), but they refer to the same result. Normalization only happens during comparison, not during extraction or storage.

#### Reference ID Reuse Rule

**CRITICAL RULE**: Do NOT deduplicate results by `reference_id` during extraction.

**Data Characteristics**:
- Same `reference_id` can **legitimately** appear in multiple hops (result reused across conversation turns)
- Same `reference_id` can appear in different web search types (webpages, questionsandanswers, news)
- Raw DCG data does NOT contain true duplicates (identical turn/hop/type/query)

**Correct Extraction Behavior**:
```python
# CORRECT: Extract all results as-is, no deduplication
for result in raw_dcg_results:
    extracted_results.append({
        "reference_id": result["ReferenceId"],
        "Type": result["Type"],
        "hop": result["Hop"],
        # ... other fields
    })
# Result: Same reference_id may appear multiple times (different hops/types)
```

**Example - Result Reuse Across Hops**:
```json
// Hop 2 results
[
  {"reference_id": "turn1search1", "hop": 2, "Type": "search_web_webpages"},
  {"reference_id": "turn1search6", "hop": 2, "Type": "search_web_webpages"}
]

// Hop 3 results (turn1search1 and turn1search6 reused)
[
  {"reference_id": "turn1search1", "hop": 3, "Type": "search_web_webpages"},
  {"reference_id": "turn1search7", "hop": 3, "Type": "search_web_webpages"},
  {"reference_id": "turn1search6", "hop": 3, "Type": "search_web_webpages"},
  {"reference_id": "turn1search10", "hop": 3, "Type": "search_web_webpages"}
]
```

**Data Rules for Implementation**:
- ✅ **REQUIRED**: Preserve all occurrences of same `reference_id` across different hops
- ✅ **REQUIRED**: Preserve all occurrences of same `reference_id` across different web search types
- ✅ **REQUIRED**: Extract results exactly as they appear in raw data
- ❌ **FORBIDDEN**: Global deduplication by `(reference_id, Type)` or any other key
- ❌ **FORBIDDEN**: Filtering duplicate `reference_id` values during extraction

#### Extraction Stage Summary

| Stage          | Action                          | Example                                                            |
| -------------- | ------------------------------- | ------------------------------------------------------------------ |
| **Extraction** | Preserve exact original content | `"ReferenceId": "Turn1Search1"` → `"reference_id": "Turn1Search1"` |
| **Storage**    | Keep original content unchanged | Store as `"reference_id": "Turn1Search1"` in extracted JSON        |
| **Matching**   | Normalize for comparison only   | Compare `"Turn1Search1".lower()` == `"turn1search1".lower()`       |

---

## Reference ID Mapping

### The Core Mapping Rule

**For `turn1searchX` format reference IDs**:
- `turn1search0` = 1st result (index 0) when CiteDCG results are sorted by Rank
- `turn1search5` = 6th result (index 5) when CiteDCG results are sorted by Rank
- `turn1search36` = 37th result (index 36) when CiteDCG results are sorted by Rank

**For email/file IDs**: Match using the `Id` field or `result_id` field directly.

### Reference ID Format

**Web/People Results** (most common):
- Format: `turn{N}search{X}` where N = turn number, X = 0-based index
- Examples: `turn1search0`, `turn1search5`, `turn2search12`
- Mapping: Extract index X, lookup in sorted CiteDCG results at index X

**Email/File/Chat Results**:
- May have both `reference_id` (turn1searchX format) AND `id` field
- Email ID format: `AAMkADM4OTQ4MjQxLTU0NmQtNGNmOS04MjgxLWNjNDY2MWRjNzg4Zg...`
- SharePoint ID format: `SPO_ZjlmOWJhZWYtYmRlNS00ZWU3LTlmMTMtMTk5M2E1MGZiYmJj...`
- Matching: Try `reference_id` first, fallback to `id` field

**CiteDCG UUID Format**:
- Format: `{uuid}.{domain}.{rank}`
- Example: `c4faaa45-6d9d-cb5a-8f5c-dfcd004ea0e9.1000.3`
- Usage: Internal CiteDCG tracking (not used for conversation matching)

### ID Field Mapping Table

| Result Type      | Conversation `reference_id` | Conversation `id` | CiteDCG Matching Field  |
| ---------------- | --------------------------- | ----------------- | ----------------------- |
| Web results      | `turn1search5`              | (none)            | Sort by Rank, use index |
| Email messages   | `turn1search12`             | `AAMk...`         | `result_id` or `Id`     |
| SharePoint files | `turn1search8`              | `SPO_...`         | `result_id` or `Id`     |
| People results   | `turn1search43`             | (may be empty)    | Sort by Rank, use index |
| Chat messages    | `turn1search22`             | Message ID        | `result_id` or `Id`     |

---

## Type-Aware Matching Logic

### The Web Search Type Problem

**Issue**: Web searches return mixed types (webpages, news, Q&A) in a single API call, but conversation and CiteDCG organize them separately by type.

**Raw Web Search Result**:
```json
{
  "search_results": [
    {"reference_id": "turn1search0", "Type": "search_web_webpages"},
    {"reference_id": "turn1search1", "Type": "search_web_news"},
    {"reference_id": "turn1search2", "Type": "search_web_webpages"},
    {"reference_id": "turn1search3", "Type": "search_web_questionsandanswers"}
  ]
}
```

**Conversation Organization** (split by domain):
```json
{
  "searches": [
    {
      "search_domain": "webpages",
      "results": [
        {"reference_id": "turn1search0", "Type": "search_web_webpages"},
        {"reference_id": "turn1search2", "Type": "search_web_webpages"}
      ]
    },
    {
      "search_domain": "news",
      "results": [
        {"reference_id": "turn1search1", "Type": "search_web_news"}
      ]
    }
  ]
}
```

**CiteDCG Organization** (split by Type):
```json
{
  "searches": [
    {
      "search_type": "search_web_webpages",
      "results": [
        {"reference_id": "turn1search0", "Type": "search_web_webpages"},
        {"reference_id": "turn1search2", "Type": "search_web_webpages"}
      ]
    },
    {
      "search_type": "search_web_news",
      "results": [
        {"reference_id": "turn1search1", "Type": "search_web_news"}
      ]
    }
  ]
}
```

### Solution: Type-Aware Matching

**For Web Searches**: Match using both query AND Type
```python
if plugin_name == 'search_web':
    # Match requires both query and Type to match
    citedcg_match = find_search(
        citedcg_searches,
        query=conv_query,
        search_type=conv_search_type  # Must match Type exactly
    )
```

**For Non-Web Searches**: Match using query AND plugin_name
```python
else:
    # Match requires query and plugin_name
    citedcg_match = find_search(
        citedcg_searches,
        query=conv_query,
        plugin_name=conv_plugin_name
    )
```

### Type Mapping Table

| Search Plugin             | Conversation `search_domain` | CiteDCG `search_type`            | Type Field                       |
| ------------------------- | ---------------------------- | -------------------------------- | -------------------------------- |
| `search_web`              | `webpages`                   | `search_web_webpages`            | `search_web_webpages`            |
| `search_web`              | `news`                       | `search_web_news`                | `search_web_news`                |
| `search_web`              | `questionsandanswers`        | `search_web_questionsandanswers` | `search_web_questionsandanswers` |
| `office365_search_files`  | `office365`                  | `office365_search_files`         | `File` or `DriveItem`            |
| `office365_search_emails` | `office365`                  | `office365_search_emails`        | `Message` or `EmailMessage`      |

---

## Matching Algorithms

### Complete Matching Flow

```
1. Load conversation and CiteDCG files
   ↓
2. Extract turn_index and searches from both
   ↓
3. For each conversation search:
   a. Extract query, search_type, plugin_name
   b. Find matching CiteDCG search:
      - Web: Match by (query, search_type)
      - Other: Match by (query, plugin_name)
   ↓
4. For each result in matched search:
   a. Extract reference_id from both
   b. Normalize to lowercase for comparison
   c. Match if reference_id values match
   ↓
5. Merge matched results:
   - Keep all conversation fields
   - Add CiteDCG quality scores
   - Add CiteDCG metadata
```

### Basic Matching Algorithm

```python
def match_search_results(conversation_searches, citedcg_searches):
    """
    Match conversation searches with CiteDCG searches
    
    For web searches: Match by (query, Type)
    For other searches: Match by (query, plugin_name)
    """
    matches = []
    
    for conv_search in conversation_searches:
        query = conv_search['query'].lower().strip()
        search_type = conv_search['search_type']
        plugin = conv_search['plugin_name']
        
        # Find matching CiteDCG search
        if plugin == 'search_web':
            # Web search: Match by query AND Type
            citedcg_match = next(
                (s for s in citedcg_searches 
                 if s['query'].lower().strip() == query 
                 and s['search_type'] == search_type),
                None
            )
        else:
            # Non-web search: Match by query AND plugin_name
            citedcg_match = next(
                (s for s in citedcg_searches 
                 if s['query'].lower().strip() == query 
                 and s['plugin_name'] == plugin),
                None
            )
        
        if citedcg_match:
            # Match results within this search
            matched_results = match_results_within_search(
                conv_search['results'],
                citedcg_match['results']
            )
            matches.append({
                'conversation_search': conv_search,
                'citedcg_search': citedcg_match,
                'matched_results': matched_results
            })
    
    return matches

def match_results_within_search(conv_results, citedcg_results):
    """Match individual results by reference_id"""
    matches = []
    
    for conv_result in conv_results:
        conv_ref_id = conv_result.get('reference_id', '').lower()
        
        citedcg_match = next(
            (r for r in citedcg_results 
             if r.get('reference_id', '').lower() == conv_ref_id),
            None
        )
        
        if citedcg_match:
            matches.append({
                'conversation_result': conv_result,
                'citedcg_result': citedcg_match,
                'quality_score': citedcg_match['CiteDCGLLMLabel']
            })
    
    return matches
```

### Unified Result Mapping

```python
def map_conversation_to_citedcg(conv_result, sorted_citedcg_results):
    """
    Map a conversation result to CiteDCG score using all available IDs
    
    Args:
        conv_result: Dict with 'reference_id' and/or 'id' fields
        sorted_citedcg_results: All CiteDCG results sorted by Rank
    
    Returns:
        CiteDCG result dict with quality score, or None
    """
    reference_id = conv_result.get('reference_id', '')
    item_id = conv_result.get('id', '')
    
    # Strategy 1: Try reference_id first (turn1searchX format)
    if reference_id and reference_id.startswith('turn') and 'search' in reference_id:
        try:
            # Extract index: "turn1search36" -> 36
            index = int(reference_id.split('search')[1])
            if index < len(sorted_citedcg_results):
                return sorted_citedcg_results[index]
        except (ValueError, IndexError):
            pass
    
    # Strategy 2: Fall back to item ID (email/file/chat)
    if item_id:
        # Try result_id field
        match = next(
            (r for r in sorted_citedcg_results 
             if r.get('result_id') == f'result_id_{item_id}'),
            None
        )
        if match:
            return match
        
        # Try Id field
        match = next(
            (r for r in sorted_citedcg_results 
             if r.get('Id') == item_id),
            None
        )
        if match:
            return match
    
    return None  # No match found
```

---

## Practical Examples

### Example 1: Complete Mapping Workflow

**Query**: "What are the best practices for creating a scorecard?"

**Step 1**: Load CiteDCG data
```python
import json

with open('results/citedcg_scores_control.json', 'r') as f:
    lines = f.readlines()
    citedcg_data = [json.loads(line) for line in lines]

# Find entry for this query
query_text = "what are the best practices for creating a scorecard?"
entry = [d for d in citedcg_data 
         if d.get('Utterance', '').lower() == query_text][0]
```

**Step 2**: Extract and sort all results
```python
# Get all results from turn 1
turn_data = entry['turnData'][0]  # Turn 1
all_results = []

for search in turn_data['searches']:
    all_results.extend(search['results'])

# Sort by Rank field (if using AllSearchResults structure)
# or keep as-is if already in correct order
sorted_results = sorted(all_results, key=lambda x: x.get('Rank', 999))

print(f"Total CiteDCG results: {len(sorted_results)}")
```

**Step 3**: Map conversation citations
```python
# Citations from final response
citations = ['turn1search5', 'turn1search36', 'turn1search39']

# Get quality scores
for ref_id in citations:
    index = int(ref_id.replace('turn1search', ''))
    if index < len(sorted_results):
        result = sorted_results[index]
        score = result['CiteDCGLLMLabel']
        result_type = result['Type']
        print(f"{ref_id}: {result_type} - Quality Score: {score}")
```

**Step 4**: Calculate citation quality metrics
```python
scores = []
for ref_id in citations:
    index = int(ref_id.replace('turn1search', ''))
    if index < len(sorted_results):
        scores.append(sorted_results[index]['CiteDCGLLMLabel'])

avg_quality = sum(scores) / len(scores)
max_quality = max(scores)

print(f"Average citation quality: {avg_quality:.2f}")
print(f"Best citation quality: {max_quality:.2f}")
```

### Example 2: Web Search with Multiple Types

**Conversation Data**:
```python
conversation_searches = [
    {
        "search_domain": "webpages",
        "search_type": "search_web_webpages",
        "query": "python async programming",
        "results": [
            {"reference_id": "turn1search0", "Type": "search_web_webpages"},
            {"reference_id": "turn1search2", "Type": "search_web_webpages"}
        ]
    },
    {
        "search_domain": "questionsandanswers",
        "search_type": "search_web_questionsandanswers",
        "query": "python async programming",
        "results": [
            {"reference_id": "turn1search1", "Type": "search_web_questionsandanswers"}
        ]
    }
]
```

**Matching Code**:
```python
def match_web_searches(conv_searches, citedcg_searches):
    matches = []
    
    for conv_search in conv_searches:
        query = conv_search['query'].lower().strip()
        search_type = conv_search['search_type']
        
        # Find CiteDCG search with matching query AND Type
        citedcg_match = next(
            (s for s in citedcg_searches 
             if s['query'].lower().strip() == query 
             and s['search_type'] == search_type),
            None
        )
        
        if citedcg_match:
            matches.append({
                'conversation': conv_search,
                'citedcg': citedcg_match,
                'type': search_type
            })
    
    return matches

# Execute matching
matched = match_web_searches(conversation_searches, citedcg_entry['turnData'][0]['searches'])
print(f"Matched {len(matched)} web searches")
```

---

## Common Pitfalls

### ❌ Pitfall 1: Using Plugin Name Only for Web Search Matching

**Problem**: Web searches with the same query but different types get matched incorrectly.

**Wrong**:
```python
# BAD: Matches any web search with same query, ignoring Type
citedcg_match = next(
    (s for s in citedcg_searches 
     if s['query'] == conv_query and s['plugin_name'] == 'search_web'),
    None
)
```

**Correct**:
```python
# GOOD: Matches web search by query AND Type
citedcg_match = next(
    (s for s in citedcg_searches 
     if s['query'] == conv_query and s['search_type'] == conv_search_type),
    None
)
```

### ❌ Pitfall 2: Case Sensitivity in Reference IDs

**Problem**: Reference IDs may have different casing in different sources.

**Solution**: Normalize to lowercase during comparison (not during extraction/storage)
```python
# Correct: Normalize for comparison only
conv_ref_id = result.get("reference_id", "").lower()
dcg_ref_id = dcg_result.get("reference_id", "").lower()
if conv_ref_id == dcg_ref_id:
    # Match found
```

### ❌ Pitfall 3: Deduplicating by Reference ID

**CRITICAL**: Do NOT deduplicate results by `reference_id` during extraction.

See [Design Principles > Reference ID Reuse Rule](#reference-id-reuse-rule) for complete details.

### ❌ Pitfall 4: Truncating Query Text for Matching

**Problem**: Truncating queries leads to incorrect matches.

**Wrong**:
```python
# BAD: Truncation causes false matches
if dcg_query[:50] == conv_query[:50]:
    # This will match "What is X" with "What is Y"
```

**Correct**:
```python
# GOOD: Full text comparison (case-insensitive)
if dcg_query.lower().strip() == conv_query.lower().strip():
    # Exact match (except casing/whitespace)
```

---

## Troubleshooting

### Issue 1: Index Out of Range

**Problem**: `turn1search36` gives index error

**Cause**: CiteDCG has fewer results than conversation file references

**Explanation**: This is **normal** - CiteDCG may evaluate fewer results than were initially searched.

**Solution**:
```python
index = int(ref_id.replace('turn1search', ''))
if index < len(sorted_results):
    result = sorted_results[index]
else:
    print(f"Warning: {ref_id} not in CiteDCG evaluation set")
```

### Issue 2: Query Not Found

**Problem**: Can't find matching query in CiteDCG file

**Cause**: Query text doesn't match exactly (different casing, punctuation, or slight rewording)

**Solution**:
```python
import re

def normalize_query(query):
    return re.sub(r'[^\w\s]', '', query.lower().strip())

query_match = next(
    (d for d in citedcg_data 
     if normalize_query(d.get('Utterance', '')) == normalize_query(your_query)),
    None
)
```

### Issue 3: Web Search Type Mismatch

**Problem**: Conversation web search not matching any CiteDCG search

**Cause**: Matching by plugin_name only, not checking Type

**Solution**: Match web searches by BOTH query AND search_type
```python
# For web searches, search_type must match exactly
if plugin_name == 'search_web':
    citedcg_match = next(
        (s for s in citedcg_searches 
         if s['query'].lower() == conv_query.lower() 
         and s['search_type'] == conv_search_type),
        None
    )
```

---

## Implementation Checklist

When implementing or debugging extraction and matching logic, verify:

- [ ] **Data Preservation**: Extract all fields exactly as they appear in raw data
- [ ] **Reference ID Format**: Preserve original casing during extraction, normalize during matching
- [ ] **No Deduplication**: Extract all results as-is (same reference_id allowed across hops/types)
- [ ] **Reference ID Reuse**: Preserve all occurrences of same reference_id in different hops or search types
- [ ] **Turn Index Matching**: Use correct turn number when matching
- [ ] **Query Text Matching**: Case-insensitive, full text comparison (no truncation)
- [ ] **Web Search Type Handling**: Match by query AND search_type for web searches
- [ ] **Non-Web Search Matching**: Match by query AND plugin_name for non-web searches
- [ ] **Type Field Usage**: Check `Type` field in CiteDCG results, not just `plugin_name`
- [ ] **Error Handling**: Gracefully handle missing matches (conversation may have queries not in CiteDCG)
- [ ] **Index Bounds**: Check array bounds before accessing by index
- [ ] **ID Fallback**: Try reference_id first, then fallback to item id if available

---

## Code References

**Extraction**:
- `seval_analysis_toolkit.py` lines 4329-4463: Web search extraction with type separation
- `get_seval_metrics.py` lines 670-850: CiteDCG score extraction

**Matching**:
- `merge_seval_results.py` lines 350-440: Search matching logic with type handling

**Key Implementation** (Lines 400-411 in `merge_seval_results.py`):
- Type-aware matching for web searches (matches by Type field, not just plugin_name)

---

**End of Document**
