# Seval Conversations TSV Format

The **converged** Seval scraper pipeline produces TSV files for both
single-turn and multi-turn evaluation. All variants share the same
7-column schema and col 6 result structure. Three variants exist:

- **Single-turn** — One user utterance per row, one result per row.
  Used for standard single-hop evaluation across many segments.
- **Multi-turn echo** — Pre-scripted user utterances replayed verbatim
  regardless of prior-turn outputs. Used for controlled A/B testing.
- **Multi-turn log-based** — A HumanBot adjusts subsequent utterances
  based on the input info (prior utterances and assumed results)
  combined with the actual results, then sends them for scraping.

Scraping results are stored in `{arm}_multiturnconversations.tsv`
where `{arm}` is the experiment arm name (e.g. `control`, `experiment`,
or `text`). Each TSV file contains all dataset rows and their
inference results.

---

## Common Schema (7 columns, tab-separated, no header row)

| Col | Name (inferred) | JSON Type | Description |
|-----|------------------|-----------|-------------|
| 0 | `conversations` | `array` of objects | Ordered list of conversation turns. Each element: `{"text": str, "author": "user"\|"bot", ...}`. |
| 1 | `metadata` | `dict` | Per-row context for the utterance(s) in col 0: segment, HumanBot mode, max turns, user/timestamp context. |
| 2 | `arm` | `string` | Experiment arm name (e.g. `"control"`, `"experiment"`). |
| 3 | `sydney_config` | `dict` | Sydney endpoint and flight configuration. |
| 4 | `humanbot_config` | `dict` | HumanBot LLM endpoint and prompt template. |
| 5 | `num_turns` | `int` | Maximum number of conversation turns allowed. |
| 6 | `results` | `array` of objects | Full Sydney response trace for each completed turn (~75 fields per turn). |

---

## Column Details

### Column 0 — `conversations`

JSON array of turn objects. Each turn:

```json
{
  "text": "Summarize the meeting customer feedback sync.",
  "author": "user",
  "location": {"country": "", "city": "", "timeZoneOffset": -7}
}
```

- `text` — The utterance text.
- `author` — `"user"` or `"bot"`. Single-turn and echo have user-only;
  log-based has both.
- `location` — (optional) User location context for the turn
  (present in echo mode only).

### Column 1 — `metadata`

```json
{
  "run_name": "",
  "segment": "5.03a - Recap: Past Meeting",
  "humanbot": "HumanBotEchoMessage",
  "max_turns": 3,
  "sydney": {
    "grounding_data_source": "Work",
    "timestamp": "2024-11-21T17:00:00Z",
    "extra_headers": {"X-ScraperService-UserId": "..."}
  },
  "querygen_rules": "N/A",
  "past_conversations": "[]",
  "rule_scenario": "default"
}
```

| Key | Type | Description |
|-----|------|-------------|
| `run_name` | str | Run identifier (may be empty). |
| `segment` | str | Evaluation segment/category name. |
| `humanbot` | str | HumanBot mode (e.g. `"HumanBotEchoMessage"`, or empty for log-based). Absent in single-turn. |
| `max_turns` | int | Max turns the scraper will execute. |
| `sydney` | dict | **Per-row** simulation context: data source, timestamp, user identity headers. Not to be confused with col 3 `sydney_config` which holds the shared endpoint/flight config (see note below). |
| `querygen_rules` | str | Query generation rules (usually `"N/A"`). |
| `past_conversations` | str | Prior conversation context (usually `"[]"`). |
| `rule_scenario` | str | Scenario tag (usually `"default"`). |

### Column 2 — `arm`

Plain string. The experiment arm name: `"control"`, `"experiment"`, or
`"text"` (single-turn).

### Column 3 — `sydney_config`

> **Note — Col 1 `sydney` vs Col 3 `sydney_config`:** These two fields
> serve different purposes despite similar names. Col 1's `sydney` dict
> carries **per-row simulation context** (which user to impersonate, what
> timestamp to simulate, grounding data source) — it varies across rows.
> Col 3 `sydney_config` carries the **endpoint and experiment
> configuration** (URL, flights/option sets, plugins, API version) —
> it is typically identical for all rows within an arm.

```json
{
  "url": "https://substrate.office.com/m365Copilot",
  "option_sets": "enterprise_flux_web,enterprise_flux_work,...",
  "extra_headers": {"X-Variants": "..."},
  "extra_params": {"debug": "True", "scenario": "officeweb", ...},
  "config_params": {"chat_api_version": "v1"},
  "plugins": [{"Version": "1.0", "id": "EnterpriseSearch", ...}],
  "extra_fields": {"tone": "Magic"},
  "options": {}
}
```

| Key | Type | Description |
|-----|------|-------------|
| `url` | str | Sydney/Substrate endpoint URL. |
| `option_sets` | str | Comma-separated flight/option-set names. |
| `extra_headers` | dict | Additional HTTP headers (e.g. X-Variants). |
| `extra_params` | dict | Additional query parameters (debug flags, scenario). |
| `config_params` | dict | API version config. |
| `plugins` | list | Plugin definitions (e.g. EnterpriseSearch). |
| `extra_fields` | dict | Extra request fields (e.g. tone). |
| `options` | dict | Additional options (usually empty). |

### Column 4 — `humanbot_config`

```json
{
  "dv3_ep": "https://prom-gpt4-turbo-nexsing-eval.eastus.inference.ml.azure.com/v1/engines/davinci/completions",
  "prompt_file": "HumanBotV5"
}
```

| Key | Type | Description |
|-----|------|-------------|
| `dv3_ep` | str | GPT-4 Turbo endpoint for generating follow-up user turns. |
| `prompt_file` | str | Prompt template name for the HumanBot. |

### Column 5 — `num_turns`

Integer. Maximum number of conversation turns.

### Column 6 — `results`

JSON array of per-turn result objects. Each element represents one
completed conversation turn with the full Sydney response trace.
**This is the largest column** (up to ~3 MB per row in echo mode).
See [Column 6 Results — Reasoning & Tool Call Structure](#column-6-results--reasoning--tool-call-structure)
for the full breakdown.

Key fields per turn element:

| Key | Type | Description |
|-----|------|-------------|
| `Human` | str | User utterance for this turn. |
| `SydneyReply` | list | Bot response text(s). |
| `SydneyCitations` | list | Citation objects. |
| `filtered_search` | list | Tool invocations (search_office365, etc.). |
| `sonic_filtered_search` | list | Sonic/reasoning tool invocations. |
| `turn_memory` | list | Accumulated tool invocations/results across turns. |
| `ConversationId` | str | Conversation session ID. |
| `ReasoningCounts` | int | Number of reasoning steps. |
| `claims` | list | Grounding claims for the response. |
| `SharedSessionId` | str | Playground link for the session. |
| `UserInfo` | dict | Simulated user profile (name, department, etc.). |
| `AvailableTools` | list | Tools available to the model. |
| `attributions` | dict | Source attribution URLs. |

(~73 fields total per turn; see the reasoning & tool call section below
for the fields most relevant to evaluation.)

---

## Variant Differences

### Single-Turn

| Property | Value |
|----------|-------|
| **Source file** | `text_multiturnconversations (3).tsv` |
| **Rows** | 237 |
| **Turns per row** | 1 (single user utterance in col 0, single result in col 6) |
| **Col 0 element keys** | `text`, `author` (no `location`) |
| **Col 1 `humanbot`** | Key absent |
| **Col 1 `max_turns`** | 1 |
| **Col 1 `sydney` sub-keys** | `timestamp`, `extra_headers`, sometimes `plugins` |
| **Col 2 arm** | `"text"` |
| **Col 3 config** | Identical for all rows (single endpoint) |
| **Col 4 `prompt_file`** | `null` (HumanBot not used) |
| **Col 5** | `1` |
| **Col 6 size** | 27–243 KB per row (avg 67 KB) |
| **Col 6 result keys** | 75–76 per item (76 total; `acf_card_triggered` in 11/237) |
| **Col 6 extra keys** | `Bot_Cutoff_Score`, `Bot_Interrupt_Score`, `Bot_Transcript_Score`, `Human_Transcript`, `audio_paths`, `original_idx` |
| **Segments** | 48 distinct (largest: "5.03a - Recap: Past Meeting" × 48) |
| **Reasoning pattern** | `sonic_model_actions` always empty; `ReasoningToolGroups` present in 235/237; `filtered_search` present in 235/237 |

**Example row (col 0):**
```json
[{"text": "Summarize my Thursday's first meeting", "author": "user"}]
```

### 403398 — Echo Mode (`HumanBotEchoMessage`)

| Property | Value |
|----------|-------|
| **Source folder** | `403398_multiturn_echo/` |
| **Rows** | 6 |
| **Turns per conversation** | 3 (user-only in col 0) |
| **`metadata.humanbot`** | `"HumanBotEchoMessage"` |
| **`metadata.max_turns`** | 3 |
| **Col 0 authors** | `"user"` only |
| **Col 0 element keys** | `text`, `author`, `location` |
| **Col 6 avg size** | ~3 MB per row |
| **Segments** | `"5.03a - Recap: Past Meeting"` (all rows) |
| **Conversation style** | Pre-scripted user turns. Bot responses only in col 6 results. |

**Example conversation:**
> [user] Summarize the meeting customer feedback sync.
> [user] Can you give me a list of the main action items?
> [user] Maybe in an email or a shared document.

### 403400 — Log-Based Mode

| Property | Value |
|----------|-------|
| **Source folder** | `403400_multiturn_logbased/` |
| **Rows** | 4 |
| **Turns per conversation** | 8 (4 user + 4 bot in col 0) |
| **`metadata.humanbot`** | `""` (empty) |
| **`metadata.max_turns`** | 10 |
| **Col 0 authors** | Both `"user"` and `"bot"` |
| **Col 0 element keys** | `text`, `author` (no `location`) |
| **Col 6 avg size** | ~250–330 KB per row |
| **Col 6 actual turn count** | Varies: 2, 4, 4, 3 |
| **Segments** | `"Scheduling and availability queries"`, `"Location based queries"`, `"Recurring meetings"` |
| **Conversation style** | Natural multi-turn with bot responses interleaved. Conversations include back-and-forth with tool use, clarifications, and follow-ups. |

**Example conversation:**
> [user] Are there any scheduling conflicts between my meetings today?
> [bot]  I'll check your calendar for today and look for any overlapping...
> [user] Okay, and do I have any back-to-back meetings with no break time?
> [bot]  Yes — I've checked your meetings today, and you do have back-to-back...
> [user] Yeah, can you add like a 15-minute buffer after the workshop?
> [bot]  I can do that — just one quick clarification...
> [user] Yeah, go ahead and move the 1:1 to 11:15.
> [bot]  All set! I've moved your 1:1 meeting to start at 11:15 PM...

---

## Column 6 Results — Reasoning & Tool Call Structure

Each turn-result object in col 6 has ~73 keys. The fields relevant to
**reasoning**, **tool calls**, and **evaluation** are documented below.

> **Important**: Neither `telemetry.metrics` nor `EvaluationData` appear in
> any variant's results. The scraper has already extracted and flattened
> the data that `foundry_preeval_script.py` would parse from raw
> `DeepLeoImprovedNetworking` metrics. The equivalent data is spread
> across several fields depending on the variant:
>
> | Data | Echo mode | Single-turn / Log-based |
> |------|-----------|------------------------|
> | Reasoning decisions (which tools to call) | `sonic_model_actions[*].toolInvocations` | `ReasoningToolGroups` |
> | Tool call results | `sonic_filtered_search` / `filtered_search` | `filtered_search` |
> | Chain-of-thought | `sonic_model_actions[*].chainOfThought` | *Not captured* |
> | Token counts | `sonic_model_actions[*].additionalMetrics.metrics` | *Not captured* |
> | Available tools for reasoning | `sonic_model_actions[*].availableTools` | `AvailableToolsReasoningRoot` |

### Reasoning: `sonic_model_actions[]`

Array of per-hop reasoning model calls within a single conversation turn.
Each element represents one reasoning loop (L1, L2, ..., Ln).

| Field | Type | Description |
|-------|------|-------------|
| `model` | str | Model deployment name (e.g. `prod-workberry-slot-a-ft`). |
| `modelApi` | str | API endpoint path (e.g. `/chat/completions`). |
| `modelTags` | list | Tags including correlation ID with `L{loop}:P{position}` suffix. |
| `chainOfThought` | str | Reasoning text emitted by the model (e.g. "Searching across all domains..."). |
| `modelOutput` | str | Raw DeepLeo output string with CallTags, token counts, etc. |
| `toolInvocations` | list | **Already-parsed** tool calls for this hop (see below). |
| `additionalMetrics.metrics` | dict | Token counts: `PromptToken`, `CompletionToken`, `CachedToken`, `ReasoningToken`, `DeepLeoInputPrompt`, `DeepLeoOutputStream`. |
| `prompt` | str | Full prompt sent to the model (can be very large, e.g. 80 KB+). |
| `availableTools` | list | Tools available to the reasoning model. |
| `latencyMilliseconds` | int | Per-hop latency. |
| `startTime` / `endTime` | str | Timestamps for the hop. |

#### `toolInvocations[]` element

| Field | Type | Description |
|-------|------|-------------|
| `function` | str | Tool name (e.g. `"office365_search"`). |
| `arguments` | str | JSON string of tool arguments. |
| `processedResult` | str | Tool result after post-processing. |
| `result` | str | Raw tool result (often empty; data is in `processedResult`). |
| `callId` | str | Unique call identifier. |
| `batchedQueries` | varies | Batched query details (if applicable). |
| `referenceIdProperty` | str | Reference ID for citation linking. |

### Tool Calls: `sonic_filtered_search[]` and `filtered_search[]`

Both arrays contain objects with `{tool_invocation, result, source}`.

| Field | Type | Description |
|-------|------|-------------|
| `tool_invocation` | str or dict | Tool call description. In echo mode: raw string like `office365_search(queries="[...]")`. In log-based mode: same format for `filtered_search`. |
| `result` | str | Tool execution result (can be very large, e.g. 56 KB for aggregated search). |
| `source` | str/int | Source indicator (typically `0`). |

**`sonic_filtered_search`** — Tool calls made during reasoning loops (corresponds 1:1 with `sonic_model_actions[*].toolInvocations`).

**`filtered_search`** — All tool call results for the turn. In echo mode these are non-reasoning tool calls (e.g. `search_office365`); in log-based mode **this is the primary location for all tool results** including reasoning-initiated ones.

### Reasoning Summary: `ReasoningToolGroups[]`

A flat list of strings, each summarizing the tool calls decided by one
reasoning iteration. Present in **both** variants when `ReasoningCounts > 0`.

In echo mode this field has the same tool calls as `sonic_model_actions`.
In log-based mode — where `sonic_model_actions` is empty — **this is the
only trace of what the reasoning model decided to call**.

Example (logbased turn 0):

```json
[
  "office365_search(queries=\"[{'domain': 'meetings', 'query': \"today's meetings\", ...}]\"}, TimeInsightsInternal_get_time_insights(startDate=\"2024-08-05\", endDate=\"2024-08-05\")"
]
```

Each string groups the tool calls from a single reasoning loop.
`len(ReasoningToolGroups)` should match `ReasoningCounts`.

### Reasoning-Available Tools: `AvailableToolsReasoningRoot[]`

List of tool description strings showing what tools the reasoning model
could choose from. Present in both variants. Each entry is a docstring
in the format:

```
`office365_search(queries: array) -> str` Search the user's enterprise...
```

Distinct from `AvailableTools` which uses `>>> help(...)` format and
may include tools not offered to the reasoning model.

### Other Relevant Fields

| Field | Type | Description |
|-------|------|-------------|
| `Human` | str | User utterance for this turn. |
| `SydneyReply` | list of str | Bot response text(s). |
| `SydneyCitations` | list | Citation objects for the response. |
| `ReasoningCounts` | int | Number of reasoning iterations. Matches `len(sonic_model_actions)` in echo mode; matches `len(ReasoningToolGroups)` in log-based mode. |
| `ReasoningToolGroups` | list of str | Summary of tool calls per reasoning iteration (see above). |
| `AvailableToolsReasoningRoot` | list of str | Tool descriptions available to the reasoning model (see above). |
| `claims` | list | Grounding claims extracted from the response. |
| `attributions` | dict | Source attribution URLs keyed by citation index. |
| `turn_memory` | list | Accumulated tool invocations/results carried forward across turns. |
| `ConversationId` | str | Conversation session ID. |
| `SharedSessionId` | str | Playground URL link for the session. |
| `ExtensionRunner` | list | Extension/feature flag configuration (~200-300 items, one dict each). |
| `option_sets` | list | Active option sets / flights (~400 items). |

### Reasoning & Tool Call Comparison Across Variants

| Aspect | Echo (403398) | Single-turn / Log-based (403400) |
|--------|---------------|-------------------|
| **`sonic_model_actions`** | Present on reasoning turns (e.g. 7 hops for turn 0) | **Always empty** — per-hop details not captured |
| **`sonic_filtered_search`** | Present on reasoning turns (matches `toolInvocations` count) | **Always empty** |
| **`ReasoningToolGroups`** | Present (same tool calls as sonic_model_actions) | **Present** — sole record of reasoning decisions |
| **`AvailableToolsReasoningRoot`** | Present (9-12 tool descriptions) | **Present** (9 tool descriptions) |
| **`filtered_search`** | Non-reasoning tool calls (e.g. `search_office365`) | **All tool call results** (e.g. `office365_search`, `TimeInsightsInternal`, `CalendarAiPlacesQuery`) |
| **`chainOfThought`** | Available per hop in `sonic_model_actions` | **Not captured** by scraper |
| **Token counts** | In `sonic_model_actions[*].additionalMetrics.metrics` | **Not captured** by scraper |
| **`ReasoningCounts`** | 0 or 1 (1 = reasoning used) | 0, 1, or 2 |
| **Reasoning evidence** | Full per-hop trace in `sonic_model_actions` | `ReasoningCounts` + `ReasoningToolGroups` + `filtered_search` results |
| **Result size per row** | ~3 MB (prompts + chain-of-thought) | Single: 27–243 KB (avg 67 KB); Log-based: 250–330 KB |

#### Echo Mode — Turn-by-Turn Example (Row 0)

| Turn | Query | sonic_model_actions | sonic_filtered_search | filtered_search | Claims | Citations |
|------|-------|--------------------|-----------------------|-----------------|--------|-----------|
| 0 | "Summarize the meeting customer feedback sync." | **7 hops** (L1-L7), 6 tool calls | 6 × `office365_search` | 1 × `search_office365` | 9 | 3 |
| 1 | "Can you give me a list of the main action items..." | 0 | 0 | 0 | 0 | 0 |
| 2 | "Maybe in an email or a shared document." | 0 | 0 | 1 × `office365_search` | 4 | 3 |

#### Log-Based Mode — Turn-by-Turn Examples

**Row 0** (Scheduling, 2 result turns):

| Turn | Query | ReasoningCounts | ReasoningToolGroups | filtered_search | Claims | Citations |
|------|-------|----|----|----|--------|--------|
| 0 | "Are there any scheduling conflicts..." | **1** | `office365_search` + `TimeInsightsInternal_get_time_insights` | 2 results (4,762 + 2,060 chars) | 10 | 10 |
| 1 | "any back-to-back meetings with no break time?" | 0 | (empty) | 0 | 0 | 8 |

**Row 1** (Location, 4 result turns):

| Turn | Query | ReasoningCounts | ReasoningToolGroups | filtered_search | Claims | Citations |
|------|-------|----|----|----|--------|--------|
| 0 | "What's the location for the 7 pm Project Sync call today?" | **2** | (2 groups of tool calls) | 2 × `office365_search` | 6 | 3 |
| 1 | "directions from main lobby to Conference Room B" | 0 | (empty) | 0 | 0 | 0 |
| 2 | "main lobby, Building 92" | **1** | `CalendarAiPlacesQuery_fetch_location_info` | 1 result (108 chars) | 0 | 0 |
| 3 | "does Conference Room B have video conferencing?" | 0 | (empty) | 0 | 0 | 0 |

### Implications for Evaluation Scripts

1. **`foundry_preeval_script.py`** extracts tool calls from
   `telemetry.metrics` → `DeepLeoImprovedNetworking` → `toolInvocations`.
   This path **does not exist** in any Seval TSV variant — the scraper
   has already consumed the raw telemetry.

2. **For echo mode**: The equivalent data is in
   `sonic_model_actions[*].toolInvocations` — already parsed, no need to
   scan raw metric strings. Chain-of-thought and token counts are also
   readily available.

3. **For single-turn and log-based mode**: Reasoning **did happen**
   (evidenced by `ReasoningCounts > 0` and populated
   `ReasoningToolGroups`), but the scraper did not capture per-hop
   details (`sonic_model_actions` is always empty). The reasoning
   decisions are summarized in `ReasoningToolGroups`, the tool results
   are in `filtered_search`, and the available tools for reasoning are
   in `AvailableToolsReasoningRoot`. Chain-of-thought and token counts
   are **not available**.

4. **A general evaluation script** for Seval TSV would need to:
   - Iterate over the col 6 results array (1 element for single-turn,
     variable for multi-turn)
   - For each turn, check `sonic_model_actions` first (echo — full detail)
   - If empty, use `ReasoningToolGroups` for reasoning decisions +
     `filtered_search` for tool results (single-turn / log-based)
   - Use `ReasoningCounts` to determine whether reasoning occurred
   - For multi-turn, aggregate across turns for conversation-level metrics

---

## Key Observations

1. **All three variants share the same 7-column schema and col 6 result
   structure.** The differences are in the number of turns, which
   metadata keys are present, and how much reasoning detail the scraper
   captures.

2. **Single-turn and log-based share the same reasoning pattern**:
   `sonic_model_actions` is always empty, reasoning decisions are in
   `ReasoningToolGroups`, and tool results are in `filtered_search`.
   Echo mode is the outlier with full per-hop detail in
   `sonic_model_actions`.

3. **Reasoning data granularity differs**: Echo mode stores full per-hop
   reasoning traces (prompts, chain-of-thought, token counts), yielding
   ~3 MB per row. Single-turn averages 67 KB per row, and log-based
   250–330 KB per row — both without per-hop internals.

4. **`telemetry.metrics` / `EvaluationData` are never present** in any
   variant. The scraper consumes the raw telemetry before writing the
   TSV. The `foundry_preeval_script.py` cannot operate on these files.

5. **Actual vs max turns**: Log-based rows have `max_turns=10` but col 6
   results array lengths are 2–4. Col 0 has 8 entries (4 user + 4 bot)
   but col 6 only contains entries for turns where Sydney produced a
   response. Single-turn always has exactly 1 in both col 0 and col 6.

6. **Single-turn has extra scoring keys** not present in multi-turn:
   `Bot_Cutoff_Score`, `Bot_Interrupt_Score`, `Bot_Transcript_Score`,
   `Human_Transcript`, `audio_paths`, `original_idx`.
