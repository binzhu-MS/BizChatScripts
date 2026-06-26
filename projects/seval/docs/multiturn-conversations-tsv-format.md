# Multi-Turn Conversations TSV Format

This document describes the format of `control_multiturnconversations.tsv` files
produced by the Seval multi-turn scraper. Two variants exist — **echo** and
**log-based** — that share the same 7-column schema but differ in conversation
generation strategy.

---

## Common Schema (7 columns, tab-separated, no header row)

| Col | Name (inferred) | JSON Type | Description |
|-----|------------------|-----------|-------------|
| 0 | `conversations` | `array` of objects | Ordered list of conversation turns. Each element: `{"text": str, "author": "user"\|"bot", ...}`. |
| 1 | `metadata` | `dict` | Run-level metadata: segment, HumanBot mode, max turns, Sydney config. |
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
- `author` — `"user"` or `"bot"`. Echo mode has user-only; log-based has both.
- `location` — (optional) User location context for the turn.

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
| `humanbot` | str | HumanBot mode (e.g. `"HumanBotEchoMessage"`, or empty for log-based). |
| `max_turns` | int | Max turns the scraper will execute. |
| `sydney` | dict | Grounding config: data source, timestamp, extra headers. |
| `querygen_rules` | str | Query generation rules (usually `"N/A"`). |
| `past_conversations` | str | Prior conversation context (usually `"[]"`). |
| `rule_scenario` | str | Scenario tag (usually `"default"`). |

### Column 2 — `arm`

Plain string. The experiment arm name: `"control"` or `"experiment"`.

### Column 3 — `sydney_config`

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

JSON array of per-turn result objects. Each element represents one completed
conversation turn with the full Sydney response trace. **This is the largest
column** (up to ~3 MB per row in echo mode).

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

(~75 fields total; see source for complete list.)

---

## Variant Differences

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
| **Col 6 avg size** | ~375 KB per row |
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

## Key Observations

1. **Echo mode** pre-scripts all user turns (col 0 has only `author: "user"`)
   and captures bot responses only in col 6 results. This is useful for
   controlled A/B testing where the user utterances are fixed.

2. **Log-based mode** captures the full interleaved conversation (both user
   and bot) in col 0. The `humanbot` field is empty, suggesting the bot
   responses were recorded from a real or simulated session rather than
   generated by a HumanBot prompt.

3. **Col 6 size difference**: Echo mode stores ~3 MB per row (3 turns with
   full telemetry), while log-based stores ~375 KB per row despite having
   more turns — likely because the log-based results are more compact or
   the grounding data is smaller.

4. **Actual vs max turns**: Log-based rows have `max_turns=10` but col 6
   results array lengths are 2–4, meaning conversations stopped before
   reaching the maximum. Col 0 has 8 entries (4 user + 4 bot) but col 6
   only tracks the turns that produced Sydney responses.
