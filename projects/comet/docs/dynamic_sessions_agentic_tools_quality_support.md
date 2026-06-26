# Supporting the `dynamic-sessions` group in CoMet Reasoning Checklist

**Status:** Implemented (judge model `dev-gpt-41-shortco-2025-04-14`, same as the session and CoMet's configured judge)
**Author:** (generated analysis)
**Scope:** Add support for the new Foundry session
`sessions/dynamic-sessions/pineapple_tool_call_quality_v1.2.json` and its
criterion `agentic_tools_quality` to the CoMet `bizchat_reasoning_checklist`
metric.

---

## 1. Summary

Foundry added a **new session group** `dynamic-sessions` containing one session,
`pineapple_tool_call_quality_v1.2.json`. It introduces a **new evaluation
criterion** — `agentic_tools_quality` — that checks whether the model invoked
(or correctly did **not** invoke) a set of **agentic / dynamic plugin tools**
(e.g. `OutlookComposeHandoffV3_outlook_compose_handoff`,
`CalendarTriage_update_calendar`, `time_insights`).

This session does **not** fit the existing office365-domain triggering criteria
(`emails_recall`, `people_recall`, …) because:

1. The expected tools are **named, dynamically-registered plugins**, asserted
   via a **regex `pattern` + `invoked` boolean** list, not a fixed domain.
2. The model dispatches these tools through a generic **`api_tool_call_tool`**
   wrapper carrying a `tool_name` argument (an "agentic" calling convention),
   rather than as first-class `office365_search`-style functions.
3. The data items carry **new fields** (`label_plugins`, `segment`,
   `is_negative`) that none of the current criteria consume.

The criterion is **LLM-judged**, so it maps cleanly onto CoMet's existing
**assessment-style LLM judge** machinery — with a few additions described below.

---

## 2. The new session at a glance

| Property | Value |
|---|---|
| Group / folder | `sessions/dynamic-sessions/` |
| File | `pineapple_tool_call_quality_v1.2.json` |
| Prompts | `control`, `treatment` (1P agentic Copilot tool set) |
| `evaluationStrategy.criteriaList` | **1** criterion: `agentic_tools_quality` |
| Data items | **726** (`sessionInputs[0].dataItems`) |
| Assertions per item | exactly **1** (`label_plugins` has 1 entry) |
| Judge model | `dev-gpt-41-shortco-2025-04-14`, `temperature=0` |
| `parsingScript` | the standard Foundry pre-eval (first-hop L1 tool-call extraction) |
| Post-`script` | extracts integer `score` from the judge's JSON |

### 2.1 Data item shape (new fields)

```json
{
  "utterance": "reply to Alex Kowalski confirming we're aligned on the June milestones",
  "label_plugins": "[{\"pattern\": \"OutlookComposeHandoffV3_outlook_compose_handoff\", \"invoked\": true}]",
  "segment": "1P: Outlook Compose",
  "is_negative": "false"
}
```

- **`label_plugins`** — JSON-encoded **array** of golden assertions. Each entry:
  - `pattern` — **regex** matched against the literal tool name.
  - `invoked` — `true` ⇒ the tool **must** be called; `false` ⇒ it **must not** be.
- **`segment`** — human label, e.g. `1P: Outlook Compose`,
  `Do not trigger: RSVP actions`. 13 segments in the dataset.
- **`is_negative`** — `"true"`/`"false"` string; negative = "should not trigger".

### 2.2 Segment / pattern distribution (726 items)

| Segment | Count |
|---|---|
| 1P: RSVP actions | 164 |
| 1P: Outlook Compose | 121 |
| Do not trigger: Outlook Compose | 98 |
| 1P: Time Analysis | 72 |
| 1P: Place Query | 49 |
| 1P: Outlook Automatic Reply | 42 |
| 1P: Calendar Instructions | 42 |
| 1P: Email Triage Action | 32 |
| 1P: Calendar Scheduling | 25 |
| Do not trigger: Calendar Scheduling | 22 |
| Do not trigger: RSVP actions | 19 |
| Do not trigger: Outlook Automatic Reply | 17 |
| Do not trigger: Time Analysis | 15 |
| Do not trigger: Email Triage Action | 8 |

`is_negative`: 547 positive / 179 negative. `invoked`: 522 `true` / 204 `false`.

Tool patterns asserted (top): `OutlookComposeHandoffV3_outlook_compose_handoff`
(219), `CalendarTriage_update_calendar` (183), `time_insights` (87),
`SetAutomaticRepliesPlugin_get_outlook_mailbox_settings` (59),
`CalendarAiPlacesQuery_fetch_location_info` (49),
`CalendarScheduling_suggest_meeting_location` (47), plus several
`CalendarAiAgent_*` and `OutlookTriageActions_*` tools.

### 2.3 The judge prompt (verbatim intent)

The criterion's system message instructs the judge to:

> Score whether an LLM's round-2 tool call matches a golden assertion.
> `label_plugins` is a JSON array of `{pattern (regex), invoked (bool)}`.
> Inspect the response for tool calls (**especially `api_tool_call_tool` with a
> `tool_name` argument**). For each entry:
> - `invoked: true` ⇒ pass if a tool call's name matches the regex `pattern`.
> - `invoked: false` ⇒ pass if **no** tool call name matches `pattern`.
> - **Match only the literal `function.name`** of a tool call. Tool names
>   appearing inside `arguments` (e.g. `api_tool_search_tools` queries) are
>   **tool-discovery searches, not invocations** — do not count them.
>
> Score `100` if all pass; `0` if any fail; partial credit
> `round(passing/total*100)`. Respond ONLY with JSON:
> `{"score", "passing_assertions", "total_assertions", "explanation"}`.

Template variables used by the prompt: `{{label_plugins}}`, `{{utterance}}`,
`{{segment}}`, `{{is_negative}}`, `{{text}}`.

The post-script simply pulls `score` out of the JSON assessment.

---

## 3. How CoMet handles sessions today (recap)

```
ReasoningChecklistInput
  signal.evaluation_data : EvaluationData      ← model's tool calls + utterance + profile
  eval_config.sessions[] : SessionEntry
        session_name      : str  (logging only)
        criteria_list     : [str]  (keys into CRITERIA_REGISTRY)
        custom_dimensions : map<str,str>  (label, segment, expectedQuery, …)
```

Pipeline: `validate → preprocess (LLM batch) → execute → postprocess →
validate_output`.

- **`_extract_tool_calls(eval_data)`** flattens every
  `turnData[-1].orchestrationIterations[*]` (recursively through
  `nestedOrchestrations`) `modelActions[].toolInvocations[]` into
  `{"function": {"name": inv.function, "arguments": inv.arguments}}` and
  `_format_tool_call_json` wraps them as
  `{"role":"assistant","content":null,"tool_calls":[…]}` — this becomes `text`.
- **`CRITERIA_REGISTRY[name] = fn(text, **ctx) -> int`**. `-1` = SKIPPED.
- LLM-judged criteria come in three flavors:
  1. **Custom-wrapped** (`meetings_rsvp`, `time_range_check(*)`,
     `query_correctness`) — bespoke `_build_*_messages` helpers.
  2. **Assessment-style** — registered in `_CRITERION_PROMPT_FILES`
     (prompt template file) + `_CRITERION_RESULT_KEY` (dedup/cache key). The
     prompt is the Sydney session text with `{{text}}`/`{{utterance}}`/
     `{{expected_reasoning_response}}` placeholders; the judge's raw reply is
     handed back to the criterion fn via `ctx["assessment"]`, which parses it.
  3. **Direct** — LLM output used as the score.
- **`preprocess()`** collects a de-duplicated set of prompts across all
  session×criterion pairs and issues **one batched** `llm_api.send_prompts(...)`.
- **`_evaluate_criterion()`** forwards **all** `custom_dimensions` as `**ctx`,
  then overlays `utterance` / `user_profile` / `assessment` for the criteria
  that need them.

`agentic_tools_quality` is an **assessment-style LLM judge** (flavor #2) — the
existing machinery is the right fit. The work is the additions in §4.

---

## 4. Gap analysis — what's missing

| # | Gap | Required change |
|---|---|---|
| G1 | Criterion `agentic_tools_quality` not registered | Add to `CRITERIA_REGISTRY` |
| G2 | Prompt template absent | Add `prompts/agentic_tools_quality.md` (the session's system message, verbatim, with `{{…}}` placeholders) |
| G3 | Prompt needs **new template vars** `label_plugins`, `segment`, `is_negative` | Extend `_build_assessment_messages` to substitute extra dims |
| G4 | Result/cache key must vary by **assertion** | Salt `_CRITERION_RESULT_KEY` for this criterion with a hash of `label_plugins` (+utterance) so distinct golden assertions don't collapse to one cached call |
| G5 | Post-processing fn | Add `_agentic_tools_quality(text, **ctx)` that parses `score` from `ctx["assessment"]` (reuse the existing `_parse_llm_score`-style helper) |
| G6 | Per-item dims plumbing | Caller (queryset → `custom_dimensions`) must supply `label_plugins`, `segment`, `is_negative` per data item |
| G7 | Judge model differs (`dev-gpt-41-shortco-2025-04-14`) | Decide: accept configured `metric_spec.yaml` judge, or allow per-criterion model override (see §6) |
| G8 | Agentic `api_tool_call_tool` convention | **No metric code change needed** — `_extract_tool_calls` passes `function.name` + `arguments` through unchanged, and the LLM judge unwraps `tool_name`. Must **verify** EvaluationData actually carries these invocations (see §7, open question). |
| G9 | Docs / registration | Update README, AGENTS.md "Supported sessions", module docstring; add tests |

---

## 5. Recommended implementation

Implement `agentic_tools_quality` as a **new assessment-style LLM criterion**,
reusing the existing batch/preprocess/inject flow with three small extensions.

### 5.1 Prompt file

Create `prompts/agentic_tools_quality.md` containing the session's system
message **verbatim**, keeping the `{{label_plugins}}`, `{{utterance}}`,
`{{segment}}`, `{{is_negative}}`, `{{text}}` placeholders. This keeps CoMet
scores comparable to Foundry-collected labels.

### 5.2 Extend `_build_assessment_messages`

Add optional kwargs and substitutions for the new variables:

```python
def _build_assessment_messages(
    prompt_filename, tool_calls_text, utterance="",
    expected_reasoning_response="", user_template="{{text}}",
    assistant_prefill="",
    extra_subs: dict | None = None,        # NEW
):
    subs = {
        "text": tool_calls_text,
        "utterance": utterance,
        "expected_reasoning_response": expected_reasoning_response,
    }
    if extra_subs:                          # NEW: label_plugins, segment, is_negative
        subs.update(extra_subs)
    ...
```

### 5.3 Register the criterion + maps

```python
# CRITERIA_REGISTRY
"agentic_tools_quality": _agentic_tools_quality,

# _LLM_CRITERIA       → add "agentic_tools_quality"
# _CRITERION_PROMPT_FILES["agentic_tools_quality"] = "agentic_tools_quality.md"
# _CRITERION_RESULT_KEY["agentic_tools_quality"]   = "agentic_tools_quality"
```

### 5.4 Per-item cache key (preprocess)

Because the golden assertion (`label_plugins`) is **per data item**, the result
key must include it (and the utterance), mirroring how `time_qualifier_check`
salts by utterance and `people_tool_query_accuracy` salts by
`expected_reasoning_response`:

```python
elif normalized == "agentic_tools_quality":
    label_plugins = cdims.get("label_plugins", "")
    utterance     = get_utterance(self._eval_data) if self._eval_data else ""
    result_key = f"agentic_tools_quality:{_stable_key(label_plugins + '|' + utterance)}"
    extra_subs = {
        "label_plugins": label_plugins,
        "segment": cdims.get("segment", ""),
        "is_negative": cdims.get("is_negative", ""),
    }
    # build_assessment_messages(..., extra_subs=extra_subs)
```

and the symmetric lookup in `_evaluate_criterion` (compute the same key from
`extra_ctx["label_plugins"]` + utterance, read `self._llm_results[key]` into
`extra_ctx["assessment"]`).

### 5.5 Criterion function

```python
def _agentic_tools_quality(text: str, **ctx) -> int:
    """LLM-judged: does the round-2 tool call satisfy the label_plugins
    golden assertion? Score from ctx['assessment'] JSON ('score' field)."""
    assessment = ctx.get("assessment", "")
    if not assessment:
        return _SKIPPED            # no judge result available
    score = _parse_llm_score(assessment)   # extracts {"score": N}
    return score if score is not None else _SKIPPED
```

(`_parse_llm_score` already strips code fences and reads a JSON `score`/number;
confirm it reads the `score` key — if not, add a tiny JSON-`score` extractor
matching the session's post-script.)

### 5.6 Caller / queryset plumbing

The scorecard (`local/code/seval_reasoning_checklist_scorecard.py`) builds
`SessionEntry.custom_dimensions` from the queryset TSV's
`personalization_metadata.metricsInput[]`. For this session each entry must
carry:

```json
{
  "session": "pineapple_tool_call_quality",
  "criteriaList": ["agentic_tools_quality"],
  "customDimensions": {
    "label_plugins": "[{\"pattern\":\"...\",\"invoked\":true}]",
    "segment": "1P: Outlook Compose",
    "is_negative": "false"
  }
}
```

No scorecard code change is required if the queryset is generated with these
dims; otherwise add a small generator on the Sydney/Foundry side.

---

## 6. Judge model decision (G7)

The session uses `dev-gpt-41-shortco-2025-04-14`; CoMet's
`config/metric_spec.yaml` configures a single judge (`dev-gpt-4o-gg`).

- **Option A (simplest):** accept the configured CoMet judge. Existing
  assessment criteria already tolerate model differences (e.g.
  `attendance_in_meetings` ran on `dev-gpt-4o-gg`). Calibrate/spot-check.
- **Option B:** add an optional per-criterion model override so the batch
  splits `agentic_tools_quality` prompts onto `dev-gpt-41-shortco-*`. Larger
  change to `preprocess()` / `llm_api` batching.

**Recommendation:** start with **Option A**, validate score agreement against a
Foundry label sample, and only pursue Option B if agreement is insufficient.

---

## 7. Open questions / things to verify before coding

1. **Agentic invocation shape in EvaluationData (G8).** Confirm that the
   `api_tool_call_tool` wrapper (with `tool_name` in `arguments`) actually
   appears in `turnData[*].orchestrationIterations[*].modelActions[].toolInvocations[]`
   for these 1P agentic runs. The metric passes `name`+`arguments` through, and
   the LLM judge unwraps `tool_name`, so **no extraction change is expected** —
   but this must be verified on a real scrape. If, instead, the dynamic tool is
   surfaced directly as `function.name` (no wrapper), the judge prompt already
   handles that too (it matches the literal name).
2. **`_parse_llm_score` reads `score`.** Verify it extracts the integer `score`
   from `{"score": N, ...}`; if it only reads bare numbers, add a JSON-`score`
   path (or a dedicated `_extract_agentic_score`).
3. **Single vs. multi-assertion.** Current data is 1 assertion/item, but the
   prompt and `passing/total` math support N. Keep the criterion N-safe.
4. **Session-name routing.** Decide the canonical `session_name`
   (e.g. `pineapple_tool_call_quality` / `dynamic-sessions`) for logging and any
   session→criteria default mapping the scorecard uses.
5. **Aggregation semantics.** Confirm negatives (`is_negative=true`) should be
   **scored** (not skipped) — unlike recall criteria, the judge already encodes
   "must not invoke", so these should produce 0/100, never `-1`.

---

## 8. Touch list (when we implement)

| File | Change |
|---|---|
| `metrics/bizchat_reasoning_checklist/prompts/agentic_tools_quality.md` | **new** — verbatim judge system prompt |
| `metrics/bizchat_reasoning_checklist/logic/metric_logic.py` | `_build_assessment_messages` extra_subs; `_agentic_tools_quality`; register in `CRITERIA_REGISTRY`, `_LLM_CRITERIA`, `_CRITERION_PROMPT_FILES`, `_CRITERION_RESULT_KEY`; per-item key in `preprocess()` + `_evaluate_criterion()`; docstring "Supported sessions" |
| `metrics/bizchat_reasoning_checklist/README.md` | document the criterion, new dims (`label_plugins`, `segment`, `is_negative`), and the `dynamic-sessions` group |
| `metrics/bizchat_reasoning_checklist/AGENTS.md` | add to supported sessions/criteria |
| `tests/test_bizchat_reasoning_checklist.py` | add cases: positive invoked-pass, negative must-not-invoke pass/fail, partial credit, missing-assertion skip, LLM mocked via `metric.llm_api.send_prompts.return_value` |

No proto change is required — `agentic_tools_quality` is just another
`criteria_list` name and the new fields ride in `custom_dimensions`
(`map<string,string>`).
