# Reasoning Checklist Metric — Refactoring Runbook

- **Date:** 2026-05-13  
- **Status:** Draft — **future-use guideline**, not yet ready to execute  
- **Metric:** `bizchat_reasoning_checklist`  
- **Scope:** Refactor the internal code organization of an existing CoMet metric while preserving all external APIs and runtime behavior.  
- **Non-goal:** No changes to the public API, proto schema, scoring semantics, criterion behavior, lifecycle, or test interfaces.

> **Execution is deferred.** CoMet still has open parity issues against
> Foundry (e.g. `user name not resolved`, `direct-reports-suffix-correct`,
> `IsEnriched`, plus the ongoing `CHI_LB_recall(domain_contains)`
> coverage gap). The shape of the eventual `criteria/<domain>.py` modules
> — which fields each criterion reads, which helpers it shares with its
> neighbors — will shift as those parity fixes land. Refactoring before
> the logic stabilizes would force us to redo module boundaries (and
> re-run all the byte-diff baselines) immediately afterwards.
>
> **Plan:** finish the parity work first; treat this document as the
> guideline to follow once the metric's behavior is settled. Update the
> module map in §5 if any parity fix changes which helpers a domain
> needs.

This is the single document that drives and verifies the refactor. It
contains:

1. The current structure and public API (which is preserved verbatim) — §2–§4.
2. The eventual (post-refactor) structure — §5.
3. The refactor execution plan — §6.
4. The verification checklists used to confirm the refactor preserved the
   public API and runtime behavior — §7–§8.

---

## 1. Motivation

The metric is implemented today in a **single 4,175-line file**
(`logic/metric_logic.py`) — by far the largest metric file in CoMet
(next-largest is `search_quality` at 2,841 lines split across 7 files).

Problems with the current layout:

- File starts with `# pylint: skip-file`; lint coverage is effectively off.
- Drift guards (`_LANGUAGE_CRITERIA`, `_UTTERANCE_CRITERIA`,
  `_ENRICHMENT_CRITERIA`, `_LLM_CRITERIA`) live ~3,500 lines from the criteria
  they gate, so PRs adding criteria repeatedly miss them.
- Prompt builders, parsers, tokenizers, ~80 criterion functions, the registry,
  and the metric class are all interleaved.
- PR diffs are unreviewable in the file viewer.

CoMet's other large metrics already follow a layered layout:

| Metric | Files | Lines | Modules in `logic/` |
|---|---|---|---|
| `search_quality` | 7 | 2,841 | `metric_logic`, `ndcg_computations`, `deduplication`, `verification_data_extractor`, `post_processor_registry`, `backwards_compatible` |
| `llm_ndcg` | 5 | 1,208 | `metric_logic`, `metric_logic_legacy`, `deduplication`, `verification_data_extractor`, `backwards_compatible` |
| `copilot_briefs_laaj` | 4 | 630 | `metric_logic`, `generation_extraction`, `scorecard_evaluation` |
| `cite_dcg` | 2 | 775 | `metric_logic`, `backwards_compatible` |

This refactor brings `bizchat_reasoning_checklist` into the same shape **with
zero behavioral change**.

---

## 2. Current Layout (Before)

### 2.1 File tree

```
cometdefinition/metrics/bizchat_reasoning_checklist/
├── bizchat_reasoning_checklist.proto
├── bizchat_reasoning_checklist_pb2.py
├── bizchat_reasoning_checklist_pb2.pyi
├── README.md
├── config/
│   └── metric_spec.yaml
├── prompts/
│   ├── chi_lb.md
│   ├── meetings_rsvp.md
│   ├── time_range_check.md
│   ├── query_correctness.md
│   └── …  (one *.md per LLM criterion)
└── logic/
    └── metric_logic.py            ← 4,175 lines, # pylint: skip-file
```

### 2.2 Anatomy of `logic/metric_logic.py`

| Approx. lines | Section | Contents |
|---|---|---|
| 1–60 | Module docstring | Overview, registry conventions, supported sessions list |
| 61–75 | Imports | stdlib + `google.protobuf` + `cometdefinition.*` + generated proto |
| 76–92 | Prompt loader | `_PROMPTS_DIR`, `_load_prompt` |
| 93–102 | Combined-tool markers | `_COMBINED_TOOL_MARKERS` |
| 103–127 | Gate frozensets | `_LANGUAGE_CRITERIA`, `_UTTERANCE_CRITERIA` |
| 130–166 | More gate frozensets | `_ENRICHMENT_CRITERIA`, `_LLM_CRITERIA` |
| 232–390 | Tokenizer / helpers | `_stable_key`, `_FOLDER_NAMES`, `_FOLDER_CONTAINER_REGEX`, `_FUNCTIONAL_TOKENS`, `_FUNCTIONAL_TOKEN_REGEX` |
| 391–512 | LLM prompt builders | `_build_rsvp_messages`, `_build_time_range_messages`, `_build_query_correctness_messages`, `_get_prompt`, `_substitute_prompt`, `_build_assessment_messages` |
| 513–598 | Parsers | `_extract_meetings_query`, `_parse_rsvp_status`, `_parse_llm_score` |
| 599–700 | Tool-call helpers | `_is_combined_tool`, `_domain_in_output`, `_get_all_queries`, `_tokenize` |
| 700–3,600 | ~80 criterion functions | Triggering, routing, meetings, completeness, URL, pronouns, language, edge context, scheduling, connectors, time range, query correctness — **interleaved**, no domain grouping |
| 3,615–3,790 | `CRITERIA_REGISTRY` | 137-key dict |
| 3,797–end | Metric class | `BizChatReasoningChecklistMetric`, `_build_shared_context`, `_evaluate_criterion`, `execute`, `preprocess`, `validate`, `validate_output` |

### 2.3 Registration points (callers — must keep working unchanged)

| File | Line |
|---|---|
| `cometdefinition/__init__.py` | `from cometdefinition.metrics.bizchat_reasoning_checklist.bizchat_reasoning_checklist_pb2 import *` |
| `cometdefinition/metrics/__init__.py` | `from cometdefinition.metrics.bizchat_reasoning_checklist.logic.metric_logic import *` |
| `cometdefinition/tests/test_bizchat_reasoning_checklist.py` | Imports `BizChatReasoningChecklistMetric`, `CRITERIA_REGISTRY`, `_LANGUAGE_CRITERIA`, `_UTTERANCE_CRITERIA`, `_LLM_CRITERIA` from `…logic.metric_logic` |

---

## 3. Public API (Preserved Verbatim)

### 3.1 Proto schema

```protobuf
message ReasoningChecklistSignal {
    EvaluationData evaluation_data = 1;
}

message SessionEntry {
    string session_name = 1;
    repeated string criteria_list = 2;
    map<string, string> custom_dimensions = 3;
}

message ReasoningChecklistEvalConfig {
    repeated SessionEntry sessions = 1;
}

message ReasoningChecklistInput {
    ReasoningChecklistSignal signal = 1 [(SingleSignal) = true];
    ReasoningChecklistEvalConfig eval_config = 2 [(EvalConfigSignal) = true];
}

message CriterionResult {
    string criteria_name = 1;
    int32 score = 2;        // 0-100 or -1 (SKIPPED)
    string message = 3;
}

message SessionResult {
    string session_name = 1;
    repeated CriterionResult criteria_results = 2;
}

message ReasoningChecklistScores {
    repeated SessionResult session_results = 1;
}

message ReasoningChecklistResult {
    ReasoningChecklistScores scores = 1 [(ScoreOutput) = true];
}

message ReasoningChecklistOutput {
    string Id = 1;
    ReasoningChecklistResult result = 2 [(SingleSignal) = true];
}
```

The `.proto` file and its generated `*_pb2.py` / `*_pb2.pyi` are not modified
by the refactor.

### 3.2 Score semantics

| Score | Meaning | Aggregation action |
|---|---|---|
| `100` | Criterion fully satisfied | Include |
| `0` | Criterion violated | Include |
| `1–99` | Partial credit (GC recall/precision, query accuracy, etc.) | Include |
| `-1` | SKIPPED — not applicable to this label/segment | **Exclude** |

### 3.3 Metric class

```python
@CopilotMetricsMap.register_metric()
class BizChatReasoningChecklistMetric(
    CopilotMetricBase[ReasoningChecklistInput, ReasoningChecklistOutput]
):
    def validate(self) -> None: ...
    async def preprocess(self) -> None: ...
    def execute(self) -> ReasoningChecklistOutput: ...
    def validate_output(self) -> None: ...

# alias used by metrics/__init__.py
metric_logic = BizChatReasoningChecklistMetric
```

Class name, module path, `@register_metric()` registration, and the
`metric_logic` alias all stay identical.

### 3.4 Criterion contract

Every callable in `CRITERIA_REGISTRY` obeys:

```python
fn(text: str, **ctx) -> int
# text  – {"role":"assistant","content":null,"tool_calls":[...]}
# **ctx – custom_dimensions ∪ injected fields (utterance, user_profile, LLM results)
# returns 0..100, or -1 for SKIPPED
```

### 3.5 `CRITERIA_REGISTRY`

- **137 keys** (criterion names exactly as they appear in Foundry session
  JSON, plus back-compat aliases including `URL_preserved_rate(*)`,
  `wokrplace_harm_recall` typo, Unicode `\u201c…\u201d` quote variants,
  `time_range_check(rule_and_LLM_combined)`).
- Still importable as
  `cometdefinition.metrics.bizchat_reasoning_checklist.logic.metric_logic.CRITERIA_REGISTRY`.

### 3.6 Gate frozensets

| Set | Members (summary) | Injection done by `execute()` |
|---|---|---|
| `_LANGUAGE_CRITERIA` | `Language Match (utterance-queries)`, `evaluate_language_consistency` | `utterance` |
| `_UTTERANCE_CRITERIA` | `URL Preserved Rate(*)`, `Question Preserved Rate(*)`, `OneDrive or SharePoint Preserved Rate(*)`, `retain my, me, i in query(*)` | `utterance` |
| `_ENRICHMENT_CRITERIA` | `IsEnriched (TruePositive) Python`, `IsEnriched (TrueNegative) Python` | `user_profile` |
| `_LLM_CRITERIA` | `meetings_rsvp`, `Time Range Check (LLM)`, `Time Range Check (Rule + LLM combined)`, `query_correctness` | `_llm_rsvp_status` / `_llm_score` (from `preprocess`) |

### 3.7 Execution pipeline

```
validate → preprocess (batch LLM calls) → execute → validate_output
```

- `preprocess()` collects de-duplicated LLM prompts for `_LLM_CRITERIA`,
  sends them as one batch via `self.llm_api.send_prompts()`, stores results
  in `self._llm_results`.
- `execute()` extracts the last-turn tool calls into the canonical JSON
  string, builds shared context (utterance + user_profile), and iterates
  `sessions × criteria`, dispatching through `CRITERIA_REGISTRY` after name
  normalization (strip whitespace, replace curly quotes).
- Only `turnData[-1]` is consulted (multi-turn conversations: previous turns
  ignored).

### 3.8 Well-known `custom_dimensions` keys

| Key | Used by |
|---|---|
| `label` | All recall/precision criteria |
| `segment` | Meeting-intent, web/work routing |
| `LLMcriteria` | Time Range Check |
| `expectedQuery` | `query_correctness` |
| `label1`, `label2` | `meetings_rsvp`, `Attendance in Meetings` |
| `tense` | `tool tense aligned JJ` |
| `assessment` | OOF, tense, delegate, category, CHI, PeopleTool_QueryAccuracy |
| `grounding` | CHI criteria |
| `expected_label` | GC criteria |
| `invocation_expected` | Edge context criteria |
| `turn`, `hasTime` | Scheduling 2nd hop |

---

## 4. Test Surface (Imports Updated, Assertions Preserved)

`cometdefinition/tests/test_bizchat_reasoning_checklist.py` has only
**internal** consumers of `…logic.metric_logic` (the test file itself plus
`local/code/seval_reasoning_checklist_scorecard.py`). The metric's
**external** API is the registered proto contract served via Polymer —
which is unaffected by Python module reorganization. We therefore do not
introduce a back-compat re-export shim. Instead:

- Each test import is updated **mechanically** to the symbol's new module
  home (e.g. `_attendance_in_meetings` becomes
  `from …logic.criteria.meetings import _attendance_in_meetings`).
- **Test bodies, fixtures, assertions, and parameterisations are not
  touched.** The diff for the test file must be import-only.
- All **510 existing tests must pass** after the import sweep, on every
  commit of the refactor.
- `local/code/seval_reasoning_checklist_scorecard.py` is updated the same
  way (import-only diff).

The complete list of internal consumers of `…logic.metric_logic` (verified
by `grep_search`):

| File | Role |
|---|---|
| `cometdefinition/metrics/__init__.py` | `import *` — keeps working as long as `BizChatReasoningChecklistMetric` is defined in (or re-imported into) the new `logic/metric_logic.py`. **No edit needed.** |
| `cometdefinition/tests/test_bizchat_reasoning_checklist.py` | Test file. **Imports updated; bodies untouched.** |
| `local/code/seval_reasoning_checklist_scorecard.py` | Scorecard tool. **Imports updated.** |

---

## 5. Target Layout (After)

```
cometdefinition/metrics/bizchat_reasoning_checklist/
├── bizchat_reasoning_checklist.proto           (unchanged)
├── bizchat_reasoning_checklist_pb2.py          (unchanged)
├── bizchat_reasoning_checklist_pb2.pyi         (unchanged)
├── README.md                                    (unchanged)
├── config/
│   └── metric_spec.yaml                         (unchanged)
├── prompts/                                     (unchanged — all *.md files)
└── logic/
    ├── __init__.py                              ← (already exists; not edited)
    ├── metric_logic.py                          ← metric class + execute pipeline
    ├── registry.py                              ← CRITERIA_REGISTRY + gate frozensets
    ├── context.py                               ← _build_shared_context, get_utterance,
    │                                              _extract_tool_calls, _format_tool_call_json
    ├── tool_calls.py                            ← _is_combined_tool, _domain_in_output,
    │                                              _get_all_queries, _tokenize,
    │                                              _FOLDER_*, _FUNCTIONAL_*, _stable_key,
    │                                              _COMBINED_TOOL_MARKERS
    ├── prompt_builders.py                       ← _PROMPTS_DIR, _load_prompt, _get_prompt,
    │                                              _substitute_prompt, _build_rsvp_messages,
    │                                              _build_time_range_messages,
    │                                              _build_query_correctness_messages,
    │                                              _build_assessment_messages
    ├── parsers.py                               ← _parse_llm_score, _parse_rsvp_status,
    │                                              _extract_meetings_query
    └── criteria/
        ├── __init__.py
        ├── triggering.py        ← emails/files/people/chat/meetings recall+precision,
        │                          python_execution_*, designer_graphic_art_*,
        │                          record_memory_*, workplace_harm_*, trigger_rate
        ├── routing.py           ← NoSearch, WebOnly, WorkOnly, WebAndWork, Web Recall,
        │                          did_early_bind, did_fanout_more_than_2,
        │                          files_invoked, canmore_*, search_web_trigger_*
        ├── meetings.py          ← Has Meeting Tool, Accuracy, Prep/Recap/Calendar
        │                          recall+precision, Bad Mistakes, keywordless_*,
        │                          keyword_early_fanout, meetings_rsvp,
        │                          Attendance in Meetings, OOF_*, tool_tense_aligned*,
        │                          time_qualifier_check, JJ_meetings_delegate_query,
        │                          Meetings Category Criteria
        ├── completeness_hints.py ← CHI-LB-Precision, CHI-LB-Recall,
        │                          meta_prioritized_incorrectly_triggered,
        │                          correct_filters_format, meta_prioritize_not_called
        ├── url_and_scope.py     ← URL Preserved*, URL Preserved Rate*,
        │                          URL Hallucination Rate*, Question Preserved Rate*,
        │                          OneDrive or SharePoint Preserved Rate*
        ├── pronouns.py          ← retain_my_me_i_in_query(strict|loose), Wrong Reasoning,
        │                          user_name_not_resolved(strict|loose),
        │                          manager_name_in_query(strict|loose)
        ├── people_tool.py       ← PeopleTool_Triggering, PeopleTool_QueryAccuracy,
        │                          people_domain_eval,
        │                          direct-reports-suffix-correct(strict|loose)
        ├── email_quality.py     ← check_folder(strict|loose),
        │                          check_folder_and_search_type_for_people_query(*),
        │                          Email MailBox Recall, hit_file_action_keyword(*)
        ├── language.py          ← Language Match (utterance-queries),
        │                          evaluate_language_consistency,
        │                          neither_unwanted(strict|loose),
        │                          miss_the_test(*), miss_past_test(*)
        ├── personalization.py   ← IsEnriched(TruePositive|TrueNegative) Python
        ├── edge_context.py      ← get_webpage_context_recall/precision, should_not_invoke
        ├── fetch_tools.py       ← fetch_email_recall, fetch_event_recall, OnlyOneTool
        ├── scheduling.py        ← scheduling 2nd-hop criteria (All parameters rate,
        │                          Empty time parameters rate, Wrong parameter rate,
        │                          Scheduling triggered, Is a Nice Serialized JSON),
        │                          People Search 1st Turn,
        │                          Emails hallucination rate 1st Turn
        ├── connectors.py        ← GC Tool Recall/Precision, GC QuSuccess, GCaaP completness
        ├── time_range.py        ← Time Range Check (LLM),
        │                          Time Range Check (Rule + LLM combined)
        └── query_correctness.py ← query_correctness, Custom Criteria,
                                   search_office365_called[_exactly_once]
```

### 5.1 Module-size guideline (soft)

**~800 lines per file is a guideline, not a hard rule.** The primary
design axis is **logical cohesion and modularity** — a domain module
should contain criteria that share helpers, share context fields, and
are reasoned about together. If a cohesive domain (e.g. `meetings.py`)
lands at 1,100 lines and splitting it would create artificial
cross-module imports between tightly-coupled helpers, **keep it
together**. Conversely, even a small file should be split if it bundles
two unrelated concerns.

Use the line count as a smell test that prompts a cohesion review, not
as an automatic trigger to split.
### 5.2 New `logic/metric_logic.py` content (minimal)

`metric_logic.py` post-refactor contains **only** the metric class and the
imports needed to wire its `execute` / `preprocess` / `validate` pipeline
to helpers in sibling modules. **No re-export shim**, no private-helper
fan-out. All other code lives in its domain module.

```python
# logic/metric_logic.py — post-refactor (sketch)

from cometdefinition.copilot_metric import CopilotMetricBase
from cometdefinition.copilot_metrics_map import CopilotMetricsMap
from cometdefinition.metrics.bizchat_reasoning_checklist.bizchat_reasoning_checklist_pb2 import (
    ReasoningChecklistInput,
    ReasoningChecklistOutput,
)
from .context import _build_shared_context
from .registry import (
    CRITERIA_REGISTRY,
    _LANGUAGE_CRITERIA, _UTTERANCE_CRITERIA,
    _ENRICHMENT_CRITERIA, _LLM_CRITERIA,
)

@CopilotMetricsMap.register_metric()
class BizChatReasoningChecklistMetric(
    CopilotMetricBase[ReasoningChecklistInput, ReasoningChecklistOutput]
):
    ...
```

That is sufficient for `metrics/__init__.py`'s `import *` to trigger the
registration decorator at import time. Test and scorecard files import
directly from `…logic.criteria.<domain>`, `…logic.registry`,
`…logic.context`, etc.

### 5.3 Dependency rule (enforced)

```
criteria/*.py  →  context.py, tool_calls.py, parsers.py, prompt_builders.py
registry.py    →  criteria/*.py
metric_logic.py →  registry.py, context.py, parsers.py, prompt_builders.py
```

`criteria/*.py` modules must NOT import from each other.
`registry.py` must NOT be imported by any `criteria/*.py`.

---

## 6. Execution Plan

Each step is a separate commit. After every commit, all unit tests must
pass and the deterministic E2E baseline must be byte-identical.

### Step 0 — Capture baselines

- `pytest -q sources/dev/MetricDefinition/cometdefinition/tests/test_bizchat_reasoning_checklist.py`
  → record pass count (expected 510/510).
- Run deterministic E2E for `bizchat_reasoning_checklist`; archive
  `expected_outputs.json` as **E2E baseline**.
- Run `seval_reasoning_checklist_scorecard.py` against the standard queryset
  with `--pairs-cache`; archive `results.md` as **scorecard baseline**.
- `python -c "from cometdefinition.metrics.bizchat_reasoning_checklist.logic.metric_logic import CRITERIA_REGISTRY; import json; print(json.dumps(sorted(CRITERIA_REGISTRY.keys()), indent=2))" > baselines/registry_keys_before.json`

### Step 1 — Extract leaf utilities

Move purely-functional helpers into their target modules. Update the
two internal consumers (`test_bizchat_reasoning_checklist.py` and
`seval_reasoning_checklist_scorecard.py`) so each moved symbol is imported
from its new home.

| Move from `metric_logic.py` | Target |
|---|---|
| `_stable_key`, `_FOLDER_NAMES`, `_FOLDER_CONTAINER_REGEX`, `_FUNCTIONAL_TOKENS`, `_FUNCTIONAL_TOKEN_REGEX`, `_tokenize`, `_get_all_queries`, `_is_combined_tool`, `_domain_in_output`, `_COMBINED_TOOL_MARKERS` | `logic/tool_calls.py` |
| `_load_prompt`, `_PROMPTS_DIR`, `_get_prompt`, `_substitute_prompt`, `_build_rsvp_messages`, `_build_time_range_messages`, `_build_query_correctness_messages`, `_build_assessment_messages` | `logic/prompt_builders.py` |
| `_parse_llm_score`, `_parse_rsvp_status`, `_extract_meetings_query` | `logic/parsers.py` |

Expected size drop: 4,175 → ~3,400 lines. Run tests; 510/510 must pass.

### Step 2 — Extract context & tool-call extraction

Move `_extract_tool_calls`, `_format_tool_call_json`, `get_utterance`, and
`_build_shared_context` to `logic/context.py`. Update imports. Run tests.

### Step 3 — Extract criterion functions, domain by domain

For each domain module in §5:

1. Cut criterion functions (with their domain-local constants) out of
   `metric_logic.py` into `criteria/<domain>.py`.
2. Update imports at the top of `criteria/<domain>.py` to pull
   `_is_combined_tool`, `_domain_in_output`, etc. from `..tool_calls`.
3. Do **not** touch `CRITERIA_REGISTRY` yet — its dict values still point to
   the (now relocated) functions through fully-qualified imports inside
   `metric_logic.py`.
4. Update the test file's imports for every moved helper to its new
   `…logic.criteria.<domain>` path. Diff for the test file must be
   import-only.
5. Run unit tests **and** drift-guard tests
   (`test_*_criteria_all_resolve_in_registry`) after every domain.

Recommended order (small/isolated first, big/shared last):

1. `edge_context`
2. `fetch_tools`
3. `connectors`
4. `scheduling`
5. `personalization`
6. `routing`
7. `url_and_scope`
8. `pronouns`
9. `language`
10. `email_quality`
11. `people_tool`
12. `completeness_hints`
13. `time_range`
14. `query_correctness`
15. `meetings`
16. `triggering`

### Step 4 — Move the registry

Cut `CRITERIA_REGISTRY = {...}` plus the four gate frozensets
(`_LANGUAGE_CRITERIA`, `_UTTERANCE_CRITERIA`, `_ENRICHMENT_CRITERIA`,
`_LLM_CRITERIA`) into `logic/registry.py`. Have it import each criterion
function from its `criteria/<domain>.py` home. Update `metric_logic.py`:

```python
from .registry import (
    CRITERIA_REGISTRY,
    _LANGUAGE_CRITERIA, _UTTERANCE_CRITERIA,
    _ENRICHMENT_CRITERIA, _LLM_CRITERIA,
)
```

### Step 5 — Strip `# pylint: skip-file`

Remove the directive from every new module. Fix only **genuine** lint
findings. Out of scope: cosmetic renames, comment edits, formatting drift.

### Step 6 — Final verification (see §7 and §8)

Run the verification checklists. If anything diverges from the baseline,
roll back the offending commit and reapply.

---

## 7. Correctness Checklist — Mid-Refactor (run after every commit)

This is the per-commit sanity gate. None of these may regress.

- [ ] `pytest -q sources/dev/MetricDefinition/cometdefinition/tests/test_bizchat_reasoning_checklist.py`
      → **510/510 passing**, zero modifications to the test file.
- [ ] Drift-guard tests still pass:
      `test_utterance_criteria_all_resolve_in_registry`,
      `test_llm_criteria_all_resolve_in_registry`,
      `test_language_criteria_all_resolve_in_registry` (if present),
      `test_enrichment_criteria_all_resolve_in_registry` (if present).
- [ ] All five symbols still importable from
      `…logic.metric_logic`:
      `BizChatReasoningChecklistMetric`, `CRITERIA_REGISTRY`,
      `_LANGUAGE_CRITERIA`, `_UTTERANCE_CRITERIA`, `_ENRICHMENT_CRITERIA`,
      `_LLM_CRITERIA`.
- [ ] `metric_logic` alias still equals `BizChatReasoningChecklistMetric`.
- [ ] No `from ..criteria` import inside any other `criteria/*.py`
      (`grep -r "from \.\.criteria" logic/criteria/` → empty).
- [ ] `git diff` for this commit shows only **moves** + necessary import
      updates (no logic edits, no signature changes).
- [ ] Per-file size reviewed against the §5.1 cohesion guideline (a file
      noticeably above ~800 lines is justified by cohesion, or has been
      split).

---

## 8. Correctness Checklist — Final (run once before raising PR)

This is the final acceptance gate. Every box must be ticked.

### 8.1 No changes outside scope

- [ ] `bizchat_reasoning_checklist.proto` — **byte-identical** to `main`.
- [ ] `bizchat_reasoning_checklist_pb2.py` / `*.pyi` — byte-identical.
- [ ] `config/metric_spec.yaml` — byte-identical.
- [ ] `prompts/*.md` — byte-identical.
- [ ] `README.md` — byte-identical.
- [ ] `cometdefinition/__init__.py` — byte-identical.
- [ ] `cometdefinition/metrics/__init__.py` — byte-identical.
- [ ] `cometdefinition/tests/test_bizchat_reasoning_checklist.py` —
      diff is **import-only**; every non-import line byte-identical.
- [ ] `local/code/seval_reasoning_checklist_scorecard.py` — diff is
      **import-only**.

### 8.2 External API identical

- [ ] Registered metric name and proto contract (input/output messages,
      `ScoreOutput` fields, custom_dimensions keys) byte-identical to
      `main` — driven by the byte-identity of `.proto`, `*_pb2.*`, and
      `metric_spec.yaml` above.
- [ ] `@CopilotMetricsMap.register_metric()` still fires at
      `cometdefinition.metrics` import time — verify the metric appears
      in `CopilotMetricsMap.list_metrics()` output.
- [ ] `CRITERIA_REGISTRY` has **exactly** the same key set as the baseline
      snapshot. Run:
      ```
      python -c "from cometdefinition.metrics.bizchat_reasoning_checklist.logic.registry import CRITERIA_REGISTRY; import json; print(json.dumps(sorted(CRITERIA_REGISTRY.keys()), indent=2))" > baselines/registry_keys_after.json
      diff baselines/registry_keys_before.json baselines/registry_keys_after.json
      ```
      → empty diff.
- [ ] `_LANGUAGE_CRITERIA`, `_UTTERANCE_CRITERIA`, `_ENRICHMENT_CRITERIA`,
      `_LLM_CRITERIA` have identical membership to baseline.
- [ ] Every name in each gate frozenset still resolves through
      `CRITERIA_REGISTRY` (drift-guard tests still green).

### 8.3 Runtime behavior identical

- [ ] Full unit test suite: **510/510 passing**.
- [ ] Deterministic E2E output **byte-identical** to Step-0 baseline.
- [ ] Seval scorecard `results.md` **byte-identical** to Step-0 baseline.
- [ ] Spot-check three criteria across different domains by invoking them
      directly with the same `(text, **ctx)` used in their pre-refactor
      tests; scores match.

### 8.4 Structural quality

- [ ] Each file in `logic/` has a clear, cohesive purpose; any file
      noticeably above ~800 lines (§5.1 guideline) is justified by
      cohesion in the PR description.
- [ ] No `# pylint: skip-file` directive remains anywhere under
      `bizchat_reasoning_checklist/`.
- [ ] No `criteria/*.py` imports another `criteria/*.py`.
- [ ] `registry.py` is not imported by any `criteria/*.py`.
- [ ] `metric_logic.py` is small — just the metric class + pipeline glue
      (target ~500 lines, not a hard cap).

### 8.5 Diff hygiene

- [ ] `git diff --stat` shows the work as moves + small import edits +
      `pylint: skip-file` removal — **no** in-place logic rewrites.
- [ ] No criterion key was renamed (the typo `wokrplace_harm_recall` is
      preserved; the back-compat aliases for `URL_preserved_rate(*)`,
      `question_preserved_rate(*)`,
      `OneDrive_or_SharePoint_preserved_rate(*)` are preserved; the Unicode
      `\u201c…\u201d` variants are preserved;
      `time_range_check(rule_and_LLM_combined)` is preserved).

---

## 9. Risk & Mitigation

| Risk | Mitigation |
|---|---|
| Hidden import cycle between `criteria/*.py` modules | Enforce §5.3 dependency rule via grep check in §7. |
| Test-file diff sneaks in non-import edits | §7 + §8.1: every commit's test-file diff must be import-only (verified by `git diff --stat` and visual review of hunks). |
| Behavioral drift introduced by cut/paste | §7 + §8.3: full pytest + deterministic E2E + scorecard byte-diff after every commit and at the end. |
| Temptation to "fix while moving" | Strict PR guideline: any code change beyond `# pylint: skip-file` removal is rejected. Out-of-scope fixes go in follow-up PRs. |
| `metrics/__init__.py` registration breaks | New `logic/metric_logic.py` still defines `BizChatReasoningChecklistMetric` with `@CopilotMetricsMap.register_metric()`, so `import *` from `metrics/__init__.py` continues to trigger registration. Verified by §8.2. |
| Drift-guard frozensets get stale during cut/paste | Run `test_*_criteria_all_resolve_in_registry` after **every** commit, not just at the end. |
| Prompt path resolution changes (`_PROMPTS_DIR`) | `_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"` will move into `prompt_builders.py`. Since `prompt_builders.py` lives in the same `logic/` directory, `.parent.parent / "prompts"` still resolves correctly. Validated by §8.3 byte-diff. |

---

## 10. Out of Scope (Save for Follow-up PRs)

- Any rename of criterion keys (including the typo `wokrplace_harm_recall`).
- Any change to context-injection rules.
- Any edit to the proto, lifecycle, or `metric_spec.yaml`.
- Any change to criterion logic — including aligning the known Foundry
  discrepancies (`user name not resolved`, `direct-reports-suffix-correct`,
  `IsEnriched` simplifications). These are documented separately and stay
  exactly as-is here.
- Cosmetic refactors (renames, comment rewrites, type-hint additions on
  code that is being moved).
- Moving `prompts/*.md` under `logic/criteria/<domain>/prompts/`.

---

## 11. Quick Reference

| Thing | Where |
|---|---|
| Public proto | `bizchat_reasoning_checklist/bizchat_reasoning_checklist.proto` |
| Metric class (entry point) | `logic/metric_logic.py::BizChatReasoningChecklistMetric` |
| Registry of 137 criteria | `logic/registry.py::CRITERIA_REGISTRY` (after refactor) |
| Gate frozensets | `logic/registry.py` (after refactor) |
| Pure helpers | `logic/tool_calls.py`, `logic/parsers.py`, `logic/prompt_builders.py`, `logic/context.py` |
| Per-domain criteria | `logic/criteria/*.py` |
| LLM prompts | `prompts/*.md` (unchanged) |
| Tests (not edited) | `cometdefinition/tests/test_bizchat_reasoning_checklist.py` |
