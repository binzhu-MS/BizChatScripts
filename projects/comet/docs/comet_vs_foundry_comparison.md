# CoMet vs Foundry: Reasoning Checklist Criteria Comparison

**Date:** 2026-04-27  
**CoMet branch:** current (metric_logic.py ~3290 lines)  
**Foundry source:** `C:\working\Sydney\services\TuringBot\tools\ReasoningPythonScript\sessions\`

> **Scope note:** Foundry evaluates raw SEVAL scraping data from `telemetry.metrics`
> and normalizes it via a `parsingScript` (pre-eval) step. CoMet receives
> already-structured `EvaluationData` from the SEVAL scraping file — a completely
> different input path. Differences caused by the pre-eval script are **out of
> scope** for this comparison; only the scoring scripts and criterion logic are
> compared.

---

## 1. Coverage Summary

| Metric | Before Phase 2 | After Phase 2 | After Phase 3 |
|---|---|---|---|
| Total criteria instances in Foundry session files | 136 | 136 | 136 |
| Unique criteria in Foundry session files | 114 | 114 | 114 |
| CRITERIA_REGISTRY keys in CoMet | 138 | **116** | **114** |
| CoMet keys exactly matching a session criterion | 38 | **114** | **114** |
| CoMet keys needing rename | 76 | **0** | **0** |
| CoMet extra keys (no session counterpart) | 24 | **2** | **0** |
| Logic discrepancies | 4 | 4 | **0** |
| Session criteria NOT covered by CoMet | 0 | 0 | 0 |

**Coverage: 114 / 114 = 100%** — all Foundry session criteria are handled by CoMet.

After Phase 2: 76 keys renamed, 22 alias keys deleted. After Phase 3:
2 standalone criteria deleted, 4 logic discrepancies fixed. Registry went
from 116 → **114 keys** (all matching session criteria, zero extras).

---

## 2. The 1 Missing Criterion

| Foundry Name | Session File | Root Cause |
|---|---|---|
| `workplace_harm_recall` | `base-sessions/triggering-workplaceharm.json` | CoMet registered under misspelled key `wokrplace_harm_recall` (swapped `r` and `k`); no entry for the correct spelling |

---

## 3. The 24 Extra CoMet Keys (Not in Any Foundry Session)

22 of the 24 extra keys are **aliases** — variant names that point to the same
Python function as an existing matched criterion. The remaining 2 (#22, #23) are
**standalone criteria** with unique implementations that have no Foundry session
counterpart. Their "berry-loop" label in the table comes from the source-code
comment at `metric_logic.py` line 2943: `# ── Berry-loop: combined tool presence ───`.

### Why these keys exist

Criterion names arrive from multiple sources: base sessions, berry-loop sessions,
different Foundry versions, and manual references. These sources use variant
spellings (typos, curly vs straight quotes, hyphens with/without spaces),
optional `(strict)` / `(loose)` suffixes, leading/trailing whitespace, or
completely different names for the same logic. The 22 alias keys ensure CoMet
accepts **any known variant** without requiring callers to normalize first.
The 2 standalone criteria (`search_office365_called`,
`search_office365_called_exactly_once`) check tool-call presence in
orchestrator output and exist only in CoMet — no Foundry session references them.

At runtime, `_evaluate_criterion()` also strips the incoming name, so
leading/trailing whitespace variants work automatically. The registry aliases
handle everything else.

### Full Extra Key Table

| # | Extra CoMet Key | Corresponding Session Criterion | Category |
|---|---|---|---|
| 1 | `wokrplace_harm_recall` | `workplace_harm_recall` | Typo (`r`↔`k`) |
| 2 | `user name not resolved` | `user_name_not_resolved(strict)` | Suffix-less (defaults to strict) |
| 3 | `miss "the" test (strict)` | `miss_"the"_test(strict)` | Straight-quote variant of curly-quote key |
| 4 | `miss "the" test (loose)` | `miss_"the"_test(loose)` | Straight-quote variant |
| 5 | `miss "the" test` | `miss_"the"_test(strict)` | Suffix-less |
| 6 | `miss "past" test` | `miss_"past"_test(strict)` | Suffix-less |
| 7 | `URL Preserved(strict)` | `URL_preserved(strict)` | Leading-space-stripped variant |
| 8 | `URL Preserved(loose)` | `URL_preserved(loose)` | Leading-space-stripped variant |
| 9 | `URL Preserved` | `URL_preserved(strict)` | Suffix-less |
| 10 | `URL Preserved Rate` | `URL_preserved_rate(strict)` | Suffix-less |
| 11 | `Question Preserved Rate` | `question_preserved_rate(strict)` | Suffix-less |
| 12 | `OneDrive or SharePoint Preserved Rate` | `OneDrive_or_SharePoint_preserved_rate(strict)` | Suffix-less |
| 13 | `URL Hallucination Rate` | `URL_hallucination_rate(strict)` | Suffix-less |
| 14 | `Hit file action keyword` | `hit_file_action_keyword(strict)` | Suffix-less |
| 15 | `neither_unwanted` | `neither_unwanted(strict)` | Suffix-less |
| 16 | `neither_unwanted (loose)` | `neither_unwanted(loose)` | Whitespace variant (space before paren) |
| 17 | `CHI-LB-Precision` | `CHI_LB_precision` | Compact-hyphen variant |
| 18 | `CHI-LB-Recall` | `CHI_LB_recall` | Compact-hyphen variant |
| 19 | `evaluate_language_consistency` | `language_match(utterance_queries)` | Wrong name — someone used the session filename as a criterion key |
| 20 | `Time Range Check (Rule + LLM combined)` | `time_range_check(LLM)` | Variant name for the same LLM judge |
| 21 | `Custom Criteria` | `query_correctness` | Variant name (international_my_manager replacement) |
| 22 | `search_office365_called` | *(none — standalone)* | Standalone criterion (berry-loop: combined tool presence) |
| 23 | `search_office365_called_exactly_once` | *(none — standalone)* | Standalone criterion (berry-loop: combined tool presence) |
| 24 | `direct-reports-suffix-correct(loose)` | `direct_reports_suffix_correct(loose)` | Whitespace variant (missing space before paren) |

### Extra Key Categories

| Category | Count | Keys |
|---|---|---|
| **Typo** | 1 | #1 |
| **Leading space stripped** | 2 | #7, #8 |
| **Straight-quote variant** | 2 | #3, #4 |
| **Suffix-less (drops `(strict)`/`(loose)`)** | 10 | #2, #5, #6, #9–15 |
| **Whitespace before paren** | 2 | #16, #24 |
| **Compact hyphens** | 2 | #17, #18 |
| **Different name for same logic** | 3 | #19, #20, #21 |
| **Standalone (unique impl, no session)** | 2 | #22, #23 |

---

## 4. Logic Discrepancies

These 4 criteria exist in both CoMet and Foundry sessions but their
implementations differ in meaningful ways. See §9 Phase 3.2 for fix status.

### 4.1 SIGNIFICANT: `user_name_not_resolved(strict/loose)`

| Aspect | CoMet (master) | Foundry Session |
|--------|---------------|----------------|
| **What is searched** | Entire raw JSON text | Only chats/emails domain queries in `office365.search` / `office365_search` tool calls |
| **Strict check** | `"bryan perkins" not in text` | `"bryan perkins" not in query_string` for each chats/emails query (AND logic) |
| **Loose check** | `"perkins" not in text` | `"bryan perkins" not in query_string` for each chats/emails query (OR logic) |
| **When no chats/emails triggered** | Returns 0 or 100 (checks entire text) | Returns SKIPPED (-1) |

**Impact:** CoMet false-positives when `"bryan perkins"` appears anywhere in
the JSON (e.g., in a different tool's output). CoMet also scores when
chats/emails were never triggered, while Foundry skips.

### 4.2 SIGNIFICANT: `direct_reports_suffix_correct(strict/loose)`

| Aspect | CoMet (master) | Foundry Session |
|--------|---------------|----------------|
| **What is checked** | Structured `domain_filters.relationship` field for `"direct_report"` | Query **text** ends with `"direct reports"` |
| **Domain filter** | Checks `domain_filters` dict | Filters to people-domain queries only |
| **When no people domain triggered** | Returns 0 or 100 | Returns SKIPPED (-1) |

**Impact:** Completely different check — CoMet looks at a structured field
while Foundry checks query text suffix. Results will diverge when the query
text doesn't end with "direct reports" but the structured field is set, or
vice versa.

### 4.3 SIGNIFICANT: `IsEnriched(TruePositive/TrueNegative)_python`

| Aspect | CoMet (master) | Foundry Session |
|--------|---------------|----------------|
| **TruePositive** | Checks if manager name appears as substring in text | Full algorithm: extract queries (FluxV2/V3/JSON), preprocess (punctuation, lowercase, stop words, stemming), find extra terms not in utterance, check if they come from any user profile field |
| **TrueNegative** | Checks if no first-person pronouns in text | Same full algorithm but returns 100 when NO profile-origin terms found |
| **Profile fields checked** | Manager name only | All profile fields (department, job title, city, etc.) |

**Impact:** CoMet's simplified check misses cases where enrichment comes from
non-manager profile fields. The Foundry algorithm is ~150 lines of text
processing with stemming and multi-format query extraction.

### 4.4 MINOR: `files_invoked`

| Aspect | CoMet (master) | Foundry Session |
|--------|---------------|----------------|
| **Legacy path** | `"search_enterprise_files" in text` | `"search_enterprise_files" in text` OR (`"search_enterprise_fanout" in text` AND `'"files"' in text`) |

**Impact:** CoMet misses files invocations that go through the fanout path.
Combined-tool path is unaffected (both check `domain == "files"`).

---

## 5. Criteria Confirmed Matching

The remaining **110** criteria (114 total minus the 4 with logic discrepancies
in §4) have **equivalent scoring logic** between Foundry session scripts and
CoMet Python implementations. Many of these criteria had different key names
in CoMet's CRITERIA_REGISTRY vs the (revised) Foundry session criterion names;
those name mismatches are documented in §5.1.

Of the 114 criteria (counting strict/loose pairs as one):
- **37 keys** already had exact-match names (no change needed)
- **70 keys** needed CoMet key renames (name only — no logic change; see §5.1)

The 4 criteria with **logic discrepancies** (7 individual keys) are documented
separately in §4.

Names shown below are the original (pre-Phase-2) CoMet CRITERIA_REGISTRY keys.

### Domain Recall/Precision (10 criteria)
`emails_recall`, `emails_precision`, `files_recall`, `files_precision`,
`people_recall`, `people_precision`, `chat_recall`, `chat_precision`,
`meetings_recall`, `meetings_precision`

All use the same `domain_in_output()` helper for combined tools and
`search_enterprise_{domain}` substring check for legacy tools. Label-skip
logic (negative/ambiguous → SKIPPED for recall, ambiguous → SKIPPED +
tool-not-invoked → SKIPPED for precision) matches exactly.

### Specialized Tool Triggering (7 criteria)
`python_execution_recall/precision`, `designer_graphic_art_recall/precision`,
`record_memory_recall/precision`, `workplace_harm_prec.`

Label variant handling (ShouldTrigger/ShouldNotTrigger/ShouldGenerate) matches.

### Canvas/Canmore (5 criteria)
`canmore_create trigger rate (higher is better)`,
`canmore_create trigger rate (lower is better)`,
`canmore_update trigger rate (higher is better)`,
`canmore_update trigger rate (lower is better)`,
`search_web trigger rate (higher is better)`

Simple substring checks match.

### Shared Mailbox (1 criterion)
`Email MailBox Recall` — combined-tool path checks `domain == "emails"` +
`"mailbox" in query`; legacy path checks `search_enterprise_emails` +
`"mailbox" in query`. Both match.

### Scheduling 1st Hop (2 criteria)
`People Search 1st Turn`, `Emails hallucination rate 1st Turn` — match.

### Web/Work Routing (5 criteria)
`NoSearch`, `WebOnly`, `WorkOnly`, `WebAndWork`, `Web Recall` — functionally
equivalent (see notes on §4.5 minor detection difference for `WebOnly`).

### People Tool (2 criteria)
`PeopleTool_Triggering`, `people domain eval` — match.
`PeopleTool_QueryAccuracy` — both read pre-computed `assessment` dimension.

### Early Binding / Fanout (2 criteria)
`did_early_bind` — both check normalized tool set == {emails}.
`did_fanout_more_than_2` — both count queries (combined) or unique tools
(legacy), filter `generate_response`.

### Meeting Prep/Recap (2 criteria)
`keywordless_early_binding`, `keyword_early_fanout` — match.

### OOF Meetings (2 criteria)
`OOF Meetings Criterion`, `OOF Meetings Criterion JJ` — both parse
`assessment` JSON, compare `doc["status"]` against `label`.

### Meeting Tense/Time (2 criteria)
`tool tense aligned` / `tool tense aligned JJ` — both parse `assessment`
JSON, compare `doc["tense"]` against expected.
`time qualifier chechk` — both parse `doc["result"]`; `"error"` → skip,
`"match"` → 100, else → 0.

### Meeting Intent (9 criteria)
`Has Meeting Tool`, `Accuracy`, `Prep Recall`, `Prep Precision`,
`Recap Recall`, `Recap Precision`, `Calendar Recall`, `Calendar Precision`,
`Bad Mistakes` — meeting_intent extraction and segment-to-intent mapping match.

### RSVP/Attendance/Delegate/Category (4 criteria)
`meetings_rsvp` — LLM status parsing and label1/label2 comparison match.
`Attendance in Meetings` — assessment JSON status parsing matches.
`JJ meetings_delegate_query`, `Meetings Category Criteria` — both parse
assessment JSON with meetings gate and label comparison.

### Pronoun Retention (3 criteria)
`Wrong Reasoning`, `retain my, me, i in query (strict)`,
`retain my, me, i in query (loose)` — CoMet's tokenizer and multi-language
pronoun set match Foundry's implementation.

### Meeting Keywords (4 criteria)
`miss \u201cthe\u201d test (strict/loose)`, `miss "past" test (strict/loose)` — both
filter to meetings-domain queries in office365.search and check keyword with
AND (strict) / OR (loose) logic.

### International Manager (2 criteria)
`manager name in query (strict/loose)` — both check `segment` value in queries.

### Japanese File Action (2 criteria)
`Hit file action keyword(strict/loose)` — keyword list and AND/OR logic match.

### URL/Scope Checks (8 criteria)
`URL Preserved Rate (strict/loose)`, `Question Preserved Rate (strict/loose)`,
`OneDrive or SharePoint Preserved Rate (strict/loose)`,
`URL Hallucination Rate (strict/loose)` — all match.

### URL Queries (2 criteria)
` URL Preserved(strict)`, ` URL Preserved(loose)` — match.

### Email Folder/Sender (4 criteria)
`check folder+search type for people query(strict/loose)`,
`check folder (strict/loose)` — match.

### Completeness Hints (5 criteria)
`CHI - LB - Precision`, `CHI - LB - Recall` — combined-tool response_length
logic and non-combined assessment-based precision/recall match.
`meta_prioritized incorrectly triggered`, `correct_filters_format`,
`meta_prioritize_not_called` — match.

### Language (1 criterion)
`Language Match (utterance-queries)` — CJK detection logic matches.

### Summarize in Language (2 criteria)
`neither_unwanted(strict)`, `neither_unwanted(loose)` — both return SKIPPED
(response-level check not possible from tool-call data alone).

### LLM-Judged (3 criteria)
`Time Range Check (LLM)`, `query_correctness`, `meetings_rsvp` — prompt
construction and score extraction match Foundry's LLM judge patterns.

### Edge Context (3 criteria)
`get_webpage_context recall`, `get_webpage_context precision`,
`should_not_invoke` — `invocation_expected` flag logic matches.

### Fetch Email/Event (3 criteria)
`fetch_email_recall`, `fetch_event_recall`, `OnlyOneTool` — match.

### Scheduling 2nd Hop (5 criteria)
`All paramters rate 2nd Turn`, `Empty time parameters rate 2nd Turn`,
`Wrong parameter rate 2nd Turn`, `Scheduling triggered 2nd Turn`,
`Is a Nice Serialized JSON` — parameter checks and JSON validation match.

### Graph Connectors (4 criteria)
`GC Tool Recall`, `GC Tool Precision` — set-based recall/precision match.
`GC QuSuccess` — parameter-level F1 computation matches.
`GCaaP completness` — response_length ratio matches.

### 5.1 Name-Only Changes (70 criteria) 

These 70 criteria have matching logic but their CoMet CRITERIA_REGISTRY keys
did not match the (revised) Foundry session criterion names. Renamed in Phase 2
(Phase 2). The 6 rename entries for logic-discrepancy criteria (§4) are excluded
from this table.

| # | Old CoMet Key | New Key (matches session) |
|---|---|---|
| 1 | `CHI - LB - Precision` | `CHI_LB_precision` |
| 2 | `CHI - LB - Recall` | `CHI_LB_recall` |
| 3 | `GC QuSuccess` | `GC_qu_success` |
| 4 | `GC Tool Precision` | `GC_tool_precision` |
| 5 | `GC Tool Recall` | `GC_tool_recall` |
| 6 | `GCaaP completness` | `GCaaP_completeness` |
| 7 | `JJ meetings_delegate_query` | `JJ_meetings_delegate_query` |
| 8 | `OOF Meetings Criterion` | `OOF_meetings_criterion` |
| 9 | `OOF Meetings Criterion JJ` | `OOF_meetings_criterion_JJ` |
| 10 | `OneDrive or SharePoint Preserved Rate (loose)` | `OneDrive_or_SharePoint_preserved_rate(loose)` |
| 11 | `OneDrive or SharePoint Preserved Rate (strict)` | `OneDrive_or_SharePoint_preserved_rate(strict)` |
| 12 | `URL Hallucination Rate (loose)` | `URL_hallucination_rate(loose)` |
| 13 | `URL Hallucination Rate (strict)` | `URL_hallucination_rate(strict)` |
| 14 | ` URL Preserved(loose)` | `URL_preserved(loose)` |
| 15 | ` URL Preserved(strict)` | `URL_preserved(strict)` |
| 16 | `URL Preserved Rate (loose)` | `URL_preserved_rate(loose)` |
| 17 | `URL Preserved Rate (strict)` | `URL_preserved_rate(strict)` |
| 18 | `All paramters rate 2nd Turn` | `all_parameters_rate_2nd_turn` |
| 19 | `Attendance in Meetings` | `attendance_in_meetings` |
| 20 | `Bad Mistakes` | `bad_mistakes` |
| 21 | `Calendar Precision` | `calendar_precision` |
| 22 | `Calendar Recall` | `calendar_recall` |
| 23 | `canmore_create trigger rate (higher is better)` | `canmore_create_trigger_rate(higher_is_better)` |
| 24 | `canmore_create trigger rate (lower is better)` | `canmore_create_trigger_rate(lower_is_better)` |
| 25 | `canmore_update trigger rate (higher is better)` | `canmore_update_trigger_rate(higher_is_better)` |
| 26 | `canmore_update trigger rate (lower is better)` | `canmore_update_trigger_rate(lower_is_better)` |
| 27 | `check folder (loose)` | `check_folder(loose)` |
| 28 | `check folder (strict)` | `check_folder(strict)` |
| 29 | `check folder+search type for people query(loose)` | `check_folder_and_search_type_for_people_query(loose)` |
| 30 | `check folder+search type for people query(strict)` | `check_folder_and_search_type_for_people_query(strict)` |
| 31 | `Email MailBox Recall` | `email_mailbox_recall` |
| 32 | `Emails hallucination rate 1st Turn` | `emails_hallucination_rate_1st_turn` |
| 33 | `Empty time parameters rate 2nd Turn` | `empty_time_parameters_rate_2nd_turn` |
| 34 | `get_webpage_context precision` | `get_webpage_context_precision` |
| 35 | `get_webpage_context recall` | `get_webpage_context_recall` |
| 36 | `Has Meeting Tool` | `has_meeting_tool` |
| 37 | `Hit file action keyword(loose)` | `hit_file_action_keyword(loose)` |
| 38 | `Hit file action keyword(strict)` | `hit_file_action_keyword(strict)` |
| 39 | `Is a Nice Serialized JSON` | `is_a_nice_serialized_JSON` |
| 40 | `Language Match (utterance-queries)` | `language_match(utterance_queries)` |
| 41 | `manager name in query (loose)` | `manager_name_in_query(loose)` |
| 42 | `manager name in query (strict)` | `manager_name_in_query(strict)` |
| 43 | `Meetings Category Criteria` | `meetings_category_criteria` |
| 44 | `meta_prioritized incorrectly triggered` | `meta_prioritized_incorrectly_triggered` |
| 45 | `miss "past" test (loose)` | `miss_"past"_test(loose)` |
| 46 | `miss "past" test (strict)` | `miss_"past"_test(strict)` |
| 47 | `miss \u201cthe\u201d test (loose)` | `miss_"the"_test(loose)` |
| 48 | `miss \u201cthe\u201d test (strict)` | `miss_"the"_test(strict)` |
| 49 | `people domain eval` | `people_domain_eval` |
| 50 | `People Search 1st Turn` | `people_search_1st_turn` |
| 51 | `PeopleTool_QueryAccuracy` | `people_tool_query_accuracy` |
| 52 | `PeopleTool_Triggering` | `people_tool_triggering` |
| 53 | `Prep Precision` | `prep_precision` |
| 54 | `Prep Recall` | `prep_recall` |
| 55 | `Question Preserved Rate (loose)` | `question_preserved_rate(loose)` |
| 56 | `Question Preserved Rate (strict)` | `question_preserved_rate(strict)` |
| 57 | `Recap Precision` | `recap_precision` |
| 58 | `Recap Recall` | `recap_recall` |
| 59 | `retain my, me, i in query (loose)` | `retain_my_me_i_in_query(loose)` |
| 60 | `retain my, me, i in query (strict)` | `retain_my_me_i_in_query(strict)` |
| 61 | `Scheduling triggered 2nd Turn` | `scheduling_triggered_2nd_turn` |
| 62 | `search_web trigger rate (higher is better)` | `search_web_trigger_rate(higher_is_better)` |
| 63 | `time qualifier chechk` | `time_qualifier_check` |
| 64 | `Time Range Check (LLM)` | `time_range_check(LLM)` |
| 65 | `tool tense aligned` | `tool_tense_aligned` |
| 66 | `tool tense aligned JJ` | `tool_tense_aligned_JJ` |
| 67 | `Trigger rate` | `trigger_rate` |
| 68 | `Web Recall` | `web_recall` |
| 69 | `Wrong parameter rate 2nd Turn` | `wrong_parameter_rate_2nd_turn` |
| 70 | `Wrong Reasoning` | `wrong_reasoning` |

---

## 6. Naming Convention & Normalization

### 6.1 Agreed Convention

Session names and criteria names follow these rules:

1. **`_`** (underscore) is the **only** word separator in both session names
   and criteria names. **`-`** (hyphen) is **not allowed**; convert to `_`
2. **CamelCase is allowed ONLY when no separator is used** (fused words like
   `NoSearch`, `OnlyOneTool`, `WebAndWork`). If the name already uses `_`
   or space as a separator → all words are **lowercase**
3. **Preserve original capitals for acronyms and special terms**: `URL`, `OOF`,
   `GC`, `LLM`, `JSON`, `CHI`, `LB`, `JJ`, `CI`, `CJK`, `GCaaP`, `OneDrive`,
   `SharePoint`, `IsEnriched`, `TruePositive`, `TrueNegative`
4. **Space is not allowed as a separator** → convert to `_`
5. **No leading or trailing whitespace**
6. **All characters must be ASCII** (no Unicode curly quotes, etc.)
7. **No spelling errors**
8. **Parenthetical qualifiers** like `(strict)`, `(loose)`, `(LLM)`,
   `(higher_is_better)` are kept as-is with **no space before `(`**;
   spaces inside qualifiers also become `_`
9. **`+`** → `_and_`; commas between list items are dropped
   (e.g. `retain my, me, i in query` → `retain_my_me_i_in_query`)

> **Note:** Subfolder names (`base-sessions`, `edge-context-first-hop`, etc.)
> are directory-structure identifiers and are **not** subject to these rules.

### 6.2 Session Name Changes

**36 changes** out of 61 session files (all `-` → `_`; plus 1 spelling fix):

| # | Current File Name | Proposed Name | Folder | Changes |
|---|---|---|---|---|
| 1 | `completeness_hints-hints_p_r_evaluation-email_focused` | `completeness_hints_hints_p_r_evaluation_email_focused` | `base-sessions` | `-` → `_` |
| 2 | `completeness_hints-hints_p_r_evaluation_1.3` | `completeness_hints_hints_p_r_evaluation_1.3` | `base-sessions` | `-` → `_` |
| 3 | `completeness_hints-latency_calibration` | `completeness_hints_latency_calibration` | `base-sessions` | `-` → `_` |
| 4 | `jp-file_action` | `jp_file_action` | `base-sessions` | `-` → `_` |
| 5 | `meeting-keywords-in-the-next-week` | `meeting_keywords_in_the_next_week` | `base-sessions` | `-` → `_` |
| 6 | `meeting-keywords-past` | `meeting_keywords_past` | `base-sessions` | `-` → `_` |
| 7 | `meetings-category-queries` | `meetings_category_queries` | `base-sessions` | `-` → `_` |
| 8 | `meetings-delegate` | `meetings_delegate` | `base-sessions` | `-` → `_` |
| 9 | `meetings-oof-everyone` | `meetings_oof_everyone` | `base-sessions` | `-` → `_` |
| 10 | `meetings-oof-others` | `meetings_oof_others` | `base-sessions` | `-` → `_` |
| 11 | `meetings-oof-self` | `meetings_oof_self` | `base-sessions` | `-` → `_` |
| 12 | `people-direct-reports` | `people_direct_reports` | `base-sessions` | `-` → `_` |
| 13 | `people-direct-reports_one_utterance_only` | `people_direct_reports_one_utterance_only` | `base-sessions` | `-` → `_` |
| 14 | `people-tool-triggering-and-accuracy` | `people_tool_triggering_and_accuracy` | `base-sessions` | `-` → `_` |
| 15 | `prompt_triggering-meetings` | `prompt_triggering_meetings` | `base-sessions` | `-` → `_` |
| 16 | `query-correctness` | `query_correctness` | `base-sessions` | `-` → `_` |
| 17 | `rsvp-status-in-meetings-queries` | `rsvp_status_in_meetings_queries` | `base-sessions` | `-` → `_` |
| 18 | `triggering-chat` | `triggering_chat` | `base-sessions` | `-` → `_` |
| 19 | `triggering-code_interpreter` | `triggering_code_interpreter` | `base-sessions` | `-` → `_` |
| 20 | `triggering-email-shared-mailbox` | `triggering_email_shared_mailbox` | `base-sessions` | `-` → `_` |
| 21 | `triggering-email` | `triggering_email` | `base-sessions` | `-` → `_` |
| 22 | `triggering-files` | `triggering_files` | `base-sessions` | `-` → `_` |
| 23 | `triggering-graphic_art` | `triggering_graphic_art` | `base-sessions` | `-` → `_` |
| 24 | `triggering-memory` | `triggering_memory` | `base-sessions` | `-` → `_` |
| 25 | `triggering-people` | `triggering_people` | `base-sessions` | `-` → `_` |
| 26 | `triggering-scheduling_handoff_first_hop` | `triggering_scheduling_handoff_first_hop` | `base-sessions` | `-` → `_` |
| 27 | `triggering-transcript_search` | `triggering_transcript_search` | `base-sessions` | `-` → `_` |
| 28 | `triggering-web_work` | `triggering_web_work` | `base-sessions` | `-` → `_` |
| 29 | `triggering-workplace_harm` | `triggering_workplace_harm` | `base-sessions` | `-` → `_` |
| 30 | `prompt_triggering-edge_context` | `prompt_triggering_edge_context` | `edge-context-first-hop` | `-` → `_` |
| 31 | `triggering-edge_context_files_calibration` | `triggering_edge_context_files_calibration` | `edge-context-first-hop` | `-` → `_` |
| 32 | `prompt_triggering-edge_context_second_hop` | `prompt_triggering_edge_context_second_hop` | `edge-context-second-hop` | `-` → `_` |
| 33 | `prompt_triggering-fetch_email` | `prompt_triggering_fetch_email` | `fetch-email` | `-` → `_` |
| 34 | `prompt_triggering-fetch_event` | `prompt_triggering_fetch_event` | `fetch-event` | `-` → `_` |
| 35 | `prompt_triggering-scheduling_handoff_second_hop` | `prompt_triggering_scheduling_handoff_second_hop` | `scheduling-handoff-second-hop` | `-` → `_` |
| 36 | `prompt_triggering-search_enterprise_connectors` | `prompt_triggering_search_enterprise_connectors` | `search_enterprise_connectors` | `-` → `_` |

The remaining 25 session names already conform (including
`canvas_create_overtriggering_doc_CI` — `CI` is a preserved acronym per rule 3).

> **Note:** The original `triggering-workplaceharm` spelling fix (→ `workplace_harm`)
> was applied in the prior round; that file is now `triggering-workplace_harm` and
> is covered by row 29 above (`-` → `_`).

### 6.3 Criteria Name Changes

**78 changes** out of 115 unique (pre-rename) criteria names. **37 names already conform.**
After rename, 114 unique names remain (1 duplicate pair collapsed — see §6.4).

| # | Current Name | Proposed Name | Session File(s) | Changes |
|---|---|---|---|---|
| 1 | ` URL Preserved(loose)` | `URL_preserved(loose)` | `url_queries` | strip whitespace; space → _ |
| 2 | ` URL Preserved(strict)` | `URL_preserved(strict)` | `url_queries` | strip whitespace; space → _ |
| 3 | `All paramters rate 2nd Turn` | `all_parameters_rate_2nd_turn` | `prompt_triggering-scheduling_handoff_second_hop` | spelling: paramters → parameters; space → _ |
| 4 | `Attendance in Meetings` | `attendance_in_meetings` | `attendance_in_meetings_search` | space → _ |
| 5 | `Bad Mistakes` | `bad_mistakes` | `meeting_intent_parameter` | space → _ |
| 6 | `CHI - LB - Precision` | `CHI_LB_precision` | `completeness_hints-hints_p_r_evaluation-email_focused`, `completeness_hints-hints_p_r_evaluation_1.3` | ' - ' → '_'; lowercase |
| 7 | `CHI - LB - Recall` | `CHI_LB_recall` | `completeness_hints-hints_p_r_evaluation-email_focused`, `completeness_hints-hints_p_r_evaluation_1.3` | ' - ' → '_'; lowercase |
| 8 | `Calendar Precision` | `calendar_precision` | `meeting_intent_parameter` | space → _ |
| 9 | `Calendar Recall` | `calendar_recall` | `meeting_intent_parameter` | space → _ |
| 10 | `Email MailBox Recall` | `email_mailbox_recall` | `triggering-email-shared-mailbox` | space → _; compound: MailBox → mailbox |
| 11 | `Emails hallucination rate 1st Turn` | `emails_hallucination_rate_1st_turn` | `triggering-scheduling_handoff_first_hop` | space → _ |
| 12 | `Empty time parameters rate 2nd Turn` | `empty_time_parameters_rate_2nd_turn` | `prompt_triggering-scheduling_handoff_second_hop` | space → _ |
| 13 | `GC QuSuccess` | `GC_qu_success` | `prompt_triggering-search_enterprise_connectors` | space → _; CamelCase split: QuSuccess → qu\_success |
| 14 | `GC Tool Precision` | `GC_tool_precision` | `prompt_triggering-search_enterprise_connectors` | space → _ |
| 15 | `GC Tool Recall` | `GC_tool_recall` | `prompt_triggering-search_enterprise_connectors` | space → _ |
| 16 | `GCaaP completness` | `GCaaP_completeness` | `prompt_triggering-search_enterprise_connectors` | spelling: completness → completeness; space → _ |
| 17 | `Has Meeting Tool` | `has_meeting_tool` | `meeting_intent_parameter` | space → _ |
| 18 | `Hit file action keyword(loose)` | `hit_file_action_keyword(loose)` | `jp-file_action` | space → _ |
| 19 | `Hit file action keyword(strict)` | `hit_file_action_keyword(strict)` | `jp-file_action` | space → _ |
| 20 | `Is a Nice Serialized JSON` | `is_a_nice_serialized_JSON` | `prompt_triggering-scheduling_handoff_second_hop` | space → _ |
| 21 | `IsEnriched (TrueNegative) Python` | `IsEnriched(TrueNegative)_python` | `personalization_with_user_profile` | remove space before qualifier per rule 8; space → _ |
| 22 | `IsEnriched (TruePositive) Python` | `IsEnriched(TruePositive)_python` | `personalization_with_user_profile` | remove space before qualifier per rule 8; space → _ |
| 23 | `JJ meetings_delegate_query` | `JJ_meetings_delegate_query` | `meetings-delegate` | space → _ |
| 24 | `Language Match (utterance-queries)` | `language_match(utterance_queries)` | `evaluate_language_consistency` | space → _; `-` → `_` in qualifier |
| 25 | `Meetings Category Criteria` | `meetings_category_criteria` | `meetings-category-queries` | space → _ |
| 26 | `OOF Meetings Criterion` | `OOF_meetings_criterion` | `meetings-oof-everyone` | space → _ |
| 27 | `OOF Meetings Criterion JJ` | `OOF_meetings_criterion_JJ` | `meetings-oof-others`, `meetings-oof-self` | space → _ |
| 28 | `OneDrive or SharePoint Preserved Rate (loose)` | `OneDrive_or_SharePoint_preserved_rate(loose)` | `files_scope_url_folder` | space → _ |
| 29 | `OneDrive or SharePoint Preserved Rate (strict)` | `OneDrive_or_SharePoint_preserved_rate(strict)` | `files_scope_url_folder` | space → _ |
| 30 | `People Search 1st Turn` | `people_search_1st_turn` | `triggering-scheduling_handoff_first_hop` | space → _ |
| 31 | `PeopleTool_QueryAccuracy` | `people_tool_query_accuracy` | `people-tool-triggering-and-accuracy` | CamelCase split: PeopleTool → people\_tool, QueryAccuracy → query\_accuracy |
| 32 | `PeopleTool_Triggering` | `people_tool_triggering` | `people-tool-triggering-and-accuracy` | CamelCase split: PeopleTool → people\_tool |
| 33 | `Prep Precision` | `prep_precision` | `meeting_intent_parameter` | space → _ |
| 34 | `Prep Recall` | `prep_recall` | `meeting_intent_parameter` | space → _ |
| 35 | `Question Preserved Rate (loose)` | `question_preserved_rate(loose)` | `files_scope_url_folder` | space → _ |
| 36 | `Question Preserved Rate (strict)` | `question_preserved_rate(strict)` | `files_scope_url_folder` | space → _ |
| 37 | `Recap Precision` | `recap_precision` | `meeting_intent_parameter` | space → _ |
| 38 | `Recap Recall` | `recap_recall` | `meeting_intent_parameter` | space → _ |
| 39 | `Scheduling triggered 2nd Turn` | `scheduling_triggered_2nd_turn` | `prompt_triggering-scheduling_handoff_second_hop` | space → _ |
| 40 | `Time Range Check (LLM)` | `time_range_check(LLM)` | `time_range_checking` | space → _ |
| 41 | `Trigger rate` | `trigger_rate` | `triggering-transcript_search` | space → _ |
| 42 | `URL Hallucination Rate (loose)` | `URL_hallucination_rate(loose)` | `files_scope_url_folder` | space → _ |
| 43 | `URL Hallucination Rate (strict)` | `URL_hallucination_rate(strict)` | `files_scope_url_folder` | space → _ |
| 44 | `URL Preserved Rate (loose)` | `URL_preserved_rate(loose)` | `files_scope_url_folder` | space → _ |
| 45 | `URL Preserved Rate (strict)` | `URL_preserved_rate(strict)` | `files_scope_url_folder` | space → _ |
| 46 | `Web Recall` | `web_recall` | `triggering-web_work` | space → _ |
| 47 | `Wrong Reasoning ` | `wrong_reasoning` | `retain_i_me_my_in_query` | strip whitespace; space → _ |
| 48 | `Wrong parameter rate 2nd Turn` | `wrong_parameter_rate_2nd_turn` | `prompt_triggering-scheduling_handoff_second_hop` | space → _ |
| 49 | `canmore_create trigger rate (higher is better)` | `canmore_create_trigger_rate(higher_is_better)` | `canvas_create` | space → _ |
| 50 | `canmore_create trigger rate (lower is better)` | `canmore_create_trigger_rate(lower_is_better)` | `canvas_create_overtriggering`, `canvas_create_overtriggering_doc_CI` | space → _ |
| 51 | `canmore_update trigger rate (higher is better)` | `canmore_update_trigger_rate(higher_is_better)` | `canvas_edit_long`, `canvas_edit_short`, `prompt_canvas_update` | space → _ |
| 52 | `canmore_update trigger rate (lower is better)` | `canmore_update_trigger_rate(lower_is_better)` | `canvas_grounding` | space → _ |
| 53 | `check folder (loose)` | `check_folder(loose)` | `email_contains_key_info` | space → _ |
| 54 | `check folder (strict)` | `check_folder(strict)` | `email_contains_key_info` | space → _ |
| 55 | `check folder+search type for people query(loose)` | `check_folder_and_search_type_for_people_query(loose)` | `email_contains_key_info` | '+' → '\_and\_'; space → _ |
| 56 | `check folder+search type for people query(strict)` | `check_folder_and_search_type_for_people_query(strict)` | `email_contains_key_info` | '+' → '\_and\_'; space → _ |
| 57 | `direct-reports-suffix-correct (loose)` | `direct_reports_suffix_correct(loose)` | `people-direct-reports` | `-` → `_`; remove space before qualifier |
| 58 | `direct-reports-suffix-correct (strict)` | `direct_reports_suffix_correct(strict)` | `people-direct-reports`, `people-direct-reports_one_utterance_only` | `-` → `_`; remove space before qualifier |
| 59 | `direct-reports-suffix-correct(loose)` | `direct_reports_suffix_correct(loose)` | `people-direct-reports_one_utterance_only` | `-` → `_` |
| 60 | `get_webpage_context precision` | `get_webpage_context_precision` | `prompt_triggering-edge_context` | space → _ |
| 61 | `get_webpage_context recall` | `get_webpage_context_recall` | `prompt_triggering-edge_context` | space → _ |
| 62 | `manager name in query (loose)` | `manager_name_in_query(loose)` | `international_my_manager` | space → _ |
| 63 | `manager name in query (strict)` | `manager_name_in_query(strict)` | `international_my_manager` | space → _ |
| 64 | `meta_prioritized incorrectly triggered` | `meta_prioritized_incorrectly_triggered` | `completeness_hints-hints_p_r_evaluation-email_focused`, `completeness_hints-hints_p_r_evaluation_1.3` | space → _ |
| 65 | `miss "past" test (loose)` | `miss_"past"_test(loose)` | `meeting-keywords-past` | space → _ |
| 66 | `miss "past" test (strict)` | `miss_"past"_test(strict)` | `meeting-keywords-past` | space → _ |
| 67 | `miss \u201cthe\u201d test (loose)` | `miss_"the"_test(loose)` | `meeting-keywords-in-the-next-week` | curly quotes → straight quotes; space → _ |
| 68 | `miss \u201cthe\u201d test (strict)` | `miss_"the"_test(strict)` | `meeting-keywords-in-the-next-week` | curly quotes → straight quotes; space → _ |
| 69 | `neither_unwanted (loose)` | `neither_unwanted(loose)` | `summarize_in_language` | remove space before qualifier |
| 70 | `people domain eval` | `people_domain_eval` | `people_eval_domain` | space → _ |
| 71 | `retain my, me, i in query (loose)` | `retain_my_me_i_in_query(loose)` | `retain_i_me_my_in_query` | remove commas; space → _ |
| 72 | `retain my, me, i in query (strict)` | `retain_my_me_i_in_query(strict)` | `retain_i_me_my_in_query` | remove commas; space → _ |
| 73 | `search_web trigger rate (higher is better)` | `search_web_trigger_rate(higher_is_better)` | `canvas_edit_web` | space → _ |
| 74 | `time qualifier chechk` | `time_qualifier_check` | `meeting_query_time_hints` | spelling: chechk → check; space → _ |
| 75 | `tool tense aligned` | `tool_tense_aligned` | `meeting_query_time_hints`, `meetings-oof-everyone` | space → _ |
| 76 | `tool tense aligned JJ` | `tool_tense_aligned_JJ` | `meetings-oof-others`, `meetings-oof-self` | space → _ |
| 77 | `user name not resolved(loose)` | `user_name_not_resolved(loose)` | `do_not_resolve_i_me_my` | space → _ |
| 78 | `user name not resolved(strict)` | `user_name_not_resolved(strict)` | `do_not_resolve_i_me_my` | space → _ |

#### Unchanged criteria (37 names already conforming)

`Accuracy`, `NoSearch`, `OnlyOneTool`, `WebAndWork`, `WebOnly`, `WorkOnly`,
`chat_precision`, `chat_recall`, `correct_filters_format`,
`designer_graphic_art_precision`, `designer_graphic_art_recall`,
`did_early_bind`, `did_fanout_more_than_2`,
`emails_precision`, `emails_recall`,
`fetch_email_recall`, `fetch_event_recall`, `files_invoked`, `files_precision`,
`files_recall`, `keyword_early_fanout`, `keywordless_early_binding`,
`meetings_precision`, `meetings_recall`, `meetings_rsvp`,
`meta_prioritize_not_called`, `neither_unwanted(strict)`, `people_precision`,
`people_recall`, `python_execution_precision`, `python_execution_recall`,
`query_correctness`, `record_memory_precision`, `record_memory_recall`,
`should_not_invoke`, `workplace_harm_prec.`, `workplace_harm_recall`

### 6.4 Duplicate After Normalization

One pair of criteria normalizes to the same name:

| Normalized Name | Original Names | Resolution |
|---|---|---|
| `direct_reports_suffix_correct(loose)` | `direct-reports-suffix-correct (loose)` (with space), `direct-reports-suffix-correct(loose)` (without space) | Both normalize to the same `_`-separated form. The with-space variant (#57) and without-space variant both need `-` → `_`. |

### 6.5 Other Observations

| Item | Notes |
|---|---|
| `completeness_hints_hints_p_r_evaluation_1.3` | Version suffix with `.` — not a separator violation |

---

## 7. Multi-Session Criteria & Script Divergence

### 7.1 Overview

The 61 session files contain **136 total criteria instances** but only **114
unique criterion names** — meaning **22 instances** are repeated (the same
criterion name appears in more than one session). **15 criteria** appear in 2+
sessions:

| # | Criterion | Session Count | Sessions |
|---|---|---|---|
| 1 | `meetings_recall` | 7 | `meetings_category_queries`, `meetings_delegate`, `meetings_oof_everyone`, `meetings_oof_others`, `meetings_oof_self`, `prompt_triggering_meetings`, `rsvp_status_in_meetings_queries` |
| 2 | `canmore_update_trigger_rate(higher_is_better)` | 3 | `canvas_edit_long`, `canvas_edit_short`, `prompt_canvas_update` |
| 3 | `correct_filters_format` | 3 | `completeness_hints_hints_p_r_evaluation_1.3`, `completeness_hints_hints_p_r_evaluation_email_focused`, `completeness_hints_latency_calibration` |
| 4 | `CHI_LB_precision` | 2 | `completeness_hints_hints_p_r_evaluation_1.3`, `completeness_hints_hints_p_r_evaluation_email_focused` |
| 5 | `CHI_LB_recall` | 2 | `completeness_hints_hints_p_r_evaluation_1.3`, `completeness_hints_hints_p_r_evaluation_email_focused` |
| 6 | `OOF_meetings_criterion_JJ` | 2 | `meetings_oof_others`, `meetings_oof_self` |
| 7 | `OnlyOneTool` | 2 | `prompt_triggering_fetch_email`, `prompt_triggering_fetch_event` |
| 8 | `canmore_create_trigger_rate(lower_is_better)` | 2 | `canvas_create_overtriggering`, `canvas_create_overtriggering_doc_CI` |
| 9 | `direct_reports_suffix_correct(loose)` | 2 | `people_direct_reports`, `people_direct_reports_one_utterance_only` |
| 10 | `direct_reports_suffix_correct(strict)` | 2 | `people_direct_reports`, `people_direct_reports_one_utterance_only` |
| 11 | `files_precision` | 2 | `triggering_files`, `triggering_edge_context_files_calibration` |
| 12 | `files_recall` | 2 | `triggering_files`, `triggering_edge_context_files_calibration` |
| 13 | `meta_prioritized_incorrectly_triggered` | 2 | `completeness_hints_hints_p_r_evaluation_1.3`, `completeness_hints_hints_p_r_evaluation_email_focused` |
| 14 | `tool_tense_aligned` | 2 | `meetings_oof_everyone`, `meeting_query_time_hints` |
| 15 | `tool_tense_aligned_JJ` | 2 | `meetings_oof_others`, `meetings_oof_self` |

Instance count check: (7 + 3 + 3 + 2×12) − 15 = 37 − 15 = **22 extra** =
136 − 114 ✓

### 7.2 Evaluation Divergence (Prompt + Script)

Of the 15 multi-session criteria, **11 have identical evaluation logic** (both
prompt and script) across all their sessions. **4 have divergence:**

- 3 have **different scripts** (1 of those also has different prompts)
- 1 has **identical scripts but a trivial prompt difference** (1 char)

Note: `meetings_recall` is purely deterministic (no LLM). `CHI_LB_precision`,
`CHI_LB_recall`, and `tool_tense_aligned` all use LLM — the session prompt
produces an `{{assessment}}` JSON that the post-processing script then parses
and scores. For LLM-based criteria, the prompt is part of the evaluation logic,
so prompt divergence is as significant as script divergence.

#### `CHI_LB_precision` — Prompt differs by 1 char; script identical

Uses LLM: yes (prompt → `{{assessment}}`). Scripts are **identical** across
both sessions. Prompts differ by a single trailing period on the final
instruction sentence — semantically identical.

| Session | Prompt Variant |
|---|---|
| `completeness_hints_hints_p_r_evaluation_1.3` | `...just follow the expected behavior)` (no period, 6918 chars) |
| `completeness_hints_hints_p_r_evaluation_email_focused` | `...just follow the expected behavior).` (with period, 6919 chars) |

#### `CHI_LB_recall` — Script differs; prompt identical

Uses LLM: yes (prompt → `{{assessment}}`). Prompts are **identical** across
both sessions (same hash, 6919-char `assistant` message).

| Session | Script Variant |
|---|---|
| `completeness_hints_hints_p_r_evaluation_1.3` | `tool in str(query.get('domain', ''))` (substring match) |
| `completeness_hints_hints_p_r_evaluation_email_focused` | `query['domain'] == tool` (exact match) |

The `email_focused` variant is stricter and likely the intended behavior.

#### `meetings_recall` — 4 script variants across 7 sessions

Uses LLM: **no** (purely deterministic).

| Variant | Sessions | Key Difference |
|---|---|---|
| A (buggy) | `meetings_category_queries` | `or "office365_search"` — always truthy (Python evaluates the non-empty string as `True`) |
| B (fixed) | `meetings_delegate`, `meetings_oof_others`, `meetings_oof_self`, `rsvp_status_in_meetings_queries` | `or name == "office365_search"` — correct comparison |
| C (fixed, trailing space) | `meetings_oof_everyone` | `or name == "office365_search" ` — same fix, cosmetic trailing space |
| D (refactored) | `prompt_triggering_meetings` | Adds `try/except`, `tool_calls` key check, and simplifies the final `print()` to unconditionally call `domain_in_output()` alongside `search_enterprise_meetings` substring check |

Variants B/C are functionally identical. Variant A has a latent bug that causes
`domain_in_output()` to always enter the combined-tool branch for any tool name,
but since it then checks `query['domain'] == domain` inside, it usually
produces the correct result anyway (false positives only when the tool is not
`office365.search`/`office365_search` but happens to have a meetings-domain
query — an unlikely scenario).

#### `tool_tense_aligned` — Both prompt and script differ substantially

Uses LLM: yes (prompt → `{{assessment}}`). Both the **prompt** and the
**post-processing script** differ between the two sessions.

**Prompt divergence:**

| Aspect | `meetings_oof_everyone` | `meeting_query_time_hints` |
|---|---|---|
| Message role | `assistant` | `user` |
| Content length | 1484 chars | 957 chars |
| Prompt hash | `fe301b390c67` | `0ff0547ecc06` |

The two prompts give the LLM different instructions and framing, which means
the `{{assessment}}` JSON they produce may differ for the same input.

**Script divergence:**

| Session | Size | Approach |
|---|---|---|
| `meetings_oof_everyone` | ~4.2 KB | Simpler assessment-JSON parser with basic `strip_code_fence()` |
| `meeting_query_time_hints` | ~7.5 KB | More robust: `extract_first_json_blob()`, better fence stripping, `is_json()` guard, extended error handling |

Both parse an `{{assessment}}` JSON object and compare `doc["tense"]` against
the expected value, but `meeting_query_time_hints` handles more edge cases
(malformed JSON, embedded JSON in text, BOM characters).

**Input-field divergence (most important):** The two scripts read the expected
tense from **different template variables**, which map to **different fields**
in the dataset items:

| Session | Script reads | Input field | Field semantics |
|---|---|---|---|
| `meetings_oof_everyone` | `{{tense}}` | `tense` (also has a separate `label` field with values like `OOF`) | Expected tense of the meetings query |
| `meeting_query_time_hints` | `{{label}}` | `label` (no `tense` field exists) | Expected tense of the meetings query |

This means the two `tool_tense_aligned` criteria are **not interchangeable**:
- `meetings_oof_everyone` uses `label` for OOF status (not tense), so the
  script must read `tense` to score tense alignment.
- `meeting_query_time_hints` repurposes `label` to mean the expected tense
  and has no `tense` field at all.

The scripts cannot be unified into a single implementation without also
restructuring the input fields of one of the sessions.

### 7.3 How CoMet Handles This

CoMet uses a **single `CRITERIA_REGISTRY` dict** — one Python function per
criterion name. When the same criterion name appears in multiple sessions,
**CoMet runs the exact same function for all of them**. There is no per-session
script dispatch.

The `execute()` method iterates over sessions, and for each session's
`criteria_list`, calls `_evaluate_criterion()` which does a simple dict lookup:

```python
criterion_fn = CRITERIA_REGISTRY.get(normalized) or CRITERIA_REGISTRY.get(criteria_name)
```

This means CoMet is **always consistent** for same-named criteria (no risk of
different scores for the same input depending on which session requests it), but
it also means CoMet must **choose one variant** when Foundry sessions disagree.

For the 3 divergent criteria, CoMet's implementations correspond to:

| Criterion | CoMet matches | Notes |
|---|---|---|
| `CHI_LB_recall` | Closer to `email_focused` (exact match) | CoMet uses structured tool-call parsing, not substring |
| `meetings_recall` | Variant B/C (the fix) | Uses correct `name == "office365_search"` comparison |
| `tool_tense_aligned` | Single implementation covering both use cases | Parses `assessment` JSON, compares `doc["tense"]` against expected |

---

## 8. LLM Usage Comparison

### 8.1 Overview

Foundry sessions define **15 unique LLM-judged criteria** (20 instances across
sessions). CoMet handles these through three different mechanisms:

| Mechanism | Count | Criteria |
|-----------|-------|----------|
| **CoMet makes own LLM calls** | 3 | `meetings_rsvp`, `time_range_check(LLM)`, `query_correctness` |
| **Assessment from custom dims** | 11 | `attendance_in_meetings`, `CHI_LB_precision`, `CHI_LB_recall`, `JJ_meetings_delegate_query`, `meetings_category_criteria`, `OOF_meetings_criterion`, `OOF_meetings_criterion_JJ`, `people_tool_query_accuracy`, `time_qualifier_check`, `tool_tense_aligned`, `tool_tense_aligned_JJ` |
| **Deterministic rule (no LLM)** | 1 | `language_match(utterance_queries)` |

### 8.2 Category A: CoMet Makes Own LLM Calls (3 criteria)

For these 3 criteria, CoMet's `preprocess()` method sends prompts to its own
LLM API and injects results into `extra_ctx` before the criterion function runs.

#### `meetings_rsvp`

| Aspect | Foundry Session | CoMet |
|--------|----------------|-------|
| **Prompt file** | Inline in `rsvp_status_in_meetings_queries.json` | `prompts/meetings_rsvp.md` |
| **Model** | `dev-gpt-5-chat-jj` | Metric-level config |
| **Temperature** | 0 | Metric-level config |
| **Message role** | `assistant` (with `<\|start\|>system` markers) | `system` + `user` |
| **Prompt length** | 1634 chars | 1018 chars |
| **Status categories** | accepted, declined, RSVP status of, cancelled, tentative, followed, non rsvp, not responded, not accepted, not declined | Same set |
| **Post-script** | 1732 chars: `safe_json_loads` → `doc['status']` vs `label1`/`label2` → 100/50/0 | `_parse_rsvp_status()` → compare to `label1`/`label2` → 100/50/0 |
| **Scoring logic** | **Match** | **Match** |

**Discrepancies:**
- **Minor:** CoMet prompt is a rewrite with slightly different wording but covers the
  same status categories and instructions. Semantically equivalent.
- **Role format:** Session uses `assistant` role with `<|start|>system` markers;
  CoMet uses proper `system` + `user` roles. Functionally equivalent.

#### `query_correctness`

| Aspect | Foundry Session | CoMet |
|--------|----------------|-------|
| **Prompt file** | Inline in `query_correctness.json` | `prompts/query_correctness.md` |
| **Model** | `dev-gpt-5-chat-jj` | Metric-level config |
| **Temperature** | 0 | Metric-level config |
| **Message roles** | `system` + `system` + `user` | `system` + `user` |
| **Prompt length** | 2121 chars (main system msg) | 383 chars |
| **Score scale** | 0 / 25 / 50 / 75 / 100 | Same |
| **Has examples** | Yes (3 worked examples) | No |
| **User message** | `result:{{text}}\nexpectedQuery:{{expectedQuery}}` | `Tool call:\n{text}\n\nExpected query:\n{expected}` |
| **Post-script** | None (LLM output = score) | `_parse_llm_score()` extracts numeric value |

**Discrepancies:**
- **P1 — Missing examples:** CoMet prompt omits all 3 worked examples that
  calibrate the LLM judge's scoring behavior. The session examples show
  concrete input/output pairs (e.g., exact match → 100, long query → 25).
  Without these, CoMet's LLM judge may score differently.
- **P2 — User message format:** Different formatting of the user message.
  May affect LLM interpretation.
- **P3 — Score description wording:** Minor semantic differences in how each
  score tier is described. Core meaning is the same.

#### `time_range_check(LLM)`

| Aspect | Foundry Session | CoMet |
|--------|----------------|-------|
| **Prompt file** | Inline in `time_range_checking.json` | `prompts/time_range_check.md` |
| **Model** | `dev-gpt-5-chat-jj` | Metric-level config |
| **Temperature** | 0 | Metric-level config |
| **Message role** | `assistant` (full prompt with example) | `system` + `user` |
| **Prompt length** | 1227 chars | 368 chars |
| **Has examples** | Yes (1 worked example with specific date) | No |
| **Hardcoded date** | "today is Tue, 11 Mar 2025" | None |
| **Output format** | JSON in triple-quote delimiters (`"""..."""`) | Plain JSON |
| **Post-script** | 1079 chars: `safe_json_loads` → extracts `score` | `_parse_llm_score()` extracts `score` |
| **Scoring logic** | **Match** (both extract `.score` from JSON) | **Match** |

**Discrepancies:**
- **P1 — Missing example:** CoMet prompt omits the worked example showing
  how to evaluate a concrete time range comparison. The session example
  demonstrates the expected JSON output format and reasoning.
- **P1 — Missing date context:** Session prompt includes "today is Tue, 11 Mar
  2025" so the LLM can resolve relative dates. CoMet has no date context,
  which may cause incorrect evaluations for relative time references
  (e.g., "next Monday", "last week").
- **P2 — Output delimiters:** Session expects `"""{"reason": "...", "score": 100}"""`;
  CoMet expects plain JSON. `_parse_llm_score()` handles both, so this is cosmetic.

### 8.3 Category B: Assessment from Custom Dimensions (11 criteria)

For these criteria, CoMet does NOT make LLM calls. Instead, it reads a
pre-computed `assessment` JSON string from the `custom_dimensions` dict
(via `ctx.get("assessment", "")`). The LLM evaluation must be performed
externally and passed in.

**Key question:** When CoMet is called via the production pipeline, who
provides the `assessment` value? If no external system supplies it, these
criteria always return `_SKIPPED` (-1).

#### Post-Script Comparison Summary

| Criterion | Session Script | CoMet Logic | Match? |
|-----------|---------------|-------------|--------|
| `attendance_in_meetings` | `doc['status'] == label1` → 100/0 | Same | ✅ |
| `CHI_LB_precision` | Complex: parse grounding + assessment, compute TP/(TP+FP) | Equivalent (different code structure) | ✅ (needs deep review) |
| `CHI_LB_recall` | Complex: parse grounding + assessment, compute TP/(TP+FN) | Equivalent (different code structure) | ✅ (needs deep review) |
| `JJ_meetings_delegate_query` | Gate + `doc['status'] == {{label1}}` | Gate + `doc['status'] == label` | ⚠️ `label1` vs `label` |
| `meetings_category_criteria` | Gate + ambiguous skip + `doc['status'] == label` | Same | ✅ |
| `OOF_meetings_criterion` | Gate + ambiguous skip + `doc['status'] == label` | Same | ✅ |
| `OOF_meetings_criterion_JJ` | Gate + ambiguous skip + `doc['status'] == label` | Same | ✅ |
| `people_tool_query_accuracy` | Last line as int, clamped [0,100] | Same | ✅ |
| `time_qualifier_check` | Gate + `result == "error"` skip, `"match"` → 100 | Same | ✅ |
| `tool_tense_aligned` | Gate + `tense == expected` → 100, ambiguous → 50, else → 0 | Same | ✅ |
| `tool_tense_aligned_JJ` | Gate + `tense == expected` → 100, ambiguous → 50, else → 0 | Same | ✅ |

#### Potential Issue: `JJ_meetings_delegate_query`

The session post-script compares `doc['status']` against `{{label1}}`,
but CoMet's `_delegate_query()` compares against `ctx.get("label", "")`.
If the data item has both `label` and `label1` fields with different values,
the results will differ.

### 8.4 Category C: LLM Downgrade — `language_match(utterance_queries)`

| Aspect | Foundry Session | CoMet |
|--------|----------------|-------|
| **Approach** | LLM-based (1112-char prompt) | Rule-based (CJK detection) |
| **Model** | `dev-gpt-5-chat-jj` | N/A |
| **What it checks** | LLM evaluates whether tool query parameters preserve the utterance's original language (dialect), allowing keyword translations | Checks only if utterance has CJK characters vs query has CJK characters |
| **Score extraction** | Regex `lang_score: (\d+)` from LLM output | Direct 0/100 |
| **Coverage** | All language pairs (English↔German, Spanish↔French, etc.) | Only CJK↔non-CJK mismatches |

**Impact:** CoMet will miss language mismatches between two non-CJK languages
(e.g., German utterance with English query) or between two CJK languages
(e.g., Japanese utterance with Chinese query). The session LLM-based
approach is significantly more capable.

### 8.5 LLM Judge Model & Parameters

| Model | Temperature | Used By (Session) | CoMet |
|-------|------------|-------------------|-------|
| `dev-gpt-5-chat-jj` | 0 | 18 of 20 instances | Metric-level config (verify) |
| `dev-gpt-4o-gg` | 0 | `attendance_in_meetings` (1 instance) | N/A (assessment from ctx) |
| `dev-gpt-5-chat-jj` | 1 | `people_tool_query_accuracy` (1 instance) | N/A (assessment from ctx) |

### 8.6 Prompt Role Conventions

Sessions use three different message role patterns:

| Pattern | Criteria | CoMet Equivalent |
|---------|----------|------------------|
| `system` + `user` | `attendance_in_meetings`, `people_tool_query_accuracy`, `query_correctness`, `time_qualifier_check` | `system` + `user` (for the 3 CoMet-owned criteria) |
| `assistant` (with `<\|start\|>system` markers) | 9 criteria (CHI_LB, JJ_meetings_delegate, OOF, meetings_rsvp, time_range_check, tool_tense_aligned, language_match, meetings_category) | N/A for assessment-based; `system`+`user` for CoMet-owned |
| `user` only (with `<\|start\|>system` markers) | `tool_tense_aligned` in `meeting_query_time_hints` | N/A (assessment from ctx) |

The `<|start|>system ... <|end|>` markers inside `assistant`/`user` messages
are a Foundry UI artifact that simulates system-role behavior. When CoMet
makes its own LLM calls, it uses proper `system` + `user` roles, which is
functionally equivalent.

### 8.7 LLM Usage Agreement Between Sessions and CoMet

The table below makes the agreement and disagreement explicit for every
direction of LLM usage.

| Direction | Count | Criteria | Notes |
|-----------|-------|----------|-------|
| **Both use LLM** | 3 | `meetings_rsvp`, `query_correctness`, `time_range_check(LLM)` | CoMet sends its own prompt via `preprocess()`. Prompt wording differs but intent matches (see §8.2). |
| **Session uses LLM, CoMet does not** | 12 | `attendance_in_meetings`, `CHI_LB_precision`, `CHI_LB_recall`, `JJ_meetings_delegate_query`, `meetings_category_criteria`, `OOF_meetings_criterion`, `OOF_meetings_criterion_JJ`, `people_tool_query_accuracy`, `time_qualifier_check`, `tool_tense_aligned`, `tool_tense_aligned_JJ`, `language_match(utterance_queries)` | 11 of these are assessment-based: CoMet reads a pre-computed `assessment` JSON from custom dimensions instead of calling the LLM itself. 1 (`language_match`) is downgraded to a deterministic CJK rule (see §8.4). |
| **CoMet uses LLM, session does not** | 0 | — | No such case exists. Every criterion in CoMet's `_LLM_CRITERIA` set also uses LLM in sessions. |

**Special case — `time_range_check(rule_and_LLM_combined)`:** This name
appears in CoMet's `_LLM_CRITERIA` frozenset but does **not** exist in any
Foundry session. It is an alias handled by the same LLM code path as
`time_range_check(LLM)` — both share the `time_range:{LLMcriteria}` prompt
key in `preprocess()`. This is an internal CoMet variant, not a disagreement.

**Summary:** There is no case where CoMet uses LLM but the corresponding
session does not. The only disagreements are in the opposite direction:
12 criteria where sessions use LLM but CoMet either reads an external
assessment (11) or falls back to a simpler rule (1).

### 8.8 Multi-Session LLM Criteria: Prompt & Script Consistency

Five of the 15 LLM criteria appear in two sessions each (10 of the 20
instances). The table below shows whether the prompt template and post-
processing script are identical across sessions for each criterion.

| Criterion | Sessions | Prompt | Script | Details |
|-----------|----------|--------|--------|---------|
| `CHI_LB_precision` | `completeness_hints_*_1.3`, `completeness_hints_*_email_focused` | **Differ** (1 char) | **Identical** | Prompt differs only by a trailing period on the final instruction sentence. Semantically identical. |
| `CHI_LB_recall` | `completeness_hints_*_1.3`, `completeness_hints_*_email_focused` | **Identical** | **Differ** | Script uses `tool in str(query.get('domain', ''))` (substring) in `_1.3` vs `query['domain'] == tool` (exact match) in `_email_focused`. This is the same divergence documented in §7.2. |
| `OOF_meetings_criterion_JJ` | `meetings_oof_others`, `meetings_oof_self` | **Identical** | **Identical** | Fully consistent across sessions. |
| `tool_tense_aligned` | `meetings_oof_everyone`, `meeting_query_time_hints` | **Differ** (major) | **Differ** (major) | Prompt: role=`assistant` 1484 chars vs role=`user` 957 chars. Script: 4196 vs 7457 chars. Substantially different implementations documented in §7.2. |
| `tool_tense_aligned_JJ` | `meetings_oof_others`, `meetings_oof_self` | **Identical** | **Identical** | Fully consistent across sessions. Shares the same prompt hash as `tool_tense_aligned` in `meetings_oof_everyone`. |

**Consistency summary:**
- **3 of 5** criteria are fully consistent (or differ by only 1 trivial character).
- **2 of 5** have meaningful divergence:
  - `CHI_LB_recall`: substring vs exact domain matching (see §7.2 for details).
  - `tool_tense_aligned`: entirely different prompt wording, message role, and
    post-processing logic between the two sessions (see §7.2 for details).

#### How CoMet Handles Multi-Session LLM Criteria

CoMet uses a single Python function per criterion name via `CRITERIA_REGISTRY`,
regardless of which session the criterion originates from:

| Criterion | CoMet Function | LLM? | Handling |
|-----------|---------------|-------|----------|
| `CHI_LB_precision` | `_chi_lb_precision()` | No (assessment) | Reads `ctx.get("assessment", "")`. Single implementation — session-level prompt/script differences are invisible to CoMet. |
| `CHI_LB_recall` | `_chi_lb_recall()` | No (assessment) | Same as above. The substring-vs-exact divergence in session scripts does not affect CoMet (it only reads the assessment result). |
| `OOF_meetings_criterion_JJ` | `_oof_meetings_criterion()` | No (assessment) | Reads `ctx.get("assessment", "")`. Identical across sessions, so no issue. |
| `tool_tense_aligned` | `_tool_tense_aligned()` | No (assessment) | Reads `ctx.get("assessment", "")`. Handles both session variants via `ctx.get("tense") or ctx.get("label")` fallback. |
| `tool_tense_aligned_JJ` | `_tool_tense_aligned()` | No (assessment) | Same function as `tool_tense_aligned`. Identical across sessions. |

**Key insight:** Because all 5 multi-session LLM criteria are in Category B
(assessment from custom dimensions), CoMet never makes its own LLM calls for
them. The prompt and script divergence in sessions does not directly affect
CoMet's evaluation logic — it only matters if the external system producing
the `assessment` value uses different prompts per session, which would then
flow through as different `assessment` inputs to the same CoMet function.

For the 3 Category A criteria (`meetings_rsvp`, `query_correctness`,
`time_range_check(LLM)`), each appears in only one session, so the
multi-session consistency question does not apply.

---

## 9. Fix Plan

### Phase 0 — Fix Misspelled Criterion Key (this repo): DONE ✅

| Old Key (misspelled) | New Key (correct) |
|---|---|
| `wokrplace_harm_recall` | `workplace_harm_recall` |

See §2 for root cause.

### Phase 1 — Fix Foundry Session Files (Sydney repo): DONE ✅

All 36 session file renames and all 78 criteria name changes have been applied
and verified. 61 session files, 114 unique criteria names — all pass convention
checks (§6.1) with zero violations.

| # | Change | Status |
|---|---|---|
| 1 | Rename 36 session files (`-` → `_`) | **DONE** ✅ |
| 2 | Update criteria `name` fields for all 78 changed names | **DONE** ✅ |

### Phase 2 — Rename CoMet CRITERIA_REGISTRY Keys (this repo): DONE ✅

Renamed 76 CRITERIA_REGISTRY keys to match session names exactly and deleted
22 alias keys. Old keys were **replaced** (not kept as aliases). Registry went
from 138 → 116 keys (114 matching session criteria + 2 standalone).

**38 criteria already match** (no change needed): `Accuracy`, `NoSearch`,
`OnlyOneTool`, `WebAndWork`, `WebOnly`, `WorkOnly`, `chat_precision`,
`chat_recall`, `correct_filters_format`, `designer_graphic_art_precision`,
`designer_graphic_art_recall`, `did_early_bind`, `did_fanout_more_than_2`,
`emails_precision`, `emails_recall`, `fetch_email_recall`, `fetch_event_recall`,
`files_invoked`, `files_precision`, `files_recall`, `keyword_early_fanout`,
`keywordless_early_binding`, `meetings_precision`, `meetings_recall`,
`meetings_rsvp`, `meta_prioritize_not_called`, `neither_unwanted(loose)`,
`neither_unwanted(strict)`, `people_precision`, `people_recall`,
`python_execution_precision`, `python_execution_recall`, `query_correctness`,
`record_memory_precision`, `record_memory_recall`, `should_not_invoke`,
`workplace_harm_prec.`, `workplace_harm_recall`

**76 keys to rename:**

| # | Old CoMet Key | New Key (matches session) |
|---|---|---|
| 1 | `CHI - LB - Precision` | `CHI_LB_precision` |
| 2 | `CHI - LB - Recall` | `CHI_LB_recall` |
| 3 | `GC QuSuccess` | `GC_qu_success` |
| 4 | `GC Tool Precision` | `GC_tool_precision` |
| 5 | `GC Tool Recall` | `GC_tool_recall` |
| 6 | `GCaaP completness` | `GCaaP_completeness` |
| 7 | `IsEnriched (TrueNegative) Python` | `IsEnriched(TrueNegative)_python` |
| 8 | `IsEnriched (TruePositive) Python` | `IsEnriched(TruePositive)_python` |
| 9 | `JJ meetings_delegate_query` | `JJ_meetings_delegate_query` |
| 10 | `OOF Meetings Criterion` | `OOF_meetings_criterion` |
| 11 | `OOF Meetings Criterion JJ` | `OOF_meetings_criterion_JJ` |
| 12 | `OneDrive or SharePoint Preserved Rate (loose)` | `OneDrive_or_SharePoint_preserved_rate(loose)` |
| 13 | `OneDrive or SharePoint Preserved Rate (strict)` | `OneDrive_or_SharePoint_preserved_rate(strict)` |
| 14 | `URL Hallucination Rate (loose)` | `URL_hallucination_rate(loose)` |
| 15 | `URL Hallucination Rate (strict)` | `URL_hallucination_rate(strict)` |
| 16 | ` URL Preserved(loose)` | `URL_preserved(loose)` |
| 17 | ` URL Preserved(strict)` | `URL_preserved(strict)` |
| 18 | `URL Preserved Rate (loose)` | `URL_preserved_rate(loose)` |
| 19 | `URL Preserved Rate (strict)` | `URL_preserved_rate(strict)` |
| 20 | `All paramters rate 2nd Turn` | `all_parameters_rate_2nd_turn` |
| 21 | `Attendance in Meetings` | `attendance_in_meetings` |
| 22 | `Bad Mistakes` | `bad_mistakes` |
| 23 | `Calendar Precision` | `calendar_precision` |
| 24 | `Calendar Recall` | `calendar_recall` |
| 25 | `canmore_create trigger rate (higher is better)` | `canmore_create_trigger_rate(higher_is_better)` |
| 26 | `canmore_create trigger rate (lower is better)` | `canmore_create_trigger_rate(lower_is_better)` |
| 27 | `canmore_update trigger rate (higher is better)` | `canmore_update_trigger_rate(higher_is_better)` |
| 28 | `canmore_update trigger rate (lower is better)` | `canmore_update_trigger_rate(lower_is_better)` |
| 29 | `check folder (loose)` | `check_folder(loose)` |
| 30 | `check folder (strict)` | `check_folder(strict)` |
| 31 | `check folder+search type for people query(loose)` | `check_folder_and_search_type_for_people_query(loose)` |
| 32 | `check folder+search type for people query(strict)` | `check_folder_and_search_type_for_people_query(strict)` |
| 33 | `direct-reports-suffix-correct (loose)` | `direct_reports_suffix_correct(loose)` |
| 34 | `direct-reports-suffix-correct (strict)` | `direct_reports_suffix_correct(strict)` |
| 35 | `Email MailBox Recall` | `email_mailbox_recall` |
| 36 | `Emails hallucination rate 1st Turn` | `emails_hallucination_rate_1st_turn` |
| 37 | `Empty time parameters rate 2nd Turn` | `empty_time_parameters_rate_2nd_turn` |
| 38 | `get_webpage_context precision` | `get_webpage_context_precision` |
| 39 | `get_webpage_context recall` | `get_webpage_context_recall` |
| 40 | `Has Meeting Tool` | `has_meeting_tool` |
| 41 | `Hit file action keyword(loose)` | `hit_file_action_keyword(loose)` |
| 42 | `Hit file action keyword(strict)` | `hit_file_action_keyword(strict)` |
| 43 | `Is a Nice Serialized JSON` | `is_a_nice_serialized_JSON` |
| 44 | `Language Match (utterance-queries)` | `language_match(utterance_queries)` |
| 45 | `manager name in query (loose)` | `manager_name_in_query(loose)` |
| 46 | `manager name in query (strict)` | `manager_name_in_query(strict)` |
| 47 | `Meetings Category Criteria` | `meetings_category_criteria` |
| 48 | `meta_prioritized incorrectly triggered` | `meta_prioritized_incorrectly_triggered` |
| 49 | `miss "past" test (loose)` | `miss_"past"_test(loose)` |
| 50 | `miss "past" test (strict)` | `miss_"past"_test(strict)` |
| 51 | `miss \u201cthe\u201d test (loose)` | `miss_"the"_test(loose)` |
| 52 | `miss \u201cthe\u201d test (strict)` | `miss_"the"_test(strict)` |
| 53 | `people domain eval` | `people_domain_eval` |
| 54 | `People Search 1st Turn` | `people_search_1st_turn` |
| 55 | `PeopleTool_QueryAccuracy` | `people_tool_query_accuracy` |
| 56 | `PeopleTool_Triggering` | `people_tool_triggering` |
| 57 | `Prep Precision` | `prep_precision` |
| 58 | `Prep Recall` | `prep_recall` |
| 59 | `Question Preserved Rate (loose)` | `question_preserved_rate(loose)` |
| 60 | `Question Preserved Rate (strict)` | `question_preserved_rate(strict)` |
| 61 | `Recap Precision` | `recap_precision` |
| 62 | `Recap Recall` | `recap_recall` |
| 63 | `retain my, me, i in query (loose)` | `retain_my_me_i_in_query(loose)` |
| 64 | `retain my, me, i in query (strict)` | `retain_my_me_i_in_query(strict)` |
| 65 | `Scheduling triggered 2nd Turn` | `scheduling_triggered_2nd_turn` |
| 66 | `search_web trigger rate (higher is better)` | `search_web_trigger_rate(higher_is_better)` |
| 67 | `time qualifier chechk` | `time_qualifier_check` |
| 68 | `Time Range Check (LLM)` | `time_range_check(LLM)` |
| 69 | `tool tense aligned` | `tool_tense_aligned` |
| 70 | `tool tense aligned JJ` | `tool_tense_aligned_JJ` |
| 71 | `Trigger rate` | `trigger_rate` |
| 72 | `user name not resolved(loose)` | `user_name_not_resolved(loose)` |
| 73 | `user name not resolved(strict)` | `user_name_not_resolved(strict)` |
| 74 | `Web Recall` | `web_recall` |
| 75 | `Wrong parameter rate 2nd Turn` | `wrong_parameter_rate_2nd_turn` |
| 76 | `Wrong Reasoning` | `wrong_reasoning` |

**Also delete these 22 alias keys** (extra keys that are not used by any session
and exist only as backward-compatible aliases — see §3). The 2 standalone
criteria (#22, #23) are kept.

| # | Key to Delete | Reason (§3 #) |
|---|---|---|
| 1 | `wokrplace_harm_recall` | Typo alias (§3 #1) |
| 2 | `user name not resolved` | Suffix-less alias (§3 #2) |
| 3 | `miss "the" test (strict)` | Straight-quote alias (§3 #3) |
| 4 | `miss "the" test (loose)` | Straight-quote alias (§3 #4) |
| 5 | `miss "the" test` | Suffix-less alias (§3 #5) |
| 6 | `miss "past" test` | Suffix-less alias (§3 #6) |
| 7 | `URL Preserved(strict)` | Leading-space-stripped alias (§3 #7) |
| 8 | `URL Preserved(loose)` | Leading-space-stripped alias (§3 #8) |
| 9 | `URL Preserved` | Suffix-less alias (§3 #9) |
| 10 | `URL Preserved Rate` | Suffix-less alias (§3 #10) |
| 11 | `Question Preserved Rate` | Suffix-less alias (§3 #11) |
| 12 | `OneDrive or SharePoint Preserved Rate` | Suffix-less alias (§3 #12) |
| 13 | `URL Hallucination Rate` | Suffix-less alias (§3 #13) |
| 14 | `Hit file action keyword` | Suffix-less alias (§3 #14) |
| 15 | `neither_unwanted` | Suffix-less alias (§3 #15) |
| 16 | `neither_unwanted (loose)` | Whitespace-before-paren alias (§3 #16) |
| 17 | `CHI-LB-Precision` | Compact-hyphen alias (§3 #17) |
| 18 | `CHI-LB-Recall` | Compact-hyphen alias (§3 #18) |
| 19 | `evaluate_language_consistency` | Filename alias (§3 #19) |
| 20 | `Time Range Check (Rule + LLM combined)` | Variant name (§3 #20) |
| 21 | `Custom Criteria` | Variant name (§3 #21) |
| 22 | `direct-reports-suffix-correct(loose)` | Whitespace-before-paren alias (§3 #24) |

> **Not deleted (kept):**
> - `search_office365_called` (§3 #22) — standalone criterion with unique implementation
> - `search_office365_called_exactly_once` (§3 #23) — standalone criterion with unique implementation
>
> **Result:** Deleted 22 alias keys + renamed 76 primary keys → registry went from
> 138 to **116 keys** (114 matching session criteria + 2 standalone). **DONE ✅**

### Phase 3 — Fix Logic Discrepancies & Remove Standalone Criteria: DONE ✅

#### 3.1 Delete 2 standalone criteria not in any session: DONE ✅

Deleted both standalone criteria from CRITERIA_REGISTRY and their function
implementations. No Foundry session references these criteria.

- [x] `search_office365_called` — deleted
- [x] `search_office365_called_exactly_once` — deleted

Registry: 116 → **114 keys** (zero extras remaining).

#### 3.2 Fix CoMet–Foundry logic discrepancies (see §4): DONE ✅

All 4 criteria rewritten to match the Foundry session scripts exactly.

- [x] **P1** `user_name_not_resolved(strict/loose)` — now parses chats/emails
  domain queries in office365.search, uses AND/OR logic, returns SKIPPED when
  no chats/emails domain triggered
- [x] **P1** `direct_reports_suffix_correct(strict/loose)` — now checks
  people-domain `query.endswith("direct reports")`, with AND/OR logic and
  SKIPPED when people domain not triggered
- [x] **P1** `IsEnriched(TruePositive/TrueNegative)_python` — full rewrite:
  query extraction (FluxV2/V3/JSON), text preprocessing (punctuation removal,
  lowercasing, stop words, stemming), query-vs-utterance comparison, profile
  field value matching across all fields
- [x] **P2** `files_invoked` — legacy path now also checks
  `search_enterprise_fanout` and `'"files"'`

#### 3.3 Resolve same-criterion script divergence across sessions (see §7.2): DONE ✅

These criteria appeared in multiple sessions with different evaluation scripts.
CoMet uses a single implementation per criterion. Resolution strategy: where
the scripts are genuinely different (not just bugs/formatting), rename the
criteria to differentiate the implementations rather than force-unifying.

- [x] **`meetings_recall`** — 4 variants across 7 sessions. Variant A had a
  latent bug (`or "office365_search"` always truthy). Variant D (from
  `meetings_category_queries_time_hints`) was the most robust (proper error
  handling, correct TailsBerry `search_office365` scoring). **All 7 sessions
  unified to Variant D.**

- [x] **`CHI_LB_recall`** — `completeness_hints_*_1.3` uses substring domain
  match (`domain in name`); `completeness_hints_*_email_focused` uses exact
  match (`name == domain`). These are intentionally different evaluation
  strategies, so they were **renamed to distinct criteria**:
  - `CHI_LB_recall(domain_contains)` — substring match (in `*_1.3`)
  - `CHI_LB_recall(domain_exact)` — exact match (in `*_email_focused`)

- [x] **`tool_tense_aligned`** — `meetings_oof_everyone` and
  `meeting_query_time_hints` differ in three ways: (1) different LLM prompts
  (different role and wording), (2) different post-script parser robustness
  (~4.2 KB simple vs ~7.5 KB robust), and most importantly (3) they read the
  expected tense from **different input fields** — `{{tense}}` vs `{{label}}`
  — because the two sessions structure their dataItems differently (in
  `meetings_oof_everyone`, `label` means OOF status; in
  `meeting_query_time_hints`, `label` means the expected tense and there is
  no `tense` field). The scripts cannot be unified without also restructuring
  one session's input fields. They were **renamed to distinct criteria**
  (named after their session context):
  - `tool_tense_aligned(oof)` — used in `meetings_oof_everyone` (reads `{{tense}}`)
  - `tool_tense_aligned(time_hints)` — used in `meeting_query_time_hints` (reads `{{label}}`)

#### 3.4 Unify trivial prompt divergence (see §7.2): DONE ✅

`CHI_LB_precision` had identical scripts but the prompt differed by a trailing
period on the final instruction sentence. The `_email_focused` version (with
period) is the correct one (proper sentence ending).

- [x] Copied `CHI_LB_precision` prompt from `completeness_hints_*_email_focused`
  into `completeness_hints_*_1.3` — both now identical (hash `1aa8d738bd2e`,
  6919 chars).

#### 3.5 Update CoMet tests: DONE ✅

- [x] Update any existing unit tests that reference old CRITERIA_REGISTRY key
  names (renamed in Phase 2) — DONE ✅ (13 tests fixed)
- [x] Rewrite tests for the 4 logic-discrepancy criteria to match new
  implementations — DONE ✅ (23 tests: 6 user_name_not_resolved,
  6 direct_reports_suffix, 9 is_enriched, 2 files_invoked).
- [x] Add tests for Phase 3.3 criterion splits — DONE ✅ (6 new tests:
  domain_contains variant, domain_exact vs contains comparison,
  non-combined path equivalence, registry name assertions).
  All 425 tests pass.

### Phase 4 — LLM Usage & Prompt Alignment (see §8)

Foundry sessions use LLM judges for 15 unique criteria (20 instances).
CoMet only makes its own LLM calls for 3 of them; 11 rely on a pre-computed
`assessment` custom dimension; 1 is downgraded to a deterministic rule.
This phase addresses prompt and post-script discrepancies.

#### 4.1 Fix CoMet prompt discrepancies (3 criteria)

- [x] **`query_correctness`** — DONE ✅. Ported the full session system prompt
  including the 3 worked examples (Example 1.1: chat+emails+meta_prioritize → 100;
  Example 1.2: meetings query_type=recap → 100; Example 1.3: office365_search
  complex query → 25). Aligned user-message format to `result:{text}\nexpectedQuery:{expected}`
  to match the session template.
- [x] **`time_range_check(LLM)`** — DONE ✅. Ported the full session system
  prompt including the worked example. Updated `_build_time_range_messages`
  user format to mirror the session's "Now please help me determine the
  following case" wrapper (Context A / Context B / `Judgement:` cue) so the
  LLM emits the triple-quoted JSON the example demonstrates. Extended
  `_parse_llm_score` to strip leading/trailing `"""` delimiters before JSON
  decoding so the triple-quote-wrapped output is parsed correctly.
- [x] **`meetings_rsvp`** — DONE ✅. Replaced the condensed CoMet rewrite with
  the full session system prompt (status-category examples, "Pending responses
  means not responded", `grouped by RSVP` / `based on RSVP` rules, JSON output
  schema with all 10 status values).

  Note: CoMet still pre-extracts the meetings query via
  `_extract_meetings_query` and skips the LLM call when no meetings query is
  present (a cost optimization). The session passes the full tool-call text to
  the LLM. Both inputs work because the prompt instructs the LLM to "look at
  the query of the `search_enterprise_meetings` tool" — the optimization is
  preserved intentionally.

#### 4.2 Align `tool_tense_aligned` prompts across sessions

The two `tool_tense_aligned` variants — `(oof)` in `meetings_oof_everyone` and
`(time_hints)` in `meeting_query_time_hints` — are genuinely distinct criteria
because they read the expected tense from **different input fields** (`{{tense}}`
vs `{{label}}`). They cannot be unified into one (see §7.2 and §3.3).

However, their **LLM prompts** also differ substantially. Since both prompts
produce a `{{assessment}}` JSON with the same schema (`tense_words`, `intent`,
`tense`), the prompt differences may cause scoring divergence on the same query:

| Aspect | `tool_tense_aligned(oof)` | `tool_tense_aligned(time_hints)` |
|--------|--------------------------|----------------------------------|
| Prompt role | `assistant` | `user` |
| Prompt length | 1484 chars | 957 chars |
| Date context | "Current date is May 15, 2025" + date examples | None |

**Post-script alignment:** Both scripts score identically on well-formed
assessment JSON (`tense == expected → 100, ambiguous → 50, else → 0`). CoMet's
`_tool_tense_aligned` matches this logic and handles both input field patterns
via `ctx.get("tense") or ctx.get("label", "")`. **Scripts match — DONE ✅.**

**Action items (prompts — deferred to Phase 4):**
- [ ] Decide whether both prompts should include date context (the `oof` prompt
  is more informative for tense resolution with date-relative queries).
- [ ] Standardize message role (`assistant` vs `user`).

#### 4.3 Fix `language_match(utterance_queries)` — LLM downgrade

- [ ] Session uses full LLM-based language evaluation (1112-char prompt, model
  `dev-gpt-5-chat-jj`, temp 0) with regex-based `lang_score` extraction.
  CoMet uses a CJK-only character-class check (no LLM). This misses
  non-CJK language mismatches (e.g. German↔English, Spanish↔French).
  Decide: upgrade CoMet to use LLM, or accept the approximation.

#### 4.4 Verify assessment-based criteria post-scripts (11 criteria)

These criteria use `ctx.get("assessment", "")` — the LLM call is external
and the assessment JSON is passed as a custom dimension. Verify that CoMet's
post-processing logic matches the session post-scripts:

- [ ] `attendance_in_meetings` — Session: `doc['status'] == label1` → 100/0.
  CoMet: same. **Likely matches.** Verify no meetings gate in session
  (confirmed: none).
- [ ] `CHI_LB_precision` — Complex (6869-char script). Session handles both
  combined-tool and non-combined paths with `remove_plugins` filtering.
  CoMet has equivalent logic. **Deep review needed.**
- [ ] `CHI_LB_recall` — Complex (8173-char script). Same structure as precision
  but recall formula. **Deep review needed.**
- [ ] `JJ_meetings_delegate_query` — Session: meetings gate + `doc['status'] == label1`.
  CoMet: meetings gate + `doc['status'] == label`. **Check: session uses
  `label1`, CoMet uses `label`. May differ if both vars exist.**
- [ ] `meetings_category_criteria` — Session: meetings gate + ambiguous skip +
  `doc['status'] == label`. CoMet: same. **Likely matches.**
- [ ] `OOF_meetings_criterion` / `OOF_meetings_criterion_JJ` — Session:
  meetings gate + ambiguous skip + `doc['status'] == label`. CoMet: same.
  **Likely matches.**
- [ ] `people_tool_query_accuracy` — Session: last line of assessment parsed
  as int, clamped to [0, 100]. CoMet: same. **Matches.**
- [ ] `time_qualifier_check` — Session: meetings gate + `result == "error"` →
  skip, `result == "match"` → 100, else → 0. CoMet: same. **Matches.**
- [ ] `tool_tense_aligned` / `tool_tense_aligned_JJ` — Session: meetings gate +
  `doc['tense'] == expected` → 100, `ambiguous` → 50, else → 0.
  CoMet: same. **Check: session `meetings_oof_*` uses `{{tense}}` var;
  `meeting_query_time_hints` uses `{{label}}`. CoMet tries both.**

#### 4.5 Verify LLM judge model alignment

- [ ] Session default model: `dev-gpt-5-chat-jj` (19/20 instances).
  Exception: `attendance_in_meetings` uses `dev-gpt-4o-gg`.
  CoMet configures model at the metric level — verify it matches.
- [ ] `people_tool_query_accuracy` uses `temperature=1` (only non-zero).
  All others use `temperature=0`.
