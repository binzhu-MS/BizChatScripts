# Session DataItem Duplication Statistics

**Generated:** 2026-04-08
**Source directory:** `sessions`
**Files analyzed:** 61

## Summary

| Metric | Value |
|--------|-------|
| Total dataItems | 12,337 |
| Unique input strings | 9,902 |
| **Type 1 extra copies** (exact input dups) | **2,435** |
| Type 1 affected utterances | 731 |
| **Type 2 utterances** (same utt, diff input) | **33** |
| Files with Type 1 dups | 33 / 61 |
| Files with Type 2 dups | 10 / 61 |
| Clean files (no dups) | 24 / 61 |

### Duplication types

- **Type 1 (exact input duplicate):** The entire `input` JSON string is identical across multiple dataItems. These are redundant copies — the `dedup_session_utterances.py` script removes them.
- **Type 2 (same utterance, different input):** The `utterance` field matches but other fields (e.g. `timestamp`, `segment`, `label`) differ. These are typically **intentional variants** (same query tested under different conditions).

## Distribution of Duplication Counts (Type 1)

How many times each duplicated input appears (including the original copy):

| Copies per utterance | Utterances | Extra copies |
|---:|---:|---:|
| 2 | 201 | 201 |
| 3 | 93 | 186 |
| 4 | 4 | 12 |
| 5 | 376 | 1,504 |
| 6 | 2 | 10 |
| 10 | 54 | 486 |
| 37 | 1 | 36 |
| **Total** | **731** | **2,435** |

The dominant pattern is **5 copies** (376 utterances, 62% of total extra copies) — likely from a seed-index expansion (indices 25, 50, 75, 100, 175). The 10-copy cluster (54 utterances in `do_not_resolve_i_me_my`) and the single 37-copy outlier (`people-direct-reports_one_utterance_only`) are also notable.

## Type 1: Exact Input Duplicates

Files with exact input duplicates, sorted by number of extra copies (descending).

| Session file | DataItems | Unique | Extra copies | Dup utterances | Dup % |
|---|---:|---:|---:|---:|---:|
| `base-sessions/prompt_triggering_meetings.json` | 1735 | 1236 | 499 | 128 | 29% |
| `base-sessions/do_not_resolve_i_me_my.json` | 510 | 51 | 459 | 51 | 90% |
| `search_enterprise_connectors/prompt_triggering-search_enterprise_connectors.json` | 260 | 52 | 208 | 52 | 80% |
| `base-sessions/fanout_calibration.json` | 250 | 50 | 200 | 50 | 80% |
| `base-sessions/time_range_checking.json` | 257 | 99 | 158 | 40 | 61% |
| `base-sessions/early_binding_for_explicit_domains.json` | 195 | 38 | 157 | 38 | 81% |
| `base-sessions/completeness_hints-hints_p_r_evaluation-email_focused.json` | 225 | 75 | 150 | 75 | 67% |
| `base-sessions/jp-file_action.json` | 185 | 36 | 149 | 36 | 81% |
| `base-sessions/email_contains_key_info.json` | 310 | 226 | 84 | 21 | 27% |
| `fetch-event/prompt_triggering-fetch_event.json` | 135 | 72 | 63 | 63 | 47% |
| `base-sessions/meetings-category-queries.json` | 191 | 129 | 62 | 56 | 32% |
| `edge-context-first-hop/triggering-edge_context_files_calibration.json` | 65 | 13 | 52 | 13 | 80% |
| `base-sessions/people-direct-reports_one_utterance_only.json` | 37 | 1 | 36 | 1 | 97% |
| `base-sessions/international_my_manager.json` | 66 | 33 | 33 | 33 | 50% |
| `base-sessions/summarize_in_language.json` | 30 | 6 | 24 | 6 | 80% |
| `base-sessions/url_queries.json` | 36 | 12 | 24 | 12 | 67% |
| `base-sessions/meeting-keywords-in-the-next-week.json` | 100 | 84 | 16 | 10 | 16% |
| `base-sessions/meeting-keywords-past.json` | 100 | 87 | 13 | 9 | 13% |
| `scheduling-handoff-second-hop/prompt_triggering-scheduling_handoff_second_hop.json` | 211 | 201 | 10 | 2 | 5% |
| `base-sessions/completeness_hints-hints_p_r_evaluation_1.3.json` | 268 | 260 | 8 | 8 | 3% |
| `base-sessions/files_scope_url_folder.json` | 304 | 298 | 6 | 6 | 2% |
| `base-sessions/meeting_query_time_hints.json` | 229 | 223 | 6 | 6 | 3% |
| `base-sessions/attendance_in_meetings_search.json` | 75 | 72 | 3 | 3 | 4% |
| `edge-context-first-hop/prompt_triggering-edge_context.json` | 118 | 115 | 3 | 1 | 3% |
| `edge-context-second-hop/prompt_triggering-edge_context_second_hop.json` | 43 | 40 | 3 | 2 | 7% |
| `base-sessions/canvas_create_overtriggering.json` | 50 | 48 | 2 | 2 | 4% |
| `base-sessions/meetings-delegate.json` | 153 | 152 | 1 | 1 | 1% |
| `base-sessions/meetings-oof-everyone.json` | 100 | 99 | 1 | 1 | 1% |
| `base-sessions/meetings-oof-others.json` | 146 | 145 | 1 | 1 | 1% |
| `base-sessions/meetings-oof-self.json` | 208 | 207 | 1 | 1 | 0% |
| `base-sessions/personalization_with_user_profile.json` | 200 | 199 | 1 | 1 | 0% |
| `base-sessions/query-correctness.json` | 21 | 20 | 1 | 1 | 5% |
| `base-sessions/triggering-transcript_search.json` | 81 | 80 | 1 | 1 | 1% |

**Total extra copies:** 2,435

## Type 2: Same Utterance, Different Input

### Distribution of Variant Counts (Type 2)

How many distinct input variants exist per utterance (only utterances with 2+ variants):

| Variants per utterance | Utterances | Extra variant items |
|---:|---:|---:|
| 2 | 25 | 25 |
| 3 | 1 | 2 |
| 4 | 1 | 3 |
| 5 | 1 | 4 |
| 9 | 1 | 8 |
| **Total** | **29** | **42** |

The vast majority (25/29) have exactly 2 variants. The 9-variant outlier is **"when is my meeting with Rajesh Singh and Gaurav Sareen together?"** in `meeting_query_time_hints.json`, tested with different `current-date` / `timestamp` combinations.

### Per-file breakdown

Files where the same utterance text appears with different input JSON (intentional variants).

| Session file | DataItems | Variant utterances | Total variant items |
|---|---:|---:|---:|
| `base-sessions/meeting_query_time_hints.json` | 229 | 17 | 47 |
| `update-canvas/canvas_grounding.json` | 79 | 5 | 10 |
| `base-sessions/files_scope_url_folder.json` | 304 | 2 | 4 |
| `base-sessions/people_eval_domain.json` | 100 | 2 | 4 |
| `base-sessions/triggering-files.json` | 530 | 2 | 4 |
| `base-sessions/completeness_hints-hints_p_r_evaluation_1.3.json` | 268 | 1 | 2 |
| `base-sessions/meetings-category-queries.json` | 191 | 1 | 2 |
| `base-sessions/prompt_triggering_meetings.json` | 1735 | 1 | 5 |
| `base-sessions/query-correctness.json` | 21 | 1 | 2 |
| `base-sessions/triggering-people.json` | 169 | 1 | 2 |

### `base-sessions/meeting_query_time_hints.json`

- **"Is there a meeting with the operations team on Tuesday?"** — 2 variants, 2 items
  - 1x with fields: `current-date`, `ideal`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `ideal`, `label`, `session`, `timestamp`
- **"What was discussed in the meeting with the finance department on tuesd..."** — 2 variants, 2 items
  - 1x with fields: `current-date`, `ideal`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `ideal`, `label`, `session`, `timestamp`
- **"Is there a meeting with the operations team?"** — 3 variants, 3 items
  - 1x with fields: `current-date`, `ideal`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `ideal`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `ideal`, `label`, `session`, `timestamp`
- **"Do I have anything lined up with the HR team on Thursday"** — 2 variants, 2 items
  - 1x with fields: `current-date`, `ideal`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `ideal`, `label`, `session`, `timestamp`
- **"meeting on Monday"** — 2 variants, 2 items
  - 1x with fields: `current-date`, `ideal`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `ideal`, `label`, `session`, `timestamp`
- **"meetings on Thursday"** — 2 variants, 2 items
  - 1x with fields: `current-date`, `ideal`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `ideal`, `label`, `session`, `timestamp`
- **"my SLT meetings on sunday"** — 2 variants, 2 items
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
- **"my meetings on sunday"** — 2 variants, 2 items
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
- **"my meetings on Friday"** — 2 variants, 2 items
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
- **"Do I have any meetings left for monday evening"** — 2 variants, 2 items
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
- **"Was there any meeting about LU on Monday"** — 2 variants, 2 items
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
- **"Is there any meeting about LU on Monday"** — 2 variants, 2 items
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
- **"when is my meeting with Manu Singhal and Avinash Kumar together?"** — 2 variants, 2 items
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
- **"when is my meeting with Rajesh Singh and Gaurav Sareen together?"** — 9 variants, 9 items
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
- **"when is my meeting with Sarah Williams and Laura Peterson together?"** — 5 variants, 5 items
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
- **"When is Sarah Peterson meeting with Laura Bond?"** — 2 variants, 2 items
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
- **"When do I have a meeting scheduled with Sarah Williams and Laura Peter..."** — 4 variants, 4 items
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`
  - 1x with fields: `current-date`, `label`, `session`, `timestamp`

### `update-canvas/canvas_grounding.json`

- **"Which statements are fact and which are fiction?"** — 2 variants, 2 items
  - 1x with fields: `filecontent`, `filename`, `segment`, `session`
  - 1x with fields: `filecontent`, `filename`, `segment`, `session`
- **"Summarize this"** — 2 variants, 2 items
  - 1x with fields: `filecontent`, `filename`, `segment`, `session`
  - 1x with fields: `filecontent`, `filename`, `segment`, `session`
- **"Summarize this page"** — 2 variants, 2 items
  - 1x with fields: `filecontent`, `filename`, `segment`, `session`
  - 1x with fields: `filecontent`, `filename`, `segment`, `session`
- **"What's on the page?"** — 2 variants, 2 items
  - 1x with fields: `filecontent`, `filename`, `segment`, `session`
  - 1x with fields: `filecontent`, `filename`, `segment`, `session`
- **"How would this look like in C++?"** — 2 variants, 2 items
  - 1x with fields: `filecontent`, `filename`, `segment`, `session`
  - 1x with fields: `filecontent`, `filename`, `segment`, `session`

### `base-sessions/files_scope_url_folder.json`

- **"LinkedId file in https://microsoft.sharepoint-df.com/sites/USRLanguage..."** — 2 variants, 2 items
  - 1x with fields: (no other keys)
  - 1x with fields: (no other keys)
- **"summarize of recent news posts from the site https://microsoft.sharepo..."** — 2 variants, 2 items
  - 1x with fields: (no other keys)
  - 1x with fields: (no other keys)

### `base-sessions/people_eval_domain.json`

- **"Who works in Cloud Migration?"** — 2 variants, 2 items
  - 1x with fields: `control_metrics`, `current_date`
  - 1x with fields: `control_metrics`, `current_date`
- **"Who works in FinTech?"** — 2 variants, 2 items
  - 1x with fields: `control_metrics`, `current_date`
  - 1x with fields: `control_metrics`, `current_date`

### `base-sessions/triggering-files.json`

- **"Can you summarize KB0065021?"** — 2 variants, 2 items
  - 1x with fields: `segment`, `session`
  - 1x with fields: `session`
- **"What is Lynx from KB0066271 knowledge base?"** — 2 variants, 2 items
  - 1x with fields: `segment`, `session`
  - 1x with fields: `session`

### `base-sessions/completeness_hints-hints_p_r_evaluation_1.3.json`

- **"Display the messages I've emailed to Milad Shokouhi"** — 2 variants, 2 items
  - 1x with fields: `grounding`
  - 1x with fields: `grounding`

### `base-sessions/meetings-category-queries.json`

- **"mijn vergaderingen met categorie urgent"** — 2 variants, 2 items
  - 1x with fields: `label`
  - 1x with fields: `label`

### `base-sessions/prompt_triggering_meetings.json`

- **"Create docs with information on TCR."** — 2 variants, 5 items
  - 4x with fields: `session`
  - 1x with fields: `session`

### `base-sessions/query-correctness.json`

- **"List the assignments from today's calls relevant to me."** — 2 variants, 2 items
  - 1x with fields: `expectedQuery`
  - 1x with fields: `expectedQuery`

### `base-sessions/triggering-people.json`

- **"What information do you have about me?"** — 2 variants, 2 items
  - 1x with fields: `label`, `session`
  - 1x with fields: `session`

## Clean Files (No Duplicates)

- `base-sessions/canvas_create.json` (84 dataItems, 84 unique utterances)
- `base-sessions/canvas_create_overtriggering_doc_CI.json` (31 dataItems, 31 unique utterances)
- `base-sessions/completeness_hints-latency_calibration.json` (256 dataItems, 256 unique utterances)
- `base-sessions/evaluate_language_consistency.json` (86 dataItems, 86 unique utterances)
- `base-sessions/meeting_intent_parameter.json` (1265 dataItems, 1265 unique utterances)
- `base-sessions/meeting_prep_recap.json` (504 dataItems, 504 unique utterances)
- `base-sessions/people-direct-reports.json` (113 dataItems, 113 unique utterances)
- `base-sessions/people-tool-triggering-and-accuracy.json` (207 dataItems, 207 unique utterances)
- `base-sessions/retain_i_me_my_in_query.json` (197 dataItems, 197 unique utterances)
- `base-sessions/rsvp-status-in-meetings-queries.json` (218 dataItems, 218 unique utterances)
- `base-sessions/triggering-chat.json` (154 dataItems, 154 unique utterances)
- `base-sessions/triggering-code_interpreter.json` (183 dataItems, 183 unique utterances)
- `base-sessions/triggering-email-shared-mailbox.json` (40 dataItems, 40 unique utterances)
- `base-sessions/triggering-email.json` (172 dataItems, 172 unique utterances)
- `base-sessions/triggering-graphic_art.json` (140 dataItems, 140 unique utterances)
- `base-sessions/triggering-memory.json` (151 dataItems, 151 unique utterances)
- `base-sessions/triggering-scheduling_handoff_first_hop.json` (147 dataItems, 147 unique utterances)
- `base-sessions/triggering-web_work.json` (255 dataItems, 255 unique utterances)
- `base-sessions/triggering-workplaceharm.json` (58 dataItems, 58 unique utterances)
- `fetch-email/prompt_triggering-fetch_email.json` (140 dataItems, 140 unique utterances)
- `update-canvas/canvas_edit_long.json` (57 dataItems, 57 unique utterances)
- `update-canvas/canvas_edit_short.json` (36 dataItems, 36 unique utterances)
- `update-canvas/canvas_edit_web.json` (25 dataItems, 25 unique utterances)
- `update-canvas/prompt_canvas_update.json` (46 dataItems, 46 unique utterances)

