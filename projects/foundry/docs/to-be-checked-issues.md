# To-be-checked issues uncovered by the May-2026 eval-script bug fixes

After fixing four silent-crash bugs in non-LLM eval scripts (commit on
`ReasoningPythonScript`, May 2026), several criteria gained ~5–10 % more
scored rows. Those newly visible rows are not noise — they correspond to a
specific class of **model output anomaly** that the previous scripts hid by
crashing. The owning teams should review whether the model behavior is
expected.

## Run comparison

- **Run 1 (before fix):** `local/Results/172715_rebuild_sessions/packed_sessions/20260509-013237-results`
- **Run 2 (after fix):** `local/Results/172715_rebuild_sessions/packed_sessions_fixed/20260509-210540-results`
- **Side-by-side report:** `local/Results/172715_rebuild_sessions/old_vs_new_with_llm.md`

## Top denominator-corrected regressions

All in `sessions/base-sessions/meeting_intent_parameter.json`. The score drop
is the *same model output* being judged against the *same criterion* — the
denominator simply grew because Foundry no longer silently dropped the
previously-crashing rows.

| Criterion          | Prompt    | Rows: R1 → R2 | Score: R1 → R2 | Δscore |
| ------------------ | --------- | ------------- | -------------- | ------ |
| `Accuracy`         | control   | 580 → 622 (+42) | 75.34 → 70.10 | −5.24 |
| `Accuracy`         | treatment | 583 → 617 (+34) | 74.10 → 70.34 | −3.76 |
| `prep_recall`      | control   | 289 → 306 (+17) | 65.74 → 60.78 | −4.96 |
| `prep_recall`      | treatment | 292 → 307 (+15) | 64.04 → 60.91 | −3.13 |
| `calendar_recall`  | control   | 124 → 143 (+19) | 69.35 → 60.14 | −9.21 |
| `calendar_recall`  | treatment | 126 → 144 (+18) | 65.87 → 58.33 | −7.54 |

## Issue: model omits `domain_filters` on `meetings` queries

> ### Why this is not a logical inconsistency with "non-LLM criterion"
>
> A reasonable concern: if Foundry decides "this criterion uses an LLM" by
> checking whether the criterion's judge prompt is empty, and we labeled this
> criterion as **non-LLM**, then how can the script depend on something an
> LLM produced?
>
> Resolution — the evaluation pipeline has **three stages**, and only one of
> them is gated by the criterion's prompt field:
>
> 1. **Seval scraping (run once, offline, prior to any Foundry run).** Sydney
>    is invoked on each utterance; its tool-call decisions and DeepLeo
>    telemetry are captured and packed into the session JSON. The model under
>    test (M) runs here. Its output is cached as `dataItems[*].input` /
>    `runEvalResults[*].inference[*].dataItemsOutputs`. **Foundry does not
>    re-run M.**
> 2. **Foundry `parsingScript`.** Reads the cached scrape via `{{text}}` and
>    normalizes its shape. No LLM call.
> 3. **Foundry criterion (`script` and optionally a judge LLM J).** If the
>    criterion's `prompt.messages` is non-empty, J is invoked and asked to
>    score the row; otherwise scoring relies entirely on the deterministic
>    Python `script`.
>
> "LLM criterion" / "non-LLM criterion" refers exclusively to whether **J**
> (stage 3) is invoked. It says nothing about M (stage 1) — the script
> always reads M's cached output, because M's output is the artefact under
> evaluation.
>
> Why both runs have byte-identical inference: both Run 1 and Run 2 reuse
> the **same Seval job results** (stage 1 is frozen). There is no
> probabilistic re-inference between runs — the inference bytes are
> identical by construction, not by LLM determinism.
>
> Verified for `meeting_intent_parameter`:
>
> - All 9 criteria have `prompt.prompt == ""` and `prompt.messages == []`.
>   No judge LLM is invoked at stage 3. ⇒ correctly classified as **non-LLM**.
> - The `parsingScript` header explicitly states it "*normalizes inference
>   output before evaluation*" — i.e., `{{text}}` is the cached Seval scrape
>   of M, not any judge's response.
>
> So the `domain_filters` field discussed below is something **M emitted
> during the original Seval scrape**. The non-LLM script is just inspecting
> it. No judge LLM (J) is involved at any point.

**Root cause analysis (Accuracy, both prompts):** of the 83 newly-counted
rows, **76 (92 %)** trace to the same model behavior — the model issues an
`office365.search` / `office365_search` tool call whose `queries[*]` entry
has `domain == "meetings"` but **no `domain_filters` key at all**:

```json
{"domain": "meetings", "response_length": "medium", "query": "Hiring Plan"}
```

### Concrete example: item 15

The two-stage data flow makes the omission unambiguous.

**Test input (`dataItems[15].input`)** — the utterance + ground-truth
labels we feed into Seval scraping. No `domain_filters` lives here:

```json
{"utterance": "Is there a pre-read for the meeting Hiring Plan?",
 "segment": "prep",
 "better_model_is_required": "false",
 "better_RAG_is_required": "false"}
```

**Cached Seval scrape (`runEvalResults[*].inference[*].dataItemsOutputs["15"]`)**
— Sydney's emitted tool call for that utterance. This is where
`domain_filters` *would* live if the model had declared it; it doesn't:

```json
{
  "function": {
    "name": "office365_search",
    "arguments": "{\"queries\":[{\"domain\":\"meetings\",
                                  \"response_length\":\"medium\",
                                  \"query\":\"Hiring Plan\"}]}"
  },
  "id": "call_CjY...",
  "type": "function"
}
```

A correctly-formed meetings query for a `prep`-segment utterance would
have looked like:

```json
{"domain": "meetings",
 "domain_filters": {"meeting_intent": "prep"},
 "query": "Hiring Plan",
 "response_length": "medium"}
```

So `domain_filters` is purely a model-output field. Both runs reuse the
same cached scrape, so its absence is identical across runs by
construction; only the script's reaction to that absence changed.

Per the meetings tool contract the model is expected to also declare
`domain_filters.meeting_intent` (one of `prep`, `recap`, `prep_and_recap`).
Without it the script (after the fix) defaults to `'prep_and_recap'`, which
is correct only for the `all`-segment cases. For segments `prep`, `recap`,
or `generic`, the row scores 0.

### Distribution of the 76 affected rows by ground-truth segment

| Segment   | Count |
| --------- | ----- |
| `prep`    |    37 |
| `generic` |    38 |
| `recap`   |     1 |

### Sample utterances to review with the domain owner

**Segment = `prep`** (model should set `meeting_intent: 'prep'`):

- item 15:  *Is there a pre-read for the meeting Hiring Plan?*
- item 20:  *Is there a pre-read for the meeting Technical Deep Dive?*
- item 34:  *Is there a pre-read for the meeting OKR Review?*
- item 41:  *Is there a pre-read for the meeting Procurement Meeting?*
- item 63:  *Is there a pre-read for the meeting Brand Strategy?*

**Segment = `generic`** (model should set `meeting_intent: 'calendar'`):

- item 1110: *What time does my next meeting with Alex start?*
- item 1115: *Which meeting did I have with Lisa on June 6?*
- item 1122: *Who were the invitees in my marketing meeting last week?*
- item 1135: *Do I have more than three meetings scheduled next week?*
- item 1138: *Show me all my meetings for this week.*

**Segment = `recap`** (model should set `meeting_intent: 'recap'`):

- item 878: *At the Quality Control session, what defects did Kimberly highlight?*

The full list is reproducible from the run-2 results by filtering the
patched-script `scriptOutput` for the literal line `{}` (the post-fix script
prints the empty fallback when `domain_filters` is absent) and joining
against `dataItems`.

### Questions for the domain owner

1. **Is the model expected to populate `domain_filters.meeting_intent`?**
   If so, the omission seen on the 76 utterances above is a real regression
   in the model / prompt and the utterances above are a starting set for
   repro. If not, the criterion's scoring contract should be updated.

2. **What does the meetings retriever actually do when
   `domain_filters.meeting_intent` is absent — and is the script's fallback
   `'prep_and_recap'` the right interpretation?**

   The script's `segment_to_output` mapping treats the four
   `meeting_intent` values as mutually exclusive *categories* of search:

   ```python
   segment_to_output = {
       "generic": "calendar",         # enumerate meetings, no prep/recap
       "prep":    "prep",             # search prep material
       "recap":   "recap",            # search recap material
       "all":     "prep_and_recap",   # search BOTH prep and recap material
   }
   ```

   So `'prep_and_recap'` is **a specific intent** (search prep+recap
   content), not a "no filter / search everything" sentinel. There are
   three plausible interpretations of "model omitted `meeting_intent`",
   and the right script behavior depends on which one matches the actual
   tool contract:

   | # | Interpretation | Effect on score |
   | - | -------------- | --------------- |
   | A | Default = `prep_and_recap` (current script convention; our fix preserves this) | `all` segments → 100; `prep` / `recap` / `generic` → 0 |
   | B | No filter at all = the meetings tool searches *every* category, so any segment counts as a match | All four segments → 100 |
   | C | Model failed to specify intent → row is unscorable | Row is dropped (`None`) |

   The existing script chose **A** (without comment) for the case
   "*`domain_filters` exists but `meeting_intent` is missing*". Our recent
   fix extends **A** to "*`domain_filters` is entirely absent*" for
   consistency with the existing convention. But this inherits whatever
   incorrectness already lived in **A**. If the meetings retriever
   actually behaves like **B**, we're under-crediting the model on
   `prep` / `recap` / `generic` rows. If it behaves like **C**, the row
   should be dropped.

   This question affects ~76 rows in `Accuracy` (and proportional shares
   of the recall / precision criteria). Confirming the retriever's actual
   behavior with the meetings-domain owner is required before we can be
   confident the fallback is correct.

## Caveat: Foundry script-execution noise (~±1–4 rows)

> **Status:** The reasoning below remains valid even with the corrected
> understanding that "inference" is the cached Seval scrape rather than a
> live Foundry-side LLM call. The argument never depended on inference
> being LLM-deterministic; it depended on inference being **identical
> across the two runs**, which is automatic once both runs reuse the same
> Seval job. So the conclusion — that Foundry's script execution
> introduces a ±1–4 row noise floor independently of any model
> non-determinism — still stands and should also be confirmed with the
> Foundry team.

When comparing two Foundry runs of the **same** sessions you should expect
roughly **±1–4 rows of drift per criterion per prompt** even on
deterministic, non-LLM scripts. We confirmed this empirically across the
two runs above:

- Both runs reuse the **same Seval scrape** (stage 1 is frozen), so the
  per-row inference passed to the eval script is byte-identical by
  construction.
- 132 non-LLM `(session, prompt, criterion)` tuples nevertheless had
  different sets of scored `dataItem` indices between Run 1 and Run 2 —
  485 differing items in total.
- For **100 % of those 485 differing items**, the inference text was
  byte-identical between the two runs **and** the criterion `script`
  bytes were identical (excluding the 7 files we patched in the May-2026
  fix). Identical script + identical input → identical output is what we
  expected; we did not get it.
- The drift was bidirectional: 51 tuples gained items in Run 2, 26 lost
  items, 55 swapped (some gained, some lost). A deterministic code change
  cannot move probability in both directions, so this drift cannot be
  caused by our bug fixes.

Most likely cause: transient script-runner failures on Foundry's evaluation
side (sandbox timeout, container restart, or storage fetch) that
occasionally yield an empty `scriptOutput` for a row, causing it to be
dropped from the denominator. The drift is small (≤1 % per row count) but
real. **When interpreting cross-run diffs, treat any single-criterion row
delta of ≤4 as within the noise floor.**

The large `meeting_intent_parameter` row gains documented above (+15 to
+42) are well outside this floor and are the genuine effect of the fix.
