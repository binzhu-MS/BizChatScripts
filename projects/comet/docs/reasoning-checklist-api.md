# CoMet Reasoning Checklist — API & Iteration Handling

This note documents the public input contract for CoMet's BizChat Reasoning
Checklist metric and how the metric internally selects which reasoning
iteration to score, so it stays aligned with Foundry's reference behavior.

## Public API

**Metric:** `BizChatReasoningChecklistMetric`
([`metric_logic.py`](../../cometdefinition/metrics/bizchat_reasoning_checklist/logic/metric_logic.py))

**Input proto:** `ReasoningChecklistInput`
([`bizchat_reasoning_checklist.proto`](../../cometdefinition/metrics/bizchat_reasoning_checklist/bizchat_reasoning_checklist.proto))

```
ReasoningChecklistInput
├── signal: ReasoningChecklistSignal
│     └── evaluation_data: EvaluationData          ← raw Seval scrape data
└── eval_config: ReasoningChecklistEvalConfig
      └── sessions: repeated SessionEntry
            ├── session_name        e.g. "triggering-email"
            ├── criteria_list       e.g. ["emails_recall", "emails_precision"]
            └── custom_dimensions   map<string,string>
                                    well-known keys: "label", "segment"
```

**Output proto:** `ReasoningChecklistOutput`

```
ReasoningChecklistOutput
└── result: ReasoningChecklistResult
      └── scores: ReasoningChecklistScores
            └── session_results: repeated SessionResult
                  ├── session_name
                  └── criteria_results: repeated CriterionResult
                        ├── criteria_name
                        ├── score      // 0..100, or -1 (SKIPPED)
                        └── message    // diagnostic
```

## What the caller passes in

The proto comment on `ReasoningChecklistInput.signal` is the contract:

> *EvaluationData from the SEVAL scraper – must contain tool invocations in
> `turnData[-1].orchestrationIterations[*].modelActions[*].toolInvocations`.*

That is, the caller passes the **full `EvaluationData` exactly as captured
from the Seval scrape**:

- All turns may be present (CoMet only inspects `turnData[-1]`).
- The last turn carries **all** of its reasoning iterations (L1, L2, L3, …)
  inside `orchestrationIterations`. Each iteration carries its
  `modelActions[]`, `toolInvocations[]`, and any `nestedOrchestrations[]`.
- The caller does **not** pre-trim to a single iteration. Iteration
  selection is CoMet's responsibility.

This contract is stable. The fix described below is purely internal — no
proto change, no behavior change for callers.

## Internal iteration selection (Foundry-aligned)

The Reasoning Checklist criteria scripts were originally calibrated inside
Foundry against the **first hop (L1) only** of the reasoning loop. The
Foundry pre-eval script
(`C:\working\Sydney\services\TuringBot\tools\ReasoningPythonScript\foundry_preeval_script.py`)
makes this explicit:

```python
# Only extract the first hop (L1).  Skip everything else,
# including entries without an L-tag (loop == -1).
loop = _get_loop_number(output_str)
if loop != 1:
    continue
...
# Break after the first successful L1 extraction
break
```

Within that single L1 entry, Foundry includes **all** `toolInvocations`.

### CoMet `_extract_tool_calls` — how it maps

In `metric_logic.py`, `_extract_tool_calls(eval_data)` is the only place
the EvaluationData proto is consumed. It now mirrors Foundry exactly:

| Foundry (telemetry.metrics)                              | CoMet (EvaluationData)                                  |
|----------------------------------------------------------|---------------------------------------------------------|
| `DeepLeoImprovedNetworking` entries with `L1`            | `turnData[-1].orchestrationIterations[0]`               |
| All `toolInvocations` inside that L1 payload             | All `modelActions[].toolInvocations[]` in that iteration|
| (no concept of nested orchestrations in telemetry)       | All `nestedOrchestrations[...]` recursively within L1   |
| Hops L2, L3, ... ignored                                 | `orchestrationIterations[1:]` ignored                   |

The wrapped output handed to each criterion script is unchanged:

```json
{"role": "assistant", "content": null, "tool_calls": [ ... ]}
```

### Rationale

- **Calibration parity** — the per-criterion thresholds and rule sets were
  authored against L1-only inputs in Foundry. Concatenating later hops
  (L2+) silently inflates `*_recall` and shifts `*_precision`,
  `query_correctness`, `time_qualifier_check`, `tool_tense_aligned*`, etc.,
  away from the Foundry rubric.
- **Comparability** — local CoMet runs and Foundry runs of the same
  Seval scrape now produce the same per-criterion scores (modulo LLM
  judge nondeterminism).

## Caller checklist

When invoking the metric with Seval data:

1. Pull the `EvaluationData` payload straight from the scrape
   (e.g. the `EvaluationData` message inside
   `requests[0].response_body.messages` of a Sydney scrape file).
2. Pass it through unchanged as `signal.evaluation_data`.
3. Provide one `SessionEntry` per session/criteria-list combination in
   `eval_config.sessions`. `custom_dimensions` carries `label`, `segment`,
   and any other per-row context the criteria need.
4. Do **not** filter, slice, or rebuild `turnData` /
   `orchestrationIterations` upstream. CoMet handles iteration selection.

## Related code

- `_extract_tool_calls` —
  [`metric_logic.py`](../../cometdefinition/metrics/bizchat_reasoning_checklist/logic/metric_logic.py)
- Proto contract —
  [`bizchat_reasoning_checklist.proto`](../../cometdefinition/metrics/bizchat_reasoning_checklist/bizchat_reasoning_checklist.proto)
- Foundry reference behavior — `foundry_preeval_script.py` (Sydney repo)
- Foundry session-criteria scoring overview —
  `C:\working\Foundry\local\docs\session-criteria-scoring.md`
