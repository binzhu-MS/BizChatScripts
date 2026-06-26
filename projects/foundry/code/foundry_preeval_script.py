# Foundry Pre-Eval Script
#
# Normalizes inference output before evaluation in Copilot Playground (Foundry).
# Handles two input formats:
#   - Sydney/Seval: extracts tool calls from telemetry.metrics (DeepLeoImprovedNetworking)
#     and converts them to the Foundry output format.
#   - Foundry native: passes through unchanged.
#
# Detection: if the input JSON contains "telemetry.metrics", it is treated as
# Sydney format; otherwise it is assumed to be Foundry-native.
#
# Output format (both paths):
#   {"role":"assistant","content":null,"tool_calls":[...],"Usage":{...}}
#
# For Sydney/Seval input, the per-request user profile resolved by
# ContextService (manager, skip manager, job title, etc.) is preserved under
# an extra "profile" key when present, so downstream scoring scripts can use
# per-row personal-profile values instead of a hard-coded fixture:
#   {"role":"assistant",...,"profile":{"Name":"...","Manager":"...",...}}
#
# Usage: paste the contents of this file into Foundry's Pre-Eval Python Script
# editor. The {{text}} placeholder is replaced by Foundry at runtime with the
# raw inference output.

import json

# {{text}} is replaced by Foundry with the inference output at runtime
JSON_INPUT = """{{text}}"""


def _parse_deep_leo_output(output_str):
    """Parse the JSON payload that follows the ``CallTags: ...`` prefix.

    Scans forward from the first ``{`` trying each position until
    ``json.loads`` succeeds.  Returns ``None`` if no valid JSON is found.
    Avoids two pitfalls:
      - A greedy first-'{' match that captures prefix garbage when
        CallTags itself contains '{'.
      - An ``rfind('{')`` match that lands inside a nested object within
        the JSON payload (e.g. toolInvocations strings).
    """
    json_start = output_str.find('{')
    while json_start != -1:
        try:
            return json.loads(output_str[json_start:])
        except json.JSONDecodeError:
            json_start = output_str.find('{', json_start + 1)
    return None


def _classify_deep_leo(output_str):
    """Classify a DeepLeoImprovedNetworking metric by its CallTags prefix.

    Returns:
      - ``"outer_invoking"``  : the root reasoning LLM picking a (router) tool.
        CallTags contains ``fluxv3:invokingfunction``.
      - ``"outer_responding"``: the root reasoning LLM emitting the final
        user-facing response.  CallTags contains ``fluxv3:responding``.
        This is a boundary marker that ends the current outer turn.
      - ``"nested"``          : any other DeepLeoImprovedNetworking call
        whose CallTags identify a sub-orchestrator (e.g. ``searchagent``,
        ``webagent``, ``workberry``).  These carry the concrete domain-tool
        invocations made *inside* an outer router call.
    """
    if "fluxv3:invokingfunction" in output_str:
        return "outer_invoking"
    if "fluxv3:responding" in output_str:
        return "outer_responding"
    return "nested"


def _convert_tool_invocations(tool_invocations, sink):
    """Append normalized tool-call dicts parsed from a ``toolInvocations`` list."""
    for inv_str in tool_invocations:
        try:
            inv = json.loads(inv_str) if isinstance(inv_str, str) else inv_str
            if isinstance(inv, dict) and "function" in inv:
                func = inv["function"]
                sink.append({
                    "id": inv.get("id", ""),
                    "type": inv.get("type", "function"),
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "{}"),
                    },
                })
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            sink.append({"raw": inv_str})


def extract_tool_calls_from_deep_leo(metrics):
    """
    Extract tool calls and token usage from DeepLeoImprovedNetworking metrics.

    The reasoning LLM's tool call decisions are in:
    - serviceName == "DeepLeoImprovedNetworking"
    - output JSON contains "toolInvocations" array
    - CallTags contains "fluxv3:invokingfunction" for the **outer** reasoning
      phase.  Sub-orchestrators (SearchAgent, WebAgent, Workberry, ...)
      emit additional ``DeepLeoImprovedNetworking`` entries with their own
      CallTags prefix (e.g. ``searchagent,...``) whose ``toolInvocations``
      contain the concrete domain-tool calls (e.g. ``office365_search``)
      that the outer router tool (e.g. ``search_office365``) delegated to.

    Concatenation policy: the input ``metrics`` list represents the
    telemetry of a *single* Sydney request (Seval scrapes one request per
    row → the last conversation turn).  All ``DeepLeoImprovedNetworking``
    entries in that input therefore belong to the same turn, just split
    across one or more orchestration iterations (workberry's L1, L2, L3,
    ... outer hops, plus their nested sub-orchestrators).  This function
    concatenates ``toolInvocations`` across every outer-invoking iteration
    *and* every nested sub-orchestrator entry it encounters, in source
    order.  An ``outer_responding`` metric is a turn-boundary marker;
    Seval scrapes shouldn't contain a later turn's metrics, but if one
    appears we stop there defensively.

    Token usage is read from the **first** outer ``fluxv3:invokingfunction``
    payload (the canonical "first hop" tokens, matching the original
    pre-workberry behavior).

    Responding-only turns: when the reasoning LLM answers the user directly
    **without invoking any function**, the turn emits a single
    ``fluxv3:responding`` metric and *no* ``fluxv3:invokingfunction`` metric.
    Such a turn still ran a full inference (its ``promptTokenCount`` is the
    real prompt size), so we capture token usage from the ``responding``
    payload as a fallback when no invoking metric was seen.  This keeps
    ``prompt_tokens`` a faithful "did inference run?" signal: a legitimate
    no-tool answer (a real recall miss) reports ``prompt_tokens > 0`` and is
    preserved by downstream ``prompt_tokens == 0`` failed-inference guards,
    while a row where inference never ran (no metric at all) still reports
    ``prompt_tokens == 0`` and is correctly dropped.

    Returns ``(tool_calls, usage_dict)``.
    """
    tool_calls = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if not isinstance(metrics, list):
        return tool_calls, usage

    first_outer_seen = False
    usage_captured = False

    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        if metric.get("serviceName", "") != "DeepLeoImprovedNetworking":
            continue
        output_str = metric.get("output", "")
        if not output_str:
            continue

        kind = _classify_deep_leo(output_str)

        # Defensive turn boundary: a later-turn ``outer_responding`` should
        # not appear in a single-request Seval scrape, but if it does, stop
        # before we bleed into a later turn's invocations.
        if kind == "outer_responding" and first_outer_seen:
            break

        output_json = _parse_deep_leo_output(output_str)
        if output_json is None:
            continue

        if kind == "outer_invoking" and not first_outer_seen:
            # Capture token usage from the first outer hop only.
            prompt_tokens = output_json.get("promptTokenCount", 0)
            completion_tokens = output_json.get("completionTokenCount", 0)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            first_outer_seen = True
            usage_captured = True
        elif kind == "outer_responding" and not usage_captured:
            # Responding-only turn (no invoking metric seen): the model ran
            # and answered without invoking a function.  Capture its real
            # token usage so this legit no-tool answer is not mistaken for a
            # row where inference never ran.  An invoking metric, if one
            # appears later, still takes precedence (branch above).
            prompt_tokens = output_json.get("promptTokenCount", 0)
            completion_tokens = output_json.get("completionTokenCount", 0)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            usage_captured = True

        # Concatenate this metric's tool invocations.  Applies to every
        # outer-invoking iteration (L1, L2, L3, ...) AND every nested
        # sub-orchestrator entry — all belong to the same turn.
        _convert_tool_invocations(
            output_json.get("toolInvocations", []), tool_calls)

    return tool_calls, usage


def extract_user_profile_from_metrics(metrics):
    """
    Extract the per-request user profile resolved by ContextService.

    Sydney records the user's personal profile for the request (e.g.
    ``Name``, ``Manager``, ``SkipManager``, ``JobTitle``, ``Department``)
    as a metric with ``serviceName == "ContextService"`` whose ``output``
    is a JSON-encoded profile object.  The build pipeline preserves this
    metric in the inference output (see ``EXTRA_METRIC_SERVICES_TO_PRESERVE``
    in build_foundry_eval_sessions.py) precisely so it can reach scoring.

    Returns the parsed profile dict, or ``None`` when no ContextService
    metric is present or its output cannot be parsed.
    """
    if not isinstance(metrics, list):
        return None
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        if metric.get("serviceName", "") != "ContextService":
            continue
        output_str = metric.get("output", "")
        if not output_str:
            continue
        try:
            profile = json.loads(output_str)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(profile, dict):
            return profile
    return None


def extract_tool_calls(json_str):
    """
    Extract tool calls from a Sydney/Seval response and return them in Foundry format.

    Parses telemetry.metrics from the JSON string, pulls tool call decisions
    from the first reasoning iteration, and returns them as:
        {"role":"assistant","content":null,"tool_calls":[...],"Usage":{...}}

    When the response carries a ContextService user profile, it is included
    under an additional ``profile`` key so downstream scoring scripts can
    read per-row personal-profile values (e.g. the user's manager name).

    Tool names and arguments are passed through AS-IS; the downstream
    evaluation script is responsible for handling naming variations.
    """
    data = json.loads(json_str)
    
    # Extract from telemetry.metrics
    telemetry = data.get("telemetry", {})
    metrics = telemetry.get("metrics", [])
    
    tool_calls, usage = extract_tool_calls_from_deep_leo(metrics)
    profile = extract_user_profile_from_metrics(metrics)
    
    # Return in Foundry-compatible format
    # This matches the output format when running directly in Foundry
    result = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
        "Usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
    }

    # Only attach the profile when ContextService resolved one, so the
    # output shape is unchanged for responses without a user profile.
    if profile is not None:
        result["profile"] = profile

    return result


if __name__ == "__main__":
    try:
        data = json.loads(JSON_INPUT)

        # Check if this is Sydney format (has telemetry.metrics)
        # If so, extract tool calls and convert to Foundry format
        # Otherwise, assume it's already Foundry format and pass through unchanged

        if "telemetry" in data and "metrics" in data.get("telemetry", {}):
            # Sydney format - extract tool calls and convert to Foundry format
            result = extract_tool_calls(JSON_INPUT)
            print(json.dumps(result))
        else:
            # Assume Foundry format - pass through unchanged
            print(JSON_INPUT.strip())

    except Exception:
        # On any error, fall back to printing the raw input
        print(JSON_INPUT.strip())
