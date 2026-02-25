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
# Usage: paste the contents of this file into Foundry's Pre-Eval Python Script
# editor. The {{text}} placeholder is replaced by Foundry at runtime with the
# raw inference output.

import json

# {{text}} is replaced by Foundry with the inference output at runtime
JSON_INPUT = """{{text}}"""


def extract_tool_calls_from_deep_leo(metrics):
    """
    Extract tool calls and token usage from DeepLeoImprovedNetworking metrics.
    
    The reasoning LLM's tool call decisions are in:
    - serviceName == "DeepLeoImprovedNetworking"
    - output JSON contains "toolInvocations" array
    - CallTags contains "fluxv3:invokingfunction" for the reasoning phase
    
    Token usage is extracted from the same output JSON:
    - completionTokenCount, promptTokenCount, etc.
    
    Returns (tool_calls, usage_dict) from the first iteration only.
    """
    tool_calls = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    first_iteration_found = False
    
    if not isinstance(metrics, list):
        return tool_calls, usage
    
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        
        service_name = metric.get("serviceName", "")
        if service_name != "DeepLeoImprovedNetworking":
            continue
        
        output_str = metric.get("output", "")
        if not output_str:
            continue
        
        # Check if this is the reasoning/tool-invoking phase
        # CallTags is at the start of output, e.g., "CallTags: fluxv3:invokingfunction,..."
        is_invoking_function = "fluxv3:invokingfunction" in output_str
        
        if not is_invoking_function:
            continue
        
        # For first iteration only, stop after first invokingfunction entry
        if first_iteration_found:
            break
        first_iteration_found = True
        
        # Extract the JSON part from output (after the metadata prefix)
        # Format: "CallTags: ..., {...json...}"
        # Scan forward from the first '{' trying each position until
        # json.loads succeeds.  This avoids two pitfalls:
        #   - A greedy first-'{' match that captures prefix garbage
        #     when CallTags itself contains '{'.
        #   - An rfind('{') match that lands inside a nested object
        #     within the JSON payload (e.g. toolInvocations strings).
        output_json = None
        json_start = output_str.find('{')
        while json_start != -1:
            try:
                output_json = json.loads(output_str[json_start:])
                break
            except json.JSONDecodeError:
                json_start = output_str.find('{', json_start + 1)
        
        if output_json is None:
            continue
        
        try:
            
            # Extract token usage from output JSON
            # Same place where we get toolInvocations
            prompt_tokens = output_json.get("promptTokenCount", 0)
            completion_tokens = output_json.get("completionTokenCount", 0)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
            
            # Extract toolInvocations array
            tool_invocations = output_json.get("toolInvocations", [])
            
            for inv_str in tool_invocations:
                try:
                    # Each invocation is a JSON string that needs to be parsed
                    inv = json.loads(inv_str) if isinstance(inv_str, str) else inv_str
                    
                    if isinstance(inv, dict) and "function" in inv:
                        func = inv["function"]
                        
                        # Pass through tool name and arguments AS-IS
                        # The evaluation script handles various tool naming conventions
                        tool_call = {
                            "id": inv.get("id", ""),
                            "type": inv.get("type", "function"),
                            "function": {
                                "name": func.get("name", ""),
                                "arguments": func.get("arguments", "{}")
                            }
                        }
                        
                        tool_calls.append(tool_call)
                except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                    # If parsing fails due to JSON, type, or attribute errors,
                    tool_calls.append({"raw": inv_str})
                    
        except json.JSONDecodeError:
            continue
    
    return tool_calls, usage


def extract_tool_calls(json_str):
    """
    Extract tool calls from a Sydney/Seval response and return them in Foundry format.

    Parses telemetry.metrics from the JSON string, pulls tool call decisions
    from the first reasoning iteration, and returns them as:
        {"role":"assistant","content":null,"tool_calls":[...],"Usage":{...}}

    Tool names and arguments are passed through AS-IS; the downstream
    evaluation script is responsible for handling naming variations.
    """
    data = json.loads(json_str)
    
    # Extract from telemetry.metrics
    telemetry = data.get("telemetry", {})
    metrics = telemetry.get("metrics", [])
    
    tool_calls, usage = extract_tool_calls_from_deep_leo(metrics)
    
    # Return in Foundry-compatible format
    # This matches the output format when running directly in Foundry
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
        "Usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
    }


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

    except json.JSONDecodeError:
        # On JSON decode error, fall back to printing the raw input
        print(JSON_INPUT.strip())
    except Exception as e:
        result = {
            "role": "assistant",
            "content": None,
            "tool_calls": [],
            "Usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "_error": f"{type(e).__name__}: {str(e)}"
        }
        print(json.dumps(result))
