"""
Universal Pre-Eval Python Script for Foundry Evaluations

Purpose:
    Normalize inference output for evaluation. Works with BOTH:
    1. Sydney/Seval output - extracts tool calls from telemetry and converts to Foundry format
    2. Foundry native output - passes through unchanged
    
Detection Logic:
    - If input has "telemetry.metrics" -> Sydney format, extract and convert
    - Otherwise -> assume Foundry format, pass through unchanged

Output Format:
    {"role":"assistant","content":null,"tool_calls":[...],"Usage":{...}}
    
This ensures the same evaluation criteria work for both Foundry and Sydney/Seval modes.

Usage: Copy this script into Foundry's Pre-Eval Python Script editor
"""

import json
import re

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
        json_match = re.search(r'\{.*\}$', output_str, re.DOTALL)
        if not json_match:
            continue
        
        try:
            output_json = json.loads(json_match.group())
            
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
                except:
                    # If parsing fails, store raw
                    tool_calls.append({"raw": inv_str})
                    
        except json.JSONDecodeError:
            continue
    
    return tool_calls, usage


def extract_tool_calls(json_str):
    """
    Extract tool calls from Sydney response.
    Returns the reasoning LLM's tool call decisions from the first iteration.
    
    OUTPUT FORMAT: Must match Foundry's native output format:
    {"role":"assistant","content":null,"tool_calls":[...],"Usage":{...}}
    
    This ensures the same evaluation criteria work for both Foundry and Seval.
    
    NOTE: Tool names and arguments are passed through AS-IS from Sydney.
    The evaluation script (meetings_recall_eval.py) already handles various
    tool naming conventions like office365_search, search_office365, 
    search_enterprise_meetings, etc.
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
            
    except json.JSONDecodeError as e:
        # Return empty in Foundry format on error
        result = {
            "role": "assistant",
            "content": None,
            "tool_calls": [],
            "Usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "_error": f"JSON parse error: {str(e)}"
        }
        print(json.dumps(result))
    except Exception as e:
        result = {
            "role": "assistant",
            "content": None,
            "tool_calls": [],
            "Usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "_error": f"{type(e).__name__}: {str(e)}"
        }
        print(json.dumps(result))
