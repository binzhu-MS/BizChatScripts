import json
import re

message_re = re.compile(r"<\|im_start\|>(?P<role>[^\n\s]+)( name=(?P<n>[^\n\s]+))?\n(?P<content>(.|\n)+?)\n<\|im_end\|>")

def de_mirror(prompt: str):
    """
    De-converts a mirrored prompt into a sequence of messages.
    Based on RSP's mirror.py implementation.
    """
    messages = []
    for match in message_re.finditer(prompt):
        message = {'role': match.group('role')}
        if match.group('n'):
            message['name'] = match.group('n')
        message = {**message, **parse_content(match.group('content'))}
        messages.append(message)
    return {'messages': messages}

def parse_content(content: str):
    """Parse message content, handling special formats."""
    if content.startswith("!raw_json"):
        doc = json.loads(content.removeprefix("!raw_json\n```json\n").removesuffix("```"))
        if "tool_calls" in doc:
            doc['tool_calls'] = [flatten_arguments(fc) for fc in doc['tool_calls']]
        return doc
    else:
        return {'content': content}

def flatten_arguments(function_call_doc):
    """Flatten function call arguments."""
    # Simplified implementation
    return function_call_doc