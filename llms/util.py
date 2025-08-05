import json
import re


def safe_json_loads(json_str):
    """
    Safely parse JSON string with fallback for escaped unicode.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed JSON object
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return json.loads(json_str.encode('utf8').decode('unicode_escape'))


def clean_json_response(response_text):
    """
    Extract JSON from response text that may contain markdown code blocks.
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Cleaned JSON string or original text if no JSON blocks found
    """
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text.strip(), re.DOTALL)
    if matches and matches[0].strip():
        return matches[0].strip()
    else:
        return response_text.strip()


def safe_filename(content, char_limit=70):
    """
    Create a safe filename from arbitrary content.
    
    Args:
        content: Input content
        char_limit: Maximum character limit
        
    Returns:
        Safe filename string
    """
    bad_chars = re.compile(r'[<>:"/\\|?*\s]+')
    return bad_chars.sub("_", content[:char_limit])