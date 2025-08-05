"""
Built-in prompt templates for when markdown files don't exist.
"""

SIMPLE_SCORER_PROMPT = {
    "messages": [
        {
            "role": "system",
            "content": "You are an expert evaluator. Rate how well the given text addresses the user's request."
        },
        {
            "role": "user", 
            "content": """Please evaluate the following:

User Request: {{{user_request}}}

Text to Evaluate: {{{text_to_evaluate}}}

Rate how well the text addresses the user's request on a scale of 0-10, where:
- 0 = Completely irrelevant or unhelpful
- 5 = Partially addresses the request
- 10 = Perfectly addresses the request

Provide your response in JSON format:
```json
{
  "score": <number>,
  "reasoning": "<explanation of your score>"
}
```"""
        }
    ]
}

SUMMARIZER_PROMPT = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant that creates concise summaries of text."
        },
        {
            "role": "user",
            "content": """Please summarize the following text in {{{max_length}}} or fewer:

Text: {{{input_text}}}

Provide a clear, concise summary that captures the main points."""
        }
    ]
}

def get_builtin_prompt(prompt_name):
    """Get built-in prompts for examples."""
    if prompt_name == "simple_scorer":
        return SIMPLE_SCORER_PROMPT
    elif prompt_name == "summarizer":
        return SUMMARIZER_PROMPT
    else:
        raise ValueError(f"Unknown prompt: {prompt_name}")