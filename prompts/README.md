# MyLLM Prompts Directory

This directory contains all prompts organized by project/function names.

## Quick Reference

For comprehensive documentation, see [`docs/llm_usage_and_auth_guide.md`](../docs/llm_usage_and_auth_guide.md).

## Directory Structure

```
prompts/
├── your_project_name/         # Each project gets its own folder
│   ├── v0.1.0.md             # Version-specific prompt files
│   ├── v0.2.0.md             # Multiple versions supported
│   └── README.md             # Project-specific documentation (optional)
├── text_summarizer/          # Example: Text summarization prompts
├── simple_scorer/            # Example: Basic scoring prompts
├── utterance_complexity_classifier/  # Example: Classification prompts
└── TEMPLATE.md               # Quick start template
```

## Quick Usage

```python
from llms import prompts

# Load any prompt by project and version
prompt = prompts.get("text_summarizer", "0.1.0")
prompt = prompts.get("simple_scorer", "0.1.0") 
prompt = prompts.get("your_project_name", "0.1.0")
```

## Adding New Prompts (Quick Steps)

1. **Create project folder**: `prompts/your_project_name/`
2. **Create prompt file**: `prompts/your_project_name/v0.1.0.md`
3. **Use in your code**:
   ```python
   DEFAULT_PROMPT = prompts.get("your_project_name", "0.1.0")
   ```

See [`TEMPLATE.md`](TEMPLATE.md) for a step-by-step guide.

## Prompt Format

Prompts use RSP-compatible markdown format:

```markdown
<|im_start|>system
Your system instructions here...
<|im_end|>
<|im_start|>user
Your user prompt with variables: {{{variable_name}}}
<|im_end|>
```

**Variables**: Use triple braces `{{{variable_name}}}` for substitution.

## Versioning

- Use semantic versioning: `v0.1.0`, `v0.2.0`, etc.
- Create new version files for prompt updates  
- Keep old versions for backward compatibility

For detailed information on prompt engineering, variable substitution, and best practices, see the [comprehensive guide](../docs/llm_usage_and_auth_guide.md#prompt-management).
