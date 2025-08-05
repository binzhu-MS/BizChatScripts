# Quick Start: Adding New Prompts

Follow these 3 simple steps to add a new prompt to your MyLLM project.

## Step 1: Create Project Folder
Create a new folder in `prompts/` with your project name:
```bash
mkdir prompts/your_project_name
```

## Step 2: Create Prompt File
Create a versioned markdown file `prompts/your_project_name/v0.1.0.md`:

```markdown
<|im_start|>system
Your system instructions here.
Describe the task and guidelines clearly.
<|im_end|>
<|im_start|>user
Your user prompt with variables like {{{input_variable}}}.

Process this data: {{{data_to_process}}}

Please provide your response in the requested format.
<|im_end|>
```

## Step 3: Use in Your Code
```python
from llms import prompts

class YourApplier(ChatCompletionLLMApplier):
    DEFAULT_PROMPT = prompts.get("your_project_name", "0.1.0")
    # ... rest of your class
```

That's it! Simple and clean.

## Tips

- **Variable Names**: Use descriptive names like `{{{input_text}}}`, `{{{instructions}}}`, `{{{data_to_process}}}`
- **System Prompts**: Be specific about the task, format, and any constraints
- **User Prompts**: Include examples if the task is complex
- **Versioning**: Start with v0.1.0 and increment for changes

For detailed documentation and advanced features, see [`../docs/llm_usage_and_auth_guide.md`](../docs/llm_usage_and_auth_guide.md).
