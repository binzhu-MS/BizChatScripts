# BizChatScripts Prompts Directory

This directory contains all prompts organized by project/function names for the BizChatScripts framework.

## Quick Reference

For comprehensive documentation, see [`docs/llm_usage_and_auth_guide.md`](../docs/llm_usage_and_auth_guide.md).

## Directory Structure

```
prompts/
├── example_simple_scorer/      # Example: Basic text scoring prompts
├── example_text_summarizer/    # Example: Text summarization prompts  
├── example_rsp_scorer/         # Example: RSP-style scoring prompts
├── example_custom_applier/     # Example: Custom applier prompts (sentiment analysis)
├── utterance_complexity_classifier/  # Utterance complexity classification prompts
├── utterance_personalizer/     # Utterance personalization prompts
├── utterance_selector/         # Selection prompts
└── README.md                   # This file
```

## Quick Usage

```python
from llms import prompts

# Load any prompt by project and version
prompt = prompts.get("example_simple_scorer", "0.1.0")
prompt = prompts.get("example_text_summarizer", "0.1.0") 
prompt = prompts.get("example_custom_applier", "0.1.0")
prompt = prompts.get("your_project_name", "0.1.0")
```

## Adding New Prompts (3 Simple Steps)

### Step 1: Create Project Folder
Create a new folder in `prompts/` with your project name:
```bash
mkdir prompts/your_project_name
```

### Step 2: Create Prompt File
Create a versioned markdown file `prompts/your_project_name/v0.1.0.md`:

```markdown
<|im_start|>system
You are an expert assistant. Your task is to [describe the task clearly].

# Guidelines:
- Be specific about the task requirements
- Include any formatting requirements
- Specify the expected output format

# Evaluation Criteria:
1. [Criterion 1]: [Description]
2. [Criterion 2]: [Description]
<|im_end|>
<|im_start|>user
Process the following input: {{{input_variable}}}

[Additional context]: {{{context_variable}}}

Please provide your response in the following format:
[Specify the expected format]
<|im_end|>
```

### Step 3: Use in Your Code
```python
from llms import ChatCompletionLLMApplier, prompts

class YourApplier(ChatCompletionLLMApplier):
    DEFAULT_PROMPT = prompts.get("your_project_name", "0.1.0")
    
    DEFAULT_MODEL_CONFIG = {
        "model": "dev-gpt-41-longco-2025-04-14",
        "temperature": 0.1,
        "max_tokens": 1000,
    }
    
    # ... rest of your implementation
```

## Prompt Format

BizChatScripts uses OpenAI-compatible message format that gets converted automatically:

```markdown
<|im_start|>system
Your system instructions here...
Provide clear task description and guidelines.
<|im_end|>
<|im_start|>user
Your user prompt with variables: {{{variable_name}}}

Input data: {{{input_data}}}
Context: {{{context}}}

Please respond in [specify format].
<|im_end|>
```

**Important Notes:**
- **Variables**: Use triple braces `{{{variable_name}}}` for substitution
- **System Messages**: Define the role, task, and guidelines clearly
- **User Messages**: Include all necessary context and format requirements
- **Output Format**: Always specify the expected response format

## Versioning

- Use semantic versioning: `v0.1.0`, `v0.2.0`, etc.
- Create new version files for prompt updates  
- Keep old versions for backward compatibility
- Test thoroughly when updating prompts

## Example Projects

### Framework Examples
- **`example_simple_scorer`**: Basic text scoring with file I/O
- **`example_text_summarizer`**: Text summarization with length control
- **`example_rsp_scorer`**: Multi-criteria scoring (RSP compatible)
- **`example_custom_applier`**: Sentiment analysis (demonstrates external prompts)

### Business Applications  
- **`utterance_complexity_classifier`**: Text classification applications
- **`utterance_personalizer`**: Personalization applications

## Tips for Writing Effective Prompts

1. **Be Specific**: Clear task descriptions lead to better results
2. **Include Examples**: Show the model what you want when the task is complex
3. **Format Requirements**: Always specify the expected output format
4. **Variable Names**: Use descriptive names like `{{{input_text}}}`, `{{{instructions}}}`
5. **System Prompts**: Set clear context and constraints
6. **Test Iteratively**: Start simple and refine based on results

For detailed information on prompt engineering, variable substitution, and best practices, see the [comprehensive guide](../docs/llm_usage_and_auth_guide.md#prompt-management).
