# MyLLM Usage and Authentication Guide

This comprehensive guide explains how to use the MyLLM framework to create LLM-powered applications and how authentication works with internal Microsoft LLM services.

## Overview

The MyLLM framework provides a structured approach to building applications that leverage Large Language Models (LLMs). It uses Microsoft Authentication Library (MSAL) to authenticate with internal Microsoft LLM services and automatically manages access tokens in `MyLLM/llms/.msal_token_cache`.

## Prerequisites

1. **Microsoft Corporate Account**: You must have a valid Microsoft corporate account with access to internal LLM services.

2. **Required Dependencies**: Install the required Python packages:
   ```bash
   pip install msal pymsalruntime
   ```

3. **Windows Environment**: While the framework supports Linux through device code flow, Windows provides the best experience with integrated authentication.

## Creating LLM Applications

### Basic Application Structure

The MyLLM framework follows a structured pattern for creating LLM applications. Here's how to build your own:

#### 1. Create an LLM Applier Class

```python
from llms import ChatCompletionLLMApplier, ApplicationModes, with_retries, prompts

class MyLLMApp(ChatCompletionLLMApplier):
    """Your custom LLM application."""
    
    DEFAULT_PROMPT = prompts.get("my_app", "0.1.0")
    DEFAULT_MODEL_CONFIG = {
        'model': "dev-gpt-41-longco-2025-04-14",
        'temperature': 0.1,
        'max_tokens': 32000
    }
    DEFAULT_THREADS = 2
    DEFAULT_RETRIES = 3
    APPLICATION_MODE = ApplicationModes.PerItem

    def process_item(self, item, i):
        """Process a single item."""
        # Extract input from item
        input_text = item.get('text', '')
        
        # Call your LLM method
        result = self.analyze(input_text)
        
        # Add result to item
        if result:
            item['result'] = result
            
        return item

    @with_retries  
    def analyze(self, input_text):
        """Your custom LLM processing method."""
        variables = {'input_text': input_text}
        
        # Format the prompt with variables
        formatted_prompt = prompts.formatting.render_messages(self.prompt, variables)
        
        # Call the LLM
        completion = self.llmapi.chat_completion(self.model_config, formatted_prompt)
        
        # Extract and return response
        return completion['choices'][0]['message']['content'].strip()
```

#### 2. Create a Prompt File

Create `prompts/my_app/v0.1.0.md`:

```markdown
<|im_start|>system
You are a helpful AI assistant that analyzes text.
<|im_end|>

<|im_start|>user
Please analyze the following text:

{{{input_text}}}

Provide your analysis:
<|im_end|>
```

#### 3. Create Application Logic

```python
def main():
    """Run your application."""
    # Create test data
    test_items = [
        {'text': 'Your input text here'},
        {'text': 'Another input text'}
    ]
    
    # Create and run your applier
    app = MyLLMApp()
    results = list(app.apply(test_items))
    
    # Display results
    for i, result in enumerate(results):
        print(f"Result {i + 1}: {result.get('result', 'No result')}")

if __name__ == "__main__":
    main()
```

### Application Patterns

The framework supports different application patterns:

#### Per-Item Processing
```python
APPLICATION_MODE = ApplicationModes.PerItem
# Processes each item independently - good for classification, scoring, summarization
```

#### Batch Processing
```python
APPLICATION_MODE = ApplicationModes.Batch
# Processes all items together - good for comparative analysis, ranking
```

#### Custom Processing
Override the `apply()` method for complete control over processing logic.

## How Authentication Works

## How Authentication Works

### Token Cache Generation

The `.msal_token_cache` file is automatically generated when you first run any MyLLM application. Here's the process:

1. **Initialization**: When you create an LLM applier (like `TextSummarizer`), the authentication provider is initialized.

2. **Token Acquisition**: The system attempts to get a token using this priority:
   - **Silent Authentication**: If cached tokens exist and are valid
   - **Windows Integrated Auth**: Interactive login using your Windows credentials
   - **Device Code Flow**: Fallback method for Linux or when interactive auth fails

3. **Token Caching**: Successfully acquired tokens are automatically cached to `llms/.msal_token_cache` for future use.

### Authentication Configuration

The authentication uses these settings (matching the RSP project):

- **Client ID**: `99c1a080-d873-4120-ba44-bd8704143c4a`
- **Authority**: `https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47`
- **Scopes**: `['https://substrate.office.com/llmapi/LLMAPI.dev']`
- **Broker**: Enabled on Windows for seamless authentication

## Running LLM Applications

### Quick Start

1. **Navigate to the MyLLM directory**:
   ```powershell
   cd c:\working\MyLLM
   ```

2. **Run an example** (e.g., text summarizer):
   ```powershell
   python using_llms\text_summarizer.py
   ```

3. **First-time authentication**: If no token cache exists, you'll see an authentication prompt:
   - On Windows: A browser window or system dialog will appear
   - On Linux: You'll get a device code to enter at https://microsoft.com/devicelogin

4. **Subsequent runs**: The cached token will be used automatically.

### Development Workflow

1. **Create your prompt**: Add a markdown file in `prompts/your_app/v0.1.0.md`
2. **Create your applier**: Inherit from `ChatCompletionLLMApplier` 
3. **Test your application**: Run it with sample data
4. **Iterate and improve**: Modify prompts and logic as needed

### Example Output

When running `text_summarizer.py`, you should see:

```
INFO:llms.base_applier:Starting TextSummarizer with 2 workers...
Running Text Summarizer Example...
==================================================

Result 1:
Original (523 chars): Artificial Intelligence (AI) has revolutionized numerous industries...
Max Length: 2 sentences
Summary (145 chars): AI has transformed industries through improved efficiency and user experience. However, rapid AI advancement raises important ethical considerations.
--------------------------------------------------
```

### Common Application Types

#### Text Analysis Applications
- **Sentiment Analysis**: Analyze text for emotional tone
- **Classification**: Categorize text into predefined categories
- **Entity Extraction**: Identify people, places, organizations

#### Content Generation Applications
- **Text Summarization**: Create concise summaries of longer texts
- **Content Rewriting**: Rephrase text for different audiences
- **Translation**: Convert text between languages

#### Scoring and Evaluation Applications
- **Quality Scoring**: Rate content quality on various dimensions
- **Compliance Checking**: Verify text meets specific criteria
- **Similarity Analysis**: Compare texts for similarity

## Authentication Flow Details

### Windows Authentication (Recommended)

1. **Token Cache Check**: System checks for existing valid tokens in `.msal_token_cache`
2. **Silent Authentication**: If cached tokens exist and are valid, they're used automatically
3. **Interactive Authentication**: If no valid tokens exist:
   - Windows authentication dialog appears
   - Enter your Microsoft corporate credentials
   - Token is acquired and cached for future use

### Linux Authentication (Device Code Flow)

1. **Device Code Initialization**: System generates a device code
2. **User Action Required**: You'll see a message like:
   ```
   To authenticate, use a web browser to visit https://microsoft.com/devicelogin 
   and enter the code ABC123DEF
   ```
3. **Browser Authentication**: Complete authentication in your browser
4. **Token Cache**: Token is saved for future use

## Model Configuration

The framework is configured to use internal Microsoft models:

```python
DEFAULT_MODEL_CONFIG = {
    'model': "dev-gpt-41-longco-2025-04-14",  # Internal Microsoft model
    'temperature': 0.1,
    'max_tokens': 32000
}
```

Required headers are automatically added:
- `X-ModelType`: Specifies the internal model type
- `X-ScenarioGUID`: Unique identifier for the scenario

## Troubleshooting

### Common Issues

1. **"Missing required dependency: pymsalruntime"**
   ```bash
   pip install pymsalruntime
   ```

2. **Authentication fails on first run**
   - Ensure you're connected to the Microsoft corporate network
   - Verify your corporate account has access to internal LLM services
   - Try deleting `.msal_token_cache` and re-authenticating

3. **"Failed to acquire token" errors**
   - Check your network connection
   - Verify you're using the correct Microsoft account
   - Try the device code flow as an alternative

4. **Cached token expires**
   - The system automatically refreshes expired tokens
   - If refresh fails, delete `.msal_token_cache` to force re-authentication

### Manual Token Cache Reset

If you encounter persistent authentication issues:

```powershell
# Delete the token cache
Remove-Item "c:\working\MyLLM\llms\.msal_token_cache" -Force

# Run your application again to re-authenticate
python using_llms\text_summarizer.py
```

## Security Considerations

1. **Token Cache Security**: The `.msal_token_cache` file contains sensitive authentication data. Keep it secure and don't commit it to version control.

2. **Network Requirements**: Authentication requires access to Microsoft's authentication servers and internal LLM services.

3. **Token Expiration**: Tokens automatically expire and are refreshed. The system handles this transparently.

## File Structure

After successful authentication, your directory structure will include:

```
MyLLM/
├── llms/
│   ├── .msal_token_cache     # Auto-generated token cache (DO NOT COMMIT)
│   ├── auth.py               # Authentication provider
│   └── llm_api.py           # LLM API client
├── using_llms/
│   ├── text_summarizer.py   # Working example
│   └── simple_scorer.py     # Another working example
├── prompts/
│   ├── text_summarizer/
│   │   └── v0.1.0.md        # Prompt for text summarizer
│   └── simple_scorer/
│       └── v0.1.0.md        # Prompt for simple scorer
└── docs/
    └── llm_usage_and_auth_guide.md  # This document
```

## Prompt Management

### Prompt Structure

Prompts use RSP-compatible markdown format with special tokens:

```markdown
<|im_start|>system
You are a helpful AI assistant.
<|im_end|>

<|im_start|>user
Process this text: {{{input_text}}}

Instructions: {{{instructions}}}
<|im_end|>
```

### Loading Prompts

Use the simple prompt loading pattern:

```python
from llms import prompts

# Load a specific prompt version
my_prompt = prompts.get("my_app", "0.1.0")
```

### Adding New Prompts

1. **Create directory**: `prompts/your_app_name/`
2. **Create prompt file**: `v0.1.0.md` (or your version)
3. **Use in applier**: `prompts.get("your_app_name", "0.1.0")`

## Example Applications

The framework includes several working examples that demonstrate different patterns:

### Text Summarizer (`using_llms/text_summarizer.py`)
- **Purpose**: Summarizes text to specified lengths
- **Pattern**: Per-item processing with variable length control
- **Key Features**: Configurable summary length, batch processing support

### Simple Scorer (`using_llms/simple_scorer.py`) 
- **Purpose**: Scores text based on specified criteria
- **Pattern**: Per-item processing with numerical scoring
- **Key Features**: Multi-criteria scoring, structured output

All examples use the same authentication pattern and will automatically generate the token cache on first run.

## Best Practices

### Application Design
1. **Single Responsibility**: Each applier should focus on one specific task
2. **Configurable Parameters**: Make key settings configurable through item properties
3. **Error Handling**: Use `@with_retries` decorator for robust LLM calls
4. **Logging**: Include appropriate logging for debugging and monitoring

### Prompt Engineering
1. **Clear Instructions**: Be explicit about what you want the LLM to do
2. **Examples**: Include examples in prompts when helpful
3. **Variable Naming**: Use clear, descriptive variable names like `{{{input_text}}}`
4. **Version Control**: Use semantic versioning for prompt updates (v0.1.0, v0.2.0, etc.)

### Performance Optimization
1. **Threading**: Use appropriate thread counts for your workload
2. **Batch Size**: Process items in appropriate batch sizes
3. **Token Management**: Monitor token usage for cost optimization
4. **Caching**: Cache results when appropriate to avoid redundant calls

## Advanced Usage

### Custom Authentication
For special cases, you can override authentication:

```python
from llms.auth import SimpleAuth

class MyCustomApp(ChatCompletionLLMApplier):
    def __init__(self, custom_token=None):
        if custom_token:
            self.auth_provider = SimpleAuth(custom_token)
        super().__init__()
```

### Custom Model Configuration
Override model settings per application:

```python
DEFAULT_MODEL_CONFIG = {
    'model': "dev-gpt-41-longco-2025-04-14",
    'temperature': 0.0,  # More deterministic
    'max_tokens': 16000,  # Shorter responses
    'top_p': 0.9,        # Additional parameters
}
```

### Parallel Processing
Configure threading for your workload:

```python
DEFAULT_THREADS = 4  # Increase for I/O bound tasks
DEFAULT_THREADS = 1  # Use single thread for CPU bound tasks
```

## Next Steps

### Getting Started
1. **Run Examples**: Start by running the provided examples to verify authentication works
2. **Study Patterns**: Examine the example code to understand the framework patterns
3. **Create Simple App**: Build a basic application following the established patterns

### Building Your First Application
1. **Define Your Use Case**: Clearly identify what you want the LLM to do
2. **Create Your Prompt**: Write a clear, specific prompt in the RSP format
3. **Implement Your Applier**: Create a class inheriting from `ChatCompletionLLMApplier`
4. **Test and Iterate**: Run with sample data and refine your approach

### Expanding Your Applications
1. **Add Error Handling**: Implement robust error handling and retry logic
2. **Optimize Performance**: Tune threading and batch processing for your workload
3. **Monitor Usage**: Track token consumption and performance metrics
4. **Version Control**: Use semantic versioning for your prompts and applications

### Resources
- **Working Examples**: Study `using_llms/text_summarizer.py` and `using_llms/simple_scorer.py`
- **Prompt Examples**: Look at `prompts/` directory for prompt formatting
- **Authentication**: This document covers all authentication scenarios
- **Framework Code**: Examine `llms/` directory for implementation details

The MyLLM framework provides a solid foundation for building reliable, scalable LLM applications with proper authentication, prompt management, and error handling built in.
