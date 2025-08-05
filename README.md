# MyLLM - A Simple LLM Framework

A structured framework for building applications that leverage Large Language Models (LLMs) with Microsoft's internal LLM services. Provides authentication, prompt management, error handling, and threading support.

## Features

- **Easy LLM API Integration**: Simplified calling with retry mechanisms and error handling
- **MSAL Authentication**: Windows integrated authentication for internal Microsoft LLM models
- **Prompt Management**: Version-controlled prompt system with RSP-compatible markdown format
- **Base Classes**: Robust foundation for building LLM applications with threading support
- **Comprehensive Examples**: Working examples for common use cases
- **Clean Architecture**: Modular design following Python best practices

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Examples
```bash
# Text summarization
python using_llms\text_summarizer.py

# Text scoring
python using_llms\simple_scorer.py

# RSP-style scoring
python using_llms\rsp_style_scorer.py
```

### 3. Authentication
The framework uses Windows MSAL authentication automatically. On first run, you may see a browser window for authentication. Tokens are cached automatically.

## Project Structure

```
MyLLM/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore patterns (excludes .msal_token_cache)
├── docs/                              # Documentation
│   └── llm_usage_and_auth_guide.md   # Complete usage and authentication guide
├── llms/                              # Core framework
│   ├── __init__.py                    # Package initialization 
│   ├── auth.py                        # MSAL authentication provider
│   ├── base_applier.py               # Base classes for LLM applications
│   ├── llm_api.py                    # Core LLM API client
│   ├── util.py                       # Utility functions
│   ├── .msal_token_cache             # Auto-generated token cache (NOT TRACKED)
│   └── prompts/                      # Prompt management system
│       ├── __init__.py               # Prompt package exports
│       ├── general_loader.py         # Universal prompt loader
│       ├── formatting.py            # Template formatting utilities
│       ├── mirror.py                # RSP-style prompt processing
│       └── templates.py             # Built-in prompt templates
├── prompts/                          # Prompt files (markdown format)
│   ├── text_summarizer/             # Text summarization prompts
│   │   └── v0.1.0.md                # Version 0.1.0 prompt
│   ├── simple_scorer/               # Text scoring prompts
│   │   └── v0.1.0.md                # Version 0.1.0 prompt
│   └── sentiment_analyzer/          # Sentiment analysis prompts
│       └── v0.1.0.md                # Version 0.1.0 prompt
├── using_llms/                       # Example applications
│   ├── text_summarizer.py           # Text summarization example
│   ├── simple_scorer.py             # Text scoring example
│   ├── rsp_style_scorer.py          # RSP-compatible scoring
│   ├── custom_applier.py            # Custom applier examples
│   └── utterance_complexity_classifier.py  # Advanced classifier
├── tests/                            # Unit tests
│   └── test_basic.py                # Basic functionality tests
├── tools/                            # Utility scripts
│   ├── conv_data_fr_json_to_tsv.py  # Data conversion tools
│   ├── merge_json_files.py          # JSON processing utilities
│   └── [other utility scripts]      # Various data processing tools
└── utils/                            # Shared utilities
    ├── constants.py                  # Application constants
    ├── markdown_reports.py           # Report generation
    └── types.py                      # Type definitions
```

## Documentation

- **[Complete Usage Guide](docs/llm_usage_and_auth_guide.md)**: Comprehensive guide covering:
  - How to create LLM applications with step-by-step examples
  - Authentication setup and token management
  - Prompt system and best practices
  - Troubleshooting and advanced usage patterns

## Basic Usage Pattern

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
        input_text = item.get('text', '')
        result = self.analyze(input_text)
        if result:
            item['result'] = result
        return item

    @with_retries  
    def analyze(self, input_text):
        """Your custom LLM processing method."""
        variables = {'input_text': input_text}
        formatted_prompt = prompts.formatting.render_messages(self.prompt, variables)
        completion = self.llmapi.chat_completion(self.model_config, formatted_prompt)
        return completion['choices'][0]['message']['content'].strip()
```

## Key Features

### Authentication
- **Windows MSAL Integration**: Seamless authentication with internal Microsoft LLM services
- **Automatic Token Management**: Tokens cached in `llms/.msal_token_cache` (excluded from Git)
- **Cross-platform Support**: Windows (integrated auth), Linux/Mac (device code flow)
- **No API Keys Required**: Uses corporate Microsoft account

### Prompt Management
- **Version Control**: Semantic versioning for prompts (v0.1.0, v0.2.0, etc.)
- **RSP-Compatible Format**: Markdown files with `<|im_start|>/<|im_end|>` tokens
- **Variable Substitution**: `{{{variable}}}` syntax for dynamic content
- **Simple Loading**: `prompts.get("project_name", "version")` pattern

### Application Framework
- **Base Classes**: Robust foundation with error handling and retry logic
- **Threading Support**: Configurable worker threads for parallel processing
- **Progress Tracking**: Built-in progress bars and logging
- **Error Isolation**: Individual failures don't stop batch processing

## Example Applications

- **Text Summarizer**: Configurable length text summarization
- **Simple Scorer**: Multi-criteria text scoring and evaluation
- **RSP-Style Scorer**: Compatible with existing RSP patterns
- **Sentiment Analyzer**: Emotional tone analysis
- **Utterance Classifier**: Complex text classification

## Requirements for Git Tracking

Essential files that should be tracked in Git:
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `llms/` - Core framework code (excluding `.msal_token_cache`)
- `prompts/` - Prompt files and templates
- `using_llms/` - Example applications
- `docs/` - Documentation files
- `tests/` - Unit tests
- `utils/` - Shared utilities
- `.gitignore` - Git ignore patterns

**Note**: The `.msal_token_cache` file contains sensitive authentication tokens and is automatically excluded from Git tracking.