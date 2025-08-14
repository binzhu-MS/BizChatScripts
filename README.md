# BizChatScripts - Microsoft Internal LLM Framework

A robust framework for building applications that leverage Microsoft's internal Large Language Models (LLMs). Designed specifically for Microsoft internal teams with authentication, prompt management, error handling, threading support, and comprehensive data processing tools.

It is based on the RSP project and python code for using Microsoft LLM API library from https://o365exchange.visualstudio.com/O365%20Core/_git/LLMApi?path=%2Fsources%2Fexamples%2FREADME.md&version=GBmaster&_a=preview.

## 📚 Table of Contents

- [🏢 Microsoft Internal Use Only](#-microsoft-internal-use-only)
- [✨ Key Features](#-key-features)  
- [🚀 Quick Start](#-quick-start)
- [⚙️ Client Types and Model Access](#️-client-types-and-model-access)
- [🎯 Model Configuration and Usage](#-model-configuration-and-usage)
- [⚡ Threading and Retry Architecture](#-threading-and-retry-architecture)
- [📁 Project Structure](#-project-structure)
- [📖 Usage Examples](#-usage-examples)
- [🏗️ Building Your Own Applications](#️-building-your-own-applications)
- [� Background: How Authentication Works](#-background-how-authentication-works)

## 🏢 **Microsoft Internal Use Only**

This framework provides access to **Microsoft's internal LLM models** and requires:
- **Microsoft corporate credentials** for automatic authentication
- **Access to Microsoft's internal LLM services** and PyPI repositories  
- **Python 3.9+** environment for framework compatibility

## ✨ **Key Features**

- **🤖 Dual LLM Access Patterns**: Choose between RSP approach (tested models) and Microsoft LLM API Client (broader model access)
- **🎯 Intelligent Model Routing**: Unified client automatically selects optimal approach based on model selection
- **📋 Advanced Prompt Management**: Separates prompts from code and features a version-controlled system with template rendering and variable substitution.
- **⚡ High-Performance Processing**: Multi-threaded batch processing with configurable retry logic and progress tracking
- **🏗️ Production-Ready Architecture**: Modular design with proper error handling and logging
- **🔄 Flexible Configuration**: Support for different model configurations, parameters, and processing modes

## 🚀 **Quick Start**

### 1. **Prerequisites**
- Microsoft corporate account with access to internal LLM models
- Python 3.9+ and conda environment
- Network access to Microsoft's internal services

### 2. **Installation**
```bash
# Clone the repository
git clone <repository-url>
cd BizChatScripts

# Create conda environment
conda create -n llm python=3.9 -y
conda activate llm

# Install dependencies
pip install -r requirements.txt

# Install Microsoft internal LLM client (Microsoft employees only)
pip install llm-api-client --index-url https://o365exchange.pkgs.visualstudio.com/_packaging/O365PythonPackagesV2/pypi/simple/
```

### 3. **Run Examples** 
*Note: Run these commands from the BizChatScripts root directory*

```bash
# Text summarization example (uses examples/example_summarization_input.json)
python -m using_llms.example_text_summarizer

# RSP-style scoring example (uses examples/example_rsp_scoring_input.json)  
python -m using_llms.example_rsp_scorer

# Custom applier example (sentiment analysis & code review)
python -m using_llms.example_custom_applier demo

# Direct LLM API usage example
python -m using_llms.example_direct_api
```

**🚀 VS Code Users**: Press `F5` to run with VS Code's debugger configurations!

## ⚙️ Client Types and Model Access

### 4. **Choosing Your LLM Access Pattern**

The framework provides three client types with different capabilities and model support:

```python
# Option 1: RSP Client (Default - Fixed model & endpoint)
# Best for: Development, testing, specific model workflows
summarizer = TextSummarizer(threads=3, retries=3)

# Option 2: Explicit RSP Client 
# Best for: When you want to ensure RSP usage even with other libraries installed
summarizer = TextSummarizer(
    threads=3, retries=3,
    client_type="rsp"
)

# Option 3: Microsoft LLM API Client (All internal models)
# Best for: Production, accessing latest models, broader model support
summarizer = TextSummarizer(
    threads=3, retries=3,
    client_type="ms_llm_client",
    ms_scenario_guid="your-scenario-guid-here"  # Optional - has default value
)

# Option 4: Unified Client (Smart routing & fallback)
# Best for: Maximum flexibility, automatic model routing, production resilience
summarizer = TextSummarizer(
    threads=3, retries=3,
    client_type="unified", 
    ms_scenario_guid="your-scenario-guid-here"  # Optional - has default for LLM API Client routing
)
```

## **Key Functional Differences:**

| Client Type        | Model Support                                                                    | Endpoint                | Use Case                      |
| ------------------ | -------------------------------------------------------------------------------- | ----------------------- | ----------------------------- |
| **RSP**            | Fixed set (e.g., `dev-gpt-41-longco-2025-04-14`)                                 | Fixed endpoint          | Development, tested workflows |
| **LLM API Client** | All Microsoft internal models (`gpt-4o-2024-05-13`, `dev-gpt-5-reasoning`, etc.) | Auto-selected endpoints | Production, latest models     |
| **Unified**        | Best of both (smart routing)                                                     | Intelligent selection   | Maximum flexibility           |

**When to Use Each:**

- **RSP** (`None` or `"rsp"`): 
  - ✅ Stable, well-tested model workflows
  - ✅ Development and debugging
  - ✅ Existing applications (no changes needed)
  - ❌ Limited to pre-configured models

- **Microsoft LLM API Client** (`"ms_llm_client"`):
  - ✅ Access to all Microsoft internal models  
  - ✅ Automatic endpoint selection
  - ✅ Latest model releases and updates
  - ✅ Production environments
  - ❌ Requires additional package installation

- **Unified** (`"unified"`):
  - ✅ Combines benefits of both approaches
  - ✅ Smart model-based routing (RSP for tested models, LLM API Client for others)
  - ✅ Automatic fallback on rate limiting/throttling
  - ✅ Production-ready resilience
  - ❌ Most complex configuration

**Practical Example - Model Selection:**
```python
# RSP Client: Works with specific pre-configured models
model_config = {"model": "dev-gpt-41-longco-2025-04-14"}  # Must use exact model names

# LLM API Client: Access to all internal models  
model_config = {"model": "gpt-4o-2024-05-13"}    
model_config = {"model": "dev-gpt-5-reasoning"}   
model_config = {"model": "dev-gpt-41-longco-2025-04-14"}  # Also available via LLM API Client

# Unified: Best of both - uses RSP for tested models, LLM API Client for others
model_config = {"model": "dev-gpt-41-longco-2025-04-14"}  # → Routes to RSP
model_config = {"model": "gpt-4o-2024-05-13"}             # → Routes to LLM API Client
```

## **Parameter Handling Details**

### **Model Input and Configuration**
```python
# Model is specified in your processing code, not during client creation:
model_config = {
    "model": "your-chosen-model",
    "temperature": 0.1,
    "max_tokens": 1000
}

# The model_config is used when calling LLM methods:
result = summarizer.llmapi.chat_completion(model_config, messages)
```

### **How Model Selection Works for Each Client Type**

#### **Option 1 & 2: RSP Client** 
- **Model Selection**: Limited to pre-configured models (e.g., `dev-gpt-41-longco-2025-04-14`)
- **Endpoint**: Fixed RSP endpoint 
- **Model Input**: Must use exact pre-configured model names

#### **Option 3: Microsoft LLM API Client**
- **Model Selection**: Access to **all Microsoft internal models** (GPT-4o, GPT-5, etc.)
- **Endpoint**: Automatically selects optimal endpoint for each model
- **Model Input**: Can use any supported model name (e.g., `gpt-4o-2024-05-13`, `dev-gpt-5-reasoning`)
- **Default Values**:
  - `ms_scenario_guid`: `"4d89af25-54b8-414a-807a-0c9186ff7539"` (if not provided)
  - Default model: `"dev-gpt-41-longco-2025-04-14"` (same as RSP for consistency)
  - Available models: `"gpt-4o-2024-05-13"`, `"dev-gpt-5-reasoning"`, and other internal models
  - Model: Specified in `model_config` during processing calls

#### **Option 4: Unified Client** 
- **Model Selection**: **Intelligent routing** based on model name:
  - **RSP Route**: Models in `RSP_PREFERRED_MODELS` (e.g., `dev-gpt-41-longco-2025-04-14`)
  - **LLM API Client Route**: All other models (e.g., `gpt-4o-2024-05-13`, `dev-gpt-5-reasoning`)
- **Endpoint**: Uses appropriate endpoint based on routing decision
- **Model Input**: Accepts any supported model name, routes automatically
- **Default Values**:
  - `ms_scenario_guid`: `"4d89af25-54b8-414a-807a-0c9186ff7539"` (for LLM API Client routes)
  - Default model: `"dev-gpt-41-longco-2025-04-14"` (routes to RSP)
  - Model: Specified in `model_config`, determines routing

**Model Selection Examples:**
```python
# Option 3: LLM API Client - Any internal model works
model_config = {"model": "gpt-4o-2024-05-13"}    
model_config = {"model": "dev-gpt-5-reasoning"}  
model_config = {"model": "dev-gpt-41-longco-2025-04-14"}  # ✅ Default model

# Option 4: Unified - Model determines routing
model_config = {"model": "dev-gpt-41-longco-2025-04-14"}  # → RSP route
model_config = {"model": "gpt-4o-2024-05-13"}             # → LLM API Client route
model_config = {"model": "dev-gpt-5-reasoning"}           # → LLM API Client route
```

**Complete Example with Default Values:**
```python
# Minimal LLM API Client setup (uses defaults)
summarizer = TextSummarizer(client_type="ms_llm_client")

# Process with recommended model
model_config = {
    "model": "gpt-4o-2024-05-13",        
    "max_tokens": 1000
}

# Call LLM
result = summarizer.llmapi.chat_completion(model_config, messages)
```

### **Parameter Usage by Client Type**

| Parameter          | RSP Client                     | LLM API Client                          | Unified Client                                      |
| ------------------ | ------------------------------ | --------------------------------------- | --------------------------------------------------- |
| `threads`          | ✅ Used for parallel processing | ✅ Used for parallel processing          | ✅ Always used                                       |
| `retries`          | ✅ Framework-level retries      | ❌ Ignored (library handles internally)  | ✅ Used per routing decision                         |
| `ms_scenario_guid` | ❌ Ignored (uses RSP auth)      | ✅ Used for authentication (has default) | ⚠️ Used only for LLM API Client routes (has default) |

### **Unified Client Parameter Details**

When using `client_type="unified"`, parameters are applied based on routing decisions:

```python
# Unified client with parameters
summarizer = TextSummarizer(
    client_type="unified",
    threads=5,                    # Always used for parallel processing
    retries=3,                    # Applied differently per route:
    ms_scenario_guid="guid-123"   # Used only for LLM API Client routes
)

# When model routes to RSP:
model_config = {"model": "dev-gpt-41-longco-2025-04-14"}
# → Uses: threads=5, retries=3 (framework level), ignores ms_scenario_guid

# When model routes to LLM API Client:
model_config = {"model": "gpt-4o-2024-05-13"}
# → Uses: threads=5, ms_scenario_guid="guid-123", retries=3 (ignored - library handles retries internally)
```

### **Default Values and Requirements**

```python
# Option 3: Microsoft LLM API Client
TextSummarizer(
    client_type="ms_llm_client",
    ms_scenario_guid="4d89af25-54b8-414a-807a-0c9186ff7539",  # Default GUID (can override)
    threads=3,                         # Default: class DEFAULT_THREADS
    retries=3                          # Default: class DEFAULT_RETRIES (limited effect)
)

# Option 4: Unified Client  
TextSummarizer(
    client_type="unified",
    ms_scenario_guid="4d89af25-54b8-414a-807a-0c9186ff7539",  # Default GUID for LLM API Client routes
    threads=3,                         # Default: class DEFAULT_THREADS  
    retries=3                          # Default: class DEFAULT_RETRIES
)

# Both options work without explicitly providing ms_scenario_guid:
TextSummarizer(client_type="ms_llm_client")  # ✅ Uses default GUID
TextSummarizer(client_type="unified")        # ✅ Uses default GUID for LLM API routes

# You can override the default if needed:
TextSummarizer(
    client_type="ms_llm_client", 
    ms_scenario_guid="your-custom-guid-here"
)
```

**Advanced: Unified Client Smart Routing**
```python
# Create unified client (requires llm-api-client installed)
summarizer = TextSummarizer(
    client_type="unified", 
    ms_scenario_guid="your-guid",
    threads=5, 
    retries=3
)

# The unified client provides intelligent routing per model:

# RSP Route Example:
model_config = {"model": "dev-gpt-41-longco-2025-04-14"}
result = summarizer.llmapi.chat_completion(model_config, messages)
# → Routes to RSP client
# → Uses: threads=5 (parallel processing), retries=3 (framework level)
# → Ignores: ms_scenario_guid (RSP has own authentication)

# LLM API Client Route Example: 
model_config = {"model": "gpt-4o-2024-05-13"}
result = summarizer.llmapi.chat_completion(model_config, messages)  
# → Routes to LLM API Client
# → Uses: threads=5 (parallel processing), ms_scenario_guid (authentication)
# → Limited: retries=3 (LLM API Client has internal retry mechanisms)

# Key behaviors:
# 1. Model determines routing: RSP for tested models, LLM API Client for others
# 2. Parameters applied contextually: ms_scenario_guid ignored for RSP routes
# 3. Automatic fallback: Switches clients on rate limiting/errors
# 4. Fail-fast validation: Clear errors if required packages/parameters missing
```

## **Common Parameter Issues & Solutions**

### **Missing or Custom ms_scenario_guid**
```python
# ✅ Works: Uses default scenario GUID  
summarizer = TextSummarizer(client_type="ms_llm_client")  # Uses default GUID

# ✅ Custom: Override with your own GUID
summarizer = TextSummarizer(
    client_type="ms_llm_client", 
    ms_scenario_guid="your-custom-guid-here"
)

# ✅ Default GUID value: "4d89af25-54b8-414a-807a-0c9186ff7539"
```

### **Unified Client Configuration Options**
```python  
# ✅ Minimal config: Uses default GUID for LLM API Client routes
summarizer = TextSummarizer(client_type="unified")  # Uses default ms_scenario_guid

# All model routes work:
model_config = {"model": "dev-gpt-41-longco-2025-04-14"}  # ✅ RSP route (ignores GUID)
model_config = {"model": "gpt-4o-2024-05-13"}             # ✅ LLM API Client route (uses default GUID)

# ✅ Custom config: Override default GUID if needed
summarizer = TextSummarizer(
    client_type="unified",
    ms_scenario_guid="your-custom-guid-here"
)
```

### **Retry Parameter Expectations**
```python
# Understanding retry behavior:
summarizer = TextSummarizer(retries=5)

# RSP Client: Full framework retry control
# - 5 retry attempts on network/API failures
# - Framework handles all retry logic

# LLM API Client: Framework retries largely ignored
# - Framework retries=5 setting has minimal effect
# - LLM API Client uses internal error_tolerance and backoff mechanisms
# - Retry behavior controlled by library's internal configuration, not framework parameter
```

## 🎯 Model Configuration and Usage

### **Available Models by Client Type**

| Client Type                  | Available Models                                                                     | Best Use Cases                            |
| ---------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------- |
| **RSP**                      | `dev-gpt-41-longco-2025-04-14` (tested)                                              | Stable workflows, development, debugging  |
| **Microsoft LLM API Client** | `gpt-4o-2024-05-13`, `dev-gpt-5-reasoning`, `dev-gpt-41-longco-2025-04-14`, and more | Latest models, production, broader access |
| **Unified**                  | All of the above (automatic routing)                                                 | Maximum flexibility, fallback resilience  |

**📋 Complete Model List**: For the full list of available models via Microsoft LLM API Client, see: https://substrate.microsoft.net/v2/llmApi/modelList

### **Model Selection Examples**

```python
# RSP Client - Limited but stable model set
model_config = {
    "model": "dev-gpt-41-longco-2025-04-14",
    "temperature": 0.1,
    "max_tokens": 1000
}

# Microsoft LLM API Client - Broader model access
model_config = {
    "model": "gpt-4o-2024-05-13",     
    "temperature": 0.7,
    "max_tokens": 2000
}

model_config = {
    "model": "dev-gpt-5-reasoning",     
    "temperature": 1,
    "max_tokens": 100000
}

# Unified Client - Automatic routing based on model
# These automatically route to the appropriate client:
model_config = {"model": "dev-gpt-41-longco-2025-04-14"}  # → RSP
model_config = {"model": "gpt-4o-2024-05-13"}             # → Microsoft LLM API Client
```

### **Common Configuration Patterns**

```python
# Text Summarization (concise, factual)
model_config = {
    "model": "gpt-4o-2024-05-13",
    "temperature": 0.1,      # Low creativity for factual summaries
    "max_tokens": 500
}

# Creative Writing (more varied, expressive)
model_config = {
    "model": "gpt-4o-2024-05-13", 
    "temperature": 0.8,      # Higher creativity for varied outputs
    "max_tokens": 2000
}

# Classification/Scoring (consistent, deterministic)  
model_config = {
    "model": "dev-gpt-41-longco-2025-04-14",
    "temperature": 0.0,      # Maximum consistency
    "max_tokens": 100
}
```

### **Artifacts You'll Notice**

Different client types create different artifacts during usage:

**RSP Client:**
- Creates `.msal_token_cache` file for authentication tokens
- Logs show "RSP" client selection in unified mode
- Direct HTTP control with custom retry logic

**Microsoft LLM API Client:**
- No visible token files (managed internally)
- Logs show "Microsoft LLM API Client" selection in unified mode  
- Library-managed authentication and retry policies

**Unified Client:**
- May create `.msal_token_cache` when using RSP routes
- Logs show automatic routing decisions
- Combines both approaches transparently

## 🔐 Background: How Authentication Works

### **Authentication Mechanisms (Background Information)**

Authentication happens automatically - you don't need to configure it. Here's what occurs behind the scenes:

**RSP Approach:**
- Uses MSAL (Microsoft Authentication Library) with browser-based login on first use
- Stores tokens locally in `.msal_token_cache` file for subsequent runs
- You might see a browser popup the first time, then it's seamless

**Microsoft LLM API Client Approach:**
- Uses library-managed authentication with your corporate credentials
- No visible token files - everything handled internally
- Completely seamless experience with no user interaction

**Unified Approach:**
- Automatically selects appropriate authentication based on model routing
- May use RSP auth for RSP models, library auth for LLM API Client models
- Creates `.msal_token_cache` only when RSP routes are used

**What You'll Experience:**
- First run: Possible browser authentication popup (RSP routes only)
- Subsequent runs: Completely automatic
- No manual token management required
- Framework handles all authentication seamlessly

## ⚡ Threading and Retry Architecture

The framework provides **layered concurrency and error handling** that works consistently across all client types:

### **Framework Level (BizChatScripts)**
- **`threads`**: Controls **parallel processing** of multiple dataset items
  - Example: Process 1000 text summaries using 5 worker threads
  - Works with all client types (RSP, Microsoft LLM API Client, Unified)
  - Each thread handles multiple items independently with progress tracking

- **`retries`**: Framework-level retry logic for **individual API calls**
  - Provides consistent retry behavior across different client types
  - Handles network issues, temporary service unavailability
  - Works as an additional layer on top of client-specific retry mechanisms

### **Client Level (Individual Libraries)**
- **Microsoft LLM API Client**: Has sophisticated internal mechanisms
  - `error_tolerance`: Internal handling of acceptable failed requests
  - `backoff`: Built-in exponential backoff for rate limiting
  - `max_requests_per_minute`: Automatic rate limiting

- **RSP Client**: Has framework-integrated retry mechanisms
  - Direct retry control through framework parameters
  - Network-level retries for connection issues
  ### **Parameter Usage by Client Type**

| Parameter | RSP Client                              | Microsoft LLM API Client                       | Unified Client                                |
| --------- | --------------------------------------- | ---------------------------------------------- | --------------------------------------------- |
| `threads` | ✅ Full control over parallel processing | ✅ Full control over parallel processing        | ✅ Always used for parallel processing         |
| `retries` | ✅ Direct framework retry control        | ❌ Largely ignored (library handles internally) | ✅ Applied based on routing decision           |
| `model`   | Limited to tested models                | All internal Microsoft models                  | Any model (auto-routes to appropriate client) |

### **Why Framework Parameters Matter**

- **Consistent Interface**: Same parameters work across all client types
- **Layered Resilience**: Framework retries + client retries = better reliability  
- **Dataset Processing**: Client libraries don't handle parallel processing of multiple items
- **Progress Tracking**: Framework provides multi-threaded progress bars and logging
- **Model Flexibility**: Unified client allows switching approaches without code changes

## 📁 **Project Structure**

```
BizChatScripts/
├── 📄 README.md                           # Project documentation
├── 📋 requirements.txt                    # Python dependencies (includes Microsoft internal packages)
├── 🚫 .gitignore                         # Git ignore patterns (excludes authentication cache files)
├── 📚 docs/                              # Documentation
├── 🧠 llms/                              # Core LLM framework
│   ├── 📦 __init__.py                    # Package initialization 
│   ├── 🔐 auth.py                        # Legacy authentication provider  
│   ├── 🏗️ base_applier.py               # Base classes with retry logic and threading
│   ├── 🤖 llm_api.py                    # Core LLM API client
│   ├── 🤖 llm_api_unified.py           # Unified LLM API with intelligent routing
│   ├── 🔗 ms_llm_api_client_adapter.py # Microsoft LLM API Client adapter
│   ├── 🛠️ util.py                       # JSON parsing and utility functions
│   └── 📝 prompts/                      # Prompt management system
│       ├── __init__.py                   # Prompt package exports
│       ├── general_loader.py            # Universal prompt loader
│       ├── formatting.py               # Template formatting utilities
│       ├── mirror.py                   # RSP-style prompt processing
│       └── templates.py                # Built-in prompt templates
├── 📝 prompts/                          # Prompt files (markdown format)
│   ├── example_text_summarizer/        # Text summarization prompts
│   └── example_rsp_scorer/             # RSP-style scoring prompts
├── 🎯 using_llms/                       # Production LLM applications
│   ├── example_simple_scorer.py        # **Basic Example**: Simple text scoring with file I/O
│   ├── example_text_summarizer.py      # **Framework Example**: Text summarization using prompts
│   ├── example_rsp_scorer.py           # **RSP Example**: Multi-criteria scoring (RSP-compatible)
│   ├── example_custom_applier.py       # **Custom Example**: Multiple appliers (sentiment + code review)
│   └── example_direct_api.py           # **API Example**: Direct LLM API usage patterns
├── 📊 examples/                         # **Example Input Data**
│   ├── example_summarization_data.json # Sample texts for summarization
│   ├── example_scoring_data.json       # Sample texts for RSP scoring
│   ├── example_api_prompts.json        # Simple prompts for API testing
│   └── README.md                       # Example usage documentation
├── 📊 projects/                         # Real-world project implementations
│   ├── gpt5_select_complex_utterances/ # Complex utterance selection pipeline
│   └── personalization/                # Utterance personalization project
├── 🧪 tests/                           # Unit tests
├── 🔧 tools/                           # Data processing utilities
└── 🛠️ utils/                           # Shared framework utilities
```

## 📖 Usage Examples
*All commands should be run from the BizChatScripts root directory*

### Example 1: Framework-based Application (Recommended)
```bash
# Text summarization with custom parameters
python -m using_llms.example_text_summarizer --threads=3 --retries=5 --max_items=2

# RSP-style scoring example
python -m using_llms.example_rsp_scorer --threads=2 --retries=3
```

### Example 2: RSP-Compatible Scoring  
```bash
# Multi-criteria text scoring with custom settings
python -m using_llms.example_rsp_scorer --threads=2 --retries=3  
python using_llms/example_rsp_scorer.py --demo
```

### Example 3: Direct API Usage
```bash  
# Low-level LLM API usage with custom control
python -m using_llms.example_direct_api
```

**🚀 VS Code Users**: Press `F5` and select an example configuration for guided debugging!

---

## 🏗️ **Building Your Own Applications**

The framework provides three main approaches for building LLM applications:

### **1. Framework Pattern (Recommended)** 

See `example_simple_scorer.py` - Basic text scoring with comprehensive file I/O and client type demonstrations.

See `example_text_summarizer.py` - Uses base classes with built-in threading, error handling, and prompt management.

**Key Parameters:**
- `--threads=N`: Number of parallel workers (1-5 recommended)
- `--retries=N`: API retry attempts (2-5 recommended)  
- `--max_items=N`: Limit processing for testing (-1 for all)

### **2. RSP-Compatible Pattern**
See `example_rsp_scorer.py` - For applications that need to integrate with existing RSP systems and scoring workflows.

### **3. Custom Multi-Applier Pattern**
See `example_custom_applier.py` - Demonstrates multiple LLM appliers in one application (sentiment analysis + code review) with both external and inline prompt management.

### **4. Direct API Pattern**
See `example_direct_api.py` - For maximum control over API interactions and custom processing flows.

### Where to Place Your Files
- **🎯 `using_llms/`**: Your main applications (e.g., `my_classifier.py`)
- **📝 `prompts/your_app/`**: Your prompt templates (e.g., `v0.1.0.md`)
- **📊 `projects/your_project/`**: Complete project implementations
- **🔧 `tools/`**: Data processing and utility scripts
- **🧪 `tests/`**: Unit tests for your applications

---

## 📁 **Git Tracking Guidelines**

Essential files that should be tracked in Git:
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `llms/` - Core framework code
- `prompts/` - Prompt files and templates
- `using_llms/` - Example applications
- `docs/` - Documentation files
- `tests/` - Unit tests
- `utils/` - Shared utilities
- `.gitignore` - Git ignore patterns

**Note**: Authentication is handled automatically by either the Microsoft LLM API Client (preferred) or RSP-style MSAL authentication with local token caching.