# BizChatScripts - BizChat Copilot Development Tools

A comprehensive collection of Python tools, scripts, and utilities for BizChat Copilot development that are not tracked in the main BizChat Copilot repository. This repository includes LLM framework integration, SEVAL analysis tools, utterance processing pipelines, data conversion utilities, and project-specific implementations.

**Note**: The LLM framework component is based on the RSP project and Microsoft LLM API library from https://o365exchange.visualstudio.com/O365%20Core/_git/LLMApi?path=%2Fsources%2Fexamples%2FREADME.md&version=GBmaster&_a=preview.

## 📚 Table of Contents

### Overview
- [🏢 Microsoft Internal Use Only](#-microsoft-internal-use-only)
- [✨ Key Features](#-key-features)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)

### Major Projects
- [🎯 Complex Utterance Generation](#1--complex-utterance-generation-projectsgpt5_gen_complex_utterances)
- [🔍 Selection of Complex Utterance](#2--selection-of-complex-utterance-projectsgpt5_select_complex_utterances)
- [👤 Utterance Personalization](#3--utterance-personalization-projectspersonalization)
- [📈 SEVAL Analysis Toolkit](#4--seval-analysis-toolkit-projectsseval)

### Tools & Utilities
- [🤖 LLM-Based Utterance Analysis Tools](#-llm-based-utterance-analysis-tools)
- [🛠️ Data Processing Tools](#️-data-processing-tools)
- [🔧 Utility Modules](#-utility-modules)

### LLM Framework Documentation
- [🧠 LLM Framework Documentation](#-llm-framework-documentation)
  - [⚙️ LLM Client Types and Model Access](#️-llm-client-types-and-model-access)
  - [🎯 Model Configuration and Usage](#-model-configuration-and-usage)
  - [🔧 Parameter Configuration Guide](#-parameter-configuration-guide)
  - [⚡ Threading and Retry Architecture](#-threading-and-retry-architecture)
  - [🔐 Authentication Mechanisms](#-authentication-mechanisms)
  - [🏗️ Building LLM Applications](#️-building-llm-applications)

### Additional Resources
- [� Example Scripts](#-example-scripts)
- [📁 Git Tracking Guidelines](#-git-tracking-guidelines)

## 🏢 **Microsoft Internal Use Only**

This repository contains tools and scripts for **BizChat Copilot development** and requires:
- **Microsoft corporate credentials** for authentication and access to internal LLMs
- **Access to Microsoft's internal services** (LLM APIs, SEVAL systems, PyPI repositories)
- **Python 3.9+** environment for compatibility

## ✨ **Key Features**

### **Core Capabilities**
- **🤖 LLM Framework Integration**: Dual access patterns (RSP and Microsoft LLM API Client) with intelligent routing
- **📊 SEVAL Analysis Toolkit**: Comprehensive tools for search evaluation, A/B testing, and metrics analysis
- **🎯 Utterance Processing Pipeline**: Classification, selection, personalization, and complexity analysis
- **🔧 Data Processing Utilities**: Universal file readers, format converters, and data manipulation tools
- **🚀 Project-Specific Tools**: Dedicated implementations for various BizChat Copilot projects

### **LLM Framework Features**
- **🎯 Intelligent Model Routing**: Unified client automatically selects optimal approach based on model selection
- **📋 Advanced Prompt Management**: Version-controlled prompts with template rendering and variable substitution
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
git clone https://github.com/binzhu-MS/BizChatScripts.git
cd BizChatScripts

# Create conda environment
conda create -n llm python=3.9 -y
conda activate llm

# Install dependencies
pip install -r requirements.txt

# Install Microsoft internal LLM client (Microsoft employees only)
pip install llm-api-client --index-url https://o365exchange.pkgs.visualstudio.com/_packaging/O365PythonPackagesV2/pypi/simple/
```

**💡 Important**: All Python scripts in this repository should be run as modules using `python -m module.name` syntax rather than direct file execution. This ensures proper import resolution and PYTHONPATH handling.

### 3. **Quick Start Examples** 
*Note: All commands should be run from the BizChatScripts root directory using Python module syntax (`python -m`) to ensure proper import resolution.*

**LLM Framework Examples:**
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

**Data Processing Examples:**
```bash
# View JSONL files interactively
python -m tools.jsonl_viewer your_file.jsonl

# Merge multiple JSON files
python -m tools.merge_json_files --output_file=merged.json file1.json file2.json file3.json

# Extract results to Excel
python -m tools.extract_results_to_excel --input_file=results.json --output_file=results.xlsx
```

**Major Project Examples:**
```bash
# Analyze complexity classification results
python -m projects.gpt5_select_complex_utterances.complexity_statistics --input_file=data/labeled.json

# SEVAL analysis - extract model statistics
python -m projects.seval.seval_analysis_toolkit extract_model_statistics --input_dir=seval_data --output_file=stats.tsv

# SEVAL metrics analysis - win/loss analysis
python -m projects.seval.seval_metrics_analysis export_win_loss_utterances --metrics_file=metrics.csv --metric=citedcg_one_centric
```

**🚀 VS Code Users**: Press `F5` to run with VS Code's debugger configurations!

---

## 📁 **Project Structure**

```
BizChatScripts/
├── 📄 README.md                           # Project documentation
├── 📋 requirements.txt                    # Python dependencies
├── 🚫 .gitignore                          # Git ignore patterns
├── 🎨 .vscode/                            # VS Code configurations (launch.json, tasks.json)
├── 📚 docs/                               # Documentation
│   ├── llm_usage_and_auth_guide.md        # LLM usage and authentication guide
│   ├── utterance_complexity_classifier.md # Complexity classification documentation
│   └── utterance_selector.md              # Utterance selector documentation
├── 🧠 llms/                               # Core LLM framework
│   ├── base_applier.py                    # Base classes with retry logic and threading
│   ├── llm_api.py                         # Core LLM API client
│   ├── llm_api_unified.py                 # Unified LLM API with intelligent routing
│   ├── ms_llm_api_client_adapter.py       # Microsoft LLM API Client adapter
│   └── prompts/                           # Prompt management system
├── 📝 prompts/                           # Prompt files (markdown format)
│   ├── example_text_summarizer/           # Text summarization prompts
│   ├── utterance_complexity_classifier/   # Complexity classification prompts
│   ├── utterance_personalizer/            # Personalization prompts
│   └── utterance_selector/                # Selection prompts
├── 🎯 using_llms/                        # LLM applications and scripts
│   ├── example_*.py                       # LLM framework examples
│   ├── utterance_complexity_classifier.py # Classify utterance complexity
│   ├── utterance_personalizer.py          # Personalize utterances
│   └── utterance_selector.py              # Select utterances based on criteria
├── 📊 examples/                          # Example input data
│   └── example_*_input.json               # Test data for examples
├── 📊 projects/                          # **Main Project Implementations**
│   ├── 🎯 gpt5_gen_complex_utterances/   # Complex Utterance Generation
│   │   └── synthetic_tenants/             # Using synthetic tenant data to generate complex utterances 
│   ├── 🔍 gpt5_select_complex_utterances/ # Select complex utterances from fine-tuning data
│   │   ├── complexity_statistics.py       # Classification statistics
│   │   ├── selected_utterance_statistics.py # Selection analysis
│   │   ├── filter_optimal_switching.py   # Optimal switching filter
│   │   └── parse_playground_results.py   # Parse test results
│   ├── 👤 personalization/              # Generation of personalized utterance patterns
│   └── 📈 seval/                        # **SEVAL Analysis Toolkit**
│       ├── seval_analysis_toolkit.py    # Comprehensive SEVAL analysis
│       └── seval_metrics_analysis.py    # A/B testing metrics analysis
├── 🧪 tests/                           # Unit tests
├── 🔧 tools/                           # Data processing utilities
│   ├── universal_file_reader.py        # Universal file format reader
│   ├── extract_results_to_excel.py     # Export results to Excel
│   ├── merge_json_files.py             # JSON file merger
│   └── conv_*.py                       # Format conversion utilities
└── 🛠️ utils/                           # Shared utilities
    ├── file_reader.py                  # File reading utilities
    ├── json_utils.py                   # JSON processing
    ├── markdown_reports.py             # Markdown report generation
    └── statistics_utils.py             # Statistical analysis
```

---

## 🚀 **Major Projects**

This repository includes several production-ready project implementations for BizChat Copilot development:

### 1. 🎯 **Complex Utterance Generation** (`projects/gpt5_gen_complex_utterances/`)
Using synthetic tenant data to generate complex utterances for reasoning model evaluation and testing.

**Key Tools:**
- `filter_treatment_winners.py` - Filter utterances based on SEVAL A/B test results
- Filters for non-conflicting treatment wins across multiple SEVAL jobs
- Integrates with SEVAL metrics analysis for data-driven test data selection

**Usage:**
```bash
# From the BizChatScripts root directory
python -m projects.gpt5_gen_complex_utterances.synthetic_tenants.filter_treatment_winners \
    --seval_results_file=projects/seval/results/combined_results.tsv \
    --test_data_file=projects/gpt5_gen_complex_utterances/synthetic_tenants/results/test_utterances.tsv \
    --output_file=projects/gpt5_gen_complex_utterances/synthetic_tenants/results/filtered_winners.tsv
```

### 2. 🔍 **Selection of Complex Utterance** (`projects/gpt5_select_complex_utterances/`)
Select complex utterances from fine-tuning data for reasoning model training and evaluation.

**Key Tools:**
- `complexity_statistics.py` - Analyze classification results with visualizations
- `selected_utterance_statistics.py` - Analyze selection patterns and distributions
- `filter_optimal_switching.py` - Filter by optimal model switching classes
- `parse_playground_results.py` - Parse playground test results
- `extract_data_to_playground_format.py` - Convert to playground format for testing

**Usage:**
```bash
# Analyze complexity classification results
python -m projects.gpt5_select_complex_utterances.complexity_statistics \
    --input_file=data/labeled_utterances.json --create_plots=True

# Analyze selected utterances
python -m projects.gpt5_select_complex_utterances.selected_utterance_statistics \
    data/results/selected_utterances.json --output_path=report.md

# Filter optimal switching classes
python -m projects.gpt5_select_complex_utterances.filter_optimal_switching \
    --parsed_results_file=data/parsed_results.json \
    --optimal_switching_file=data/Merged_Optimal_Switching.json \
    --output_file=data/filtered_optimal.json
```

### 3. 👤 **Utterance Personalization** (`projects/personalization/`)
Generation of personalized utterance patterns for diverse user contexts and scenarios.

**Usage:**
```bash
python -m using_llms.utterance_personalizer \
    --input_file=data/base_utterances.json \
    --output_file=data/personalized_utterances.json \
    --threads=5
```

**Key Features:**
- Generate personalized variations of utterances
- Adapt utterances for specific contexts and user personas
- LLM framework integration for context-aware personalization

### 4. �📈 **SEVAL Analysis Toolkit** (`projects/seval/`)
Comprehensive tools for analyzing SEVAL (Search Evaluation) results and A/B testing metrics.

**Key Tools:**
- `seval_analysis_toolkit.py` - Multi-purpose SEVAL analysis tool
  - Query search across SEVAL files
  - Model statistics extraction
  - Search results pattern analysis
  - Conversation details extraction
- `seval_metrics_analysis.py` - A/B testing statistical analysis
  - Win/loss/tie analysis for treatment vs control
  - Multi-job comparison and aggregation
  - Non-conflicting winner export
  - Comprehensive statistical reporting

**SEVAL Analysis Toolkit Features:**
```bash
# Search for queries in SEVAL files
python -m projects.seval.seval_analysis_toolkit search_query --query "microsoft" --exp both

# Extract model statistics (which reasoning models were used)
python -m projects.seval.seval_analysis_toolkit extract_model_statistics \
    --input_dir=seval_data/raw_data \
    --output_file=results/model_stats.tsv \
    --threads=16

# Analyze search results patterns
python -m projects.seval.seval_analysis_toolkit analyze_search_results \
    --mappings_file=results/query_mappings.tsv \
    --output_file=results/search_analysis.tsv
```

**SEVAL Metrics Analysis Features:**
```bash
# Export win/loss/tie analysis for single job
python -m projects.seval.seval_metrics_analysis export_win_loss_utterances \
    --metrics_file=seval_data/metrics/all_metrics_paired.csv \
    --metric=citedcg_one_centric \
    --output_file=results/win_loss_analysis.tsv

# Compare and combine two SEVAL jobs
python -m projects.seval.seval_metrics_analysis export_two_jobs_utterances \
    --job1_metrics_path=seval_data/job1/all_metrics_paired.csv \
    --job2_metrics_path=seval_data/job2/all_metrics_paired.csv \
    --metric=citedcg_one_centric \
    --output_file=results/combined_analysis.tsv

# Statistical analysis with visualizations
python -m projects.seval.seval_metrics_analysis analyze_metrics \
    --metrics_file=seval_data/metrics/all_metrics_paired.csv \
    --metric=citedcg_one_centric
```

---

## 🤖 **LLM-Based Utterance Analysis Tools**

Collection of LLM-powered tools in the `using_llms/` directory for utterance complexity classification and intelligent selection.

### 🎓 **Utterance Complexity Classification** (`utterance_complexity_classifier.py`)
Classify utterances as requiring chat model (simple) or reasoning model (complex).

**Usage:**
```bash
python -m using_llms.utterance_complexity_classifier \
    --input_file=data/utterances.json \
    --output_file=data/classified_utterances.json \
    --threads=8
```

**Key Features:**
- Classifies utterances as "chat" (simple) or "cot" (chain-of-thought/reasoning)
- Confidence scoring for each classification
- Detailed reasoning for classification decisions
- Multi-threaded batch processing

### 📊 **Utterance Selection** (`utterance_selector.py`)
Select diverse, representative utterances from large datasets based on configurable criteria.

**Usage:**
```bash
python -m using_llms.utterance_selector \
    --input_file=data/all_utterances.json \
    --output_file=data/selected_utterances.json \
    --selection_criteria=diversity \
    --max_utterances=1000
```

**Key Features:**
- Intelligent selection based on diversity and representativeness
- Configurable selection criteria
- Handles large datasets efficiently
- Multi-threaded processing for performance

---

## 🛠️ **Data Processing Tools**

The `tools/` directory contains utilities for data processing and format conversion:

- **`universal_file_reader.py`** - Read various file formats (JSON, JSONL, TSV, CSV, Excel, TXT)
- **`extract_results_to_excel.py`** - Export analysis results to Excel spreadsheets
- **`merge_json_files.py`** - Merge multiple JSON files with various strategies
- **`jsonl_viewer.py`** - View and navigate JSONL files
- **`JinjaTemplateLoader.py`** - Load and process Jinja2 templates
- **Conversion utilities** (`conv_*.py`) - Convert between different data formats

---

## 🔧 **Utility Modules**

The `utils/` directory provides shared utilities used across projects:

- **`file_reader.py`** - Unified file reading interface
- **`json_utils.py`** - JSON parsing and manipulation
- **`markdown_reports.py`** - Generate formatted markdown reports
- **`statistics_utils.py`** - Statistical analysis functions
- **`types.py`** - Common type definitions

---

## 📖 **Example Scripts**

The `examples/` directory contains sample input files for testing LLM framework applications:

```bash
# Text summarization example
python -m using_llms.example_text_summarizer

# RSP-style scoring example
python -m using_llms.example_rsp_scorer

# Custom applier example (sentiment analysis & code review)
python -m using_llms.example_custom_applier demo

# Direct LLM API usage example
python -m using_llms.example_direct_api
```

---

## 🧠 **LLM Framework Documentation**

Comprehensive guide to using the LLM framework for building AI-powered applications with Microsoft internal models.

### ⚙️ **LLM Client Types and Model Access**

The LLM framework component provides three client types with different capabilities and model support:

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

#### **Client Type Comparison and Selection**

**Key Functional Differences:**

| Client Type        | Model Support                                                                    | Endpoint                | Use Case                      |
| ------------------ | -------------------------------------------------------------------------------- | ----------------------- | ----------------------------- |
| **RSP**            | Fixed set (e.g., `dev-gpt-41-longco-2025-04-14`)                                 | Fixed endpoint          | Development, tested workflows |
| **LLM API Client** | All Microsoft internal models (`gpt-4o-2024-05-13`, `dev-gpt-5-reasoning`, etc.) | Auto-selected endpoints | Production, latest models     |
| **Unified**        | Best of both (smart routing)                                                     | Intelligent selection   | Maximum flexibility           |

**When to Use Each Client Type:**

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

---

### 🎯 **Model Configuration and Usage**

#### **Available Models by Client Type**

| Client Type                  | Available Models                                                                     | Best Use Cases                            |
| ---------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------- |
| **RSP**                      | `dev-gpt-41-longco-2025-04-14` (tested)                                              | Stable workflows, development, debugging  |
| **Microsoft LLM API Client** | `gpt-4o-2024-05-13`, `dev-gpt-5-reasoning`, `dev-gpt-41-longco-2025-04-14`, and more | Latest models, production, broader access |
| **Unified**                  | All of the above (automatic routing)                                                 | Maximum flexibility, fallback resilience  |

**📋 Complete Model List**: For the full list of available models via Microsoft LLM API Client, see: https://substrate.microsoft.net/v2/llmApi/modelList

#### **Model Selection Examples**

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

#### **Common Configuration Patterns**

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

#### **Artifacts You'll Notice**

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

---

### 🔧 **Parameter Configuration Guide**

#### **Model Input and Configuration**
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

#### **How Model Selection Works for Each Client Type**

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

#### **Parameter Usage by Client Type**

| Parameter          | RSP Client                     | LLM API Client                          | Unified Client                                      |
| ------------------ | ------------------------------ | --------------------------------------- | --------------------------------------------------- |
| `threads`          | ✅ Used for parallel processing | ✅ Used for parallel processing          | ✅ Always used                                       |
| `retries`          | ✅ Framework-level retries      | ❌ Ignored (library handles internally)  | ✅ Used per routing decision                         |
| `ms_scenario_guid` | ❌ Ignored (uses RSP auth)      | ✅ Used for authentication (has default) | ⚠️ Used only for LLM API Client routes (has default) |

#### **Unified Client Parameter Details**

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

#### **Default Values and Requirements**

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

#### **Common Parameter Issues & Solutions**

**Missing or Custom ms_scenario_guid:**
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

**Unified Client Configuration Options:**
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

**Retry Parameter Expectations:**
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

---

### ⚡ **Threading and Retry Architecture**

The framework provides **layered concurrency and error handling** that works consistently across all client types:

#### **Framework Level (BizChatScripts)**
- **`threads`**: Controls **parallel processing** of multiple dataset items
  - Example: Process 1000 text summaries using 5 worker threads
  - Works with all client types (RSP, Microsoft LLM API Client, Unified)
  - Each thread handles multiple items independently with progress tracking

- **`retries`**: Framework-level retry logic for **individual API calls**
  - Provides consistent retry behavior across different client types
  - Handles network issues, temporary service unavailability
  - Works as an additional layer on top of client-specific retry mechanisms

#### **Client Level (Individual Libraries)**
- **Microsoft LLM API Client**: Has sophisticated internal mechanisms
  - `error_tolerance`: Internal handling of acceptable failed requests
  - `backoff`: Built-in exponential backoff for rate limiting
  - `max_requests_per_minute`: Automatic rate limiting

- **RSP Client**: Has framework-integrated retry mechanisms
  - Direct retry control through framework parameters
  - Network-level retries for connection issues

#### **Parameter Usage by Client Type**

| Parameter | RSP Client                              | Microsoft LLM API Client                       | Unified Client                                |
| --------- | --------------------------------------- | ---------------------------------------------- | --------------------------------------------- |
| `threads` | ✅ Full control over parallel processing | ✅ Full control over parallel processing        | ✅ Always used for parallel processing         |
| `retries` | ✅ Direct framework retry control        | ❌ Largely ignored (library handles internally) | ✅ Applied based on routing decision           |
| `model`   | Limited to tested models                | All internal Microsoft models                  | Any model (auto-routes to appropriate client) |

#### **Why Framework Parameters Matter**

- **Consistent Interface**: Same parameters work across all client types
- **Layered Resilience**: Framework retries + client retries = better reliability  
- **Dataset Processing**: Client libraries don't handle parallel processing of multiple items
- **Progress Tracking**: Framework provides multi-threaded progress bars and logging
- **Model Flexibility**: Unified client allows switching approaches without code changes

---

### 🔐 **Authentication Mechanisms**

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

---

### 🏗️ **Building LLM Applications**

The framework provides four main approaches for building LLM applications:

#### **1. Framework Pattern (Recommended)** 

See `example_simple_scorer.py` - Basic text scoring with comprehensive file I/O and client type demonstrations.

See `example_text_summarizer.py` - Uses base classes with built-in threading, error handling, and prompt management.

**Key Parameters:**
- `--threads=N`: Number of parallel workers (1-5 recommended)
- `--retries=N`: API retry attempts (2-5 recommended)  
- `--max_items=N`: Limit processing for testing (-1 for all)

#### **2. RSP-Compatible Pattern**
See `example_rsp_scorer.py` - For applications that need to integrate with existing RSP systems and scoring workflows.

#### **3. Custom Multi-Applier Pattern**
See `example_custom_applier.py` - Demonstrates multiple LLM appliers in one application (sentiment analysis + code review) with both external and inline prompt management.

#### **4. Direct API Pattern**
See `example_direct_api.py` - For maximum control over API interactions and custom processing flows.

#### **Where to Place Your Files**
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