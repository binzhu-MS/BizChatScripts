# Utterance Selector

## What's New

### Simplified Single-File Architecture (Latest Update)
- **Single Output File**: Eliminated dual-file complexity - all data in one user-specified file
- **Embedded Resume Information**: Resume state stored directly in output file, no separate checkpoints needed
- **Thread-Safe Operations**: Built-in file locking for multi-threading compatibility
- **Streamlined Resume**: Can resume from any existing output file automatically
- **Simplified Debug**: Single file contains both production data and analysis information
- **Enhanced Reliability**: No checkpoint file coordination issues or sync problems

## Overview

The Utterance Selector is an intelligent, enterprise-grade utterance selection system that uses an iterative approach to select balanced, diverse, and high-quality utterances from large datasets. Designed specifically for Microsoft Copilot enterprise search scenarios, it provides fast, reliable selection with comprehensive progress tracking and resume capabilities.

## Features

- **Iterative Balanced Selection**: Ensures proportional representation across all categories
- **Enterprise Search Context**: Understands Copilot enterprise, web, and hybrid search scenarios  
- **Multi-hop Complexity Recognition**: Identifies utterances requiring disambiguation and multi-step reasoning
- **Embedded Resume Capability**: Resume state stored directly in output file - no separate checkpoints needed
- **Single Output File**: All data (production results + analysis) in one user-specified file
- **Thread-Safe Operations**: Built-in file locking prevents data corruption during multi-threading
- **Enhanced Analytics**: Comprehensive selection analysis and human-readable justifications
- **Fast Performance**: Processes 1000+ utterances/minute without LLM analysis
- **LLM-Guided Selection**: Optional intelligent selection within categories for quality optimization
- **Fire CLI Integration**: Full command-line interface with extensive parameter customization

## Architecture

### Core Components

1. **UtteranceSelector**: Main CLI interface class (`using_llms/utterance_selector.py`)
2. **IterativeUtteranceSelector**: Core selection engine with LLM analysis (`using_llms/utterance_selector_core.py`) 
3. **Prompt System**: Enterprise-aware selection prompts (`prompts/utterance_selector/v0.1.0.md`)
4. **Automatic Resume System**: State persistence directly in output file
5. **Thread-Safe Operations**: File locking for concurrent access protection

### Selection Strategy

The system employs a **popularity-based iterative approach**:

1. **Category Organization**: Groups utterances by category (e.g., `enterprise_queries|chat`, `web_queries|cot`)
2. **Popularity Sorting**: Orders categories by utterance count (largest first)
3. **Incremental Selection**: Selects small increments (1-3) from each category per round
4. **Balanced Distribution**: Ensures all categories get representation
5. **LLM Enhancement**: Optional intelligent selection using **comprehensive reference strategy** that:
   - Uses **ALL selected examples** from most popular and current categories (not limited samples)
   - Provides **current round guidance** with real-time examples from ongoing selections
   - Applies **adaptive quality standards** based on category size ratios
   - **Prioritizes complexity, diversity, and enterprise relevance equally** (high priority), with **personalization as secondary consideration** (medium priority)
   - Ensures comprehensive coverage of COT reasoning scenarios, varied use cases, and enterprise-focused utterances
   - Maintains **token efficiency** through strategic example selection rather than random sampling

### Quality Consistency Across Categories

The system ensures selected utterances from different categories maintain similar quality profiles through its **comprehensive reference strategy**:

1. **Complete Category Context**: Instead of limited sampling, the system provides:
   - **ALL selected utterances** from the most popular category (excluding current round) 
   - **ALL selected utterances** from the current category being processed
   - **Current round selections** from the most popular category for immediate guidance
   - **Adaptive quality standards** that adjust based on category size ratios

2. **Comprehensive Reference Framework**: Each utterance selection benefits from complete context:
   - **Full category patterns**: LLM sees complete selection history, not just 3 samples
   - **Consistent standards**: Same criteria applied with full visibility into successful selections
   - **Real-time guidance**: Current round examples provide immediate quality benchmarks
   - **Strategic efficiency**: Uses all relevant examples while maintaining token efficiency

3. **Enhanced Selection Quality**: The comprehensive approach ensures:
   - **Pattern recognition**: LLM understands complete selection patterns within and across categories
   - **Quality consistency**: Full context enables more reliable quality maintenance
   - **Adaptive flexibility**: Smaller categories get more flexible standards while maintaining core quality
   - **Token optimization**: Strategic example selection provides maximum guidance within token limits
4. **Unified Quality Standards**: The comprehensive reference strategy maintains consistency by:
   - **Complete context awareness**: Every selection decision has full visibility into successful patterns
   - **Cross-category learning**: Current category learns from complete most popular category history  
   - **Adaptive thresholds**: Quality standards adjust appropriately for smaller vs larger categories
   - **Token-efficient guidance**: Strategic use of ALL relevant examples without token waste

This approach ensures that whether an utterance comes from `enterprise_queries|cot` or `web_queries|chat`, it meets the same high standards for complexity, enterprise relevance, and diversity, resulting in a cohesive final dataset.

### Enterprise Context Understanding

The system recognizes three search types:
- **Enterprise-only**: Internal company data (emails, documents, SharePoint, Teams)
- **Web-only**: External web content and information
- **Hybrid**: Combined enterprise and web search

It prioritizes utterances with **multi-hop complexity**:
- **Disambiguation**: Requiring clarification of intent or terms
- **Multi-step reasoning**: Breaking into multiple analytical steps
- **Context gathering**: Needing multiple searches for complete context
- **Cross-domain synthesis**: Combining information from different sources

### Comprehensive Reference Strategy

The system uses an advanced **comprehensive reference strategy** for LLM-guided selection that maximizes selection quality while maintaining token efficiency:

#### Strategy Components

1. **Complete Category Context**: 
   - **ALL selected utterances** from the most popular category (excluding current round)
   - **ALL selected utterances** from the current category being processed
   - **Current round selections** from most popular category for immediate guidance

2. **Adaptive Quality Standards**:
   - Category size ratios determine flexibility levels
   - Smaller categories (<50% of most popular) get more flexible standards
   - Core quality criteria (complexity, enterprise relevance) remain consistent
   - Style and format variations allowed for smaller categories

3. **Token Efficiency**:
   - **128K context window**: GPT-4 provides ample capacity for comprehensive examples
   - **Strategic selection**: Uses ALL relevant examples instead of random sampling
   - **4000 output tokens**: Enhanced capacity for detailed selection reasoning
   - **Smart organization**: Examples grouped by category with clear section headers

#### Benefits Over Limited Sampling

- **No 3-sample limitation**: LLM sees complete selection patterns
- **Better pattern recognition**: Full context enables more reliable quality decisions
- **Consistent standards**: Complete visibility ensures uniform quality across categories
- **Real-time guidance**: Current round examples provide immediate quality benchmarks
- **Adaptive flexibility**: Appropriate standards for different category sizes

## Installation & Setup

### Prerequisites

```bash
# Core dependencies
pip install fire>=0.6.0
pip install requests tqdm pydantic python-dotenv  
pip install pystache>=0.5.4 msal>=1.20,<2

# Or install from requirements.txt
pip install -r requirements.txt
```

### Authentication Setup

The selector uses Microsoft MSAL authentication for LLM access. Ensure your authentication is properly configured:

```bash
# Set up authentication (one-time)
python using_llms/auth_setup.py
```

## Usage

### Command Line Interface

#### Basic Usage

```bash
# Basic selection with all default features
python using_llms/utterance_selector.py input_data.json

# Custom target count and output file
python using_llms/utterance_selector.py input_data.json \
    --output_file selected_utterances.json \
    --target_utterances 2000
```

#### Advanced Features

```bash
# Automatic resume capability (no separate checkpoint file needed)
python using_llms/utterance_selector.py input_data.json \
    --output_file selection_results.json

# Custom selection parameters
python using_llms/utterance_selector.py input_data.json \
    --increment_per_category 3 \
    --target_utterances 5000

# Test mode with specific parameters
python using_llms/utterance_selector.py input_data.json \
    --test=True \
    --test_utterances 50 \
    --test_categories 5

# Disable incremental saving (only final save)
python using_llms/utterance_selector.py input_data.json \
    --incremental_saving False
```

#### Parameter Reference

| Parameter                | Default                    | Description                                           |
| ------------------------ | -------------------------- | ----------------------------------------------------- |
| `input_file`             | Required                   | Path to JSON file with utterance data                 |
| `output_file`            | `selected_utterances.json` | Path for complete results file (includes resume info) |
| `test`                   | `False`                    | Whether to run in test mode                           |
| `test_utterances`        | `12`                       | Number of utterances to select in test mode           |
| `test_categories`        | `3`                        | Number of top categories to use in test mode          |
| `target_utterances`      | `2000`                     | Number of utterances to select in production mode     |
| `increment_per_category` | `2`                        | Utterances per category per round                     |
| `resume`                 | `True`                     | Whether to attempt resuming from existing output      |
| `incremental_saving`     | `True`                     | Save results after each round                         |

### Python API

#### Basic API Usage

```python
from using_llms.utterance_selector_core import IterativeUtteranceSelector

# Initialize selector
selector = IterativeUtteranceSelector(
    increment_per_category=2
)

# Run selection
results = selector.select_utterances_iteratively(
    data=your_utterance_data,
    target_count=2000,
    use_llm_analysis=True
)

print(f"Selected {results['total_selected']} utterances")
```

#### Advanced API with Resume

```python
from using_llms.utterance_selector_core import IterativeUtteranceSelector

# Initialize with custom configuration
selector = IterativeUtteranceSelector(
    increment_per_category=3,
    model_config={
        "model": "dev-gpt-41-longco-2025-04-14", 
        "temperature": 0.1,
        "max_tokens": 4000  # Enhanced output capacity for comprehensive analysis
    }
)

# Run with full feature set
results = selector.select_utterances_iteratively(
    data=large_dataset,
    target_count=5000,
    use_llm_analysis=True,
    output_file="large_selection_results.json"
)
```

#### High-Level CLI API

```python
from using_llms.utterance_selector import UtteranceSelector

# Initialize with parameters
selector = UtteranceSelector(
    increment_per_category=2
)

# Process files directly
results = selector.select_utterances(
    input_file="input_data.json",
    output_file="results.json",
    target_count=2000
)
```

## Input Data Format

### Supported Formats

#### Format 1: Segmented Data (Recommended)
```json
{
  "enterprise_queries": [
    {
      "utterance": "Find my recent emails from John about the Q4 project",
      "switching_class": "chat"
    },
    {
      "utterance": "Analyze the performance metrics across all departments and identify areas for improvement based on recent feedback",
      "switching_class": "cot"  
    }
  ],
  "web_queries": [
    {
      "utterance": "What's the weather forecast for Seattle?",
      "switching_class": "chat"
    }
  ]
}
```

#### Format 2: Flat List
```json
[
  {
    "utterance": "Show me today's calendar",
    "segment": "enterprise_queries",
    "switching_class": "chat"
  },
  {
    "utterance": "Compare market trends for renewable energy",
    "segment": "web_queries", 
    "switching_class": "cot"
  }
]
```

### Category Formation

Categories are automatically formed by combining:
- **Segment**: Data source/type (`enterprise_queries`, `web_queries`, etc.)
- **Switching Class**: Complexity indicator (`chat`, `cot`, etc.)

Example categories: `enterprise_queries|chat`, `enterprise_queries|cot`, `web_queries|chat`

## Output Data Structure

The system now uses a **simplified single-file approach** that combines all data:

**Single Output File**: Contains production-ready utterance data + comprehensive analysis + embedded resume information

> **📋 User Notification**: The system will inform you about the output file upon completion:
> ```
> 📄 File Created:
>    • Complete results: selected_utterances.json
>    Contains production data, analysis, and resume information in one file.
>    Ready for immediate use - no additional files needed.
> ```

### Unified Output File (e.g., `selected_utterances.json`)

**Purpose**: Complete results file containing production data, analysis, and resume information.

```json
{
  "metadata": {
    "total_selected": 2000,
    "target_count": 2000,
    "rounds_completed": 8,
    "categories_used": 6,
    "selection_method": "simple",
    "timestamp": "2025-01-07T15:30:45.123456"
  },
  "category_distribution": {
    "enterprise_queries|chat": 450,
    "enterprise_queries|cot": 380,
    "web_queries|chat": 420,
    "web_queries|cot": 350,
    "hybrid_queries|chat": 200,
    "hybrid_queries|cot": 200
  },
  "selected_utterances": [
    {
      "utterance": "Find my recent emails from marketing team",
      "switching_class": "chat",
      "segment": "enterprise_queries",
      "id": "utterance_123",
      "category": "enterprise_queries|chat",
      "round_selected": 1
    }
    // ... more utterances
  ],
  "resume_info": {
    "last_round": 8,
    "category_order": [
      "enterprise_queries|chat",
      "enterprise_queries|cot",
      "web_queries|chat", 
      "web_queries|cot",
      "hybrid_queries|chat",
      "hybrid_queries|cot"
    ],
    "completed": true
  }
}
```

### Resume Capability

The system can automatically resume from any existing output file:

**Resume Detection**: The system automatically detects:
- Completed selections (`resume_info.completed: true`)  
- Partial selections that can be continued
- Incompatible files that need fresh start

**Resume Process**:
1. **Automatic Detection**: System reads existing output file and checks `resume_info` section
2. **Smart Resume**: If file is compatible, continues from last completed round
3. **Completion Detection**: If already complete, returns existing results instantly
4. **Fresh Start**: If incompatible parameters, starts new selection

**Example Resume Scenarios**:
```bash
# First run - creates output file with resume_info
python using_llms/utterance_selector.py data.json --output_file=results.json --target_utterances=1000

# Second run - automatically detects completion
python using_llms/utterance_selector.py data.json --output_file=results.json --target_utterances=1000
# Output: "🎯 Already completed: 1000/1000"

# Different parameters - starts fresh 
python using_llms/utterance_selector.py data.json --output_file=results.json --target_utterances=2000
# Output: "Starting fresh selection (parameters changed)"
```

### File Usage

The single output file contains everything needed for both production and analysis:

#### Typical Workflow

```bash
# 1. Run selection - creates complete output file
python utterance_selector.py input.json --output_file=results.json

# 2. Use directly for production (selected_utterances array)
python process_data.py results.json

# 3. Resume capability built-in - same command continues from where it left off
python utterance_selector.py input.json --output_file=results.json  # Automatically resumes or returns existing results

# 4. Extract specific sections as needed
python analyze_categories.py results.json  # Uses category_distribution
python review_metadata.py results.json     # Uses metadata section
```

### Incremental Progress Files

During execution, the main output file is updated after each round:

```json
{
  "metadata": {
    "status": "in_progress",           // "in_progress" or "completed" 
    "last_update_round": 3,
    "total_selected": 600,
    "target_count": 2000
  },
  "round_summary": {
    "current_round": 3,
    "total_rounds": 3,
    "round_selections": [              // Last 3 rounds for context
      {
        "round": 1,
        "selected_count": 200,
        "category_breakdown": {...}
      }
    ]
  }
}
```

## Performance Characteristics

### Speed Benchmarks

| Mode                   | Speed                   | Use Case                   |
| ---------------------- | ----------------------- | -------------------------- |
| **Without LLM**        | ~1000 utterances/minute | Fast bulk selection        |
| **With LLM Analysis**  | ~100 utterances/minute  | Quality-focused selection  |
| **Resume from Output** | Same as above           | Recovery from interruption |

### Memory Usage

- **Linear scaling**: Memory usage grows linearly with dataset size
- **Efficient processing**: Processes categories incrementally
- **Thread-safe**: Built-in file locking for concurrent access
- **Embedded state**: Resume information stored in output file, minimal overhead

### Token Usage and Efficiency

The system is designed for large-scale datasets with comprehensive token management:

#### Token Capacity
- **Model**: GPT-4 with 128K context window (massive capacity for examples)
- **Output tokens**: 4000 tokens (enhanced from 2000 for detailed analysis)
- **Input efficiency**: Strategic use of ALL relevant examples without token waste

#### Comprehensive Reference Strategy Efficiency
- **Smart scaling**: Uses complete category context when beneficial, fallback when needed
- **Strategic organization**: Examples grouped by relevance (popular category → current category → current round)
- **Adaptive sizing**: Quality standards adjust based on available context size
- **No random sampling**: Uses ALL relevant examples for maximum selection quality

#### Large Dataset Handling
- **2K+ utterances**: Comprehensive strategy provides full context without token issues
- **Efficiency gains**: ~95%+ token reduction vs naive approach (using all selected utterances)
- **Quality improvement**: Complete category context leads to better selection decisions
- **Scalable approach**: Works efficiently with datasets up to 50K+ utterances

### Scalability

- **Small datasets** (< 1K utterances): Completes in seconds
- **Medium datasets** (1K-50K): Minutes to hours depending on LLM usage
- **Large datasets** (50K+): Hours with embedded resume capability for reliability

## Advanced Configuration

### Custom Model Configuration

```python
custom_config = {
    "model": "gpt-4o-2024-11-20",
    "temperature": 0.05,     # Very low for consistency
    "max_tokens": 4000,      # Enhanced output capacity for comprehensive analysis
    "top_p": 0.95
}

selector = IterativeUtteranceSelector(
    model_config=custom_config,
    increment_per_category=1  # More conservative
)
```

### Custom Prompt Usage

```python
from llms import prompts

# Load custom prompt
custom_prompt = prompts.get("my_custom_selector", "1.0.0")

selector = IterativeUtteranceSelector(
    prompt=custom_prompt
)
```

### Debug and Development

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# Run with maximum verbosity
results = selector.select_utterances_iteratively(
    data=debug_data,
    target_count=50,
    use_llm_analysis=True
)
```

## Troubleshooting

### Common Issues

#### 1. **Resume Compatibility Errors**
```
Starting fresh selection (parameters changed)
```
**Solution**: Resume info is tied to specific parameters. Use same `target_utterances` and other settings, or delete the output file to start completely fresh.

#### 2. **Authentication Failures**
```
ERROR: Failed to authenticate with LLM API
```
**Solution**: Re-run authentication setup:
```bash
python using_llms/auth_setup.py
```

#### 3. **Token Usage Warnings**
```
WARNING: Token limit management: Using X examples from Y,ZZZ selected utterances
```
**Understanding**: This is normal behavior for large datasets. The comprehensive reference strategy automatically manages token usage by:
- Using ALL relevant examples when possible
- Applying strategic selection when approaching token limits
- Providing efficiency summaries in logs

**No action needed**: The system handles token optimization automatically.

#### 4. **Memory Issues with Large Datasets**
```
ERROR: Out of memory during processing
```
**Solution**: Use smaller `increment_per_category` and enable `incremental_saving`:
```bash
python using_llms/utterance_selector.py large_data.json \
    --increment_per_category 1 \
    --incremental_saving True
```

#### 5. **No Categories Found**
```
WARNING: No valid categories found in data
```
**Solution**: Ensure data has proper format with `utterance` field and category indicators.

### Performance Optimization

#### For Speed Priority
```bash
python using_llms/utterance_selector.py data.json \
    --increment_per_category 5 \
    --target_utterances 1000
```

#### For Quality Priority  
```bash
python using_llms/utterance_selector.py data.json \
    --increment_per_category 1 \
    --target_utterances 2000
```

#### For Large Datasets
```bash
python using_llms/utterance_selector.py large_data.json \
    --output_file large_results.json \
    --increment_per_category 2 \
    --target_utterances 5000
```

## Integration Examples

### Integration with Data Processing Pipeline

```python
import json
from using_llms.utterance_selector_core import IterativeUtteranceSelector

def process_dataset(input_path, output_path, selection_config):
    """Integrate utterance selection into data pipeline."""
    
    # Load and preprocess data
    with open(input_path, 'r') as f:
        raw_data = json.load(f)
    
    # Initialize selector with config
    selector = IterativeUtteranceSelector(**selection_config)
    
    # Run selection with resume capability
    results = selector.select_utterances_iteratively(
        data=raw_data,
        target_count=selection_config.get('target_count', 2000),
        use_llm_analysis=True,
        output_file=output_path
    )
    
    # Post-process results
    return {
        'selected_count': results['total_selected'],
        'categories_used': results['selection_summary']['categories_used'],
        'balance_quality': calculate_balance_quality(results),
        'output_file': output_path
    }

# Configuration for different scenarios
configs = {
    'development': {
        'increment_per_category': 1,
        'target_count': 100
    },
    'production': {
        'increment_per_category': 3,
        'target_count': 5000
    }
}

# Run processing
results = process_dataset(
    'raw_utterances.json',
    'selected_utterances.json', 
    configs['production']
)
```

### Batch Processing Multiple Files

```python
import glob
from pathlib import Path

def batch_select_utterances(input_pattern, output_dir, common_config):
    """Process multiple files with utterance selection."""
    
    input_files = glob.glob(input_pattern)
    results_summary = []
    
    for input_file in input_files:
        input_path = Path(input_file)
        output_file = Path(output_dir) / f"{input_path.stem}_selected.json"
        
        selector = IterativeUtteranceSelector(**common_config)
        
        try:
            results = selector.select_utterances_iteratively(
                data=json.load(open(input_file)),
                output_file=str(output_file),
                **common_config
            )
            
            results_summary.append({
                'file': input_file,
                'status': 'success',
                'selected': results['total_selected']
            })
            
        except Exception as e:
            results_summary.append({
                'file': input_file, 
                'status': 'error',
                'error': str(e)
            })
    
    return results_summary

# Process all JSON files in directory
results = batch_select_utterances(
    'data/utterances_*.json',
    'output/selected/',
    {
        'target_count': 1000,
        'increment_per_category': 2
    }
)
```

## Best Practices

### 1. **Production Deployment**
- Resume capability is automatic - no separate checkpoint files needed
- Use appropriate `increment_per_category` based on dataset size
- Monitor log output for selection quality
- Output files contain all necessary resume information

### 2. **Quality Optimization with Comprehensive Strategy**
- **Enable LLM analysis** for quality-critical selections to leverage comprehensive reference strategy
- **Trust the adaptive standards**: System automatically adjusts quality thresholds for smaller categories
- **Monitor token efficiency logs**: Review efficiency summaries to understand comprehensive strategy benefits
- **Use lower `increment_per_category`** for more careful selection with maximum context utilization
- **Validate category balance** in results - comprehensive strategy maintains better consistency

### 3. **Large Dataset Best Practices**
- **Leverage comprehensive strategy**: For 2K+ utterances, the system provides maximum quality with optimal token usage
- **Monitor efficiency gains**: Check logs for token reduction percentages vs naive approaches
- **Enable incremental saving**: Track comprehensive selection quality improvements over rounds
- **Use adaptive standards**: Allow system to apply appropriate flexibility for smaller categories

### 4. **Performance Optimization**
- Use higher `increment_per_category` for faster completion
- Consider parallel processing for multiple independent datasets
- Monitor selection quality through logs

### 5. **Monitoring and Debugging**
- Enable INFO or DEBUG logging for detailed progress tracking
- Review output files for comprehensive selection analysis
- Check category balance scores for quality validation
- Resume capability is automatic from output files

## Related Tools

This module works well with other tools in the BizChatScripts ecosystem:

- **utterance_complexity_classifier.py**: Pre-classify utterances for complexity
- **rsp_style_scorer.py**: Score utterances for response style preferences
- **text_summarizer.py**: Generate summaries of selected utterance sets
- **custom_applier.py**: Apply custom processing to selected utterances

## Contributing

When extending the utterance selector:

1. **Adding New Selection Criteria**: Extend the prompt in `prompts/utterance_selector/`
2. **Custom Selection Logic**: Override methods in `IterativeUtteranceSelector`
3. **New Output Formats**: Extend the output generation methods
4. **Performance Improvements**: Focus on the core iteration loop in `select_utterances_iteratively`

## License

This module is part of the BizChatScripts project and follows the same licensing terms.
