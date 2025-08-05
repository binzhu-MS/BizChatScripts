# Utterance Complexity Classifier

## Overview

The Utterance Complexity Classifier is a machine learning-powered tool that analyzes enterprise utterances and classifies them as requiring either a **chat model** (for simple queries) or a **reasoning model** (for complex/Chain-of-Thought processing). This classification helps route user queries to the most appropriate AI model, optimizing both response quality and computational resources.

## Features

- **Intelligent Classification**: Uses advanced LLM analysis to determine utterance complexity
- **Multi-threaded Processing**: Configurable worker threads for high-performance batch processing
- **Incremental Saving**: Real-time progress saving for large datasets (20K+ utterances)
- **Thread-Safe Operations**: File locking prevents data corruption during concurrent processing
- **Flexible Output Formats**: Supports both incremental and batch saving modes
- **Enterprise Search Context**: Handles enterprise, web, and hybrid search scenarios
- **Progress Monitoring**: Real-time logging and progress tracking
- **Fire CLI Integration**: Command-line interface with extensive parameter customization

## Architecture

### Core Components

1. **UtteranceComplexityClassifier**: Main processing class extending `ChatCompletionLLMApplier`
2. **Incremental Saving System**: Thread-safe file operations with progress persistence
3. **Prompt System**: Advanced classification prompts with contextual reasoning
4. **Threading Manager**: Configurable multi-threading with load balancing

### Classification Logic

The system evaluates utterances based on several complexity indicators:

**Chat Model Candidates (Simple)**:
- Direct information retrieval
- Single-step queries
- Straightforward factual questions
- Basic web searches

**Reasoning Model Candidates (Complex/COT)**:
- Multi-step analysis required
- Comparative analysis across sources
- Cross-functional synthesis
- Contextual reasoning
- Subjective judgment tasks
- Analytical thinking requirements

## Installation & Setup

### Prerequisites

```bash
# Required dependencies
pip install fire>=0.6.0
pip install requests tqdm pydantic python-dotenv
pip install pystache>=0.5.4 msal>=1.20,<2
```

### Authentication Setup

The classifier uses Microsoft MSAL authentication for LLM access. Ensure your authentication is properly configured before running.

## Usage

### Command Line Interface

#### Basic Usage

```bash
# Process first 20 utterances (default for testing)
python using_llms\utterance_complexity_classifier.py

# Process ALL utterances in the dataset
python using_llms\utterance_complexity_classifier.py --max_utterances=-1

# Process with custom parameters
python using_llms\utterance_complexity_classifier.py \
    --max_utterances=1000 \
    --threads=8 \
    --save_batch_size=50 \
    --incremental_save=True
```

#### Advanced Examples

```bash
# Large dataset processing with incremental saving
python using_llms\utterance_complexity_classifier.py \
    --max_utterances=-1 \
    --threads=6 \
    --save_batch_size=100 \
    --output_suffix="_production"

# Fast processing for small datasets (save only at end)
python using_llms\utterance_complexity_classifier.py \
    --max_utterances=500 \
    --threads=4 \
    --incremental_save=False

# Custom input file processing
python using_llms\utterance_complexity_classifier.py \
    --input_file="path/to/your/utterances.json" \
    --max_utterances=-1 \
    --save_batch_size=200
```

### Parameters Reference

| Parameter          | Type | Default       | Description                            |
| ------------------ | ---- | ------------- | -------------------------------------- |
| `input_file`       | str  | Auto-detected | Path to input JSON file                |
| `max_utterances`   | int  | 20            | Max utterances to process (-1 for all) |
| `output_suffix`    | str  | "_labeled"    | Suffix for output filename             |
| `threads`          | int  | 3             | Number of worker threads               |
| `save_batch_size`  | int  | 100           | Utterances per incremental save        |
| `incremental_save` | bool | True          | Enable incremental saving              |

## Input/Output Formats

### Input Format

The classifier expects JSON files with the following structure:

```json
{
  "category_name": [
    "First utterance text...",
    "Second utterance text...",
    "Third utterance text..."
  ],
  "another_category": [
    "More utterance examples..."
  ]
}
```

### Output Format

The classifier outputs enhanced JSON with classification results:

```json
{
  "category_name": [
    {
      "utterance": "Original utterance text...",
      "classification": "chat|cot",
      "confidence": 0.95,
      "reasoning": "Detailed explanation of classification decision...",
      "complexity_indicators": [
        "Multi-step analysis",
        "Comparative analysis",
        "Contextual reasoning"
      ]
    }
  ]
}
```

### Classification Values

- **"chat"**: Simple queries suitable for chat models
- **"cot"**: Complex queries requiring Chain-of-Thought/reasoning models

## Performance Optimization

### For Large Datasets (10K+ utterances)

```bash
# Recommended settings for 20K utterances
python using_llms\utterance_complexity_classifier.py \
    --max_utterances=-1 \
    --threads=8 \
    --save_batch_size=100 \
    --incremental_save=True
```

**Benefits**:
- Progress saved every ~12-13 utterances per thread
- Fault tolerance - resume from interruptions
- Real-time monitoring of results
- Memory efficient processing

### For Small Datasets (<1K utterances)

```bash
# Optimized for speed
python using_llms\utterance_complexity_classifier.py \
    --max_utterances=500 \
    --threads=4 \
    --incremental_save=False
```

**Benefits**:
- Faster processing (no I/O overhead)
- Simpler workflow
- Single final output file

## Threading and Concurrency

### Thread Distribution

The system automatically distributes work across threads:

- **Batch Size**: 100 utterances
- **Threads**: 6
- **Per-Thread Save**: ~17 utterances per thread before save

### Thread Safety

- **File Locking**: `threading.Lock()` ensures safe concurrent file access
- **Duplicate Prevention**: Automatic deduplication during incremental saves
- **Progress Tracking**: Per-thread counters with synchronized updates

## Monitoring and Logging

### Real-time Progress

```
INFO:__main__:Input file: dataset.json
INFO:__main__:Max utterances: -1
INFO:__main__:Worker threads: 6
INFO:__main__:Save batch size: 100
INFO:__main__:Running Utterance Complexity Classifier...
INFO:__main__:Results will be saved incrementally every ~16 utterances per thread
INFO:__main__:Processing Category: enterprise_queries (1543 utterances)
INFO:__main__:  Processing utterance #1/1543: Analyze the market trends...
INFO:__main__:Classified utterance #1: cot
INFO:__main__:Incrementally saved 16 utterances to output.json
```

### Summary Reports

```
Classification Summary by Category:
================================================================================

Category: enterprise_queries
  Total: 1543
  Chat Model: 892 (57.8%)
  Reasoning Model: 651 (42.2%)

Overall Summary:
  Total Utterances: 1543
  Chat Model: 892 (57.8%)
  Reasoning Model: 651 (42.2%)
```

## VS Code Integration

### Debug Configurations

Three pre-configured debug profiles are available:

1. **"Run Utterance Complexity Classifier (Test)"**
   - 10 utterances, 3 threads, no incremental saving
   - Perfect for development and testing

2. **"Run Utterance Complexity Classifier (Incremental)"**
   - 100 utterances, 6 threads, save every 20 utterances
   - Demonstrates incremental saving features

3. **"Run Utterance Complexity Classifier (All - 6 threads)"**
   - All utterances, 6 threads, save every 100 utterances
   - Production-ready configuration

### Running in VS Code

1. Open `utterance_complexity_classifier.py`
2. Press `F5` or go to Run > Start Debugging
3. Select desired configuration
4. Monitor progress in the integrated terminal

## Error Handling

### Common Issues

**Authentication Errors**:
```bash
ERROR: Microsoft MSAL authentication failed
```
Solution: Run `python examples\auth_setup.py` to configure authentication

**File Access Errors**:
```bash
ERROR: Permission denied writing to output file
```
Solution: Ensure output directory has write permissions

**Memory Issues**:
```bash
ERROR: Out of memory processing large dataset
```
Solution: Reduce `save_batch_size` or use incremental saving

### Recovery from Interruptions

When using incremental saving, interrupted processes can be resumed:

1. The output file contains all processed utterances
2. Restart with the same parameters
3. The system will skip already-processed utterances (some overlap may occur)
4. Processing continues from the interruption point

## API Reference

### UtteranceComplexityClassifier Class

```python
class UtteranceComplexityClassifier(ChatCompletionLLMApplier):
    def __init__(
        self, 
        threads=None, 
        save_batch_size=100, 
        incremental_save=True, 
        output_file_path=None, 
        **kwargs
    ):
        """
        Initialize the classifier with configuration options.
        
        Args:
            threads: Number of worker threads
            save_batch_size: Utterances per incremental save
            incremental_save: Enable/disable incremental saving
            output_file_path: Path for output file
        """
```

### Main Functions

```python
def classify_utterances(
    input_file=None, 
    max_utterances=20, 
    output_suffix="_labeled", 
    threads=3,
    save_batch_size=100,
    incremental_save=True
):
    """
    Main entry point for utterance classification.
    
    Returns: None (results saved to file)
    """

def load_utterance_data(json_file_path, max_utterances=20):
    """
    Load and preprocess utterance data from JSON file.
    
    Returns: List of processed data items
    """

def print_summary(results):
    """
    Print classification summary statistics.
    """
```

## Best Practices

### Performance Optimization

1. **Thread Count**: Use 4-8 threads for optimal performance
2. **Batch Size**: 50-200 utterances per save for large datasets
3. **Memory Management**: Enable incremental saving for datasets >5K utterances
4. **Monitoring**: Use logging to track progress on long-running jobs

### Data Quality

1. **Input Validation**: Ensure JSON format is correct before processing
2. **Output Verification**: Check classification distribution in summary
3. **Error Handling**: Review error logs for failed classifications
4. **Confidence Scores**: Monitor confidence levels for quality assessment

### Production Deployment

1. **Authentication**: Secure MSAL credentials in production environment
2. **File Permissions**: Ensure write access to output directories
3. **Resource Limits**: Monitor CPU and memory usage during processing
4. **Backup Strategy**: Regular backups of classification results

## Troubleshooting

### Performance Issues

**Slow Processing**:
- Increase thread count: `--threads=8`
- Reduce batch size: `--save_batch_size=50`
- Disable incremental saving for small datasets

**High Memory Usage**:
- Enable incremental saving: `--incremental_save=True`
- Reduce batch size: `--save_batch_size=50`
- Process in smaller chunks using `--max_utterances`

### File Issues

**Permission Denied**:
- Check write permissions on output directory
- Ensure file is not locked by another process
- Try different output location

**Corrupted Output**:
- Thread concurrency issue - check file locking implementation
- Disk space shortage - ensure adequate storage
- Interrupted process - use incremental saving for recovery

## Examples and Use Cases

### Enterprise Query Routing

```bash
# Process enterprise knowledge base queries
python using_llms\utterance_complexity_classifier.py \
    --input_file="enterprise_queries.json" \
    --max_utterances=-1 \
    --threads=6 \
    --save_batch_size=100 \
    --output_suffix="_enterprise_classified"
```

Use the results to route queries:
- **Chat model**: Simple fact lookups, basic Q&A
- **Reasoning model**: Analysis, comparisons, complex problem-solving

### Customer Support Optimization

```bash
# Classify customer support tickets
python using_llms\utterance_complexity_classifier.py \
    --input_file="support_tickets.json" \
    --max_utterances=-1 \
    --threads=4 \
    --save_batch_size=200 \
    --output_suffix="_support_classified"
```

Benefits:
- Route simple queries to fast chat models
- Escalate complex issues to reasoning models
- Improve response times and accuracy

### Research Dataset Processing

```bash
# Process large research datasets incrementally
python using_llms\utterance_complexity_classifier.py \
    --input_file="research_queries.json" \
    --max_utterances=-1 \
    --threads=8 \
    --save_batch_size=50 \
    --incremental_save=True \
    --output_suffix="_research_analysis"
```

Monitor progress in real-time and analyze partial results while processing continues.

## Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure authentication: `python examples\auth_setup.py`
4. Run tests: `python using_llms\test_utterance_classifier.py`

### Code Structure

- **Main Classifier**: `using_llms\utterance_complexity_classifier.py`
- **Prompt Templates**: `prompts\utterance_complexity_classifier\v0.1.0.md`
- **Test Script**: `using_llms\test_utterance_classifier.py`
- **Documentation**: `docs\utterance_complexity_classifier.md`

### Testing

```bash
# Quick functionality test
python using_llms\utterance_complexity_classifier.py --max_utterances=5

# Incremental saving test
python using_llms\utterance_complexity_classifier.py \
    --max_utterances=10 \
    --save_batch_size=3 \
    --incremental_save=True
```

## License

This project is part of the SimpleLLM framework. Please refer to the main project license for usage terms.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review log output for specific error messages
3. Verify authentication and file permissions
4. Test with smaller datasets to isolate issues

---

*Last updated: August 4, 2025*
