# Example Data Files and Demonstrations

This folder contains sample input data and demonstration scripts for the BizChatScripts framework.

## Data Files

### `example_simple_scorer_input.json`
Sample texts for the simple scoring example.
- **Usage**: `python -m using_llms.example_simple_scorer examples/example_simple_scorer_input.json`
- **Format**: Array of objects with `id`, `text`, and `criteria` fields
- **Size**: 5 sample texts with different evaluation scenarios

### `example_summarization_input.json`
Sample texts with various length requirements for the text summarization example.
- **Usage**: `python -m using_llms.example_text_summarizer examples/example_summarization_input.json`
- **Format**: Array of objects with `id`, `input_text`, and `max_length` fields
- **Size**: 4+ sample texts covering AI, climate, Renaissance, and quantum topics

### `example_rsp_scoring_input.json` 
Sample texts with scoring criteria for the RSP-style scorer example.
- **Usage**: `python -m using_llms.example_rsp_scorer examples/example_rsp_scoring_input.json`
- **Format**: Array of objects with `id`, `text`, and `criteria` fields
- **Size**: Multiple evaluation scenarios with different criteria

### `example_custom_applier_input.json`
Mixed data for custom applier demonstrations including sentiment analysis and code review.
- **Usage**: `python -m using_llms.example_custom_applier process examples/example_custom_applier_input.json`
- **Format**: Array of objects with `type` field ("sentiment" or "code_review") and corresponding data
- **Size**: 5 items demonstrating both sentiment analysis and code review functionality

### `example_direct_api_input.json`
Simple prompts for direct API usage demonstrations.
- **Usage**: `python -m using_llms.example_direct_api examples/example_direct_api_input.json`
- **Format**: Array of objects with `id` and `prompt` fields
- **Content**: Educational prompts about quantum computing, renewable energy, programming, etc.

## Demonstration Scripts

### `file_reader_example.py`
Comprehensive demonstration of the UniversalFileReader capabilities.
- **Usage**: `python examples/file_reader_example.py`
- **Features**: 
  - Reading individual files (PDF, DOCX, XLSX, PPTX, TXT)
  - Batch processing directories
  - Integration with LLM processing workflows
  - Error handling and fallback methods
- **Output**: Demonstrates file content extraction with detailed logging

## VS Code Launch Configurations

The `.vscode/launch.json` file contains pre-configured debug sessions for all examples:

- **📚 Example: Simple Scorer** - Basic text scoring with file I/O
- **📚 Example: Simple Scorer (Demo Mode)** - Demonstration with hardcoded data
- **📚 Example: Text Summarizer** - Process summarization data with custom parameters
- **📚 Example: Text Summarizer (Demo Mode)** - Show parameter configuration demos
- **📚 Example: RSP Scorer** - Multi-criteria text evaluation
- **📚 Example: RSP Scorer (Demo Mode)** - Parameter demonstration
- **📚 Example: Custom Applier** - Sentiment analysis and code review from file
- **📚 Example: Custom Applier (Demo Mode)** - Mixed applier demonstrations
- **📚 Example: Direct API Usage** - Low-level API interactions
- **📚 Example: Direct API Usage (Patterns)** - Show common patterns

## Related Tools and Utilities

The BizChatScripts framework includes additional tools and utilities in the `tools/` directory:

- **UniversalFileReader** - Multi-format file content extraction (PDF, DOCX, XLSX, PPTX)
- **Data Conversion Tools** - JSON/TSV/Excel format converters
- **Statistics and Metrics** - Analysis tools for generated data
- **Template Processing** - Jinja2 template utilities
- **Data Validation** - JSON validation and processing tools

For detailed information about these tools, see `tools/README_UniversalFileReader.md`.

## Quick Start

*Important: Run all commands from the BizChatScripts root directory*

1. **Run with default parameters:**
   ```bash
   # Simple scorer with file I/O
   python -m using_llms.example_simple_scorer
   
   # Text summarization
   python -m using_llms.example_text_summarizer
   
   # Custom applier with mixed sentiment/code review
   python -m using_llms.example_custom_applier process
   
   # File reading demonstration
   python examples/file_reader_example.py
   ```

2. **Run demo modes with hardcoded data:**
   ```bash
   python -m using_llms.example_simple_scorer demo
   python -m using_llms.example_text_summarizer demo
   python -m using_llms.example_custom_applier demo
   ```

3. **Run with custom parameters:**
   ```bash
   python -m using_llms.example_text_summarizer examples/example_summarization_input.json --threads=3 --retries=5 --max_items=2
   python -m using_llms.example_custom_applier process examples/example_custom_applier_input.json --client_type=unified --threads=3
   ```

4. **Use VS Code debugger:**
   - Press `F5` and select an example configuration
   - Set breakpoints to inspect the framework behavior

5. **Explore file processing capabilities:**
   ```bash
   # Run universal file reader examples
   python examples/file_reader_example.py
   
   # Use tools for data conversion and analysis
   python tools/universal_file_reader_example.py
   python tools/get_json_statistics.py input_file.json
   ```

## Output Files

Example applications will create output files in this directory:
- `example_simple_scorer_output.json` - Simple scoring results
- `example_summarization_output.json` - Text summarization results
- `example_rsp_scoring_output.json` - RSP scoring results
- `example_custom_applier_output.json` - Mixed sentiment analysis and code review results
- `example_direct_api_output.json` - Direct API call results

These output files show the structure of processed data and can be used as templates for your own applications.

## Integration Examples

The examples demonstrate key integration patterns:

1. **File I/O Processing**: Read input data, process with LLMs, save structured output
2. **Batch Processing**: Handle multiple items with threading and error recovery
3. **Multi-format File Reading**: Extract content from various document formats
4. **Custom Prompt Engineering**: Apply different LLM techniques for specific tasks
5. **Error Handling**: Robust processing with retries and fallback strategies
6. **Configuration Management**: Flexible parameter handling for different use cases

For production implementations, refer to the `projects/` directory which contains real-world applications using these patterns.
