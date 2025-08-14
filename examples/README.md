# Example Data Files

This folder contains sample input data for demonstrating the BizChatScripts framework.

## Files

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

## Output Files

Example applications will create output files in this directory:
- `example_simple_scorer_output.json` - Simple scoring results
- `example_summarization_output.json` - Text summarization results
- `example_rsp_scoring_output.json` - RSP scoring results
- `example_custom_applier_output.json` - Mixed sentiment analysis and code review results
- `example_direct_api_output.json` - Direct API call results

These output files show the structure of processed data and can be used as templates for your own applications.
