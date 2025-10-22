# Utterance Selection and Analysis Tools

This directory contains a comprehensive suite of tools for utterance complexity classification, selection, analysis, and conversion to playground format for testing reasoning models.

## Files

### Analysis & Statistics Tools
- `complexity_statistics.py` - Analyzes complexity classification results with graphical visualizations
- `selected_utterance_statistics.py` - Analyzes selected utterances from the selector output
- `playground_results_statistics.py` - Analyzes playground test results

### Data Processing & Filtering
- `filter_optimal_switching.py` - Filters utterances based on optimal switching classes
- `deduplicate_selected_utterances.py` - Removes duplicate utterances from selection results
- `testdata_for_utterance_selector.py` - Generates test data for the utterance selector

### Data Conversion Tools
- `extract_data_to_playground_format.py` - Converts classification data to playground input format
- `playground_converter.py` - General utility module for playground format conversion
- `parse_playground_results.py` - Parses playground output results

### Data Files
- `data/` - Directory containing classification data and results
  - `multiple_tool_call_utterance_for_all_segments.json` - Raw utterance data
  - `multiple_tool_call_utterance_for_all_segments_labeled.json` - Classification results
  - `playground_cot_data.json` - CoT utterances in playground format
  - `Merged_Optimal_Switching.json` - Optimal switching class definitions
  - `detailed_statistics.json` - Detailed statistics output
  - `statistics_report.json` - Statistics analysis reports
  - `results/` - Generated results and analysis outputs

## Tools Overview

### 1. Complexity Statistics Analyzer (`complexity_statistics.py`)
Analyzes utterance complexity classification results and generates comprehensive statistics with optional graphical visualizations.

### 2. Selected Utterance Statistics (`selected_utterance_statistics.py`)
Analyzes the output of the utterance selector to provide insights on selection patterns, category distribution, and selection rounds.

### 3. Optimal Switching Filter (`filter_optimal_switching.py`)
Filters parsed playground results to include only utterances with optimal switching classes defined in the configuration file.

### 4. Playground Data Converter (`extract_data_to_playground_format.py`)
Extracts specific classification types (CoT, Chat, etc.) and converts them to playground input format while preserving all metadata.

### 5. Playground Results Parser (`parse_playground_results.py`)
Parses and analyzes results from playground testing to extract switching classes and performance metrics.

## Usage

### 1. Complexity Classification Analysis

#### Basic Analysis (Console Output Only)
```bash
python complexity_statistics.py --input_file="data/multiple_tool_call_utterance_for_all_segments_labeled.json"
```

#### Analysis with Detailed Category Stats
```bash
python complexity_statistics.py --input_file="data/results.json" --show_detailed_stats=True
```

#### Analysis with JSON Output
```bash
python complexity_statistics.py --input_file="data/results.json" --output_file="statistics_report.json"
```

#### Analysis with Graphical Plots
```bash
python complexity_statistics.py --input_file="data/results.json" --create_plots=True
```

#### Full Analysis (Detailed Stats + JSON Output + Plots)
```bash
python complexity_statistics.py --input_file="data/results.json" --output_file="statistics_report.json" --show_detailed_stats=True --create_plots=True
```

---

### 2. Selected Utterance Statistics Analysis

#### Basic Analysis
```bash
python selected_utterance_statistics.py data/results/selected_utterances.json
```

#### Save Markdown Report
```bash
python selected_utterance_statistics.py data/results/selected_utterances.json --output_path=report.md
```

#### Generate with Plots
```bash
python selected_utterance_statistics.py data/results/selected_utterances.json --output_path=report.md --create_plots=True
```

#### Generate Plots with Custom Directory
```bash
python selected_utterance_statistics.py data/results/selected_utterances.json --create_plots=True --plot_output_dir=visualizations
```

---

### 3. Optimal Switching Filter

#### Filter Parsed Results
```bash
python filter_optimal_switching.py \
    --parsed_results_file="data/parsed_playground_results.json" \
    --optimal_switching_file="data/Merged_Optimal_Switching.json" \
    --output_file="data/filtered_optimal_results.json"
```

---

### 4. Playground Data Conversion

#### Extract CoT (Chain of Thought) Utterances (Default)
```bash
python extract_data_to_playground_format.py --input_file="data/results.json" --output_file="playground_cot.json"
```

#### Extract Chat Model Utterances
```bash
python extract_data_to_playground_format.py --input_file="data/results.json" --output_file="playground_chat.json" --classification=chat
```

#### Extract All Utterances
```bash
python extract_data_to_playground_format.py --input_file="data/results.json" --output_file="playground_all.json" --classification=all
```

#### Extract with Custom Source Name
```bash
python extract_data_to_playground_format.py --input_file="data/results.json" --output_file="playground.json" --source="my-experiment" --classification=cot
```

---

### 5. Parse Playground Results

#### Parse Results from Playground Testing
```bash
python parse_playground_results.py --input_file="playground_output.json" --output_file="parsed_results.json"
```

---

### 6. Deduplicate Selected Utterances

#### Remove Duplicates from Selection
```bash
python deduplicate_selected_utterances.py --input_file="selected_utterances.json" --output_file="deduplicated_utterances.json"
```

## Playground Data Format

The playground format is structured as follows:

```json
[
  {
    "input": {
      "parameters": "{\"utterance\": \"user query text\", \"source\": \"data-source\", \"segment\": \"category-name\"}"
    },
    "output": {
      "completions": []
    },
    "results": [
      {
        "name": "classification_metadata",
        "output": "metadata",
        "evaluations": {
          "Classification": {
            "score": 100,
            "assessment": "Classified as: cot"
          },
          "Confidence": {
            "score": 95,
            "assessment": "Confidence: 0.95"
          },
          "Category": {
            "score": 100,
            "assessment": "Category: cwc - generate_response"
          }
        }
      }
    ]
  }
]
```

## Graphical Visualizations

When `--create_plots=True` is specified, the analyzer creates four types of visualizations:

1. **Overall Distribution Pie Chart** (`overall_distribution.png`)
   - Shows the overall distribution of Chat Model vs Reasoning Model classifications
   - Includes error counts if present
   - Color-coded: Green (Chat), Orange (Reasoning), Red (Errors)

2. **Top Categories Bar Chart** (`top_categories_distribution.png`)
   - Horizontal stacked bar chart showing the top 15 categories by utterance count
   - Each bar is divided to show Chat vs Reasoning Model distribution within the category
   - Useful for identifying which categories have the most utterances and their complexity distribution

3. **Complexity Ratio Distribution** (`complexity_ratio_distribution.png`)
   - Histogram showing how complexity ratios (CoT/Total) are distributed across all categories
   - Color-coded bins: Green (low complexity <20%), Orange (medium 20-50%), Red (high >50%)
   - Shows mean and median lines for reference

4. **Category Comparison Scatter Plot** (`category_comparison_scatter.png`)
   - 2D scatter plot with Total Utterances (log scale) vs Complexity Ratio
   - Point sizes scale with utterance count, colors indicate complexity level
   - Annotates top categories by size and highest complexity categories
   - Useful for identifying outliers and patterns

## Requirements

- Python 3.7+
- fire (for CLI interface)
- matplotlib and numpy (for graphical visualizations, optional)
- Standard library modules: json, os, logging, pathlib, collections

Install plotting dependencies:
```bash
pip install matplotlib numpy
```

## Example Output

The analyzer provides comprehensive statistics including:

- **Overall Summary**: Total categories, utterances, and classification distribution
- **Classification Distribution**: Percentage breakdown of Chat Model vs Reasoning Model classifications
- **Confidence Analysis**: Average confidence scores and ranges for each classification type
- **Category Rankings**: Top categories by utterance count and complexity ratio
- **Detailed Category Statistics**: Per-category breakdown with confidence metrics
- **Graphical Visualizations**: Four types of plots showing distribution patterns and trends

### Sample Results

```
================================================================================
UTTERANCE COMPLEXITY CLASSIFICATION STATISTICS
================================================================================

📊 OVERALL SUMMARY:
  Total Categories: 111
  Total Utterances: 22452
  Successfully Classified: 22452

🎯 CLASSIFICATION DISTRIBUTION:
  Chat Model: 18139 (80.8%)
  Reasoning Model: 4313 (19.2%)

📈 CONFIDENCE ANALYSIS:
  CHAT: avg=0.955, range=[0.85-1.0], median=0.95
  COT: avg=0.933, range=[0.85-1.0], median=0.95

📋 TOP 10 CATEGORIES BY UTTERANCE COUNT:
   1. meeting_prep_recap_keyword                                   (3700 utterances)
   2. cwc - search_web                                             (2006 utterances)
   3. cwc - generate_response                                      (663 utterances)
   ...

🧠 TOP 10 CATEGORIES BY COMPLEXITY RATIO:
   1. Creation - Draft project Plan                                (0.905)
   2. Creation - Draft user feedback survey                        (0.700)
   3. Bizchat - Bizchat Security                                   (0.667)
   ...
```

## Workflow Overview

The typical workflow for utterance selection and analysis:

1. **Classify Utterances** → Generate labeled data with complexity classifications
2. **Analyze Classifications** → Use `complexity_statistics.py` to review distribution
3. **Convert to Playground** → Use `extract_data_to_playground_format.py` for testing
4. **Test in Playground** → Run reasoning model tests
5. **Parse Results** → Use `parse_playground_results.py` to extract switching classes
6. **Filter Optimal** → Use `filter_optimal_switching.py` to select best performers
7. **Analyze Selection** → Use `selected_utterance_statistics.py` for final insights
8. **Deduplicate** → Use `deduplicate_selected_utterances.py` if needed

## VS Code Integration

The tools are integrated with VS Code through launch configurations for easy debugging and execution. Available configurations include:
- Complexity Statistics Analysis (with/without plots)
- Selected Utterance Statistics Analysis
- Optimal Switching Filter
- Playground Data Conversion
- Parse Playground Results
- Deduplication

## Requirements

### Core Dependencies
- Python 3.7+
- fire (for CLI interface)
- Standard library modules: json, os, logging, pathlib, collections

### Optional Dependencies (for visualizations)
```bash
pip install matplotlib numpy
```

## Data Formats

### Classification Results Format
```json
{
  "category_name": [
    {
      "utterance": "sample utterance text",
      "classification": "chat" | "cot",
      "confidence": 0.95,
      "reasoning": "explanation text"
    }
  ]
}
```

### Selected Utterances Format
```json
{
  "selected_round_1": [...],
  "selected_round_2": [...],
  "metadata": {...}
}
```

### Optimal Switching Classes Format
```json
{
  "ReasoningClasses": [
    "Category Name 1",
    "Category Name 2"
  ]
}
```
