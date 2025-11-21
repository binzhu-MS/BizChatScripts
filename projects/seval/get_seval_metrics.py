"""
SEVAL Metrics Reader and Extractor

This module provides two main types of functionality for SEVAL experiment data:

1. CSV-BASED METRICS PROCESSING (A/B Test Results):
   - Reads aggregated metric scores from CSV files in the metrics job folder
   - Used for comparing Control vs Treatment experiments
   - Input: CSV files from folders like "100335_metrics/consolidated_*"
   - Output: DataFrames with metrics comparison and markdown reports

2. JSON-BASED SCORE EXTRACTION (Per-Result Scores):
   - Extracts CiteDCG scores for individual search results for each utterance from 
     results.json for either Control or Treatment experiment.
   - Used for detailed per-utterance and per-search-result analysis
   - Input: JSON files from folders like "130949_metrics/Consolidated NDCG and CiteDCG Labels Control/results.json"
   - Output: JSON with scores for each search result

USAGE:

A. CSV Metrics Processing (existing functionality):
    # List available CSV files
    python -m projects.seval.get_seval_metrics list_csv_files --job_path="c:/path/to/100335_metrics"
    
    # Read and display metrics summary
    python -m projects.seval.get_seval_metrics read_metrics --job_path="c:/path/to/100335_metrics"
    
    # Get comprehensive metrics summary
    python -m projects.seval.get_seval_metrics get_metrics_summary --job_path="c:/path/to/100335_metrics"
    
    # Read metrics with reasoning class column
    python -m projects.seval.get_seval_metrics read_metrics --job_path="c:/path/to/100335_metrics" --add_reasoning_class=True --reasoning_json_path="path/to/data.json"
    
    # Generate CiteDCG report by reasoning class
    python -m projects.seval.get_seval_metrics generate_citedcg_report --job_path="c:/path/to/100335_metrics" --add_reasoning_class=True --reasoning_json_path="path/to/data.json"
    
    ---
    
    # Programmatic usage
    from projects.seval.get_seval_metrics import MetricsDataReader, MetricsAnalyzer
    
    reader = MetricsDataReader(job_path="/path/to/100335_metrics")
    df = reader.read_metrics()
    
    analyzer = MetricsAnalyzer(job_path="/path/to/100335_metrics")
    results = analyzer.extract_metric_pairs(['metric1_control', 'metric1_treatment'])

B. Per-Result CiteDCG Score Extraction (new functionality):
    # Command line - module calling (preferred)
    python -m projects.seval.get_seval_metrics extract_per_result_citedcg --metrics_folder="130949_metrics" --experiment="control" --output_file="scores.json"
    
    # Or direct file execution (also works)
    python get_seval_metrics.py extract_per_result_citedcg --metrics_folder="130949_metrics" --experiment="control" --output_file="scores.json"
    
    # Extract for treatment experiment
    python -m projects.seval.get_seval_metrics extract_per_result_citedcg --metrics_folder="130949_metrics" --experiment="treatment" --output_file="scores.json"
    
    # Filter by specific utterance
    python -m projects.seval.get_seval_metrics extract_per_result_citedcg --metrics_folder="130949_metrics" --experiment="control" --utterance="scorecard" --output_file="scores.json"
    
    # Programmatic usage
    from projects.seval.get_seval_metrics import PerResultCiteDCGExtractor
    
    extractor = PerResultCiteDCGExtractor()
    extractor.extract(metrics_folder="130949_metrics", experiment="control", output_file="scores.json")
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fire
import pandas as pd
from fire.core import FireExit

# Import from the utils package (sibling folder)
from utils.statistics_utils import tdiff

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerResultCiteDCGExtractor:
    """
    Extract per-result CiteDCG scores from SEVAL results.json files.
    
    Extracts individual search result scores (as opposed to aggregated CSV metrics).
    
    Provides standardized interface:
    - Input: metrics_folder name + experiment (auto-constructs path)
    - Optional utterance filtering
    - JSON output with per-result scores
    """
    
    def __init__(self):
        """
        Initialize extractor.
        
        Uses default base path: projects/seval/seval_data/
        """
        self.base_path = Path("seval_data")
    
    def extract(
        self,
        metrics_folder: str,
        experiment: str,
        output_file: str,
        utterance: Optional[str] = None
    ):
        """
        Extract CiteDCG scores from results.json.
        
        Args:
            metrics_folder: Metrics folder name (e.g., "130949_metrics")
            experiment: "control" or "treatment"
            output_file: Path for output JSON file
            utterance: Optional - filter by utterance substring
        """
        # Construct path to results.json
        exp_capitalized = experiment.capitalize()
        results_path = (
            self.base_path / metrics_folder /
            f"Consolidated NDCG and CiteDCG Labels {exp_capitalized}" /
            "results.json"
        )
        
        if not results_path.exists():
            logger.error(f"Results file not found: {results_path}")
            return
        
        logger.info(f"Reading CiteDCG data from: {results_path}")
        logger.info(f"  Metrics folder: {metrics_folder}, Experiment: {experiment}")
        if utterance:
            logger.info(f"  Filtering by utterance: '{utterance}'")
        
        # Load and parse JSON (handles single or concatenated objects)
        all_data = []
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Try standard JSON first
            try:
                data = json.loads(content)
                all_data = [data] if isinstance(data, dict) else data
            except json.JSONDecodeError as e:
                # Standard parsing failed, try concatenated JSON objects
                logger.warning(
                    "Standard JSON parsing failed, "
                    "attempting concatenated objects"
                )
                decoder = json.JSONDecoder()
                idx = 0
                while idx < len(content):
                    try:
                        obj, end_idx = decoder.raw_decode(content, idx)
                        if isinstance(obj, dict):
                            all_data.append(obj)
                        idx = end_idx
                        # Skip whitespace between objects
                        while idx < len(content) and content[idx].isspace():
                            idx += 1
                    except json.JSONDecodeError:
                        break
                
                if not all_data:
                    logger.error(f"Failed to load results.json: {e}")
                    return
                else:
                    logger.info(
                        f"Parsed {len(all_data)} concatenated JSON objects"
                    )
        except Exception as e:
            logger.error(f"Failed to load results.json: {e}")
            return
        
        # Extract CiteDCG data from all JSON objects
        extracted_data = []
        for data_obj in all_data:
            extracted_data.extend(self._extract_cite_dcg_data(data_obj))
        
        if not extracted_data:
            logger.warning("No CiteDCG data found in results file")
            return
        
        logger.info(f"Extracted {len(extracted_data)} results")
        
        # Filter by utterance if specified
        if utterance:
            filtered_data = [
                item for item in extracted_data
                if utterance.lower() in item.get('utterance', '').lower()
            ]
            logger.info(
                f"Filtered to {len(filtered_data)} results "
                f"matching '{utterance}'"
            )
            extracted_data = filtered_data
        
        # Group by query and write JSON output
        grouped_data = self._group_by_query(extracted_data)
        self._write_json_output(grouped_data, output_file)
        logger.info(f"Wrote {len(grouped_data)} queries to {output_file}")
    
    def _extract_cite_dcg_data(self, data: dict) -> list:
        """Extract CiteDCG scores from JSON structure."""
        results = []
        all_search_results = data.get("AllSearchResults", {})
        
        for turn_index, query_data in all_search_results.items():
            if not query_data:
                continue
            
            for search_domain, domain_data in query_data.items():
                if not isinstance(domain_data, list):
                    continue
                
                for domain_item in domain_data:
                    # Extract plugin metadata
                    plugin_name = domain_item.get("PluginName", "")
                    plugin_invocation = domain_item.get(
                        "PluginInvocation", ""
                    )
                    query_string = self._extract_query_from_invocation(
                        plugin_invocation
                    )
                    
                    results_list = domain_item.get("Results", [])
                    
                    for result in results_list:
                        if "CiteDCGLLMLabel" not in result:
                            continue
                        
                        extracted = self._extract_single_result(
                            result,
                            turn_index,
                            search_domain,
                            plugin_name,
                            query_string,
                        )
                        results.append(extracted)
        
        return results
    
    def _extract_query_from_invocation(self, invocation_str: str) -> str:
        """Extract query string from PluginInvocation field."""
        if not invocation_str:
            return ""
        
        try:
            # Extract the 'query' parameter value from invocation string
            # Example: "office365_search({'domain': 'files', ...})"
            import re
            match = re.search(r"'query':\s*'([^']+)'", invocation_str)
            if match:
                return match.group(1)
            
            # Try double quotes
            match = re.search(r'"query":\s*"([^"]+)"', invocation_str)
            if match:
                return match.group(1)
        except Exception as e:
            logger.debug(f"Error extracting query from invocation: {e}")
        
        return ""
    
    def _extract_single_result(
        self,
        result: dict,
        turn_index: str,
        search_domain: str,
        plugin_name: str = "",
        query_string: str = "",
    ) -> dict:
        """Extract relevant fields from a single result."""
        extracted = {
            "turn_index": turn_index,
            "search_domain": search_domain,
            "plugin_name": plugin_name,
            "query_string": query_string,
            "ReferenceId": result.get("ReferenceId", ""),
            "CiteDCGLLMLabel": result.get("CiteDCGLLMLabel", ""),
            "ResultType": result.get("ResultType", ""),
            "Type": result.get("Type", ""),
        }
        
        # Extract source metadata
        source = result.get("Source", {})
        extracted["Title"] = source.get("Title", "")
        extracted["Subject"] = source.get("Subject", "")
        
        # Extract utterance from prompt
        prompt_list = result.get("CiteDCGLLMPrompt", [])
        if prompt_list and len(prompt_list) > 0:
            prompt_content = prompt_list[0].get("content", "")
            extracted["utterance"] = self._extract_field_from_prompt(
                prompt_content, "Utterance = "
            )
            extracted["timestamp"] = self._extract_field_from_prompt(
                prompt_content, "Timestamp =  "
            )
        else:
            extracted["utterance"] = ""
            extracted["timestamp"] = ""
        
        return extracted
    
    def _extract_field_from_prompt(
        self, prompt_content: str, field_name: str
    ) -> str:
        """Extract field value from prompt content."""
        if not prompt_content or field_name not in prompt_content:
            return ""
        
        try:
            start_idx = prompt_content.find(field_name)
            if start_idx == -1:
                return ""
            
            start_idx += len(field_name)
            end_idx = prompt_content.find("\n", start_idx)
            
            if end_idx == -1:
                value = prompt_content[start_idx:].strip()
            else:
                value = prompt_content[start_idx:end_idx].strip()
            
            # Remove quotes
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            
            return value
        except Exception as e:
            logger.warning(f"Error extracting {field_name}: {e}")
            return ""
    
    def _group_by_query(self, data: list) -> list:
        """Group results by query/utterance, domain, and query_string."""
        from collections import defaultdict
        
        query_groups = defaultdict(lambda: {"results": []})
        
        for item in data:
            utterance = item.get("utterance", "")
            turn_index = item.get("turn_index", "")
            
            key = utterance
            
            if not query_groups[key].get("turn_index"):
                query_groups[key]["turn_index"] = turn_index
                query_groups[key]["utterance"] = utterance
                query_groups[key]["timestamp"] = item.get("timestamp", "")
            
            query_groups[key]["results"].append(item)
        
        # Convert to list with hierarchical grouping
        grouped_list = []
        for group_data in query_groups.values():
            results = group_data["results"]
            
            # Group by (domain, query_string) for search-level grouping
            search_groups = defaultdict(list)
            for result in results:
                domain = result.get("search_domain", "unknown")
                query_string = result.get("query_string", "")
                plugin_name = result.get("plugin_name", "")
                key = (domain, query_string, plugin_name)
                
                # Remove redundant fields for result item
                result_item = {
                    k: v
                    for k, v in result.items()
                    if k not in [
                        "turn_index",
                        "utterance",
                        "timestamp",
                        "search_domain",
                        "plugin_name",
                        "query_string",
                    ]
                }
                search_groups[key].append(result_item)
            
            # Create search entries
            searches = []
            total_results = 0
            all_scores = []
            
            for (domain, query_string, plugin_name), search_results in (
                search_groups.items()
            ):
                # Calculate stats for this search
                scores = [
                    r.get("CiteDCGLLMLabel", 0)
                    for r in search_results
                    if r.get("CiteDCGLLMLabel")
                ]
                avg_score = (
                    round(sum(scores) / len(scores), 2) if scores else None
                )
                
                searches.append(
                    {
                        "search_domain": domain,
                        "plugin_name": plugin_name,
                        "query_string": query_string,
                        "result_count": len(search_results),
                        "avg_cite_dcg_label": avg_score,
                        "results": search_results,
                    }
                )
                
                total_results += len(search_results)
                all_scores.extend(scores)
            
            # Sort searches by domain then query_string
            searches.sort(
                key=lambda x: (x["search_domain"], x["query_string"])
            )
            
            # Calculate overall average
            overall_avg = (
                round(sum(all_scores) / len(all_scores), 2)
                if all_scores
                else None
            )
            
            # Count results by domain
            domain_counts = {}
            for search in searches:
                domain = search["search_domain"]
                domain_counts[domain] = (
                    domain_counts.get(domain, 0) + search["result_count"]
                )
            
            ordered_group = {
                "turn_index": group_data.get("turn_index", ""),
                "utterance": group_data.get("utterance", ""),
                "timestamp": group_data.get("timestamp", ""),
                "total_results": total_results,
                "avg_cite_dcg_label": overall_avg,
                "results_by_domain": domain_counts,
                "searches": searches,
            }
            grouped_list.append(ordered_group)
        
        # Sort by turn_index
        grouped_list.sort(key=lambda x: int(x.get("turn_index", 0)))
        return grouped_list
    
    def _write_json_output(self, data: list, output_file: str):
        """Write data to JSONL file (one JSON object per line)."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


class ReasoningClassExtractor:
    """
    A class to extract utterance-to-reasoning-class mappings from JSON files.
    Processes test data files to create lookup tables for reasoning classification.
    """
    
    def __init__(self, json_file_path: Optional[str] = None):
        """
        Initialize the ReasoningClassExtractor.
        
        Args:
            json_file_path: Path to the JSON file containing test data
        """
        self.json_file_path = json_file_path
        self.utterance_to_class_map = {}
    
    def load_reasoning_mappings(self, json_file_path: Optional[str] = None) -> Dict[str, str]:
        """
        Load utterance-to-reasoning-class mappings from JSON file.
        
        Args:
            json_file_path: Path to JSON file (optional if set in constructor)
            
        Returns:
            dict: Mapping of utterances to reasoning classes
        """
        if json_file_path:
            self.json_file_path = json_file_path
        
        if not self.json_file_path:
            raise ValueError("JSON file path not provided")
        
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            utterance_map = {}
            
            for item in data:
                # Skip empty items
                if not item or 'input' not in item or 'results' not in item:
                    continue
                
                # Extract utterance from input parameters
                input_params = item.get('input', {}).get('parameters', '')
                if input_params:
                    try:
                        # Parse the JSON string in parameters
                        params_dict = json.loads(input_params)
                        utterance = params_dict.get('utterance', '')
                        
                        if utterance:
                            # Look for v11 entries in results array
                            results = item.get('results', [])
                            
                            # Check for entries with names starting with v11
                            for result in results:
                                if isinstance(result, dict):
                                    name = result.get('name', '')
                                    if name.startswith('v11'):
                                        reasoning_class = result.get('output', '')
                                        if reasoning_class:
                                            utterance_map[utterance] = reasoning_class
                                            logger.debug(f"Mapped utterance to class: {utterance[:50]}... -> {reasoning_class}")
                                            break  # Use first v11 match
                    
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.debug(f"Skipping item due to parsing error: {e}")
                        continue
            
            self.utterance_to_class_map = utterance_map
            logger.info(f"Loaded {len(utterance_map)} utterance-to-reasoning-class mappings")
            
            return utterance_map
            
        except FileNotFoundError:
            logger.error(f"JSON file not found: {self.json_file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading reasoning mappings: {e}")
            raise
    
    def get_reasoning_class(self, utterance: str) -> Optional[str]:
        """
        Get reasoning class for a given utterance.
        
        Args:
            utterance: The utterance to look up
            
        Returns:
            str or None: The reasoning class if found, None otherwise
        """
        return self.utterance_to_class_map.get(utterance)
    
    def get_all_mappings(self) -> Dict[str, str]:
        """
        Get all utterance-to-reasoning-class mappings.
        
        Returns:
            dict: All loaded mappings
        """
        return self.utterance_to_class_map.copy()

class MetricsAnalyzer:
    """
    A class to analyze specific metrics data and generate comparison structures.
    Extracts and processes specific columns for generating comparison results.
    """
    
    def __init__(self, 
                 job_path: Optional[str] = None,
                 metrics_file_name: str = "all_metrics_paired.csv",
                 metrics_relative_path: str = "offline_scorecard_generator_output"):
        """
        Initialize the MetricsAnalyzer.
        
        Args:
            job_path: Full path to the job folder (e.g., "/path/to/100335_metrics")
            metrics_file_name: Name of the main metrics CSV file 
            metrics_relative_path: Relative path from job folder to metrics files
        """
        self.job_path = Path(job_path) if job_path else None
        self.metrics_file_name = metrics_file_name
        self.metrics_relative_path = metrics_relative_path
        self.df: Optional[pd.DataFrame] = None
    
    def set_job_path(self, job_path: str) -> None:
        """Set the job path and clear cached data."""
        self.job_path = Path(job_path)
        self.df = None
    
    def set_file_config(self, metrics_file_name: Optional[str] = None, metrics_relative_path: Optional[str] = None) -> None:
        """Update file configuration and clear cached data."""
        if metrics_file_name:
            self.metrics_file_name = metrics_file_name
        if metrics_relative_path:
            self.metrics_relative_path = metrics_relative_path
        self.df = None
    
    def _load_data(self) -> None:
        """Load data from the configured job path if not already loaded."""
        if self.df is None:
            if not self.job_path:
                raise ValueError("Job path not set. Use set_job_path() first.")
            
            reader = MetricsDataReader(
                job_path=str(self.job_path),
                metrics_file_name=self.metrics_file_name,
                metrics_relative_path=self.metrics_relative_path
            )
            self.df = reader.read_metrics()
    
    def get_dataframe(self, add_reasoning_class: bool = False, reasoning_json_path: Optional[str] = None) -> pd.DataFrame:
        """
        Get the loaded dataframe, optionally with reasoning class column added.
        
        Args:
            add_reasoning_class: Whether to add reasoning class column based on utterance matching
            reasoning_json_path: Path to JSON file containing reasoning class mappings
            
        Returns:
            pd.DataFrame: The metrics dataframe
        """
        # Ensure data is loaded
        self._load_data()
        
        assert self.df is not None, "DataFrame should be loaded"
        
        # Make a copy to avoid modifying the original
        df_copy = self.df.copy()
        
        if add_reasoning_class and reasoning_json_path:
            # Load reasoning class mappings
            extractor = ReasoningClassExtractor(reasoning_json_path)
            try:
                utterance_map = extractor.load_reasoning_mappings()
                
                # Add reasoning_class column
                if 'utterance' in df_copy.columns:
                    df_copy['reasoning_class'] = df_copy['utterance'].map(
                        lambda x: utterance_map.get(x, '')
                    )
                    
                    # Log mapping statistics
                    mapped_count = (df_copy['reasoning_class'] != '').sum()
                    total_count = len(df_copy)
                    logger.debug(f"Added reasoning_class column: {mapped_count}/{total_count} utterances mapped")
                    
                    # Debug: Show unique reasoning classes found
                    # unique_classes = df_copy[df_copy['reasoning_class'] != '']['reasoning_class'].unique()
                    # logger.info(f"Unique reasoning classes: {list(unique_classes)}")
                    
                    # Debug: Show first few mappings
                    # sample_mappings = df_copy[df_copy['reasoning_class'] != ''][['utterance', 'reasoning_class']].head(3)
                    # logger.info(f"Sample mappings:\n{sample_mappings}")
                else:
                    logger.warning("No 'utterance' column found in dataframe, cannot add reasoning_class")
                    
            except Exception as e:
                logger.error(f"Failed to add reasoning class column: {e}")
                # Add empty reasoning_class column as fallback
                df_copy['reasoning_class'] = ''
        
        return df_copy
    
    def _load_switching_strategy(self, switching_json_path: str) -> list:
        """
        Load switching strategy from JSON file.
        
        Args:
            switching_json_path: Path to JSON file containing switching strategy
            
        Returns:
            list: List of reasoning classes that should use treatment
        """
        try:
            with open(switching_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            reasoning_classes = data.get('ReasoningClasses', [])
            logger.info(f"Loaded switching strategy with {len(reasoning_classes)} reasoning classes for treatment")
            return reasoning_classes
            
        except Exception as e:
            logger.error(f"Failed to load switching strategy: {e}")
            return []
    
    def extract_metric_pairs(self, target_columns: list, segment_column: Optional[str] = None, paired_test: bool = True, use_enhanced_df: Optional[pd.DataFrame] = None, switching_json_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract specific metric pairs (control/treatment) and calculate comparison statistics.
        
        Args:
            target_columns: List of column names to analyze (must include both _control and _treatment versions)
            segment_column: Name of the column containing segment information. If None, performs overall comparison only.
            paired_test: Use paired t-test (True) or independent t-test (False) for statistical analysis
            use_enhanced_df: Optional enhanced dataframe to use instead of self.df (e.g., with reasoning_class column)
            
        Returns:
            dict: Structured comparison results ready for formatting
        """
        # Use enhanced dataframe if provided, otherwise load default data
        if use_enhanced_df is not None:
            df_to_use = use_enhanced_df
        else:
            # Ensure data is loaded
            self._load_data()
            df_to_use = self.df
        
        # Type assertions for mypy
        assert df_to_use is not None, "DataFrame should be loaded"
        
        # Determine segments based on segment_column parameter
        if segment_column and segment_column in df_to_use.columns:
            segments = list(df_to_use[segment_column].unique())
            # Filter out empty reasoning classes if using reasoning_class column
            if segment_column == 'reasoning_class':
                segments = [seg for seg in segments if seg != '']
                # Debug: Print what reasoning classes we found
                logger.debug(f"Found reasoning classes: {segments}")
                if not segments:
                    logger.warning("No non-empty reasoning classes found, falling back to 'All Data'")
                    segments = ['All Data']
        else:
            segments = ['All Data']
        
        # Find available columns from target list
        available_columns = [col for col in target_columns if col in df_to_use.columns]
        
        # Separate control and treatment columns
        control_cols = [col for col in available_columns if '_control' in col]
        treatment_cols = [col for col in available_columns if '_treatment' in col]
        
        # Load switching strategy once if provided (to avoid duplicate loading)
        treatment_reasoning_classes = []
        if switching_json_path:
            treatment_reasoning_classes = self._load_switching_strategy(switching_json_path)
            
            # Log switching strategy statistics once for all metrics
            if treatment_reasoning_classes and 'reasoning_class' in df_to_use.columns:
                all_reasoning_classes = df_to_use['reasoning_class'].unique()
                treatment_count = 0
                control_count = 0
                for rc in all_reasoning_classes:
                    if rc and rc in treatment_reasoning_classes:
                        treatment_count += 1
                    elif rc:  # Only count non-empty reasoning classes
                        control_count += 1
                
                # Count utterances assigned to treatment vs control
                treatment_utterances = 0
                control_utterances = 0
                for _, row in df_to_use.iterrows():
                    reasoning_class = row.get('reasoning_class', '')
                    if reasoning_class in treatment_reasoning_classes:
                        treatment_utterances += 1
                    elif reasoning_class:  # Only count utterances with reasoning classes
                        control_utterances += 1
                
                logger.info(f"🔄 Current Switching Strategy Overview:")
                logger.info(f"   📊 Reasoning Classes: {treatment_count} → Treatment, {control_count} → Control")
                logger.info(f"   📈 Utterances: {treatment_utterances} → Treatment, {control_utterances} → Control")
        
        # Calculate performance gains for sorting
        metric_performance = []
        for control_col in control_cols:
            treatment_col = control_col.replace('_control', '_treatment')
            
            if treatment_col in treatment_cols:
                metric_name = control_col.replace('_control', '')
                
                control_values = df_to_use[control_col].dropna()
                treatment_values = df_to_use[treatment_col].dropna()
                
                if len(control_values) > 0 and len(treatment_values) > 0:
                    # Calculate overall statistics with proper statistical testing
                    control_values_list = control_values.tolist()
                    treatment_values_list = treatment_values.tolist()
                    
                    # Calculate optimal values per utterance (choose higher value for each utterance)
                    optimal_values = []
                    current_switching_values = []
                    
                    # Track switching statistics for calculations (no logging needed since we log once above)
                    treatment_count = 0
                    control_count = 0
                    treatment_utterances = 0
                    control_utterances = 0
                    
                    for i in range(len(control_values)):
                        if i < len(treatment_values):
                            optimal_val = max(control_values.iloc[i], treatment_values.iloc[i])
                            optimal_values.append(optimal_val)
                            
                            # Calculate current switching value
                            if switching_json_path and 'reasoning_class' in df_to_use.columns:
                                # Get the reasoning class for this utterance
                                reasoning_class = df_to_use.iloc[control_values.index[i]]['reasoning_class']
                                if reasoning_class in treatment_reasoning_classes:
                                    # Use treatment for this reasoning class
                                    current_switching_values.append(treatment_values.iloc[i])
                                    treatment_utterances += 1
                                else:
                                    # Use control for this reasoning class
                                    current_switching_values.append(control_values.iloc[i])
                                    control_utterances += 1
                            else:
                                # If no switching strategy, default to control
                                current_switching_values.append(control_values.iloc[i])
                                control_utterances += 1
                    
                    # Count unique reasoning classes in treatment vs control (for this metric's data only)
                    if switching_json_path and 'reasoning_class' in df_to_use.columns:
                        metric_reasoning_classes = df_to_use.iloc[control_values.index]['reasoning_class'].unique()
                        for rc in metric_reasoning_classes:
                            if rc and rc in treatment_reasoning_classes:
                                treatment_count += 1
                            elif rc:
                                control_count += 1
                    
                    # Calculate overall means
                    overall_optimal_mean = sum(optimal_values) / len(optimal_values) if optimal_values else 0
                    overall_current_switching_mean = sum(current_switching_values) / len(current_switching_values) if current_switching_values else 0
                    
                    # Perform overall statistical test
                    overall_stats_result = {}
                    try:
                        # Suppress scipy warnings for small sample sizes or identical values
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
                            overall_stats_result = tdiff(control_values_list, treatment_values_list, paired=paired_test)
                    except Exception as e:
                        logger.debug(f"Statistical test failed for {metric_name}: {e}")
                        # Provide fallback values
                        overall_stats_result = {
                            'control': control_values.mean(),
                            'experiment': treatment_values.mean(),
                            'diff': treatment_values.mean() - control_values.mean(),
                            'prop_diff': ((treatment_values.mean() - control_values.mean()) / control_values.mean()) if control_values.mean() != 0 else 0,
                            'pval': 0.5
                        }
                    
                    # Perform switching comparison statistical test (Current vs Optimal)
                    switching_p_value = 0.5  # Default value
                    if switching_json_path and current_switching_values and optimal_values:
                        try:
                            # Check if current and optimal values are identical (would cause issues)
                            if len(set(current_switching_values)) == 1 and len(set(optimal_values)) == 1:
                                switching_p_value = 1.0  # No difference
                            else:
                                with warnings.catch_warnings():
                                    warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
                                    # Use scipy directly to get raw p-value with full precision
                                    from scipy.stats import ttest_rel
                                    t_stat, raw_p_value = ttest_rel(optimal_values, current_switching_values)
                                    switching_p_value = float(raw_p_value)
                                
                                # Log debug info for switching p-values
                                current_mean = sum(current_switching_values)/len(current_switching_values)
                                optimal_mean = sum(optimal_values)/len(optimal_values)
                                logger.info(f"🔬 Switching analysis for {metric_name}:")
                                logger.info(f"   Current_Switching mean: {current_mean:.6f}")
                                logger.info(f"   Optimal_Switching mean: {optimal_mean:.6f}")
                                logger.info(f"   Raw P-value: {raw_p_value}")
                                logger.info(f"   T-statistic: {t_stat}")
                                
                                if switching_p_value < 0.001:
                                    logger.info(f"   ⚡ Extremely significant difference detected (p < 0.001)")
                        except Exception as e:
                            logger.debug(f"Failed to calculate switching p-value for {metric_name}: {e}")
                            switching_p_value = 0.5
                    
                    # Extract overall results with safe conversion to numeric
                    control_mean = overall_stats_result.get('control') or control_values.mean()
                    treatment_mean = overall_stats_result.get('experiment') or treatment_values.mean()
                    overall_diff = overall_stats_result.get('diff') or (treatment_mean - control_mean)
                    overall_prop_diff = overall_stats_result.get('prop_diff') or ((overall_diff / control_mean) if control_mean != 0 else 0)
                    overall_p_value = overall_stats_result.get('pval') or 0.5
                    
                    # Calculate overall variance
                    overall_control_var = control_values.var() if len(control_values) > 1 else 0.0
                    overall_treatment_var = treatment_values.var() if len(treatment_values) > 1 else 0.0
                    
                    # Ensure all overall values are numeric (handle string "Inf" cases)
                    control_mean = self._safe_numeric(control_mean)
                    treatment_mean = self._safe_numeric(treatment_mean)
                    overall_diff = self._safe_numeric(overall_diff)
                    overall_prop_diff = self._safe_numeric(overall_prop_diff)
                    overall_p_value = self._safe_numeric(overall_p_value)
                    overall_control_var = self._safe_numeric(overall_control_var)
                    overall_treatment_var = self._safe_numeric(overall_treatment_var)
                    
                    change = ((treatment_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0
                    
                    # Calculate segment-level statistics
                    segment_stats = []
                    for segment in segments:
                        segment_data = self._get_segment_data(segment, segment_column, df_to_use)
                        seg_control = segment_data[control_col].dropna()
                        seg_treatment = segment_data[treatment_col].dropna()
                        
                        if len(seg_control) > 0 and len(seg_treatment) > 0:
                            n = min(len(seg_control), len(seg_treatment))
                            
                            # Use proper statistical testing via tdiff function
                            # Convert pandas Series to lists for the statistical test
                            control_values_list = seg_control.tolist()
                            treatment_values_list = seg_treatment.tolist()
                            
                            # Perform statistical test - use the paired_test parameter
                            stats_result = {}
                            try:
                                # Suppress scipy warnings for small sample sizes or identical values
                                import warnings
                                with warnings.catch_warnings():
                                    warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
                                    stats_result = tdiff(control_values_list, treatment_values_list, paired=paired_test)
                            except Exception as e:
                                logger.debug(f"Statistical test failed for segment {segment} in {metric_name}: {e}")
                                # Provide fallback values
                                stats_result = {
                                    'control': seg_control.mean(),
                                    'experiment': seg_treatment.mean(),
                                    'diff': seg_treatment.mean() - seg_control.mean(),
                                    'prop_diff': ((seg_treatment.mean() - seg_control.mean()) / seg_control.mean()) if seg_control.mean() != 0 else 0,
                                    'pval': 0.5
                                }
                            
                            # Extract results from statistical test with safe conversion to numeric
                            seg_control_mean = stats_result.get('control') or seg_control.mean()
                            seg_treatment_mean = stats_result.get('experiment') or seg_treatment.mean()
                            seg_diff = stats_result.get('diff')
                            if seg_diff is None:
                                seg_diff = seg_treatment_mean - seg_control_mean
                            
                            seg_prop_diff = stats_result.get('prop_diff')
                            if seg_prop_diff is None:
                                seg_prop_diff = (seg_diff / seg_control_mean) if seg_control_mean != 0 else 0
                            
                            seg_p_value = stats_result.get('pval') or 0.5  # Use calculated p-value or default to 0.5
                            
                            # Calculate variance for control and treatment
                            seg_control_var = seg_control.var() if len(seg_control) > 1 else 0.0
                            seg_treatment_var = seg_treatment.var() if len(seg_treatment) > 1 else 0.0
                            
                            # Ensure all values are numeric (handle string "Inf" cases)
                            seg_control_mean = self._safe_numeric(seg_control_mean)
                            seg_treatment_mean = self._safe_numeric(seg_treatment_mean)
                            seg_diff = self._safe_numeric(seg_diff)
                            seg_prop_diff = self._safe_numeric(seg_prop_diff)
                            seg_p_value = self._safe_numeric(seg_p_value)
                            seg_control_var = self._safe_numeric(seg_control_var)
                            seg_treatment_var = self._safe_numeric(seg_treatment_var)
                            
                            segment_display = str(segment).replace('_', ' ').title() if segment != 'All Data' else 'All Data'
                            
                            segment_stats.append({
                                'segment_display': segment_display,
                                'n': n,
                                'control_mean': seg_control_mean,
                                'treatment_mean': seg_treatment_mean,
                                'control_var': seg_control_var,
                                'treatment_var': seg_treatment_var,
                                'diff': seg_diff,
                                'prop_diff': seg_prop_diff,
                                'p_value': seg_p_value
                            })
                    
                    # Sort segments by Diff (highest gain to lowest gain)
                    segment_stats.sort(key=lambda x: x['diff'], reverse=True)
                    
                    metric_performance.append({
                        'metric_name': metric_name,
                        'control_col': control_col,
                        'treatment_col': treatment_col,
                        'control_mean': control_mean,
                        'treatment_mean': treatment_mean,
                        'control_var': overall_control_var,
                        'treatment_var': overall_treatment_var,
                        'optimal_mean': overall_optimal_mean,
                        'current_switching_mean': overall_current_switching_mean,
                        'switching_p_value': switching_p_value,
                        'percent_change': change,
                        'overall_diff': overall_diff,
                        'overall_prop_diff': overall_prop_diff,
                        'overall_p_value': overall_p_value,
                        'segment_stats': segment_stats
                    })
        
        # Sort by performance gain (highest to lowest)
        metric_performance.sort(key=lambda x: x['percent_change'], reverse=True)
        
        return {
            'total_utterances': len(df_to_use),
            'segments': segments,
            'metrics_count': len(metric_performance),
            'metric_performance': metric_performance
        }
    
    def _safe_numeric(self, value):
        """
        Safely convert a value to numeric, handling string cases like 'Inf' and '-Inf'.
        
        Args:
            value: The value to convert
            
        Returns:
            float: Numeric value, with special handling for infinity cases
        """
        if value is None:
            return 0.0
        if isinstance(value, str):
            if value == "Inf":
                return float('inf')
            elif value == "-Inf":
                return float('-inf')
            else:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _get_segment_data(self, segment, segment_column: Optional[str] = None, df_to_use: Optional[pd.DataFrame] = None):
        """Get data for a specific segment."""
        if df_to_use is None:
            assert self.df is not None, "DataFrame should be loaded"
            df_to_use = self.df
            
        if segment_column and segment_column in df_to_use.columns:
            return df_to_use[df_to_use[segment_column] == segment]
        else:
            return df_to_use


class MetricsComparisonMarkdownGenerator:
    """
    A class to generate markdown reports from structured comparison results.
    General purpose markdown formatter for metric comparison analysis.
    """
    
    def __init__(self, analysis_results: Dict[str, Any], report_title: str = "Metrics by Segment"):
        """
        Initialize the markdown generator.
        
        Args:
            analysis_results: Structured results from MetricsAnalyzer
            report_title: Title for the markdown report
        """
        self.results = analysis_results
        self.report_title = report_title
    
    def generate_report(self) -> str:
        """
        Generate a complete markdown report.
        
        Returns:
            str: Formatted markdown report
        """
        markdown_lines = []
        
        # Header and summary
        markdown_lines.extend(self._generate_header())
        markdown_lines.extend(self._generate_summary())
        
        # Overall summary table (across all segments)
        markdown_lines.extend(self._generate_overall_summary_table())
        
        # Switching strategy comparison table (if switching data available)
        markdown_lines.extend(self._generate_switching_comparison_table())
        
        # Individual metric sections
        for metric_data in self.results['metric_performance']:
            markdown_lines.extend(self._generate_metric_section(metric_data))
        
        return "\n".join(markdown_lines)
    
    def _generate_header(self) -> list:
        """Generate the report header."""
        return [
            f"# {self.report_title}",
            ""
        ]
    
    def _generate_summary(self) -> list:
        """Generate the summary section."""
        # Determine if we're using reasoning classes based on the report title
        using_reasoning_classes = "Reasoning Class" in self.report_title
        segment_label = "reasoning classes" if using_reasoning_classes else "segments"
        
        return [
            "## Summary",
            f"- **Total utterances**: {self.results['total_utterances']}",
            f"- **Total {segment_label}**: {len(self.results['segments'])}",
            f"- **Metrics analyzed**: {self.results['metrics_count']}",
            ""
        ]
    
    def _generate_overall_summary_table(self) -> list:
        """Generate an overall summary table across all segments."""
        # Determine if we're using reasoning classes based on the report title
        using_reasoning_classes = "Reasoning Class" in self.report_title
        segment_label = "reasoning classes" if using_reasoning_classes else "segments"
        
        lines = [
            f"## Overall Performance Summary (across all {segment_label})",
            "",
            "| Metric | Control | Control Var | Treatment | Treatment Var | Diff | Prop diff | P |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"
        ]
        
        # Calculate overall metrics across all segments for each metric
        for metric_data in self.results['metric_performance']:
            # Get the overall means and statistics from the metric data
            control_mean = metric_data['control_mean']
            treatment_mean = metric_data['treatment_mean']
            control_var = metric_data.get('control_var', 0.0)
            treatment_var = metric_data.get('treatment_var', 0.0)
            overall_diff = metric_data.get('overall_diff', treatment_mean - control_mean)
            overall_prop_diff = metric_data.get('overall_prop_diff', (overall_diff / control_mean) if control_mean != 0 else 0)
            overall_p_value = metric_data.get('overall_p_value', 0.5)
            
            lines.append(
                f"| {metric_data['metric_name'].upper()} | {control_mean:.3f} | {control_var:.6f} | "
                f"{treatment_mean:.3f} | {treatment_var:.6f} | {overall_diff:+.3f} | {overall_prop_diff:+.3f} | "
                f"{overall_p_value:.3f} |"
            )
        
        # Add footnote explaining the table
        lines.extend([
            "",
            f"*Note: This table shows overall performance comparison between control and treatment across all {segment_label}.*",
            "",
            "---",
            ""
        ])
        return lines
    
    def _generate_switching_comparison_table(self) -> list:
        """Generate a comparison table between Current Switching and Optimal Switching."""
        # Check if we have current switching data
        has_switching_data = any(
            'current_switching_mean' in metric_data 
            for metric_data in self.results['metric_performance']
        )
        
        if not has_switching_data:
            return []
        
        lines = [
            "## Switching Strategy Comparison",
            "",
            "| Metric | Current_Switching | Optimal_Switching | Diff | Prop diff | P |",
            "| --- | --- | --- | --- | --- | --- |"
        ]
        
        # Calculate comparison for each metric
        for metric_data in self.results['metric_performance']:
            current_switching_mean = metric_data.get('current_switching_mean', 0)
            optimal_switching_mean = metric_data.get('optimal_mean', 0)
            
            # Calculate difference and proportional difference
            diff = optimal_switching_mean - current_switching_mean
            prop_diff = (diff / current_switching_mean) if current_switching_mean != 0 else 0
            
            # Get the p-value from switching comparison if available
            switching_p_value = metric_data.get('switching_p_value', 0.5)
            
            # Format p-value with appropriate precision
            if switching_p_value == 0.0:
                p_value_str = "< 1E-300"  # Indicate extremely small p-value when it underflows to zero
            elif switching_p_value < 1e-10:
                p_value_str = f"{switching_p_value:.3E}"  # Scientific notation with uppercase E for extremely small values
            elif switching_p_value < 0.001:
                p_value_str = f"{switching_p_value:.6E}"  # More precision with uppercase E for very small values
            else:
                p_value_str = f"{switching_p_value:.3f}"
            
            lines.append(
                f"| {metric_data['metric_name'].upper()} | {current_switching_mean:.3f} | "
                f"{optimal_switching_mean:.3f} | {diff:+.3f} | {prop_diff:+.3f} | "
                f"{p_value_str} |"
            )
        
        lines.extend([
            "",
            "*Note: Current_Switching shows performance using the current switching strategy, while Optimal_Switching represents the best achievable performance using our data-driven optimized switching.*",
            "",
            "---",
            ""
        ])
        return lines
    
    def _generate_metric_section(self, metric_data: Dict[str, Any]) -> list:
        """Generate a section for a specific metric."""
        # Determine if we're using reasoning classes based on the report title
        using_reasoning_classes = "Reasoning Class" in self.report_title
        segment_column_header = "Reasoning Class" if using_reasoning_classes else "Segment"
        
        lines = [
            f"### {metric_data['metric_name'].upper()}",
            "",
            f"| {segment_column_header} | N | Control | Control Var | Treatment | Treatment Var | Diff | Prop diff | P |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        ]
        
        # Add data rows
        for stat in metric_data['segment_stats']:
            control_var = stat.get('control_var', 0.0)
            treatment_var = stat.get('treatment_var', 0.0)
            lines.append(
                f"| {stat['segment_display']} | {stat['n']} | {stat['control_mean']:.3f} | {control_var:.6f} | "
                f"{stat['treatment_mean']:.3f} | {treatment_var:.6f} | {stat['diff']:+.3f} | {stat['prop_diff']:+.3f} | "
                f"{stat['p_value']:.3f} |"
            )
        
        lines.append("")
        return lines

class MetricsDataReader:
    """
    A class to read metrics data from standardized folder structure.
    Focuses on data loading and basic file operations.
    """
    
    def __init__(self, 
                 job_path: Optional[str] = None,
                 metrics_file_name: str = "all_metrics_paired.csv",
                 metrics_relative_path: str = "offline_scorecard_generator_output"):
        """
        Initialize the MetricsReader.
        
        Args:
            job_path: Full path to the job folder (e.g., "/path/to/100335_metrics"). 
                     If None, must be set before using the class methods.
            metrics_file_name: Name of the main metrics CSV file (default: "all_metrics_paired.csv")
            metrics_relative_path: Relative path from job folder to metrics files 
                                 (default: "offline_scorecard_generator_output")
        """
        self.job_path = Path(job_path) if job_path else None
        self.metrics_file_name = metrics_file_name
        self.metrics_relative_path = metrics_relative_path
        
        if self.job_path:
            self._validate_job_path()
        else:
            logger.info("MetricsReader initialized without job_path. Set job_path before using.")
    
    def set_job_path(self, job_path: str) -> None:
        """
        Set the full path to the job folder.
        
        Args:
            job_path: Full path to the job folder (e.g., "/path/to/100335_metrics")
        """
        self.job_path = Path(job_path)
        self._validate_job_path()
        logger.info(f"Job path set to: {self.job_path}")
    
    def set_metrics_file_name(self, file_name: str) -> None:
        """
        Set the name of the main metrics file.
        
        Args:
            file_name: Name of the metrics CSV file
        """
        self.metrics_file_name = file_name
        logger.info(f"Metrics file name set to: {self.metrics_file_name}")
    
    def set_metrics_relative_path(self, relative_path: str) -> None:
        """
        Set the relative path from job folder to metrics files.
        
        Args:
            relative_path: Relative path from job folder to metrics files
        """
        self.metrics_relative_path = relative_path
        logger.info(f"Metrics relative path set to: {self.metrics_relative_path}")
    
    def _validate_job_path(self) -> None:
        """
        Validate that the job path exists and log information about it.
        """
        if not self.job_path or not self.job_path.exists():
            logger.warning(f"Job path does not exist: {self.job_path}")
            logger.info("Make sure the job path points to the job folder (e.g., '/path/to/100335_metrics')")
        else:
            logger.debug(f"Job path validated: {self.job_path}")
            
            # Check if it has the expected structure
            metrics_dir = self.job_path / self.metrics_relative_path
            metrics_file = metrics_dir / self.metrics_file_name
            
            if metrics_dir.exists():
                logger.debug(f"Found metrics directory: {metrics_dir}")
                csv_files = list(metrics_dir.glob("*.csv"))
                # logger.info(f"Found {len(csv_files)} CSV files in metrics directory")
            else:
                logger.warning(f"Metrics directory not found: {metrics_dir}")
            
            if metrics_file.exists():
                logger.debug(f"Found main metrics file: {metrics_file}")
            else:
                logger.warning(f"Main metrics file not found: {metrics_file}")
    
    def _check_job_structure(self) -> bool:
        """
        Check if the job folder has the expected structure.
        
        Returns:
            True if structure is valid, False otherwise
        """
        if not self.job_path:
            raise ValueError("Job path not set. Use set_job_path() first.")
        
        metrics_dir = self.job_path / self.metrics_relative_path
        metrics_file = metrics_dir / self.metrics_file_name
        
        if not self.job_path.exists():
            logger.error(f"Job folder not found: {self.job_path}")
            return False
        
        if not metrics_dir.exists():
            logger.error(f"Metrics directory not found: {metrics_dir}")
            return False
        
        if not metrics_file.exists():
            logger.error(f"Main metrics file not found: {metrics_file}")
            return False
        
        return True
    
    def read_metrics(self) -> pd.DataFrame:
        """
        Read the main metrics file.
        
        Returns:
            pandas.DataFrame: The loaded metrics data
            
        Raises:
            ValueError: If job path is not set
            FileNotFoundError: If the metrics file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
            pd.errors.ParserError: If the file can't be parsed as CSV
        """
        if not self.job_path:
            raise ValueError("Job path not set. Use set_job_path() or initialize with job_path.")
        
        # Validate job structure first
        if not self._check_job_structure():
            raise FileNotFoundError(f"Invalid job structure for {self.job_path}")
        
        # Construct the file path
        metrics_file_path = (self.job_path / 
                           self.metrics_relative_path / self.metrics_file_name)
        
        logger.info(f"Reading metrics from: {metrics_file_path}")
        
        try:
            # Read the CSV file with low_memory=False to handle mixed column types
            df = pd.read_csv(metrics_file_path, low_memory=False)
            logger.info(f"Successfully loaded metrics data: {df.shape[0]} rows, {df.shape[1]} columns")
            
            return df
            
        except pd.errors.EmptyDataError:
            logger.error(f"The file {metrics_file_path} is empty")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file {metrics_file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading {metrics_file_path}: {e}")
            raise
    
    def list_available_files(self) -> list:
        """
        List all available CSV files in the metrics output folder.
        
        Returns:
            list: List of available CSV file names
            
        Raises:
            ValueError: If job path is not set
        """
        if not self.job_path:
            raise ValueError("Job path not set. Use set_job_path() or initialize with job_path.")
        
        output_dir = (self.job_path / self.metrics_relative_path)
        
        if not output_dir.exists():
            logger.warning(f"Output directory not found: {output_dir}")
            return []
        
        csv_files = [f.name for f in output_dir.glob("*.csv")]
        csv_files.sort()
        
        logger.info(f"Found {len(csv_files)} CSV files in {output_dir}")
        return csv_files
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the main metrics file.
        
        Returns:
            dict: Summary information about the metrics
            
        Raises:
            ValueError: If job path is not set
        """
        if not self.job_path:
            raise ValueError("Job path not set. Use set_job_path() or initialize with job_path.")
        
        df = self.read_metrics()
        
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_utterances": df['utterance'].nunique() if 'utterance' in df.columns else None,
            "sample_data": df.head(3).to_dict('records') if not df.empty else []
        }
        
        return summary


def generate_optimal_switching_json(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate optimal switching JSON output organized by metric.
    
    Args:
        analysis_results: Analysis results from extract_metric_pairs
        
    Returns:
        dict: JSON structure with reasoning classes categorized by performance
    """
    optimal_switching = {}
    
    for metric_data in analysis_results['metric_performance']:
        metric_name = metric_data['metric_name'].upper()
        
        # Initialize metric entry
        optimal_switching[metric_name] = {
            "AllowedReasoningCapabilities": [],
            "ReasoningClasses": []
        }
        
        # Process each reasoning class for this metric
        for segment_stat in metric_data['segment_stats']:
            reasoning_class = segment_stat['segment_display']

            capability = reasoning_class.split(' 1 ')[0] if ' 1 ' in reasoning_class else reasoning_class.split(' 2 ')[0]

            if capability not in optimal_switching[metric_name]["AllowedReasoningCapabilities"]:
                optimal_switching[metric_name]["AllowedReasoningCapabilities"].append(capability)

            # Skip 'All Data' entry
            if reasoning_class == 'All Data':
                continue
                
            control_mean = segment_stat['control_mean']
            treatment_mean = segment_stat['treatment_mean']
            
            # If treatment - control >= 0.02, add to ReasoningClasses
            if treatment_mean - control_mean >= 0.02:
                optimal_switching[metric_name]["ReasoningClasses"].append(reasoning_class)
        
        # Sort the lists alphabetically for consistency
        optimal_switching[metric_name]["AllowedReasoningCapabilities"].sort()
        optimal_switching[metric_name]["ReasoningClasses"].sort()
    
    return optimal_switching


def list_csv_files(
    job_path: str,
    metrics_file_name: str = "all_metrics_paired.csv",
    metrics_relative_path: str = "offline_scorecard_generator_output"
) -> None:
    """
    List all available CSV files in the metrics job folder.
    
    Args:
        job_path: Full path to the job folder (e.g., "/path/to/100335_metrics")
        metrics_file_name: Name of the main metrics CSV file
        metrics_relative_path: Relative path from job folder to metrics files
    
    Example:
        python -m projects.seval.get_seval_metrics list_csv_files \\
            --job_path="c:/path/to/100335_metrics"
    """
    try:
        reader = MetricsDataReader(
            job_path=job_path,
            metrics_file_name=metrics_file_name,
            metrics_relative_path=metrics_relative_path
        )
        
        job_name = Path(job_path).name
        available_files = reader.list_available_files()
        
        print(f"\n=== Available Files in {job_name} ===")
        print(f"Total files: {len(available_files)}")
        if available_files:
            print("Files:")
            for i, file_name in enumerate(available_files, 1):
                print(f"  {i}. {file_name}")
        else:
            print("No CSV files found")
            
    except Exception as e:
        logger.error(f"Error listing files in {job_path}: {e}")
        sys.exit(1)


def get_metrics_summary(
    job_path: str,
    add_reasoning_class: bool = False,
    reasoning_json_path: Optional[str] = None,
    metrics_file_name: str = "all_metrics_paired.csv",
    metrics_relative_path: str = "offline_scorecard_generator_output"
) -> None:
    """
    Generate comprehensive summary of the metrics data.
    
    Args:
        job_path: Full path to the job folder (e.g., "/path/to/100335_metrics")
        add_reasoning_class: Add reasoning class column based on utterance matching
        reasoning_json_path: Path to JSON file containing reasoning class mappings
        metrics_file_name: Name of the main metrics CSV file
        metrics_relative_path: Relative path from job folder to metrics files
    
    Example:
        python -m projects.seval.get_seval_metrics get_metrics_summary \\
            --job_path="c:/path/to/100335_metrics"
    """
    try:
        analyzer = MetricsAnalyzer(
            job_path=job_path,
            metrics_file_name=metrics_file_name,
            metrics_relative_path=metrics_relative_path
        )
        
        df = analyzer.get_dataframe(
            add_reasoning_class=add_reasoning_class,
            reasoning_json_path=reasoning_json_path
        )
        
        job_name = Path(job_path).name
        print(f"\n=== Comprehensive Summary for {job_name} ===")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        
        if 'utterance' in df.columns:
            print(f"Unique utterances: {df['utterance'].nunique()}")
        
        if add_reasoning_class and 'reasoning_class' in df.columns:
            mapped_count = (df['reasoning_class'] != '').sum()
            unique_classes = df[df['reasoning_class'] != '']['reasoning_class'].nunique()
            print(f"Reasoning classes: {mapped_count}/{len(df)} utterances mapped to {unique_classes} unique classes")
        
        missing_values_total = df.isnull().sum().sum()
        print(f"Missing values (total): {missing_values_total}")
        
        type_counts = {}
        for dtype in df.dtypes.values:
            type_name = str(dtype)
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        print(f"Data types: {dict(type_counts)}")
        
    except Exception as e:
        logger.error(f"Error generating summary for {job_path}: {e}")
        sys.exit(1)


def generate_citedcg_report(
    job_path: str,
    add_reasoning_class: bool = False,
    reasoning_json_path: Optional[str] = None,
    switching_json_path: Optional[str] = None,
    output_optimal_switching: bool = False,
    metrics_file_name: str = "all_metrics_paired.csv",
    metrics_relative_path: str = "offline_scorecard_generator_output",
    paired_test: bool = True
) -> None:
    """
    Generate CiteDCG report in markdown format and save to file.
    
    Generates a performance comparison report sorted by performance gain.
    If add_reasoning_class=True, segments by reasoning class; otherwise segments by 'segment 2' column.
    
    Args:
        job_path: Full path to the job folder (e.g., "/path/to/100335_metrics")
        add_reasoning_class: Add reasoning class column based on utterance matching
        reasoning_json_path: Path to JSON file containing reasoning class mappings
        switching_json_path: Path to JSON file containing switching strategy
        output_optimal_switching: Generate optimal switching JSON output
        metrics_file_name: Name of the main metrics CSV file
        metrics_relative_path: Relative path from job folder to metrics files
        paired_test: Use paired t-test (True) or independent t-test (False)
    
    Example:
        python -m projects.seval.get_seval_metrics generate_citedcg_report \\
            --job_path="c:/path/to/100335_metrics" \\
            --add_reasoning_class=True \\
            --reasoning_json_path="path/to/data.json"
    """
    try:
        job_name = Path(job_path).name
        print(f"\n=== Generating CiteDCG Report for {job_name} ===")
        
        # Define target CiteDCG metrics
        target_columns = [
            'citedcg_one_centric_control', 'citedcg_one_centric_treatment',
            'citedcg_num_enterprise_cites_control', 'citedcg_num_enterprise_cites_treatment'
        ]
        
        # Initialize analyzer with job configuration
        analyzer = MetricsAnalyzer(
            job_path=job_path,
            metrics_file_name=metrics_file_name,
            metrics_relative_path=metrics_relative_path
        )
        
        # Add reasoning class if requested
        if add_reasoning_class and reasoning_json_path:
            df_with_reasoning = analyzer.get_dataframe(
                add_reasoning_class=True,
                reasoning_json_path=reasoning_json_path
            )
            
            # Extract metrics comparison results
            analysis_results = analyzer.extract_metric_pairs(
                target_columns,
                segment_column='reasoning_class',
                paired_test=paired_test,
                use_enhanced_df=df_with_reasoning,
                switching_json_path=switching_json_path
            )
        else:
            # Extract metrics comparison results with segment 2 analysis
            analysis_results = analyzer.extract_metric_pairs(
                target_columns,
                segment_column='segment 2',
                paired_test=paired_test,
                switching_json_path=switching_json_path
            )
        
        # Generate markdown report
        if add_reasoning_class and reasoning_json_path:
            report_title = "CiteDCG Metrics by Reasoning Class"
        else:
            report_title = "CiteDCG Metrics by Segment"
        
        markdown_generator = MetricsComparisonMarkdownGenerator(
            analysis_results,
            report_title=report_title
        )
        report_content = markdown_generator.generate_report()
        
        # Save to file with job-specific name
        job_base_name = job_name.replace('_metrics', '') if job_name.endswith('_metrics') else job_name
        output_file = Path(job_path).parent / f"{job_base_name}_CiteDCG_Report.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ CiteDCG report saved to: {output_file}")
        if add_reasoning_class and reasoning_json_path:
            print(f"📊 Report contains {analysis_results['metrics_count']} specific CiteDCG metrics across {len(analysis_results['segments'])} reasoning classes")
        else:
            print(f"📊 Report contains {analysis_results['metrics_count']} specific CiteDCG metrics across {len(analysis_results['segments'])} segments")
        
        # Generate optimal switching JSON output if requested
        if output_optimal_switching and add_reasoning_class and reasoning_json_path:
            optimal_switching_output = generate_optimal_switching_json(analysis_results)
            json_output_file = Path(job_path).parent / f"{job_base_name}_Optimal_Switching.json"
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(optimal_switching_output, f, indent=2)
            print(f"📄 Optimal switching JSON saved to: {json_output_file}")
        
        # Print performance summary to console
        print(f"\nPerformance Summary (sorted by gain):")
        for rank, metric_data in enumerate(analysis_results['metric_performance'][:5], 1):
            metric_name = metric_data['metric_name'].upper()
            change = metric_data['percent_change']
            direction = "🔼" if change > 0 else "🔽" if change < 0 else "➡️"
            print(f"  {rank}. {metric_name}: {change:+.1f}% {direction}")
        
        if analysis_results['metrics_count'] > 5:
            print(f"  ... and {analysis_results['metrics_count'] - 5} more metrics")
            
    except Exception as e:
        logger.error(f"Error generating CiteDCG report for {job_path}: {e}")
        sys.exit(1)


def read_metrics(
    job_path: str,
    add_reasoning_class: bool = False,
    reasoning_json_path: Optional[str] = None,
    metrics_file_name: str = "all_metrics_paired.csv",
    metrics_relative_path: str = "offline_scorecard_generator_output"
) -> None:
    """
    Read and display basic information about the metrics data.
    
    Args:
        job_path: Full path to the job folder (e.g., "/path/to/100335_metrics")
        add_reasoning_class: Add reasoning class column based on utterance matching
        reasoning_json_path: Path to JSON file containing reasoning class mappings
        metrics_file_name: Name of the main metrics CSV file
        metrics_relative_path: Relative path from job folder to metrics files
    
    Example:
        python -m projects.seval.get_seval_metrics read_metrics \\
            --job_path="c:/path/to/100335_metrics"
    """
    try:
        analyzer = MetricsAnalyzer(
            job_path=job_path,
            metrics_file_name=metrics_file_name,
            metrics_relative_path=metrics_relative_path
        )
        
        df = analyzer.get_dataframe(
            add_reasoning_class=add_reasoning_class,
            reasoning_json_path=reasoning_json_path
        )
        
        job_name = Path(job_path).name
        print(f"\n=== Metrics Summary for {job_name} ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        if add_reasoning_class and 'reasoning_class' in df.columns:
            mapped_count = (df['reasoning_class'] != '').sum()
            unique_classes = df[df['reasoning_class'] != '']['reasoning_class'].nunique()
            print(f"Reasoning classes: {mapped_count}/{len(df)} utterances mapped to {unique_classes} unique classes")
            
            if unique_classes > 0:
                print("\nTop 5 reasoning classes:")
                class_counts = df[df['reasoning_class'] != '']['reasoning_class'].value_counts().head(5)
                for class_name, count in class_counts.items():
                    print(f"  {class_name}: {count} utterances")
        
        if 'utterance' in df.columns:
            print(f"Unique utterances: {df['utterance'].nunique()}")
            print(f"\nFirst 3 utterances:")
            for i, utterance in enumerate(df['utterance'].head(3), 1):
                reasoning_info = ""
                if add_reasoning_class and 'reasoning_class' in df.columns:
                    reasoning_class = df.iloc[i-1]['reasoning_class']
                    if reasoning_class:
                        reasoning_info = f" [Class: {reasoning_class}]"
                print(f"  {i}. {utterance}{reasoning_info}")
        
        print(f"\nColumn names (first 10):")
        for i, col in enumerate(df.columns[:10], 1):
            print(f"  {i}. {col}")
        
        if len(df.columns) > 10:
            print(f"  ... and {len(df.columns) - 10} more columns")
        
        print(f"\n=== Available Files in {job_name} ===")
        reader = MetricsDataReader(
            job_path=job_path,
            metrics_file_name=metrics_file_name,
            metrics_relative_path=metrics_relative_path
        )
        available_files = reader.list_available_files()
        print(f"Total files: {len(available_files)}")
        print("First 10 files:")
        for i, file_name in enumerate(available_files[:10], 1):
            print(f"  {i}. {file_name}")
        
        if len(available_files) > 10:
            print(f"  ... and {len(available_files) - 10} more files")
            
    except Exception as e:
        logger.error(f"Error reading metrics from {job_path}: {e}")
        sys.exit(1)


def extract_per_result_citedcg(
    metrics_folder: str,
    experiment: str,
    output_file: str,
    utterance: Optional[str] = None
) -> None:
    """
    Extract per-result CiteDCG scores from SEVAL results.json.
    
    Extracts individual search result scores (as opposed to aggregated CSV metrics).
    
    This function reads CiteDCG scores from the consolidated results file
    and outputs them in JSONL format (one JSON object per line).
    Provides per-result scores with optional filtering by utterance.
    
    Args:
        metrics_folder: Metrics folder name (e.g., "130949_metrics")
        experiment: Experiment name - "control" or "treatment"
        output_file: Path for output JSONL file (one JSON object per line)
        utterance: Optional - filter results by utterance substring
    
    Example usage:
        # Extract all per-result CiteDCG scores for control experiment (module calling - preferred)
        python -m projects.seval.get_seval_metrics extract_per_result_citedcg \\
            --metrics_folder="130949_metrics" \\
            --experiment="control" \\
            --output_file="scores.json"
        
        # Or direct file execution (also works)
        python get_seval_metrics.py extract_per_result_citedcg \\
            --metrics_folder="130949_metrics" \\
            --experiment="control" \\
            --output_file="scores.json"
        
        # Extract for treatment experiment
        python -m projects.seval.get_seval_metrics extract_per_result_citedcg \\
            --metrics_folder="130949_metrics" \\
            --experiment="treatment" \\
            --output_file="scores.json"
        
        # Filter by specific utterance
        python -m projects.seval.get_seval_metrics extract_per_result_citedcg \\
            --metrics_folder="130949_metrics" \\
            --experiment="control" \\
            --utterance="scorecard" \\
            --output_file="scorecard_scores.json"
    """
    try:
        extractor = PerResultCiteDCGExtractor()
        extractor.extract(
            metrics_folder=metrics_folder,
            experiment=experiment,
            output_file=output_file,
            utterance=utterance
        )
    except Exception as e:
        logger.error(f"Error extracting CiteDCG scores: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Use fire for command line arguments
    # Exposes all modular functions as commands
    try:
        fire.Fire({
            'list_csv_files': list_csv_files,
            'get_metrics_summary': get_metrics_summary,
            'generate_citedcg_report': generate_citedcg_report,
            'read_metrics': read_metrics,
            'extract_per_result_citedcg': extract_per_result_citedcg
        })
    except FireExit as e:
        # Handle Fire's exit (including --help) gracefully in debug mode
        # FireExit with code 0 means successful exit (like --help)
        # FireExit with non-zero code means error exit
        sys.exit(e.code)
