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
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import fire
import pandas as pd
from fire.core import FireExit

# Import from the utils package (sibling folder)
from utils.statistics_utils import tdiff

# Add parent directory to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    
    def __init__(self, verbose: bool = True):
        """
        Initialize extractor.
        
        Uses base path relative to this module: module_dir/seval_data/
        This ensures paths work regardless of current working directory.
        
        Args:
            verbose: If True, show detailed statistics. If False, minimal output.
        """
        # Use path relative to this module location (projects/seval/)
        self.module_dir = Path(__file__).parent
        self.base_path = self.module_dir / "seval_data"
        self.verbose = verbose
    
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
            return 0
        
        print(f"Extracting CiteDCG scores for {experiment}...")
        if self.verbose:
            print(f"  → Reading from: {results_path}")
        logger.info(f"Reading CiteDCG data from: {results_path}")
        logger.info(f"  Metrics folder: {metrics_folder}, Experiment: {experiment}")
        if utterance:
            logger.info(f"  Filtering by utterance: '{utterance}'")
        
        # Load and parse JSONL (newline-delimited JSON, one object per line)
        all_data = []
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                all_data.append(obj)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Skipping invalid JSON on line {line_num}: {e}"
                            )
                
                if not all_data:
                    logger.error(
                        "Failed to parse any valid JSON objects from JSONL file"
                    )
                    return 0
        except Exception as e:
            logger.error(f"Failed to load results.json: {e}")
            return 0
        
        # Extract CiteDCG data from all JSON objects
        extracted_data = []
        for data_obj in all_data:
            extracted_data.extend(self._extract_cite_dcg_data(data_obj))
        
        if not extracted_data:
            logger.warning("No CiteDCG data found in results file")
            return 0
        
        # Verify multi-turn hop pattern
        if self.verbose:
            print("")
            print("=" * 60)
            print("MULTI-TURN HOP PATTERN VERIFICATION")
            print("=" * 60)
        verification = self._verify_multiturn_hop_pattern(all_data)
        
        if self.verbose:
            print(
                f"Total multi-turn conversations: "
                f"{verification['total_multi_turn']}"
            )
            print(
                f"  Last turn has hops: "
                f"{verification['last_turn_has_hops']} "
                f"({verification['last_turn_has_hops'] / max(1, verification['total_multi_turn']) * 100:.1f}%)"
            )
            print(
                f"  First turn has hops: "
                f"{verification['first_turn_has_hops']}"
            )
            print(
                f"  Middle turn has hops: "
                f"{verification['middle_turn_has_hops']}"
            )
            print(
                f"  Multiple turns with hops: "
                f"{verification['multiple_turns_with_hops']}"
            )
            
            if verification['pattern_by_turns']:
                print("")
                print("Pattern breakdown by # of turns:")
                for turns_key in sorted(verification['pattern_by_turns'].keys()):
                    patterns = verification['pattern_by_turns'][turns_key]
                    print(f"  {turns_key}:")
                    for pattern, count in sorted(
                        patterns.items(), key=lambda x: -x[1]
                    ):
                        print(f"    {pattern}: {count}")
            
            print("=" * 60)
            print("")
        
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
        
        # Calculate statistics from grouped data (utterances)
        stats = self._calculate_utterance_stats(grouped_data)
        
        if self.verbose:
            print("=" * 60)
            print("EXTRACTION STATISTICS:")
        total = stats["total"]
        if self.verbose:
            print(f"  Total utterances: {total}")
        
        with_scores = stats["with_scores"]
        with_scores_pct = (with_scores / total * 100) if total > 0 else 0
        
        empty = stats["empty_results"]
        no_searches = stats["no_searches"]
        errors = stats["errors_count"]
        
        # Always show utterances with scores
        print(
            f"  Utterances with CiteDCG scores: "
            f"{with_scores} ({with_scores_pct:.1f}%)"
        )
        
        # Calculate total without scores
        total_no_scores = empty + no_searches + errors
        no_scores_pct = (total_no_scores / total * 100) if total > 0 else 0
        
        # Always show total without scores
        print(
            f"  Utterances without scores: "
            f"{total_no_scores} ({no_scores_pct:.1f}%)"
        )
        
        # For verbose=True, show breakdown
        if self.verbose:
            empty_pct = (empty / total * 100) if total > 0 else 0
            print(
                f"    - Empty search results: "
                f"{empty} ({empty_pct:.1f}%)"
            )
            
            no_searches_pct = (no_searches / total * 100) if total > 0 else 0
            print(
                f"    - No searches: "
                f"{no_searches} ({no_searches_pct:.1f}%)"
            )
            
            errors_pct = (errors / total * 100) if total > 0 else 0
            print(
                f"    - Errors (non-empty results, no labels): "
                f"{errors} ({errors_pct:.1f}%)"
            )
        
        print("")  # Add blank line for separation
        
        # Output turn statistics (verbose only)
        if self.verbose and stats['turn_stats']:
            print("")
            print("  Utterance breakdown by # of turns:")
            print("    (Non-empty turn = turn with search results/CiteDCG scores)")
            total = stats['total']
            for num_turns in sorted(stats['turn_stats'].keys()):
                count = stats['turn_stats'][num_turns]
                ratio = (count / total * 100) if total > 0 else 0
                if num_turns == 1:
                    print(
                        f"    {num_turns} turn: {count} ({ratio:.1f}%)"
                    )
                else:
                    # Multi-turn = retries, only last turn has search results
                    print(
                        f"    {num_turns} turns (1 non-empty: last turn has results): "
                        f"{count} ({ratio:.1f}%)"
                    )
        
        # Error examples (verbose only)
        if self.verbose and stats['error_examples']:
            print("")
            print("  Error case examples:")
            for err in stats['error_examples'][:5]:  # Show first 5
                print(f"    - {err}")
            if len(stats['error_examples']) > 5:
                print(
                    f"    ... and {len(stats['error_examples']) - 5} more"
                )
            
            # Show which plugins have no DCG labels
            if stats['error_plugins']:
                print("")
                print("  Plugins without CiteDCG labels (error cases):")
                sorted_plugins = sorted(
                    stats['error_plugins'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for plugin, count in sorted_plugins:
                    print(f"    - {plugin}: {count} results")
        
        if self.verbose:
            print("=" * 60)
        
        self._write_json_output(grouped_data, output_file)
        logger.info(
            f"Wrote {len(grouped_data)} utterances "
            f"(JSONL rows) to {output_file}"
        )
        return len(grouped_data)
    
    def _calculate_utterance_stats(self, grouped_data: list) -> dict:
        """
        Calculate statistics about utterances from grouped data.
        
        Returns:
            dict with keys:
                - total: total number of utterances
                - with_scores: utterances with CiteDCG scores
                - empty_results: utterances with searches but empty results
                - no_searches: utterances without any searches
                - errors_count: utterances with results but no labels
                - error_examples: list of error utterance texts
                - error_plugins: dict mapping plugin_name -> count of results without labels
                - turn_stats: dict mapping num_turns -> count
                - multi_turn_with_hops: count of multi-turn utterances with hops
        """
        stats = {
            'total': len(grouped_data),
            'with_scores': 0,
            'empty_results': 0,
            'no_searches': 0,
            'errors_count': 0,
            'error_examples': [],
            'error_plugins': {},
            'turn_stats': {},
            'multi_turn_with_hops': 0
        }
        
        for utterance_data in grouped_data:
            has_scores = utterance_data.get('has_cite_dcg_scores', True)
            num_turns = utterance_data.get('num_turns', 1)
            non_empty_turns = utterance_data.get('non_empty_turns', 0)
            
            # Track turn statistics
            if num_turns not in stats['turn_stats']:
                stats['turn_stats'][num_turns] = 0
            stats['turn_stats'][num_turns] += 1
            
            # Track multi-turn utterances with hops
            if num_turns > 1 and non_empty_turns > 0:
                stats['multi_turn_with_hops'] += 1
            
            if has_scores:
                # Normal case: has scores
                stats['with_scores'] += 1
            else:
                # No scores - check reason
                reason = utterance_data.get('reason', '')
                
                if reason == 'no_search_results':
                    # Searches executed but returned empty results
                    stats['empty_results'] += 1
                elif reason == 'no_searches_executed':
                    # No searches performed
                    stats['no_searches'] += 1
                else:
                    # Check if any search has results (error case)
                    searches = utterance_data.get('searches', [])
                    has_results = any(
                        search.get('result_count', 0) > 0
                        for search in searches
                    )
                    if has_results:
                        # Error: has results but no scores
                        stats['errors_count'] += 1
                        utterance = utterance_data.get('utterance', 'Unknown')
                        stats['error_examples'].append(utterance)
                        
                        # Collect plugin names for error cases
                        for search in searches:
                            if search.get('result_count', 0) > 0:
                                plugin = search.get('plugin_name', 'Unknown')
                                if plugin not in stats['error_plugins']:
                                    stats['error_plugins'][plugin] = 0
                                stats['error_plugins'][plugin] += 1
        
        return stats
    
    def _calculate_extraction_stats(self, data: list) -> dict:
        """
        Calculate statistics about extracted data.
        
        Returns:
            dict with keys:
                - with_scores: count of results with CiteDCG scores
                - no_results: count without scores (empty search results)
                - no_searches: count without scores (no searches)
                - errors: list of utterances with non-empty results
                          but no CiteDCG labels
        """
        stats = {
            'with_scores': 0,
            'no_results': 0,
            'no_searches': 0,
            'errors': []
        }
        
        for item in data:
            has_scores = item.get('has_cite_dcg_scores', True)
            
            if has_scores:
                stats['with_scores'] += 1
            else:
                reason = item.get('reason', '')
                is_error = item.get('is_error', False)
                
                if is_error:
                    # Non-empty results but no CiteDCG labels - ERROR!
                    utterance = item.get('utterance', 'Unknown')
                    stats['errors'].append(utterance)
                elif reason == 'no_search_results':
                    stats['no_results'] += 1
                elif reason == 'no_searches_executed':
                    stats['no_searches'] += 1
        
        return stats
    
    def _verify_multiturn_hop_pattern(
        self, data_list: list
    ) -> dict:
        """
        Verify which turns have hops in multi-turn conversations.
        
        This checks the hypothesis that in multi-turn conversations,
        only the LAST turn has hops (orchestrationIterations),
        while previous turns are failed attempts.
        
        Args:
            data_list: List of raw DCG data entries
            
        Returns:
            dict with verification statistics
        """
        verification = {
            'total_multi_turn': 0,
            'last_turn_has_hops': 0,
            'first_turn_has_hops': 0,
            'middle_turn_has_hops': 0,
            'multiple_turns_with_hops': 0,
            'pattern_by_turns': {},  # num_turns -> pattern description
            'examples': []  # Store some examples for review
        }
        
        for data in data_list:
            evaluation_data = data.get("EvaluationData", {})
            turn_data = evaluation_data.get("turnData", [])
            num_turns = len(turn_data) if turn_data else 0
            
            # Only analyze multi-turn conversations
            if num_turns <= 1:
                continue
            
            verification['total_multi_turn'] += 1
            
            # Check which turns have orchestrationIterations (hops)
            turns_with_hops = []
            for turn_idx, turn in enumerate(turn_data):
                orchestration_iterations = turn.get(
                    "orchestrationIterations", []
                )
                if orchestration_iterations and len(
                    orchestration_iterations
                ) > 0:
                    turns_with_hops.append(turn_idx + 1)  # 1-indexed
            
            # Categorize the pattern
            num_turns_with_hops = len(turns_with_hops)
            
            if num_turns_with_hops == 0:
                pattern = "no_hops"
            elif num_turns_with_hops == 1:
                hop_turn = turns_with_hops[0]
                if hop_turn == num_turns:  # Last turn
                    verification['last_turn_has_hops'] += 1
                    pattern = "last_turn_only"
                elif hop_turn == 1:  # First turn
                    verification['first_turn_has_hops'] += 1
                    pattern = "first_turn_only"
                else:  # Middle turn
                    verification['middle_turn_has_hops'] += 1
                    pattern = f"turn_{hop_turn}_only"
            else:  # Multiple turns with hops
                verification['multiple_turns_with_hops'] += 1
                pattern = f"turns_{','.join(map(str, turns_with_hops))}"
            
            # Track pattern by number of turns
            key = f"{num_turns}_turns"
            if key not in verification['pattern_by_turns']:
                verification['pattern_by_turns'][key] = {}
            if pattern not in verification['pattern_by_turns'][key]:
                verification['pattern_by_turns'][key][pattern] = 0
            verification['pattern_by_turns'][key][pattern] += 1
            
            # Store examples (first 5 of each unusual pattern)
            if pattern != "last_turn_only" and len(
                verification['examples']
            ) < 10:
                verification['examples'].append({
                    'utterance': data.get("Utterance", "")[:80],
                    'num_turns': num_turns,
                    'turns_with_hops': turns_with_hops,
                    'pattern': pattern
                })
        
        return verification
    
    def _extract_queries_from_batched_queries(self, data: dict) -> dict:
        """Extract query strings from EvaluationData.batchedQueries.
        
        This handles the case where PluginInvocation in AllSearchResults is empty
        (e.g., 'search_web({})') but the actual queries are stored in 
        EvaluationData.turnData[].orchestrationIterations[].modelActions[]
            .toolInvocations[].batchedQueries[].arguments
        
        Note: AllSearchResults uses a flattened turn index that corresponds to
        orchestration iterations across all turns, not just turnData index.
        For example, if turnData[0] has 4 iterations, AllSearchResults will have
        turns "1", "2", "3", "4" corresponding to those iterations.
        
        Returns:
            dict mapping (turn_index, query_index) to query_string
            where turn_index matches AllSearchResults keys and query_index
            corresponds to the order within that turn's search results
        """
        query_map = {}
        
        evaluation_data = data.get("EvaluationData", {})
        turn_data = evaluation_data.get("turnData", [])
        
        # Track the flattened turn index (1-based, matching AllSearchResults)
        flattened_turn_idx = 0
        
        for turn in turn_data:
            orchestration_iterations = turn.get("orchestrationIterations", [])
            
            for iteration in orchestration_iterations:
                flattened_turn_idx += 1
                turn_key = str(flattened_turn_idx)
                
                # Track query index within this iteration/turn
                query_idx = 0
                
                model_actions = iteration.get("modelActions", [])
                
                for action in model_actions:
                    tool_invocations = action.get("toolInvocations", [])
                    
                    for invocation in tool_invocations:
                        batched_queries = invocation.get("batchedQueries", [])
                        # Get tool/function name from either 'function' or 'name' field
                        tool_name = invocation.get("function", "") or invocation.get("name", "")
                        
                        # Only process search-related tools
                        is_search_tool = any(
                            search_type in tool_name.lower()
                            for search_type in ['search', 'bing']
                        ) if tool_name else True  # Default to True if no tool name
                        
                        if not is_search_tool:
                            continue
                        
                        # Map each batched query to its index
                        for batch_query in batched_queries:
                            arguments = batch_query.get("arguments", "")
                            
                            # Parse arguments to extract the query
                            query_string = ""
                            if isinstance(arguments, str) and arguments:
                                # Try to parse as JSON first (search_web uses JSON string)
                                try:
                                    parsed = json.loads(arguments)
                                    if isinstance(parsed, dict):
                                        query_string = (
                                            parsed.get("query") or
                                            parsed.get("search_query") or
                                            parsed.get("queries") or
                                            arguments  # fallback to original string
                                        )
                                    else:
                                        query_string = arguments
                                except (json.JSONDecodeError, TypeError):
                                    query_string = arguments
                            elif isinstance(arguments, dict):
                                # Try common query field names
                                query_string = (
                                    arguments.get("query") or
                                    arguments.get("search_query") or
                                    arguments.get("queries") or
                                    str(arguments)
                                )
                            
                            # Store with composite key
                            key = (turn_key, query_idx)
                            query_map[key] = query_string
                            query_idx += 1
        
        return query_map
    
    def _build_block_to_query_map(self, data: dict) -> dict:
        """
        Build a mapping from (hop, plugin, block_index) to query string.
        
        This matches AllSearchResults blocks to batchedQueries using ref_id overlap.
        Each block in AllSearchResults corresponds to one batchedQuery, but ref_ids
        can be reused across different batchedQueries. We use overlap scoring to
        find the best match.
        
        The approach:
        1. Collect all ref_ids per batchedQuery from processedResult
        2. For each block in AllSearchResults, find the batchedQuery with highest
           ref_id overlap
        3. Return mapping of (hop, plugin, block_index) -> query_string
        
        Args:
            data: The full raw DCG data containing AllSearchResults and EvaluationData
            
        Returns:
            dict mapping (hop, plugin_name, block_index) -> query_string
        """
        block_to_query = {}
        
        evaluation_data = data.get("EvaluationData", {})
        turn_data = evaluation_data.get("turnData", [])
        all_search_results = data.get("AllSearchResults", {})
        
        # Step 1: Collect ref_ids and queries per batchedQuery, organized by hop and tool
        # Structure: {hop: {tool_name: [(query, set_of_ref_ids), ...]}}
        bq_data_by_hop = {}
        
        flattened_turn_idx = 0
        for turn in turn_data:
            orchestration_iterations = turn.get("orchestrationIterations", [])
            
            for iteration in orchestration_iterations:
                flattened_turn_idx += 1
                hop_key = str(flattened_turn_idx)
                
                if hop_key not in bq_data_by_hop:
                    bq_data_by_hop[hop_key] = {}
                
                model_actions = iteration.get("modelActions", [])
                
                for action in model_actions:
                    tool_invocations = action.get("toolInvocations", [])
                    
                    for invocation in tool_invocations:
                        tool_name = invocation.get("function", "") or invocation.get("name", "")
                        
                        if tool_name not in bq_data_by_hop[hop_key]:
                            bq_data_by_hop[hop_key][tool_name] = []
                        
                        batched_queries = invocation.get("batchedQueries", [])
                        
                        for batch_query in batched_queries:
                            # Extract query from arguments
                            arguments = batch_query.get("arguments", "")
                            query_string = ""
                            if isinstance(arguments, str) and arguments:
                                try:
                                    parsed = json.loads(arguments)
                                    if isinstance(parsed, dict):
                                        query_string = (
                                            parsed.get("query") or
                                            parsed.get("search_query") or
                                            parsed.get("queries") or
                                            arguments
                                        )
                                    else:
                                        query_string = arguments
                                except (json.JSONDecodeError, TypeError):
                                    query_string = arguments
                            elif isinstance(arguments, dict):
                                query_string = arguments.get("query", str(arguments))
                            
                            # Collect ALL ref_ids from processedResult
                            all_refs = set()
                            processed_result = batch_query.get("processedResult", "")
                            if processed_result:
                                try:
                                    if isinstance(processed_result, str):
                                        processed_data = json.loads(processed_result)
                                    else:
                                        processed_data = processed_result
                                    
                                    if isinstance(processed_data, dict):
                                        for result_type, items in processed_data.items():
                                            if isinstance(items, list):
                                                for item in items:
                                                    if isinstance(item, dict):
                                                        ref_id = item.get("reference_id", "")
                                                        if ref_id:
                                                            all_refs.add(ref_id)
                                except (json.JSONDecodeError, TypeError, AttributeError):
                                    pass
                            
                            bq_data_by_hop[hop_key][tool_name].append((query_string, all_refs))
        
        # Step 2: For each block in AllSearchResults, find best matching batchedQuery
        for hop_key, hop_data in all_search_results.items():
            if not isinstance(hop_data, dict):
                continue
            
            for plugin_name, blocks in hop_data.items():
                if not isinstance(blocks, list):
                    continue
                
                # Normalize plugin name for lookup in batchedQueries
                # AllSearchResults uses "office365_search_files" but batchedQueries uses "office365_search"
                normalized_plugin_name = plugin_name
                if plugin_name.startswith("office365_search_"):
                    normalized_plugin_name = "office365_search"
                
                # Get batchedQueries for this hop and tool
                bq_list = bq_data_by_hop.get(hop_key, {}).get(normalized_plugin_name, [])
                
                for block_idx, block in enumerate(blocks):
                    # Collect ref_ids from this block
                    block_refs = set()
                    results_list = block.get("Results", [])
                    for result in results_list:
                        ref_id = result.get("reference_id", "")
                        if ref_id:
                            block_refs.add(ref_id)
                    
                    # Find batchedQuery with highest overlap
                    best_query = ""
                    best_overlap = 0
                    
                    for query, bq_refs in bq_list:
                        overlap = len(block_refs & bq_refs)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_query = query
                    
                    # Store the mapping
                    key = (hop_key, plugin_name, block_idx)
                    block_to_query[key] = best_query
        
        return block_to_query
    
    def _build_first_ref_to_query_map(self, evaluation_data: dict) -> dict:
        """
        Build a mapping from first reference_id of each search to query string.
        
        This extracts queries from batchedQueries in EvaluationData.
        For each batchedQuery, we extract:
        - The query string from 'arguments'
        - The first reference_id from processedResult (WebPages, News, etc.)
        
        This allows us to match AllSearchResults entries (which have the first reference_id
        but empty PluginInvocation for search_web) to their actual query strings.
        
        Args:
            evaluation_data: The EvaluationData dict containing turnData with batchedQueries
            
        Returns:
            dict mapping first_reference_id -> query_string
        """
        first_ref_to_query = {}
        
        turn_data = evaluation_data.get("turnData", [])
        
        for turn in turn_data:
            orchestration_iterations = turn.get("orchestrationIterations", [])
            
            for iteration in orchestration_iterations:
                model_actions = iteration.get("modelActions", [])
                
                for action in model_actions:
                    tool_invocations = action.get("toolInvocations", [])
                    
                    for invocation in tool_invocations:
                        tool_name = invocation.get("function", "") or invocation.get("name", "")
                        
                        # Only process search_web (the problematic case with empty PluginInvocation)
                        if "search_web" not in tool_name.lower():
                            continue
                        
                        batched_queries = invocation.get("batchedQueries", [])
                        
                        for batch_query in batched_queries:
                            # Extract query from arguments
                            arguments = batch_query.get("arguments", "")
                            query_string = ""
                            if isinstance(arguments, str) and arguments:
                                # Try to parse as JSON first (search_web uses JSON string)
                                try:
                                    parsed = json.loads(arguments)
                                    if isinstance(parsed, dict):
                                        query_string = (
                                            parsed.get("query") or
                                            parsed.get("search_query") or
                                            parsed.get("queries") or
                                            arguments  # fallback to original string
                                        )
                                    else:
                                        query_string = arguments
                                except (json.JSONDecodeError, TypeError):
                                    query_string = arguments
                            elif isinstance(arguments, dict):
                                query_string = arguments.get("query", str(arguments))
                            
                            if not query_string:
                                continue
                            
                            # Parse processedResult to get reference_ids
                            processed_result = batch_query.get("processedResult", "")
                            if not processed_result:
                                continue
                            
                            try:
                                if isinstance(processed_result, str):
                                    processed_data = json.loads(processed_result)
                                else:
                                    processed_data = processed_result
                                
                                # Extract first reference_id from any result type
                                # (WebPages, News, QuestionsAndAnswers, etc.)
                                if isinstance(processed_data, dict):
                                    for result_type in ["WebPages", "News", "QuestionsAndAnswers", "Sports", "Videos"]:
                                        items = processed_data.get(result_type, [])
                                        if items and isinstance(items, list):
                                            first_ref = items[0].get("reference_id", "")
                                            if first_ref:
                                                first_ref_to_query[first_ref] = query_string
                                                # Don't break - store all first refs from all types
                            except (json.JSONDecodeError, TypeError, AttributeError):
                                continue
        
        return first_ref_to_query
    
    def _build_ref_to_query_map(self, evaluation_data: dict) -> dict:
        """
        Build a mapping from reference_id to query string.
        
        This extracts queries from citedResponseAttributions and uncitedResponseAttributions
        in EvaluationData. Each attribution has a 'searchQuery' field and a 'referenceMetadata'
        field containing the 'citationRefId' (e.g., 'turn1search9').
        
        This is the most reliable source for query-to-reference mapping as it comes directly
        from the response attribution data.
        
        Args:
            evaluation_data: The EvaluationData dict containing attribution arrays
            
        Returns:
            dict mapping reference_id -> query_string
        """
        ref_to_query = {}
        
        # Process both cited and uncited attributions
        attribution_sources = [
            evaluation_data.get('citedResponseAttributions', []),
            evaluation_data.get('uncitedResponseAttributions', [])
        ]
        
        for attributions in attribution_sources:
            if not attributions:
                continue
                
            for attr in attributions:
                search_query = attr.get('searchQuery', '')
                if not search_query:
                    continue
                    
                # Extract citationRefId from referenceMetadata
                ref_metadata = attr.get('referenceMetadata', '')
                if not ref_metadata:
                    continue
                    
                # referenceMetadata is a JSON string containing citationRefId
                # e.g., '{"type":"Web",...,"citationRefId":"turn1search9"}'
                try:
                    if isinstance(ref_metadata, str):
                        metadata_dict = json.loads(ref_metadata)
                    else:
                        metadata_dict = ref_metadata
                        
                    citation_ref_id = metadata_dict.get('citationRefId', '')
                    if citation_ref_id:
                        ref_to_query[citation_ref_id] = search_query
                except (json.JSONDecodeError, TypeError):
                    continue
        
        return ref_to_query
    
    def _extract_cite_dcg_data(self, data: dict) -> list:
        """
        Extract CiteDCG scores from JSON structure.
        If no results with CiteDCG labels found, return placeholder entry
        to preserve utterance in output.
        """
        results = []
        all_search_results = data.get("AllSearchResults", {})
        utterance = data.get("Utterance", "")
        
        # Extract num_turns from EvaluationData
        evaluation_data = data.get("EvaluationData", {})
        turn_data = evaluation_data.get("turnData", [])
        num_turns = len(turn_data) if turn_data else 1
        
        # Build block-to-query map using ref_id overlap matching
        # This is the primary method for resolving queries for blocks
        block_to_query_map = self._build_block_to_query_map(data)
        
        # Build reference_id to query map from attributions (fallback for remaining cases)
        ref_to_query_map = self._build_ref_to_query_map(evaluation_data)
        
        # If no utterance, skip this entry entirely
        if not utterance:
            return results
        
        # Track if we found any searches and scoreable results
        has_any_searches = False
        has_results_with_scores = False
        has_non_empty_results = False  # Track if ANY Results list is non-empty
        
        # Also collect results without scores for error cases
        results_without_scores = []
        
        for turn_index, query_data in all_search_results.items():
            if not query_data:
                continue
            
            has_any_searches = True
            
            for search_domain, domain_data in query_data.items():
                if not isinstance(domain_data, list):
                    continue
                
                for domain_item_idx, domain_item in enumerate(domain_data):
                    # Extract plugin metadata - preserve original format
                    plugin_name = domain_item.get("PluginName", "")
                    plugin_invocation = domain_item.get(
                        "PluginInvocation", ""
                    )
                    
                    # For office365_search, extract domain as separate field
                    # The domain is inside PluginInvocation: 
                    # office365_search({'queries': [{'domain': 'files', ...}]})
                    # Matching logic will combine plugin_name + domain when needed
                    search_domain_value = ""
                    if plugin_name == "office365_search":
                        search_domain_value = self._extract_domain_from_invocation(plugin_invocation)
                    
                    # Get results list early (needed for reference_id fallback)
                    results_list = domain_item.get("Results", [])
                    
                    # Only extract queries from search plugins
                    # Skip non-search plugins (record_memory, fetch_*, etc.)
                    is_search_plugin = any(
                        search_type in plugin_name.lower()
                        for search_type in ['search', 'bing', 'office365']
                    ) and not any(
                        fetch_type in plugin_name.lower()
                        for fetch_type in ['fetch_', 'record_']
                    )
                    
                    query_string = ""
                    if is_search_plugin:
                        # First try to extract from PluginInvocation
                        query_string = self._extract_query_from_invocation(
                            plugin_invocation, plugin_name
                        )
                        
                        # Fallback: Use block-to-query map (matches blocks via ref_id overlap)
                        # This correctly handles cases where ref_ids are reused across queries
                        if not query_string:
                            block_key = (turn_index, plugin_name, domain_item_idx)
                            query_string = block_to_query_map.get(block_key, "")
                        
                        # Fallback 2: Try reference_id lookup from attributions
                        if not query_string and results_list:
                            first_ref_id = results_list[0].get("reference_id", "")
                            if first_ref_id:
                                query_string = ref_to_query_map.get(first_ref_id, "")
                    
                    # Track if this search has non-empty results
                    if results_list:
                        has_non_empty_results = True
                    
                    # Extract ContentDomainName from first result (for Graph Connectors)
                    # This is needed to match with conversation data
                    content_domain_name = None
                    if results_list and ("search_enterprise_connectors" in plugin_name or "search_enterprise" in plugin_name):
                        first_result = results_list[0]
                        content_domain = first_result.get("ContentDomain", {})
                        if isinstance(content_domain, dict):
                            content_domain_name = content_domain.get("Name")
                    
                    for result in results_list:
                        if "CiteDCGLLMLabel" not in result:
                            # Collect results without scores for error tracking
                            extracted = self._extract_single_result(
                                result,
                                turn_index,
                                search_domain,
                                plugin_name,
                                query_string,
                                plugin_invocation,
                                include_null_score=True,
                                domain=search_domain_value
                            )
                            # Store block_index to preserve original AllSearchResults block structure
                            extracted["_block_index"] = domain_item_idx
                            # Store content_domain_name temporarily for grouping (will be moved to search level)
                            if content_domain_name:
                                extracted["_content_domain_name"] = content_domain_name
                            results_without_scores.append(extracted)
                            continue

                        
                        has_results_with_scores = True
                        extracted = self._extract_single_result(
                            result,
                            turn_index,
                            search_domain,
                            plugin_name,
                            query_string,
                            plugin_invocation,
                            domain=search_domain_value
                        )
                        extracted["num_turns"] = num_turns
                        # Store block_index to preserve original AllSearchResults block structure
                        extracted["_block_index"] = domain_item_idx
                        # Store content_domain_name temporarily for grouping (will be moved to search level)
                        if content_domain_name:
                            extracted["_content_domain_name"] = content_domain_name
                        results.append(extracted)
        
        # If no scoreable results found, add placeholder
        # This includes: searches with empty results, or no searches at all
        if not has_results_with_scores:
            reason = (
                "no_search_results" if has_any_searches
                else "no_searches_executed"
            )
            
            # ERROR: Non-empty results but no CiteDCG labels
            is_error = has_non_empty_results
            
            # For error cases, include the results without scores
            if is_error and results_without_scores:
                # Include actual results but mark as having no scores
                for item in results_without_scores:
                    item["is_error"] = True
                    item["has_cite_dcg_scores"] = False
                    item["num_turns"] = num_turns
                    # Ensure utterance is set from parent data if missing
                    if not item.get("utterance"):
                        item["utterance"] = utterance
                return results_without_scores
            else:
                # Empty results or no searches - return placeholder
                results.append({
                    "utterance": utterance,
                    "num_turns": num_turns,
                    "has_cite_dcg_scores": False,
                    "reason": reason,
                    "searches": [],
                    "is_error": is_error
                })
        
        # No deduplication needed - same reference_id can legitimately appear
        # in multiple hops when search results are reused across turns
        
        return results
    
    def _extract_query_from_invocation(
        self, invocation_str: str, plugin_name: str = ""
    ) -> str:
        """Extract query string from PluginInvocation field.
        
        Handles both simple string queries and complex structured queries.
        For web search, the query might be a dict with 'search_query' field.
        
        Args:
            invocation_str: The PluginInvocation string
            plugin_name: Name of the plugin (for logging)
        
        Examples:
            office365_search({'domain': 'files', 'query': 'my query'})
            bing_search({'search_query': 'my search', 'market': 'en-US'})
        """
        if not invocation_str:
            return ""
        
        try:
            # First try to parse as eval (safe for dict literals)
            # Extract the part inside parentheses
            import ast
            import re

            # Find the dict/parameter part: function_name({...})
            # Use a more robust approach to extract the dict
            # Find opening paren after function name
            paren_start = invocation_str.find('(')
            if paren_start != -1:
                # Find matching closing paren by counting nesting
                brace_count = 0
                in_string = False
                string_char = None
                escaped = False
                
                for i in range(paren_start + 1, len(invocation_str)):
                    char = invocation_str[i]
                    
                    if escaped:
                        escaped = False
                        continue
                    
                    if char == '\\':
                        escaped = True
                        continue
                    
                    # Track string state
                    if char in ('"', "'"):
                        if not in_string:
                            in_string = True
                            string_char = char
                        elif char == string_char:
                            in_string = False
                            string_char = None
                        continue
                    
                    # Only count parens outside strings
                    if not in_string:
                        if char == '(':
                            brace_count += 1
                        elif char == ')':
                            if brace_count == 0:
                                # Found matching closing paren
                                dict_str = invocation_str[paren_start + 1:i]
                                try:
                                    # Safely parse the dict literal
                                    params = ast.literal_eval(dict_str)
                                    if isinstance(params, dict):
                                        # Try 'query' first (office365_search)
                                        if 'query' in params:
                                            query_val = params['query']
                                            # Handle nested dict (web search case)
                                            if isinstance(query_val, dict):
                                                # Try common web search fields
                                                return (
                                                    query_val.get('search_query') or
                                                    query_val.get('query') or
                                                    str(query_val)
                                                )
                                            return str(query_val)
                                        
                                        # Try 'search_query' (bing_search)
                                        if 'search_query' in params:
                                            return str(params['search_query'])
                                        
                                        # Try 'querykeywords' (enterprise connectors)
                                        if 'querykeywords' in params:
                                            return str(params['querykeywords'])
                                except (ValueError, SyntaxError):
                                    pass
                                break
                            else:
                                brace_count -= 1
            
            # Fallback: regex extraction for simple string queries
            # NOTE: These regexes may truncate queries with embedded quotes
            # The ast.literal_eval approach above should handle most cases
            # Try single quotes: 'query': 'value'
            match = re.search(r"'query':\s*'([^']+)'", invocation_str)
            if match:
                return match.group(1)
            
            # Try double quotes: "query": "value"
            match = re.search(r'"query":\s*"([^"]+)"', invocation_str)
            if match:
                return match.group(1)
            
            # Try search_query field
            match = re.search(r"'search_query':\s*'([^']+)'", invocation_str)
            if match:
                return match.group(1)
            
            match = re.search(r'"search_query":\s*"([^"]+)"', invocation_str)
            if match:
                return match.group(1)
            
            # Try querykeywords field (enterprise connectors)
            match = re.search(r"'querykeywords':\s*'([^']*)'", invocation_str)
            if match:
                return match.group(1)  # May be empty string for searches with no keywords
            
            match = re.search(r'"querykeywords":\s*"([^"]*)"', invocation_str)
            if match:
                return match.group(1)  # May be empty string for searches with no keywords
            
            # Handle queries with escaped quotes or nested quotes
            # Pattern: 'query': "value with 'quotes'"
            match = re.search(r"'query':\s*\"([^\"]+)\"", invocation_str)
            if match:
                return match.group(1)
            
            # Pattern: "query": 'value with "quotes"'
            match = re.search(r'"query":\s*\'([^\']+)\'', invocation_str)
            if match:
                return match.group(1)
                
        except Exception as e:
            logger.debug(f"Error extracting query from invocation: {e}")
        
        # If we reach here for a search plugin, log debug info
        # (not a warning since some searches legitimately have empty queries)
        if invocation_str and plugin_name:
            logger.debug(
                f"Could not extract query from search plugin "
                f"'{plugin_name}': {invocation_str[:100]}"
            )
        
        return ""

    def _extract_domain_from_invocation(self, invocation_str: str) -> str:
        """Extract domain from PluginInvocation field for office365_search.
        
        The domain (files, people, events, etc.) is stored inside the invocation string:
        office365_search({'queries': [{'domain': 'files', 'query': '...'}]})
        
        This is needed for matching with conversation data which uses format:
        office365_search_files, office365_search_people, etc.
        
        Args:
            invocation_str: The PluginInvocation string
            
        Returns:
            The domain string (e.g., 'files', 'people') or empty string if not found
        """
        if not invocation_str:
            return ""
        
        try:
            import ast
            import re

            # Find the dict/parameter part: function_name({...})
            paren_start = invocation_str.find('(')
            if paren_start != -1:
                # Find matching closing paren
                brace_count = 0
                in_string = False
                string_char = None
                escaped = False
                
                for i in range(paren_start + 1, len(invocation_str)):
                    char = invocation_str[i]
                    
                    if escaped:
                        escaped = False
                        continue
                    
                    if char == '\\':
                        escaped = True
                        continue
                    
                    if char in ('"', "'"):
                        if not in_string:
                            in_string = True
                            string_char = char
                        elif char == string_char:
                            in_string = False
                            string_char = None
                        continue
                    
                    if not in_string:
                        if char == '(':
                            brace_count += 1
                        elif char == ')':
                            if brace_count == 0:
                                dict_str = invocation_str[paren_start + 1:i]
                                try:
                                    params = ast.literal_eval(dict_str)
                                    if isinstance(params, dict):
                                        # Try 'domain' directly
                                        if 'domain' in params:
                                            return str(params['domain'])
                                        # Try 'queries' list (office365_search format)
                                        if 'queries' in params and isinstance(params['queries'], list):
                                            for q in params['queries']:
                                                if isinstance(q, dict) and 'domain' in q:
                                                    return str(q['domain'])
                                except (ValueError, SyntaxError):
                                    pass
                                break
                            else:
                                brace_count -= 1
            
            # Fallback: regex extraction
            match = re.search(r"'domain':\s*'([^']+)'", invocation_str)
            if match:
                return match.group(1)
            
            match = re.search(r'"domain":\s*"([^"]+)"', invocation_str)
            if match:
                return match.group(1)
                
        except Exception as e:
            logger.debug(f"Error extracting domain from invocation: {e}")
        
        return ""
    
    def _extract_single_result(
        self,
        result: dict,
        turn_index: str,
        search_domain: str,
        plugin_name: str = "",
        query_string: str = "",
        plugin_invocation: str = "",
        include_null_score: bool = False,
        domain: str = "",
    ) -> dict:
        """Extract relevant fields from a single result."""
        # Set default CiteDCG label based on whether we're including nulls
        default_label = None if include_null_score else ""
        
        extracted = {
            "turn_index": turn_index,
            "search_domain": search_domain,
            "plugin_name": plugin_name,
            "domain": domain,  # Extracted from PluginInvocation for office365_search
            "query_string": query_string,
            "plugin_invocation": plugin_invocation,
            # Extract reference_id: Office365 uses "ReferenceId" (PascalCase),
            # Web searches use "reference_id" (lowercase). Check both.
            "reference_id": (
                result.get("reference_id", "")
                or result.get("ReferenceId", "")
            ),
            "CiteDCGLLMLabel": result.get("CiteDCGLLMLabel", default_label),
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
            num_turns = item.get("num_turns", 1)
            
            key = utterance
            
            if not query_groups[key].get("utterance"):
                query_groups[key]["utterance"] = utterance
                query_groups[key]["timestamp"] = item.get("timestamp", "")
                query_groups[key]["num_turns"] = num_turns
                # Track if this is a placeholder entry
                query_groups[key]["has_cite_dcg_scores"] = item.get(
                    "has_cite_dcg_scores", True
                )
                query_groups[key]["reason"] = item.get("reason", "")
            
            query_groups[key]["results"].append(item)
        
        # Convert to list with hierarchical grouping
        grouped_list = []
        for group_data in query_groups.values():
            results = group_data["results"]
            
            # Check if this is a placeholder entry (no DCG scores)
            has_dcg = group_data.get("has_cite_dcg_scores", True)
            
            # Check if any result has search_domain (indicating actual search data)
            has_search_data = any(
                "search_domain" in result for result in results
            )
            
            if not has_dcg and not has_search_data:
                # Create simplified entry for utterances without searches
                # or empty search results
                ordered_group = {
                    "utterance": group_data.get("utterance", ""),
                    "timestamp": group_data.get("timestamp", ""),
                    "num_turns": group_data.get("num_turns", 1),
                    "num_hops": 0,
                    "non_empty_turns": 0,
                    "has_cite_dcg_scores": False,
                    "reason": group_data.get("reason", ""),
                    "total_results": 0,
                    "results_by_plugin": {},
                    "searches": [],
                }
                grouped_list.append(ordered_group)
                continue
            
            # Process results with search data (normal or error cases)
            search_groups = {}  # Use regular dict since we initialize each key explicitly
            plugin_invocations = {}  # Store plugin_invocation per group
            content_domain_names = {}  # Store content_domain_name per group (for Graph Connectors)
            extracted_domains = {}  # Store extracted domain (files/people/etc.) for office365_search
            for result in results:
                search_domain_key = result.get("search_domain", "unknown")
                query_string = result.get("query_string", "")
                plugin_name = result.get("plugin_name", "")
                plugin_invocation = result.get("plugin_invocation", "")
                hop = result.get("turn_index", "")
                extracted_domain = result.get("domain", "")  # Extracted from PluginInvocation
                block_index = result.get("_block_index", 0)  # Block index from AllSearchResults
                result_type = result.get("Type", "")  # e.g., "search_web_webpages", "search_web_news"
                
                # Grouping strategy:
                # - For search_web: group by (hop, type, query_string, plugin_name) 
                #   This matches conversation data which has separate queries for webpages/news
                # - For others: group by (hop, search_domain_key, block_index, plugin_name)
                #   This preserves AllSearchResults block structure
                if plugin_name == "search_web" and result_type:
                    # Use Type and query to create separate groups
                    key = (hop, result_type, query_string, plugin_name)
                else:
                    # Use block_index for non-web searches
                    key = (hop, search_domain_key, block_index, plugin_name)
                
                # Store plugin_invocation for this group (same for all results in group)
                if key not in plugin_invocations:
                    plugin_invocations[key] = plugin_invocation
                
                # Store query_string for this group - only initialize once per key
                if key not in search_groups or not isinstance(search_groups[key], dict):
                    search_groups[key] = {"_query": query_string, "results": [], "_types": []}
                
                # Store content_domain_name for this group (for Graph Connectors)
                if key not in content_domain_names:
                    content_domain_names[key] = result.get("_content_domain_name")
                
                # Store extracted domain for this group (for office365_search)
                if key not in extracted_domains:
                    extracted_domains[key] = extracted_domain
                
                # Track Type values for this group (for web searches)
                result_type = result.get("Type", "")
                if result_type:
                    search_groups[key]["_types"].append(result_type)
                
                # Remove redundant fields for result item
                result_item = {
                    k: v
                    for k, v in result.items()
                    if k not in [
                        "turn_index",  # Now at search level as "hop"
                        "utterance",
                        "timestamp",
                        "search_domain",
                        "plugin_name",
                        "domain",  # Now at search level
                        "query_string",
                        "plugin_invocation",  # Now at search level
                        "is_error",  # Remove internal tracking field
                        "_content_domain_name",  # Temporary field, now at search level
                        "_block_index",  # Internal field for grouping
                    ]
                }
                search_groups[key]["results"].append(result_item)
            
            # Create search entries
            searches = []
            total_results = 0
            
            for key, search_group_data in search_groups.items():
                # Key structure varies:
                # - For search_web: (hop, result_type, query_string, plugin_name)
                # - For others: (hop, search_domain_key, block_index, plugin_name)
                hop = key[0]
                second_element = key[1]  # result_type for search_web, search_domain_key for others
                third_element = key[2]  # query_string for search_web, block_index for others
                plugin_name = key[3]
                
                query_string = search_group_data.get("_query", "")
                search_results = search_group_data.get("results", [])
                type_list = search_group_data.get("_types", [])
                
                # Get content_domain_name from stored dict (for Graph Connectors)
                content_domain_name = content_domain_names.get(key)
                
                # Get extracted domain for office365_search (files/people/etc.)
                extracted_domain = extracted_domains.get(key, "")
                
                # Determine the type for web searches
                # For search_web, second_element IS the result_type (from grouping key)
                result_type = None
                if plugin_name == "search_web":
                    # The second element is the result_type for search_web
                    result_type = second_element if isinstance(second_element, str) and second_element.startswith("search_web_") else None
                
                search_dict = {
                    "hop": hop,
                    "plugin_name": plugin_name,
                    "query_string": query_string,
                    "plugin_invocation": plugin_invocations.get(key, ""),
                    "result_count": len(search_results),
                    "results": search_results,
                }
                
                # Add type for web searches (for matching with conversation data)
                if result_type:
                    search_dict["type"] = result_type
                
                # Add domain for office365_search (extracted from PluginInvocation)
                # This is used for matching with conversation data
                if extracted_domain:
                    search_dict["domain"] = extracted_domain
                
                # Add content_domain_name at search level for Graph Connectors
                if content_domain_name:
                    search_dict["content_domain_name"] = content_domain_name
                
                searches.append(search_dict)
                
                total_results += len(search_results)
            
            # Sort searches by hop, then plugin_name, then block_index to preserve order
            searches.sort(
                key=lambda x: (x["hop"], x["plugin_name"], x.get("query_string", ""))
            )
            
            # Count results by plugin
            plugin_counts = {}
            for search in searches:
                plugin = search["plugin_name"]
                plugin_counts[plugin] = (
                    plugin_counts.get(plugin, 0) + search["result_count"]
                )
            
            # Calculate num_hops (unique hop numbers)
            unique_hops = set(search["hop"] for search in searches)
            num_hops = len(unique_hops)
            
            # For current data: always 1 non-empty turn (the successful one)
            # Future: can track multiple turns with hops if pattern changes
            non_empty_turns = 1 if num_hops > 0 else 0
            
            ordered_group = {
                "utterance": group_data.get("utterance", ""),
                "timestamp": group_data.get("timestamp", ""),
                "num_turns": group_data.get("num_turns", 1),
                "num_hops": num_hops,
                "non_empty_turns": non_empty_turns,
                "has_cite_dcg_scores": has_dcg,
                "total_results": total_results,
                "results_by_plugin": plugin_counts,
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


def extract_conv_details_and_dcg_from_raw_dcgfiles(raw_data: dict) -> dict:
    """
    Extract unified conversation details and CiteDCG scores from raw results.json.
    
    Extracts BOTH conversation metadata and per-result DCG scores from a single
    raw data record, enabling unified analysis without cross-referencing multiple
    data sources. Uses reference_id mapping to resolve queries for plugins with
    empty PluginInvocation (e.g., search_web({})).
    
    Query Resolution Strategy:
        1. Primary: Extract from PluginInvocation (office365_search, bing_search, etc.)
        2. Fallback: Map reference_id -> query via batchedQueries.processedResult.WebPages
    
    Args:
        raw_data: A single parsed JSON record from results.json containing:
            - Utterance, ConversationId: Conversation metadata
            - EvaluationData.turnData: Turn and query information
            - AllSearchResults: Search results with CiteDCG scores
        
    Returns:
        dict: Unified extraction result with structure:
            {
                'utterance': str,           # Original user query
                'conversation_id': str,     # Unique conversation identifier
                'num_turns': int,           # Number of conversation turns
                'has_cite_dcg_scores': bool,# Whether any results have DCG labels
                'total_results': int,       # Total search results count
                'searches': [               # List of search operations
                    {
                        'hop': str,             # Search hop/turn index
                        'plugin_name': str,     # Plugin used (search_web, etc.)
                        'query_string': str,    # Resolved query string
                        'plugin_invocation': str,# Raw plugin invocation
                        'result_count': int,    # Results in this search
                        'results': [            # Individual search results
                            {
                                'reference_id': str,        # Result reference ID
                                'CiteDCGLLMLabel': float|None, # DCG score (0-4)
                                'query_string': str,        # Query for this result
                                'ResultType': str,          # Result type category
                                'Type': str,                # Content type
                                'Title': str,               # Result title
                            }
                        ]
                    }
                ]
            }
    
    Example:
        >>> with open('results.json') as f:
        ...     for line in f:
        ...         record = json.loads(line)
        ...         extracted = extract_conv_details_and_dcg_from_raw_dcgfiles(record)
        ...         print(f"Utterance: {extracted['utterance'][:50]}...")
        ...         print(f"Total results: {extracted['total_results']}")
    """
    result = {
        'utterance': raw_data.get('Utterance', ''),
        'conversation_id': raw_data.get('ConversationId', ''),
        'num_turns': 0,
        'has_cite_dcg_scores': False,
        'total_results': 0,
        'searches': []
    }
    
    eval_data = raw_data.get('EvaluationData', {})
    turn_data = eval_data.get('turnData', [])
    all_search = raw_data.get('AllSearchResults', {})
    
    result['num_turns'] = len(turn_data)
    
    # Step 1: Build block-index -> query mapping from batchedQueries
    # This maps (turn_key, plugin_name, block_index) -> query_string
    # to align AllSearchResults blocks with batchedQueries by index
    block_to_query = _build_block_to_query_map(turn_data)
    
    # Also build first_ref -> query map as fallback for cases where
    # block alignment doesn't work (e.g., different plugin ordering)
    first_ref_to_query = _build_first_ref_to_query_map_unified(turn_data)
    
    # Step 2: Process AllSearchResults to get DCG scores and merge with queries
    # Track block index per (turn_key, plugin_name) for alignment
    block_indices = {}
    
    for turn_key, search_data in sorted(all_search.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        for domain, items in search_data.items():
            if not isinstance(items, list):
                continue
            
            for item_idx, item in enumerate(items):
                plugin_name = item.get('PluginName', '')
                plugin_invocation = item.get('PluginInvocation', '')
                results_list = item.get('Results', [])
                
                if not results_list:
                    continue
                
                # Track block index for this (turn_key, plugin_name)
                block_key = (turn_key, plugin_name)
                if block_key not in block_indices:
                    block_indices[block_key] = 0
                current_block_idx = block_indices[block_key]
                block_indices[block_key] += 1
                
                # Try to extract query from PluginInvocation first
                query_from_invocation = _extract_query_from_invocation_unified(
                    plugin_invocation, plugin_name
                )
                
                # Fallback 1: Use block-index alignment with batchedQueries
                block_query = ''
                if not query_from_invocation:
                    # Normalize plugin name for lookup (AllSearchResults uses 'office365_search_files'
                    # but batchedQueries uses 'office365_search')
                    normalized_plugin = plugin_name
                    if plugin_name.startswith('office365_search_'):
                        normalized_plugin = 'office365_search'
                    block_query_key = (turn_key, normalized_plugin, current_block_idx)
                    block_query = block_to_query.get(block_query_key, '')
                
                # Fallback 2: Use first result's reference_id to find query
                first_ref_query = ''
                if not query_from_invocation and not block_query:
                    first_ref = results_list[0].get('reference_id', '')
                    if first_ref:
                        first_ref_query = first_ref_to_query.get(first_ref, '')
                
                # Determine the query to use for all results in this block
                # Priority: PluginInvocation > block-index alignment > first_ref lookup
                block_level_query = query_from_invocation or block_query or first_ref_query
                
                # Group results and determine query for each
                search_entry = {
                    'hop': turn_key,
                    'plugin_name': plugin_name,
                    'query_string': block_level_query,
                    'plugin_invocation': plugin_invocation,
                    'result_count': len(results_list),
                    'results': []
                }
                
                for r in results_list:
                    ref_id = r.get('reference_id', '')
                    dcg_label = r.get('CiteDCGLLMLabel')
                    
                    # All results in the same block get the same query
                    # (Don't do per-result ref_id lookup as it can be wrong)
                    result_entry = {
                        'reference_id': ref_id,
                        'CiteDCGLLMLabel': dcg_label,
                        'query_string': block_level_query,  # Same query for all results in block
                        'ResultType': r.get('ResultType', ''),
                        'Type': r.get('Type', ''),
                        'Title': r.get('Title', ''),
                    }
                    
                    if dcg_label is not None:
                        result['has_cite_dcg_scores'] = True
                    
                    search_entry['results'].append(result_entry)
                    result['total_results'] += 1
                
                result['searches'].append(search_entry)
    
    return result


def _build_block_to_query_map(turn_data: list) -> dict:
    """
    Build a mapping from (turn_key, plugin_name, block_index) to query string.
    
    This enables alignment between AllSearchResults blocks and batchedQueries
    by block index within each (turn, plugin) combination.
    
    Args:
        turn_data: List of turn data from EvaluationData.turnData
        
    Returns:
        dict mapping (turn_key, plugin_name, block_index) -> query_string
    """
    block_to_query = {}
    
    # Track the flattened turn index (1-based, matching AllSearchResults keys)
    flattened_turn_idx = 0
    
    for turn in turn_data:
        orchestration_iterations = turn.get('orchestrationIterations', [])
        
        for iteration in orchestration_iterations:
            flattened_turn_idx += 1
            turn_key = str(flattened_turn_idx)
            
            # Track block index per plugin_name within this turn
            plugin_block_indices = {}
            
            model_actions = iteration.get('modelActions', [])
            
            for action in model_actions:
                tool_invocations = action.get('toolInvocations', [])
                
                for invocation in tool_invocations:
                    tool_name = invocation.get('function', '') or invocation.get('name', '')
                    batched_queries = invocation.get('batchedQueries', [])
                    
                    # Normalize tool name to match AllSearchResults plugin names
                    plugin_name = tool_name  # e.g., 'search_web'
                    
                    if plugin_name not in plugin_block_indices:
                        plugin_block_indices[plugin_name] = 0
                    
                    for bq in batched_queries:
                        query = bq.get('arguments', '')
                        
                        # Store with (turn_key, plugin_name, block_index)
                        block_idx = plugin_block_indices[plugin_name]
                        key = (turn_key, plugin_name, block_idx)
                        block_to_query[key] = query
                        plugin_block_indices[plugin_name] += 1
    
    return block_to_query


def _build_first_ref_to_query_map_unified(turn_data: list) -> dict:
    """
    Build a mapping from first reference_id of each search to query string.
    
    This extracts queries from batchedQueries and maps them to the first
    reference_id found in processedResult (WebPages or News).
    
    Args:
        turn_data: List of turn data from EvaluationData.turnData
        
    Returns:
        dict mapping first_reference_id -> query_string
    """
    first_ref_to_query = {}
    
    for turn in turn_data:
        for iteration in turn.get('orchestrationIterations', []):
            for action in iteration.get('modelActions', []):
                for inv in action.get('toolInvocations', []):
                    for bq in inv.get('batchedQueries', []):
                        query = bq.get('arguments', '')
                        processed = bq.get('processedResult', '')
                        
                        if not query or not processed:
                            continue
                        
                        # Parse processedResult
                        try:
                            if isinstance(processed, str):
                                processed = json.loads(processed)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        
                        if not isinstance(processed, dict):
                            continue
                        
                        # Get first ref from WebPages or News (not all refs!)
                        # This is used as fallback when block alignment fails
                        for result_type in ['WebPages', 'News', 'Videos', 'QuestionsAndAnswers']:
                            items = processed.get(result_type, [])
                            if items and isinstance(items, list):
                                first_ref = items[0].get('reference_id', '')
                                if first_ref and first_ref not in first_ref_to_query:
                                    first_ref_to_query[first_ref] = query
                                break  # Only use first available result type
    
    return first_ref_to_query


def _build_ref_to_query_map(turn_data: list) -> dict:
    """
    Build a mapping from reference_id to query string.
    
    This extracts queries from batchedQueries in EvaluationData and maps them
    to reference_ids found in processedResult.WebPages.
    
    Args:
        turn_data: List of turn data from EvaluationData.turnData
        
    Returns:
        dict mapping reference_id -> query_string
    """
    ref_to_query = {}
    
    for turn in turn_data:
        for iteration in turn.get('orchestrationIterations', []):
            for action in iteration.get('modelActions', []):
                for inv in action.get('toolInvocations', []):
                    for bq in inv.get('batchedQueries', []):
                        query = bq.get('arguments', '')
                        processed = bq.get('processedResult', '')
                        
                        # processedResult might be a JSON string
                        if isinstance(processed, str):
                            try:
                                processed = json.loads(processed)
                            except (json.JSONDecodeError, TypeError):
                                continue
                        
                        if isinstance(processed, dict):
                            webpages = processed.get('WebPages', [])
                            for wp in webpages:
                                ref_id = wp.get('reference_id', '')
                                if ref_id:
                                    ref_to_query[ref_id] = query
    
    return ref_to_query


def _extract_query_from_invocation_unified(invocation_str: str, plugin_name: str = "") -> str:
    """
    Extract query from PluginInvocation string.
    
    Args:
        invocation_str: The PluginInvocation string (e.g., "office365_search({...})")
        plugin_name: Name of the plugin (for context)
        
    Returns:
        Extracted query string, or empty string if not found
    """
    if not invocation_str:
        return ""
    
    # Check for empty invocation like "search_web({})"
    if invocation_str.endswith("({})"):
        return ""
    
    try:
        import ast
        import re

        # Find the dict/parameter part: function_name({...})
        paren_start = invocation_str.find('(')
        if paren_start == -1:
            return ""
        
        # Extract content between parentheses
        paren_end = invocation_str.rfind(')')
        if paren_end <= paren_start:
            return ""
        
        dict_str = invocation_str[paren_start + 1:paren_end]
        
        try:
            params = ast.literal_eval(dict_str)
            if isinstance(params, dict):
                # Try 'query' first (office365_search, etc.)
                if 'query' in params:
                    return str(params['query'])
                # Try 'search_query' (bing_search)
                if 'search_query' in params:
                    return str(params['search_query'])
                # Try 'querykeywords' (enterprise connectors)
                if 'querykeywords' in params:
                    return str(params['querykeywords'])
        except (ValueError, SyntaxError):
            pass
        
        # Fallback: regex extraction
        match = re.search(r"'query':\s*['\"]([^'\"]+)['\"]", invocation_str)
        if match:
            return match.group(1)
        
        match = re.search(r"'search_query':\s*['\"]([^'\"]+)['\"]", invocation_str)
        if match:
            return match.group(1)
        
    except Exception:
        pass
    
    return ""


def extract_conv_details_and_dcg_from_raw(
    raw_file: str,
    output_file: str = None,
    utterance_filter: str = None
) -> list:
    """
    CLI wrapper to extract unified conversation + DCG data from raw results.json.
    
    Processes a raw SEVAL results file (JSONL format) and extracts both
    conversation details and CiteDCG scores in a unified structure.
    
    Args:
        raw_file: Path to raw file. Supports both:
            - JSONL format (one JSON record per line, like results.json)
            - Single JSON file (one complete JSON object)
        output_file: Optional path for JSON output. If None, prints to stdout.
        utterance_filter: Optional substring to filter by utterance text
        
    Returns:
        list: List of extracted records
        
    Example:
        python get_seval_metrics.py extract_conv_details_and_dcg_from_raw \\
            --raw_file=../../temp/dcg_catering_raw.json \\
            --output_file=results/unified_output.json
    """
    if not os.path.exists(raw_file):
        print(f"Error: File not found: {raw_file}")
        return []
    
    results = []
    
    # Try to detect file format: JSONL vs single JSON
    with open(raw_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try parsing as single JSON object first
    try:
        raw_data = json.loads(content)
        # If it's a dict with expected keys, treat as single record
        if isinstance(raw_data, dict) and ('Utterance' in raw_data or 'AllSearchResults' in raw_data):
            utterance = raw_data.get('Utterance', '')
            if not utterance_filter or utterance_filter.lower() in utterance.lower():
                extracted = extract_conv_details_and_dcg_from_raw_dcgfiles(raw_data)
                results.append(extracted)
        # If it's a list, treat each item as a record
        elif isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, dict):
                    utterance = item.get('Utterance', '')
                    if utterance_filter and utterance_filter.lower() not in utterance.lower():
                        continue
                    extracted = extract_conv_details_and_dcg_from_raw_dcgfiles(item)
                    results.append(extracted)
    except json.JSONDecodeError:
        # Fall back to JSONL format (one JSON object per line)
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                raw_data = json.loads(line)
            except json.JSONDecodeError as e:
                # Only warn for lines that look like they should be JSON
                if line.startswith('{'):
                    print(f"Warning: Skipping line {line_num} - JSON parse error: {e}")
                continue
            
            # Apply utterance filter if specified
            utterance = raw_data.get('Utterance', '')
            if utterance_filter and utterance_filter.lower() not in utterance.lower():
                continue
            
            extracted = extract_conv_details_and_dcg_from_raw_dcgfiles(raw_data)
            results.append(extracted)
    
    # Note: Removed print here - the caller (extract_unified_dcg_batch) handles progress output
    # print(f"Processed {len(results)} records from {raw_file}")
    
    if output_file:
        # Always write to file, even if results is empty (will write empty array [])
        os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Output written to: {output_file} ({len(results)} records)")
    
    # Always return results (can be empty list)
    return results


# Note: extract_unified_dcg_batch has been moved to seval_batch_processor.py
# for better modularity (batch processing functions belong there)


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
    utterance: Optional[str] = None,
    verbose: bool = True
) -> int:
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
        verbose: If True, show detailed statistics. If False, minimal output.
    
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
        extractor = PerResultCiteDCGExtractor(verbose=verbose)
        count = extractor.extract(
            metrics_folder=metrics_folder,
            experiment=experiment,
            output_file=output_file,
            utterance=utterance
        )
        return count
    except Exception as e:
        logger.error(f"Error extracting CiteDCG scores: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Use fire for command line arguments
    # Exposes all modular functions as commands
    # Note: extract_unified_dcg_batch has been moved to seval_batch_processor.py
    try:
        fire.Fire({
            'list_csv_files': list_csv_files,
            'get_metrics_summary': get_metrics_summary,
            'generate_citedcg_report': generate_citedcg_report,
            'read_metrics': read_metrics,
            'extract_per_result_citedcg': extract_per_result_citedcg,
            'extract_conv_details_and_dcg_from_raw': extract_conv_details_and_dcg_from_raw,
        })
    except FireExit as e:
        # Handle Fire's exit (including --help) gracefully in debug mode
        # FireExit with code 0 means successful exit (like --help)
        # FireExit with non-zero code means error exit
        sys.exit(e.code)
        sys.exit(e.code)
        sys.exit(e.code)
        sys.exit(e.code)
        sys.exit(e.code)
        sys.exit(e.code)
