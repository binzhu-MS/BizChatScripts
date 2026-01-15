"""
Transform SEVAL raw data files to llm_ndcg input format.

This module converts SEVAL control/treatment file pairs into the JSONL format
expected by the llm_ndcg metric for evaluating search result quality.

Description:
    - Pairs control and experiment files by query_hash
    - Extracts search results from the last turn (for multi-turn conversations)
    - Transforms to llm_ndcg input format with all_search_results_control/treatment

Usage:
    python transform_seval_to_llm_ndcg.py transform \
        --input_dir "../seval_data/123665_scraping_raw_data_output" \
        --output_file "llm_ndcg_input.jsonl" \
        --max_pairs 100

    python transform_seval_to_llm_ndcg.py transform \
        --input_dir "../seval_data/123665_scraping_raw_data_output" \
        --output_file "llm_ndcg_input.jsonl" \
        --threads 16
"""

import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import fire

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SEVALToLLMNDCGTransformer:
    """
    Transform SEVAL raw data files to llm_ndcg input format.

    This class:
    1. Pairs control and experiment files by query_hash
    2. Extracts search results from the last turn
    3. Transforms to llm_ndcg input format
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize transformer.

        Args:
            verbose: If True, show detailed progress and statistics.
        """
        self.verbose = verbose
        self._file_locks: Dict[str, Lock] = {}
        self._file_locks_mutex = Lock()

    def _get_file_lock(self, file_path: str) -> Lock:
        """Get or create a lock for a specific file path."""
        normalized_path = os.path.normpath(file_path)
        with self._file_locks_mutex:
            if normalized_path not in self._file_locks:
                self._file_locks[normalized_path] = Lock()
            return self._file_locks[normalized_path]

    def _read_json_file_safely(self, file_path: str) -> Dict:
        """Safely read a JSON file with proper locking."""
        file_lock = self._get_file_lock(file_path)
        with file_lock:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def transform(
        self,
        input_dir: str,
        output_file: str,
        max_pairs: Optional[int] = None,
        threads: int = 8,
    ) -> int:
        """
        Transform SEVAL files to llm_ndcg format.

        Args:
            input_dir: Directory containing SEVAL control/experiment JSON files.
            output_file: Output JSONL file path.
            max_pairs: Maximum number of pairs to process (None for all).
            threads: Number of threads for parallel processing.

        Returns:
            Number of successfully transformed pairs.
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return 0

        # Step 1: Pair files by query_hash
        print("=" * 60)
        print("STEP 1: Pairing control and experiment files by query_hash")
        print("=" * 60)

        pairs = self._pair_files_by_query_hash(input_path, threads)

        complete_pairs = [(qh, info) for qh, info in pairs.items()
                         if info.get("control") and info.get("experiment")]

        print(f"\nPairing Statistics:")
        print(f"  Total unique queries: {len(pairs)}")
        print(f"  Complete pairs (control + experiment): {len(complete_pairs)}")
        print(f"  Control only: {sum(1 for info in pairs.values() if info.get('control') and not info.get('experiment'))}")
        print(f"  Experiment only: {sum(1 for info in pairs.values() if info.get('experiment') and not info.get('control'))}")

        if not complete_pairs:
            logger.error("No complete pairs found!")
            return 0

        # Limit pairs if requested
        if max_pairs and max_pairs < len(complete_pairs):
            complete_pairs = complete_pairs[:max_pairs]
            print(f"\n  Processing limited to {max_pairs} pairs")

        # Step 2: Transform each pair
        print("\n" + "=" * 60)
        print("STEP 2: Transforming pairs to llm_ndcg format")
        print("=" * 60)

        transformed_records = []
        errors = []
        # Track failure reasons for analysis - now stores (query_hash, control_uuid, experiment_uuid)
        failure_reasons: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Create mapping with file UUIDs for better tracking
            future_to_info = {}
            for query_hash, info in complete_pairs:
                control_uuid = self._extract_uuid_from_filepath(info["control"])
                experiment_uuid = self._extract_uuid_from_filepath(info["experiment"])
                future = executor.submit(
                    self._transform_pair_with_reason,
                    query_hash,
                    info["control"],
                    info["experiment"],
                    info.get("query_text", "")
                )
                future_to_info[future] = (query_hash, control_uuid, experiment_uuid)

            for i, future in enumerate(as_completed(future_to_info)):
                query_hash, control_uuid, experiment_uuid = future_to_info[future]
                try:
                    result, reason = future.result()
                    if result:
                        transformed_records.append(result)
                    else:
                        errors.append(query_hash)
                        failure_reasons[reason].append((query_hash, control_uuid, experiment_uuid))
                except Exception as e:
                    logger.warning(f"Error transforming {query_hash}: {e}")
                    errors.append(query_hash)
                    failure_reasons[f"exception: {type(e).__name__}"].append((query_hash, control_uuid, experiment_uuid))

                if self.verbose and (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(complete_pairs)} pairs...")

        # Step 3: Write output
        print("\n" + "=" * 60)
        print("STEP 3: Writing output")
        print("=" * 60)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for record in transformed_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"\nTransformation Complete:")
        print(f"  Successfully transformed: {len(transformed_records)}")
        print(f"  Errors/skipped: {len(errors)}")
        print(f"  Output file: {output_path.absolute()}")

        # Show failure analysis
        if failure_reasons:
            self._show_failure_analysis(failure_reasons)

        # Show sample statistics
        if transformed_records:
            self._show_sample_statistics(transformed_records)

        return len(transformed_records)

    def _extract_uuid_from_filepath(self, filepath: str) -> str:
        """
        Extract UUID from a SEVAL file path.

        File names are like: control_sydney_response_<UUID>.json
                         or: experiment_sydney_response_<UUID>.json

        Args:
            filepath: Full path to the SEVAL JSON file.

        Returns:
            The UUID string, or the filename if UUID cannot be extracted.
        """
        import re
        filename = Path(filepath).stem  # Get filename without extension
        # Pattern: control_sydney_response_<UUID> or experiment_sydney_response_<UUID>
        match = re.search(r'(?:control|experiment)_sydney_response_([a-f0-9-]+)', filename, re.IGNORECASE)
        if match:
            return match.group(1)
        # Fallback: return the whole filename
        return filename

    def _pair_files_by_query_hash(
        self,
        input_path: Path,
        threads: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Pair control and experiment files by query_hash.

        Args:
            input_path: Path to directory containing SEVAL files.
            threads: Number of threads for parallel processing.

        Returns:
            Dict mapping query_hash to file info.
        """
        control_files = list(input_path.glob("control_sydney_response_*.json"))
        experiment_files = list(input_path.glob("experiment_sydney_response_*.json"))

        print(f"  Found {len(control_files)} control files")
        print(f"  Found {len(experiment_files)} experiment files")

        pairs: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Process control files
        print("  Processing control files...")
        control_results = self._extract_query_info_parallel(
            control_files, "control", threads
        )
        for result in control_results:
            query_hash = result["query_hash"]
            pairs[query_hash]["control"] = result["filepath"]
            pairs[query_hash]["query_text"] = result.get("query_text", "")

        # Process experiment files
        print("  Processing experiment files...")
        experiment_results = self._extract_query_info_parallel(
            experiment_files, "experiment", threads
        )
        for result in experiment_results:
            query_hash = result["query_hash"]
            pairs[query_hash]["experiment"] = result["filepath"]
            if "query_text" not in pairs[query_hash]:
                pairs[query_hash]["query_text"] = result.get("query_text", "")

        return dict(pairs)

    def _extract_query_info_parallel(
        self,
        json_files: List[Path],
        exp_type: str,
        threads: int
    ) -> List[Dict[str, Any]]:
        """Extract query info from files in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_file = {
                executor.submit(self._extract_query_info, json_file): json_file
                for json_file in json_files
            }

            for future in as_completed(future_to_file):
                json_file = future_to_file[future]
                try:
                    query_info = future.result()
                    if query_info:
                        results.append(query_info)
                except Exception as e:
                    logger.warning(f"Error extracting from {json_file.name}: {e}")

        return results

    def _extract_query_info(self, json_file: Path) -> Optional[Dict[str, Any]]:
        """Extract query_hash and query_text from a single file."""
        try:
            data = self._read_json_file_safely(str(json_file))

            query_obj = data.get("query", {})
            query_hash = query_obj.get("query_hash", "")
            query_text = query_obj.get("id", "")

            if not query_hash:
                # Try to extract from messages if query object is missing
                messages = data.get("requests", [{}])[0].get("response_body", {}).get("messages", [])
                for msg in messages:
                    if msg.get("author") == "user":
                        query_text = msg.get("text", "")
                        # Create a hash from the query text
                        query_hash = str(hash(query_text.strip().lower()))
                        break

            if not query_hash:
                return None

            return {
                "filepath": str(json_file),
                "query_hash": query_hash,
                "query_text": query_text,
            }

        except Exception as e:
            logger.debug(f"Error reading {json_file.name}: {e}")
            return None

    def _transform_pair_with_reason(
        self,
        query_hash: str,
        control_file: str,
        experiment_file: str,
        query_text: str
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Transform a single control/experiment pair to llm_ndcg format with failure reason.

        Args:
            query_hash: The query hash for this pair.
            control_file: Path to control JSON file.
            experiment_file: Path to experiment JSON file.
            query_text: The query text.

        Returns:
            Tuple of (transformed record, failure_reason). If successful, failure_reason is empty.
        """
        try:
            control_data = self._read_json_file_safely(control_file)
            experiment_data = self._read_json_file_safely(experiment_file)

            # Extract search results from last turn with detailed info
            control_results, control_info = self._extract_search_results_last_turn_with_info(control_data)
            experiment_results, experiment_info = self._extract_search_results_last_turn_with_info(experiment_data)

            # Skip if both are empty - analyze why
            if not control_results and not experiment_results:
                reason = self._analyze_empty_results_reason(control_info, experiment_info)
                return None, reason

            # Get utterance and user profile
            utterance, user_profile = self._extract_utterance_and_profile(control_data)
            if not utterance:
                utterance, user_profile = self._extract_utterance_and_profile(experiment_data)

            if not utterance:
                utterance = query_text

            # Build the llm_ndcg format
            record = {
                "id": query_hash,
                "utterance": utterance,
                "user_profile": user_profile or {},
                "all_search_results_control": control_results,
                "all_search_results_treatment": experiment_results,
            }

            return record, ""

        except Exception as e:
            logger.debug(f"Error transforming pair {query_hash}: {e}")
            return None, f"exception: {type(e).__name__}: {str(e)[:50]}"

    def _transform_pair(
        self,
        query_hash: str,
        control_file: str,
        experiment_file: str,
        query_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Transform a single control/experiment pair to llm_ndcg format.
        (Legacy method - kept for backward compatibility)
        """
        result, _ = self._transform_pair_with_reason(query_hash, control_file, experiment_file, query_text)
        return result

    def _analyze_empty_results_reason(
        self,
        control_info: Dict[str, Any],
        experiment_info: Dict[str, Any]
    ) -> str:
        """
        Analyze why both control and experiment have empty results.

        Args:
            control_info: Diagnostic info from control extraction.
            experiment_info: Diagnostic info from experiment extraction.

        Returns:
            Reason string describing why results are empty.
        """
        control_reason = control_info.get("reason", "unknown")
        experiment_reason = experiment_info.get("reason", "unknown")

        # Combine reasons if different
        if control_reason == experiment_reason:
            return control_reason
        else:
            return f"ctrl:{control_reason}|exp:{experiment_reason}"

    def _show_failure_analysis(self, failure_reasons: Dict[str, List[Tuple[str, str, str]]]) -> None:
        """
        Display analysis of failure reasons.

        Args:
            failure_reasons: Dict mapping reason to list of (query_hash, control_uuid, experiment_uuid) tuples.
        """
        print("\n" + "=" * 60)
        print("FAILURE/SKIP ANALYSIS")
        print("=" * 60)

        total_failures = sum(len(items) for items in failure_reasons.values())
        print(f"\nTotal failed/skipped: {total_failures}")
        print(f"Number of distinct failure reasons: {len(failure_reasons)}")

        print("\nFailure Reasons Breakdown:")
        print("-" * 50)

        # Sort by count (most common first)
        sorted_reasons = sorted(
            failure_reasons.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        for reason, failure_items in sorted_reasons:
            count = len(failure_items)
            percentage = (count / total_failures) * 100 if total_failures > 0 else 0
            print(f"  {reason}:")
            print(f"    Count: {count} ({percentage:.1f}%)")
            # Show sample file UUIDs for each reason (first 3)
            if failure_items:
                print(f"    Sample files (control_uuid / experiment_uuid):")
                for query_hash, ctrl_uuid, exp_uuid in failure_items[:3]:
                    print(f"      - ctrl: {ctrl_uuid}")
                    print(f"        exp:  {exp_uuid}")
                if len(failure_items) > 3:
                    print(f"      ... and {len(failure_items) - 3} more")
            print()

    def _extract_utterance_and_profile(
        self,
        data: Dict[str, Any]
    ) -> Tuple[str, Optional[Dict]]:
        """
        Extract utterance and user profile from SEVAL data.

        Args:
            data: Parsed SEVAL JSON data.

        Returns:
            Tuple of (utterance, user_profile).
        """
        utterance = ""
        user_profile = None

        try:
            messages = data.get("requests", [{}])[0].get("response_body", {}).get("messages", [])

            # Get utterance from user message
            for msg in messages:
                if msg.get("author") == "user":
                    utterance = msg.get("text", "")
                    break

            # Get user profile from EvaluationData
            for msg in messages:
                if msg.get("messageType") == "EvaluationData":
                    eval_data = msg.get("evaluationData", {})
                    user_profile = eval_data.get("userProfile")
                    # Also try to get utterance from turnData if not found
                    if not utterance:
                        turn_data = eval_data.get("turnData", [])
                        if turn_data:
                            utterance = turn_data[0].get("userInput", "")
                    break

        except Exception as e:
            logger.debug(f"Error extracting utterance/profile: {e}")

        return utterance, user_profile

    def _extract_search_results_last_turn_with_info(
        self,
        data: Dict[str, Any]
    ) -> Tuple[Dict[str, Dict[str, List[Dict]]], Dict[str, Any]]:
        """
        Extract search results from the last turn with diagnostic information.

        This method returns both the search results and diagnostic info
        about why results might be empty.

        Args:
            data: Parsed SEVAL JSON data.

        Returns:
            Tuple of (search_results dict, diagnostic_info dict).
            diagnostic_info contains 'reason' key explaining empty results.
        """
        all_search_results: Dict[str, Dict[str, List[Dict]]] = {}
        diagnostic_info: Dict[str, Any] = {
            "reason": "",
            "has_messages": False,
            "has_eval_data": False,
            "has_turn_data": False,
            "num_turns": 0,
            "has_orchestration": False,
            "num_iterations": 0,
            "has_tool_invocations": False,
            "tool_functions_found": [],
            "has_batched_queries": False,
            "num_batched_queries": 0,
            "has_processed_results": False,
            "processed_result_types": [],
        }

        try:
            messages = data.get("requests", [{}])[0].get("response_body", {}).get("messages", [])
            diagnostic_info["has_messages"] = bool(messages)

            if not messages:
                diagnostic_info["reason"] = "no_messages"
                return all_search_results, diagnostic_info

            # Find EvaluationData message
            eval_data = None
            for msg in messages:
                if msg.get("messageType") == "EvaluationData":
                    eval_data = msg.get("evaluationData", {})
                    break

            diagnostic_info["has_eval_data"] = bool(eval_data)
            if not eval_data:
                diagnostic_info["reason"] = "no_evaluation_data"
                return all_search_results, diagnostic_info

            turn_data = eval_data.get("turnData", [])
            diagnostic_info["has_turn_data"] = bool(turn_data)
            diagnostic_info["num_turns"] = len(turn_data)

            if not turn_data:
                diagnostic_info["reason"] = "no_turn_data"
                return all_search_results, diagnostic_info

            # Use the LAST turn
            last_turn = turn_data[-1]
            orchestration_iterations = last_turn.get("orchestrationIterations", [])
            diagnostic_info["has_orchestration"] = bool(orchestration_iterations)
            diagnostic_info["num_iterations"] = len(orchestration_iterations)

            if not orchestration_iterations:
                diagnostic_info["reason"] = "no_orchestration_iterations"
                return all_search_results, diagnostic_info

            # Track iteration index (1-based for llm_ndcg format)
            for iter_idx, iteration in enumerate(orchestration_iterations, start=1):
                iter_key = str(iter_idx)
                iter_results: Dict[str, List[Dict]] = {}

                model_actions = iteration.get("modelActions", [])
                for action in model_actions:
                    tool_invocations = action.get("toolInvocations", [])
                    if tool_invocations:
                        diagnostic_info["has_tool_invocations"] = True

                    for invocation in tool_invocations:
                        function_name = invocation.get("function", "unknown_plugin")
                        diagnostic_info["tool_functions_found"].append(function_name)

                        # Parse arguments to get queries with domain info
                        arguments = invocation.get("arguments", {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {}
                        
                        # Get list of queries from arguments (for domain info)
                        arg_queries = arguments.get("queries", [])

                        # Try two formats:
                        # Format 1: processedResult directly on invocation
                        # Format 2: processedResult inside batchedQueries
                        
                        # Format 1: Direct processedResult on invocation
                        direct_processed_result = invocation.get("processedResult", "")
                        if direct_processed_result:
                            diagnostic_info["has_processed_results"] = True
                            # Track the type/structure of processed results
                            try:
                                if isinstance(direct_processed_result, str):
                                    pr_data = json.loads(direct_processed_result)
                                else:
                                    pr_data = direct_processed_result
                                pr_keys = list(pr_data.keys()) if isinstance(pr_data, dict) else ["non_dict"]
                                diagnostic_info["processed_result_types"].extend(pr_keys[:5])
                            except:
                                diagnostic_info["processed_result_types"].append("parse_error")

                            results, result_type = self._parse_processed_result({"processedResult": direct_processed_result})
                            if results:
                                # Extract domain and query from REQUEST arguments
                                # Different tools have different argument structures:
                                #   - office365_search: args.queries[].domain, args.queries[].query
                                #   - search_web: args.query (no domain)
                                #   - fetch_file: no domain, no query
                                #   - search_enterprise_connectors: args.query (no domain in request)
                                domain = ""
                                query = ""
                                
                                # Method 1: Check args.queries[] (office365_search style)
                                if arg_queries and isinstance(arg_queries[0], dict):
                                    domain = arg_queries[0].get("domain", "")
                                    query = arg_queries[0].get("query", "")
                                
                                # Method 2: Check args.query (search_web, search_enterprise_connectors style)
                                if not query:
                                    query = arguments.get("query", "")
                                
                                plugin_invocation = self._build_plugin_invocation(
                                    invocation, domain=domain, query=query
                                )
                                
                                if function_name not in iter_results:
                                    iter_results[function_name] = []
                                iter_results[function_name].append({
                                    "Results": results,
                                    "PluginInvocation": plugin_invocation,
                                    "ResultType": result_type,
                                })

                        # Format 2: batchedQueries with processedResult
                        batched_queries = invocation.get("batchedQueries", [])
                        if batched_queries:
                            diagnostic_info["has_batched_queries"] = True
                            diagnostic_info["num_batched_queries"] += len(batched_queries)

                        for idx, batch_query in enumerate(batched_queries):
                            processed_result = batch_query.get("processedResult", "")
                            if processed_result:
                                diagnostic_info["has_processed_results"] = True
                                # Track the type/structure of processed results
                                try:
                                    if isinstance(processed_result, str):
                                        pr_data = json.loads(processed_result)
                                    else:
                                        pr_data = processed_result
                                    pr_keys = list(pr_data.keys()) if isinstance(pr_data, dict) else ["non_dict"]
                                    diagnostic_info["processed_result_types"].extend(pr_keys[:5])
                                except:
                                    diagnostic_info["processed_result_types"].append("parse_error")

                            results, result_type = self._parse_processed_result(batch_query)
                            if results:
                                # Extract domain and query from REQUEST arguments
                                # Priority: batch_query fields > arg_queries[idx] > args.query
                                domain = batch_query.get("domain", "")
                                query = batch_query.get("query", "")
                                
                                # If not in batch_query, try to get from corresponding arg_query
                                if not domain and idx < len(arg_queries) and isinstance(arg_queries[idx], dict):
                                    domain = arg_queries[idx].get("domain", "")
                                if not query and idx < len(arg_queries) and isinstance(arg_queries[idx], dict):
                                    query = arg_queries[idx].get("query", "")
                                
                                # Fallback: try args.query (search_web, search_enterprise_connectors style)
                                if not query:
                                    query = arguments.get("query", "")
                                
                                plugin_invocation = self._build_plugin_invocation(
                                    invocation, domain=domain, query=query
                                )
                                
                                if function_name not in iter_results:
                                    iter_results[function_name] = []
                                iter_results[function_name].append({
                                    "Results": results,
                                    "PluginInvocation": plugin_invocation,
                                    "ResultType": result_type,
                                })

                if iter_results:
                    all_search_results[iter_key] = iter_results

            # Determine reason if still empty
            if not all_search_results:
                if not diagnostic_info["has_tool_invocations"]:
                    diagnostic_info["reason"] = "no_tool_invocations"
                elif not diagnostic_info["has_processed_results"]:
                    # No processedResult found in either format (direct or batchedQueries)
                    diagnostic_info["reason"] = "no_processed_results"
                else:
                    # Has processed results but none parsed successfully
                    unique_types = list(set(diagnostic_info["processed_result_types"]))[:5]
                    diagnostic_info["reason"] = f"unparseable_results:{','.join(unique_types)}"
            else:
                diagnostic_info["reason"] = ""

        except Exception as e:
            logger.debug(f"Error extracting search results: {e}")
            diagnostic_info["reason"] = f"exception:{type(e).__name__}"

        return all_search_results, diagnostic_info

    def _build_plugin_invocation(
        self,
        invocation: Dict[str, Any],
        domain: Optional[str] = None,
        query: Optional[str] = None
    ) -> str:
        """
        Build a PluginInvocation string from the tool invocation data.

        This method is generic and supports any tool type. The format preserves
        domain and query information from the REQUEST where available:

        Supported tool types:
            - office365_search: 'office365_search(domain="meetings", query="...")'
              Domain from request args.queries[].domain
            - search_web: 'search_web(query="...")'
              No domain in request, query from args.query
            - fetch_file: 'fetch_file()'
              No domain or query in request
            - search_enterprise_connectors_*: 'search_enterprise_connectors_abc(query="...")'
              No domain in request, query from args.query
            - Any other tool: 'tool_name(...)' with whatever params are provided

        Note: result_type is NOT included here because it's derived from the RESPONSE,
        which may differ between control and experiment. result_type is stored
        separately in the ResultType field for informational purposes.

        Args:
            invocation: The toolInvocation dict from SEVAL data.
            domain: Optional domain (for office365_search: emails, files, etc.).
            query: Optional query text.

        Returns:
            PluginInvocation string.
        """
        function_name = invocation.get("function", "unknown_plugin")
        
        # Build parts list based on what's available from the REQUEST
        parts = []
        
        # Use domain for office365_search (from request arguments)
        if domain:
            parts.append(f'domain="{domain}"')
        
        if query:
            parts.append(f'query="{query}"')
        
        if parts:
            return f'{function_name}({", ".join(parts)})'
        else:
            return f"{function_name}()"

    def _parse_processed_result(
        self,
        batch_query: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Parse processedResult from a batchedQuery into llm_ndcg result format.

        This method handles multiple result formats from different tool types:
            - office365_search: "results" array
            - search_web: "WebPages", "News", "Sports", "QuestionsAndAnswers" arrays
            - fetch_file: "results" array
            - search_enterprise_connectors_*: "results" array
            - Any new format: Falls back to finding first list-valued key

        Args:
            batch_query: A batchedQuery dict containing processedResult.

        Returns:
            Tuple of (results list in llm_ndcg format, result_type string).
            result_type indicates the source key name (e.g., "results", "WebPages", etc.)
        """
        results = []
        result_type = ""

        processed_result = batch_query.get("processedResult", "")
        if not processed_result:
            return results, result_type

        try:
            # processedResult is often a JSON string
            if isinstance(processed_result, str):
                processed_data = json.loads(processed_result)
            else:
                processed_data = processed_result

            # Extract results from various possible locations
            raw_results = []

            # Known result keys in priority order
            # "results" is most common (office365_search, fetch_file, search_enterprise_connectors)
            # Web search types follow
            known_result_keys = [
                "results",           # office365_search, fetch_file, search_enterprise_connectors
                "WebPages",          # search_web
                "News",              # search_web
                "Sports",            # search_web
                "QuestionsAndAnswers",  # search_web
                "Images",            # search_web (potential)
                "Videos",            # search_web (potential)
                "RelatedSearches",   # search_web (potential)
            ]

            # Try known keys first
            for key in known_result_keys:
                if key in processed_data and isinstance(processed_data[key], list):
                    raw_results = processed_data[key]
                    result_type = key
                    break

            # Fallback: find any list-valued key for unknown/new result formats
            if not raw_results and isinstance(processed_data, dict):
                for key, value in processed_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        # Check if it looks like search results (list of dicts)
                        if isinstance(value[0], dict):
                            raw_results = value
                            result_type = key
                            logger.debug(f"Using fallback result key: {key}")
                            break

            # Transform each result to llm_ndcg format
            for raw_result in raw_results:
                transformed = self._transform_single_result(raw_result)
                if transformed:
                    results.append(transformed)

        except json.JSONDecodeError:
            logger.debug("Failed to parse processedResult as JSON")
        except Exception as e:
            logger.debug(f"Error parsing processedResult: {e}")

        return results, result_type

    def _transform_single_result(
        self,
        raw_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Transform a single search result to llm_ndcg format.

        SEVAL format fields:
            - reference_id: Unique ID
            - result.type: Document type
            - result.title: Title
            - result.snippet: Content snippet
            - result.author: Author
            - result.fileName: File name

        llm_ndcg format fields:
            - Id: Unique ID
            - ContentBody: Content/snippet
            - Title: Title
            - DocType: Document type

        Args:
            raw_result: Raw search result from SEVAL.

        Returns:
            Transformed result in llm_ndcg format.
        """
        try:
            # Get the reference_id (primary identifier)
            ref_id = raw_result.get("reference_id", "")
            if not ref_id:
                ref_id = raw_result.get("id", "")

            # Get nested result data if present
            result_data = raw_result.get("result", raw_result)

            # Extract fields
            title = result_data.get("title", "")
            snippet = result_data.get("snippet", "")
            doc_type = result_data.get("type", "")
            author = result_data.get("author", "")
            file_name = result_data.get("fileName", "")

            # Build llm_ndcg format result
            transformed = {
                "Id": ref_id,
                "ContentBody": snippet,
                "Title": title,
                "DocType": doc_type,
            }

            # Add optional fields if present
            if author:
                transformed["Author"] = author
            if file_name:
                transformed["FileName"] = file_name

            # Only return if we have meaningful content
            if ref_id or title or snippet:
                return transformed

            return None

        except Exception as e:
            logger.debug(f"Error transforming result: {e}")
            return None

    def _show_sample_statistics(self, records: List[Dict[str, Any]]) -> None:
        """Show statistics about the transformed records."""
        print("\n" + "=" * 60)
        print("SAMPLE STATISTICS")
        print("=" * 60)

        # Count plugins
        control_plugins: Dict[str, int] = defaultdict(int)
        treatment_plugins: Dict[str, int] = defaultdict(int)
        total_control_results = 0
        total_treatment_results = 0

        for record in records:
            for iter_key, plugins in record.get("all_search_results_control", {}).items():
                for plugin, result_lists in plugins.items():
                    for result_list in result_lists:
                        count = len(result_list.get("Results", []))
                        control_plugins[plugin] += count
                        total_control_results += count

            for iter_key, plugins in record.get("all_search_results_treatment", {}).items():
                for plugin, result_lists in plugins.items():
                    for result_list in result_lists:
                        count = len(result_list.get("Results", []))
                        treatment_plugins[plugin] += count
                        total_treatment_results += count

        print("\nControl Results:")
        print("  Total results across all records: {total_control_results}")
        print("  Results by plugin:")
        for plugin, count in sorted(control_plugins.items(), key=lambda x: -x[1]):
            print(f"    - {plugin}: {count}")

        print("\nTreatment Results:")
        print(f"  Total results across all records: {total_treatment_results}")
        print("  Results by plugin:")
        for plugin, count in sorted(treatment_plugins.items(), key=lambda x: -x[1]):
            print(f"    - {plugin}: {count}")

        # Show a sample record structure
        if records:
            sample = records[0]
            print("\nSample Record Structure:")
            print(f"  id: {sample.get('id', '')[:50]}...")
            print(f"  utterance: {sample.get('utterance', '')[:80]}...")
            print(f"  user_profile keys: {list(sample.get('user_profile', {}).keys())[:5]}")
            print(f"  control iterations: {list(sample.get('all_search_results_control', {}).keys())}")
            print(f"  treatment iterations: {list(sample.get('all_search_results_treatment', {}).keys())}")


def transform(
    input_dir: str,
    output_file: str,
    max_pairs: Optional[int] = None,
    threads: int = 8,
    verbose: bool = True,
) -> int:
    """
    Transform SEVAL files to llm_ndcg format.

    Args:
        input_dir: Directory containing SEVAL control/experiment JSON files.
        output_file: Output JSONL file path.
        max_pairs: Maximum number of pairs to process (None for all).
        threads: Number of threads for parallel processing.
        verbose: Show detailed progress.

    Returns:
        Number of successfully transformed pairs.
    """
    transformer = SEVALToLLMNDCGTransformer(verbose=verbose)
    return transformer.transform(input_dir, output_file, max_pairs, threads)


if __name__ == "__main__":
    fire.Fire({
        "transform": transform,
    })
