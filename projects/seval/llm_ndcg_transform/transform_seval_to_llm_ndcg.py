# transform_seval_with_dedup.py
#
# Copyright (c) Microsoft Corporation. All rights reserved.

"""
Transform SEVAL raw data files to llm_ndcg input format WITH DEDUPLICATION FIX.

This is a fixed version of transform_seval_to_llm_ndcg.py that deduplicates
results within the same tool invocation using (query, reference_id) tuples.

Author: GitHub Copilot
Created: 2026-01-27
Description: Transforms SEVAL control/treatment file pairs into llm_ndcg format,
             with proper deduplication to avoid extracting same results from both
             direct processedResult and batchedQueries formats.

Usage:
    python transform_seval_with_dedup.py transform \
        --input_dir "\\\\m365fileservices\\m365seval\\SevalRawData\\144683_scraping_raw_data_output" \
        --output_file "llm_ndcg_input_deduped.jsonl" \
        --max_pairs 10
"""

import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import fire

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SEVALToLLMNDCGTransformerWithDedup:
    """
    Transform SEVAL raw data files to llm_ndcg input format WITH DEDUPLICATION.

    This class:
    1. Pairs control and experiment files by query_hash
    2. Extracts search results from the last turn
    3. DEDUPLICATES results within same tool invocation using (query, reference_id)
    4. Transforms to llm_ndcg input format
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
        
        # Dedup statistics
        self.dedup_stats = {
            "total_results_before_dedup": 0,
            "total_results_after_dedup": 0,
            "total_duplicates_avoided": 0,
        }

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
        Transform SEVAL files to llm_ndcg format with deduplication.

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
        print("STEP 2: Transforming pairs to llm_ndcg format (WITH DEDUP)")
        print("=" * 60)

        transformed_records = []
        errors = []
        failure_reasons: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

        with ThreadPoolExecutor(max_workers=threads) as executor:
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

        # Show dedup statistics
        print("\n" + "=" * 60)
        print("DEDUPLICATION STATISTICS")
        print("=" * 60)
        print(f"  Results before dedup: {self.dedup_stats['total_results_before_dedup']}")
        print(f"  Results after dedup:  {self.dedup_stats['total_results_after_dedup']}")
        print(f"  Duplicates avoided:   {self.dedup_stats['total_duplicates_avoided']}")
        if self.dedup_stats['total_results_before_dedup'] > 0:
            reduction_pct = (self.dedup_stats['total_duplicates_avoided'] / 
                           self.dedup_stats['total_results_before_dedup'] * 100)
            print(f"  Reduction percentage: {reduction_pct:.1f}%")

        # Show failure analysis
        if failure_reasons:
            self._show_failure_analysis(failure_reasons)

        # Show sample statistics
        if transformed_records:
            self._show_sample_statistics(transformed_records)

        return len(transformed_records)

    def _extract_uuid_from_filepath(self, filepath: str) -> str:
        """Extract UUID from a SEVAL file path."""
        import re
        filename = Path(filepath).stem
        match = re.search(r'(?:control|experiment)_sydney_response_([a-f0-9-]+)', filename, re.IGNORECASE)
        if match:
            return match.group(1)
        return filename

    def _pair_files_by_query_hash(
        self,
        input_path: Path,
        threads: int
    ) -> Dict[str, Dict[str, Any]]:
        """Pair control and experiment files by query_hash."""
        control_files = list(input_path.glob("control_sydney_response_*.json"))
        experiment_files = list(input_path.glob("experiment_sydney_response_*.json"))

        print(f"  Found {len(control_files)} control files")
        print(f"  Found {len(experiment_files)} experiment files")

        pairs: Dict[str, Dict[str, Any]] = defaultdict(dict)

        print("  Processing control files...")
        control_results = self._extract_query_info_parallel(control_files, "control", threads)
        for result in control_results:
            query_hash = result["query_hash"]
            pairs[query_hash]["control"] = result["filepath"]
            pairs[query_hash]["query_text"] = result.get("query_text", "")

        print("  Processing experiment files...")
        experiment_results = self._extract_query_info_parallel(experiment_files, "experiment", threads)
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
                try:
                    query_info = future.result()
                    if query_info:
                        results.append(query_info)
                except Exception:
                    pass
        return results

    def _extract_query_info(self, json_file: Path) -> Optional[Dict[str, Any]]:
        """Extract query_hash and query_text from a single file."""
        try:
            data = self._read_json_file_safely(str(json_file))
            query_obj = data.get("query", {})
            query_hash = query_obj.get("query_hash", "")
            query_text = query_obj.get("id", "")

            if not query_hash:
                messages = data.get("requests", [{}])[0].get("response_body", {}).get("messages", [])
                for msg in messages:
                    if msg.get("author") == "user":
                        query_text = msg.get("text", "")
                        query_hash = str(hash(query_text.strip().lower()))
                        break

            if not query_hash:
                return None

            return {
                "filepath": str(json_file),
                "query_hash": query_hash,
                "query_text": query_text,
            }
        except Exception:
            return None

    def _transform_pair_with_reason(
        self,
        query_hash: str,
        control_file: str,
        experiment_file: str,
        query_text: str
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Transform a single control/experiment pair to llm_ndcg format."""
        try:
            control_data = self._read_json_file_safely(control_file)
            experiment_data = self._read_json_file_safely(experiment_file)

            control_results, control_info = self._extract_search_results_last_turn_with_info(control_data)
            experiment_results, experiment_info = self._extract_search_results_last_turn_with_info(experiment_data)

            if not control_results and not experiment_results:
                reason = self._analyze_empty_results_reason(control_info, experiment_info)
                return None, reason

            utterance, user_profile = self._extract_utterance_and_profile(control_data)
            if not utterance:
                utterance, user_profile = self._extract_utterance_and_profile(experiment_data)
            if not utterance:
                utterance = query_text

            record = {
                "id": query_hash,
                "utterance": utterance,
                "user_profile": user_profile or {},
                "all_search_results_control": control_results,
                "all_search_results_treatment": experiment_results,
            }

            return record, ""
        except Exception as e:
            return None, f"exception: {type(e).__name__}: {str(e)[:50]}"

    def _analyze_empty_results_reason(
        self,
        control_info: Dict[str, Any],
        experiment_info: Dict[str, Any]
    ) -> str:
        """Analyze why both control and experiment have empty results."""
        control_reason = control_info.get("reason", "unknown")
        experiment_reason = experiment_info.get("reason", "unknown")
        if control_reason == experiment_reason:
            return control_reason
        return f"ctrl:{control_reason}|exp:{experiment_reason}"

    def _show_failure_analysis(self, failure_reasons: Dict[str, List[Tuple[str, str, str]]]) -> None:
        """Display analysis of failure reasons."""
        print("\n" + "=" * 60)
        print("FAILURE/SKIP ANALYSIS")
        print("=" * 60)
        total_failures = sum(len(items) for items in failure_reasons.values())
        print(f"\nTotal failed/skipped: {total_failures}")
        sorted_reasons = sorted(failure_reasons.items(), key=lambda x: len(x[1]), reverse=True)
        for reason, failure_items in sorted_reasons[:5]:
            print(f"  {reason}: {len(failure_items)}")

    def _extract_utterance_and_profile(
        self,
        data: Dict[str, Any]
    ) -> Tuple[str, Optional[Dict]]:
        """Extract utterance and user profile from SEVAL data."""
        utterance = ""
        user_profile = None
        try:
            messages = data.get("requests", [{}])[0].get("response_body", {}).get("messages", [])
            for msg in messages:
                if msg.get("author") == "user":
                    utterance = msg.get("text", "")
                    break
            for msg in messages:
                if msg.get("messageType") == "EvaluationData":
                    eval_data = msg.get("evaluationData", {})
                    user_profile = eval_data.get("userProfile")
                    if not utterance:
                        turn_data = eval_data.get("turnData", [])
                        if turn_data:
                            utterance = turn_data[0].get("userInput", "")
                    break
        except Exception:
            pass
        return utterance, user_profile

    def _extract_search_results_last_turn_with_info(
        self,
        data: Dict[str, Any]
    ) -> Tuple[Dict[str, Dict[str, List[Dict]]], Dict[str, Any]]:
        """
        Extract search results from the last turn with DEDUPLICATION.

        This method uses (query, reference_id) tuples to deduplicate results
        within the same tool invocation, avoiding double-counting when both
        direct processedResult and batchedQueries formats exist.

        Args:
            data: Parsed SEVAL JSON data.

        Returns:
            Tuple of (search_results dict, diagnostic_info dict).
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
            # Dedup stats for this file
            "results_before_dedup": 0,
            "results_after_dedup": 0,
            "duplicates_avoided": 0,
        }

        try:
            messages = data.get("requests", [{}])[0].get("response_body", {}).get("messages", [])
            diagnostic_info["has_messages"] = bool(messages)

            if not messages:
                diagnostic_info["reason"] = "no_messages"
                return all_search_results, diagnostic_info

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

            last_turn = turn_data[-1]
            orchestration_iterations = last_turn.get("orchestrationIterations", [])
            diagnostic_info["has_orchestration"] = bool(orchestration_iterations)
            diagnostic_info["num_iterations"] = len(orchestration_iterations)

            if not orchestration_iterations:
                diagnostic_info["reason"] = "no_orchestration_iterations"
                return all_search_results, diagnostic_info

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
                        
                        arg_queries = arguments.get("queries", [])
                        
                        # === DEDUPLICATION FIX ===
                        # Avoid extracting duplicates from BOTH direct processedResult AND
                        # batchedQueries when they contain the same results.
                        #
                        # Strategy:
                        # 1. Check if reference_id is unique within each format
                        # 2. If unique in both → use simple reference_id deduplication
                        # 3. If reused → split by query and dedupe within matching subsets
                        #
                        # Scope preserved:
                        # - Different iterations: NOT deduplicated (processed separately)
                        # - Different tool invocations: NOT deduplicated (different invocations)
                        # - Same tool, different queries: NOT deduplicated (different query subsets)
                        # - Same query appearing in both formats: DEDUPLICATED by reference_id
                        
                        batched_queries = invocation.get("batchedQueries", [])
                        direct_processed_result = invocation.get("processedResult", "")
                        
                        # Step 1: Analyze reference_id usage in both formats
                        direct_ref_ids: Dict[str, int] = defaultdict(int)  # ref_id -> count
                        batched_ref_ids: Dict[str, int] = defaultdict(int)  # ref_id -> count
                        batched_ref_ids_by_query: Dict[str, Set[str]] = defaultdict(set)  # query_key -> ref_ids
                        
                        # Collect ref_ids from direct format
                        direct_results_parsed = []
                        if direct_processed_result:
                            try:
                                if isinstance(direct_processed_result, str):
                                    direct_data = json.loads(direct_processed_result)
                                else:
                                    direct_data = direct_processed_result
                                
                                for key in ["results", "WebPages", "News", "Sports", "QuestionsAndAnswers"]:
                                    if key in direct_data and isinstance(direct_data[key], list):
                                        direct_results_parsed = direct_data[key]
                                        break
                                
                                if not direct_results_parsed and isinstance(direct_data, dict):
                                    for key, value in direct_data.items():
                                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                                            direct_results_parsed = value
                                            break
                                
                                for result in direct_results_parsed:
                                    result_data = result.get("result", result)
                                    ref_id = result.get("reference_id", "") or result_data.get("reference_id", "")
                                    if ref_id:
                                        direct_ref_ids[ref_id] += 1
                            except:
                                pass
                        
                        # Collect ref_ids from batched format (per query)
                        batched_parsed_by_idx: Dict[int, List[Dict]] = {}
                        for idx, batch_query in enumerate(batched_queries):
                            batch_pr = batch_query.get("processedResult", "")
                            if batch_pr:
                                try:
                                    if isinstance(batch_pr, str):
                                        batch_data = json.loads(batch_pr)
                                    else:
                                        batch_data = batch_pr
                                    
                                    batch_results = []
                                    for key in ["results", "WebPages", "News", "Sports", "QuestionsAndAnswers"]:
                                        if key in batch_data and isinstance(batch_data[key], list):
                                            batch_results = batch_data[key]
                                            break
                                    
                                    if not batch_results and isinstance(batch_data, dict):
                                        for key, value in batch_data.items():
                                            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                                                batch_results = value
                                                break
                                    
                                    batched_parsed_by_idx[idx] = batch_results
                                    
                                    # Get query key for this batch
                                    query_string = batch_query.get("query", "")
                                    if not query_string and idx < len(arg_queries) and isinstance(arg_queries[idx], dict):
                                        query_string = arg_queries[idx].get("query", "")
                                    domain = batch_query.get("domain", "")
                                    if not domain and idx < len(arg_queries) and isinstance(arg_queries[idx], dict):
                                        domain = arg_queries[idx].get("domain", "")
                                    query_key = f"{domain}|{query_string}"
                                    
                                    for result in batch_results:
                                        result_data = result.get("result", result)
                                        ref_id = result.get("reference_id", "") or result_data.get("reference_id", "")
                                        if ref_id:
                                            batched_ref_ids[ref_id] += 1
                                            batched_ref_ids_by_query[query_key].add(ref_id)
                                except:
                                    batched_parsed_by_idx[idx] = []
                        
                        # Step 2: Check for reference_id reuse within each format
                        direct_has_reuse = any(count > 1 for count in direct_ref_ids.values())
                        batched_has_reuse = any(count > 1 for count in batched_ref_ids.values())
                        
                        # Step 3: Determine dedup strategy
                        # If no reuse in either format, we can use simple ref_id dedup
                        # If reuse exists, we need to be careful about which subset to check
                        
                        if not direct_has_reuse and not batched_has_reuse:
                            # Simple case: ref_id is unique within each format
                            # Just check if ref_id from batched exists in direct's set
                            seen_ref_ids: Set[str] = set(direct_ref_ids.keys())
                            
                            # Process direct format
                            if direct_processed_result and direct_results_parsed:
                                diagnostic_info["has_processed_results"] = True
                                
                                direct_domain = ""
                                direct_query_display = ""
                                if arg_queries and isinstance(arg_queries[0], dict):
                                    direct_domain = arg_queries[0].get("domain", "")
                                    direct_query_display = arg_queries[0].get("query", "")
                                if not direct_query_display:
                                    direct_query_display = arguments.get("query", "")
                                
                                results, result_type = self._transform_results_list(direct_results_parsed)
                                diagnostic_info["results_before_dedup"] += len(direct_results_parsed)
                                diagnostic_info["results_after_dedup"] += len(results)
                                
                                if results:
                                    plugin_invocation = self._build_plugin_invocation(
                                        invocation, domain=direct_domain, query=direct_query_display
                                    )
                                    if function_name not in iter_results:
                                        iter_results[function_name] = []
                                    iter_results[function_name].append({
                                        "Results": results,
                                        "PluginInvocation": plugin_invocation,
                                        "ResultType": result_type,
                                    })
                            
                            # Process batched format, skip ref_ids already in direct
                            if batched_queries:
                                diagnostic_info["has_batched_queries"] = True
                                diagnostic_info["num_batched_queries"] += len(batched_queries)
                                
                                for idx, batch_query in enumerate(batched_queries):
                                    if idx not in batched_parsed_by_idx:
                                        continue
                                    
                                    batch_results = batched_parsed_by_idx[idx]
                                    if not batch_results:
                                        continue
                                    
                                    diagnostic_info["has_processed_results"] = True
                                    
                                    query_string = batch_query.get("query", "")
                                    if not query_string and idx < len(arg_queries) and isinstance(arg_queries[idx], dict):
                                        query_string = arg_queries[idx].get("query", "")
                                    if not query_string:
                                        query_string = arguments.get("query", "")
                                    
                                    domain = batch_query.get("domain", "")
                                    if not domain and idx < len(arg_queries) and isinstance(arg_queries[idx], dict):
                                        domain = arg_queries[idx].get("domain", "")
                                    
                                    # Filter out results already seen in direct
                                    filtered_results = []
                                    for result in batch_results:
                                        result_data = result.get("result", result)
                                        ref_id = result.get("reference_id", "") or result_data.get("reference_id", "")
                                        
                                        diagnostic_info["results_before_dedup"] += 1
                                        
                                        if ref_id and ref_id in seen_ref_ids:
                                            diagnostic_info["duplicates_avoided"] += 1
                                            continue
                                        
                                        if ref_id:
                                            seen_ref_ids.add(ref_id)
                                        
                                        transformed = self._transform_single_result(result)
                                        if transformed:
                                            filtered_results.append(transformed)
                                            diagnostic_info["results_after_dedup"] += 1
                                    
                                    if filtered_results:
                                        plugin_invocation = self._build_plugin_invocation(
                                            invocation, domain=domain, query=query_string
                                        )
                                        if function_name not in iter_results:
                                            iter_results[function_name] = []
                                        iter_results[function_name].append({
                                            "Results": filtered_results,
                                            "PluginInvocation": plugin_invocation,
                                            "ResultType": "results",
                                        })
                        
                        else:
                            # Complex case: ref_id is reused within a format
                            # Need to check by query subset
                            # For each batched query, only dedupe against direct results that
                            # match the same query (if we can determine it)
                            
                            # Since direct doesn't have per-query info, we use a conservative approach:
                            # Only dedupe if the SAME ref_id appears in BOTH formats
                            # and the result count matches (suggesting they're the same)
                            
                            all_direct_ref_ids = set(direct_ref_ids.keys())
                            all_batched_ref_ids = set(batched_ref_ids.keys())
                            overlapping_ref_ids = all_direct_ref_ids & all_batched_ref_ids
                            
                            # Process direct format
                            if direct_processed_result and direct_results_parsed:
                                diagnostic_info["has_processed_results"] = True
                                
                                direct_domain = ""
                                direct_query_display = ""
                                if arg_queries and isinstance(arg_queries[0], dict):
                                    direct_domain = arg_queries[0].get("domain", "")
                                    direct_query_display = arg_queries[0].get("query", "")
                                if not direct_query_display:
                                    direct_query_display = arguments.get("query", "")
                                
                                results, result_type = self._transform_results_list(direct_results_parsed)
                                diagnostic_info["results_before_dedup"] += len(direct_results_parsed)
                                diagnostic_info["results_after_dedup"] += len(results)
                                
                                if results:
                                    plugin_invocation = self._build_plugin_invocation(
                                        invocation, domain=direct_domain, query=direct_query_display
                                    )
                                    if function_name not in iter_results:
                                        iter_results[function_name] = []
                                    iter_results[function_name].append({
                                        "Results": results,
                                        "PluginInvocation": plugin_invocation,
                                        "ResultType": result_type,
                                    })
                            
                            # Process batched format
                            if batched_queries:
                                diagnostic_info["has_batched_queries"] = True
                                diagnostic_info["num_batched_queries"] += len(batched_queries)
                                
                                for idx, batch_query in enumerate(batched_queries):
                                    if idx not in batched_parsed_by_idx:
                                        continue
                                    
                                    batch_results = batched_parsed_by_idx[idx]
                                    if not batch_results:
                                        continue
                                    
                                    diagnostic_info["has_processed_results"] = True
                                    
                                    query_string = batch_query.get("query", "")
                                    if not query_string and idx < len(arg_queries) and isinstance(arg_queries[idx], dict):
                                        query_string = arg_queries[idx].get("query", "")
                                    if not query_string:
                                        query_string = arguments.get("query", "")
                                    
                                    domain = batch_query.get("domain", "")
                                    if not domain and idx < len(arg_queries) and isinstance(arg_queries[idx], dict):
                                        domain = arg_queries[idx].get("domain", "")
                                    
                                    # Only skip results whose ref_id overlaps with direct
                                    filtered_results = []
                                    for result in batch_results:
                                        result_data = result.get("result", result)
                                        ref_id = result.get("reference_id", "") or result_data.get("reference_id", "")
                                        
                                        diagnostic_info["results_before_dedup"] += 1
                                        
                                        # Skip if this ref_id exists in both direct and batched (it's a duplicate)
                                        if ref_id and ref_id in overlapping_ref_ids:
                                            diagnostic_info["duplicates_avoided"] += 1
                                            continue
                                        
                                        transformed = self._transform_single_result(result)
                                        if transformed:
                                            filtered_results.append(transformed)
                                            diagnostic_info["results_after_dedup"] += 1
                                    
                                    if filtered_results:
                                        plugin_invocation = self._build_plugin_invocation(
                                            invocation, domain=domain, query=query_string
                                        )
                                        if function_name not in iter_results:
                                            iter_results[function_name] = []
                                        iter_results[function_name].append({
                                            "Results": filtered_results,
                                            "PluginInvocation": plugin_invocation,
                                            "ResultType": "results",
                                        })

                if iter_results:
                    all_search_results[iter_key] = iter_results

            # Update global dedup stats
            self.dedup_stats["total_results_before_dedup"] += diagnostic_info["results_before_dedup"]
            self.dedup_stats["total_results_after_dedup"] += diagnostic_info["results_after_dedup"]
            self.dedup_stats["total_duplicates_avoided"] += diagnostic_info["duplicates_avoided"]

            if not all_search_results:
                if not diagnostic_info["has_tool_invocations"]:
                    diagnostic_info["reason"] = "no_tool_invocations"
                elif not diagnostic_info["has_processed_results"]:
                    diagnostic_info["reason"] = "no_processed_results"
                else:
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
        """Build a PluginInvocation string from the tool invocation data."""
        function_name = invocation.get("function", "unknown_plugin")
        parts = []
        if domain:
            parts.append(f'domain="{domain}"')
        if query:
            parts.append(f'query="{query}"')
        if parts:
            return f'{function_name}({", ".join(parts)})'
        return f"{function_name}()"

    def _parse_processed_result_with_dedup(
        self,
        batch_query: Dict[str, Any],
        seen_ref_ids: Set[str]
    ) -> Tuple[List[Dict[str, Any]], str, int, int]:
        """
        Parse processedResult with deduplication based on reference_id.

        Deduplication scope: same tool invocation (same tool call in same iteration).
        - Results from different iterations are NOT deduplicated
        - Results from different tool invocations are NOT deduplicated  
        - Results from same invocation appearing in both direct and batched 
          formats ARE deduplicated using reference_id

        Our analysis confirmed: direct processedResult is a SUBSET of batchedQueries
        results, with the SAME reference_id for matching results. So we can safely
        dedupe by reference_id within each invocation.

        Args:
            batch_query: A batchedQuery dict containing processedResult.
            seen_ref_ids: Set of already seen reference_ids within this invocation.

        Returns:
            Tuple of (results list, result_type, count_before_dedup, count_after_dedup).
        """
        results = []
        result_type = ""
        count_before_dedup = 0
        count_after_dedup = 0

        processed_result = batch_query.get("processedResult", "")
        if not processed_result:
            return results, result_type, count_before_dedup, count_after_dedup

        try:
            if isinstance(processed_result, str):
                processed_data = json.loads(processed_result)
            else:
                processed_data = processed_result

            raw_results = []

            known_result_keys = [
                "results",
                "WebPages",
                "News",
                "Sports",
                "QuestionsAndAnswers",
                "Images",
                "Videos",
                "RelatedSearches",
            ]

            for key in known_result_keys:
                if key in processed_data and isinstance(processed_data[key], list):
                    raw_results = processed_data[key]
                    result_type = key
                    break

            if not raw_results and isinstance(processed_data, dict):
                for key, value in processed_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict):
                            raw_results = value
                            result_type = key
                            break

            # Transform each result WITH DEDUPLICATION
            for raw_result in raw_results:
                count_before_dedup += 1
                
                # Extract reference_id for deduplication
                # reference_id is at top level of result (e.g., "turn1search1")
                result_data = raw_result.get("result", raw_result)
                reference_id = raw_result.get("reference_id", "") or result_data.get("reference_id", "")
                
                # Check for duplicate within this invocation
                if reference_id:
                    if reference_id in seen_ref_ids:
                        # Skip duplicate result (already extracted from direct or another batch)
                        continue
                    seen_ref_ids.add(reference_id)
                
                transformed = self._transform_single_result(raw_result)
                if transformed:
                    results.append(transformed)
                    count_after_dedup += 1

        except json.JSONDecodeError:
            pass
        except Exception:
            pass

        return results, result_type, count_before_dedup, count_after_dedup

    def _transform_results_list(
        self,
        raw_results: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Transform a list of raw results to llm_ndcg format.
        
        Args:
            raw_results: List of raw result dicts.
            
        Returns:
            Tuple of (transformed results list, result_type string).
        """
        results = []
        result_type = "results"  # Default
        
        for raw_result in raw_results:
            transformed = self._transform_single_result(raw_result)
            if transformed:
                results.append(transformed)
        
        return results, result_type

    def _transform_single_result(
        self,
        raw_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Transform a single search result to llm_ndcg format."""
        try:
            result_data = raw_result.get("result", raw_result)
            entity_type = result_data.get("type", "")

            source_json_str = result_data.get("sourceJson", "")
            source_json = {}
            if source_json_str:
                try:
                    source_json = json.loads(source_json_str) if isinstance(source_json_str, str) else source_json_str
                except json.JSONDecodeError:
                    pass

            source = result_data.get("Source", {})
            if isinstance(source, str):
                try:
                    source = json.loads(source)
                except json.JSONDecodeError:
                    source = {}

            # Extract IDs for deduplication (not using reference_id!)
            extracted_ids = {}
            artifact_id = source_json.get("ArtifactId", "")
            if artifact_id:
                extracted_ids["ArtifactId"] = artifact_id
            file_id = source.get("FileId", "") or source.get("fileid", "")
            if file_id:
                extracted_ids["FileId"] = file_id
            unique_id = source.get("UniqueId", "") or source.get("uniqueid", "")
            if unique_id:
                extracted_ids["UniqueId"] = unique_id

            url_for_id = result_data.get("url", "")
            if url_for_id:
                extracted_ids["Url"] = url_for_id

            transformed = {"Type": entity_type}
            if extracted_ids:
                transformed["ExtractedIds"] = extracted_ids

            title = result_data.get("title", "")
            subject = result_data.get("subject", "")
            if title:
                transformed["Title"] = title
            if subject:
                transformed["Subject"] = subject
                if not title:
                    transformed["Title"] = subject

            snippet = result_data.get("snippet", "")
            if snippet:
                transformed["Snippet"] = snippet
                transformed["ContentBody"] = snippet

            url = result_data.get("url", "")
            if url:
                transformed["Url"] = url

            merged_source = dict(source) if source else {}
            if source_json:
                for key, value in source_json.items():
                    if key not in merged_source:
                        merged_source[key] = value
            if merged_source:
                transformed["Source"] = merged_source

            # Additional fields (simplified)
            for field in ["fileName", "fileType", "author", "lastModifiedBy", "lastModifiedTime",
                         "from", "to", "dateTimeReceived", "start", "end", "organizerName",
                         "organizerEmail", "displayName", "pluginId"]:
                value = result_data.get(field, "")
                if value:
                    # Convert camelCase to PascalCase
                    pascal_field = field[0].upper() + field[1:]
                    transformed[pascal_field] = value

            if extracted_ids or transformed.get("Title") or transformed.get("Snippet"):
                return transformed
            return None

        except Exception:
            return None

    def _show_sample_statistics(self, records: List[Dict[str, Any]]) -> None:
        """Show statistics about the transformed records."""
        print("\n" + "=" * 60)
        print("SAMPLE STATISTICS")
        print("=" * 60)

        control_plugins: Dict[str, int] = defaultdict(int)
        treatment_plugins: Dict[str, int] = defaultdict(int)
        total_control = 0
        total_treatment = 0

        for record in records:
            for iter_key, plugins in record.get("all_search_results_control", {}).items():
                for plugin, result_lists in plugins.items():
                    for rl in result_lists:
                        count = len(rl.get("Results", []))
                        control_plugins[plugin] += count
                        total_control += count

            for iter_key, plugins in record.get("all_search_results_treatment", {}).items():
                for plugin, result_lists in plugins.items():
                    for rl in result_lists:
                        count = len(rl.get("Results", []))
                        treatment_plugins[plugin] += count
                        total_treatment += count

        print(f"\nControl Results: {total_control}")
        for plugin, count in sorted(control_plugins.items(), key=lambda x: -x[1]):
            print(f"    - {plugin}: {count}")

        print(f"\nTreatment Results: {total_treatment}")
        for plugin, count in sorted(treatment_plugins.items(), key=lambda x: -x[1]):
            print(f"    - {plugin}: {count}")


def transform(
    input_dir: str,
    output_file: str,
    max_pairs: Optional[int] = None,
    threads: int = 8,
    verbose: bool = True,
) -> int:
    """
    Transform SEVAL files to llm_ndcg format with deduplication.

    Args:
        input_dir: Directory containing SEVAL control/experiment JSON files.
        output_file: Output JSONL file path.
        max_pairs: Maximum number of pairs to process (None for all).
        threads: Number of threads for parallel processing.
        verbose: Show detailed progress.

    Returns:
        Number of successfully transformed pairs.
    """
    transformer = SEVALToLLMNDCGTransformerWithDedup(verbose=verbose)
    return transformer.transform(input_dir, output_file, max_pairs, threads)


if __name__ == "__main__":
    fire.Fire({
        "transform": transform,
    })
