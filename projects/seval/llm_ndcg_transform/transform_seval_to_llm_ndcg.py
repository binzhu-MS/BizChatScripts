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

        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_pair = {
                executor.submit(
                    self._transform_pair,
                    query_hash,
                    info["control"],
                    info["experiment"],
                    info.get("query_text", "")
                ): query_hash
                for query_hash, info in complete_pairs
            }

            for i, future in enumerate(as_completed(future_to_pair)):
                query_hash = future_to_pair[future]
                try:
                    result = future.result()
                    if result:
                        transformed_records.append(result)
                    else:
                        errors.append(query_hash)
                except Exception as e:
                    logger.warning(f"Error transforming {query_hash}: {e}")
                    errors.append(query_hash)

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

        # Show sample statistics
        if transformed_records:
            self._show_sample_statistics(transformed_records)

        return len(transformed_records)

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

    def _transform_pair(
        self,
        query_hash: str,
        control_file: str,
        experiment_file: str,
        query_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Transform a single control/experiment pair to llm_ndcg format.

        Args:
            query_hash: The query hash for this pair.
            control_file: Path to control JSON file.
            experiment_file: Path to experiment JSON file.
            query_text: The query text.

        Returns:
            Transformed record in llm_ndcg format, or None if transformation fails.
        """
        try:
            control_data = self._read_json_file_safely(control_file)
            experiment_data = self._read_json_file_safely(experiment_file)

            # Extract search results from last turn
            control_results = self._extract_search_results_last_turn(control_data)
            experiment_results = self._extract_search_results_last_turn(experiment_data)

            # Skip if both are empty
            if not control_results and not experiment_results:
                return None

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

            return record

        except Exception as e:
            logger.debug(f"Error transforming pair {query_hash}: {e}")
            return None

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

    def _extract_search_results_last_turn(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Extract search results from the last turn's orchestration iterations.

        This uses the last turn because in multi-turn conversations,
        only the last turn typically has successful search results.

        Args:
            data: Parsed SEVAL JSON data.

        Returns:
            Dict in llm_ndcg format:
            {
                "1": {"plugin_name": [{"Results": [...]}]},
                "2": {"plugin_name": [{"Results": [...]}]}
            }
        """
        all_search_results: Dict[str, Dict[str, List[Dict]]] = {}

        try:
            messages = data.get("requests", [{}])[0].get("response_body", {}).get("messages", [])

            # Find EvaluationData message
            eval_data = None
            for msg in messages:
                if msg.get("messageType") == "EvaluationData":
                    eval_data = msg.get("evaluationData", {})
                    break

            if not eval_data:
                return all_search_results

            turn_data = eval_data.get("turnData", [])
            if not turn_data:
                return all_search_results

            # Use the LAST turn (most likely to have successful search results)
            last_turn = turn_data[-1]
            orchestration_iterations = last_turn.get("orchestrationIterations", [])

            # Track iteration index (1-based for llm_ndcg format)
            for iter_idx, iteration in enumerate(orchestration_iterations, start=1):
                iter_key = str(iter_idx)
                iter_results: Dict[str, List[Dict]] = {}

                model_actions = iteration.get("modelActions", [])
                for action in model_actions:
                    tool_invocations = action.get("toolInvocations", [])

                    for invocation in tool_invocations:
                        # Get the function/plugin name
                        function_name = invocation.get("function", "unknown_plugin")

                        # Build PluginInvocation string from function name and arguments
                        plugin_invocation = self._build_plugin_invocation(invocation)

                        # Process batchedQueries
                        batched_queries = invocation.get("batchedQueries", [])
                        for batch_query in batched_queries:
                            results = self._parse_processed_result(batch_query)
                            if results:
                                if function_name not in iter_results:
                                    iter_results[function_name] = []
                                # Include PluginInvocation in the result dict
                                iter_results[function_name].append({
                                    "Results": results,
                                    "PluginInvocation": plugin_invocation,
                                })

                if iter_results:
                    all_search_results[iter_key] = iter_results

        except Exception as e:
            logger.debug(f"Error extracting search results: {e}")

        return all_search_results

    def _build_plugin_invocation(
        self,
        invocation: Dict[str, Any]
    ) -> str:
        """
        Build a PluginInvocation string from the tool invocation data.

        The format should match what llm_ndcg expects:
            'function_name(query="search query text")'

        Args:
            invocation: The toolInvocation dict from SEVAL data.

        Returns:
            PluginInvocation string.
        """
        function_name = invocation.get("function", "unknown_plugin")
        arguments = invocation.get("arguments", {})

        # arguments can be a string (JSON) or dict
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        # Extract query from arguments
        query = ""
        if "queries" in arguments and arguments["queries"]:
            first_query = arguments["queries"][0]
            if isinstance(first_query, dict):
                query = first_query.get("query", "")
            elif isinstance(first_query, str):
                query = first_query
        elif "query" in arguments:
            query = arguments.get("query", "")

        # Build the invocation string
        if query:
            return f'{function_name}(query="{query}")'
        else:
            return f"{function_name}()"

    def _parse_processed_result(
        self,
        batch_query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse processedResult from a batchedQuery into llm_ndcg result format.

        Args:
            batch_query: A batchedQuery dict containing processedResult.

        Returns:
            List of results in llm_ndcg format.
        """
        results = []

        processed_result = batch_query.get("processedResult", "")
        if not processed_result:
            return results

        try:
            # processedResult is often a JSON string
            if isinstance(processed_result, str):
                processed_data = json.loads(processed_result)
            else:
                processed_data = processed_result

            # Extract results from various possible locations
            raw_results = []

            # Check direct "results" key
            if "results" in processed_data:
                raw_results = processed_data["results"]
            # Check WebPages (Bing search format)
            elif "WebPages" in processed_data:
                raw_results = processed_data["WebPages"]
            # Check News
            elif "News" in processed_data:
                raw_results = processed_data["News"]

            # Transform each result to llm_ndcg format
            for raw_result in raw_results:
                transformed = self._transform_single_result(raw_result)
                if transformed:
                    results.append(transformed)

        except json.JSONDecodeError:
            logger.debug("Failed to parse processedResult as JSON")
        except Exception as e:
            logger.debug(f"Error parsing processedResult: {e}")

        return results

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
            - result_id: Same as Id
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
                "result_id": ref_id,
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
