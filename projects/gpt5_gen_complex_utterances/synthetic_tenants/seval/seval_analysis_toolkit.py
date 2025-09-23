#!/usr/bin/env python3
"""
SEVAL Raw File Searcher

This program searches through JSON files in the SEVAL results directory to find files
by experiment type (control/experiment/both) and partial query text matching.

Usage:
    python seval_raw_file_searcher.py search_query --exp control --query "microsoft"
    python seval_raw_file_searcher.py search_query --exp experiment --query "azure"
    python seval_raw_file_searcher.py search_query --exp both --query "teams"
    python seval_raw_file_searcher.py search_query --query "search without exp filter"

    # Generate query mappings for faster searches
    python seval_raw_file_searcher.py extract_query_mappings --threads 16

    # Fast search using pre-generated mappings
    python seval_raw_file_searcher.py search_using_mappings --query "microsoft"

    # Extract model statistics from SEVAL files
    python seval_raw_file_searcher.py extract_model_statistics --input_dir "seval_data" --output_file "model_stats.tsv"
    python seval_raw_file_searcher.py extract_model_statistics --input_dir "seval_data" --output_file "model_stats.tsv" --exp control
    python seval_raw_file_searcher.py extract_model_statistics --input_dir "seval_data" --output_file "model_stats.tsv" --exp experiment --threads 16
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import fire
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_results(results: List[Dict[str, Any]]) -> None:
    """Format and display search results."""
    if not results:
        print("No matches found.")
        return

    print(f"\nFound {len(results)} matching files:")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"{i:2d}. File: {result['filename']}")
        print(f"    Exp Name: {result['exp_name']}")
        print(f"    Query: {result['query_text']}")
        print()


class SEVALRawFileSearcher:
    """
    SEVAL Raw File Search Tool using Fire CLI framework.

    Search JSON files by experiment type and query text with streamlined interface.
    """

    def __init__(self):
        """Initialize with thread-safe result collection."""
        self.results_lock = Lock()

    def _search_in_file(
        self, json_file: Path, query_text: str
    ) -> Optional[Dict[str, Any]]:
        """Search within a single JSON file."""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Get query text from JSON of the file
            query = data.get("query", {}).get("id", "No query found")

            # If no query text specified, return None
            if not query_text:
                return None

            # Perform case-insensitive text matching
            if query_text.lower() in query.lower():
                file_info = {
                    "filename": json_file.name,
                    "filepath": str(json_file),
                    "exp_name": data.get("exp_name", "unknown"),
                    "query_text": query,
                }

                return file_info
            else:
                return None

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in file {json_file.name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error reading file {json_file.name}: {str(e)}")
            return None

    def search_query(
        self,
        query: str,
        exp: str = "both",
        threads: int = 10,
        search_dir: str = r"c:\working\BizChatScripts\projects\gpt5_gen_complex_utterances\synthetic_tenants\seval\seval_data\212953_scraping_raw_data_output",
    ):
        """
        Search query to find matching Seval raw files using multithreading.

        Args:
            query: Partial text to search for in query field (case-insensitive)
            exp: Filter by experiment type ('control', 'experiment', or 'both')
            threads: Number of threads to use for parallel processing (default: 4)
            search_dir: Directory where raw Seval files are located
        """

        try:
            search_path = Path(search_dir)
            if not search_path.exists():
                logger.error(f"Search directory does not exist: {search_path}")
                return

            # Validate thread count
            if threads < 1:
                logger.error("Thread count must be at least 1")
                return

            all_results = []

            # Handle 'both' setting by searching both control and experiment
            if exp == "both":
                exp_types = ["control", "experiment"]
            else:
                exp_types = [exp]

            for exp_type in exp_types:
                # Determine file pattern based on setting
                file_pattern = f"{exp_type.lower()}_sydney_response_*.json"

                # Search for matching files
                json_files = list(search_path.glob(file_pattern))
                logger.info(
                    f"Found {len(json_files)} {exp_type} files to process with {threads} threads"
                )

                if not json_files:
                    continue

                # Process files in parallel
                exp_results = self._process_files_parallel(json_files, query, threads)
                all_results.extend(exp_results)

            # Display results
            format_results(all_results)

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return

    def extract_query_mappings(
        self,
        threads: int = 10,
        search_dir: str = r"c:\working\BizChatScripts\projects\gpt5_gen_complex_utterances\synthetic_tenants\seval\seval_data\212953_scraping_raw_data_output",
        output_file: str = "query_file_mappings.tsv",
    ):
        """
        Extract all queries and map them to their control and experiment file pairs.
        Creates a TSV file with query, control_file, and experiment_file columns.

        Args:
            threads: Number of threads to use for parallel processing (default: 10)
            search_dir: Directory where raw Seval files are located
            output_file: Output TSV file name (default: query_file_mappings.tsv)
        """
        import csv
        from collections import defaultdict

        try:
            search_path = Path(search_dir)
            if not search_path.exists():
                logger.error(f"Search directory does not exist: {search_path}")
                return

            # Validate thread count
            if threads < 1:
                logger.error("Thread count must be at least 1")
                return

            logger.info(
                f"Starting query mapping extraction with {threads} threads from: {search_path}"
            )

            # Collect all files
            control_files = list(search_path.glob("control_sydney_response_*.json"))
            experiment_files = list(
                search_path.glob("experiment_sydney_response_*.json")
            )

            logger.info(
                f"Found {len(control_files)} control files and {len(experiment_files)} experiment files"
            )

            # Extract queries from all files
            query_to_files = defaultdict(lambda: {"control": None, "experiment": None})

            # Process control files
            control_results = self._extract_queries_from_files(
                control_files, "control", threads
            )
            for result in control_results:
                query_hash = result["query_hash"]
                query_to_files[query_hash]["control"] = result["filename"]
                query_to_files[query_hash]["query_text"] = result["query_text"]

            # Process experiment files
            experiment_results = self._extract_queries_from_files(
                experiment_files, "experiment", threads
            )
            for result in experiment_results:
                query_hash = result["query_hash"]
                query_to_files[query_hash]["experiment"] = result["filename"]
                if "query_text" not in query_to_files[query_hash]:
                    query_to_files[query_hash]["query_text"] = result["query_text"]

            # Write results to TSV
            output_path = Path(output_file)
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", newline="", encoding="utf-8") as tsvfile:
                fieldnames = [
                    "query_hash",
                    "query_text",
                    "control_file",
                    "experiment_file",
                ]
                writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")

                writer.writeheader()

                complete_pairs = 0
                control_only = 0
                experiment_only = 0

                for query_hash, files in query_to_files.items():
                    writer.writerow(
                        {
                            "query_hash": query_hash,
                            "query_text": files.get(
                                "query_text", "No query text found"
                            ),
                            "control_file": files["control"] or "",
                            "experiment_file": files["experiment"] or "",
                        }
                    )

                    # Count pair statistics
                    if files["control"] and files["experiment"]:
                        complete_pairs += 1
                    elif files["control"] and not files["experiment"]:
                        control_only += 1
                    elif files["experiment"] and not files["control"]:
                        experiment_only += 1

            # Display statistics
            total_queries = len(query_to_files)
            print(f"\n📊 Query Mapping Statistics (processed with {threads} threads):")
            print(f"{'='*50}")
            print(f"📂 Input directory: {search_path}")
            print(f"Total unique queries: {total_queries}")
            print(f"Complete pairs (control + experiment): {complete_pairs}")
            print(f"Control only: {control_only}")
            print(f"Experiment only: {experiment_only}")

            logger.info(
                f"Query mapping extraction completed. Results saved to: {output_path}"
            )

        except Exception as e:
            logger.error(f"Query mapping extraction failed: {str(e)}")
            return

    def _extract_queries_from_files(
        self, json_files: List[Path], exp_type: str, threads: int
    ) -> List[Dict[str, Any]]:
        """Extract query information from files in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    self._extract_query_from_file, json_file, exp_type
                ): json_file
                for json_file in json_files
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                json_file = future_to_file[future]
                try:
                    query_info = future.result()
                    if query_info:
                        results.append(query_info)
                except Exception as e:
                    logger.warning(
                        f"Error extracting query from file {json_file.name}: {str(e)}"
                    )
                    continue

        logger.info(
            f"Extracted queries from {len(results)} {exp_type} files using {threads} threads"
        )
        return results

    def _extract_query_from_file(
        self, json_file: Path, exp_type: str
    ) -> Optional[Dict[str, Any]]:
        """Extract query information from a single JSON file."""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            query_obj = data.get("query", {})
            query_hash = query_obj.get("query_hash", "")
            query_text = query_obj.get("id", "No query found")

            if not query_hash:
                logger.warning(f"No query_hash found in file {json_file.name}")
                return None

            return {
                "filename": json_file.name,
                "filepath": str(json_file),
                "exp_type": exp_type,
                "query_hash": query_hash,
                "query_text": query_text,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in file {json_file.name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error reading file {json_file.name}: {str(e)}")
            return None

    def search_using_mappings(
        self,
        query: str,
        mappings_file: str,
        threads: int = 8,
        search_dir: str = r"c:\working\BizChatScripts\projects\gpt5_gen_complex_utterances\synthetic_tenants\seval\seval_data\212953_scraping_raw_data_output",
    ):
        """
        Fast search using pre-generated query mappings TSV file with multithreading support.
        Much faster than searching through all JSON files individually.

        Args:
            query: Partial text to search for in query field (case-insensitive)
            mappings_file: TSV file with query mappings (default: query_file_mappings.tsv)
            threads: Number of threads to use for parallel processing (default: 8)
            search_dir: Directory where the JSON files are located (for constructing full file paths)
        """
        import csv

        try:
            # Handle mappings_file as an independent path
            mappings_path = Path(mappings_file)

            if not mappings_path.exists():
                logger.error(f"Mappings file not found: {mappings_path}")
                logger.info(
                    "Run 'extract_query_mappings' first to generate the mappings file."
                )
                return

            # Validate thread count
            if threads < 1:
                logger.error("Thread count must be at least 1")
                return

            logger.info(
                f"Searching using mappings file: {mappings_path} with {threads} threads"
            )

            # Load all rows from TSV file
            all_rows = []
            with open(mappings_path, "r", newline="", encoding="utf-8") as tsvfile:
                reader = csv.DictReader(tsvfile, delimiter="\t")
                all_rows = list(reader)

            logger.info(f"Loaded {len(all_rows)} rows from mappings file")

            # Process rows in parallel using multithreading
            results = self._process_mappings_parallel(
                all_rows, query, search_dir, threads
            )

            # Display results
            print(
                f"\n🔍 Search Results (searched {len(all_rows)} queries with {threads} threads):"
            )
            format_results(results)

        except Exception as e:
            logger.error(f"Mappings-based search failed: {str(e)}")
            return

    def _process_files_parallel(
        self, json_files: List[Path], query: str, threads: int
    ) -> List[Dict[str, Any]]:
        """Process files in parallel using ThreadPoolExecutor."""
        results = []

        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._search_in_file, json_file, query): json_file
                for json_file in json_files
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                json_file = future_to_file[future]
                try:
                    match = future.result()
                    if match:
                        results.append(match)
                except Exception as e:
                    logger.warning(f"Error processing file {json_file.name}: {str(e)}")
                    continue

        return results

    def _process_mappings_parallel(
        self, all_rows: List[Dict[str, str]], query: str, search_dir: str, threads: int
    ) -> List[Dict[str, Any]]:
        """Process mapping rows in parallel using ThreadPoolExecutor."""
        results = []
        results_lock = Lock()

        def process_row_batch(rows_batch):
            """Process a batch of rows in a single thread."""
            batch_results = []
            for row in rows_batch:
                result = self._process_single_mapping_row(row, query, search_dir)
                if result:
                    batch_results.extend(result)
            return batch_results

        # Split rows into batches for better thread utilization
        batch_size = max(1, len(all_rows) // (threads * 2))  # 2 batches per thread
        row_batches = [
            all_rows[i : i + batch_size] for i in range(0, len(all_rows), batch_size)
        ]

        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit all batch tasks
            future_to_batch = {
                executor.submit(process_row_batch, batch): batch
                for batch in row_batches
            }

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    if batch_results:
                        with results_lock:
                            results.extend(batch_results)
                except Exception as e:
                    logger.warning(f"Error processing batch: {str(e)}")
                    continue

        return results

    def _process_single_mapping_row(
        self, row: Dict[str, str], query: str, search_dir: str
    ) -> List[Dict[str, Any]]:
        """Process a single mapping row to check for query matches."""
        results = []

        try:
            query_text = row.get("query_text", "")

            # Perform case-insensitive text matching
            if query.lower() in query_text.lower():
                # Add control file if exists
                if row.get("control_file"):
                    results.append(
                        {
                            "filename": row["control_file"],
                            "filepath": str(Path(search_dir) / row["control_file"]),
                            "exp_name": "control",
                            "query_text": query_text,
                            "query_hash": row.get("query_hash", ""),
                        }
                    )

                # Add experiment file if exists
                if row.get("experiment_file"):
                    results.append(
                        {
                            "filename": row["experiment_file"],
                            "filepath": str(Path(search_dir) / row["experiment_file"]),
                            "exp_name": "experiment",
                            "query_text": query_text,
                            "query_hash": row.get("query_hash", ""),
                        }
                    )

        except Exception as e:
            logger.warning(f"Error processing row: {str(e)}")

        return results

    def extract_model_statistics(
        self,
        input_dir: str,
        output_file: str,
        exp: str = "both",
        threads: int = 8,
    ):
        """
        Extract reasoning model statistics from SEVAL JSON files.

        Focuses specifically on models used for the reasoning/planning phase (fluxv3:invokingfunction)
        where the model determines which tools to invoke. This is the core reasoning capability.

        Since all fluxv3:invokingfunction instances within a single query use the same model,
        we optimize by finding just the first instance per file.

        Args:
            input_dir: Directory containing SEVAL JSON files to analyze
            output_file: Path where TSV results file will be saved
            exp: Filter by experiment type ('control', 'experiment', or 'both') (default: 'both')
            threads: Number of threads for parallel processing (default: 8)
        """
        import json
        import re
        import os
        import concurrent.futures
        from datetime import datetime
        from pathlib import Path

        logger.info(f"Extracting model statistics from: {input_dir}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Using {threads} threads")
        logger.info(f"Experiment filter: {exp}")

        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return

        # Validate exp parameter
        if exp not in ["control", "experiment", "both"]:
            logger.error(
                f"Invalid exp parameter '{exp}'. Must be 'control', 'experiment', or 'both'"
            )
            return

        logger.info(f"Starting model statistics extraction at: {datetime.now()}")

        # Find JSON files based on experiment type filter
        json_files = []
        search_path = Path(input_dir)

        # Determine file patterns based on exp filter
        if exp == "both":
            file_patterns = [
                "control_sydney_response_*.json",
                "experiment_sydney_response_*.json",
            ]
        elif exp == "control":
            file_patterns = ["control_sydney_response_*.json"]
        elif exp == "experiment":
            file_patterns = ["experiment_sydney_response_*.json"]

        # Collect files matching the patterns
        for pattern in file_patterns:
            matching_files = list(search_path.glob(pattern))
            json_files.extend([str(f) for f in matching_files])
            logger.info(
                f"Found {len(matching_files)} files matching pattern: {pattern}"
            )

        # If no pattern matches found, fall back to finding all JSON files
        if not json_files:
            logger.info("No SEVAL pattern files found, searching for all JSON files...")
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith(".json"):
                        # Apply exp filter to generic JSON files if possible
                        if exp == "both" or exp in file.lower():
                            json_files.append(os.path.join(root, file))

        logger.info(f"Found {len(json_files)} JSON files")

        if not json_files:
            logger.warning("No JSON files found in the directory.")
            return

        # Statistics collection
        model_stats = {}
        file_stats = []

        def extract_reasoning_models_and_categorize(file_path):
            """Extract reasoning model information and categorize file based on success/failure status."""
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                file_models = set()
                reasoning_models_found = (
                    []
                )  # Track all reasoning model instances for consistency check

                # Categorize file based on conversation success and error conditions
                file_category = "other_errors"  # default

                # Check for conversation success indicators
                conversation_success = data.get("conversation_success", True)
                is_success = data.get("is_success", True)
                error_reason = data.get("error", "")

                # Check for specific error patterns
                error_message = str(error_reason).lower()

                if not conversation_success or not is_success:
                    if "504" in error_message or "timeout" in error_message:
                        file_category = "timeout_errors"
                    elif "scraperservice" in error_message:
                        file_category = "scraper_errors"
                    elif "request" in error_message or "http" in error_message:
                        file_category = "request_failed"
                    else:
                        file_category = "conversation_failed"
                elif conversation_success and is_success:
                    file_category = (
                        "successful_without_reasoning"  # Will update if reasoning found
                    )

                # Extract models from various locations in the JSON structure
                def extract_models_recursive(obj, path=""):
                    nonlocal file_models, reasoning_models_found

                    # Priority-based model detection - stop at first successful match
                    primary_model = None
                    detection_method = None

                    def search_base_model_name(obj, current_path=""):
                        """Priority 1: Search for base model name in direct model fields"""
                        if isinstance(obj, dict):
                            if "model" in obj and isinstance(obj["model"], str):
                                model = obj["model"].strip()
                                tag = obj.get("tag", "")
                                if (
                                    model
                                    and (
                                        "reasoning" in model.lower()
                                        or "gpt" in model.lower()
                                    )
                                    and "invokingfunction" in tag
                                ):
                                    return (
                                        model,
                                        "direct_model_field",
                                        tag,
                                        current_path,
                                    )

                            for key, value in obj.items():
                                result = search_base_model_name(
                                    value,
                                    f"{current_path}.{key}" if current_path else key,
                                )
                                if result[0]:  # If model found
                                    return result
                        elif isinstance(obj, list):
                            for i, item in enumerate(obj):
                                result = search_base_model_name(
                                    item,
                                    (
                                        f"{current_path}[{i}]"
                                        if current_path
                                        else f"[{i}]"
                                    ),
                                )
                                if result[0]:  # If model found
                                    return result
                        return None, None, None, None

                    def search_extended_data_loop1(obj, current_path=""):
                        """Priority 2: Extended Data with LoopCount=1"""
                        if isinstance(obj, dict):
                            if "extendedData" in obj:
                                extended_data_raw = obj["extendedData"]
                                extended_data = None

                                # Handle different extendedData formats
                                if isinstance(extended_data_raw, str):
                                    try:
                                        extended_data = json.loads(extended_data_raw)
                                    except json.JSONDecodeError:
                                        pass
                                elif isinstance(extended_data_raw, dict):
                                    extended_data = extended_data_raw

                                # Parse the extended data if we have it
                                if extended_data and isinstance(extended_data, dict):
                                    tag = extended_data.get("Tag", "")
                                    model_name = extended_data.get(
                                        "ModelName", ""
                                    ).strip()
                                    loop_count = extended_data.get("LoopCount", "")

                                    if (
                                        model_name
                                        and "fluxv3:invokingfunction" in tag
                                        and loop_count == "1"
                                    ):
                                        return (
                                            model_name,
                                            "extended_data_loop1",
                                            tag,
                                            current_path,
                                        )

                            for key, value in obj.items():
                                result = search_extended_data_loop1(
                                    value,
                                    f"{current_path}.{key}" if current_path else key,
                                )
                                if result[0]:
                                    return result
                        elif isinstance(obj, list):
                            for i, item in enumerate(obj):
                                result = search_extended_data_loop1(
                                    item,
                                    (
                                        f"{current_path}[{i}]"
                                        if current_path
                                        else f"[{i}]"
                                    ),
                                )
                                if result[0]:
                                    return result
                        return None, None, None, None

                    def search_orchestration_iter1(obj, current_path=""):
                        """Priority 3: orchestrationIterations with iteration=1"""
                        if isinstance(obj, dict):
                            if "orchestrationIterations" in obj and isinstance(
                                obj["orchestrationIterations"], list
                            ):
                                for iteration in obj["orchestrationIterations"]:
                                    if (
                                        isinstance(iteration, dict)
                                        and iteration.get("iteration") == 1
                                        and "modelActions" in iteration
                                    ):

                                        model_actions = iteration["modelActions"]
                                        if isinstance(model_actions, list):
                                            for model_action in model_actions:
                                                if isinstance(model_action, dict):
                                                    model = model_action.get(
                                                        "model", ""
                                                    ).strip()
                                                    tag = model_action.get("tag", "")

                                                    if (
                                                        model
                                                        and "reasoning" in model.lower()
                                                        and "invokingfunction" in tag
                                                    ):
                                                        return (
                                                            model,
                                                            "orchestration_iter1",
                                                            tag,
                                                            f"{current_path}.orchestrationIterations[iteration=1]",
                                                        )

                            for key, value in obj.items():
                                result = search_orchestration_iter1(
                                    value,
                                    f"{current_path}.{key}" if current_path else key,
                                )
                                if result[0]:
                                    return result
                        elif isinstance(obj, list):
                            for i, item in enumerate(obj):
                                result = search_orchestration_iter1(
                                    item,
                                    (
                                        f"{current_path}[{i}]"
                                        if current_path
                                        else f"[{i}]"
                                    ),
                                )
                                if result[0]:
                                    return result
                        return None, None, None, None

                    def search_fallback_invokingfunction(obj, current_path=""):
                        """Priority 4: Fallback - any invokingfunction tag"""
                        if isinstance(obj, dict):
                            if "tag" in obj and "model" in obj:
                                tag = obj.get("tag", "")
                                model = obj.get("model", "").strip()

                                if model and "invokingfunction" in tag:
                                    return (
                                        model,
                                        "fallback_invokingfunction",
                                        tag,
                                        current_path,
                                    )

                            for key, value in obj.items():
                                result = search_fallback_invokingfunction(
                                    value,
                                    f"{current_path}.{key}" if current_path else key,
                                )
                                if result[0]:
                                    return result
                        elif isinstance(obj, list):
                            for i, item in enumerate(obj):
                                result = search_fallback_invokingfunction(
                                    item,
                                    (
                                        f"{current_path}[{i}]"
                                        if current_path
                                        else f"[{i}]"
                                    ),
                                )
                                if result[0]:
                                    return result
                        return None, None, None, None

                    # Execute priority-based search
                    # Priority 1: Direct model field with reasoning
                    primary_model, detection_method, tag, model_path = (
                        search_base_model_name(obj, path)
                    )

                    if not primary_model:
                        # Priority 2: Extended Data LoopCount=1
                        primary_model, detection_method, tag, model_path = (
                            search_extended_data_loop1(obj, path)
                        )

                    if not primary_model:
                        # Priority 3: orchestrationIterations iteration=1
                        primary_model, detection_method, tag, model_path = (
                            search_orchestration_iter1(obj, path)
                        )

                    if not primary_model:
                        # Priority 4: Fallback invokingfunction
                        primary_model, detection_method, tag, model_path = (
                            search_fallback_invokingfunction(obj, path)
                        )

                    # If we found a model, record it (only once per file)
                    if primary_model:
                        reasoning_models_found.append(
                            {
                                "model": primary_model,
                                "tag": tag,
                                "path": model_path,
                                "detection_method": detection_method,
                            }
                        )

                        # Add only the primary model (no context variations)
                        file_models.add(primary_model)

                # Execute the recursive search only once per file
                extract_models_recursive(data)

                # Update file category if reasoning models were found
                if (
                    reasoning_models_found
                    and file_category == "successful_without_reasoning"
                ):
                    file_category = "successful_with_reasoning"

                # Sanity check: Verify model consistency across reasoning instances
                consistency_info = {
                    "consistent": True,
                    "reasoning_model": None,
                    "instances_count": 0,
                    "inconsistencies": [],
                }

                if reasoning_models_found:
                    consistency_info["instances_count"] = len(reasoning_models_found)
                    first_model = reasoning_models_found[0]["model"]
                    consistency_info["reasoning_model"] = first_model

                    # Check if all instances use the same model
                    for instance in reasoning_models_found:
                        if instance["model"] != first_model:
                            consistency_info["consistent"] = False
                            consistency_info["inconsistencies"].append(
                                {
                                    "expected": first_model,
                                    "found": instance["model"],
                                    "path": instance["path"],
                                }
                            )

                return {
                    "file": file_path,
                    "models": list(file_models),
                    "category": file_category,
                    "consistency": consistency_info,
                }

            except Exception as e:
                return {
                    "file": file_path,
                    "models": [],
                    "category": "processing_errors",
                    "error": str(e),
                    "consistency": {
                        "consistent": True,
                        "reasoning_model": None,
                        "instances_count": 0,
                        "inconsistencies": [],
                    },
                }

        # Process files with threading
        consistency_stats = {
            "total_files": 0,
            "files_with_reasoning": 0,
            "consistent_files": 0,
            "inconsistent_files": 0,
            "total_reasoning_instances": 0,
            "inconsistencies": [],
        }

        # Enhanced file categorization
        file_categories = {
            "successful_with_reasoning": 0,
            "successful_without_reasoning": 0,
            "conversation_failed": 0,
            "request_failed": 0,
            "processing_errors": 0,
            "timeout_errors": 0,
            "scraper_errors": 0,
            "other_errors": 0,
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_file = {
                executor.submit(
                    extract_reasoning_models_and_categorize, file_path
                ): file_path
                for file_path in json_files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    file_stats.append(result)

                    # Update consistency statistics
                    consistency_stats["total_files"] += 1
                    consistency_info = result.get("consistency", {})

                    # Update file categorization statistics
                    file_category = result.get("category", "other_errors")
                    if file_category in file_categories:
                        file_categories[file_category] += 1
                    else:
                        file_categories["other_errors"] += 1

                    if consistency_info.get("instances_count", 0) > 0:
                        consistency_stats["files_with_reasoning"] += 1
                        consistency_stats[
                            "total_reasoning_instances"
                        ] += consistency_info["instances_count"]

                        if consistency_info.get("consistent", True):
                            consistency_stats["consistent_files"] += 1
                        else:
                            consistency_stats["inconsistent_files"] += 1
                            # Store details of inconsistencies for reporting
                            consistency_stats["inconsistencies"].append(
                                {
                                    "file": os.path.basename(file_path),
                                    "expected_model": consistency_info.get(
                                        "reasoning_model"
                                    ),
                                    "inconsistency_details": consistency_info.get(
                                        "inconsistencies", []
                                    ),
                                }
                            )

                    # Update global model statistics
                    for model in result["models"]:
                        if model not in model_stats:
                            model_stats[model] = 0
                        model_stats[model] += 1

                except Exception as e:
                    consistency_stats["total_files"] += 1
                    file_categories["processing_errors"] += 1
                    file_stats.append(
                        {
                            "file": file_path,
                            "models": [],
                            "category": "processing_errors",
                            "error": str(e),
                            "consistency": {
                                "consistent": True,
                                "reasoning_model": None,
                                "instances_count": 0,
                                "inconsistencies": [],
                            },
                        }
                    )
                    logger.error(f"Error processing {file_path}: {e}")

        # Write results to TSV file
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            # Write summary statistics
            f.write("# SEVAL Model Statistics Summary\n")
            f.write(f"# Generated at: {datetime.now()}\n")
            f.write(f"# Input directory: {input_dir}\n")
            f.write(f"# Experiment filter: {exp}\n")
            f.write(f"# Total files found: {len(json_files)}\n")
            f.write(f"# Total files processed: {len(json_files)}\n")

            # Write reasoning model analysis statistics
            files_without_reasoning = (
                len(json_files) - consistency_stats["files_with_reasoning"]
            )
            f.write("\n# Reasoning Model Analysis:\n")
            reasoning_percentage = (
                consistency_stats["files_with_reasoning"] / len(json_files)
            ) * 100
            no_reasoning_percentage = (files_without_reasoning / len(json_files)) * 100
            f.write(
                f"# Files with reasoning models: {consistency_stats['files_with_reasoning']} ({reasoning_percentage:.1f}%)\n"
            )

            # Add model breakdown directly under the files with reasoning models line
            if model_stats:
                for model, count in sorted(
                    model_stats.items(), key=lambda x: x[1], reverse=True
                ):
                    percentage = (
                        count / consistency_stats["files_with_reasoning"]
                    ) * 100
                    total_percentage = (count / len(json_files)) * 100
                    f.write(
                        f"#   {model}: {count} files ({percentage:.1f}% of reasoning files, {total_percentage:.1f}% of all files)\n"
                    )

            f.write(
                f"# Files without reasoning models: {files_without_reasoning} ({no_reasoning_percentage:.1f}%)\n"
            )

            # Add file categorization statistics
            f.write("\n# File Categorization Statistics:\n")
            total_categorized = sum(file_categories.values())
            for category, count in file_categories.items():
                if count > 0:
                    percentage = (count / total_categorized) * 100
                    f.write(
                        f"# {category.replace('_', ' ').title()}: {count} files ({percentage:.1f}%)\n"
                    )

            # Only include inconsistency details if they exist
            if consistency_stats["inconsistencies"]:
                f.write("\n# INCONSISTENCY DETAILS:\n")
                for inconsistency in consistency_stats["inconsistencies"]:
                    f.write(
                        f"# - {inconsistency['file']}: expected '{inconsistency['expected_model']}'\n"
                    )
                    for detail in inconsistency["inconsistency_details"]:
                        f.write(f"#   Found: '{detail['found']}' at {detail['path']}\n")

            f.write("\n# Detailed File-by-File Results:\n")

            # Write TSV header
            f.write(
                "file\tmodels_found\tmodel_list\treasoning_instances\tconsistent\treasoning_model\tcategory\terror\n"
            )

            # Write detailed results
            for stat in file_stats:
                filename = os.path.basename(stat["file"])
                models_count = len(stat["models"])
                models_list = "; ".join(stat["models"]) if stat["models"] else ""
                error = stat.get("error", "")

                # Add consistency information
                consistency_info = stat.get("consistency", {})
                reasoning_instances = consistency_info.get("instances_count", 0)
                is_consistent = consistency_info.get("consistent", True)
                reasoning_model = consistency_info.get("reasoning_model", "")
                category = stat.get("category", "unknown")

                f.write(
                    f"{filename}\t{models_count}\t{models_list}\t{reasoning_instances}\t{is_consistent}\t{reasoning_model}\t{category}\t{error}\n"
                )

        print(f"\n📊 Model Statistics Summary:")
        print(f"{'='*50}")
        print(f"Input directory: {input_dir}")
        print(f"Experiment filter: {exp}")
        print(f"Total files found: {len(json_files)}")
        print(f"Total files processed: {len(json_files)}")

        # Print reasoning model analysis results
        files_without_reasoning = (
            len(json_files) - consistency_stats["files_with_reasoning"]
        )
        print(f"\n🔍 Reasoning Model Analysis:")
        reasoning_percentage = (
            consistency_stats["files_with_reasoning"] / len(json_files)
        ) * 100
        no_reasoning_percentage = (files_without_reasoning / len(json_files)) * 100
        print(
            f"Files with reasoning models: {consistency_stats['files_with_reasoning']} ({reasoning_percentage:.1f}%)"
        )

        # Show model breakdown directly under the files with reasoning models line
        if model_stats:
            for model, count in sorted(
                model_stats.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                percentage = (count / consistency_stats["files_with_reasoning"]) * 100
                total_percentage = (count / len(json_files)) * 100
                print(
                    f"  {model}: {count} files ({percentage:.1f}% of reasoning files, {total_percentage:.1f}% of all files)"
                )

        print(
            f"Files without reasoning models: {files_without_reasoning} ({no_reasoning_percentage:.1f}%)"
        )

        # Print file categorization statistics
        print(f"\n📂 File Categorization Breakdown:")
        total_categorized = sum(file_categories.values())
        for category, count in file_categories.items():
            if count > 0:
                percentage = (count / total_categorized) * 100
                category_name = category.replace("_", " ").title()
                print(f"  {category_name}: {count} files ({percentage:.1f}%)")

        # Only show inconsistencies if they exist (should be rare with priority-based detection)
        if consistency_stats["inconsistent_files"] > 0:
            print(f"\n⚠️  INCONSISTENCIES FOUND:")
            for inconsistency in consistency_stats["inconsistencies"]:
                print(
                    f"  {inconsistency['file']}: expected '{inconsistency['expected_model']}'"
                )
                for detail in inconsistency["inconsistency_details"]:
                    print(f"    Found: '{detail['found']}' at {detail['path']}")
        else:
            print("✅ All files use consistent reasoning models!")

        print(f"Unique reasoning models discovered: {len(model_stats)}")

        print(f"\nResults written to: {output_file}")
        print(f"Model statistics extraction completed at: {datetime.now()}")
        logger.info(
            f"Model statistics extraction completed successfully. Results saved to: {output_file}"
        )


def main():
    """Main function to run the SEVAL Raw File Searcher with Fire."""
    fire.Fire(SEVALRawFileSearcher)


if __name__ == "__main__":
    main()
