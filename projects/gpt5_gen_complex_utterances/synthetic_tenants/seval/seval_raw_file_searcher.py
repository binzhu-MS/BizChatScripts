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
                print(f"Error: Search directory does not exist: {search_path}")
                return

            # Validate thread count
            if threads < 1:
                print("Error: Thread count must be at least 1")
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
                print(f"Error: Search directory does not exist: {search_path}")
                return

            # Validate thread count
            if threads < 1:
                print("Error: Thread count must be at least 1")
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
                print(f"Error: Mappings file not found: {mappings_path}")
                print(
                    "Run 'extract_query_mappings' first to generate the mappings file."
                )
                return

            # Validate thread count
            if threads < 1:
                print("Error: Thread count must be at least 1")
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


def main():
    """Main function to run the SEVAL Raw File Searcher with Fire."""
    fire.Fire(SEVALRawFileSearcher)


if __name__ == "__main__":
    main()
