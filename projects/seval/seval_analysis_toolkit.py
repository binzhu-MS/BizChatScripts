#!/usr/bin/env python3
"""
SEVAL Analysis Toolkit

This program provides comprehensive analysis tools for SEVAL results including
search functionality, model statistics extraction, and search results analysis.

Usage:
    python seval_analysis_toolkit.py search_query --exp control --query "microsoft"
    python seval_analysis_toolkit.py search_query --exp experiment --query "azure"
    python seval_analysis_toolkit.py search_query --exp both --query "teams"
    python seval_analysis_toolkit.py search_query --query "search without exp filter"

    # Generate query mappings for faster searches
    python seval_analysis_toolkit.py extract_query_mappings --threads 16

    # Fast search using pre-generated mappings
    python seval_analysis_toolkit.py search_using_mappings --query "microsoft"

    # Extract model statistics from SEVAL files
    python seval_analysis_toolkit.py extract_model_statistics --input_dir "seval_data" --output_file "model_stats.tsv"
    python seval_analysis_toolkit.py extract_model_statistics --input_dir "seval_data" --output_file "model_stats.tsv" --exp control
    python seval_analysis_toolkit.py extract_model_statistics --input_dir "seval_data" --output_file "model_stats.tsv" --exp experiment --threads 16

    # Analyze search results patterns (requires query mappings)
    python seval_analysis_toolkit.py analyze_search_results --mappings_file "results/query_file_mappings.tsv" --output_file "results/search_analysis.tsv"
    python seval_analysis_toolkit.py analyze_search_results --threads 16 --max_queries 100
"""

import json
import re
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
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


class SEVALAnalysisToolkit:
    """
    Comprehensive SEVAL analysis toolkit providing search, model statistics, and search results analysis.

    This class offers multiple analysis methods for SEVAL (Search Evaluation) JSON files:

    1. search_query: Find files by experiment type and partial query matching
    2. extract_query_mappings: Generate query-to-file mappings for faster searches
    3. search_using_mappings: Fast search using pre-generated mappings
    4. extract_model_statistics: Extract reasoning model usage statistics
    5. analyze_search_results: Analyze search operation patterns and success/failure rates

    The toolkit handles both direct message formats and SEVAL request/response structures,
    providing detailed analysis of search operations, content access patterns, and
    comparative statistics between control and experiment conditions.
    """

    def __init__(self):
        """Initialize with thread-safe result collection."""
        self.results_lock = Lock()

    def _extract_search_information(self, seval_data: Dict) -> Dict:
        """
        Extract comprehensive search information from SEVAL JSON structure.

        This method parses the message flow to extract:
        - User query (Message 0)
        - Search query executed (Message 3)
        - Search results (Message 9)
        - Final response (Message 10)
        - File access info (Message 11)
        - Error messages (Messages 4-6)

        Args:
            seval_data: Parsed SEVAL JSON data

        Returns:
            Dict containing structured search information
        """
        try:
            messages = seval_data["requests"][0]["response_body"]["messages"]
        except (KeyError, IndexError):
            logger.warning("Could not extract messages from SEVAL data structure")
            return self._get_empty_search_info()

        search_info = {
            "user_query": "",
            "search_query": "",
            "results_found": [],
            "files_accessed": [],
            "final_response": "",
            "search_success": True,
            "error_messages": [],
            "response_length": 0,
        }

        for i, message in enumerate(messages):
            author = message.get("author", "unknown")
            text = message.get("text", "")

            try:
                # Extract user query (Message 0)
                if i == 0 and author == "user":
                    search_info["user_query"] = text

                # Extract search query (Message 3)
                elif i == 3 and author == "bot" and "search for" in text.lower():
                    search_info["search_query"] = text

                # Check for "No content returned" errors (Messages 4-6)
                elif "no content returned" in text.lower():
                    search_info["error_messages"].append(f"Message {i}: {text}")
                    search_info["search_success"] = False

                # Extract search results (Message 9)
                elif text.startswith('{"results":'):
                    try:
                        results_json = json.loads(text)
                        for item in results_json.get("results", []):
                            result_info = item.get("result", {})
                            search_info["results_found"].append(
                                {
                                    "type": result_info.get("type", "unknown"),
                                    "title": result_info.get("title", ""),
                                    "reference_id": result_info.get("reference_id", ""),
                                    "snippet": result_info.get("snippet", ""),
                                }
                            )
                    except json.JSONDecodeError:
                        search_info["error_messages"].append(
                            f"Message {i}: JSON parse error in results"
                        )

                # Extract final response (Message 10)
                elif i == 10 and author == "bot" and len(text) > 100:
                    search_info["final_response"] = text
                    search_info["response_length"] = len(text)

                # Extract file access URLs (Message 11)
                elif text.startswith('{"storageResults":'):
                    try:
                        storage_json = json.loads(text)
                        for item in storage_json.get("storageResults", []):
                            url = item.get("url", "")
                            filename = self._extract_filename_from_url(url)
                            search_info["files_accessed"].append(
                                {
                                    "filename": filename,
                                    "url": url,
                                    "id": item.get("id", ""),
                                    "type": item.get("type", "unknown"),
                                }
                            )
                    except json.JSONDecodeError:
                        search_info["error_messages"].append(
                            f"Message {i}: JSON parse error in storage"
                        )

            except Exception as e:
                logger.warning(f"Error processing message {i}: {e}")
                continue

        return search_info

    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from SharePoint or file URL."""
        if not url:
            return "unknown_file"

        if "file=" in url:
            try:
                return url.split("file=")[1].split("&")[0]
            except IndexError:
                pass

        if "/" in url:
            try:
                return url.split("/")[-1].split("?")[0]
            except IndexError:
                pass

        return "unknown_file"

    def _get_empty_search_info(self) -> Dict:
        """Return empty search info structure."""
        return {
            "user_query": "",
            "search_query": "",
            "results_found": [],
            "files_accessed": [],
            "final_response": "",
            "search_success": False,
            "error_messages": ["Could not parse message structure"],
            "response_length": 0,
        }

    def _determine_access_level(self, search_info: Dict) -> str:
        """
        Determine the level of content access achieved.

        Args:
            search_info: Search information extracted from JSON structure

        Returns:
            Access level: 'no_access', 'partial_access', 'full_access', or 'unknown_access'
        """
        # No Access: Search failed or returned errors
        if not search_info["search_success"] or any(
            "no content returned" in msg.lower()
            for msg in search_info["error_messages"]
        ):
            return "no_access"

        # No Access: No results found
        if not search_info["results_found"]:
            return "no_access"

        # Full Access: Files were found AND accessed AND content details provided
        if (
            search_info["files_accessed"]
            and search_info["results_found"]
            and search_info["response_length"] > 500
        ):

            # Check for specific content indicators in final response
            response = search_info["final_response"].lower()
            full_access_indicators = [
                "shows",
                "contains",
                "includes",
                "data",
                "according to",
                "sheet shows",
                "column",
                "values",
                "metrics",
                "analysis of",
                "based on",
                "document indicates",
            ]

            if any(indicator in response for indicator in full_access_indicators):
                return "full_access"

        # Partial Access: Files found but limited content provided
        if search_info["results_found"]:
            return "partial_access"

        return "unknown_access"

    def _calculate_json_based_similarity(
        self, control_info: Dict, experiment_info: Dict
    ) -> Dict:
        """
        Calculate similarity between control and experiment search results using JSON structure.

        Args:
            control_info: Search information from control
            experiment_info: Search information from experiment

        Returns:
            Dictionary with similarity metrics and categorization
        """
        # Extract filenames from both searches
        control_files = set()
        experiment_files = set()

        # From results_found
        for result in control_info["results_found"]:
            filename = self._extract_filename_from_title(result["title"])
            if filename:
                control_files.add(filename.lower())

        for result in experiment_info["results_found"]:
            filename = self._extract_filename_from_title(result["title"])
            if filename:
                experiment_files.add(filename.lower())

        # From files_accessed
        control_files.update(
            item["filename"].lower()
            for item in control_info["files_accessed"]
            if item["filename"] != "unknown_file"
        )
        experiment_files.update(
            item["filename"].lower()
            for item in experiment_info["files_accessed"]
            if item["filename"] != "unknown_file"
        )

        # Calculate overlap
        intersection = control_files.intersection(experiment_files)
        union = control_files.union(experiment_files)

        if not union:
            content_overlap_pct = 100.0  # Both found nothing
        else:
            content_overlap_pct = (len(intersection) / len(union)) * 100

        # Determine similarity category
        if content_overlap_pct >= 80:
            similarity_category = "Identical_Files"
        elif content_overlap_pct >= 50:
            similarity_category = "Similar_Files"
        elif content_overlap_pct >= 30:
            similarity_category = "Some_Overlap"
        else:
            similarity_category = "Different_Files"

        return {
            "content_overlap_pct": content_overlap_pct,
            "similarity_category": similarity_category,
            "control_files": list(control_files),
            "experiment_files": list(experiment_files),
            "common_files": list(intersection),
            "control_access_level": self._determine_access_level(control_info),
            "experiment_access_level": self._determine_access_level(experiment_info),
        }

    def _extract_filename_from_title(self, title: str) -> str:
        """Extract filename from result title like '<File>filename</File>'."""
        if not title:
            return ""

        import re

        match = re.search(r"<File>([^<]+)</File>", title)
        if match:
            return match.group(1).strip()

        # Fallback: return title as-is if no File tags found
        return title.strip()

    def _classify_search_domain(self, search_info: Dict) -> str:
        """
        Classify business domain based on search query and results.

        Args:
            search_info: Search information extracted from JSON structure

        Returns:
            Domain classification string
        """
        # Combine query and results text for analysis
        analysis_text = (
            search_info["user_query"]
            + " "
            + search_info["search_query"]
            + " "
            + " ".join(result["snippet"] for result in search_info["results_found"])
        ).lower()

        domain_keywords = {
            "HR": [
                "employee",
                "hr",
                "human resources",
                "personnel",
                "staff",
                "recruitment",
                "performance review",
                "onboarding",
                "benefits",
                "payroll",
            ],
            "Finance": [
                "budget",
                "financial",
                "accounting",
                "revenue",
                "cost",
                "expense",
                "invoice",
                "payment",
                "fiscal",
                "profit",
                "loss",
            ],
            "IT": [
                "technical",
                "system",
                "software",
                "database",
                "server",
                "network",
                "deployment",
                "configuration",
                "monitoring",
                "infrastructure",
            ],
            "Legal": [
                "contract",
                "legal",
                "compliance",
                "policy",
                "regulation",
                "terms",
                "agreement",
                "clause",
                "law",
                "governance",
            ],
            "Sales": [
                "sales",
                "customer",
                "client",
                "deal",
                "prospect",
                "revenue",
                "opportunity",
                "lead",
                "pipeline",
                "territory",
            ],
            "Operations": [
                "operations",
                "process",
                "workflow",
                "logistics",
                "supply",
                "procedure",
                "efficiency",
                "quality",
                "production",
            ],
            "Testing": [
                "test",
                "testing",
                "qa",
                "quality assurance",
                "verification",
                "validation",
                "bug",
                "defect",
                "automation",
                "cycle",
            ],
            "Engineering": [
                "engineering",
                "development",
                "code",
                "algorithm",
                "architecture",
                "design",
                "implementation",
                "optimization",
                "performance",
            ],
        }

        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in analysis_text)
            domain_scores[domain] = score

        if domain_scores and max(domain_scores.values()) > 0:
            return max(domain_scores.keys(), key=lambda k: domain_scores[k])
        return "General"

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
        search_dir: str = r"seval_data\212953_scraping_raw_data_output",
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
        search_dir: str = r"seval_data\212953_scraping_raw_data_output",
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
        search_dir: str = r"seval_data\212953_scraping_raw_data_output",
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

                # Extract utterance text
                utterance = data.get("query", {}).get("id", "No utterance found")

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
                    "utterance": utterance,
                    "models": list(file_models),
                    "category": file_category,
                    "consistency": consistency_info,
                }

            except Exception as e:
                return {
                    "file": file_path,
                    "utterance": "Error extracting utterance",
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
                "utterance\tfile\tmodels_found\tmodel_list\treasoning_model\tconsistent\tcategory\terror\n"
            )

            # Write detailed results
            for stat in file_stats:
                filename = os.path.basename(stat["file"])
                utterance = stat.get("utterance", "No utterance found")
                models_count = len(stat["models"])
                models_list = "; ".join(stat["models"]) if stat["models"] else ""
                error = stat.get("error", "")

                # Add consistency information
                consistency_info = stat.get("consistency", {})
                is_consistent = consistency_info.get("consistent", True)
                reasoning_model = consistency_info.get("reasoning_model", "")
                category = stat.get("category", "unknown")

                # Clean utterance text (remove tabs and newlines for TSV format)
                utterance_clean = (
                    utterance.replace("\t", " ")
                    .replace("\n", " ")
                    .replace("\r", " ")
                    .strip()
                )

                f.write(
                    f"{utterance_clean}\t{filename}\t{models_count}\t{models_list}\t{reasoning_model}\t{is_consistent}\t{category}\t{error}\n"
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

    def analyze_search_results(
        self,
        mappings_file: str = "results/query_file_mappings.tsv",
        output_file: str = "results/search_results_analysis.tsv",
        search_dir: str = "seval_data/212953_scraping_raw_data_output",
        threads: int = 8,
        max_queries: int = 0,
    ) -> None:
        """
        Analyze search results from SEVAL files to extract queries, search results, and content access patterns.

        This function:
        1. Extracts query, search results, and content access info for all rounds
        2. Labels which searches succeeded/failed and what content was accessed
        3. Compares control vs experiment files for each query
        4. Outputs detailed TSV format with query comparisons
        5. Provides console statistics

        Args:
            mappings_file: Path to TSV file containing query-to-file mappings from extract_query_mappings
            output_file: Path to output TSV file for detailed results
            search_dir: Directory containing SEVAL JSON files
            threads: Number of threads for parallel processing
            max_queries: Maximum number of queries to process (0 = all)
        """
        from datetime import datetime

        print("🔍 Starting SEVAL Search Results Analysis...")
        logger.info(f"Analyzing search results using mappings from: {mappings_file}")

        # Load query mappings
        query_mappings = self._load_query_mappings(mappings_file)
        if not query_mappings:
            logger.error(f"No query mappings found in {mappings_file}")
            return

        # Filter queries if max_queries is specified
        if max_queries > 0:
            query_mappings = dict(list(query_mappings.items())[:max_queries])

        logger.info(f"Processing {len(query_mappings)} unique queries...")

        # Process queries with threading
        results = []
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_query = {
                executor.submit(
                    self._analyze_query_search_results, query, file_info, search_dir
                ): query
                for query, file_info in query_mappings.items()
            }

            completed = 0
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    completed += 1
                    if completed % 50 == 0:
                        logger.info(
                            f"Processed {completed}/{len(query_mappings)} queries..."
                        )
                except Exception as exc:
                    logger.error(f"Query '{query}' generated an exception: {exc}")

        # Sort results by query for consistent output
        results.sort(key=lambda x: x["query"])

        # Convert old format results to new format for JSON-based analysis
        converted_results = []
        for result in results:
            # Extract control and experiment file paths
            control_file = ""
            experiment_file = ""

            if result.get("control_analysis"):
                control_files = [
                    a.get("filename", "") for a in result["control_analysis"]
                ]
                control_file = control_files[0] if control_files else ""

            if result.get("experiment_analysis"):
                experiment_files = [
                    a.get("filename", "") for a in result["experiment_analysis"]
                ]
                experiment_file = experiment_files[0] if experiment_files else ""

            converted_result = {
                "query_id": f"query_{len(converted_results) + 1}",
                "query": result["query"],
                "control_file": (
                    os.path.join(search_dir, control_file) if control_file else ""
                ),
                "experiment_file": (
                    os.path.join(search_dir, experiment_file) if experiment_file else ""
                ),
            }
            converted_results.append(converted_result)

        # Create clean TSV output using JSON-based analysis
        clean_output_file = output_file.replace(".tsv", "_clean_json_based.tsv")
        self.create_clean_tsv_output(converted_results, clean_output_file)

        # Also create traditional output for compatibility
        self._write_search_analysis_tsv(results, output_file)

        # Display console statistics
        self._display_search_statistics(results)

        print(f"\n📊 Search results analysis completed!")
        print(f"Traditional results written to: {output_file}")
        print(f"Clean JSON-based results written to: {clean_output_file}")
        logger.info(
            f"Search results analysis completed. Processed {len(results)} queries."
        )

    def _load_query_mappings(self, mappings_file: str) -> Dict[str, Dict]:
        """Load query-to-file mappings from TSV file."""
        query_mappings = {}
        try:
            with open(mappings_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    # Handle both old format (query) and new format (query_text)
                    query = row.get("query") or row.get("query_text", "")
                    if not query:
                        continue

                    if query not in query_mappings:
                        query_mappings[query] = {"control": [], "experiment": []}

                    # Handle different column name formats
                    control_file = row.get("control_file", "")
                    experiment_file = row.get("experiment_file", "")

                    # If using the old format with experiment_type column
                    if "experiment_type" in row:
                        exp_type = row["experiment_type"]
                        filename = row.get("filename", "")
                        file_path = row.get("file_path", "")

                        if exp_type in ["control", "experiment"] and filename:
                            query_mappings[query][exp_type].append(
                                {"filename": filename, "file_path": file_path}
                            )
                    else:
                        # New format with separate control/experiment file columns
                        if control_file:
                            query_mappings[query]["control"].append(
                                {"filename": control_file, "file_path": control_file}
                            )
                        if experiment_file:
                            query_mappings[query]["experiment"].append(
                                {
                                    "filename": experiment_file,
                                    "file_path": experiment_file,
                                }
                            )

        except Exception as e:
            logger.error(f"Error loading query mappings: {e}")
            return {}

        return query_mappings

    def _analyze_query_search_results(
        self, query: str, file_info: Dict, search_dir: str
    ) -> Optional[Dict]:
        """Analyze search results for a single query across control and experiment files."""

        result = {"query": query, "control_analysis": None, "experiment_analysis": None}

        # Analyze control files
        if file_info.get("control"):
            control_analyses = []
            for file_info_item in file_info["control"]:
                file_path = Path(search_dir) / file_info_item["filename"]
                analysis = self._analyze_file_search_results(file_path, query)
                if analysis:
                    control_analyses.append(analysis)
            result["control_analysis"] = control_analyses

        # Analyze experiment files
        if file_info.get("experiment"):
            experiment_analyses = []
            for file_info_item in file_info["experiment"]:
                file_path = Path(search_dir) / file_info_item["filename"]
                analysis = self._analyze_file_search_results(file_path, query)
                if analysis:
                    experiment_analyses.append(analysis)
            result["experiment_analysis"] = experiment_analyses

        return (
            result
            if (result["control_analysis"] or result["experiment_analysis"])
            else None
        )

    def _analyze_file_search_results(
        self, file_path: Path, query: str
    ) -> Optional[Dict]:
        """Analyze search results within a single SEVAL JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            analysis = {
                "filename": file_path.name,
                "query": query,
                "conversation_success": data.get("conversation_success", False),
                "search_rounds": [],
                "search_summary": {
                    "total_searches": 0,
                    "successful_searches": 0,
                    "failed_searches": 0,
                    "content_access_attempts": 0,
                    "successful_content_access": 0,
                },
            }

            # Extract search operations from conversation messages
            # Handle both direct 'messages' and 'requests.response_body.messages' structures
            messages = []
            if "messages" in data:
                messages = data["messages"]
            elif "requests" in data and data["requests"]:
                # SEVAL format: messages are in requests[0].response_body.messages
                first_request = data["requests"][0]
                if isinstance(first_request, dict) and "response_body" in first_request:
                    response_body = first_request["response_body"]
                    if isinstance(response_body, dict) and "messages" in response_body:
                        messages = response_body["messages"]

            for message in messages:
                # Look for office365_search function calls
                invocation = message.get("invocation", "")
                if "office365_search" in invocation:
                    search_info = self._extract_search_info_from_invocation(
                        invocation, message
                    )
                    if search_info:
                        analysis["search_rounds"].extend(search_info)

            # Update summary statistics
            analysis["search_summary"]["total_searches"] = len(
                analysis["search_rounds"]
            )
            for search_round in analysis["search_rounds"]:
                if search_round["success"]:
                    analysis["search_summary"]["successful_searches"] += 1
                else:
                    analysis["search_summary"]["failed_searches"] += 1

                if search_round["content_accessed"]:
                    analysis["search_summary"]["content_access_attempts"] += 1
                    if search_round["content_access_success"]:
                        analysis["search_summary"]["successful_content_access"] += 1

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return None

    def _extract_search_info_from_invocation(
        self, invocation: str, message: Dict
    ) -> List[Dict]:
        """Extract search information from function invocation and message context."""
        search_info = []

        try:
            # Parse invocation to extract search queries - handle escaped JSON
            if "office365_search" in invocation:
                # The invocation is escaped JSON inside a string, need to carefully parse
                # Look for the queries pattern with proper unescaping

                # First, try to find all query objects by looking for domain/query pairs
                domain_query_pairs = []

                # Pattern to match domain and query in escaped JSON
                domain_pattern = r'\\"domain\\":\s*\\"([^"\\]+)\\"'
                query_pattern = r'\\"query\\":\s*\\"([^"\\]+)\\"'

                domains = re.findall(domain_pattern, invocation)
                queries = re.findall(query_pattern, invocation)

                # Match domains with queries (they should appear in pairs)
                for i, domain in enumerate(domains):
                    if i < len(queries):
                        search_query = queries[i]

                        # Check for search results in message
                        search_result = self._extract_search_results_from_message(
                            message, domain
                        )

                        search_info.append(
                            {
                                "domain": domain,
                                "search_query": search_query,
                                "success": search_result["success"],
                                "total_results": search_result["total_results"],
                                "results_snippet": search_result["results_snippet"],
                                "content_accessed": search_result["content_accessed"],
                                "content_access_success": search_result[
                                    "content_access_success"
                                ],
                                "failure_reason": search_result["failure_reason"],
                            }
                        )

        except Exception as e:
            logger.debug(f"Error extracting search info from invocation: {e}")

        return search_info

    def _extract_search_results_from_message(self, message: Dict, domain: str) -> Dict:
        """Extract search results information from a message."""
        result = {
            "success": False,
            "total_results": 0,
            "results_snippet": "",
            "content_accessed": False,
            "content_access_success": False,
            "failure_reason": "",
        }

        try:
            # Check message text for search results
            text = message.get("text", "")

            # Look for search metadata patterns
            if "searchMetadata" in text:
                # Parse search metadata
                if (
                    '"status":"No results found."' in text
                    or '"status": "No results found."' in text
                ):
                    result["failure_reason"] = "No results found"
                elif '"totalResults":0' in text or '"totalResults": 0' in text:
                    result["failure_reason"] = "Zero results"
                elif "totalResults" in text:
                    # Extract total results number - try multiple patterns
                    total_results_matches = re.findall(
                        r'"totalResults":(\d+)|"totalResults":\s*"(\d+)"', text
                    )
                    if total_results_matches:
                        # Get the first non-empty match
                        for match_group in total_results_matches:
                            for match in match_group:
                                if match:
                                    result["total_results"] = int(match)
                                    result["success"] = True
                                    break
                            if result["success"]:
                                break

                        # Extract results snippet if successful
                        if result["success"]:
                            if '"results":' in text:
                                results_start = text.find('"results":')
                                results_end = text.find(
                                    ',"searchMetadata"', results_start
                                )
                                if results_end == -1:
                                    results_end = text.find(
                                        '},"searchMetadata"', results_start
                                    )
                                if results_end == -1:
                                    results_end = text.find(
                                        '"searchMetadata"', results_start
                                    )
                                if results_end != -1:
                                    snippet_text = text[results_start:results_end]
                                    # Limit snippet length
                                    if len(snippet_text) > 500:
                                        snippet_text = snippet_text[:500] + "..."
                                    result["results_snippet"] = snippet_text

            # Check telemetry for grounding response (indicates content access)
            telemetry = message.get("telemetry", {})
            if "groundingResponse" in telemetry:
                result["content_accessed"] = True
                grounding_response = telemetry["groundingResponse"]

                # Check if content access was successful
                if isinstance(grounding_response, dict):
                    grounding_total = grounding_response.get("searchMetadata", {}).get(
                        "totalResults", 0
                    )
                    if isinstance(grounding_total, str):
                        try:
                            grounding_total = int(grounding_total)
                        except:
                            grounding_total = 0

                    if grounding_total > 0:
                        result["content_access_success"] = True
                    elif "status" in grounding_response.get("searchMetadata", {}):
                        status = grounding_response["searchMetadata"]["status"]
                        if "No results found" in status:
                            result["failure_reason"] = (
                                "Content access failed - no results"
                            )

            # Additional check: Look for empty results arrays
            if '"results":[]' in text:
                result["failure_reason"] = "Empty results array"
                result["total_results"] = 0

        except Exception as e:
            logger.debug(f"Error extracting search results from message: {e}")
            result["failure_reason"] = f"Parsing error: {str(e)}"

        return result

    def _write_search_analysis_tsv(self, results: List[Dict], output_file: str) -> None:
        """Write detailed search analysis results to TSV file (pure TSV format for Excel)."""
        from datetime import datetime

        os.makedirs(Path(output_file).parent, exist_ok=True)

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")

            # Write header (pure TSV, no comments)
            header = [
                "query",
                "control_files_count",
                "experiment_files_count",
                "control_final_search_results",
                "experiment_final_search_results",
                "search_result_similarity",
                "mentions_specific_content",
                "content_found_by_both",
                "both_accessed_content",
                "fair_comparison_candidate",
                "control_total_searches",
                "control_successful_searches",
                "experiment_total_searches",
                "experiment_successful_searches",
                "control_files",
                "experiment_files",
                "control_final_search_details",
                "experiment_final_search_details",
            ]
            writer.writerow(header)

            # Write data rows, focusing on final search results for fair comparison
            for result in results:
                control_analyses = result.get("control_analysis", []) or []
                experiment_analyses = result.get("experiment_analysis", []) or []

                # Skip queries without both control and experiment data
                if not (control_analyses and experiment_analyses):
                    continue

                # Aggregate control statistics
                control_stats = self._aggregate_analysis_stats(control_analyses)
                experiment_stats = self._aggregate_analysis_stats(experiment_analyses)

                # Extract final search results (last successful search rounds)
                control_final_results = self._extract_final_search_results(
                    control_analyses
                )
                experiment_final_results = self._extract_final_search_results(
                    experiment_analyses
                )

                # Calculate search result similarity
                similarity = self._calculate_search_similarity(
                    control_final_results, experiment_final_results
                )

                # Analyze content matching for this specific query
                query_content_analysis = self._analyze_query_content_matching(result)

                # Prepare file lists
                control_files = ";".join([a["filename"] for a in control_analyses])
                experiment_files = ";".join(
                    [a["filename"] for a in experiment_analyses]
                )

                # Prepare final search details (focusing on last successful searches)
                control_final_details = self._format_final_search_details(
                    control_analyses
                )
                experiment_final_details = self._format_final_search_details(
                    experiment_analyses
                )

                row = [
                    result["query"],
                    len(control_analyses),
                    len(experiment_analyses),
                    control_final_results,
                    experiment_final_results,
                    similarity,
                    query_content_analysis["mentions_specific_content"],
                    query_content_analysis["content_found_by_both"],
                    query_content_analysis["both_accessed_content"],
                    query_content_analysis["fair_comparison_candidate"],
                    control_stats["total_searches"],
                    control_stats["successful_searches"],
                    experiment_stats["total_searches"],
                    experiment_stats["successful_searches"],
                    control_files,
                    experiment_files,
                    control_final_details,
                    experiment_final_details,
                ]
                writer.writerow(row)

    def _aggregate_analysis_stats(self, analyses: List[Dict]) -> Dict:
        """Aggregate statistics from multiple file analyses."""
        stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "content_access_attempts": 0,
            "successful_content_access": 0,
        }

        for analysis in analyses:
            summary = analysis.get("search_summary", {})
            stats["total_searches"] += summary.get("total_searches", 0)
            stats["successful_searches"] += summary.get("successful_searches", 0)
            stats["failed_searches"] += summary.get("failed_searches", 0)
            stats["content_access_attempts"] += summary.get(
                "content_access_attempts", 0
            )
            stats["successful_content_access"] += summary.get(
                "successful_content_access", 0
            )

        return stats

    def _extract_final_search_results(self, analyses: List[Dict]) -> str:
        """Extract final search results (last successful searches) for comparison."""
        final_results = []

        for analysis in analyses:
            search_rounds = analysis.get("search_rounds", [])
            if not search_rounds:
                continue

            # Find the last few successful search rounds that produced results
            successful_rounds = [
                r for r in search_rounds if r["success"] and r["total_results"] > 0
            ]

            if successful_rounds:
                # Take last 3 successful rounds or all if fewer
                last_rounds = successful_rounds[-3:]
                for round_info in last_rounds:
                    result_info = (
                        f"{round_info['domain']}({round_info['total_results']})"
                    )
                    if result_info not in final_results:
                        final_results.append(result_info)

        return ";".join(final_results) if final_results else "No results"

    def _calculate_search_similarity(
        self, control_results: str, experiment_results: str
    ) -> str:
        """Calculate similarity between search results to identify comparable queries."""
        if control_results == "No results" and experiment_results == "No results":
            return "Both_Empty"

        if control_results == "No results" or experiment_results == "No results":
            return "One_Empty"

        # Parse domain-based results
        control_domains = set()
        experiment_domains = set()

        if control_results != "No results":
            for item in control_results.split(";"):
                if "(" in item:
                    domain = item.split("(")[0]
                    control_domains.add(domain)

        if experiment_results != "No results":
            for item in experiment_results.split(";"):
                if "(" in item:
                    domain = item.split("(")[0]
                    experiment_domains.add(domain)

        # Calculate domain overlap
        if control_domains and experiment_domains:
            overlap = len(control_domains.intersection(experiment_domains))
            total_unique = len(control_domains.union(experiment_domains))

            if overlap == 0:
                return "No_Overlap"
            elif overlap == len(control_domains) == len(experiment_domains):
                return "Identical_Domains"
            elif overlap > 0:
                similarity_pct = int((overlap / total_unique) * 100)
                return f"Partial_Overlap_{similarity_pct}%"

        return "Unknown"

    def _format_final_search_details(self, analyses: List[Dict]) -> str:
        """Format final search details focusing on last successful searches."""
        details = []

        for analysis in analyses:
            search_rounds = analysis.get("search_rounds", [])
            if not search_rounds:
                continue

            # Find last successful searches
            successful_rounds = [
                r for r in search_rounds if r["success"] and r["total_results"] > 0
            ]

            if successful_rounds:
                # Take last 3 successful rounds
                last_rounds = successful_rounds[-3:]
                round_details = []

                for search_round in last_rounds:
                    round_info = f"{search_round['domain']}:{search_round['total_results']}_results"
                    round_details.append(round_info)

                if round_details:
                    file_detail = f"{analysis['filename']}[{','.join(round_details)}]"
                    details.append(file_detail)

        return " | ".join(details)

    def _format_search_details(self, analyses: List[Dict]) -> str:
        """Format detailed search information for TSV output."""
        details = []

        for analysis in analyses:
            file_details = f"{analysis['filename']}:"
            search_rounds = analysis.get("search_rounds", [])

            round_details = []
            for search_round in search_rounds:
                round_info = (
                    f"domain={search_round['domain']}, "
                    f"success={search_round['success']}, "
                    f"results={search_round['total_results']}"
                )
                if search_round["failure_reason"]:
                    round_info += f", failure={search_round['failure_reason']}"
                round_details.append(round_info)

            if round_details:
                file_details += "[" + "; ".join(round_details) + "]"
            details.append(file_details)

        return " | ".join(details)

    def _analyze_content_matching(self, results: List[Dict]) -> Dict:
        """Analyze content matching between control and experiment for fair comparison."""
        import re

        analysis = {
            "specific_content_queries": 0,
            "both_found_same_content": 0,
            "both_full_content_access": 0,
            "fair_comparison_candidates": 0,
        }

        # File extensions and email patterns to identify specific content mentions
        file_patterns = [
            r"\w+\.(docx?|xlsx?|pptx?|pdf|txt|csv|json|xml)",  # Common file extensions
            r"\w+_\w+.*\.(docx?|xlsx?|pptx?|pdf)",  # Files with underscores (common pattern)
        ]

        email_patterns = [
            r"\w+@\w+\.\w+",  # Email addresses
            r"email.*from.*\w+",  # "email from person"
            r"emails.*between.*\w+",  # "emails between people"
        ]

        for result in results:
            query = result["query"].lower()

            # Check if query mentions specific files or emails
            mentions_specific_content = False

            # Check for specific file mentions
            for pattern in file_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    mentions_specific_content = True
                    break

            # Check for email mentions
            if not mentions_specific_content:
                for pattern in email_patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        mentions_specific_content = True
                        break

            # Also check for explicit file/email keywords
            if not mentions_specific_content:
                content_keywords = [
                    "file",
                    "document",
                    "attachment",
                    "email",
                    "message",
                    "report",
                    "spreadsheet",
                    "presentation",
                    "pdf",
                    "docx",
                ]
                for keyword in content_keywords:
                    if keyword in query and any(
                        char in query for char in [".", "_", "-"]
                    ):
                        mentions_specific_content = True
                        break

            if not mentions_specific_content:
                continue

            analysis["specific_content_queries"] += 1

            # Analyze if both control and experiment found similar content
            control_analyses = result.get("control_analysis", []) or []
            experiment_analyses = result.get("experiment_analysis", []) or []

            if not (control_analyses and experiment_analyses):
                continue

            # Extract found content from both sides
            control_content = self._extract_found_content(control_analyses)
            experiment_content = self._extract_found_content(experiment_analyses)

            # Check for content similarity
            content_similarity = self._calculate_content_similarity(
                control_content, experiment_content
            )

            if content_similarity["has_overlap"]:
                analysis["both_found_same_content"] += 1

                # Check for full content access
                if content_similarity["both_have_full_access"]:
                    analysis["both_full_content_access"] += 1

                # Mark as fair comparison candidate if reasonable overlap
                if content_similarity["similarity_score"] >= 0.3:  # 30% or more overlap
                    analysis["fair_comparison_candidates"] += 1

        return analysis

    def _extract_found_content(self, analyses: List[Dict]) -> Dict:
        """Extract information about found content from analyses."""
        content_info = {
            "successful_searches": [],
            "content_access_attempts": 0,
            "successful_content_access": 0,
            "domains_searched": set(),
            "total_results_found": 0,
        }

        for analysis in analyses:
            search_rounds = analysis.get("search_rounds", [])

            for search_round in search_rounds:
                if search_round["success"] and search_round["total_results"] > 0:
                    content_info["successful_searches"].append(
                        {
                            "domain": search_round["domain"],
                            "results": search_round["total_results"],
                            "content_accessed": search_round.get(
                                "content_accessed", False
                            ),
                            "content_access_success": search_round.get(
                                "content_access_success", False
                            ),
                        }
                    )
                    content_info["domains_searched"].add(search_round["domain"])
                    content_info["total_results_found"] += search_round["total_results"]

                if search_round.get("content_accessed", False):
                    content_info["content_access_attempts"] += 1
                    if search_round.get("content_access_success", False):
                        content_info["successful_content_access"] += 1

        return content_info

    def _calculate_content_similarity(
        self, control_content: Dict, experiment_content: Dict
    ) -> Dict:
        """Calculate similarity between found content in control and experiment."""
        similarity = {
            "has_overlap": False,
            "both_have_full_access": False,
            "similarity_score": 0.0,
            "domain_overlap": 0,
        }

        # Check domain overlap
        control_domains = control_content["domains_searched"]
        experiment_domains = experiment_content["domains_searched"]

        if control_domains and experiment_domains:
            overlap = len(control_domains.intersection(experiment_domains))
            total_unique = len(control_domains.union(experiment_domains))

            similarity["domain_overlap"] = overlap
            similarity["has_overlap"] = overlap > 0

            if total_unique > 0:
                similarity["similarity_score"] = overlap / total_unique

        # Check if both have successful content access
        similarity["both_have_full_access"] = (
            control_content["successful_content_access"] > 0
            and experiment_content["successful_content_access"] > 0
        )

        return similarity

    def _analyze_query_content_matching(self, result: Dict) -> Dict:
        """Analyze content matching for a single query using JSON structure analysis."""

        # Extract search information from control and experiment files
        control_info = None
        experiment_info = None

        # Load control file if available
        control_file = result.get("control_file")
        if control_file:
            try:
                with open(control_file, "r", encoding="utf-8") as f:
                    control_data = json.load(f)
                control_info = self._extract_search_information(control_data)
            except Exception as e:
                logger.warning(f"Could not parse control file {control_file}: {e}")

        # Load experiment file if available
        experiment_file = result.get("experiment_file")
        if experiment_file:
            try:
                with open(experiment_file, "r", encoding="utf-8") as f:
                    experiment_data = json.load(f)
                experiment_info = self._extract_search_information(experiment_data)
            except Exception as e:
                logger.warning(
                    f"Could not parse experiment file {experiment_file}: {e}"
                )

        # Default analysis results
        analysis = {
            "mentions_specific_content": False,
            "content_found_by_both": False,
            "both_accessed_content": False,
            "fair_comparison_candidate": False,
            "control_files_found": [],
            "experiment_files_found": [],
            "control_files_accessed": [],
            "experiment_files_accessed": [],
            "content_overlap_pct": 0.0,
            "similarity_category": "No_Data",
            "control_access_level": "unknown",
            "experiment_access_level": "unknown",
            "control_search_success": False,
            "experiment_search_success": False,
            "control_domain": "General",
            "experiment_domain": "General",
        }

        # Check if query mentions specific content (files, documents, etc.)
        query = result["query"].lower()
        specific_content_indicators = [
            ".docx",
            ".xlsx",
            ".pptx",
            ".pdf",
            ".txt",
            ".csv",
            "file",
            "document",
            "email",
            "attachment",
            "report",
            "spreadsheet",
            "presentation",
            "_",
            "-",
        ]

        analysis["mentions_specific_content"] = any(
            indicator in query for indicator in specific_content_indicators
        )

        # Perform detailed analysis if we have data from both sides
        if control_info and experiment_info:
            # Calculate similarity using JSON structure
            similarity_data = self._calculate_json_based_similarity(
                control_info, experiment_info
            )

            # Update analysis with detailed results
            analysis.update(
                {
                    "content_found_by_both": len(similarity_data["common_files"]) > 0,
                    "both_accessed_content": (
                        similarity_data["control_access_level"] == "full_access"
                        and similarity_data["experiment_access_level"] == "full_access"
                    ),
                    "fair_comparison_candidate": (
                        similarity_data["content_overlap_pct"] >= 30
                        and control_info["search_success"]
                        and experiment_info["search_success"]
                    ),
                    "control_files_found": [
                        r["title"] for r in control_info["results_found"]
                    ],
                    "experiment_files_found": [
                        r["title"] for r in experiment_info["results_found"]
                    ],
                    "control_files_accessed": [
                        f["filename"] for f in control_info["files_accessed"]
                    ],
                    "experiment_files_accessed": [
                        f["filename"] for f in experiment_info["files_accessed"]
                    ],
                    "content_overlap_pct": similarity_data["content_overlap_pct"],
                    "similarity_category": similarity_data["similarity_category"],
                    "control_access_level": similarity_data["control_access_level"],
                    "experiment_access_level": similarity_data[
                        "experiment_access_level"
                    ],
                    "control_search_success": control_info["search_success"],
                    "experiment_search_success": experiment_info["search_success"],
                    "control_domain": self._classify_search_domain(control_info),
                    "experiment_domain": self._classify_search_domain(experiment_info),
                }
            )

        elif control_info:
            # Only control data available
            analysis.update(
                {
                    "control_files_found": [
                        r["title"] for r in control_info["results_found"]
                    ],
                    "control_files_accessed": [
                        f["filename"] for f in control_info["files_accessed"]
                    ],
                    "control_access_level": self._determine_access_level(control_info),
                    "control_search_success": control_info["search_success"],
                    "control_domain": self._classify_search_domain(control_info),
                }
            )

        elif experiment_info:
            # Only experiment data available
            analysis.update(
                {
                    "experiment_files_found": [
                        r["title"] for r in experiment_info["results_found"]
                    ],
                    "experiment_files_accessed": [
                        f["filename"] for f in experiment_info["files_accessed"]
                    ],
                    "experiment_access_level": self._determine_access_level(
                        experiment_info
                    ),
                    "experiment_search_success": experiment_info["search_success"],
                    "experiment_domain": self._classify_search_domain(experiment_info),
                }
            )

        return analysis

    def _display_search_statistics(self, results: List[Dict]) -> None:
        """Display console statistics for search results analysis."""
        print(f"\n🔍 SEVAL Search Results Analysis Summary")
        print("=" * 80)

        total_queries = len(results)
        queries_with_control = sum(1 for r in results if r.get("control_analysis"))
        queries_with_experiment = sum(
            1 for r in results if r.get("experiment_analysis")
        )
        queries_with_both = sum(
            1
            for r in results
            if r.get("control_analysis") and r.get("experiment_analysis")
        )

        # Calculate detailed distribution
        queries_control_only = queries_with_control - queries_with_both
        queries_experiment_only = queries_with_experiment - queries_with_both

        print(f"📊 Query Coverage:")
        print(f"  Total queries analyzed: {total_queries}")
        print(
            f"  Queries with both types: {queries_with_both} ({100*queries_with_both/total_queries:.1f}%)"
        )
        print(
            f"  Queries with control only: {queries_control_only} ({100*queries_control_only/total_queries:.1f}%)"
        )
        print(
            f"  Queries with experiment only: {queries_experiment_only} ({100*queries_experiment_only/total_queries:.1f}%)"
        )

        # Analyze content matching for fair comparison
        content_analysis = self._analyze_content_matching(results)
        print(f"\n📋 Content Matching Analysis (for fair comparison):")
        print(
            f"  Queries mentioning specific files/emails: {content_analysis['specific_content_queries']}"
        )
        print(
            f"  Both found same specific content: {content_analysis['both_found_same_content']} ({100*content_analysis['both_found_same_content']/max(content_analysis['specific_content_queries'], 1):.1f}%)"
        )
        print(
            f"  Both accessed full content: {content_analysis['both_full_content_access']} ({100*content_analysis['both_full_content_access']/max(content_analysis['specific_content_queries'], 1):.1f}%)"
        )
        print(
            f"  Fair comparison candidates: {content_analysis['fair_comparison_candidates']} queries"
        )

        # Aggregate search statistics
        control_total_searches = 0
        control_successful_searches = 0
        experiment_total_searches = 0
        experiment_successful_searches = 0
        control_content_access = 0
        experiment_content_access = 0

        for result in results:
            control_analyses = result.get("control_analysis", []) or []
            experiment_analyses = result.get("experiment_analysis", []) or []

            control_stats = self._aggregate_analysis_stats(control_analyses)
            experiment_stats = self._aggregate_analysis_stats(experiment_analyses)

            control_total_searches += control_stats["total_searches"]
            control_successful_searches += control_stats["successful_searches"]
            control_content_access += control_stats["successful_content_access"]

            experiment_total_searches += experiment_stats["total_searches"]
            experiment_successful_searches += experiment_stats["successful_searches"]
            experiment_content_access += experiment_stats["successful_content_access"]

        print(f"\n🔍 Search Operation Statistics:")
        print(
            f"  Control searches: {control_successful_searches}/{control_total_searches} successful ({100*control_successful_searches/max(control_total_searches, 1):.1f}%)"
        )
        print(
            f"  Experiment searches: {experiment_successful_searches}/{experiment_total_searches} successful ({100*experiment_successful_searches/max(experiment_total_searches, 1):.1f}%)"
        )

        print(f"\n📄 Content Access Statistics:")
        print(f"  Control successful content access: {control_content_access}")
        print(f"  Experiment successful content access: {experiment_content_access}")

        # Search failure analysis
        failure_patterns = {"control": {}, "experiment": {}}
        for result in results:
            for exp_type in ["control", "experiment"]:
                analyses = result.get(f"{exp_type}_analysis", []) or []
                for analysis in analyses:
                    for search_round in analysis.get("search_rounds", []):
                        if (
                            not search_round["success"]
                            and search_round["failure_reason"]
                        ):
                            failure_reason = search_round["failure_reason"]
                            failure_patterns[exp_type][failure_reason] = (
                                failure_patterns[exp_type].get(failure_reason, 0) + 1
                            )

        print(f"\n❌ Search Failure Patterns:")
        for exp_type in ["control", "experiment"]:
            if failure_patterns[exp_type]:
                print(f"  {exp_type.title()}:")
                for reason, count in sorted(
                    failure_patterns[exp_type].items(), key=lambda x: -x[1]
                ):
                    print(f"    '{reason}': {count} occurrences")
            else:
                print(f"  {exp_type.title()}: No failures detected")

    def create_clean_tsv_output(self, results: List[Dict], output_file: str) -> str:
        """Create a clean TSV file without header comments, using JSON-based analysis."""

        # Define TSV headers
        headers = [
            "query_id",
            "query",
            "control_file",
            "experiment_file",
            "mentions_specific_content",
            "content_found_by_both",
            "both_accessed_content",
            "fair_comparison_candidate",
            "control_files_found",
            "experiment_files_found",
            "control_files_accessed",
            "experiment_files_accessed",
            "content_overlap_pct",
            "similarity_category",
            "control_access_level",
            "experiment_access_level",
            "control_search_success",
            "experiment_search_success",
            "control_domain",
            "experiment_domain",
        ]

        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")

            # Write header row (no comments)
            writer.writeheader()

            # Process each result with JSON-based analysis
            for result in results:
                content_analysis = self._analyze_query_content_matching(result)

                # Prepare row data
                row = {
                    "query_id": result.get("query_id", ""),
                    "query": result.get("query", ""),
                    "control_file": result.get("control_file", ""),
                    "experiment_file": result.get("experiment_file", ""),
                    "mentions_specific_content": content_analysis[
                        "mentions_specific_content"
                    ],
                    "content_found_by_both": content_analysis["content_found_by_both"],
                    "both_accessed_content": content_analysis["both_accessed_content"],
                    "fair_comparison_candidate": content_analysis[
                        "fair_comparison_candidate"
                    ],
                    "control_files_found": "|".join(
                        content_analysis["control_files_found"]
                    ),
                    "experiment_files_found": "|".join(
                        content_analysis["experiment_files_found"]
                    ),
                    "control_files_accessed": "|".join(
                        content_analysis["control_files_accessed"]
                    ),
                    "experiment_files_accessed": "|".join(
                        content_analysis["experiment_files_accessed"]
                    ),
                    "content_overlap_pct": f"{content_analysis['content_overlap_pct']:.1f}",
                    "similarity_category": content_analysis["similarity_category"],
                    "control_access_level": content_analysis["control_access_level"],
                    "experiment_access_level": content_analysis[
                        "experiment_access_level"
                    ],
                    "control_search_success": content_analysis[
                        "control_search_success"
                    ],
                    "experiment_search_success": content_analysis[
                        "experiment_search_success"
                    ],
                    "control_domain": content_analysis["control_domain"],
                    "experiment_domain": content_analysis["experiment_domain"],
                }

                writer.writerow(row)

        logger.info(f"Created clean TSV output with {len(results)} rows: {output_file}")
        return output_file


def main():
    """Main function to run the SEVAL Analysis Toolkit with Fire."""
    fire.Fire(SEVALAnalysisToolkit)


if __name__ == "__main__":
    main()
