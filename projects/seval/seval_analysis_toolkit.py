#!/usr/bin/env python3
"""
SEVAL Analysis Toolkit

This program provides comprehensive analysis tools for SEVAL results including
search functionality, model statistics extraction, and search results analysis.

Usage:
    # Search for SEVAL files (file names) by partial query text and experiment type.
    python seval_analysis_toolkit.py search_query --exp control --query "microsoft"
    python seval_analysis_toolkit.py search_query --exp experiment --query "azure"
    python seval_analysis_toolkit.py search_query --exp both --query "teams"
    python seval_analysis_toolkit.py search_query --query "search without exp filter"

    # Generate query mappings (each query to control/experiment file pairs) for faster searches
    python seval_analysis_toolkit.py extract_query_mappings --threads 16

    # Fast search using pre-generated mappings
    python seval_analysis_toolkit.py search_using_mappings --query "microsoft"

    # Extract model usage statistics from SEVAL files
    python seval_analysis_toolkit.py extract_model_statistics --input_dir "seval_data" --output_file "model_stats.tsv"
    python seval_analysis_toolkit.py extract_model_statistics --input_dir "seval_data" --output_file "model_stats.tsv" --exp control
    python seval_analysis_toolkit.py extract_model_statistics --input_dir "seval_data" --output_file "model_stats.tsv" --exp experiment --threads 16

    # Analyze search results patterns (requires query mappings)
    python seval_analysis_toolkit.py analyze_search_results --mappings_file "results/query_file_mappings.tsv" --output_file "results/search_analysis.tsv"
    python seval_analysis_toolkit.py analyze_search_results --threads 16 --max_queries 100

    # Extract detailed conversation information from SEVAL file (extracts turn/hop data, tool invocations, search results from messages[] array and EvaluationData message)
    python seval_analysis_toolkit.py extract_conversation_details --input_file "seval_data/experiment_file.json" --output_file "analysis_report.md"
"""

# DATA STRUCTURE REFERENCE FOR SEVAL FILES:
#
# Search results and tool invocations can be found in multiple locations:
#
# 1. EvaluationData Message (COMPLETE SOURCE - contains ALL search results):
#    Path: messages[] → {"messageType": "EvaluationData"} → evaluationData
#    Structure:
#      evaluationData:
#        turnData: [                                    # Array of conversation turns
#          {
#            userInput: "user's query text",          # The user's question/request
#            orchestrationIterations: [                # Multiple search/reasoning iterations per turn
#              {
#                modelActions: [                       # Actions taken by the model
#                  {
#                    toolInvocations: [                # Tool calls (search, etc.)
#                      {
#                        batchedQueries: [             # Parallel queries to different domains
#                          {
#                            arguments: "JSON string",  # Parse to get: {"domain": "emails|files|chats|people", "query": "search text"}
#                            processedResult: "JSON string",  # Parse to get: {"results": [{reference_id, type, title, snippet, author, ...}]}
#                            result: "raw search API response"
#                          }
#                        ]
#                      }
#                    ]
#                  }
#                ]
#              }
#            ]
#          }
#        ]
#    Note: Both 'arguments' and 'processedResult' are JSON-encoded strings, not objects
#
# 2. InternalSearchResult Messages (PARTIAL - only shows results displayed to user):
#    Path: messages[] → {"messageType": "InternalSearchResult"}
#    Contains: Subset of search results that were actually cited or shown
#    Note: May not include all results returned by search API
#
# 3. SearchInvocations in Message Metadata (QUERY INFO ONLY):
#    Path: messages[] → searchInvocations → queries[]
#    Contains: Query text and domain, but NOT the actual results
#
# For complete result extraction with reference_ids for CiteDCG mapping,
# ALWAYS use EvaluationData.turnData.orchestrationIterations path.

import csv
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import fire

# Configure logging only if not already configured
if not logging.getLogger().hasHandlers():
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
        """Initialize with thread-safe result collection and file access locks."""
        self.results_lock = Lock()
        # Dictionary to track file locks - one lock per unique file path
        self._file_locks = {}
        self._file_locks_mutex = Lock()  # Protects the file_locks dictionary itself

    def _get_file_lock(self, file_path: str) -> Lock:
        """
        Get or create a lock for a specific file path.

        This ensures that each unique file has its own lock, preventing race conditions
        when multiple threads try to access the same file simultaneously.

        Args:
            file_path: Path to the file

        Returns:
            Lock object specific to this file path
        """
        # Normalize the path to ensure consistent locking
        normalized_path = os.path.normpath(file_path)

        with self._file_locks_mutex:
            if normalized_path not in self._file_locks:
                self._file_locks[normalized_path] = Lock()
            return self._file_locks[normalized_path]

    def _read_json_file_safely(self, file_path: str) -> Dict:
        """
        Safely read a JSON file with proper locking to prevent race conditions.

        Args:
            file_path: Path to the JSON file to read

        Returns:
            Parsed JSON data as dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            IOError: For other file access issues
        """
        file_lock = self._get_file_lock(file_path)

        with file_lock:
            # Set current file being processed for debugging
            self._current_file_being_processed = Path(file_path).name

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            finally:
                # Clean up the current file tracker
                if hasattr(self, "_current_file_being_processed"):
                    delattr(self, "_current_file_being_processed")

    def _process_seval_file(self, file_path: str) -> Dict:
        """
        Process a SEVAL file with proper file locking to prevent race conditions.

        Args:
            file_path: Path to the SEVAL JSON file

        Returns:
            Dict containing processed search information
        """
        try:
            # Read the file safely with proper locking
            data = self._read_json_file_safely(file_path)

            # Extract search information, passing the file path for better error reporting
            return self._extract_search_information(data, file_path)

        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return self._get_empty_search_info()

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in file {file_path}: {e}")
            return self._get_empty_search_info()

        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            return self._get_empty_search_info()

    def _extract_search_information(
        self, seval_data: Dict, file_path: Optional[str] = None
    ) -> Dict:
        """
        Extract comprehensive search information from SEVAL JSON structure.

        This method parses the message flow to extract:
        - User query (first message with author="user")
        - Search results (InternalSearchResult messages)
        - Final response (bot messages without messageType)
        - File access info (InternalStorageMetaData messages)
        - Error messages (messages with "no content returned")

        Args:
            seval_data: Parsed SEVAL JSON data
            file_path: Optional file path for better error reporting

        Returns:
            Dict containing structured search information
        """
        # Handle two different JSON structure patterns
        try:
            # Pattern 1: Standard conversation with messages
            messages = seval_data["requests"][0]["response_body"]["messages"]
            return self._extract_from_messages(messages)
        except (KeyError, IndexError):
            # Pattern 2: Error result structure (failed conversations)
            try:
                result = seval_data["requests"][0]["response_body"]["result"]
                if isinstance(result, dict) and (
                    "error" in result or "exception" in result
                ):
                    # logger.info(
                    #     "Processing error conversation (no search activity performed)"
                    # )
                    return self._extract_from_error_result(result)
            except (KeyError, IndexError):
                pass

            # Neither pattern matched - provide detailed diagnostic information
            # Get the filename for debugging - use provided file_path or fall back to other methods
            filename = "unknown_file"
            if file_path:
                filename = Path(file_path).name
            else:
                try:
                    if hasattr(self, "_current_file_being_processed"):
                        filename = getattr(
                            self, "_current_file_being_processed", "unknown_file"
                        )
                    elif (
                        "conversation_id" in seval_data
                        and seval_data["conversation_id"]
                    ):
                        filename = f"conversation_{seval_data['conversation_id']}"
                except:
                    pass

            # Create detailed diagnostic message about the actual structure found
            diagnostic_info = self._get_structure_diagnostic(
                seval_data, filename, file_path or filename
            )

            logger.warning(
                f"Could not extract messages from SEVAL data structure.\nFile: {filename}\nReason: {diagnostic_info}"
            )

            return self._get_empty_search_info()

    def _get_structure_diagnostic(
        self, seval_data: Dict, filename: str, full_file_path: Optional[str] = None
    ) -> str:
        """
        Generate detailed diagnostic information about the structure of a SEVAL file
        that doesn't match expected patterns.

        Args:
            seval_data: The parsed JSON data from the file
            filename: The filename for context (short name)
            full_file_path: The full file path for detailed error reporting

        Returns:
            String with detailed diagnostic information
        """
        try:
            # If filename is still unknown, try to get it from the current file being processed
            if filename == "unknown_file" and hasattr(
                self, "_current_file_being_processed"
            ):
                filename = getattr(
                    self, "_current_file_being_processed", "unknown_file"
                )

            # Check if data is a dictionary
            if not isinstance(seval_data, dict):
                return f"Root structure is {type(seval_data).__name__}, expected dict"

            top_keys = list(seval_data.keys())

            # Check for requests key
            if "requests" not in seval_data:
                return f"Missing 'requests' key. Found keys: {top_keys}"

            requests = seval_data["requests"]
            if not requests:
                return f"'requests' is empty. Top-level keys: {top_keys}"

            if not isinstance(requests, list):
                return f"'requests' is {type(requests).__name__}, expected list. Top-level keys: {top_keys}"

            if len(requests) == 0:
                return f"'requests' list is empty. Top-level keys: {top_keys}"

            # Check first request
            first_req = requests[0]
            if not isinstance(first_req, dict):
                return f"requests[0] is {type(first_req).__name__}, expected dict"

            req_keys = list(first_req.keys())

            # Check for response_body
            if "response_body" not in first_req:
                return f"Missing 'response_body' in requests[0]. Found keys: {req_keys}"

            response_body = first_req["response_body"]
            if not isinstance(response_body, dict):
                return f"response_body is {type(response_body).__name__}, expected dict"

            rb_keys = list(response_body.keys())

            # Check for expected content
            has_messages = "messages" in response_body
            has_result = "result" in response_body

            if not has_messages and not has_result:
                # This is the specific case we want to track - print to console
                display_path = full_file_path if full_file_path else filename
                print(f"🚨 PROBLEMATIC FILE DETECTED: {filename}")
                print(f"   File path: {display_path}")
                print(f"   Issue: response_body missing both 'messages' and 'result'")
                print(f"   Found keys in response_body: {rb_keys}")
                print(f"   This file has an error-only response structure")
                print("-" * 60)

                return f"response_body missing both 'messages' and 'result'. Found keys: {rb_keys}"

            # If we get here, it should have worked - likely a race condition
            if has_messages:
                messages = response_body["messages"]
                if not isinstance(messages, list):
                    return f"'messages' is {type(messages).__name__}, expected list. response_body keys: {rb_keys}"
                return f"Structure appears correct with {len(messages)} messages - possible race condition during file access"

            elif has_result:
                result = response_body["result"]
                result_type = type(result).__name__
                if isinstance(result, dict):
                    result_keys = list(result.keys())
                    return f"Structure has 'result' ({result_type}) with keys {result_keys} - should match Pattern 2 but failed parsing"
                else:
                    return f"Structure has 'result' of type {result_type} - unexpected result type"

            # Fallback case (should not reach here)
            return f"Structure analysis completed but no specific issue identified. Keys: {rb_keys}"

        except Exception as e:
            return f"Error analyzing structure: {str(e)}"

    def _extract_from_messages(self, messages: List[Dict]) -> Dict:
        """Extract search information from messages array (Pattern 1)."""
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
            message_type = message.get("messageType", "")

            try:
                # Extract user query (first message with author="user")
                if author == "user" and not search_info["user_query"]:
                    search_info["user_query"] = text

                # Extract search progress messages
                elif message_type == "Progress" and "search" in text.lower():
                    if not search_info["search_query"]:
                        search_info["search_query"] = text

                # Check for "No content returned" errors in any message
                if "no content returned" in text.lower():
                    search_info["error_messages"].append(f"Message {i}: {text}")
                    search_info["search_success"] = False

                # Extract search results from InternalSearchResult messages
                elif message_type == "InternalSearchResult" and text.startswith(
                    '{"results":'
                ):
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
                                    "file_name": result_info.get("fileName", ""),
                                    "author": result_info.get("author", ""),
                                    "file_type": result_info.get("fileType", ""),
                                }
                            )
                    except json.JSONDecodeError:
                        search_info["error_messages"].append(
                            f"Message {i}: JSON parse error in InternalSearchResult"
                        )

                # Extract final response (bot message without messageType, substantial content)
                elif (
                    author == "bot"
                    and not message_type
                    and len(text) > 100
                    and not search_info["final_response"]
                ):
                    search_info["final_response"] = text
                    search_info["response_length"] = len(text)

                # Extract file access URLs from InternalStorageMetaData messages
                elif message_type == "InternalStorageMetaData" and text.startswith(
                    '{"storageResults":'
                ):
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
                            f"Message {i}: JSON parse error in InternalStorageMetaData"
                        )

            except Exception as e:
                logger.warning(f"Error processing message {i}: {e}")
                continue

        return search_info

    def _extract_from_error_result(self, result: Dict) -> Dict:
        """Extract search information from error result structure (Pattern 2)."""
        # Error conversations have no search activity
        error_message = result.get("message", "")
        error_details = result.get("error", "")

        return {
            "user_query": "",
            "search_query": "",
            "results_found": [],
            "files_accessed": [],
            "final_response": error_message,
            "search_success": False,
            "error_messages": (
                [f"Conversation failed: {error_details[:200]}..."]
                if error_details
                else ["Conversation failed"]
            ),
            "response_length": len(error_message),
        }

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

    def _extract_requested_files_emails_from_query(self, query_text: str) -> List[str]:
        """
        Extract specific files and emails mentioned in the query text.

        This function identifies explicit file references (with extensions) and email subjects
        that the user is specifically requesting to search for.

        Args:
            query_text: The user's query text

        Returns:
            List of requested files/emails found in the query
        """
        import re

        requested_items = []

        # Pattern 1: Files with extensions (.xlsx, .docx, .pdf, .pptx, etc.)
        file_pattern = r"\b([A-Za-z0-9_\-]+\.[A-Za-z]{2,5})\b"
        file_matches = re.findall(file_pattern, query_text)
        requested_items.extend(file_matches)

        # Pattern 2: Email subjects in quotes
        email_subject_pattern = r'"([^"]+)"'
        email_matches = re.findall(email_subject_pattern, query_text)
        requested_items.extend(email_matches)

        # Pattern 3: Email folders/categories (Sent Items, AutomatedReports, etc.)
        email_folder_pattern = r"\b(Sent Items|AutomatedReports|Inbox|Drafts)\b"
        folder_matches = re.findall(email_folder_pattern, query_text, re.IGNORECASE)
        # Don't add folders as requested items, they are search locations

        # Pattern 4: Specific document types mentioned with context
        # e.g., "whitepaper", "policy emails", "alert rules v2 emails"
        contextual_pattern = (
            r"\b(\w+\s+emails?|\w+\s+whitepaper|\w+\s+rules?\s+v?\d*|\w+\s+policy)\b"
        )
        contextual_matches = re.findall(contextual_pattern, query_text, re.IGNORECASE)
        requested_items.extend(contextual_matches)

        # Clean up and normalize
        cleaned_items = []
        for item in requested_items:
            item = item.strip()
            if len(item) > 2:  # Skip very short matches
                cleaned_items.append(item)

        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in cleaned_items:
            if item.lower() not in seen:
                seen.add(item.lower())
                unique_items.append(item)

        return unique_items

    def _determine_targeted_search_success(
        self, search_info: Dict, requested_items: List[str]
    ) -> bool:
        """
        Determine if the search successfully found/accessed the specifically requested files/emails.

        This function focuses on whether the requested items were found, ignoring failures
        in ancillary searches (like searching emails when only files were requested).

        Args:
            search_info: Search information extracted from JSON structure
            requested_items: List of specific files/emails mentioned in the query

        Returns:
            True if the search found/accessed the specifically requested items, False otherwise
        """
        if not requested_items:
            # If no specific items were requested, fall back to original search_success logic
            return search_info["search_success"]

        # Extract filenames from search results
        found_files = set()
        accessed_files = set()

        # From results_found (search results)
        for result in search_info["results_found"]:
            title = result.get("title", "").lower()
            filename = result.get("file_name", "").lower()
            found_files.add(title)
            found_files.add(filename)

        # From files_accessed (actual file access)
        for file_info in search_info["files_accessed"]:
            filename = file_info.get("filename", "").lower()
            accessed_files.add(filename)

        # Check if each requested item was found or accessed
        found_requested_items = 0
        for requested_item in requested_items:
            requested_lower = requested_item.lower()

            # Check for exact matches or partial matches
            found_in_results = any(
                requested_lower in found_file
                or self._normalize_filename(requested_lower)
                in self._normalize_filename(found_file)
                for found_file in found_files
                if found_file
            )

            found_in_accessed = any(
                requested_lower in accessed_file
                or self._normalize_filename(requested_lower)
                in self._normalize_filename(accessed_file)
                for accessed_file in accessed_files
                if accessed_file
            )

            if found_in_results or found_in_accessed:
                found_requested_items += 1

        # Consider search successful if we found at least 50% of requested items
        # This handles cases where some requested files might not exist or have slightly different names
        success_threshold = max(
            1, len(requested_items) // 2
        )  # At least 1, or half of requested items

        return found_requested_items >= success_threshold

    def _normalize_filename(self, filename: str) -> str:
        """
        Normalize filename for comparison by removing extensions and special characters.

        Args:
            filename: Original filename

        Returns:
            Normalized filename for comparison
        """
        if not filename:
            return ""

        # Remove file extension
        name_without_ext = filename.split(".")[0]

        # Remove special characters and normalize spacing
        import re

        normalized = re.sub(r"[_\-\s]+", "", name_without_ext.lower())

        return normalized

    def _determine_access_level(self, search_info: Dict) -> str:
        """
        Determine the level of content access achieved.

        Args:
            search_info: Search information extracted from JSON structure

        Returns:
            Access level: 'no_access', 'partial_access', 'full_access', or 'unknown_access'
        """
        detailed_info = self._get_detailed_access_breakdown(search_info)
        return detailed_info["access_level"]

    def _get_detailed_access_breakdown(self, search_info: Dict) -> Dict:
        """
        Get detailed breakdown of access level determination with specific reasons.

        Returns:
            Dict with access_level, detailed_category, reason, and additional metrics
        """
        # Check for explicit search failure indicators
        has_no_content_errors = any(
            "no content returned" in msg.lower()
            for msg in search_info["error_messages"]
        )

        results_count = len(search_info["results_found"])
        files_accessed_count = len(search_info["files_accessed"])
        response_length = search_info["response_length"]
        has_error_messages = len(search_info["error_messages"]) > 0
        search_success = search_info["search_success"]

        # Detailed categorization for NO ACCESS
        if not search_info["results_found"]:
            # Check if this is a failed conversation (error result pattern)
            if any(
                "Conversation failed" in msg for msg in search_info["error_messages"]
            ):
                return {
                    "access_level": "no_access",
                    "detailed_category": "conversation_failed",
                    "reason": "Conversation failed before search could be performed",
                    "results_count": 0,
                    "files_accessed_count": 0,
                    "response_length": response_length,
                    "has_errors": has_error_messages,
                    "search_attempted": False,
                }
            elif has_no_content_errors:
                return {
                    "access_level": "no_access",
                    "detailed_category": "search_no_content",
                    "reason": "Search was performed but returned no content",
                    "results_count": 0,
                    "files_accessed_count": files_accessed_count,
                    "response_length": response_length,
                    "has_errors": has_error_messages,
                    "search_attempted": True,
                }
            elif not search_success:
                return {
                    "access_level": "no_access",
                    "detailed_category": "search_failed",
                    "reason": "Search operation failed or encountered errors",
                    "results_count": 0,
                    "files_accessed_count": files_accessed_count,
                    "response_length": response_length,
                    "has_errors": has_error_messages,
                    "search_attempted": True,
                }
            else:
                return {
                    "access_level": "no_access",
                    "detailed_category": "no_results_found",
                    "reason": "Search completed but found no matching results",
                    "results_count": 0,
                    "files_accessed_count": files_accessed_count,
                    "response_length": response_length,
                    "has_errors": has_error_messages,
                    "search_attempted": True,
                }

        # If we have results, check for FULL ACCESS
        if search_info["results_found"] and response_length > 300:
            # Check if response contains specific content from the found files
            response = search_info["final_response"].lower()

            # Strong indicators of full access to file content
            full_access_indicators = [
                "shows",
                "contains",
                "includes",
                "according to",
                "based on",
                "document indicates",
                "file defines",
                "specification",
                "thresholds",
                "metrics",
                "panel",
                "dashboard",
                "alert",
                "warning",
                "critical",
                "configuration",
                "guidelines",
            ]

            # Check for specific content details in snippets
            has_detailed_content = False
            detailed_snippets_count = 0
            for result in search_info["results_found"]:
                snippet = result.get("snippet", "").lower()
                if len(snippet) > 200 and any(
                    indicator in snippet
                    for indicator in [
                        "threshold",
                        "metric",
                        "specification",
                        "configuration",
                    ]
                ):
                    has_detailed_content = True
                    detailed_snippets_count += 1

            # Check for substantial file content in response
            has_substantial_response = any(
                indicator in response for indicator in full_access_indicators
            )

            # Full access if we have detailed content OR substantial response
            if has_detailed_content or has_substantial_response:
                return {
                    "access_level": "full_access",
                    "detailed_category": "substantial_content_access",
                    "reason": f"Found {results_count} results with substantial content access",
                    "results_count": results_count,
                    "files_accessed_count": files_accessed_count,
                    "response_length": response_length,
                    "has_errors": has_error_messages,
                    "search_attempted": True,
                    "detailed_snippets_count": detailed_snippets_count,
                    "has_substantial_response": has_substantial_response,
                }

        # PARTIAL ACCESS cases - we have results but limited/mixed success
        if search_info["results_found"]:
            if has_no_content_errors:
                return {
                    "access_level": "partial_access",
                    "detailed_category": "mixed_success_with_errors",
                    "reason": f"Found {results_count} results but some content access failed",
                    "results_count": results_count,
                    "files_accessed_count": files_accessed_count,
                    "response_length": response_length,
                    "has_errors": has_error_messages,
                    "search_attempted": True,
                    "has_no_content_errors": True,
                }
            elif response_length <= 300:
                return {
                    "access_level": "partial_access",
                    "detailed_category": "limited_content_response",
                    "reason": f"Found {results_count} results but limited response content ({response_length} chars)",
                    "results_count": results_count,
                    "files_accessed_count": files_accessed_count,
                    "response_length": response_length,
                    "has_errors": has_error_messages,
                    "search_attempted": True,
                }
            else:
                return {
                    "access_level": "partial_access",
                    "detailed_category": "results_without_detailed_content",
                    "reason": f"Found {results_count} results but without detailed file content indicators",
                    "results_count": results_count,
                    "files_accessed_count": files_accessed_count,
                    "response_length": response_length,
                    "has_errors": has_error_messages,
                    "search_attempted": True,
                }

        # Fallback cases
        if has_no_content_errors and not search_success:
            return {
                "access_level": "no_access",
                "detailed_category": "explicit_search_failure",
                "reason": "Search explicitly failed with no content returned",
                "results_count": results_count,
                "files_accessed_count": files_accessed_count,
                "response_length": response_length,
                "has_errors": has_error_messages,
                "search_attempted": True,
            }

        return {
            "access_level": "unknown_access",
            "detailed_category": "unclassified",
            "reason": "Could not determine access level from available information",
            "results_count": results_count,
            "files_accessed_count": files_accessed_count,
            "response_length": response_length,
            "has_errors": has_error_messages,
            "search_attempted": True,
        }

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

    def _extract_mentioned_files(self, query_text: str) -> list:
        """Extract specific filenames and emails mentioned in user queries.

        This function detects:
        - Filenames with common extensions (.xlsx, .docx, .pdf, .txt, .ppt, .xls, etc.)
        - Email addresses
        - Document references (even without extensions)

        Args:
            query_text: The user query text to analyze

        Returns:
            List of mentioned files/emails found in the query
        """
        if not query_text:
            return []

        import re

        mentioned_files = set()

        # Pattern 1: Files with common extensions
        file_extensions = [
            "xlsx",
            "docx",
            "pdf",
            "txt",
            "ppt",
            "pptx",
            "xls",
            "csv",
            "doc",
            "rtf",
            "odt",
            "ods",
            "odp",
            "json",
            "xml",
            "html",
            "log",
            "md",
            "zip",
            "rar",
            "7z",
            "tar",
            "gz",
        ]

        # Look for filenames with extensions (case-insensitive)
        for ext in file_extensions:
            pattern = r"\b([A-Za-z0-9_\-\.]+\." + ext + r")\b"
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            mentioned_files.update(matches)

        # Pattern 2: Email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        email_matches = re.findall(email_pattern, query_text)
        mentioned_files.update(email_matches)

        # Pattern 3: Common document naming patterns (without extensions)
        # Look for quoted filenames or capitalized multi-word names
        doc_patterns = [
            r'"([A-Za-z0-9_\-\s]+)"',  # Quoted names
            r"\b([A-Z][A-Za-z0-9_]*(?:[A-Z][A-Za-z0-9_]*)+)\b",  # CamelCase names
        ]

        for pattern in doc_patterns:
            matches = re.findall(pattern, query_text)
            # Only add if it looks like a document name (contains underscore or multiple capitals)
            for match in matches:
                if "_" in match or sum(1 for c in match if c.isupper()) >= 2:
                    mentioned_files.add(match)

        return sorted(list(mentioned_files))

    def _check_mentioned_files_access(
        self, mentioned_files: list, found_files: list, accessed_files: list
    ) -> tuple:
        """Check which mentioned files were found and accessed by a system.

        Args:
            mentioned_files: List of files mentioned in the user query
            found_files: List of file titles found by the system
            accessed_files: List of filenames accessed by the system

        Returns:
            Tuple of (found_mentioned, accessed_mentioned) lists
        """
        if not mentioned_files:
            return [], []

        found_mentioned = []
        accessed_mentioned = []

        # Convert to lowercase for case-insensitive matching
        mentioned_lower = [f.lower() for f in mentioned_files]
        found_lower = [f.lower() for f in found_files]
        accessed_lower = [f.lower() for f in accessed_files]

        for mentioned_file in mentioned_files:
            mentioned_lower_file = mentioned_file.lower()

            # Check if mentioned file was found (partial matching)
            for found_file in found_files:
                if (
                    mentioned_lower_file in found_file.lower()
                    or found_file.lower() in mentioned_lower_file
                    or
                    # Check without extension
                    mentioned_lower_file.split(".")[0] in found_file.lower()
                ):
                    found_mentioned.append(mentioned_file)
                    break

            # Check if mentioned file was accessed (partial matching)
            for accessed_file in accessed_files:
                if (
                    mentioned_lower_file in accessed_file.lower()
                    or accessed_file.lower() in mentioned_lower_file
                    or
                    # Check without extension
                    mentioned_lower_file.split(".")[0] in accessed_file.lower()
                ):
                    accessed_mentioned.append(mentioned_file)
                    break

        return found_mentioned, accessed_mentioned

    def _evaluate_fair_comparison_candidate(
        self,
        mentioned_files: list,
        control_info: dict,
        experiment_info: dict,
        similarity_data: dict,
        control_found_mentioned: list,
        experiment_found_mentioned: list,
        control_accessed_mentioned: list,
        experiment_accessed_mentioned: list,
        requested_items: Optional[List[str]] = None,
    ) -> bool:
        """
        Evaluate if a query is a fair comparison candidate based on targeted search criteria.

        Updated Criteria:
        1. Both systems must have targeted search success (found the specifically requested items)
        2. If both control and experiment have full access to requested items, they are fair comparison candidates
        3. If both found the requested items but either has full access, they are fair comparison candidates
        4. For queries without specific items, fall back to original logic

        Args:
            mentioned_files: List of files specifically mentioned in the query
            control_info: Control system search information
            experiment_info: Experiment system search information
            similarity_data: File overlap similarity data
            control_found_mentioned: Mentioned files found by control
            experiment_found_mentioned: Mentioned files found by experiment
            control_accessed_mentioned: Mentioned files accessed by control
            experiment_accessed_mentioned: Mentioned files accessed by experiment
            requested_items: Specific files/emails requested in the query

        Returns:
            Boolean indicating if this is a fair comparison candidate
        """
        # Use requested_items if provided, otherwise fall back to mentioned_files
        target_items = requested_items if requested_items else mentioned_files

        # Basic requirement: both systems must have targeted search success
        control_search_success = self._determine_targeted_search_success(
            control_info, target_items
        )
        experiment_search_success = self._determine_targeted_search_success(
            experiment_info, target_items
        )

        if not (control_search_success and experiment_search_success):
            return False

        # Enhanced fair comparison logic based on your requirements
        if target_items:
            # Case 1: Query has specific requested items
            control_has_full_access = (
                similarity_data["control_access_level"] == "full_access"
            )
            experiment_has_full_access = (
                similarity_data["experiment_access_level"] == "full_access"
            )

            # Both have full access to requested items
            if control_has_full_access and experiment_has_full_access:
                return True

            # Both found the items but either has full access
            both_found_items = (
                len(control_found_mentioned) > 0 and len(experiment_found_mentioned) > 0
            )
            either_has_full_access = (
                control_has_full_access or experiment_has_full_access
            )

            if both_found_items and either_has_full_access:
                return True

            return False

        else:
            # Case 2: No specific items requested - use original overlap logic
            # Both systems must find at least one file/email
            control_found_any = len(control_info["results_found"]) > 0
            experiment_found_any = len(experiment_info["results_found"]) > 0

            if not (control_found_any and experiment_found_any):
                return False

            # Check overlap criteria based on the system with fewer found files
            control_files = set(f.lower() for f in similarity_data["control_files"])
            experiment_files = set(
                f.lower() for f in similarity_data["experiment_files"]
            )

            if not control_files or not experiment_files:
                return False

            # Calculate overlap based on the system with fewer files
            if len(control_files) <= len(experiment_files):
                # Control has fewer files - check if control files are contained/overlap in experiment
                intersection = control_files.intersection(experiment_files)
                overlap_percentage = len(intersection) / len(control_files) * 100
            else:
                # Experiment has fewer files - check if experiment files are contained/overlap in control
                intersection = control_files.intersection(experiment_files)
                overlap_percentage = len(intersection) / len(experiment_files) * 100

            # Files must be fully contained (100%) or overlap by >=50%
            return overlap_percentage >= 50.0

    def _search_in_file(
        self, json_file: Path, query_text: str
    ) -> Optional[Dict[str, Any]]:
        """Search within a single JSON file."""
        try:
            data = self._read_json_file_safely(str(json_file))

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
            data = self._read_json_file_safely(str(json_file))

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
        import concurrent.futures
        import json
        import os
        import re
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
                data = self._read_json_file_safely(file_path)

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

    def extract_model_usage_statistics(
        self,
        input_dir: str,
        output_file: str,
        exp: str = "both",
        threads: int = 8,
    ):
        """
        Extract detailed model usage statistics from SEVAL JSON files.

        This function extracts model information for EACH occurrence (each iteration and modelAction)
        from the path: evaluationData -> turnData -> orchestrationIterations -> modelActions -> model

        Unlike extract_model_statistics() which finds only the first reasoning model per file,
        this function collects ALL model invocations to provide complete usage statistics.

        Args:
            input_dir: Directory containing SEVAL JSON files to analyze
            output_file: Path where TSV results file will be saved
            exp: Filter by experiment type ('control', 'experiment', or 'both') (default: 'both')
            threads: Number of threads for parallel processing (default: 8)

        Output:
            - Per-file detailed model usage (iterations, model calls, latencies)
            - Aggregate statistics by experiment group
            - Model distribution across all iterations
        """
        import concurrent.futures
        from collections import defaultdict
        from datetime import datetime
        from pathlib import Path

        logger.info(f"Extracting detailed model usage statistics from: {input_dir}")
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

        logger.info(f"Found {len(json_files)} JSON files total")

        if not json_files:
            logger.warning("No JSON files found in the directory.")
            return

        def extract_all_model_usages(file_path):
            """Extract ALL model usage information from a single file."""
            try:
                data = self._read_json_file_safely(file_path)
                
                # Determine experiment type from filename
                filename = os.path.basename(file_path)
                if filename.startswith("control_"):
                    exp_type = "control"
                elif filename.startswith("experiment_"):
                    exp_type = "experiment"
                else:
                    exp_type = "unknown"

                # Extract utterance
                utterance = data.get("query", {}).get("id", "No utterance found")

                # Check conversation success
                conversation_success = data.get("conversation_success", True)
                is_success = data.get("is_success", True)

                model_usages = []
                total_iterations = 0
                total_model_actions = 0

                # Navigate to orchestrationIterations
                try:
                    requests = data.get("requests", [])
                    if requests:
                        response_body = requests[0].get("response_body", {})
                        messages = response_body.get("messages", [])
                        
                        # Find message with evaluationData (usually the last one)
                        for msg in messages:
                            eval_data = msg.get("evaluationData", {})
                            turn_data = eval_data.get("turnData", [])
                            
                            for turn_idx, turn in enumerate(turn_data):
                                orch_iterations = turn.get("orchestrationIterations", [])
                                
                                for orch_iter in orch_iterations:
                                    iteration_num = orch_iter.get("iteration", 0)
                                    total_iterations += 1
                                    
                                    iter_start = orch_iter.get("startTime", "")
                                    iter_end = orch_iter.get("endTime", "")
                                    
                                    model_actions = orch_iter.get("modelActions", [])
                                    
                                    for action_idx, action in enumerate(model_actions):
                                        total_model_actions += 1
                                        
                                        model_name = action.get("model", "").strip()
                                        tag = action.get("tag", "")
                                        model_tags = action.get("modelTags", [])
                                        model_api = action.get("modelApi", "")
                                        latency_ms = action.get("latencyMilliseconds", 0)
                                        action_start = action.get("startTime", "")
                                        action_end = action.get("endTime", "")
                                        
                                        # Extract token usage from additionalMetrics if available
                                        additional_metrics = action.get("additionalMetrics", {})
                                        metrics = additional_metrics.get("metrics", {})
                                        input_tokens = metrics.get("DeepLeoInputPrompt", 0)
                                        output_tokens = metrics.get("DeepLeoOutputStream", 0)
                                        
                                        # Check for tool invocations
                                        tool_invocations = action.get("toolInvocations", [])
                                        num_tool_calls = len(tool_invocations) if tool_invocations else 0
                                        
                                        model_usages.append({
                                            "iteration": iteration_num,
                                            "action_index": action_idx,
                                            "model": model_name,
                                            "tag": tag,
                                            "model_api": model_api,
                                            "latency_ms": latency_ms,
                                            "input_tokens": input_tokens,
                                            "output_tokens": output_tokens,
                                            "num_tool_calls": num_tool_calls,
                                            "start_time": action_start,
                                            "end_time": action_end,
                                        })
                                        
                except Exception as e:
                    logger.debug(f"Error extracting model usages from {file_path}: {e}")

                return {
                    "file": file_path,
                    "filename": filename,
                    "exp_type": exp_type,
                    "utterance": utterance,
                    "conversation_success": conversation_success and is_success,
                    "total_iterations": total_iterations,
                    "total_model_actions": total_model_actions,
                    "model_usages": model_usages,
                    "error": None,
                }

            except Exception as e:
                return {
                    "file": file_path,
                    "filename": os.path.basename(file_path),
                    "exp_type": "unknown",
                    "utterance": "Error",
                    "conversation_success": False,
                    "total_iterations": 0,
                    "total_model_actions": 0,
                    "model_usages": [],
                    "error": str(e),
                }

        # Process files with threading
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_file = {
                executor.submit(extract_all_model_usages, file_path): file_path
                for file_path in json_files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.error(f"Error processing {file_path}: {e}")

        # Aggregate statistics by experiment type
        stats_by_exp = defaultdict(lambda: {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "files_with_model_actions": 0,
            "files_without_model_actions": 0,
            "total_iterations": 0,
            "total_model_actions": 0,
            "model_counts": defaultdict(int),
            "tag_counts": defaultdict(int),
            "total_latency_ms": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tool_calls": 0,
            "latencies": [],
        })

        # Process results
        for result in all_results:
            exp_type = result["exp_type"]
            stats = stats_by_exp[exp_type]
            
            stats["total_files"] += 1
            
            if result["conversation_success"]:
                stats["successful_files"] += 1
            else:
                stats["failed_files"] += 1
            
            if result["total_model_actions"] > 0:
                stats["files_with_model_actions"] += 1
            else:
                stats["files_without_model_actions"] += 1
            
            stats["total_iterations"] += result["total_iterations"]
            stats["total_model_actions"] += result["total_model_actions"]
            
            for usage in result["model_usages"]:
                model = usage["model"]
                tag = usage["tag"]
                
                if model:
                    stats["model_counts"][model] += 1
                if tag:
                    stats["tag_counts"][tag] += 1
                
                stats["total_latency_ms"] += usage["latency_ms"]
                stats["total_input_tokens"] += usage["input_tokens"]
                stats["total_output_tokens"] += usage["output_tokens"]
                stats["total_tool_calls"] += usage["num_tool_calls"]
                
                if usage["latency_ms"] > 0:
                    stats["latencies"].append(usage["latency_ms"])

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

        # Write results to TSV file
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            f.write("# SEVAL Model Usage Statistics Report\n")
            f.write(f"# Generated at: {datetime.now()}\n")
            f.write(f"# Input directory: {input_dir}\n")
            f.write(f"# Experiment filter: {exp}\n")
            f.write(f"# Total files processed: {len(all_results)}\n")
            f.write("\n")

            # Write aggregate statistics for each experiment type
            for exp_type in sorted(stats_by_exp.keys()):
                stats = stats_by_exp[exp_type]
                
                f.write(f"# ========== {exp_type.upper()} GROUP STATISTICS ==========\n")
                f.write(f"# Total files: {stats['total_files']}\n")
                f.write(f"# Successful conversations: {stats['successful_files']}\n")
                f.write(f"# Failed conversations: {stats['failed_files']}\n")
                f.write(f"# Files with model actions: {stats['files_with_model_actions']}\n")
                f.write(f"# Files without model actions: {stats['files_without_model_actions']}\n")
                f.write(f"# Total orchestration iterations: {stats['total_iterations']}\n")
                f.write(f"# Total model actions: {stats['total_model_actions']}\n")
                
                if stats['total_model_actions'] > 0:
                    avg_actions_per_file = stats['total_model_actions'] / stats['total_files']
                    f.write(f"# Avg model actions per file: {avg_actions_per_file:.2f}\n")
                
                # Model distribution
                f.write(f"#\n# Model Distribution:\n")
                for model, count in sorted(stats["model_counts"].items(), key=lambda x: -x[1]):
                    pct = (count / stats['total_model_actions'] * 100) if stats['total_model_actions'] > 0 else 0
                    f.write(f"#   {model}: {count} ({pct:.1f}%)\n")
                
                # Tag distribution
                f.write(f"#\n# Tag Distribution:\n")
                for tag, count in sorted(stats["tag_counts"].items(), key=lambda x: -x[1]):
                    pct = (count / stats['total_model_actions'] * 100) if stats['total_model_actions'] > 0 else 0
                    f.write(f"#   {tag}: {count} ({pct:.1f}%)\n")
                
                # Latency statistics
                if stats["latencies"]:
                    avg_latency = sum(stats["latencies"]) / len(stats["latencies"])
                    min_latency = min(stats["latencies"])
                    max_latency = max(stats["latencies"])
                    sorted_latencies = sorted(stats["latencies"])
                    p50 = sorted_latencies[len(sorted_latencies) // 2]
                    p90_idx = int(len(sorted_latencies) * 0.9)
                    p90 = sorted_latencies[p90_idx] if p90_idx < len(sorted_latencies) else max_latency
                    
                    f.write(f"#\n# Latency Statistics (ms):\n")
                    f.write(f"#   Avg: {avg_latency:.0f}, Min: {min_latency}, Max: {max_latency}\n")
                    f.write(f"#   P50: {p50}, P90: {p90}\n")
                
                # Token statistics
                f.write(f"#\n# Token Usage:\n")
                f.write(f"#   Total input tokens: {stats['total_input_tokens']:,}\n")
                f.write(f"#   Total output tokens: {stats['total_output_tokens']:,}\n")
                f.write(f"#   Total tool calls: {stats['total_tool_calls']:,}\n")
                
                if stats['total_model_actions'] > 0:
                    avg_input = stats['total_input_tokens'] / stats['total_model_actions']
                    avg_output = stats['total_output_tokens'] / stats['total_model_actions']
                    f.write(f"#   Avg input tokens per action: {avg_input:.0f}\n")
                    f.write(f"#   Avg output tokens per action: {avg_output:.0f}\n")
                
                f.write("\n")

            # Write detailed per-file results header
            f.write("# ========== DETAILED PER-FILE RESULTS ==========\n")
            f.write("exp_type\tutterance\tfilename\tconversation_success\ttotal_iterations\ttotal_model_actions\t")
            f.write("iteration\taction_index\tmodel\ttag\tmodel_api\tlatency_ms\tinput_tokens\toutput_tokens\tnum_tool_calls\n")

            # Write detailed results for each file
            for result in sorted(all_results, key=lambda x: (x["exp_type"], x["filename"])):
                exp_type = result["exp_type"]
                utterance = result["utterance"].replace("\t", " ").replace("\n", " ").strip()
                filename = result["filename"]
                success = result["conversation_success"]
                total_iters = result["total_iterations"]
                total_actions = result["total_model_actions"]
                
                if result["model_usages"]:
                    # Write one row per model action
                    for usage in result["model_usages"]:
                        f.write(f"{exp_type}\t{utterance}\t{filename}\t{success}\t{total_iters}\t{total_actions}\t")
                        f.write(f"{usage['iteration']}\t{usage['action_index']}\t{usage['model']}\t{usage['tag']}\t")
                        f.write(f"{usage['model_api']}\t{usage['latency_ms']}\t{usage['input_tokens']}\t")
                        f.write(f"{usage['output_tokens']}\t{usage['num_tool_calls']}\n")
                else:
                    # Write a single row for files with no model actions
                    f.write(f"{exp_type}\t{utterance}\t{filename}\t{success}\t{total_iters}\t{total_actions}\t")
                    f.write("0\t0\t\t\t\t0\t0\t0\t0\n")

        # Print summary to console
        print(f"\n{'='*60}")
        print("SEVAL Model Usage Statistics Report")
        print(f"{'='*60}")
        print(f"Input directory: {input_dir}")
        print(f"Experiment filter: {exp}")
        print(f"Total files processed: {len(all_results)}")

        for exp_type in sorted(stats_by_exp.keys()):
            stats = stats_by_exp[exp_type]
            print(f"\n--- {exp_type.upper()} GROUP ---")
            print(f"Total files: {stats['total_files']}")
            print(f"Files with model actions: {stats['files_with_model_actions']} ({stats['files_with_model_actions']/stats['total_files']*100:.1f}%)")
            print(f"Total orchestration iterations: {stats['total_iterations']}")
            print(f"Total model actions: {stats['total_model_actions']}")
            
            if stats["model_counts"]:
                print("\nTop models:")
                for model, count in sorted(stats["model_counts"].items(), key=lambda x: -x[1])[:5]:
                    pct = (count / stats['total_model_actions'] * 100) if stats['total_model_actions'] > 0 else 0
                    print(f"  {model}: {count} ({pct:.1f}%)")
            
            if stats["latencies"]:
                avg_latency = sum(stats["latencies"]) / len(stats["latencies"])
                print(f"\nAvg latency: {avg_latency:.0f}ms")

        print(f"\nResults written to: {output_file}")
        logger.info(f"Model usage statistics extraction completed. Results saved to: {output_file}")

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

        print("Starting SEVAL Search Results Analysis...")
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

        # Create results directly from query mappings for JSON-based analysis
        converted_results = []
        for query, file_info in query_mappings.items():
            # Get file paths
            control_file = ""
            experiment_file = ""

            if file_info.get("control"):
                control_files = [f["filename"] for f in file_info["control"]]
                control_file = control_files[0] if control_files else ""

            if file_info.get("experiment"):
                experiment_files = [f["filename"] for f in file_info["experiment"]]
                experiment_file = experiment_files[0] if experiment_files else ""

            converted_result = {
                "query": query,
                "control_file": (
                    os.path.join(search_dir, control_file) if control_file else ""
                ),
                "experiment_file": (
                    os.path.join(search_dir, experiment_file) if experiment_file else ""
                ),
            }
            converted_results.append(converted_result)

        # Sort results by query for consistent output
        converted_results.sort(key=lambda x: x["query"])

        # Create JSON-based TSV output
        self.create_clean_tsv_output(converted_results, output_file)

        # Display JSON-based statistics
        self._display_detailed_access_statistics(converted_results)

        print(f"\n📊 Search results analysis completed!")
        print(f"JSON-based results written to: {output_file}")
        logger.info(
            f"Search results analysis completed. Processed {len(converted_results)} queries."
        )

    def _display_detailed_access_statistics(self, results: List[Dict]) -> None:
        """Display detailed statistics with breakdown of access levels and error details."""
        print(f"\n📊 Detailed Access Level Analysis:")
        print(f"{'='*60}")

        # Analyze all results to get detailed breakdown
        detailed_stats = {
            "total_queries": len(results),
            "access_level_breakdown": {"control": {}, "experiment": {}},
            "detailed_category_breakdown": {"control": {}, "experiment": {}},
            "error_analysis": {"control": {}, "experiment": {}},
        }

        # Process each result to get detailed information
        for result in results:
            for exp_type in ["control", "experiment"]:
                file_path = result.get(f"{exp_type}_file", "")
                if not file_path or not os.path.exists(file_path):
                    # Missing file
                    access_level = "missing_file"
                    detailed_category = "file_not_found"
                else:
                    # Analyze the file with retry logic
                    try:
                        # Use the new retry method
                        search_info = self._process_seval_file(file_path)
                        breakdown = self._get_detailed_access_breakdown(search_info)

                        access_level = breakdown["access_level"]
                        detailed_category = breakdown["detailed_category"]

                        # Track error details
                        if breakdown.get("has_errors", False):
                            reason = breakdown.get("reason", "Unknown error")
                            if reason not in detailed_stats["error_analysis"][exp_type]:
                                detailed_stats["error_analysis"][exp_type][reason] = 0
                            detailed_stats["error_analysis"][exp_type][reason] += 1

                    except Exception as e:
                        access_level = "processing_error"
                        detailed_category = "file_processing_failed"

                        error_msg = f"Processing failed: {str(e)[:50]}..."
                        if error_msg not in detailed_stats["error_analysis"][exp_type]:
                            detailed_stats["error_analysis"][exp_type][error_msg] = 0
                        detailed_stats["error_analysis"][exp_type][error_msg] += 1

                # Update counts
                if (
                    access_level
                    not in detailed_stats["access_level_breakdown"][exp_type]
                ):
                    detailed_stats["access_level_breakdown"][exp_type][access_level] = 0
                detailed_stats["access_level_breakdown"][exp_type][access_level] += 1

                if (
                    detailed_category
                    not in detailed_stats["detailed_category_breakdown"][exp_type]
                ):
                    detailed_stats["detailed_category_breakdown"][exp_type][
                        detailed_category
                    ] = 0
                detailed_stats["detailed_category_breakdown"][exp_type][
                    detailed_category
                ] += 1

        # Display results
        total_queries = detailed_stats["total_queries"]
        print(f"Total queries analyzed: {total_queries}")

        # Display access level breakdown for each experiment type
        for exp_type in ["control", "experiment"]:
            print(f"\n🔐 {exp_type.upper()} ACCESS LEVELS:")
            access_counts = detailed_stats["access_level_breakdown"][exp_type]
            total_files = sum(access_counts.values())

            for access_level, count in sorted(
                access_counts.items(), key=lambda x: -x[1]
            ):
                percentage = (count / total_files) * 100 if total_files > 0 else 0
                print(
                    f"  {access_level.replace('_', ' ')}: {count} ({percentage:.1f}%)"
                )

            # Show detailed breakdown for no_access and partial_access
            print(f"\n📋 {exp_type.upper()} DETAILED BREAKDOWN:")
            detailed_counts = detailed_stats["detailed_category_breakdown"][exp_type]

            # Focus on the categories user is interested in
            focus_categories = {}
            for category, count in detailed_counts.items():
                if any(
                    keyword in category
                    for keyword in [
                        "no_",
                        "partial_",
                        "failed",
                        "error",
                        "conversation",
                    ]
                ):
                    focus_categories[category] = count

            if focus_categories:
                print("  Problem categories:")
                for category, count in sorted(
                    focus_categories.items(), key=lambda x: -x[1]
                ):
                    percentage = (count / total_files) * 100 if total_files > 0 else 0
                    print(
                        f"    {category.replace('_', ' ')}: {count} ({percentage:.1f}%)"
                    )

            # Show error details if any
            error_details = detailed_stats["error_analysis"][exp_type]
            if error_details:
                print(f"\n⚠️  {exp_type.upper()} ERROR DETAILS:")
                for error_msg, count in sorted(
                    error_details.items(), key=lambda x: -x[1]
                )[
                    :10
                ]:  # Top 10 errors
                    percentage = (count / total_files) * 100 if total_files > 0 else 0
                    print(f"    {error_msg}: {count} ({percentage:.1f}%)")

        print(f"\n{'='*60}")

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
            data = self._read_json_file_safely(str(file_path))

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
                "mentioned_files",
                "control_found_mentioned",
                "experiment_found_mentioned",
                "control_accessed_mentioned",
                "experiment_accessed_mentioned",
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
                    "|".join(query_content_analysis.get("mentioned_files", [])),
                    "|".join(query_content_analysis.get("control_found_mentioned", [])),
                    "|".join(
                        query_content_analysis.get("experiment_found_mentioned", [])
                    ),
                    "|".join(
                        query_content_analysis.get("control_accessed_mentioned", [])
                    ),
                    "|".join(
                        query_content_analysis.get("experiment_accessed_mentioned", [])
                    ),
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

    def _analyze_query_content_matching(
        self, result: Dict, requested_items: Optional[List[str]] = None
    ) -> Dict:
        """Analyze content matching for a single query using JSON structure analysis."""

        # Extract requested items from query if not provided
        if requested_items is None:
            query_text = result.get("query", "")
            requested_items = self._extract_requested_files_emails_from_query(
                query_text
            )

        # Extract search information from control and experiment files
        control_info = None
        experiment_info = None

        # Load control file if available
        control_file = result.get("control_file")
        if control_file:
            try:
                control_data = self._read_json_file_safely(control_file)
                control_info = self._extract_search_information(
                    control_data, control_file
                )
            except Exception as e:
                logger.warning(f"Could not parse control file {control_file}: {e}")

        # Load experiment file if available
        experiment_file = result.get("experiment_file")
        if experiment_file:
            try:
                experiment_data = self._read_json_file_safely(experiment_file)
                experiment_info = self._extract_search_information(
                    experiment_data, experiment_file
                )
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
            "mentioned_files": [],
            "control_found_mentioned": [],
            "experiment_found_mentioned": [],
            "control_accessed_mentioned": [],
            "experiment_accessed_mentioned": [],
            "content_overlap_pct": 0.0,
            "similarity_category": "No_Data",
            "control_access_level": "unknown",
            "experiment_access_level": "unknown",
            "control_search_success": False,
            "experiment_search_success": False,
        }

        # Extract specific file/email mentions from user query
        query_text = result.get("query", "")
        mentioned_files = self._extract_mentioned_files(query_text)
        analysis["mentioned_files"] = mentioned_files

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

            # Check which mentioned files were found/accessed
            control_found_mentioned, control_accessed_mentioned = (
                self._check_mentioned_files_access(
                    mentioned_files,
                    [r["title"] for r in control_info["results_found"]],
                    [f["filename"] for f in control_info["files_accessed"]],
                )
            )

            experiment_found_mentioned, experiment_accessed_mentioned = (
                self._check_mentioned_files_access(
                    mentioned_files,
                    [r["title"] for r in experiment_info["results_found"]],
                    [f["filename"] for f in experiment_info["files_accessed"]],
                )
            )

            # Enhanced fair comparison logic using sophisticated criteria
            fair_comparison_candidate = self._evaluate_fair_comparison_candidate(
                mentioned_files,
                control_info,
                experiment_info,
                similarity_data,
                control_found_mentioned,
                experiment_found_mentioned,
                control_accessed_mentioned,
                experiment_accessed_mentioned,
                requested_items,
            )

            # Update analysis with detailed results
            analysis.update(
                {
                    "content_found_by_both": len(similarity_data["common_files"]) > 0,
                    "both_accessed_content": (
                        similarity_data["control_access_level"] == "full_access"
                        and similarity_data["experiment_access_level"] == "full_access"
                    ),
                    "fair_comparison_candidate": fair_comparison_candidate,
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
                    "control_found_mentioned": control_found_mentioned,
                    "experiment_found_mentioned": experiment_found_mentioned,
                    "control_accessed_mentioned": control_accessed_mentioned,
                    "experiment_accessed_mentioned": experiment_accessed_mentioned,
                    "content_overlap_pct": similarity_data["content_overlap_pct"],
                    "similarity_category": similarity_data["similarity_category"],
                    "control_access_level": similarity_data["control_access_level"],
                    "experiment_access_level": similarity_data[
                        "experiment_access_level"
                    ],
                    "control_search_success": self._determine_targeted_search_success(
                        control_info, requested_items
                    ),
                    "experiment_search_success": self._determine_targeted_search_success(
                        experiment_info, requested_items
                    ),
                }
            )

        elif control_info:
            # Only control data available
            control_found_mentioned, control_accessed_mentioned = (
                self._check_mentioned_files_access(
                    mentioned_files,
                    [r["title"] for r in control_info["results_found"]],
                    [f["filename"] for f in control_info["files_accessed"]],
                )
            )

            analysis.update(
                {
                    "control_files_found": [
                        r["title"] for r in control_info["results_found"]
                    ],
                    "control_files_accessed": [
                        f["filename"] for f in control_info["files_accessed"]
                    ],
                    "control_found_mentioned": control_found_mentioned,
                    "control_accessed_mentioned": control_accessed_mentioned,
                    "control_access_level": self._determine_access_level(control_info),
                    "control_search_success": self._determine_targeted_search_success(
                        control_info, requested_items
                    ),
                }
            )

        elif experiment_info:
            # Only experiment data available
            experiment_found_mentioned, experiment_accessed_mentioned = (
                self._check_mentioned_files_access(
                    mentioned_files,
                    [r["title"] for r in experiment_info["results_found"]],
                    [f["filename"] for f in experiment_info["files_accessed"]],
                )
            )

            analysis.update(
                {
                    "experiment_files_found": [
                        r["title"] for r in experiment_info["results_found"]
                    ],
                    "experiment_files_accessed": [
                        f["filename"] for f in experiment_info["files_accessed"]
                    ],
                    "experiment_found_mentioned": experiment_found_mentioned,
                    "experiment_accessed_mentioned": experiment_accessed_mentioned,
                    "experiment_access_level": self._determine_access_level(
                        experiment_info
                    ),
                    "experiment_search_success": self._determine_targeted_search_success(
                        experiment_info, requested_items
                    ),
                }
            )

        return analysis

    def create_clean_tsv_output(self, results: List[Dict], output_file: str) -> str:
        """Create a clean TSV file without header comments, using JSON-based analysis."""

        # Define TSV headers with detailed breakdown columns
        headers = [
            "query",
            "control_file",
            "experiment_file",
            "requested_files_emails",
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
            "control_detailed_category",
            "experiment_detailed_category",
            "control_access_reason",
            "experiment_access_reason",
            "control_results_count",
            "experiment_results_count",
            "control_response_length",
            "experiment_response_length",
            "control_search_success",
            "experiment_search_success",
        ]

        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")

            # Write header row (no comments)
            writer.writeheader()

            # Process each result with JSON-based analysis
            for result in results:
                query_text = result.get("query", "")
                requested_items = self._extract_requested_files_emails_from_query(
                    query_text
                )
                content_analysis = self._analyze_query_content_matching(
                    result, requested_items
                )

                # Get detailed breakdown for both control and experiment
                control_breakdown = {
                    "detailed_category": "unknown",
                    "reason": "Unknown",
                    "results_count": 0,
                    "response_length": 0,
                }
                experiment_breakdown = {
                    "detailed_category": "unknown",
                    "reason": "Unknown",
                    "results_count": 0,
                    "response_length": 0,
                }

                # Analyze control file if it exists
                control_file = result.get("control_file", "")
                if control_file and os.path.exists(control_file):
                    search_info = self._process_seval_file(control_file)
                    if search_info.get("error_messages") and any(
                        "Could not parse" in msg
                        for msg in search_info["error_messages"]
                    ):
                        control_breakdown = {
                            "detailed_category": "processing_error",
                            "reason": "File processing failed",
                            "results_count": 0,
                            "response_length": 0,
                        }
                    else:
                        control_breakdown = self._get_detailed_access_breakdown(
                            search_info
                        )

                # Analyze experiment file if it exists
                experiment_file = result.get("experiment_file", "")
                if experiment_file and os.path.exists(experiment_file):
                    search_info = self._process_seval_file(experiment_file)
                    if search_info.get("error_messages") and any(
                        "Could not parse" in msg
                        for msg in search_info["error_messages"]
                    ):
                        experiment_breakdown = {
                            "detailed_category": "processing_error",
                            "reason": "File processing failed",
                            "results_count": 0,
                            "response_length": 0,
                        }
                    else:
                        experiment_breakdown = self._get_detailed_access_breakdown(
                            search_info
                        )

                # Prepare row data with detailed breakdown
                query_text = result.get("query", "")
                requested_items = self._extract_requested_files_emails_from_query(
                    query_text
                )

                row = {
                    "query": query_text,
                    "control_file": result.get("control_file", ""),
                    "experiment_file": result.get("experiment_file", ""),
                    "requested_files_emails": "|".join(requested_items),
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
                    "control_detailed_category": control_breakdown["detailed_category"],
                    "experiment_detailed_category": experiment_breakdown[
                        "detailed_category"
                    ],
                    "control_access_reason": control_breakdown["reason"],
                    "experiment_access_reason": experiment_breakdown["reason"],
                    "control_results_count": control_breakdown.get("results_count", 0),
                    "experiment_results_count": experiment_breakdown.get(
                        "results_count", 0
                    ),
                    "control_response_length": control_breakdown.get(
                        "response_length", 0
                    ),
                    "experiment_response_length": experiment_breakdown.get(
                        "response_length", 0
                    ),
                    "control_search_success": content_analysis[
                        "control_search_success"
                    ],
                    "experiment_search_success": content_analysis[
                        "experiment_search_success"
                    ],
                }

                writer.writerow(row)

        logger.info(f"Created clean TSV output with {len(results)} rows: {output_file}")
        return output_file

    def extract_conversation_details(
        self,
        input_file: str,
        output_file: str,
    ):
        """
        Extract detailed conversation information from a SEVAL JSON file.

        This tool analyzes the conversation flow, tool calls, search results, and user-visible
        outputs from a SEVAL JSON file and generates comprehensive JSON and markdown reports.

        Args:
            input_file: Path to the SEVAL JSON file to analyze
            output_file: Path where the JSON analysis will be saved (a markdown report will also be generated)

        Outputs:
            1. Tool calls per round (tool name, messages, parameters)
            2. Search results with file/email names and content snippets
            3. Intermediate and final user-visible outputs
            4. JSON file with complete structured data

        """

        logger.info(f"Extracting conversation details from: {input_file}")
        logger.info(f"Output JSON file: {output_file}")

        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return

        try:
            # Load the SEVAL JSON file
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract basic information
            conversation_id = data.get("conversation_id", "Unknown")
            exp_name = data.get("exp_name", "Unknown")
            query_text = data.get("query", {}).get("id", "Unknown")
            conversation_success = data.get("conversation_success", False)

            # Get messages from the response
            messages = []
            if data.get("requests") and len(data["requests"]) > 0:
                request = data["requests"][0]
                messages = request.get("response_body", {}).get("messages", [])

            logger.info(
                f"Processing {len(messages)} messages from {exp_name} experiment"
            )

            # Analyze the conversation
            analysis = self._analyze_conversation_flow(
                messages,
                query_text,
                exp_name,
                conversation_id,
                conversation_success,
                input_file,
            )

            # Add summary counts to metadata
            eval_summary = analysis["evaluation_data_results"].get("summary", {})
            analysis["metadata"].update(
                {
                    "total_unique_tool_functions": eval_summary.get("total_tool_invocations_count", 0),
                    "total_tool_invocations_count": eval_summary.get("total_tool_invocations_count", 0),
                    "total_queries": eval_summary.get("total_queries", 0),
                    "total_search_results": eval_summary.get("total_search_results", 0),
                    "total_user_messages": len(analysis["user_visible_messages"]),
                    "total_rounds": len(analysis["rounds"]),
                }
            )

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully generated JSON analysis: {output_file}")
            
            # Log evaluation data summary
            eval_results = analysis.get("evaluation_data_results", {})
            summary = eval_results.get("summary", {})
            if summary:
                logger.info(
                    f"EvaluationData: {summary.get('total_search_results', 0)} search results from "
                    f"{summary.get('total_queries', 0)} queries in "
                    f"{summary.get('total_tool_invocations_count', 0)} tool invocations across "
                    f"{summary.get('total_turns', 0)} turns"
                )
            logger.info(
                f"User-visible messages: {len(analysis['user_visible_messages'])}"
            )

        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}")
            raise

    def _analyze_conversation_flow(
        self,
        messages: List[Dict],
        query_text: str,
        exp_name: str,
        conversation_id: str,
        conversation_success: bool,
        input_file: str,
    ) -> Dict:
        """Analyze the conversation flow and extract key information."""
        import os
        from datetime import datetime

        # Extract filename and SEVAL job ID from input path
        filename = os.path.basename(input_file)

        # Extract SEVAL job ID from path (e.g., "212953" from "212953_scraping_raw_data_output")
        seval_job_id = "Unknown"
        path_parts = input_file.replace("\\", "/").split("/")
        for part in path_parts:
            if "_scraping_raw_data_output" in part:
                seval_job_id = part.split("_")[0]
                break

        # Extract all search results from evaluationData
        evaluation_results = self._extract_evaluation_data_results(
            messages, conversation_id, exp_name
        )

        analysis = {
            "metadata": {
                "conversation_id": conversation_id,
                "exp_name": exp_name,
                "query_text": query_text,
                "conversation_success": conversation_success,
                "analysis_timestamp": datetime.now().isoformat(),
                "filename": filename,
                "seval_job_id": seval_job_id,
            },
            "evaluation_data_results": evaluation_results,
            "user_visible_messages": [],
            "cited_reference_ids": [],
            "rounds": [],
        }

        current_round = 1
        round_data = {
            "round_number": current_round,
            "tool_calls": [],
            "search_invocations": [],
            "search_results": [],
            "progress_messages": [],
            "latest_invocation_id": None,
        }

        for i, msg in enumerate(messages):
            msg_type = msg.get("messageType", "")
            author = msg.get("author", "")
            text = msg.get("text", "")
            timestamp = msg.get("timestamp", "")
            content_type = msg.get("contentType", "")

            # NOTE: Tool calls, search invocations, and search results are extracted
            # from evaluationData (authoritative source), not from messages[] array
            # to avoid duplicates. Messages[] contains multiple Internal messages
            # with repeated invocation info as search progresses.

            # Track progress messages
            if msg_type == "Progress" and author == "bot":
                progress_msg = {
                    "message_index": i,
                    "timestamp": timestamp,
                    "text": text[:200] + ("..." if len(text) > 200 else ""),
                    "full_text": text,
                }
                round_data["progress_messages"].append(progress_msg)

            # Track user-visible messages (final responses)
            elif author == "bot" and msg_type == "" and len(text) > 100:
                user_msg = {
                    "message_index": i,
                    "timestamp": timestamp,
                    "text": text,
                    "citations": msg.get("citations", []),
                    "length": len(text),
                    "is_final_response": True,
                }
                analysis["user_visible_messages"].append(user_msg)

            # NOTE: Round transitions are now determined from evaluationData.turnData
            # instead of detecting from messages[] array

        # Populate rounds from evaluationData instead of messages[] array
        # This ensures we have clean, non-duplicated tool call and search data
        self._populate_rounds_from_evaluation_data(
            analysis, evaluation_results, round_data
        )

        # Extract cited reference_ids from final response
        cited_ref_ids = set()
        for user_msg in analysis["user_visible_messages"]:
            text = user_msg.get("text", "")
            # Find all turn1searchX patterns in the response
            import re

            matches = re.findall(r"turn1search\d+", text)
            cited_ref_ids.update(matches)

        # Mark search results as cited or not cited
        for round_data in analysis["rounds"]:
            for search_result in round_data.get("search_results", []):
                for result in search_result.get("results", []):
                    ref_id = result.get("reference_id", "")
                    result["was_cited"] = ref_id in cited_ref_ids

        # Update cited reference IDs in analysis (already initialized in correct position)
        analysis["cited_reference_ids"] = sorted(list(cited_ref_ids))

        return analysis

    def _populate_rounds_from_evaluation_data(self, analysis, evaluation_results, round_data):
        """
        Populate rounds from already-extracted evaluationData.
        
        Args:
            analysis: The analysis dictionary to populate
            evaluation_results: The evaluation_data_results (already extracted)
            round_data: The current round data dictionary (with progress_messages)
        """
        if not evaluation_results or "turns" not in evaluation_results:
            # No evaluationData available, append current round_data if it has content
            if round_data.get("progress_messages"):
                analysis["rounds"].append(round_data)
            return
        
        # Process each turn from the already-extracted evaluation_results
        for turn in evaluation_results.get("turns", []):
            turn_num = turn.get("turn_number", 1)
            
            # Create round entry for this turn
            turn_round_data = {
                "round_number": turn_num,
                "progress_messages": round_data.get("progress_messages", [])
            }
            
            # Append this turn's round data
            analysis["rounds"].append(turn_round_data)

    def _extract_tool_call_info(
        self, msg: Dict, msg_index: int, round_number: int = 1, invocation_id: str = None
    ) -> Dict:
        """Extract tool call information from an InvokeAction message.

        Args:
            msg: Message dictionary
            msg_index: Index of message in messages array
            round_number: Current conversation round/turn number
            invocation_id: Unique identifier for this invocation
        """
        invocation = msg.get("invocation", "")
        text = msg.get("text", "")
        timestamp = msg.get("timestamp", "")

        # Parse tool name and parameters from invocation
        tool_name = "Unknown"
        parameters = {}

        try:
            # Parse the invocation JSON array
            if invocation.startswith("[") and invocation.endswith("]"):
                import json

                invocation_data = json.loads(invocation)
                if invocation_data and isinstance(invocation_data, list):
                    # The first element might be a string containing JSON
                    first_element = invocation_data[0]
                    if isinstance(first_element, str):
                        # Parse the nested JSON string
                        try:
                            first_call = json.loads(first_element)
                        except json.JSONDecodeError:
                            first_call = first_element
                    else:
                        first_call = first_element

                    if isinstance(first_call, dict) and "function" in first_call:
                        function_data = first_call["function"]
                        if "name" in function_data:
                            tool_name = function_data["name"]

                        # Try to parse arguments for parameters
                        if "arguments" in function_data:
                            try:
                                args_data = json.loads(function_data["arguments"])
                                if isinstance(args_data, dict):
                                    parameters = args_data
                            except json.JSONDecodeError:
                                pass

            # Fallback: Look for tool name in invocation string (old method)
            if tool_name == "Unknown" and '"name":' in invocation:
                name_match = re.search(r'"name":"([^"]+)"', invocation)
                if name_match:
                    tool_name = name_match.group(1)

            # Special handling for office365_search (legacy support)
            if "office365_search" in invocation:
                tool_name = "office365_search"
                # Extract search queries
                query_pattern = r'"query":"([^"]*)"'
                queries = re.findall(query_pattern, invocation)
                if queries:
                    parameters["queries"] = [
                        q.replace('\\"', '"').replace("\\\\", "\\")
                        for q in queries
                        if len(q) > 5
                    ]

            # Extract domain if present
            if "domain" in invocation:
                domain_pattern = r'"domain":"([^"]*)"'
                domain_match = re.search(domain_pattern, invocation)
                if domain_match:
                    parameters["domain"] = domain_match.group(1)

        except Exception as e:
            logger.warning(f"Could not parse tool invocation: {e}")

        return {
            "invocation_id": invocation_id,
            "message_index": msg_index,
            "round_number": round_number,
            "timestamp": timestamp,
            "tool_name": tool_name,
            "parameters": parameters,
            "progress_text": text,
            "raw_invocation": invocation[:500]
            + ("..." if len(invocation) > 500 else ""),
        }

    def _extract_search_invocations(
        self, msg: Dict, msg_index: int, round_number: int = 1, invocation_ref: str = None
    ) -> Optional[Dict]:
        """Extract search invocation details including all queries.

        Args:
            msg: Message dictionary
            msg_index: Index of message in messages array
            round_number: Current conversation round/turn number
            invocation_ref: Reference to the tool invocation that triggered this search
        """
        invocation = msg.get("invocation", "")
        timestamp = msg.get("timestamp", "")
        msg_type = msg.get("messageType", "")

        invocation_info = {
            "invocation_ref": invocation_ref,
            "message_index": msg_index,
            "round_number": round_number,
            "timestamp": timestamp,
            "message_type": msg_type,
            "function_name": "",
            "queries": [],
        }

        if not invocation or not isinstance(invocation, str):
            return None

        try:
            inv_data = json.loads(invocation)
            if isinstance(inv_data, list) and len(inv_data) > 0:
                for inv_item in inv_data:
                    func = inv_item.get("function", {})
                    func_name = func.get("name", "")

                    if "office365_search" in func_name:
                        invocation_info["function_name"] = func_name
                        args_str = func.get("arguments", "")

                        try:
                            args = json.loads(args_str)
                            queries = args.get("queries", [])

                            for query in queries:
                                invocation_info["queries"].append(
                                    {
                                        "domain": query.get("domain", ""),
                                        "query": query.get("query", ""),
                                        "response_length": query.get(
                                            "response_length", ""
                                        ),
                                    }
                                )
                        except:
                            pass
        except:
            pass

        return invocation_info if invocation_info["function_name"] else None

    def _extract_evaluation_data_results(
        self, messages: list, conversation_id: str = "Unknown", exp_name: str = "Unknown"
    ) -> Dict:
        """Extract all search results from EvaluationData message.

        This method extracts the COMPLETE set of search results from the EvaluationData message.
        The data is organized hierarchically:

        Data Source Path:
          messages[]
            → {"messageType": "EvaluationData"}
              → evaluationData
                → turnData[]                           # Each conversation turn (hop)
                  → orchestrationIterations[]          # Search/reasoning iterations within turn
                    → modelActions[]                   # Actions taken by model
                      → toolInvocations[]              # Tool calls (search API, etc.)
                        → batchedQueries[]             # Parallel queries to different domains
                          → arguments (JSON string)    # Contains: {"domain": "emails|files|chats|people", "query": "search text"}
                          → processedResult (JSON)     # Contains: {"results": [{reference_id, type, title, snippet, ...}]}

        Return Structure:
          {
            "turns": [                               # One entry per conversation turn
              {
                "turn_number": 1,                    # Turn identifier (1-indexed)
                "user_input": "user's query",        # The user's question for this turn
                "hops": [                            # Orchestration iterations (reasoning steps)
                  {
                    "hop_number": 1,                 # Hop identifier within turn (1-indexed)
                    "invocations": [                 # Tool invocations in this hop
                      {
                        "invocation_number": 1,      # Invocation counter within hop
                        "tool_name": "...",          # Name of tool invoked
                        "queries": [                 # Batched queries in this invocation
                          {
                            "query_number": 1,       # Query counter within invocation
                            "domain": "emails",      # Search domain (emails, files, chats, people)
                            "query": "search text",  # Actual query text sent to search API
                            "result_count": 10,      # Number of results returned
                            "results": [             # Array of search results
                              {
                                "reference_id": "turn1search1",  # Unique ID for CiteDCG mapping
                                "type": "Email|File|Chat|Person",
                                "title": "...",
                                "snippet": "...",
                                "author": "...",
                                # ... additional metadata fields
                              }
                            ]
                          }
                        ],
                        "total_results": 67          # Total results in this invocation
                      }
                    ],
                    "total_results": 67              # Total results in this hop
                  }
                ],
                "total_results": 67                  # Total results in this turn
              }
            ],
            "total_results": 67,                     # Total results across all turns
            "total_hops": 1,                         # Total hops across all turns
            "total_invocations": 1                   # Total invocations across all turns
          }

        Note: This extracts ALL search results, not just those cited in the final response.
        Use reference_id to map results to CiteDCG quality scores.
        """
        results_by_turn = []

        # Find EvaluationData message (contains complete search history)
        eval_message = None
        for msg in messages:
            if msg.get("messageType") == "EvaluationData":
                eval_message = msg
                break

        if not eval_message:
            return {"turns": [], "total_results": 0, "total_invocations": 0}

        eval_data = eval_message.get("evaluationData", {})
        turn_data_list = eval_data.get("turnData", [])

        # Verify multi-turn hop pattern
        if len(turn_data_list) > 1:
            verification = self._verify_multiturn_hop_pattern_for_conversation(turn_data_list)
            logger.info("")
            logger.info("MULTI-TURN HOP PATTERN VERIFICATION (Conversation Extraction)")
            logger.info(f"Total multi-turn conversations: {verification['total_multi_turn']}")
            logger.info(
                f"  Last turn has hops: "
                f"{verification['last_turn_has_hops']} "
                f"({verification['last_turn_has_hops'] / max(1, verification['total_multi_turn']) * 100:.1f}%)"
            )
            logger.info(f"  First turn has hops: {verification['first_turn_has_hops']}")
            logger.info(f"  Middle turn has hops: {verification['middle_turn_has_hops']}")
            logger.info(f"  Multiple turns with hops: {verification['multiple_turns_with_hops']}")
            
            if verification['pattern_by_turns']:
                logger.info("")
                logger.info("Pattern breakdown by # of turns:")
                for turns_key in sorted(verification['pattern_by_turns'].keys()):
                    patterns = verification['pattern_by_turns'][turns_key]
                    logger.info(f"  {turns_key}:")
                    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
                        logger.info(f"    {pattern}: {count}")
            
            logger.info("=" * 60)
            logger.info("")

        total_results = 0
        total_tool_invocations_count = 0
        total_queries = 0
        total_hops = 0

        for turn_idx, turn_data in enumerate(turn_data_list):
            turn_results = {
                "turn_number": turn_idx + 1,
                "user_input": turn_data.get("userInput", ""),
                "hops": [],  # Changed from "invocations" to "hops"
                "total_results": 0,
            }

            iterations = turn_data.get("orchestrationIterations", [])

            for hop_idx, iteration in enumerate(iterations):
                total_hops += 1
                # Create hop structure
                hop_results = {
                    "hop_number": hop_idx + 1,
                    "invocations": [],  # Invocations now nested under hops
                    "total_results": 0,
                }

                model_actions = iteration.get("modelActions", [])

                for model_action in model_actions:
                    tool_invocations = model_action.get("toolInvocations", [])

                    for tool_inv in tool_invocations:
                        # Extract tool name from invocation
                        tool_name = tool_inv.get("function", "unknown")
                        
                        total_tool_invocations_count += 1
                        invocation_results = {
                            "invocation_number": len(
                                hop_results["invocations"]
                            )
                            + 1,
                            "tool_name": tool_name,
                            "queries": [],
                            "total_results": 0,
                        }
                        
                        # Check for batchedQueries (office365_search and search_web patterns)
                        batched_queries = tool_inv.get("batchedQueries", [])

                        for query_idx, batched_query in enumerate(
                            batched_queries
                        ):
                            # Handle search_web batchedQueries differently
                            # search_web has: arguments=plain query string, processedResult={WebPages, News, Sports, etc.}
                            if tool_name == "search_web":
                                # For search_web, arguments is a plain query string, not JSON
                                query_text = batched_query.get("arguments", "")
                                
                                # Parse processedResult which has WebPages, News, Sports, etc.
                                processed_result_str = batched_query.get(
                                    "processedResult", ""
                                )
                                
                                if processed_result_str:
                                    try:
                                        processed_data = json.loads(processed_result_str)
                                        
                                        # Extract results by type (same as non-batched search_web)
                                        results_by_type = {}
                                        
                                        if isinstance(processed_data, dict):
                                            # Extract all possible result types from the dict
                                            if "QuestionsAndAnswers" in processed_data:
                                                results_by_type["questionsandanswers"] = processed_data.get("QuestionsAndAnswers", [])
                                            if "WebPages" in processed_data:
                                                results_by_type["webpages"] = processed_data.get("WebPages", [])
                                            if "News" in processed_data:
                                                results_by_type["news"] = processed_data.get("News", [])
                                            if "Sports" in processed_data:
                                                results_by_type["sports"] = processed_data.get("Sports", [])
                                        
                                        # Create separate query entries for each type
                                        for search_type, items in results_by_type.items():
                                            if not items:
                                                continue
                                            
                                            total_queries += 1
                                            query_results = {
                                                "query_number": len(invocation_results["queries"]) + 1,
                                                "domain": search_type,
                                                "query": query_text,
                                                "result_count": len(items),
                                                "results": [],
                                            }
                                            
                                            # Extract each result for this type
                                            for item in items:
                                                # Determine display type
                                                if search_type == "questionsandanswers":
                                                    item_type = "QA"
                                                elif search_type == "news":
                                                    item_type = "News"
                                                elif search_type == "sports":
                                                    item_type = "Sports"
                                                else:
                                                    item_type = "WebPage"
                                                
                                                # Extract snippets if available
                                                snippets = item.get("snippets", [])
                                                snippet_text = ""
                                                if snippets:
                                                    snippet_text = " ".join(str(s) for s in snippets)[:200]
                                                
                                                result_item = {
                                                    "reference_id": item.get("reference_id", ""),
                                                    "type": item_type,
                                                    "title": item.get("title", ""),
                                                    "snippet": snippet_text,
                                                    "author": "",
                                                    "lastModifiedTime": "",
                                                    "fileName": "",
                                                    "fileType": "",
                                                    "url": item.get("url", ""),
                                                }
                                                query_results["results"].append(result_item)
                                            
                                            invocation_results["queries"].append(query_results)
                                            invocation_results["total_results"] += len(items)
                                            hop_results["total_results"] += len(items)
                                            turn_results["total_results"] += len(items)
                                            total_results += len(items)
                                    except Exception as e:
                                        # Log error but continue
                                        user_input = turn_results.get("user_input", "")
                                        print(f"\nWARNING: Error parsing batched search_web result")
                                        print(f"  Conversation ID: {conversation_id}")
                                        print(f"  Error: {e}")
                                
                                # Continue to next batched query (skip office365_search logic)
                                continue
                            
                            # office365_search pattern: arguments is JSON with domain/query, processedResult has results array
                            total_queries += 1
                            # Parse arguments to get query details
                            arguments_str = batched_query.get("arguments", "")
                            domain = "unknown"
                            query_text = ""

                            if arguments_str:
                                try:
                                    arguments = json.loads(arguments_str)
                                    domain = arguments.get("domain", "unknown")
                                    query_text = arguments.get("query", "")
                                except Exception:
                                    pass

                            # Parse processedResult to get results
                            processed_result_str = batched_query.get(
                                "processedResult", ""
                            )
                            results = []

                            if processed_result_str:
                                try:
                                    processed_data = json.loads(
                                        processed_result_str
                                    )
                                    results = processed_data.get("results", [])
                                except Exception:
                                    pass

                            query_results = {
                                "query_number": query_idx + 1,
                                "domain": domain,
                                "query": query_text,
                                "result_count": len(results),
                                "results": [],
                            }

                            # Extract key info from each result
                            for result in results:
                                result_type = result.get("type", "")
                                
                                # Handle PeopleInferenceAnswer differently
                                if result_type == "PeopleInferenceAnswer":
                                    # Extract & clean displayName
                                    display_name = result.get(
                                        "displayName", ""
                                    )
                                    if display_name:
                                        display_name = display_name.replace(
                                            "<Person>", ""
                                        ).replace("</Person>", "")
                                    
                                    result_data = {
                                        "reference_id": result.get(
                                            "reference_id", ""
                                        ),
                                        "type": result_type,
                                        "title": display_name,
                                        "snippet": result.get(
                                            "profession", ""
                                        ),
                                        "author": result.get(
                                            "userPrincipalName", ""
                                        ),
                                        "lastModifiedTime": "",
                                        "fileName": result.get(
                                            "companyName", ""
                                        ),
                                        "fileType": result.get(
                                            "department", ""
                                        ),
                                        # Additional people-specific fields
                                        "emailAddresses": result.get(
                                            "emailAddresses", ""
                                        ),
                                        "officeLocation": result.get(
                                            "officeLocation", ""
                                        ),
                                    }
                                elif result_type == "TeamsMessage":
                                    # Handle TeamsMessage type-specific fields
                                    result_data = {
                                        "reference_id": result.get(
                                            "reference_id", ""
                                        ),
                                        "type": result_type,
                                        "title": result.get("title", ""),
                                        "snippet": result.get(
                                            "snippet", ""
                                        )[:200],
                                        "author": result.get("to", ""),
                                        "lastModifiedTime": result.get(
                                            "dateTimeSent", ""
                                        ),
                                        "fileName": "",
                                        "fileType": "",
                                    }
                                elif result_type == "EmailMessage":
                                    # Handle EmailMessage type-specific fields
                                    # Use dateTimeReceived for received emails, dateTimeSent for sent emails
                                    date_time = result.get("dateTimeReceived") or result.get("dateTimeSent", "")
                                    result_data = {
                                        "reference_id": result.get(
                                            "reference_id", ""
                                        ),
                                        "type": result_type,
                                        "title": result.get("subject", ""),
                                        "snippet": result.get(
                                            "snippet", ""
                                        )[:200],
                                        "author": result.get("from", ""),
                                        "lastModifiedTime": date_time,
                                        "fileName": "",
                                        "fileType": "",
                                    }
                                else:
                                    # Generic extraction for other types
                                    result_data = {
                                        "reference_id": result.get(
                                            "reference_id", ""
                                        ),
                                        "type": result_type,
                                        "title": result.get("title", ""),
                                        "snippet": result.get(
                                            "snippet", ""
                                        )[:200],
                                        "author": result.get("author", ""),
                                        "lastModifiedTime": result.get(
                                            "lastModifiedTime", ""
                                        ),
                                        "fileName": result.get(
                                            "fileName", ""
                                        ),
                                        "fileType": result.get(
                                            "fileType", ""
                                        ),
                                    }
                                
                                query_results["results"].append(result_data)

                            invocation_results["queries"].append(query_results)
                            invocation_results["total_results"] += len(results)
                            hop_results["total_results"] += len(results)
                            turn_results["total_results"] += len(results)
                            total_results += len(results)

                        # Fallback: When batchedQueries is empty, extract from toolInvocations directly
                        # This handles GPT-5 treatment cases where batchedQueries is [] but data exists in arguments/processedResult
                        if not batched_queries and tool_name == "office365_search" and "processedResult" in tool_inv:
                            # Extract queries from arguments field
                            arguments_str = tool_inv.get("arguments", "")
                            queries_list = []
                            
                            if arguments_str:
                                try:
                                    if isinstance(arguments_str, str):
                                        arguments = json.loads(arguments_str)
                                    else:
                                        arguments = arguments_str
                                    queries_list = arguments.get("queries", [])
                                except Exception:
                                    pass
                            
                            # Extract all results from processedResult
                            processed_result_str = tool_inv.get("processedResult", "")
                            all_results = []
                            
                            if processed_result_str:
                                try:
                                    if isinstance(processed_result_str, str):
                                        processed_data = json.loads(processed_result_str)
                                    else:
                                        processed_data = processed_result_str
                                    if isinstance(processed_data, dict):
                                        all_results = processed_data.get("results", [])
                                except Exception:
                                    pass
                            
                            # Process each query from the queries list
                            for query_idx, query_item in enumerate(queries_list):
                                total_queries += 1
                                
                                # Handle query_item being string or dict
                                if isinstance(query_item, str):
                                    try:
                                        query_item = json.loads(query_item)
                                    except Exception:
                                        query_item = {}
                                
                                if not isinstance(query_item, dict):
                                    query_item = {}
                                
                                domain = query_item.get("domain", "unknown")
                                query_text = query_item.get("query", "")
                                
                                # Note: We cannot accurately split results per-query when batchedQueries is empty
                                # For single-query cases, all results belong to that query
                                # For multi-query cases, we assign all results to the first query only
                                # to avoid inflating result counts
                                if query_idx == 0:
                                    results_for_query = all_results
                                else:
                                    results_for_query = []  # Don't duplicate results for other queries
                                
                                query_results = {
                                    "query_number": query_idx + 1,
                                    "domain": domain,
                                    "query": query_text,
                                    "result_count": len(results_for_query),
                                    "results": [],
                                }
                                
                                # Extract key info from each result
                                for result in results_for_query:
                                    if not isinstance(result, dict):
                                        continue
                                    
                                    result_type = result.get("type", "")
                                    
                                    # Handle PeopleInferenceAnswer differently
                                    if result_type == "PeopleInferenceAnswer":
                                        display_name = result.get("displayName", "")
                                        if display_name:
                                            display_name = display_name.replace("<Person>", "").replace("</Person>", "")
                                        
                                        result_data = {
                                            "reference_id": result.get("reference_id", ""),
                                            "type": result_type,
                                            "title": display_name,
                                            "snippet": result.get("profession", ""),
                                            "author": result.get("userPrincipalName", ""),
                                            "lastModifiedTime": "",
                                            "fileName": result.get("companyName", ""),
                                            "fileType": result.get("department", ""),
                                            "emailAddresses": result.get("emailAddresses", ""),
                                            "officeLocation": result.get("officeLocation", ""),
                                        }
                                    elif result_type == "TeamsMessage":
                                        result_data = {
                                            "reference_id": result.get("reference_id", ""),
                                            "type": result_type,
                                            "title": result.get("title", ""),
                                            "snippet": result.get("snippet", "")[:200] if result.get("snippet") else "",
                                            "author": result.get("to", ""),
                                            "lastModifiedTime": result.get("dateTimeSent", ""),
                                            "fileName": "",
                                            "fileType": "",
                                        }
                                    elif result_type == "EmailMessage":
                                        date_time = result.get("dateTimeReceived") or result.get("dateTimeSent", "")
                                        result_data = {
                                            "reference_id": result.get("reference_id", ""),
                                            "type": result_type,
                                            "title": result.get("subject", ""),
                                            "snippet": result.get("snippet", "")[:200] if result.get("snippet") else "",
                                            "author": result.get("from", ""),
                                            "lastModifiedTime": date_time,
                                            "fileName": "",
                                            "fileType": "",
                                        }
                                    else:
                                        # Generic extraction for other types (File, etc.)
                                        result_data = {
                                            "reference_id": result.get("reference_id", ""),
                                            "type": result_type,
                                            "title": result.get("title", ""),
                                            "snippet": result.get("snippet", "")[:200] if result.get("snippet") else "",
                                            "author": result.get("author", ""),
                                            "lastModifiedTime": result.get("lastModifiedTime", ""),
                                            "fileName": result.get("fileName", ""),
                                            "fileType": result.get("fileType", ""),
                                        }
                                    
                                    query_results["results"].append(result_data)
                                
                                invocation_results["queries"].append(query_results)
                                invocation_results["total_results"] += len(results_for_query)
                                hop_results["total_results"] += len(results_for_query)
                                turn_results["total_results"] += len(results_for_query)
                                total_results += len(results_for_query)

                        # Check for web search results (search_web pattern)
                        # Web search stores results in 'processedResult' field, not batchedQueries
                        if not batched_queries and "processedResult" in tool_inv and tool_name == "search_web":
                            # Extract query from arguments field
                            arguments_str = tool_inv.get("arguments", "")
                            query_text = ""
                            if arguments_str:
                                try:
                                    arguments = json.loads(arguments_str)
                                    query_text = arguments.get("query", "")
                                except Exception:
                                    pass
                            
                            result_str = tool_inv.get("processedResult", "")
                            # Only process if we have non-empty content after stripping
                            if result_str:
                                result_str = result_str.strip()
                                if result_str:
                                    try:
                                        # Parse result JSON
                                        result_data = json.loads(result_str)
                                        
                                        # Handle three different structures:
                                        # 1. Dict with 'News' and 'WebPages' keys (standard format)
                                        # 2. Dict with 'QuestionsAndAnswers' key (site-restricted search)
                                        # 3. List of messages with groundingInfo (legacy format)
                                        
                                        # Extract results by type
                                        results_by_type = {}
                                        
                                        if isinstance(result_data, dict):
                                            # Extract all possible result types from the dict
                                            if "QuestionsAndAnswers" in result_data:
                                                results_by_type["questionsandanswers"] = result_data.get("QuestionsAndAnswers", [])
                                            if "WebPages" in result_data:
                                                results_by_type["webpages"] = result_data.get("WebPages", [])
                                            if "News" in result_data:
                                                results_by_type["news"] = result_data.get("News", [])
                                        elif isinstance(result_data, list) and result_data:
                                            # Legacy format: list of messages with groundingInfo
                                            web_pages = []
                                            news_items = []
                                            for msg in result_data:
                                                grounding = msg.get("groundingInfo", {})
                                                web_pages.extend(grounding.get("WebPages", []))
                                                news_items.extend(grounding.get("News", []))
                                            if web_pages:
                                                results_by_type["webpages"] = web_pages
                                            if news_items:
                                                results_by_type["news"] = news_items
                                        
                                        # Create separate query entries for each type (to match CiteDCG structure)
                                        for search_type, items in results_by_type.items():
                                            if not items:
                                                continue
                                            
                                            total_queries += 1
                                            # Create a query entry for this specific type
                                            # Domain format: search_web_{type} to match CiteDCG
                                            query_results = {
                                                "query_number": len(invocation_results["queries"]) + 1,
                                                "domain": search_type,  # Type-specific domain
                                                "query": query_text,
                                                "result_count": len(items),
                                                "results": [],
                                            }
                                            
                                            # Extract each result for this type
                                            for item in items:
                                                # Determine display type
                                                if search_type == "questionsandanswers":
                                                    item_type = "QA"
                                                elif search_type == "news":
                                                    item_type = "News"
                                                else:
                                                    item_type = "WebPage"
                                                
                                                # Extract snippets if available
                                                snippets = item.get("snippets", [])
                                                snippet_text = ""
                                                if snippets:
                                                    snippet_text = " ".join(str(s) for s in snippets)[:200]
                                                
                                                result_item = {
                                                    "reference_id": item.get("reference_id", ""),
                                                    "type": item_type,
                                                    "title": item.get("title", ""),
                                                    "snippet": snippet_text,
                                                    "author": "",
                                                    "lastModifiedTime": "",
                                                    "fileName": "",
                                                    "fileType": "",
                                                    "url": item.get("url", ""),
                                                }
                                                query_results["results"].append(result_item)
                                            
                                            invocation_results["queries"].append(query_results)
                                            invocation_results["total_results"] += len(items)
                                            hop_results["total_results"] += len(items)
                                            turn_results["total_results"] += len(items)
                                            total_results += len(items)
                                        
                                        # Check if NO types had results
                                        if not results_by_type:
                                            # Web search invocation with no results - log for investigation
                                            user_input = turn_results.get("user_input", "")
                                            print(f"\n{'='*80}")
                                            print("WARNING: Web search with no News/WebPages results")
                                            print(f"  Conversation ID: {conversation_id}")
                                            print(f"  Experiment: {exp_name}")
                                            print(f"  Utterance: {user_input[:100]}...")
                                            print(f"  Turn: {turn_results.get('turn_number')}, Hop: {hop_results.get('hop_number')}")
                                            print(f"  Search query: {query_text[:100] if query_text else 'no query'}")
                                            print(f"  processedResult type: {type(result_data)}")
                                            if isinstance(result_data, dict):
                                                print(f"  processedResult keys: {list(result_data.keys())}")
                                            print(f"{'='*80}\n")
                                    except Exception as e:
                                        # Log error but continue processing
                                        user_input = turn_results.get("user_input", "")
                                        print(f"\n{'='*80}")
                                        print("WARNING: Error parsing web search result")
                                        print(f"  Conversation ID: {conversation_id}")
                                        print(f"  Experiment: {exp_name}")
                                        print(f"  Utterance: {user_input[:100]}...")
                                        print(f"  Turn: {turn_results.get('turn_number')}, Hop: {hop_results.get('hop_number')}")
                                        print(f"  Error: {e}")
                                        print(f"{'='*80}\n")
                                else:
                                    # Empty processedResult after stripping
                                    user_input = turn_results.get("user_input", "")
                                    print(f"\n{'='*80}")
                                    print("WARNING: Web search with empty processedResult")
                                    print(f"  Conversation ID: {conversation_id}")
                                    print(f"  Experiment: {exp_name}")
                                    print(f"  Utterance: {user_input[:100]}...")
                                    print(f"  Turn: {turn_results.get('turn_number')}, Hop: {hop_results.get('hop_number')}")
                                    print(f"{'='*80}\n")
                            else:
                                # Empty or missing processedResult
                                if not batched_queries:  # Only for web search pattern
                                    user_input = turn_results.get("user_input", "")
                                    print(f"\n{'='*80}")
                                    print("WARNING: Web search pattern but no processedResult")
                                    print(f"  Conversation ID: {conversation_id}")
                                    print(f"  Experiment: {exp_name}")
                                    print(f"  Utterance: {user_input[:100]}...")
                                    print(f"  Turn: {turn_results.get('turn_number')}, Hop: {hop_results.get('hop_number')}")
                                    print(f"{'='*80}\n")
                        
                        # Check for Graph Connector searches (e.g., search_enterprise_connectors_*)
                        # These have arguments and processedResult directly on tool_inv, not in batchedQueries
                        if (not batched_queries and 
                            "processedResult" in tool_inv and 
                            tool_name != "search_web" and
                            ("search_enterprise_connectors" in tool_name or 
                             "search_enterprise" in tool_name)):
                            
                            # Extract query from arguments field
                            arguments_str = tool_inv.get("arguments", "")
                            query_text = ""
                            query_keywords = ""
                            
                            if arguments_str:
                                try:
                                    arguments = json.loads(arguments_str)
                                    query_text = arguments.get("query", arguments.get("QueryKeywords", ""))
                                    query_keywords = arguments.get("QueryKeywords", "")
                                except Exception:
                                    pass
                            
                            # Parse processedResult to get results
                            result_str = tool_inv.get("processedResult", "")
                            if result_str:
                                result_str = result_str.strip()
                                if result_str:
                                    try:
                                        # Parse result JSON
                                        result_data = json.loads(result_str)
                                        
                                        # Graph connector results are in "results" array
                                        results = []
                                        if isinstance(result_data, dict):
                                            results = result_data.get("results", [])
                                        
                                        if results:
                                            total_queries += 1
                                            
                                            # Extract ContentDomainName from first result's sourceJson
                                            # This matches DCG structure where ContentDomain.Name is the domain
                                            domain = None
                                            if results and isinstance(results[0], dict):
                                                source_json_str = results[0].get("sourceJson", "")
                                                if source_json_str:
                                                    try:
                                                        source_json = json.loads(source_json_str)
                                                        domain = source_json.get("ContentDomainName")
                                                    except Exception:
                                                        pass
                                            
                                            # If extraction failed, log error and skip this query
                                            if not domain:
                                                user_input = turn_results.get("user_input", "")
                                                print(f"\n{'='*80}")
                                                print("ERROR: Could not extract ContentDomainName from Graph Connector result")
                                                print(f"  Conversation ID: {conversation_id}")
                                                print(f"  Tool: {tool_name}")
                                                print(f"  Utterance: {user_input[:100]}...")
                                                print(f"  Turn: {turn_results.get('turn_number')}, Hop: {hop_results.get('hop_number')}")
                                                print(f"{'='*80}\n")
                                                continue
                                            
                                            # Create a query entry for this Graph Connector search
                                            query_results = {
                                                "query_number": len(invocation_results["queries"]) + 1,
                                                "domain": domain,
                                                "query": query_text or query_keywords,
                                                "result_count": len(results),
                                                "results": [],
                                            }
                                            
                                            # Extract each result
                                            for item in results:
                                                result_item = {
                                                    "reference_id": item.get("reference_id", ""),
                                                    "type": item.get("type", "External"),
                                                    "title": item.get("title", ""),
                                                    "snippet": item.get("snippet", "")[:200],
                                                    "author": "",
                                                    "lastModifiedTime": item.get("lastModifiedTime", ""),
                                                    "fileName": "",
                                                    "fileType": "",
                                                }
                                                query_results["results"].append(result_item)
                                            
                                            invocation_results["queries"].append(query_results)
                                            invocation_results["total_results"] += len(results)
                                            hop_results["total_results"] += len(results)
                                            turn_results["total_results"] += len(results)
                                            total_results += len(results)
                                    except Exception as e:
                                        # Log error but continue processing
                                        user_input = turn_results.get("user_input", "")
                                        print(f"\n{'='*80}")
                                        print("WARNING: Error parsing Graph Connector search result")
                                        print(f"  Conversation ID: {conversation_id}")
                                        print(f"  Experiment: {exp_name}")
                                        print(f"  Tool: {tool_name}")
                                        print(f"  Utterance: {user_input[:100]}...")
                                        print(f"  Turn: {turn_results.get('turn_number')}, Hop: {hop_results.get('hop_number')}")
                                        print(f"  Error: {e}")
                                        print(f"{'='*80}\n")

                        # Handle fetch_file invocations
                        # fetch_file retrieves a file by ID/name and returns content
                        # CiteDCG evaluates these fetched files with scores
                        if (not batched_queries and 
                            "processedResult" in tool_inv and 
                            tool_name == "fetch_file"):
                            
                            # Parse processedResult to get results
                            result_str = tool_inv.get("processedResult", "")
                            if result_str:
                                result_str = result_str.strip()
                                if result_str:
                                    # Skip known non-JSON error responses (expired, failed, etc.)
                                    if result_str.lower() in ('expired', 'failed', 'error', 'null', 'none'):
                                        # Silently skip - this is expected for failed fetches
                                        pass
                                    else:
                                        try:
                                            # Parse result JSON
                                            result_data = json.loads(result_str)
                                            
                                            # fetch_file results are in "results" array
                                            results = []
                                            if isinstance(result_data, dict):
                                                results = result_data.get("results", [])
                                            
                                            if results:
                                                total_queries += 1
                                                
                                                # Create a query entry for this fetch_file invocation
                                                # Domain is "fetch_file" to match CiteDCG plugin_name
                                                query_results = {
                                                    "query_number": len(invocation_results["queries"]) + 1,
                                                    "domain": "fetch_file",
                                                    "query": "",  # fetch_file has no query string
                                                    "result_count": len(results),
                                                    "results": [],
                                                }
                                                
                                                # Extract each result
                                                for item in results:
                                                    # Get title from various possible fields
                                                    title = item.get("title", "") or item.get("fileName", "") or item.get("name", "")
                                                    
                                                    result_item = {
                                                        "reference_id": item.get("reference_id", ""),
                                                        "type": "fetch_file_results",  # Match CiteDCG Type
                                                        "title": title,
                                                        "snippet": item.get("snippet", item.get("content", ""))[:200],
                                                        "author": item.get("author", ""),
                                                        "lastModifiedTime": item.get("lastModifiedTime", ""),
                                                        "fileName": item.get("fileName", item.get("name", "")),
                                                        "fileType": item.get("fileType", item.get("type", "")),
                                                        "itemId": item.get("itemId", item.get("id", "")),
                                                    }
                                                    query_results["results"].append(result_item)
                                                
                                                invocation_results["queries"].append(query_results)
                                                invocation_results["total_results"] += len(results)
                                                hop_results["total_results"] += len(results)
                                                turn_results["total_results"] += len(results)
                                                total_results += len(results)
                                        except json.JSONDecodeError:
                                            # Non-JSON result string - silently skip
                                            # This includes error messages like "Expired", "Failed", etc.
                                            pass
                                        except Exception as e:
                                            # Log unexpected errors but continue processing
                                            user_input = turn_results.get("user_input", "")
                                            logger.warning(
                                                f"Unexpected error parsing fetch_file result: {e}, "
                                                f"conv_id={conversation_id}, "
                                                f"turn={turn_results.get('turn_number')}"
                                            )

                        # Only add invocation if it has queries (batched, web search, or graph connector)
                        if invocation_results["queries"]:
                            hop_results["invocations"].append(invocation_results)

                # Always add hop to preserve complete iteration structure
                # (even if empty - shows reasoning steps without tool calls)
                turn_results["hops"].append(hop_results)

            results_by_turn.append(turn_results)

        return {
            "summary": {
                "total_turns": len(results_by_turn),
                "total_hops": total_hops,
                "total_tool_invocations_count": total_tool_invocations_count,
                "total_queries": total_queries,
                "total_search_results": total_results,
            },
            "turns": results_by_turn,
        }

    def _verify_multiturn_hop_pattern_for_conversation(self, turn_data_list: list) -> dict:
        """Verify which turns have hops in multi-turn conversations (conversation extraction).
        
        Args:
            turn_data_list: List of turn data from evaluationData.turnData
            
        Returns:
            Dictionary with verification statistics
        """
        verification = {
            'total_multi_turn': 0,
            'last_turn_has_hops': 0,
            'first_turn_has_hops': 0,
            'middle_turn_has_hops': 0,
            'multiple_turns_with_hops': 0,
            'pattern_by_turns': {},
            'examples': []
        }
        
        # We only have one conversation in this method, check if it's multi-turn
        if len(turn_data_list) <= 1:
            return verification
        
        verification['total_multi_turn'] = 1
        num_turns = len(turn_data_list)
        
        # Check which turns have orchestrationIterations (hops)
        turns_with_hops = []
        for turn_idx, turn in enumerate(turn_data_list):
            iterations = turn.get("orchestrationIterations", [])
            if iterations:
                turns_with_hops.append(turn_idx + 1)
        
        # Categorize the pattern
        num_turns_with_hops = len(turns_with_hops)
        pattern = "unknown"
        
        if num_turns_with_hops == 0:
            pattern = "no_hops"
        elif num_turns_with_hops == 1:
            turn_num = turns_with_hops[0]
            if turn_num == num_turns:
                pattern = "last_turn_only"
                verification['last_turn_has_hops'] = 1
            elif turn_num == 1:
                pattern = "first_turn_only"
                verification['first_turn_has_hops'] = 1
            else:
                pattern = "middle_turn_only"
                verification['middle_turn_has_hops'] = 1
        else:
            pattern = "multiple_turns"
            verification['multiple_turns_with_hops'] = 1
        
        # Track pattern by number of turns
        turns_key = f"{num_turns}_turns"
        if turns_key not in verification['pattern_by_turns']:
            verification['pattern_by_turns'][turns_key] = {}
        verification['pattern_by_turns'][turns_key][pattern] = 1
        
        return verification

    def _extract_search_result_info(
        self, msg: Dict, msg_index: int, round_number: int = 1, invocation_ref: str = None
    ) -> Dict:
        """Extract search result information.

        Args:
            msg: Message dictionary
            msg_index: Index of message in messages array
            round_number: Current conversation round/turn number
            invocation_ref: Reference to the tool invocation that triggered this search
        """
        text = msg.get("text", "")
        timestamp = msg.get("timestamp", "")
        msg_type = msg.get("messageType", "")
        content_type = msg.get("contentType", "")

        # Try to extract from adaptiveCards if text is empty
        if not text or len(text) < 10:
            if "adaptiveCards" in msg and len(msg["adaptiveCards"]) > 0:
                card = msg["adaptiveCards"][0]
                if "body" in card and len(card["body"]) > 0:
                    body_item = card["body"][0]
                    if "inlines" in body_item and len(body_item["inlines"]) > 0:
                        inline_item = body_item["inlines"][0]
                        text = inline_item.get("text", "")

        result_info = {
            "invocation_ref": invocation_ref,
            "message_index": msg_index,
            "round_number": round_number,
            "timestamp": timestamp,
            "message_type": msg_type,
            "content_type": content_type,
            "items_found": 0,
            "results": [],
            "raw_content": text[:1000] + ("..." if len(text) > 1000 else ""),
        }

        # Parse JSON results if present
        try:
            if text.startswith("{") and '"results":' in text:
                parsed = json.loads(text)
                if "results" in parsed:
                    results = parsed["results"]
                    result_info["items_found"] = len(results)

                    # Also capture searchMetadata if present
                    if "searchMetadata" in parsed:
                        result_info["searchMetadata"] = parsed["searchMetadata"]

                    # Extract details for each result
                    for result in results:  # Process all results, not just first 5
                        # Results are already in the correct format (not nested under "result" key)
                        result_detail = {
                            "reference_id": result.get("reference_id", ""),
                            "id": result.get(
                                "id", ""
                            ),  # SharePoint/Email ID for CiteDCG mapping
                            "type": result.get("type", ""),
                            # Email fields
                            "subject": result.get("subject", ""),
                            "from_email": result.get("from", ""),
                            "to_email": result.get("to", ""),
                            "sent_time": result.get("sentTime", ""),
                            "dateTimeReceived": result.get("dateTimeReceived", ""),
                            "dateTimeSent": result.get("dateTimeSent", ""),
                            "isRead": result.get("isRead", None),
                            # File fields
                            "file_name": result.get("fileName", ""),
                            "title": result.get("title", ""),
                            "file_type": result.get("fileType", ""),
                            "author": result.get("author", ""),
                            "lastModifiedTime": result.get("lastModifiedTime", ""),
                            "lastModifiedBy": result.get("lastModifiedBy", ""),
                            # People fields
                            "displayName": result.get("displayName", ""),
                            "emailAddresses": result.get("emailAddresses", ""),
                            "userPrincipalName": result.get("userPrincipalName", ""),
                            "profession": result.get("profession", ""),
                            "department": result.get("department", ""),
                            "officeLocation": result.get("officeLocation", ""),
                            # Common fields
                            "snippet": result.get("snippet", "")[:300]
                            + ("..." if len(result.get("snippet", "")) > 300 else ""),
                            "url": result.get("url", ""),
                            # Related entities for people results
                            "relatedEntities": result.get("relatedEntities", []),
                        }
                        result_info["results"].append(result_detail)

                # Also check telemetry.groundingResponse for additional results
                if "telemetry" in msg:
                    telemetry = msg.get("telemetry", {})
                    if "groundingResponse" in telemetry:
                        grounding = telemetry["groundingResponse"]
                        if isinstance(grounding, dict) and "results" in grounding:
                            grounding_results = grounding["results"]
                            # Add grounding results that aren't already in the main results
                            existing_ref_ids = set(
                                r.get("reference_id") for r in result_info["results"]
                            )
                            for result in grounding_results:
                                ref_id = result.get("reference_id", "")
                                if ref_id and ref_id not in existing_ref_ids:
                                    result_detail = {
                                        "reference_id": ref_id,
                                        "id": result.get("id", ""),
                                        "type": result.get("type", ""),
                                        # Email fields
                                        "subject": result.get("subject", ""),
                                        "from_email": result.get("from", ""),
                                        "to_email": result.get("to", ""),
                                        "sent_time": result.get("sentTime", ""),
                                        "dateTimeReceived": result.get(
                                            "dateTimeReceived", ""
                                        ),
                                        "dateTimeSent": result.get("dateTimeSent", ""),
                                        "isRead": result.get("isRead", None),
                                        # File fields
                                        "file_name": result.get("fileName", ""),
                                        "title": result.get("title", ""),
                                        "file_type": result.get("fileType", ""),
                                        "author": result.get("author", ""),
                                        "lastModifiedTime": result.get(
                                            "lastModifiedTime", ""
                                        ),
                                        "lastModifiedBy": result.get(
                                            "lastModifiedBy", ""
                                        ),
                                        # People fields
                                        "displayName": result.get("displayName", ""),
                                        "emailAddresses": result.get(
                                            "emailAddresses", ""
                                        ),
                                        "userPrincipalName": result.get(
                                            "userPrincipalName", ""
                                        ),
                                        "profession": result.get("profession", ""),
                                        "department": result.get("department", ""),
                                        "officeLocation": result.get(
                                            "officeLocation", ""
                                        ),
                                        # Common fields
                                        "snippet": result.get("snippet", "")[:300]
                                        + (
                                            "..."
                                            if len(result.get("snippet", "")) > 300
                                            else ""
                                        ),
                                        "url": result.get("url", ""),
                                        "relatedEntities": result.get(
                                            "relatedEntities", []
                                        ),
                                        "source": "grounding",  # Mark as coming from grounding
                                    }
                                    result_info["results"].append(result_detail)
                                    result_info["items_found"] += 1

            elif text.startswith("[") and len(text) > 10:
                # Handle array format results
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    result_info["items_found"] = len(parsed)
                    for item in parsed[:5]:
                        if isinstance(item, dict):
                            result_detail = {
                                "query": item.get("query", ""),
                                "result": str(item.get("result", ""))[:300]
                                + (
                                    "..."
                                    if len(str(item.get("result", ""))) > 300
                                    else ""
                                ),
                            }
                            result_info["results"].append(result_detail)

        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            if "No content returned" not in text and len(text) > 20:
                result_info["results"].append(
                    {
                        "type": "text",
                        "content": text[:300] + ("..." if len(text) > 300 else ""),
                    }
                )

        return result_info

    def _generate_markdown_report(self, analysis: Dict, output_file: str):
        """Generate a comprehensive markdown report."""
        metadata = analysis["metadata"]

        # Ensure output directory exists
        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
            exist_ok=True,
        )

        with open(output_file, "w", encoding="utf-8") as f:
            # Header
            f.write(f"# SEVAL Conversation Analysis Report\n\n")
            f.write(f"**Generated:** {metadata['analysis_timestamp']}\n\n")

            # Metadata section
            f.write(f"## Conversation Metadata\n\n")
            f.write(f"- **File Name:** `{metadata['filename']}`\n")
            f.write(f"- **SEVAL Job ID:** `{metadata['seval_job_id']}`\n")
            f.write(f"- **Conversation ID:** `{metadata['conversation_id']}`\n")
            f.write(f"- **Experiment Type:** `{metadata['exp_name']}`\n")
            f.write(f"- **Success:** `{metadata['conversation_success']}`\n")
            f.write(f"- **Total Messages:** {metadata['total_messages']}\n")
            f.write(f"- **Tool Calls:** {len(analysis['tool_calls'])}\n")
            f.write(f"- **Search Results:** {len(analysis['search_results'])}\n\n")

            # Query section
            f.write(f"## Original Query\n\n")
            f.write(f"```\n{metadata['query_text']}\n```\n\n")

            # Rounds analysis
            f.write(f"## Conversation Rounds ({len(analysis['rounds'])} rounds)\n\n")

            for round_data in analysis["rounds"]:
                f.write(f"### Round {round_data['round_number']}\n\n")

                # Tool calls in this round
                if round_data["tool_calls"]:
                    f.write(f"#### Tool Calls ({len(round_data['tool_calls'])})\n\n")
                    for i, tool_call in enumerate(round_data["tool_calls"], 1):
                        f.write(f"**Call {i}: {tool_call['tool_name']}**\n")
                        f.write(f"- **Invocation ID:** `{tool_call.get('invocation_id', 'N/A')}`\n")
                        f.write(f"- **Timestamp:** {tool_call['timestamp']}\n")
                        f.write(f"- **Message Index:** {tool_call['message_index']}\n")

                        if tool_call["parameters"]:
                            f.write(f"- **Parameters:**\n")
                            for key, value in tool_call["parameters"].items():
                                if isinstance(value, list):
                                    f.write(f"  - **{key}:** {len(value)} items\n")
                                    for j, item in enumerate(value[:3], 1):
                                        f.write(f"    {j}. `{item}`\n")
                                    if len(value) > 3:
                                        f.write(f"    ... and {len(value) - 3} more\n")
                                else:
                                    f.write(f"  - **{key}:** `{value}`\n")

                        if tool_call["progress_text"]:
                            f.write(f"- **Progress:** {tool_call['progress_text']}\n")

                        f.write(f"\n")

                # Search invocations in this round
                if round_data["search_invocations"]:
                    f.write(
                        f"#### Search Invocations ({len(round_data['search_invocations'])})\n\n"
                    )
                    for i, search_inv in enumerate(round_data["search_invocations"], 1):
                        f.write(f"**Invocation {i}: {search_inv['function_name']}**\n")
                        f.write(f"- **Triggered By:** `{search_inv.get('invocation_ref', 'N/A')}`\n")
                        f.write(f"- **Timestamp:** {search_inv['timestamp']}\n")
                        f.write(f"- **Message Index:** {search_inv['message_index']}\n")
                        f.write(f"- **Message Type:** {search_inv['message_type']}\n")
                        f.write(
                            f"- **Number of Queries:** {len(search_inv['queries'])}\n\n"
                        )

                        if search_inv["queries"]:
                            f.write(f"**Queries:**\n\n")
                            for q_idx, query in enumerate(search_inv["queries"], 1):
                                f.write(f"{q_idx}. **Domain:** `{query['domain']}`\n")
                                f.write(
                                    f"   - **Query:** `{query['query'][:100]}{'...' if len(query['query']) > 100 else ''}`\n"
                                )
                                f.write(
                                    f"   - **Response Length:** `{query['response_length']}`\n\n"
                                )

                        f.write(f"\n")

                # Search results in this round
                if round_data["search_results"]:
                    f.write(
                        f"#### Search Results ({len(round_data['search_results'])})\n\n"
                    )
                    for i, search_result in enumerate(round_data["search_results"], 1):
                        f.write(f"**Result {i}**\n")
                        f.write(f"- **From Invocation:** `{search_result.get('invocation_ref', 'N/A')}`\n")
                        f.write(f"- **Timestamp:** {search_result['timestamp']}\n")
                        f.write(
                            f"- **Message Index:** {search_result['message_index']}\n"
                        )
                        f.write(
                            f"- **Type:** {search_result['message_type']} / {search_result['content_type']}\n"
                        )
                        f.write(
                            f"- **Items Found:** {search_result['items_found']}\n\n"
                        )

                        if search_result["results"]:
                            f.write(f"**Found Items:**\n")
                            for j, item in enumerate(search_result["results"], 1):
                                f.write(f"{j}. ")
                                if item.get("type") == "EmailMessage":
                                    f.write(
                                        f"**Email:** `{item.get('subject', 'No subject')}`\n"
                                    )
                                    if item.get("reference_id"):
                                        cited_marker = (
                                            " ✓ CITED" if item.get("was_cited") else ""
                                        )
                                        f.write(
                                            f"   - **Reference ID:** `{item['reference_id']}`{cited_marker}\n"
                                        )
                                    if item.get("id"):
                                        f.write(f"   - **ID:** `{item['id']}`\n")
                                    if item.get("from_email"):
                                        f.write(
                                            f"   - **From:** {item['from_email']}\n"
                                        )
                                    if item.get("sent_time"):
                                        f.write(f"   - **Sent:** {item['sent_time']}\n")
                                    if item.get("snippet"):
                                        f.write(
                                            f"   - **Snippet:** {item['snippet']}\n"
                                        )
                                elif item.get("type") in [
                                    "File",
                                    "PeopleInferenceAnswer",
                                    "PeopleNotes",
                                    "TeamsMessage",
                                    "External",
                                ]:
                                    # Handle all typed results
                                    type_label = item.get("type", "Unknown")
                                    display_name = (
                                        item.get("file_name")
                                        or item.get("title")
                                        or item.get("displayName")
                                        or item.get("subject")
                                        or "(No name)"
                                    )
                                    f.write(f"**{type_label}:** `{display_name}`\n")

                                    if item.get("reference_id"):
                                        cited_marker = (
                                            " ✓ CITED" if item.get("was_cited") else ""
                                        )
                                        f.write(
                                            f"   - **Reference ID:** `{item['reference_id']}`{cited_marker}\n"
                                        )
                                    if item.get("id"):
                                        f.write(f"   - **ID:** `{item['id']}`\n")

                                    # Type-specific fields
                                    if item.get("file_type"):
                                        f.write(
                                            f"   - **File Type:** {item['file_type']}\n"
                                        )
                                    if item.get("author"):
                                        f.write(f"   - **Author:** {item['author']}\n")
                                    if item.get("lastModifiedTime"):
                                        f.write(
                                            f"   - **Modified:** {item['lastModifiedTime']}\n"
                                        )
                                    if item.get("profession"):
                                        f.write(
                                            f"   - **Role:** {item['profession']}\n"
                                        )
                                    if item.get("department"):
                                        f.write(
                                            f"   - **Department:** {item['department']}\n"
                                        )
                                    if item.get("emailAddresses"):
                                        f.write(
                                            f"   - **Email:** {item['emailAddresses']}\n"
                                        )
                                    if item.get("url"):
                                        f.write(
                                            f"   - **URL:** {item['url'][:100]}...\n"
                                        )

                                    if item.get("snippet"):
                                        f.write(
                                            f"   - **Snippet:** {item['snippet']}\n"
                                        )

                                    # Show related entities count for people
                                    if (
                                        item.get("relatedEntities")
                                        and len(item["relatedEntities"]) > 0
                                    ):
                                        f.write(
                                            f"   - **Related Entities:** {len(item['relatedEntities'])} items\n"
                                        )
                                elif item.get("file_name"):
                                    # Fallback for files without type field
                                    f.write(f"**File:** `{item['file_name']}`\n")
                                    if item.get("reference_id"):
                                        f.write(
                                            f"   - **Reference ID:** `{item['reference_id']}`\n"
                                        )
                                    if item.get("id"):
                                        f.write(f"   - **ID:** `{item['id']}`\n")
                                    if item.get("file_type"):
                                        f.write(f"   - **Type:** {item['file_type']}\n")
                                    if item.get("snippet"):
                                        f.write(
                                            f"   - **Content:** {item['snippet']}\n"
                                        )
                                elif item.get("query"):
                                    f.write(f"**Query Result:** `{item['query']}`\n")
                                    if item.get("reference_id"):
                                        f.write(
                                            f"   - **Reference ID:** `{item['reference_id']}`\n"
                                        )
                                    if item.get("id"):
                                        f.write(f"   - **ID:** `{item['id']}`\n")
                                    f.write(
                                        f"   - **Result:** {item.get('result', '')}\n"
                                    )
                                else:
                                    # Generic item type
                                    f.write(
                                        f"**Item:** {item.get('content', str(item))}\n"
                                    )
                                    if item.get("reference_id"):
                                        f.write(
                                            f"   - **Reference ID:** `{item['reference_id']}`\n"
                                        )
                                    if item.get("id"):
                                        f.write(f"   - **ID:** `{item['id']}`\n")
                                f.write(f"\n")
                        f.write(f"\n")

                # Progress messages
                if round_data["progress_messages"]:
                    f.write(
                        f"#### Progress Messages ({len(round_data['progress_messages'])})\n\n"
                    )
                    for i, progress in enumerate(round_data["progress_messages"], 1):
                        f.write(
                            f"{i}. **[{progress['timestamp']}]** {progress['text']}\n"
                        )

            f.write(f"---\n\n")

            # User-visible outputs
            f.write(f"## User-Visible Outputs\n\n")

            if analysis["user_visible_messages"]:
                for i, msg in enumerate(analysis["user_visible_messages"], 1):
                    f.write(f"### Output {i}\n\n")
                    f.write(f"- **Timestamp:** {msg['timestamp']}\n")
                    f.write(f"- **Message Index:** {msg['message_index']}\n")
                    f.write(f"- **Length:** {msg['length']} characters\n")
                    f.write(f"- **Citations:** {len(msg['citations'])} items\n\n")

                    f.write(f"**Content:**\n")
                    f.write(f"```\n{msg['text']}\n```\n\n")

                    if msg["citations"]:
                        f.write(f"**Citations:**\n")
                        for j, citation in enumerate(msg["citations"], 1):
                            f.write(f"{j}. {citation}\n")
                        f.write(f"\n")
            else:
                f.write(f"No user-visible outputs found.\n\n")

            # Summary statistics
            f.write(f"## Summary Statistics\n\n")
            f.write(f"- **Total Rounds:** {len(analysis['rounds'])}\n")
            f.write(f"- **Total Tool Calls:** {len(analysis['tool_calls'])}\n")
            f.write(
                f"- **Total Search Invocations:** {len(analysis['search_invocations'])}\n"
            )
            f.write(f"- **Total Search Results:** {len(analysis['search_results'])}\n")
            f.write(
                f"- **User-Visible Messages:** {len(analysis['user_visible_messages'])}\n"
            )

            # Calculate search query statistics
            total_queries = sum(
                len(inv["queries"]) for inv in analysis["search_invocations"]
            )
            f.write(f"- **Total Search Queries:** {total_queries}\n")

            # Domain breakdown
            domain_counts = {}
            for inv in analysis["search_invocations"]:
                for query in inv["queries"]:
                    domain = query.get("domain", "unknown")
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1

            if domain_counts:
                f.write(f"- **Queries by Domain:**\n")
                for domain in sorted(domain_counts.keys()):
                    f.write(f"  - {domain}: {domain_counts[domain]}\n")

            # Calculate search efficiency
            total_items_found = sum(
                result["items_found"] for result in analysis["search_results"]
            )
            if analysis["tool_calls"]:
                search_efficiency = total_items_found / len(analysis["tool_calls"])
                f.write(
                    f"- **Search Efficiency:** {search_efficiency:.1f} items per tool call\n"
                )

            f.write(f"- **Total Items Found:** {total_items_found}\n\n")

            # Add CiteDCG mapping section
            f.write("\n## CiteDCG Mapping Reference\n\n")
            f.write(
                "This section provides a summary of all search results with their reference IDs for mapping to CiteDCG quality scores.\n\n"
            )

            # Collect all search results across all rounds
            all_search_results = []
            for round_data in analysis["rounds"]:
                for search_result in round_data.get("search_results", []):
                    if search_result.get("results"):
                        for result in search_result["results"]:
                            if result.get("reference_id"):
                                all_search_results.append(result)

            if all_search_results:
                cited_count = sum(1 for r in all_search_results if r.get("was_cited"))
                not_cited_count = len(all_search_results) - cited_count

                f.write(f"**Total Search Results Found:** {len(all_search_results)}\n")
                f.write(f"**Cited in Final Response:** {cited_count}\n")
                f.write(f"**Not Cited:** {not_cited_count}\n\n")

                # Group by type
                by_type = {}
                for result in all_search_results:
                    result_type = result.get("type", "Unknown")
                    if result_type not in by_type:
                        by_type[result_type] = []
                    by_type[result_type].append(result)

                f.write("### Results by Type\n\n")
                for result_type, results in sorted(by_type.items()):
                    f.write(f"#### {result_type} ({len(results)} results)\n\n")
                    f.write("| Reference ID | Cited | ID/Email | Name/Subject |\n")
                    f.write("|--------------|-------|----------|-------------|\n")

                    for result in results:
                        ref_id = result.get("reference_id", "")
                        was_cited = "✓" if result.get("was_cited") else ""
                        item_id = (
                            result.get("id", "")
                            or result.get("emailAddresses", "")
                            or "-"
                        )
                        name = (
                            result.get("file_name")
                            or result.get("title")
                            or result.get("displayName")
                            or result.get("subject")
                            or "(No name)"
                        )

                        # Truncate long values
                        if len(str(item_id)) > 50:
                            item_id = str(item_id)[:47] + "..."
                        if len(str(name)) > 60:
                            name = str(name)[:57] + "..."

                        f.write(
                            f"| `{ref_id}` | {was_cited} | `{item_id}` | {name} |\n"
                        )

                    f.write("\n")

                # Add mapping instructions
                f.write("### CiteDCG Mapping Instructions\n\n")
                f.write("To map these results to CiteDCG quality scores:\n\n")
                f.write(
                    "1. The `reference_id` (e.g., `turn1search5`) maps to CiteDCG results **by rank order**\n"
                )
                f.write(
                    "2. Load the CiteDCG results file and sort all results by their `Rank` field\n"
                )
                f.write(
                    "3. Extract the index from `turn1searchX` (e.g., `turn1search5` → index 5)\n"
                )
                f.write(
                    "4. The result at index X in the sorted CiteDCG array corresponds to this search result\n"
                )
                f.write(
                    "5. Extract the `CiteDCGLLMLabel` from the matched CiteDCG result\n\n"
                )
                f.write("**Example mapping code:**\n")
                f.write("```python\n")
                f.write("# Load and sort CiteDCG results\n")
                f.write("citedcg_results = load_citedcg_file(citedcg_path)\n")
                f.write(
                    "sorted_results = sorted(citedcg_results, key=lambda x: x.get('Rank', 999))\n\n"
                )
                f.write("# Map reference_id to CiteDCG score\n")
                f.write("reference_id = 'turn1search5'\n")
                f.write("index = int(reference_id.replace('turn1search', ''))\n")
                f.write("citedcg_score = sorted_results[index]['CiteDCGLLMLabel']\n")
                f.write("```\n\n")
            else:
                f.write(
                    "No search results with reference IDs found in this conversation.\n\n"
                )

            # Add EvaluationData search results section
            eval_data = analysis.get("evaluation_data_results", {})
            if eval_data and eval_data.get("total_results", 0) > 0:
                f.write("\n## EvaluationData Search Results\n\n")
                f.write(
                    "This section contains all search results extracted from the EvaluationData message, "
                    "organized by turn, invocation, and query. These results include complete metadata "
                    "and reference IDs for mapping to CiteDCG quality scores.\n\n"
                )

                f.write(f"**Total Results:** {eval_data['total_results']}\n")
                f.write(f"**Total Invocations:** {eval_data['total_invocations']}\n")
                f.write(f"**Total Turns:** {len(eval_data.get('turns', []))}\n\n")

                # Process each turn
                for turn in eval_data.get("turns", []):
                    turn_num = turn.get("turn_number", 0)
                    user_input = turn.get("user_input", "")

                    f.write(f"### Turn {turn_num}\n\n")
                    f.write(f"**User Input:** {user_input}\n\n")
                    f.write(
                        f"**Total Results in Turn:** {turn.get('total_results', 0)}\n\n"
                    )

                    # Process each invocation in the turn
                    for invocation in turn.get("invocations", []):
                        inv_num = invocation.get("invocation_number", 0)
                        f.write(f"#### Invocation {inv_num}\n\n")
                        f.write(
                            f"**Results in Invocation:** {invocation.get('total_results', 0)}\n\n"
                        )

                        # Process each query in the invocation
                        for query in invocation.get("queries", []):
                            query_num = query.get("query_number", 0)
                            domain = query.get("domain", "unknown")
                            query_text = query.get("query", "")
                            result_count = query.get("result_count", 0)

                            f.write(f"##### Query {query_num} ({domain} domain)\n\n")
                            f.write(f"**Query Text:** {query_text}\n\n")
                            f.write(f"**Results Found:** {result_count}\n\n")

                            # Display results in a table
                            results = query.get("results", [])
                            if results:
                                f.write(
                                    "| Reference ID | Type | Title/Name | Author/From |\n"
                                )
                                f.write(
                                    "|--------------|------|------------|-------------|\n"
                                )

                                for result in results:
                                    ref_id = result.get("reference_id", "")
                                    result_type = result.get("type", "Unknown")

                                    # Get title/name based on result type
                                    title = (
                                        result.get("title")
                                        or result.get("displayName")
                                        or result.get("fileName")
                                        or result.get("subject")
                                        or "(No title)"
                                    )

                                    # Get author/from based on result type
                                    author = (
                                        result.get("author")
                                        or result.get("from")
                                        or result.get("givenName")
                                        or "-"
                                    )

                                    # Truncate long values
                                    if len(str(title)) > 50:
                                        title = str(title)[:47] + "..."
                                    if len(str(author)) > 30:
                                        author = str(author)[:27] + "..."

                                    f.write(
                                        f"| `{ref_id}` | {result_type} | {title} | {author} |\n"
                                    )

                                f.write("\n")

                                # Add expandable details section for complete metadata
                                f.write(
                                    "<details>\n<summary>Click to view complete result metadata</summary>\n\n"
                                )
                                f.write("```json\n")
                                import json

                                f.write(json.dumps(results, indent=2))
                                f.write("\n```\n")
                                f.write("</details>\n\n")
                            else:
                                f.write("No results found for this query.\n\n")

                # Add summary table by domain
                f.write("### Summary by Domain\n\n")
                domain_summary = {}
                for turn in eval_data.get("turns", []):
                    for invocation in turn.get("invocations", []):
                        for query in invocation.get("queries", []):
                            domain = query.get("domain", "unknown")
                            count = query.get("result_count", 0)
                            domain_summary[domain] = (
                                domain_summary.get(domain, 0) + count
                            )

                if domain_summary:
                    f.write("| Domain | Total Results |\n")
                    f.write("|--------|---------------|\n")
                    for domain in sorted(domain_summary.keys()):
                        f.write(f"| {domain} | {domain_summary[domain]} |\n")
                    f.write("\n")


def main():
    """Main function to run the SEVAL Analysis Toolkit with Fire."""
    fire.Fire(SEVALAnalysisToolkit)


if __name__ == "__main__":
    main()
