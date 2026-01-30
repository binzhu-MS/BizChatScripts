"""
Batch processing script for llm_ndcg metrics on SEVAL data.

This script processes a JSONL file containing llm_ndcg input records and
calculates metrics for each record, outputting results in the same format
as the standard testing_metric.py script.

Description:
    - Processes JSONL input files (one record per line)
    - Supports Retrieved Good Gain flight flag
    - Supports multi-threading for faster processing
    - Outputs results to JSONL file
    - Shows progress and handles errors gracefully

Usage (command line):
    python batch_run_llm_ndcg.py \
        --input "path/to/input.jsonl" \
        --output "path/to/output.jsonl" \
        --flights "ndcg-retrieved-good-gain" \
        --max_records 10 \
        --multithreaded

Usage (VS Code Debug):
    Use the launch.json configuration "Batch Run LLM NDCG"
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from types import SimpleNamespace
from typing import List, Optional, Tuple

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRIC_DEF_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, METRIC_DEF_DIR)

from google.protobuf.json_format import MessageToDict
from msal import PublicClientApplication

# Import after path setup
from scripts.testing_metric import TestingMetric

# Lock for thread-safe file writing
output_lock = Lock()

# Global cached token for thread-safe reuse
_cached_token = None
_token_lock = Lock()


def get_cached_token():
    """Get or acquire a token with thread-safe caching."""
    global _cached_token
    with _token_lock:
        if _cached_token is None:
            print("Acquiring authentication token (you may see a login prompt)...")
            app = PublicClientApplication(
                "c278a72c-22cc-4870-b10f-700808cc6466",
                authority="https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47",
                enable_broker_on_windows=True,
                enable_broker_on_mac=True,
            )
            SCOPES = ["https://substrate.office.com/llmapi/LLMAPI.dev"]
            accounts = app.get_accounts()
            result = None

            if accounts:
                chosen = accounts[0]
                result = app.acquire_token_silent(SCOPES, account=chosen)

            if not result:
                result = app.acquire_token_interactive(
                    scopes=SCOPES,
                    parent_window_handle=app.CONSOLE_WINDOW_HANDLE
                )
                if "error" in result:
                    raise ValueError(f"Failed to acquire token: {result.get('error_description', result)}")

            _cached_token = result["access_token"]
            print("Authentication successful!")
        return _cached_token


def get_token_from_cache(self):
    """Replacement for get_token_local that uses the pre-cached token."""
    return get_cached_token()


def process_record(
    record: dict,
    record_index: int,
    flights: Optional[List[str]] = None,
    multithreaded: bool = False,
) -> Tuple[int, Optional[dict], float]:
    """
    Process a single llm_ndcg input record.

    Args:
        record: The input record in llm_ndcg format.
        record_index: Index of the record (for logging).
        flights: List of flight flags to enable.
        multithreaded: Use multiple threads for LLM calls within this record.

    Returns:
        Tuple of (record_index, result_dict or None, elapsed_time).
    """
    record_id = record.get("id", f"record_{record_index}")
    start_time = time.time()

    try:
        # Add flights if specified
        if flights:
            record["request_context"] = SimpleNamespace(flights=flights)

        # Create metric tester (no LLM mock - use real LLM)
        metric_tester = TestingMetric(
            metric_name="llm_ndcg",
            input_json=record,
            llm_mock=None,  # Use real LLM
            multithreaded=multithreaded,
        )

        # Calculate metric
        result = metric_tester.calculate_metric()

        # Convert protobuf to dict
        result_dict = MessageToDict(
            result,
            including_default_value_fields=True,
            preserving_proto_field_name=True,
        )

        # Add input context fields for traceability
        result_dict["utterance"] = record.get("utterance", "")
        result_dict["user_profile"] = record.get("user_profile", "")

        elapsed = time.time() - start_time
        return (record_index, result_dict, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ERROR processing {record_id}: {e}")
        return (record_index, None, elapsed)


def batch_run(
    input_file: str,
    output_file: str,
    flights: Optional[str] = None,
    max_records: Optional[int] = None,
    start_from: int = 0,
    threads: int = 1,
    multithreaded: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Batch process llm_ndcg input file.

    Args:
        input_file: Path to input JSONL file.
        output_file: Path to output JSONL file.
        flights: Comma-separated list of flight flags.
        max_records: Maximum number of records to process (None for all).
        start_from: Record index to start from (0-based).
        threads: Number of parallel threads for batch processing (default: 1).
        multithreaded: Enable multi-threading within each metric calculation.
        verbose: Show detailed progress.

    Returns:
        Statistics dictionary.
    """
    # Parse flights
    flight_list = flights.split(",") if flights else None

    print("=" * 70)
    print("LLM NDCG Batch Processing")
    print("=" * 70)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Flights:     {flight_list}")
    print(f"Max records: {max_records or 'all'}")
    print(f"Start from:  {start_from}")
    print(f"Threads:     {threads}")
    print(f"Multithreaded (per-record): {multithreaded}")
    print("=" * 70)

    # Pre-authenticate before starting batch processing
    # This ensures the auth popup appears once, before any threads are spawned
    import scripts.testing_metric as testing_metric_module
    print("\nPre-authenticating for LLM API access...")
    try:
        get_cached_token()  # This will trigger auth popup if needed
        # Monkey-patch the module's get_token_local to use our cached version
        testing_metric_module.get_token_local = get_token_from_cache
        print("Token cached and ready for multi-threaded access.\n")
    except Exception as e:
        print(f"ERROR: Authentication failed: {e}")
        return {"error": f"Authentication failed: {e}"}

    # Validate input file
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return {"error": "Input file not found"}

    # Count total records
    with open(input_file, "r", encoding="utf-8") as f:
        total_records = sum(1 for _ in f)
    print(f"\nTotal records in file: {total_records}")

    # Calculate records to process
    records_to_process = total_records - start_from
    if max_records:
        records_to_process = min(records_to_process, max_records)
    print(f"Records to process: {records_to_process}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Load records to process
    records_data = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if i < start_from:
                continue
            if max_records and len(records_data) >= max_records:
                break
            record = json.loads(line.strip())
            records_data.append((i, record))

    # Process records
    stats = {
        "total": len(records_data),
        "success": 0,
        "failed": 0,
        "skipped": start_from,
        "start_time": datetime.now().isoformat(),
    }

    start_time = time.time()
    results_dict = {}  # Store results by index for ordered output
    completed_count = 0

    if threads == 1:
        # Sequential processing
        with open(output_file, "w", encoding="utf-8") as fout:
            for idx, (record_index, record) in enumerate(records_data):
                record_id = record.get("id", f"record_{record_index}")

                if verbose:
                    print(f"\n[{idx + 1}/{records_to_process}] Processing: {record_id}")

                _, result, elapsed = process_record(
                    record, record_index, flight_list, multithreaded
                )

                if result:
                    stats["success"] += 1
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()

                    if verbose:
                        print(f"  SUCCESS ({elapsed:.1f}s)")
                        _print_rgg_summary(result)
                else:
                    stats["failed"] += 1
                    if verbose:
                        print(f"  FAILED ({elapsed:.1f}s)")
    else:
        # Parallel processing with ThreadPoolExecutor
        print(f"\nUsing {threads} parallel threads...")

        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    process_record, record, record_index, flight_list, multithreaded
                ): (record_index, record.get("id", f"record_{record_index}"))
                for record_index, record in records_data
            }

            # Process results as they complete
            for future in as_completed(futures):
                record_index, record_id = futures[future]
                completed_count += 1

                try:
                    idx, result, elapsed = future.result()

                    if result:
                        stats["success"] += 1
                        results_dict[idx] = result

                        if verbose:
                            print(
                                f"[{completed_count}/{records_to_process}] "
                                f"SUCCESS: {record_id} ({elapsed:.1f}s)"
                            )
                    else:
                        stats["failed"] += 1
                        if verbose:
                            print(
                                f"[{completed_count}/{records_to_process}] "
                                f"FAILED: {record_id} ({elapsed:.1f}s)"
                            )

                except Exception as e:
                    stats["failed"] += 1
                    print(f"[{completed_count}/{records_to_process}] ERROR: {record_id}: {e}")

        # Write results in original order
        print("\nWriting results to output file (in original order)...")
        with open(output_file, "w", encoding="utf-8") as fout:
            for record_index, _ in records_data:
                if record_index in results_dict:
                    fout.write(json.dumps(results_dict[record_index], ensure_ascii=False) + "\n")

    # Calculate elapsed time
    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = elapsed
    stats["end_time"] = datetime.now().isoformat()

    # Print summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total processed: {stats['total']}")
    print(f"Successful:      {stats['success']}")
    print(f"Failed:          {stats['failed']}")
    print(f"Elapsed time:    {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if stats["total"] > 0:
        print(f"Avg per record:  {elapsed/stats['total']:.1f}s")
    print(f"Output file:     {output_file}")
    print("=" * 70)

    return stats


def _print_rgg_summary(result: dict) -> None:
    """Print Retrieved Good Gain summary if present."""
    ctrl_rgg = result.get("control", {}).get("retrieved_good_gain")
    if ctrl_rgg:
        summary = ctrl_rgg.get("summary", {})
        print(
            f"  Control RGG: iterations={summary.get('search_iteration_count', 0)}, "
            f"total_results={summary.get('total_results', 0)}, "
            f"good_results={summary.get('total_good_results', 0)}, "
            f"duplicates={summary.get('total_duplicates', 0)}"
        )

    treat_rgg = result.get("treatment", {}).get("retrieved_good_gain")
    if treat_rgg:
        summary = treat_rgg.get("summary", {})
        print(
            f"  Treatment RGG: iterations={summary.get('search_iteration_count', 0)}, "
            f"total_results={summary.get('total_results', 0)}, "
            f"good_results={summary.get('total_good_results', 0)}, "
            f"duplicates={summary.get('total_duplicates', 0)}"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch process llm_ndcg metrics on SEVAL data"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--flights",
        required=False,
        default=None,
        help="Comma-separated list of flight flags (e.g., 'ndcg-retrieved-good-gain')"
    )
    parser.add_argument(
        "--max_records",
        type=int,
        required=False,
        default=None,
        help="Maximum number of records to process (default: all)"
    )
    parser.add_argument(
        "--start_from",
        type=int,
        required=False,
        default=0,
        help="Record index to start from (0-based, default: 0)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        required=False,
        default=1,
        help="Number of parallel threads for batch processing (default: 1)"
    )
    parser.add_argument(
        "--multithreaded",
        action="store_true",
        help="Enable multi-threading within each metric calculation (for LLM calls)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed progress output"
    )

    args = parser.parse_args()

    batch_run(
        input_file=args.input,
        output_file=args.output,
        flights=args.flights,
        max_records=args.max_records,
        start_from=args.start_from,
        threads=args.threads,
        multithreaded=args.multithreaded,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
