# batch_run_llm_ndcg_sdk.py
#
# Copyright (c) Microsoft Corporation. All rights reserved.

"""
Batch Run LLM NDCG via CoMet SDK (MetricsAPIClient).

Author: GitHub Copilot
Created: 2026-02-11
Description: Batch processes a JSONL input file by calling the CoMet SDK
             (MetricsAPIClient from cometclientsdk) in dev mode. This
             demonstrates how an external system like SEVAL calls CoMet.

Usage:
    python batch_run_llm_ndcg_sdk.py \\
        --input tests/135056_top200_new.jsonl \\
        --output tests/135056_top200_llm_ndcg_output_sdk.jsonl \\
        --metric llm_ndcg \\
        --flights ndcg-retrieved-good-gain
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from cometclientsdk.MetricsAPIClient import MetricsAPIClient
from cometdefinition.metrics.llm_ndcg.llm_ndcg_pb2 import (
    LLMNDCGMetricInput,
    LLMNDCGOptions,
)
from google.protobuf.json_format import MessageToDict


def _to_string(value: Any) -> str:
    """
    Convert a value to string, serializing dicts/lists to JSON.

    The protobuf input expects string fields, but JSONL records may
    contain nested objects that need to be serialized first.

    Args:
        value: The value to convert (str, dict, list, or other).

    Returns:
        The value as a string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _build_metric_input(record: Dict[str, Any]) -> LLMNDCGMetricInput:
    """
    Build a LLMNDCGMetricInput protobuf from a raw JSONL record.

    Args:
        record: The input record dictionary with keys like utterance,
                user_profile, all_search_results_control, etc.

    Returns:
        A populated LLMNDCGMetricInput protobuf object.
    """
    metric_input = LLMNDCGMetricInput()
    metric_input.utterance = _to_string(record.get("utterance", ""))
    metric_input.user_profile = _to_string(record.get("user_profile", ""))
    metric_input.timestamp = _to_string(record.get("timestamp", ""))
    metric_input.all_search_results_control = _to_string(
        record.get("all_search_results_control", "")
    )
    metric_input.all_search_results_treatment = _to_string(
        record.get("all_search_results_treatment", "")
    )

    # Set options if present
    options = record.get("options", {})
    if options or record.get("all_plugins"):
        metric_input.options.CopyFrom(
            LLMNDCGOptions(
                all_plugins=options.get("all_plugins", record.get("all_plugins", "")),
                debug=options.get("debug", record.get("debug", False)),
                cache=options.get("cache", record.get("cache", False)),
                use_full_entity=options.get("use_full_entity", False),
            )
        )

    return metric_input


def _process_record(
    record: Dict[str, Any],
    record_index: int,
    client: MetricsAPIClient,
    metric_name: str,
    version: str,
    flights: Optional[List[str]] = None,
) -> Tuple[int, Optional[Dict[str, Any]], float]:
    """
    Process a single record by calling the CoMet SDK (MetricsAPIClient).

    Args:
        record: The input record dictionary.
        record_index: Index of the record for logging.
        client: The MetricsAPIClient instance.
        metric_name: Name of the metric to calculate.
        version: Version of the metric.
        flights: List of flight flags to enable.

    Returns:
        Tuple of (record_index, result_dict or None, elapsed_seconds).
    """
    record_id = record.get("id", f"record_{record_index}")
    start_time = time.time()

    try:
        # Build protobuf input from JSON record
        metric_input = _build_metric_input(record)

        # Call metric via CoMet SDK
        result = client.compute_metric_any_direct(
            metric=metric_name,
            version=version,
            custom_metric_input=metric_input,
            id=str(record_id),
            flights=flights or [],
        )

        # Convert protobuf output to dict
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


def _print_rgg_summary(result: Dict[str, Any]) -> None:
    """
    Print Retrieved Good Gain summary if present in the result.

    Args:
        result: The metric result dictionary.
    """
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


def batch_run(
    input_file: str,
    output_file: str,
    metric_name: str = "llm_ndcg",
    version: str = "1.0.0",
    flights: Optional[str] = None,
    max_records: Optional[int] = None,
    start_from: int = 0,
    threads: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Batch process input records by calling the CoMet SDK (MetricsAPIClient).

    Args:
        input_file: Path to input JSONL file.
        output_file: Path to output JSONL file.
        metric_name: Name of the metric to calculate (default: "llm_ndcg").
        version: Metric version string (default: "1.0.0").
        flights: Comma-separated list of flight flags.
        max_records: Maximum number of records to process (None for all).
        start_from: Record index to start from (0-based).
        threads: Number of parallel threads for batch processing.
        verbose: Show detailed progress.

    Returns:
        Statistics dictionary with processing results.
    """
    # Parse flights
    flight_list = flights.split(",") if flights else []

    print("=" * 70)
    print("LLM NDCG Batch Processing (via CoMet SDK — MetricsAPIClient)")
    print("=" * 70)
    print(f"Metric:      {metric_name}")
    print(f"Version:     {version}")
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Flights:     {flight_list}")
    print(f"Max records: {max_records or 'all'}")
    print(f"Start from:  {start_from}")
    print(f"Threads:     {threads}")
    print("=" * 70)

    # Validate input file
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return {"error": "Input file not found"}

    # Initialize CoMet SDK client in dev mode (local computation)
    print("\nInitializing CoMet SDK (MetricsAPIClient) in dev mode...")
    client = MetricsAPIClient(is_dev=True)
    print(f"Using MetricsAPIClient for metric: {metric_name} v{version}\n")

    # Count total records
    with open(input_file, "r", encoding="utf-8") as f:
        total_records = sum(1 for _ in f)
    print(f"Total records in file: {total_records}")

    # Calculate records to process
    records_to_process = total_records - start_from
    if max_records:
        records_to_process = min(records_to_process, max_records)
    print(f"Records to process: {records_to_process}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Load records
    records_data: List[Tuple[int, Dict[str, Any]]] = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if i < start_from:
                continue
            if max_records and len(records_data) >= max_records:
                break
            record = json.loads(line.strip())
            records_data.append((i, record))

    # Process records
    stats: Dict[str, Any] = {
        "total": len(records_data),
        "success": 0,
        "failed": 0,
        "skipped": start_from,
        "start_time": datetime.now().isoformat(),
    }

    start_time = time.time()
    results_dict: Dict[int, Dict[str, Any]] = {}

    if threads == 1:
        # Sequential processing
        with open(output_file, "w", encoding="utf-8") as fout:
            for idx, (record_index, record) in enumerate(records_data):
                record_id = record.get("id", f"record_{record_index}")

                if verbose:
                    print(f"\n[{idx + 1}/{records_to_process}] Processing: {record_id}")

                _, result, elapsed = _process_record(
                    record, record_index, client, metric_name, version, flight_list
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
        completed_count = 0

        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    _process_record, record, record_index,
                    client, metric_name, version, flight_list
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
                    print(
                        f"[{completed_count}/{records_to_process}] "
                        f"ERROR: {record_id}: {e}"
                    )

        # Write results in original order
        print("\nWriting results to output file (in original order)...")
        with open(output_file, "w", encoding="utf-8") as fout:
            for record_index, _ in records_data:
                if record_index in results_dict:
                    fout.write(
                        json.dumps(results_dict[record_index], ensure_ascii=False) + "\n"
                    )

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
    print(f"Elapsed time:    {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    if stats["total"] > 0:
        print(f"Avg per record:  {elapsed / stats['total']:.1f}s")
    print(f"Output file:     {output_file}")
    print("=" * 70)

    return stats


def main() -> None:
    """Main entry point for the batch processing script."""
    parser = argparse.ArgumentParser(
        description="Batch process CoMet metrics via CoMet SDK (MetricsAPIClient)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--metric",
        default="llm_ndcg",
        help="Name of the metric to calculate (default: llm_ndcg)",
    )
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Metric version (default: 1.0.0)",
    )
    parser.add_argument(
        "--flights",
        required=False,
        default=None,
        help="Comma-separated list of flight flags (e.g., 'ndcg-retrieved-good-gain')",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        required=False,
        default=None,
        help="Maximum number of records to process (default: all)",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        required=False,
        default=0,
        help="Record index to start from (0-based, default: 0)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        required=False,
        default=1,
        help="Number of parallel threads for batch processing (default: 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed progress output",
    )

    args = parser.parse_args()

    batch_run(
        input_file=args.input,
        output_file=args.output,
        metric_name=args.metric,
        version=args.version,
        flights=args.flights,
        max_records=args.max_records,
        start_from=args.start_from,
        threads=args.threads,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
