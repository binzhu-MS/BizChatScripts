# extract_debug.py
#
# Copyright (c) Microsoft Corporation. All rights reserved.

"""
Extract debug data from llm_ndcg output files.

Author: Copilot
Created: 2026-01-08
Description: Extracts the 'debug' field from llm_ndcg output files and saves
             as raw JSON. Use VS Code's format document to view.
"""

import argparse
import json
import sys
from pathlib import Path


def extract_debug(record: dict) -> dict | None:
    """
    Extract and parse the debug field from a record.

    Args:
        record: A single output record.

    Returns:
        Parsed debug data as dict, or None if not present.
    """
    debug_str = record.get("debug")
    if not debug_str:
        return None

    if isinstance(debug_str, dict):
        return debug_str

    try:
        return json.loads(debug_str)
    except json.JSONDecodeError:
        return None


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract debug data from llm_ndcg output files to JSON."
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file path (JSON or JSONL)"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (default: <input>_debug.json)"
    )

    parser.add_argument(
        "--single",
        action="store_true",
        help="Input is a single JSON file instead of JSONL"
    )

    parser.add_argument(
        "--record",
        type=int,
        default=1,
        help="Record number to extract from JSONL (1-based, default: 1)"
    )

    args = parser.parse_args()

    # Load record
    if args.single:
        with open(args.input, "r", encoding="utf-8") as f:
            record = json.load(f)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if idx == args.record:
                    record = json.loads(line.strip())
                    break
            else:
                print(f"Error: Record {args.record} not found", file=sys.stderr)
                sys.exit(1)

    # Extract debug data
    debug_data = extract_debug(record)
    if not debug_data:
        print("Error: No debug data found in record", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    output_path = args.output or str(Path(args.input).with_suffix("")) + "_debug.json"

    # Save raw JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False)

    print(f"Extracted debug data to: {output_path}")


if __name__ == "__main__":
    main()
