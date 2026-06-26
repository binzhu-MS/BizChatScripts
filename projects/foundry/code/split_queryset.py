#!/usr/bin/env python3
"""Split a queryset TSV into 4 files, each with a unique user_id.

Each output row is identical to the original except that
``info["sydney"]["user_id"]`` is set to a per-file value.
Rows are distributed round-robin so each file gets roughly the same
number of queries.

Output files are written alongside the input file::

    queryset.tsv  ->  queryset_part1.tsv
                      queryset_part2.tsv
                      queryset_part3.tsv
                      queryset_part4.tsv

Usage::

    python local/code/split_queryset.py --input local/Results/queryset.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

USER_IDS = [
    "alexkumar@reasoningchecklist-sdf-eval@SyntheticTenant",
    "markfields@reasoningchecklist-sdf-eval@SyntheticTenant",
    "alexmorgan@reasoningchecklist-sdf-eval@SyntheticTenant",
    "laurapark@reasoningchecklist-sdf-eval@SyntheticTenant",
]

NUM_PARTS = len(USER_IDS)


def split_queryset(input_path: Path) -> None:
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    stem = input_path.stem          # e.g. "checklists_queryset"
    suffix = input_path.suffix      # e.g. ".tsv"
    out_dir = input_path.parent

    # Read all lines
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if not lines:
        print("ERROR: Input file is empty")
        sys.exit(1)

    # Open output files
    out_files = []
    out_paths: list[Path] = []
    for i in range(NUM_PARTS):
        p = out_dir / f"{stem}_part{i + 1}{suffix}"
        out_paths.append(p)
        out_files.append(open(p, "w", encoding="utf-8", newline="\n"))

    counts = [0] * NUM_PARTS
    idx = 0

    for line in lines:
        parts = line.split("\t", 1)
        if len(parts) != 2:
            # Malformed line — skip
            continue

        query_col, info_col = parts

        # Parse info JSON, inject user_id
        try:
            info_obj = json.loads(info_col)
        except json.JSONDecodeError:
            # Cannot parse — skip
            continue

        sydney = info_obj.setdefault("sydney", {})
        bucket = idx % NUM_PARTS
        sydney["user_id"] = USER_IDS[bucket]

        info_json = json.dumps(info_obj, ensure_ascii=False)
        out_files[bucket].write(f"{query_col}\t{info_json}\n")
        counts[bucket] += 1
        idx += 1

    for fh in out_files:
        fh.close()

    total = sum(counts)
    print(f"Split {total} rows from {input_path.name} into {NUM_PARTS} files:")
    for i in range(NUM_PARTS):
        print(f"  {out_paths[i].name}: {counts[i]} rows  (user_id: {USER_IDS[i]})")


def main():
    parser = argparse.ArgumentParser(
        description="Split a queryset TSV into 4 files with unique user_ids.",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the input queryset TSV file.",
    )
    args = parser.parse_args()
    split_queryset(Path(args.input))


if __name__ == "__main__":
    main()
