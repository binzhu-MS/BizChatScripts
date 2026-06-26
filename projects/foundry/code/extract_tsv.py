#!/usr/bin/env python3
"""Extract specific rows and columns from a multi-turn conversations TSV file.

Usage:
    python extract_tsv.py <tsv_file> --rows <range> --cols <range> [options]

Row/column ranges:
    Single:  --rows 0        --cols 2
    Range:   --rows 0-3      --cols 1-4
    List:    --rows 0,2,5    --cols 0,3,6
    All:     --rows all      --cols all   (default)

Options:
    --json-indent N   Pretty-print JSON fields with N spaces (default: 2)
    --raw             Print raw values without JSON formatting
    --truncate N      Truncate output to N chars per cell (0 = no limit)
    --out FILE        Write output to file instead of stdout
    --sep SEP         Output separator between cells (default: section dividers)

Examples:
    # Single cell: row 0, col 0 (conversations)
    python extract_tsv.py data.tsv --rows 0 --cols 0

    # All rows, just the metadata column
    python extract_tsv.py data.tsv --cols 1

    # Rows 0-2, columns 0 and 6, truncated
    python extract_tsv.py data.tsv --rows 0-2 --cols 0,6 --truncate 500

    # Raw output to file
    python extract_tsv.py data.tsv --rows 0 --cols 6 --raw --out result.json
"""

import argparse
import csv
import json
import sys
import os


def parse_range(spec: str, max_val: int) -> list[int]:
    """Parse a range spec like '0', '0-3', '0,2,5', or 'all'."""
    if spec.lower() == "all":
        return list(range(max_val))

    indices = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))

    # Validate
    for i in indices:
        if i < 0 or i >= max_val:
            print(f"Error: index {i} out of range (0-{max_val - 1})", file=sys.stderr)
            sys.exit(1)

    return indices


def try_parse_json(value: str):
    """Try to parse a string as JSON; return parsed object or original string."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def format_cell(value: str, json_indent: int | None, truncate: int) -> str:
    """Format a single cell value for display."""
    if json_indent is not None:
        parsed = try_parse_json(value)
        if isinstance(parsed, (dict, list)):
            formatted = json.dumps(parsed, indent=json_indent, ensure_ascii=False)
        else:
            formatted = str(value)
    else:
        formatted = str(value)

    if truncate > 0 and len(formatted) > truncate:
        formatted = formatted[:truncate] + f"\n... [truncated, {len(formatted)} chars total]"

    return formatted


def main():
    parser = argparse.ArgumentParser(
        description="Extract specific rows/columns from a multi-turn TSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("tsv_file", help="Path to the TSV file")
    parser.add_argument("--rows", default="all", help="Row range: 0, 0-3, 0,2,5, or all (default: all)")
    parser.add_argument("--cols", default="all", help="Column range: 0, 0-3, 0,2,5, or all (default: all)")
    parser.add_argument("--json-indent", type=int, default=2, help="JSON indent spaces (default: 2)")
    parser.add_argument("--raw", action="store_true", help="Print raw values, no JSON formatting")
    parser.add_argument("--truncate", type=int, default=0, help="Truncate each cell to N chars (0 = no limit)")
    parser.add_argument("--out", help="Output file (default: stdout)")
    parser.add_argument("--sep", default=None, help="Custom separator between cells")

    args = parser.parse_args()

    # Increase field size limit for large JSON cells
    csv.field_size_limit(10 * 1024 * 1024)

    # Read TSV
    tsv_path = args.tsv_file
    if not os.path.isabs(tsv_path):
        tsv_path = os.path.abspath(tsv_path)

    if not os.path.exists(tsv_path):
        print(f"Error: file not found: {tsv_path}", file=sys.stderr)
        sys.exit(1)

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        all_rows = list(reader)

    if not all_rows:
        print("Error: TSV file is empty", file=sys.stderr)
        sys.exit(1)

    num_rows = len(all_rows)
    num_cols = max(len(r) for r in all_rows)

    # Parse ranges
    row_indices = parse_range(args.rows, num_rows)
    col_indices = parse_range(args.cols, num_cols)

    json_indent = None if args.raw else args.json_indent

    # Build output
    out_lines = []

    # Header summary
    out_lines.append(f"# Source: {os.path.basename(tsv_path)}")
    out_lines.append(f"# Total: {num_rows} rows x {num_cols} cols")
    out_lines.append(f"# Extracting: rows {args.rows}, cols {args.cols}")
    out_lines.append(f"#   → {len(row_indices)} row(s) x {len(col_indices)} col(s) = {len(row_indices) * len(col_indices)} cell(s)")
    out_lines.append("")

    single_cell = len(row_indices) == 1 and len(col_indices) == 1

    for ri in row_indices:
        row = all_rows[ri]
        for ci in col_indices:
            if ci >= len(row):
                cell_value = ""
            else:
                cell_value = row[ci]

            formatted = format_cell(cell_value, json_indent, args.truncate)

            if single_cell:
                # No header for single cell extraction
                out_lines.append(formatted)
            else:
                if args.sep:
                    out_lines.append(args.sep)
                else:
                    out_lines.append(f"{'='*60}")
                    out_lines.append(f"  Row {ri}, Col {ci}  (len={len(cell_value)} chars)")
                    out_lines.append(f"{'='*60}")
                out_lines.append(formatted)
                out_lines.append("")

    output = "\n".join(out_lines)

    # Write output
    if args.out:
        out_path = args.out
        if not os.path.isabs(out_path):
            out_path = os.path.abspath(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Written to {out_path} ({len(output)} chars)", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
