#!/usr/bin/env python3
"""
Extract LLM usage info from a reference comparison MD file and apply it to a target MD file.

Usage:
    python apply_llm_column.py <reference_md_with_llm> <target_md_without_llm> [-o OUTPUT]

The reference file must have an "LLM" column (Yes/No) in its comparison tables.
The script extracts the criteria→LLM mapping, then injects the LLM column into
the target file's tables (matching by criteria name).

If -o is not specified, writes to <target>_with_llm.md alongside the target file.
"""

import argparse
import re
import sys
from pathlib import Path


def extract_llm_mapping(ref_path: str) -> dict[str, str]:
    """Parse the reference MD and return {criteria_name: 'Yes'|'No'|'?'}."""
    mapping: dict[str, str] = {}
    with open(ref_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            # Match table rows: | Name | Criteria | LLM | ... |
            # The LLM column is the 3rd pipe-delimited field.
            # Rows may have an empty Name column (continuation rows for same session).
            if not line.startswith("|"):
                continue
            cols = [c.strip() for c in line.split("|")]
            # split on | gives ['', col1, col2, ..., ''] for well-formed rows
            if len(cols) < 5:
                continue
            # Skip header / separator rows (but NOT continuation rows with empty Name)
            if cols[1] == "Name" or cols[2] in ("", "Criteria") or set(cols[3]) <= {"-", " ", ""}:
                continue
            criteria = cols[2]
            llm_val = cols[3]
            if llm_val in ("Yes", "No", "?"):
                mapping[criteria] = llm_val
    return mapping


def build_llm_criteria_section(mapping: dict[str, str]) -> str:
    """Build the summary section listing criteria that use LLM."""
    llm_criteria = sorted(c for c, v in mapping.items() if v == "Yes")
    if not llm_criteria:
        return ""
    lines = [
        "",
        "> **LLM column** -- `Yes` means the criterion uses an LLM judge during evaluation "
        "(in both control and treatment arms); `No` means script-only deterministic scoring. "
        "`?` means the criterion was not found in any session JSON.",
        "",
        "### Criteria that use LLM during evaluation",
        "",
    ]
    for c in llm_criteria:
        lines.append(f"- `{c}`")
    lines.append("")
    return "\n".join(lines)


def apply_llm_to_target(target_path: str, mapping: dict[str, str], output_path: str) -> None:
    """Read the target MD, insert LLM column into tables, and write to output."""
    with open(target_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result: list[str] = []
    i = 0
    side_by_side_inserted = False

    while i < len(lines):
        line = lines[i].rstrip("\n")

        # Insert the LLM summary block right after "## Side-by-Side Comparison"
        if line.strip() == "## Side-by-Side Comparison" and not side_by_side_inserted:
            result.append(line)
            result.append("")
            result.append(build_llm_criteria_section(mapping).strip())
            side_by_side_inserted = True
            i += 1
            continue

        # Detect table header rows: | Name | Criteria | <data cols> |
        if line.startswith("|") and "Criteria" in line and "Name" in line:
            cols = [c.strip() for c in line.split("|")]
            # Insert "LLM" after "Criteria" column (index 2)
            criteria_idx = None
            for idx, c in enumerate(cols):
                if c == "Criteria":
                    criteria_idx = idx
                    break

            if criteria_idx is not None:
                # Header row
                cols.insert(criteria_idx + 1, "LLM")
                result.append("| " + " | ".join(cols[1:-1]) + " |")
                i += 1

                # Separator row (next line should be |---|---|...)
                if i < len(lines) and lines[i].strip().startswith("|"):
                    sep_cols = [c.strip() for c in lines[i].rstrip("\n").split("|")]
                    sep_cols.insert(criteria_idx + 1, "---")
                    result.append("| " + " | ".join(sep_cols[1:-1]) + " |")
                    i += 1

                # Data rows
                while i < len(lines):
                    dline = lines[i].rstrip("\n")
                    if not dline.startswith("|"):
                        break
                    dcols = [c.strip() for c in dline.split("|")]
                    if len(dcols) < 4:
                        break
                    # Check if it's a separator row or sub-header (shouldn't be but guard)
                    if all(set(c) <= {"-", " ", ""} for c in dcols[1:-1]):
                        break

                    criteria_name = dcols[criteria_idx]
                    # For continuation rows (empty Name), criteria is still in the criteria column
                    llm_val = mapping.get(criteria_name, "?")
                    dcols.insert(criteria_idx + 1, llm_val)
                    result.append("| " + " | ".join(dcols[1:-1]) + " |")
                    i += 1
                continue
            else:
                result.append(line)
                i += 1
                continue
        else:
            result.append(line)
            i += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result))
        if not result[-1].endswith("\n"):
            f.write("\n")

    print(f"Wrote {output_path}")
    print(f"  LLM mapping: {sum(1 for v in mapping.values() if v == 'Yes')} Yes, "
          f"{sum(1 for v in mapping.values() if v == 'No')} No, "
          f"{sum(1 for v in mapping.values() if v == '?')} unknown")


def main():
    parser = argparse.ArgumentParser(
        description="Extract LLM column from a reference comparison MD and apply it to a target MD."
    )
    parser.add_argument("reference", help="Path to the reference MD file (with LLM column)")
    parser.add_argument("target", help="Path to the target MD file (without LLM column)")
    parser.add_argument("-o", "--output", help="Output path (default: <target>_with_llm.md)")
    args = parser.parse_args()

    ref_path = args.reference
    target_path = args.target
    output_path = args.output
    if not output_path:
        p = Path(target_path)
        output_path = str(p.with_stem(p.stem + "_with_llm"))

    if not Path(ref_path).exists():
        print(f"Error: reference file not found: {ref_path}", file=sys.stderr)
        sys.exit(1)
    if not Path(target_path).exists():
        print(f"Error: target file not found: {target_path}", file=sys.stderr)
        sys.exit(1)

    mapping = extract_llm_mapping(ref_path)
    if not mapping:
        print("Warning: no LLM mapping extracted from reference file.", file=sys.stderr)

    apply_llm_to_target(target_path, mapping, output_path)


if __name__ == "__main__":
    main()
