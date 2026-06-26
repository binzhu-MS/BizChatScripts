"""
Combine Seval E2E_metrics.csv from multiple queryset-part folders and
generate a results.md in the same format as Foundry's
run_reasoning_checklist output.

Usage:
    python combine_seval_results.py <seval_dir> [--output <path>]

    seval_dir   Directory containing checklists_queryset_part* subfolders,
                each with a work_metrics/E2E_metrics.csv.

    --output    Path for the output markdown file.
                Default: <seval_dir>/seval_results.md

The script:
1. Discovers all *part* subdirectories containing work_metrics/E2E_metrics.csv.
2. Reads the ``control`` and ``treatment`` rows where segment == "All".
3. Merges scores across parts via weighted average on valid_count:
       combined_score = sum(score_i * valid_i) / sum(valid_i)
4. Emits a markdown table grouped by directory (session subfolder),
   matching the format produced by check_reasoning_checklist.py.
"""

import argparse
import csv
import glob
import os
from collections import defaultdict


def load_e2e_metrics(csv_path):
    """Load E2E_metrics.csv and return rows for control/treatment, segment=All."""
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    result = []
    for r in rows:
        if r["segment"] != "All":
            continue
        if r["exp_name"] not in ("control", "treatment"):
            continue
        result.append(r)
    return result


def parse_metric_field(metric_str):
    """Parse 'reasoning_checklist/path/to/session.json/criteria' into
    (directory, session_name, criteria)."""
    parts = metric_str.split("/")
    if len(parts) < 3:
        return None, None, None

    # parts[0] = "reasoning_checklist", parts[1:-1] = session path, parts[-1] = criteria
    session_path = "/".join(parts[1:-1])  # e.g. "base-sessions/triggering-files.json"
    criteria = parts[-1].strip()

    # Split session_path into directory and filename
    session_path_clean = session_path.replace(".json", "")
    if "/" in session_path_clean:
        directory, session_name = session_path_clean.rsplit("/", 1)
    else:
        directory = ""
        session_name = session_path_clean

    return directory, session_name, criteria


def _find_csv_relpath(part_dir):
    """Search for E2E_metrics.csv in part_dir or one subfolder below it.
    Returns the relative path from part_dir, or None."""
    direct = os.path.join(part_dir, "E2E_metrics.csv")
    if os.path.isfile(direct):
        return "E2E_metrics.csv"
    for sub in sorted(os.listdir(part_dir)):
        candidate = os.path.join(part_dir, sub, "E2E_metrics.csv")
        if os.path.isfile(candidate):
            return os.path.join(sub, "E2E_metrics.csv")
    return None


def combine_parts(seval_dir):
    """Read E2E_metrics.csv from all part folders, combine by weighted average.

    Returns:
        {(directory, session_name, criteria): {arm: {"score": float|None, "valid": int, "total": int}}}
    """
    # Find part folders
    pattern = os.path.join(seval_dir, "*part*")
    part_dirs = sorted(glob.glob(pattern))
    part_dirs = [d for d in part_dirs if os.path.isdir(d)]

    if not part_dirs:
        print(f"ERROR: No *part* directories found in {seval_dir}")
        return {}

    print(f"Found {len(part_dirs)} part directories:")
    for d in part_dirs:
        print(f"  {os.path.basename(d)}")

    # Auto-detect the relative path to E2E_metrics.csv from the first part dir
    csv_relpath = _find_csv_relpath(part_dirs[0])
    if csv_relpath:
        print(f"  CSV location: <part>/{csv_relpath}")
    else:
        print(f"  CSV location: <part>/E2E_metrics.csv (fallback)")

    # Accumulate: key → arm → list of (score, valid_count, total_count)
    accum = defaultdict(lambda: defaultdict(list))

    for part_dir in part_dirs:
        csv_path = os.path.join(part_dir, csv_relpath) if csv_relpath else None
        # If detected relpath doesn't exist here, try searching this dir too
        if not csv_path or not os.path.isfile(csv_path):
            alt = _find_csv_relpath(part_dir)
            csv_path = os.path.join(part_dir, alt) if alt else None
        if not csv_path or not os.path.isfile(csv_path):
            print(f"  WARNING: No work_metrics/E2E_metrics.csv in {os.path.basename(part_dir)}")
            continue

        rows = load_e2e_metrics(csv_path)
        part_name = os.path.basename(part_dir)
        print(f"  {part_name}: {len(rows)} rows (control+treatment, All)")

        for r in rows:
            directory, session_name, criteria = parse_metric_field(r["Metric"])
            if session_name is None:
                continue

            arm = r["exp_name"]
            score_str = r["score"]
            total = int(r["total_count"]) if r["total_count"] else 0
            valid = int(r["valid_count"]) if r["valid_count"] else 0
            score = float(score_str) if score_str else None

            key = (directory, session_name, criteria)
            accum[key][arm].append((score, valid, total))

    # Merge via weighted average
    combined = {}
    for key, arms in accum.items():
        combined[key] = {}
        for arm, entries in arms.items():
            total_valid = sum(v for _, v, _ in entries)
            total_total = sum(t for _, _, t in entries)

            if total_valid > 0:
                # Weighted average: sum(score_i * valid_i) / sum(valid_i)
                weighted_sum = sum(
                    s * v for s, v, _ in entries
                    if s is not None and v > 0
                )
                valid_with_score = sum(
                    v for s, v, _ in entries
                    if s is not None and v > 0
                )
                if valid_with_score > 0:
                    merged_score = weighted_sum / valid_with_score
                else:
                    merged_score = None
            else:
                merged_score = None

            combined[key][arm] = {
                "score": merged_score,
                "valid": total_valid,
                "total": total_total,
            }

    return combined


def generate_results_md(combined, output_path):
    """Write a results.md in the Foundry format."""

    # Group by directory
    by_dir = defaultdict(list)
    for (directory, session_name, criteria), arms in combined.items():
        by_dir[directory].append((session_name, criteria, arms))

    # Sort entries within each directory
    for d in by_dir:
        by_dir[d].sort(key=lambda x: (x[0], x[1]))

    # Determine directory order: "base-sessions" first, then alphabetical
    dir_order = sorted(by_dir.keys(),
                       key=lambda d: ("0" + d if d == "base-sessions" else "1" + d))

    lines = []
    for directory in dir_order:
        lines.append(f"# Directory: {directory}\n")
        lines.append("")
        lines.append("| Name | Criteria | [Seval] control (valid/total rows) | [Seval] treatment (valid/total rows) |")
        lines.append("|---|---|---|---|")

        for session_name, criteria, arms in by_dir[directory]:
            ctrl = arms.get("control", {})
            treat = arms.get("treatment", {})

            def fmt(arm_data):
                s = arm_data.get("score")
                v = arm_data.get("valid", 0)
                t = arm_data.get("total", 0)
                if s is not None:
                    return f"{s:.2f} ({v}/{t} rows)"
                else:
                    return f"N/A ({v}/{t} rows)"

            lines.append(
                f"| {session_name} | {criteria} "
                f"| {fmt(ctrl)} | {fmt(treat)} |"
            )

        lines.append("")

    content = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Count stats
    total_entries = len(combined)
    with_scores = sum(
        1 for arms in combined.values()
        if any(a.get("score") is not None for a in arms.values())
    )
    print(f"\nGenerated: {output_path}")
    print(f"  {total_entries} session×criteria entries")
    print(f"  {with_scores} with at least one score")
    print(f"  Directories: {', '.join(dir_order)}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine Seval E2E_metrics.csv from multiple part "
                    "folders into a single results.md."
    )
    parser.add_argument(
        "seval_dir",
        help="Directory containing *part* subfolders with E2E_metrics.csv",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output markdown file path (default: <seval_dir>/seval_results.md)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.seval_dir):
        print(f"ERROR: Directory not found: {args.seval_dir}")
        return

    output_path = args.output or os.path.join(args.seval_dir, "seval_results.md")

    print(f"Seval directory: {args.seval_dir}")
    combined = combine_parts(args.seval_dir)

    if not combined:
        print("ERROR: No data to combine.")
        return

    generate_results_md(combined, output_path)


if __name__ == "__main__":
    main()
