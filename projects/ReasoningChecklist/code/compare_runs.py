"""
Compare two Foundry evaluation runs by parsing results.md from each run folder.
Matches rows by (directory, name, criteria, prompt) and produces a side-by-side
markdown comparison with summary, top regressions/improvements, and per-directory tables.

Usage:
    python compare_runs.py --run1 <run1_folder> --run2 <run2_folder> [--output comparison.md]

    --output defaults to comparison.md if omitted.
"""

import os
import re
import argparse
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_score_cell(cell):
    """Parse a score cell like '96.67 (1710 rows)' into (score, row_count).
    Returns (None, None) if unparseable.
    """
    cell = cell.strip()
    if not cell or cell == "0.00 (0 rows)":
        return 0.0, 0
    m = re.match(r'^([\d.]+)\s*\((\d+)\s*rows?\)', cell)
    if m:
        return float(m.group(1)), int(m.group(2))
    # Try just a number
    try:
        return float(cell), 0
    except ValueError:
        return None, None


def parse_results_md(results_md_path):
    """Parse a results.md file.
    
    Returns a list of dicts, each with keys:
        directory, name, criteria, prompt, score, rows
    """
    entries = []
    if not os.path.exists(results_md_path):
        print(f"Error: {results_md_path} not found")
        return entries

    with open(results_md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_dir = "unknown"
    headers = []

    for line in lines:
        line = line.rstrip('\n')

        # Directory header: # Directory: base-sessions
        dir_match = re.match(r'^#\s+Directory:\s+(.+)', line)
        if dir_match:
            current_dir = dir_match.group(1).strip()
            headers = []
            continue

        # Table header row: | Name | Criteria | Link | prompt1 | prompt2 |
        if line.startswith('|') and 'Name' in line and 'Criteria' in line:
            headers = [h.strip() for h in line.split('|')]
            # Remove empty strings from leading/trailing pipes
            headers = [h for h in headers if h]
            continue

        # Separator row: |---|---|---|---|---|
        if line.startswith('|') and set(line.replace('|', '').replace('-', '').strip()) == set():
            continue

        # Data row
        if line.startswith('|') and headers:
            cells = [c.strip() for c in line.split('|')]
            cells = [c for c in cells if c is not None]
            # Remove empty strings from leading/trailing pipes
            if cells and cells[0] == '':
                cells = cells[1:]
            if cells and cells[-1] == '':
                cells = cells[:-1]

            if len(cells) < len(headers):
                continue

            row = {}
            for i, h in enumerate(headers):
                if i < len(cells):
                    row[h] = cells[i]

            name = row.get('Name', 'Unknown')
            criteria = row.get('Criteria', 'Unknown')

            # Prompt columns are all columns after Name, Criteria, Link
            for h in headers:
                if h in ('Name', 'Criteria', 'Link'):
                    continue
                cell_value = row.get(h, '')
                score, rows = parse_score_cell(cell_value)
                if score is not None:
                    entries.append({
                        'directory': current_dir,
                        'name': name,
                        'criteria': criteria,
                        'prompt': h,
                        'score': score,
                        'rows': rows,
                    })

    return entries


def compare_runs(run1_folder, run2_folder, output_path):
    """Compare two runs by parsing their results.md files."""
    lines = []

    def out(text=""):
        lines.append(text)

    run1_md = os.path.join(run1_folder, "results.md")
    run2_md = os.path.join(run2_folder, "results.md")

    # Use relative paths for display (normalize to forward slashes)
    run1_display = os.path.relpath(run1_folder, SCRIPT_DIR).replace('\\', '/')
    run2_display = os.path.relpath(run2_folder, SCRIPT_DIR).replace('\\', '/')

    print(f"Loading run 1: {run1_display}")
    entries1 = parse_results_md(run1_md)
    print(f"  {len(entries1)} score entries")

    print(f"Loading run 2: {run2_display}")
    entries2 = parse_results_md(run2_md)
    print(f"  {len(entries2)} score entries")

    # Build lookup dicts keyed by (directory, name, criteria, prompt)
    def build_lookup(entries):
        lookup = {}
        for e in entries:
            key = (e['directory'], e['name'], e['criteria'], e['prompt'])
            lookup[key] = (e['score'], e['rows'])
        return lookup

    scores1 = build_lookup(entries1)
    scores2 = build_lookup(entries2)

    matched_keys = set(scores1.keys()) & set(scores2.keys())
    run1_only = set(scores1.keys()) - set(scores2.keys())
    run2_only = set(scores2.keys()) - set(scores1.keys())

    diffs = []
    identical = 0
    different = 0
    for key in sorted(matched_keys):
        s1, c1 = scores1[key]
        s2, c2 = scores2[key]
        diff = s2 - s1
        directory, session_name, criteria_name, prompt_title = key
        diffs.append({
            "directory": directory,
            "name": session_name,
            "criteria": criteria_name,
            "prompt": prompt_title,
            "run1_score": s1,
            "run1_rows": c1,
            "run2_score": s2,
            "run2_rows": c2,
            "diff": diff,
        })
        if abs(diff) < 0.005:
            identical += 1
        else:
            different += 1

    # Summary
    out("")
    out("# Summary")
    out("")
    out(f"- Total matched score entries: {len(matched_keys)}")
    out(f"  - Score entries with identical scores: {identical}")
    out(f"  - Score entries with different scores: {different}")
    if diffs:
        abs_diffs = [abs(d["diff"]) for d in diffs if abs(d["diff"]) >= 0.005]
        if abs_diffs:
            out(f"    - Max absolute diff: {max(abs_diffs):.2f}")
            out(f"    - Mean absolute diff: {sum(abs_diffs) / len(abs_diffs):.2f}")
    out(f"- Total mismatched score entries: {len(run1_only) + len(run2_only)}")
    out(f"  - Entries in run1 only: {len(run1_only)}")
    out(f"  - Entries in run2 only: {len(run2_only)}")

    # Top regressions / improvements
    if diffs:
        sorted_by_diff = sorted(diffs, key=lambda d: d["diff"])
        regressions = [d for d in sorted_by_diff if d["diff"] < -0.005]
        improvements = [d for d in reversed(sorted_by_diff) if d["diff"] > 0.005]

        if regressions:
            out("")
            out("## Top 10 Regressions (Run2 lower)")
            out("| Name | Criteria | Prompt | Run1 | Run2 | Diff |")
            out("|---|---|---|---|---|---|")
            for d in regressions[:10]:
                out(f"| {d['name']} | {d['criteria']} | {d['prompt']} | {d['run1_score']:.2f} ({d['run1_rows']}) | {d['run2_score']:.2f} ({d['run2_rows']}) | {d['diff']:.2f} |")

        if improvements:
            out("")
            out("## Top 10 Improvements (Run2 higher)")
            out("| Name | Criteria | Prompt | Run1 | Run2 | Diff |")
            out("|---|---|---|---|---|---|")
            for d in improvements[:10]:
                out(f"| {d['name']} | {d['criteria']} | {d['prompt']} | {d['run1_score']:.2f} ({d['run1_rows']}) | {d['run2_score']:.2f} ({d['run2_rows']}) | +{d['diff']:.2f} |")

    # Per-directory breakdown
    by_dir = defaultdict(list)
    for d in diffs:
        by_dir[d["directory"]].append(d)

    for directory in sorted(by_dir.keys()):
        out("")
        out(f"# Directory: {directory}")
        out("")
        out("| Name | Criteria | Prompt | Run1 Score | Run1 Rows | Run2 Score | Run2 Rows | Score Diff |")
        out("|---|---|---|---|---|---|---|---|")
        dir_diffs = sorted(by_dir[directory], key=lambda d: (d["name"], d["criteria"], d["prompt"]))
        for d in dir_diffs:
            sign = "+" if d["diff"] >= 0 else ""
            out(f"| {d['name']} | {d['criteria']} | {d['prompt']} | {d['run1_score']:.2f} | {d['run1_rows']} | {d['run2_score']:.2f} | {d['run2_rows']} | {sign}{d['diff']:.2f} |")

    # Write output
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nWritten to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare two Foundry evaluation runs")
    parser.add_argument("--run1", required=True, help="Path to first run results folder")
    parser.add_argument("--run2", required=True, help="Path to second run results folder")
    parser.add_argument("--output", help="Output markdown file path", default="comparison.md")
    args = parser.parse_args()

    compare_runs(args.run1, args.run2, args.output)


if __name__ == "__main__":
    main()
