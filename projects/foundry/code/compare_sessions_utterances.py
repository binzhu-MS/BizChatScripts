"""
Compare two versions of session files and report all dataItem differences.

Walks all *.json session files under --original-dir, finds the matching
file in --revised-dir, and compares every dataItem field (input, pinned,
index, etc.) between the two versions at both locations:
  - top-level dataItems (deprecated)
  - sessionInputs[0].dataItems (active, used by Foundry backend)

For each file with changes at either location, generates a markdown report
listing all differences. Prints overall statistics to console.

Usage:
    python compare_sessions_utterances.py --original-dir <DIR> --revised-dir <DIR> --output-dir <DIR>

Arguments:
    --original-dir  Root directory containing the baseline session JSON files.
    --revised-dir   Root directory containing the revised session JSON files.
                    Must have the same subdirectory structure as --original-dir.
    --output-dir    Directory where per-session diff report markdown files
                    are written. Created automatically if it does not exist.

Examples:
    # Compare original sessions against deduped output
    python compare_sessions_utterances.py \\
        --original-dir sessions \\
        --revised-dir local/sessions_deduped \\
        --output-dir local/docs/session-diffs

    # Compare two arbitrary versions
    python compare_sessions_utterances.py \\
        --original-dir local/sessions_v1 \\
        --revised-dir local/sessions_v2 \\
        --output-dir local/docs/v1-v2-diffs

Output:
    - Per-session markdown diff reports in --output-dir
    - Console statistics: files compared, files changed, items changed,
      field-level change distribution
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_session(path: str) -> dict:
    """Load a session JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_input(input_field) -> dict | None:
    """Parse input field to a dict for comparison."""
    if isinstance(input_field, str):
        try:
            return json.loads(input_field)
        except (json.JSONDecodeError, TypeError):
            return None
    elif isinstance(input_field, dict):
        return input_field
    return None


def canonical(obj) -> str:
    """Canonical JSON string for comparison."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def diff_items(orig_item: dict | None, rev_item: dict | None) -> list[dict]:
    """Compare two dataItem dicts and return a list of field-level diffs.

    Each diff is {field, original, revised}.
    """
    diffs = []
    if orig_item is None and rev_item is None:
        return diffs

    orig_keys = set(orig_item.keys()) if orig_item else set()
    rev_keys = set(rev_item.keys()) if rev_item else set()
    all_keys = sorted(orig_keys | rev_keys)

    for key in all_keys:
        orig_val = orig_item.get(key) if orig_item else None
        rev_val = rev_item.get(key) if rev_item else None

        # For "input" field, compare parsed JSON to ignore whitespace diffs
        if key == "input":
            orig_parsed = normalize_input(orig_val)
            rev_parsed = normalize_input(rev_val)
            if canonical(orig_parsed) == canonical(rev_parsed):
                continue
            # Find which sub-fields changed
            if orig_parsed and rev_parsed:
                sub_keys = sorted(
                    set(orig_parsed.keys()) | set(rev_parsed.keys())
                )
                for sk in sub_keys:
                    ov = orig_parsed.get(sk)
                    rv = rev_parsed.get(sk)
                    if ov != rv:
                        diffs.append({
                            "field": f"input.{sk}",
                            "original": ov,
                            "revised": rv,
                        })
            else:
                diffs.append({
                    "field": "input",
                    "original": orig_val,
                    "revised": rev_val,
                })
        else:
            if orig_val != rev_val:
                diffs.append({
                    "field": key,
                    "original": orig_val,
                    "revised": rev_val,
                })

    return diffs


def format_value(val) -> str:
    """Format a value for display in markdown."""
    if val is None:
        return "*(missing)*"
    if isinstance(val, str):
        return val.replace("|", "\\|").replace("\n", "↵")
    return str(val).replace("|", "\\|")


def get_si_data_items(data: dict) -> list[dict] | None:
    """Return sessionInputs[0].dataItems if it exists, else None."""
    si = data.get("sessionInputs", [])
    if si and isinstance(si, list) and si[0].get("dataItems"):
        return si[0]["dataItems"]
    return None


def compare_item_lists(
    orig_items: list[dict],
    rev_items: list[dict],
) -> tuple[list[tuple[int, str, list[dict]]], dict]:
    """Compare two dataItems lists.

    Returns (item_diffs, stats) where:
        item_diffs: list of (index, change_type, field_diffs)
        stats: dict with items_compared/changed/added/removed, field_counts
    """
    max_idx = max(len(orig_items), len(rev_items)) if (orig_items or rev_items) else 0
    item_diffs: list[tuple[int, str, list[dict]]] = []
    stats: dict = {
        "items_compared": 0,
        "items_changed": 0,
        "items_added": 0,
        "items_removed": 0,
        "field_counts": defaultdict(int),
    }

    for idx in range(max_idx):
        orig_item = orig_items[idx] if idx < len(orig_items) else None
        rev_item = rev_items[idx] if idx < len(rev_items) else None
        stats["items_compared"] += 1

        if orig_item is None and rev_item is not None:
            stats["items_added"] += 1
            item_diffs.append((idx, "added", []))
        elif orig_item is not None and rev_item is None:
            stats["items_removed"] += 1
            item_diffs.append((idx, "removed", []))
        else:
            field_diffs = diff_items(orig_item, rev_item)
            if field_diffs:
                stats["items_changed"] += 1
                for fd in field_diffs:
                    stats["field_counts"][fd["field"]] += 1
                item_diffs.append((idx, "modified", field_diffs))

    return item_diffs, stats


def format_diff_section(
    item_diffs: list[tuple[int, str, list[dict]]],
    orig_items: list[dict],
    rev_items: list[dict],
    section_title: str,
) -> list[str]:
    """Render a markdown section for one dataItems location."""
    if not item_diffs:
        return []

    lines: list[str] = []
    lines.append(f"## {section_title}\n")
    lines.append(f"**Original items:** {len(orig_items)}  ")
    lines.append(f"**Revised items:** {len(rev_items)}  ")
    lines.append(f"**Items with changes:** {len(item_diffs)}\n")

    modified = [(i, d) for i, t, d in item_diffs if t == "modified"]
    added = [i for i, t, _ in item_diffs if t == "added"]
    removed = [i for i, t, _ in item_diffs if t == "removed"]

    if modified:
        lines.append(f"### Modified Items ({len(modified)})\n")
        for idx, field_diffs in modified:
            orig_item = orig_items[idx]
            utt = None
            inp = normalize_input(orig_item.get("input"))
            if inp:
                utt = inp.get("utterance", "")
            header = f"#### Item {idx}"
            if utt:
                header += f": {utt}"
            lines.append(header + "\n")

            lines.append("| Field | Original | Revised |")
            lines.append("|-------|----------|---------|")
            for fd in field_diffs:
                field = fd["field"]
                orig_disp = format_value(fd["original"])
                rev_disp = format_value(fd["revised"])
                lines.append(f"| `{field}` | {orig_disp} | {rev_disp} |")
            lines.append("")

    if added:
        lines.append(f"### Added Items ({len(added)})\n")
        for idx in added:
            rev_item = rev_items[idx]
            inp = normalize_input(rev_item.get("input"))
            utt = inp.get("utterance", "") if inp else ""
            lines.append(f"- **[{idx}]** {utt}")
        lines.append("")

    if removed:
        lines.append(f"### Removed Items ({len(removed)})\n")
        for idx in removed:
            orig_item = orig_items[idx]
            inp = normalize_input(orig_item.get("input"))
            utt = inp.get("utterance", "") if inp else ""
            lines.append(f"- **[{idx}]** {utt}")
        lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Compare two versions of session files and report "
                    "all dataItem differences"
    )
    parser.add_argument("--original-dir", required=True,
                        help="Root directory of baseline session files")
    parser.add_argument("--revised-dir", required=True,
                        help="Root directory of revised session files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for per-session diff reports")
    args = parser.parse_args()

    original_dir = args.original_dir
    revised_dir = args.revised_dir
    output_dir = args.output_dir

    for d in (original_dir, revised_dir):
        if not os.path.isdir(d):
            print(f"ERROR: Directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Collect all JSON files from original dir
    json_files: list[str] = []
    for root, _dirs, files in os.walk(original_dir):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            filepath = os.path.join(root, fn)
            rel = os.path.relpath(filepath, original_dir).replace("\\", "/")
            json_files.append(rel)

    # --- Stats (tracked per location) ---
    total_files = 0
    files_changed = 0
    files_unchanged = 0
    files_missing = 0
    reports_generated = 0

    # Per-location aggregate stats
    loc_stats: dict[str, dict] = {
        "top": {"items_compared": 0, "items_changed": 0,
                "items_added": 0, "items_removed": 0,
                "field_counts": defaultdict(int)},
        "si":  {"items_compared": 0, "items_changed": 0,
                "items_added": 0, "items_removed": 0,
                "field_counts": defaultdict(int)},
    }

    for rel in json_files:
        total_files += 1
        orig_path = os.path.join(original_dir, rel)
        rev_path = os.path.join(revised_dir, rel)

        if not os.path.exists(rev_path):
            files_missing += 1
            continue

        orig_data = load_session(orig_path)
        rev_data = load_session(rev_path)

        # --- Top-level dataItems ---
        orig_top = orig_data.get("dataItems", [])
        rev_top = rev_data.get("dataItems", [])
        top_diffs, top_st = compare_item_lists(orig_top, rev_top)

        # --- sessionInputs[0].dataItems ---
        orig_si = get_si_data_items(orig_data)
        rev_si = get_si_data_items(rev_data)
        si_diffs: list[tuple[int, str, list[dict]]] = []
        si_st: dict = {"items_compared": 0, "items_changed": 0,
                       "items_added": 0, "items_removed": 0,
                       "field_counts": defaultdict(int)}
        has_si = orig_si is not None or rev_si is not None
        if has_si:
            si_diffs, si_st = compare_item_lists(
                orig_si or [], rev_si or [],
            )

        # Accumulate global stats
        for key in ("items_compared", "items_changed",
                    "items_added", "items_removed"):
            loc_stats["top"][key] += top_st[key]
            loc_stats["si"][key] += si_st[key]
        for f, c in top_st["field_counts"].items():
            loc_stats["top"]["field_counts"][f] += c
        for f, c in si_st["field_counts"].items():
            loc_stats["si"]["field_counts"][f] += c

        if not top_diffs and not si_diffs:
            files_unchanged += 1
            continue

        files_changed += 1

        # --- Generate report ---
        lines: list[str] = []
        lines.append(f"# Diff Report: `{rel}`\n")

        # Top-level section
        lines.extend(
            format_diff_section(top_diffs, orig_top, rev_top,
                                "Top-Level dataItems")
        )

        # sessionInputs section
        if has_si:
            lines.extend(
                format_diff_section(si_diffs, orig_si or [], rev_si or [],
                                    "sessionInputs[0].dataItems")
            )
        else:
            lines.append("## sessionInputs[0].dataItems\n")
            lines.append("*(not present in either file)*\n")

        report = "\n".join(lines)
        out_name = rel.replace("/", "_").replace(".json", ".md")
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        reports_generated += 1
        n_top = len(top_diffs)
        n_si = len(si_diffs)
        print(f"  {out_name}  (top: {n_top}, SI: {n_si})")

    # --- Overall Statistics ---
    print(f"\n{'=' * 60}")
    print(f"Session Comparison Statistics")
    print(f"{'=' * 60}")
    print(f"Original dir:         {original_dir}")
    print(f"Revised dir:          {revised_dir}")
    print(f"Total files:          {total_files}")
    print(f"Files changed:        {files_changed}")
    print(f"Files unchanged:      {files_unchanged}")
    print(f"Files missing:        {files_missing}")
    print(f"Reports generated:    {reports_generated} (in {output_dir}/)")

    for label, key in [("Top-Level dataItems", "top"),
                       ("sessionInputs[0].dataItems", "si")]:
        s = loc_stats[key]
        print(f"\n--- {label} ---")
        print(f"  Items compared:   {s['items_compared']}")
        print(f"  Items modified:   {s['items_changed']}")
        print(f"  Items added:      {s['items_added']}")
        print(f"  Items removed:    {s['items_removed']}")
        fc = s["field_counts"]
        if fc:
            print(f"  Field changes:")
            for field, count in sorted(fc.items(), key=lambda x: -x[1]):
                print(f"    {field:<25}  {count:>6}")


if __name__ == "__main__":
    main()
