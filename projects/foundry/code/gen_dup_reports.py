"""Generate per-session duplicate report markdown files.

For each session file that contains duplicates, creates a markdown file
listing:
  - Type 1 (exact input duplicates): the input and how many copies exist.
  - Type 2 (same utterance, different input): all variant inputs grouped
    under the deduplicated utterance.

Usage:
    python local/code/gen_dup_reports.py --input-dir <DIR> --output-dir <DIR>

Arguments:
    --input-dir   Root directory containing session JSON files (searched
                  recursively). Default: sessions
    --output-dir  Directory where markdown report files are written.
                  Default: local/docs/dup-reports

Examples:
    # Scan original sessions
    python local/code/gen_dup_reports.py \\
        --input-dir sessions --output-dir local/docs/dup-reports

    # Verify deduped sessions have no duplicates
    python local/code/gen_dup_reports.py \\
        --input-dir local/sessions_deduped --output-dir local/docs/dup-reports-deduped
"""

import argparse
import json
import os
from collections import defaultdict


def load_items(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("dataItems", data.get("items", []))


def extract_utterance(inp_obj):
    """Extract utterance text from an input object."""
    if isinstance(inp_obj, dict):
        if "parts" in inp_obj:
            for p in inp_obj["parts"]:
                if isinstance(p, dict) and p.get("author") == "user":
                    return p.get("text", "")
        elif "utterance" in inp_obj:
            return inp_obj["utterance"]
    return None


def parse_input(raw_input):
    """Parse the input field (may be str or dict)."""
    if isinstance(raw_input, str):
        return json.loads(raw_input)
    return raw_input


def format_input_block(inp_obj):
    """Format an input object as a fenced JSON block."""
    return "```json\n" + json.dumps(inp_obj, indent=2, ensure_ascii=False) + "\n```"


def generate_report(session_rel, items):
    """Return (markdown_string, stats_dict) for one session.

    Returns (None, stats) if no dups found. stats is always returned.
    """
    # Parse all inputs and extract utterances
    parsed = []
    for item in items:
        raw = item.get("input", {})
        try:
            inp_obj = parse_input(raw)
        except Exception:
            continue
        utt = extract_utterance(inp_obj)
        if utt is None:
            continue
        input_str = json.dumps(inp_obj, sort_keys=True, ensure_ascii=False)
        parsed.append((utt, input_str, inp_obj))

    # --- Type 1: exact input duplicates ---
    input_counts = defaultdict(int)
    input_example = {}
    for utt, input_str, inp_obj in parsed:
        input_counts[input_str] += 1
        if input_str not in input_example:
            input_example[input_str] = (utt, inp_obj)

    type1 = {k: v for k, v in input_counts.items() if v > 1}

    # --- Type 2: same utterance, different input ---
    utt_inputs = defaultdict(dict)  # utt -> {input_str: inp_obj}
    for utt, input_str, inp_obj in parsed:
        utt_inputs[utt][input_str] = inp_obj

    type2 = {utt: variants for utt, variants in utt_inputs.items() if len(variants) > 1}

    # Build stats
    type1_extra = sum(v - 1 for v in type1.values())  # extra copies
    type1_copy_counts = [v for v in type1.values()]    # copy count per dup input
    type2_variant_counts = [len(v) for v in type2.values()]  # variants per utterance
    stats = {
        "total_items": len(parsed),
        "type1_groups": len(type1),
        "type1_extra": type1_extra,
        "type1_copy_counts": type1_copy_counts,
        "type2_groups": len(type2),
        "type2_extra": sum(c - 1 for c in type2_variant_counts),
        "type2_variant_counts": type2_variant_counts,
    }

    if not type1 and not type2:
        return None, stats

    lines = []
    lines.append(f"# Duplicate Report: `{session_rel}`\n")
    lines.append(f"**Total dataItems:** {len(items)}  ")
    lines.append(f"**Type 1 duplicated inputs:** {len(type1)}  ")
    lines.append(f"**Type 2 variant utterances:** {len(type2)}\n")

    # Type 1 section
    if type1:
        lines.append("## Type 1: Exact Input Duplicates\n")
        lines.append(
            "Each entry below is an input that appears multiple times "
            "(identical copies). Only one copy is needed.\n"
        )
        # Sort by count descending, then by utterance
        sorted_type1 = sorted(
            type1.items(), key=lambda kv: (-kv[1], input_example[kv[0]][0])
        )
        for input_str, count in sorted_type1:
            utt, inp_obj = input_example[input_str]
            lines.append(f"### \"{utt}\" — **{count} copies**\n")
            lines.append(format_input_block(inp_obj))
            lines.append("")

    # Type 2 section
    if type2:
        lines.append("## Type 2: Same Utterance, Different Input\n")
        lines.append(
            "Each group below shares the same utterance text but has different "
            "input fields. These need utterance rewording to avoid SEVAL dedup.\n"
        )
        # Sort by number of variants descending
        sorted_type2 = sorted(type2.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        for utt, variants in sorted_type2:
            lines.append(f"### \"{utt}\" — **{len(variants)} variants**\n")
            for i, (input_str, inp_obj) in enumerate(variants.items(), 1):
                lines.append(f"**Variant {i}:**\n")
                lines.append(format_input_block(inp_obj))
                lines.append("")

    return "\n".join(lines), stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-session duplicate report markdown files"
    )
    parser.add_argument("--input-dir", default="sessions",
                        help="Root directory containing session JSON files "
                             "(default: sessions)")
    parser.add_argument("--output-dir", default="local/docs/dup-reports",
                        help="Directory for output markdown reports "
                             "(default: local/docs/dup-reports)")
    args = parser.parse_args()

    sessions_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    generated = []
    clean_files = 0
    total_files = 0
    total_items = 0
    total_type1_groups = 0
    total_type1_extra = 0
    total_type2_groups = 0
    total_type2_extra = 0
    all_type1_copy_counts: list[int] = []
    all_type2_variant_counts: list[int] = []

    for root, dirs, files in os.walk(sessions_dir):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, sessions_dir).replace("\\", "/")
            items = load_items(path)
            total_files += 1
            report, stats = generate_report(rel, items)

            total_items += stats["total_items"]
            total_type1_groups += stats["type1_groups"]
            total_type1_extra += stats["type1_extra"]
            total_type2_groups += stats["type2_groups"]
            total_type2_extra += stats["type2_extra"]
            all_type1_copy_counts.extend(stats["type1_copy_counts"])
            all_type2_variant_counts.extend(stats["type2_variant_counts"])

            if report is None:
                clean_files += 1
                continue

            # Create output filename from relative path
            out_name = rel.replace("/", "_").replace(".json", ".md")
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report)
            generated.append((rel, out_name))
            print(f"  {out_name}")

    print(f"\nGenerated {len(generated)} report(s) in {output_dir}/")
    print(f"Clean files (no dups): {clean_files}")

    # --- Overall statistics ---
    files_with_dups = total_files - clean_files
    print(f"\n{'=' * 50}")
    print(f"Overall Duplication Statistics")
    print(f"{'=' * 50}")
    print(f"Total session files:     {total_files}")
    print(f"Files with duplicates:   {files_with_dups}")
    print(f"Files without dups:      {clean_files}")
    print(f"Total dataItems:         {total_items}")

    print(f"\n--- Type 1: Exact Input Duplicates ---")
    print(f"Duplicated inputs:       {total_type1_groups}")
    print(f"Extra copies:            {total_type1_extra}")
    if all_type1_copy_counts:
        t1_dist: dict[int, int] = defaultdict(int)
        for c in all_type1_copy_counts:
            t1_dist[c] += 1
        print(f"\n  {'Copies':>8}  {'Inputs':>8}")
        for copies in sorted(t1_dist.keys()):
            print(f"  {copies:>8}  {t1_dist[copies]:>8}")

    print(f"\n--- Type 2: Same Utterance, Different Input ---")
    print(f"Utterances with variants: {total_type2_groups}")
    print(f"Extra variant items:      {total_type2_extra}")
    if all_type2_variant_counts:
        t2_dist: dict[int, int] = defaultdict(int)
        for c in all_type2_variant_counts:
            t2_dist[c] += 1
        print(f"\n  {'Variants':>10}  {'Utterances':>10}")
        for variants in sorted(t2_dist.keys()):
            print(f"  {variants:>10}  {t2_dist[variants]:>10}")

    total_extra = total_type1_extra + total_type2_extra
    print(f"\nTotal extra items (Type 1 + Type 2): {total_extra}")


if __name__ == "__main__":
    main()
