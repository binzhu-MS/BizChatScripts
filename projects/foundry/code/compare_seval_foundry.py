"""Compare two results.md files side-by-side.

Usage:
    python compare_seval_foundry.py <file_a> <file_b> <output> [--threshold <N>]

    file_a       First results markdown file (e.g. seval_results.md)
    file_b       Second results markdown file (e.g. foundry results.md)
    output       Output markdown file path for the comparison report
    --threshold  Score delta threshold to flag mismatches (default: 3)

The script auto-detects the source type (Seval vs Foundry) from table headers
and the column format (with/without Link column).
"""

import argparse
import os
import re
import sys


def parse_results(path):
    """Parse a results.md file into a dict of (dir/session, criteria) -> {control: (score, rows), treatment: (score, rows)}.

    Auto-detects whether the table has a Link column (4 vs 5 data columns).
    """
    entries = {}
    current_dir = ""
    has_link = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Directory:"):
                current_dir = line.split(":", 1)[1].strip()
                has_link = None  # re-detect per table
            elif line.startswith("| Name"):
                # Auto-detect: count columns in header
                cols = [p.strip() for p in line.split("|")[1:-1]]
                has_link = len(cols) >= 5
            elif line.startswith("|---"):
                continue
            elif line.startswith("|") and has_link is not None:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if has_link:
                    if len(parts) < 5:
                        continue
                    name, criteria = parts[0], parts[1]
                    ctrl_str, treat_str = parts[3], parts[4]
                else:
                    if len(parts) < 4:
                        continue
                    name, criteria = parts[0], parts[1]
                    ctrl_str, treat_str = parts[2], parts[3]

                key = (current_dir + "/" + name, criteria.strip())
                entries[key] = {
                    "control": _parse_val(ctrl_str),
                    "treatment": _parse_val(treat_str),
                }
    return entries


def detect_source(path):
    """Detect whether a results file is from Seval or Foundry based on header content."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "[Seval]" in line:
                return "Seval"
            if "[Prompt]" in line:
                return "Foundry"
    return None


def _parse_val(s):
    # Match "0.00 (84/84 rows)" — valid/total format
    m = re.match(r"([\d.]+)\s+\((\d+)/(\d+)\s+rows\)", s)
    if m:
        return float(m.group(1)), int(m.group(2)), int(m.group(3))
    # Match "82.63 (1710 rows)" — single count format
    m = re.match(r"([\d.]+)\s+\((\d+)\s+rows\)", s)
    if m:
        return float(m.group(1)), int(m.group(2)), int(m.group(2))
    # Match "N/A (0/72 rows)"
    m = re.match(r"N/A\s+\((\d+)/(\d+)\s+rows\)", s)
    if m:
        return None, int(m.group(1)), int(m.group(2))
    # Match "N/A (0 rows)"
    m = re.match(r"N/A\s+\((\d+)\s+rows\)", s)
    if m:
        return None, int(m.group(1)), int(m.group(1))
    return None, 0, 0


def fmt(score, valid, total=None):
    if total is None:
        total = valid
    if score is None:
        return f"N/A ({valid}/{total})"
    return f"{score:.2f} ({valid}/{total})"


def _disambiguate_labels(path_a, path_b, base_label):
    """Derive distinguishing labels from paths when both files have the same source type.

    Walks the path components of each file and returns the first component
    that differs between the two.  Falls back to ``<label> (A)`` / ``<label> (B)``
    if no unique component is found.
    """
    parts_a = os.path.normpath(path_a).split(os.sep)
    parts_b = os.path.normpath(path_b).split(os.sep)
    diff_a, diff_b = None, None
    for pa, pb in zip(parts_a, parts_b):
        if pa != pb:
            diff_a, diff_b = pa, pb
            break
    if diff_a is None:
        # Paths are identical up to common length; try remaining parts
        if len(parts_a) > len(parts_b):
            diff_a, diff_b = parts_a[len(parts_b)], ""
        elif len(parts_b) > len(parts_a):
            diff_a, diff_b = "", parts_b[len(parts_a)]
    if diff_a and diff_b:
        return f"{base_label} ({diff_a})", f"{base_label} ({diff_b})"
    return f"{base_label} (A)", f"{base_label} (B)"


def main():
    parser = argparse.ArgumentParser(
        description="Compare two results.md files side-by-side."
    )
    parser.add_argument("file_a", help="First results markdown file")
    parser.add_argument("file_b", help="Second results markdown file")
    parser.add_argument(
        "--threshold", type=float, default=3,
        help="Score delta threshold to flag mismatches (default: 3)",
    )
    parser.add_argument(
        "output",
        help="Output markdown file path for the comparison report",
    )
    parser.add_argument(
        "--label-a", default=None,
        help="Custom label for file_a (overrides auto-detect)",
    )
    parser.add_argument(
        "--label-b", default=None,
        help="Custom label for file_b (overrides auto-detect)",
    )
    args = parser.parse_args()

    path_a = args.file_a
    path_b = args.file_b
    threshold = args.threshold

    # Auto-detect source types
    source_a = args.label_a or detect_source(path_a)
    source_b = args.label_b or detect_source(path_b)

    # Disambiguate when no custom labels provided
    if not args.label_a and not args.label_b:
        # Both same type (including both None) or one is None → disambiguate
        if source_a == source_b or source_a is None or source_b is None:
            base = source_a or source_b or "Results"
            source_a, source_b = _disambiguate_labels(path_a, path_b, base)
    else:
        source_a = source_a or os.path.basename(path_a)
        source_b = source_b or os.path.basename(path_b)

    # Auto-order: Foundry first, Seval second (skip if custom labels provided)
    # Delta = Seval - Foundry, so positive means Seval is higher
    if not args.label_a and not args.label_b:
        if source_a == "Seval" and source_b == "Foundry":
            path_a, path_b = path_b, path_a
            source_a, source_b = source_b, source_a

    foundry_label = source_a
    seval_label = source_b

    foundry = parse_results(path_a)
    seval = parse_results(path_b)

    all_keys = sorted(set(foundry.keys()) | set(seval.keys()))

    # Categorize
    warn_threshold = 10  # hard boundary between warning and mismatch
    matched = []         # |Δ| <= threshold
    improved = []        # Seval notably higher (good) — counted as matched in stats
    warning = []         # Seval lower, threshold < drop <= warn_threshold
    mismatched = []      # Seval lower, drop > warn_threshold (or N/A)
    foundry_only = []
    seval_only = []

    for key in all_keys:
        fd = foundry.get(key)
        sv = seval.get(key)
        if fd and not sv:
            foundry_only.append(key)
        elif sv and not fd:
            seval_only.append(key)
        else:
            max_drop = 0   # max amount Seval is BELOW Foundry (bad)
            max_gain = 0   # max amount Seval is ABOVE Foundry (good)
            has_na = False
            for arm in ("control", "treatment"):
                fv, _, _ = fd[arm]
                ev, _, _ = sv[arm]
                if fv is None or ev is None:
                    if (fv is not None and fv > 0) or (ev is not None and ev > 0):
                        has_na = True
                elif fv is not None and ev is not None:
                    diff = ev - fv  # positive = Seval higher
                    if diff < 0:
                        max_drop = max(max_drop, -diff)
                    else:
                        max_gain = max(max_gain, diff)
            if has_na or max_drop > warn_threshold:
                mismatched.append(key)
            elif max_drop > threshold:
                warning.append(key)
            elif max_gain > threshold:
                improved.append(key)
            else:
                matched.append(key)

    lines = []

    # --- Summary ---
    lines.append(f"# Comparison: {seval_label} vs {foundry_label}\n")
    lines.append(f"- **{foundry_label}**: {os.path.basename(path_a)}")
    lines.append(f"- **{seval_label}**: {os.path.basename(path_b)}")
    lines.append(f"- **Threshold**: {threshold}")
    lines.append(f"- **Total entries**: {len(all_keys)}")
    lines.append(f"- **Matched** \u2705 (|\u0394| <= {threshold}): {len(matched)}")
    lines.append(f"- **Improved** \U0001F53C ({seval_label} notably higher): {len(improved)}")
    lines.append(f"- **Warning** \u26A0\uFE0F ({seval_label} lower, {threshold} < drop <= {warn_threshold}): {len(warning)}")
    lines.append(f"- **Mismatched** \u274C ({seval_label} lower, drop > {warn_threshold}): {len(mismatched)}")
    lines.append(f"- **{foundry_label}-only**: {len(foundry_only)}")
    lines.append(f"- **{seval_label}-only**: {len(seval_only)}")
    lines.append(f"- **\u0394 direction**: {seval_label} \u2212 {foundry_label} (positive = {seval_label} higher)")
    lines.append("")

    # --- Full side-by-side table ---
    all_both_keys = sorted(foundry.keys() & seval.keys())
    by_dir = {}
    for key in all_both_keys:
        session, crit = key
        d = session.rsplit("/", 1)[0] if "/" in session else ""
        by_dir.setdefault(d, []).append(key)

    lines.append("## Side-by-Side Comparison\n")

    for directory in sorted(by_dir.keys()):
        lines.append(f"### {directory}\n")
        lines.append(
            f"| Name | Criteria "
            f"| {foundry_label} control | {seval_label} control | \u0394 ctrl "
            f"| {foundry_label} treatment | {seval_label} treatment | \u0394 treat | Flag |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")

        # Group by session name within directory
        by_session = {}
        for key in by_dir[directory]:
            session, crit = key
            name = session.rsplit("/", 1)[-1]
            by_session.setdefault(name, []).append(key)

        for name in sorted(by_session.keys()):
            session_keys = sorted(by_session[name], key=lambda k: k[1])
            for i, key in enumerate(session_keys):
                fd = foundry[key]
                sv = seval[key]
                _, crit = key

                display_name = name if i == 0 else ""

                fc, fv_c, ft_c = fd["control"]
                sc, sv_c, st_c = sv["control"]
                ft_score, fv_t, ft_t = fd["treatment"]
                st_score, sv_t, st_t = sv["treatment"]

                def delta(seval_val, foundry_val):
                    if seval_val is None or foundry_val is None:
                        return "---"
                    return f"{seval_val - foundry_val:+.2f}"

                dc = delta(sc, fc)
                dt = delta(st_score, ft_score)

                # Flags: ✅ matched, 🔼 improved, ⚠️ warning, ❌ mismatch, ➖ N/A
                if dc == "---" or dt == "---":
                    flag = "\u2796"
                else:
                    fdc, fdt = float(dc), float(dt)
                    max_drop = max(0, -fdc, -fdt)
                    max_gain = max(0, fdc, fdt)
                    if max_drop > warn_threshold:
                        flag = "\u274C"
                    elif max_drop > threshold:
                        flag = "\u26A0\uFE0F"
                    elif max_gain > threshold:
                        flag = "\U0001F53C"
                    else:
                        flag = "\u2705"

                lines.append(
                    f"| {display_name} | {crit} "
                    f"| {fmt(fc,fv_c,ft_c)} | {fmt(sc,sv_c,st_c)} | {dc} "
                    f"| {fmt(ft_score,fv_t,ft_t)} | {fmt(st_score,sv_t,st_t)} | {dt} | {flag} |"
                )

        lines.append("")

    # --- Warning details ---
    if warning:
        lines.append(f"## Warning Entries ({len(warning)})\n")
        lines.append(
            f"| Session | Criteria | Arm "
            f"| {foundry_label} | {seval_label} | \u0394 | Valid Rows Diff |"
        )
        lines.append("|---|---|---|---|---|---|---|")

        for key in warning:
            fd = foundry[key]
            sv = seval[key]
            session, crit = key
            for arm in ("control", "treatment"):
                fv, fvld, ftot = fd[arm]
                ev, evld, etot = sv[arm]
                if ev is not None and fv is not None:
                    d_str = f"{ev-fv:+.2f}"
                elif fv is None:
                    d_str = f"{foundry_label}=N/A"
                elif ev is None:
                    d_str = f"{seval_label}=N/A"
                else:
                    d_str = "both N/A"
                rd_str = f"{evld-fvld:+d}"
                lines.append(
                    f"| {session} | {crit} | {arm} "
                    f"| {fmt(fv,fvld,ftot)} | {fmt(ev,evld,etot)} | {d_str} | {rd_str} |"
                )
        lines.append("")

    # --- Mismatched details ---
    if mismatched:
        lines.append(f"## Mismatched Entries ({len(mismatched)})\n")
        lines.append(
            f"| Session | Criteria | Arm "
            f"| {foundry_label} | {seval_label} | \u0394 | Valid Rows Diff |"
        )
        lines.append("|---|---|---|---|---|---|---|")

        for key in mismatched:
            fd = foundry[key]
            sv = seval[key]
            session, crit = key
            for arm in ("control", "treatment"):
                fv, fvld, ftot = fd[arm]
                ev, evld, etot = sv[arm]
                if ev is not None and fv is not None:
                    d_str = f"{ev-fv:+.2f}"
                elif fv is None:
                    d_str = f"{foundry_label}=N/A"
                elif ev is None:
                    d_str = f"{seval_label}=N/A"
                else:
                    d_str = "both N/A"
                rd_str = f"{evld-fvld:+d}"
                lines.append(
                    f"| {session} | {crit} | {arm} "
                    f"| {fmt(fv,fvld,ftot)} | {fmt(ev,evld,etot)} | {d_str} | {rd_str} |"
                )
        lines.append("")

    # --- Foundry-only ---
    if foundry_only:
        lines.append(f"## {foundry_label}-Only Entries ({len(foundry_only)})\n")
        lines.append("| Session | Criteria | control | treatment |")
        lines.append("|---|---|---|---|")
        for key in foundry_only:
            fd = foundry[key]
            session, crit = key
            fc, fv_c, ft_c = fd["control"]
            ft_score, fv_t, ft_t = fd["treatment"]
            lines.append(f"| {session} | {crit} | {fmt(fc,fv_c,ft_c)} | {fmt(ft_score,fv_t,ft_t)} |")
        lines.append("")

    # --- Seval-only ---
    if seval_only:
        lines.append(f"## {seval_label}-Only Entries ({len(seval_only)})\n")
        lines.append("| Session | Criteria | control | treatment |")
        lines.append("|---|---|---|---|")
        for key in seval_only:
            sv = seval[key]
            session, crit = key
            sc, sv_c, st_c = sv["control"]
            st_score, sv_t, st_t = sv["treatment"]
            lines.append(f"| {session} | {crit} | {fmt(sc,sv_c,st_c)} | {fmt(st_score,sv_t,st_t)} |")
        lines.append("")

    # --- Delta statistics ---
    deltas_ctrl = []
    deltas_treat = []
    for key in matched + improved + warning + mismatched:
        fd = foundry[key]
        sv = seval[key]
        fc, _, _ = fd["control"]
        sc, _, _ = sv["control"]
        ft, _, _ = fd["treatment"]
        st, _, _ = sv["treatment"]
        if fc is not None and sc is not None:
            deltas_ctrl.append(sc - fc)
        if ft is not None and st is not None:
            deltas_treat.append(st - ft)

    lines.append("## Delta Statistics\n")
    for label, deltas in [("Control", deltas_ctrl), ("Treatment", deltas_treat)]:
        if not deltas:
            continue
        abs_deltas = [abs(d) for d in deltas]
        lines.append(f"### {label} arm ({len(deltas)} pairs)\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Mean delta | {sum(deltas)/len(deltas):+.2f} |")
        lines.append(f"| Mean |delta| | {sum(abs_deltas)/len(abs_deltas):.2f} |")
        lines.append(f"| Max |delta| | {max(abs_deltas):.2f} |")
        lines.append(f"| Pairs |d| <= 1 | {sum(1 for d in abs_deltas if d <= 1)} |")
        lines.append(f"| Pairs |d| <= 3 | {sum(1 for d in abs_deltas if d <= 3)} |")
        lines.append(f"| Pairs |d| <= 5 | {sum(1 for d in abs_deltas if d <= 5)} |")
        lines.append(f"| Pairs |d| <= 10 | {sum(1 for d in abs_deltas if d <= 10)} |")
        lines.append(f"| Pairs |d| > 10 | {sum(1 for d in abs_deltas if d > 10)} |")
        lines.append("")

    content = "\n".join(lines)
    with open(args.output, "w", encoding="utf-8") as out:
        out.write(content)

    print(f"Report saved to: {args.output}")
    print(f"  {len(all_keys)} entries, {len(matched)} matched, {len(improved)} improved, "
          f"{len(warning)} warning, {len(mismatched)} mismatched, "
          f"{len(foundry_only)} {foundry_label}-only, {len(seval_only)} {seval_label}-only")


if __name__ == "__main__":
    main()
