"""Replace C1 control characters (0x80-0x9F) with their CP1252 Unicode
equivalents, replace NBSP with ASCII space (except inside URLs), and
strip leading/trailing whitespace from utterances in session JSON files.

All session files (with and without changes) are copied to --output-dir,
preserving the original directory structure. Original files are never modified.

Usage:
    python fix_cp1252_and_strip_space.py --sessions-dir DIR --output-dir DIR
"""

import argparse
import json
import re
import shutil
from pathlib import Path

# CP1252 mapping for the C1 control character range (0x80-0x9F).
# These bytes are valid in CP1252 but are undefined/control chars in Unicode.
# Quote characters are normalized to ASCII for LLM/eval compatibility:
#   smart quotes → ASCII quotes (better tokenization, grep, exact-match)
# Other characters map to their proper Unicode equivalents.
CP1252_TO_UNICODE = {
    0x80: "\u20AC",  # € Euro sign
    0x81: "\u0081",  # <undefined - keep as-is>
    0x82: "'",       # ‚ Single low-9 quotation mark → ASCII apostrophe
    0x83: "\u0192",  # ƒ Latin small letter f with hook
    0x84: '"',       # „ Double low-9 quotation mark → ASCII double quote
    0x85: "\u2026",  # … Horizontal ellipsis
    0x86: "\u2020",  # † Dagger
    0x87: "\u2021",  # ‡ Double dagger
    0x88: "\u02C6",  # ˆ Modifier letter circumflex accent
    0x89: "\u2030",  # ‰ Per mille sign
    0x8A: "\u0160",  # Š Latin capital letter S with caron
    0x8B: "\u2039",  # ‹ Single left-pointing angle quotation mark
    0x8C: "\u0152",  # Œ Latin capital ligature OE
    0x8D: "\u008D",  # <undefined - keep as-is>
    0x8E: "\u017D",  # Ž Latin capital letter Z with caron
    0x8F: "\u008F",  # <undefined - keep as-is>
    0x90: "\u0090",  # <undefined - keep as-is>
    0x91: "'",       # ' Left single quotation mark → ASCII apostrophe
    0x92: "'",       # ' Right single quotation mark → ASCII apostrophe
    0x93: '"',       # " Left double quotation mark → ASCII double quote
    0x94: '"',       # " Right double quotation mark → ASCII double quote
    0x95: "\u2022",  # • Bullet
    0x96: "\u2013",  # – En dash
    0x97: "\u2014",  # — Em dash
    0x98: "\u02DC",  # ˜ Small tilde
    0x99: "\u2122",  # ™ Trade mark sign
    0x9A: "\u0161",  # š Latin small letter s with caron
    0x9B: "\u203A",  # › Single right-pointing angle quotation mark
    0x9C: "\u0153",  # œ Latin small ligature oe
    0x9D: "\u009D",  # <undefined - keep as-is>
    0x9E: "\u017E",  # ž Latin small letter z with caron
    0x9F: "\u0178",  # Ÿ Latin capital letter Y with diaeresis
}

C1_RANGE = set(range(0x80, 0xA0))


def fix_cp1252(text: str) -> str:
    """Replace any C1 control chars with their CP1252 Unicode equivalents."""
    if not any(ord(c) in C1_RANGE for c in text):
        return text
    chars = []
    for c in text:
        cp = ord(c)
        if cp in CP1252_TO_UNICODE:
            chars.append(CP1252_TO_UNICODE[cp])
        else:
            chars.append(c)
    return "".join(chars)


def _find_url_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) index ranges for URLs in *text*.

    The regex accepts NBSP (\\xa0) within a URL so that paths like
    ``Shared\\xa0Documents`` are treated as part of the URL.
    """
    return [(m.start(), m.end())
            for m in re.finditer(r'https?://(?:\S|\xa0)+', text)]


def fix_nbsp(text: str) -> str:
    """Replace NBSP (\\xa0) with ASCII space, except inside URLs.

    NBSP inside a URL (e.g. SharePoint ``Shared Documents``) is preserved
    because replacing it with a regular space would split the URL.

    To avoid introducing redundant whitespace, the NBSP is *dropped*
    (rather than replaced with a space) when:
    - the preceding character is already whitespace, or
    - the following character is already whitespace or NBSP, or
    - the following character is punctuation that should not be
      preceded by a space (e.g. ``?  ! . , ; : ) ] }``).
    """
    if '\xa0' not in text:
        return text
    url_spans = _find_url_spans(text)
    # Build result char-by-char so we can inspect the preceding output.
    _NO_SPACE_BEFORE = set('?!.,;:)]}\u201d\u2019')  # incl. curly quotes
    result: list[str] = []
    for i, ch in enumerate(text):
        if ch != '\xa0' or any(s <= i < e for s, e in url_spans):
            result.append(ch)
            continue
        # Outside URL — decide: replace with space or drop.
        prev = result[-1] if result else ''
        nxt = text[i + 1] if i + 1 < len(text) else ''
        if prev in (' ', '\t', '\n', '\r'):
            continue  # already whitespace before — drop
        if nxt in (' ', '\t', '\n', '\r', '\xa0'):
            continue  # whitespace/NBSP follows — drop
        if nxt in _NO_SPACE_BEFORE:
            continue  # punctuation follows — drop
        result.append(' ')
    return ''.join(result)


def _fix_data_items(data_items: list[dict], stats: dict) -> bool:
    """Apply CP1252/NBSP/strip fixes to a list of dataItem dicts in place.

    Returns True if any items were modified.
    """
    modified = False
    for item in data_items:
        input_str = item.get("input", "")
        if not input_str:
            continue
        try:
            obj = json.loads(input_str)
        except (json.JSONDecodeError, TypeError):
            continue

        utt = obj.get("utterance") or obj.get("input")
        if not utt:
            continue

        new_utt = utt

        # 1) Fix CP1252 C1 control chars
        fixed = fix_cp1252(new_utt)
        if fixed != new_utt:
            stats["cp1252_fixed"] += 1
            new_utt = fixed

        # 2) Replace NBSP with ASCII space (except inside URLs)
        nbsp_fixed = fix_nbsp(new_utt)
        if nbsp_fixed != new_utt:
            stats["nbsp_fixed"] += 1
            new_utt = nbsp_fixed

        # 3) Strip whitespace from both ends
        stripped = new_utt.strip()
        if stripped != new_utt:
            stats["stripped"] += 1
            new_utt = stripped

        if new_utt != utt:
            # Update the field that held the utterance
            if "utterance" in obj:
                obj["utterance"] = new_utt
            else:
                obj["input"] = new_utt
            item["input"] = json.dumps(obj, ensure_ascii=False)
            modified = True
    return modified


def _check_input_consistency(data: dict, rel: str) -> str | None:
    """Check whether top-level and sessionInputs[0] dataItems match.

    Returns None if identical (or only one copy exists), otherwise a
    warning string describing the mismatch.
    """
    top_di = data.get("dataItems", [])
    si = data.get("sessionInputs", [])
    si_di = si[0].get("dataItems", []) if si else []
    if not top_di or not si_di:
        return None

    def _utt(item):
        try:
            return json.loads(item.get("input", "")).get("utterance", "")
        except (json.JSONDecodeError, TypeError):
            return ""

    top_utts = [_utt(d) for d in top_di]
    si_utts = [_utt(d) for d in si_di]

    if top_utts == si_utts:
        return None

    from collections import Counter
    top_counts = Counter(top_utts)
    si_counts = Counter(si_utts)

    if top_counts == si_counts:
        return (f"  INPUT WARNING: {rel} — copies have same utterances "
                f"but different ordering (both will be identical in output)")

    top_only = top_counts - si_counts
    si_only = si_counts - top_counts
    parts = []
    if top_only:
        parts.append(f"{sum(top_only.values())} top-level-only")
    if si_only:
        parts.append(f"{sum(si_only.values())} SI-only")
    if len(top_utts) != len(si_utts):
        parts.append(f"length {len(top_utts)} vs {len(si_utts)}")
    return (f"  INPUT WARNING: {rel} — copies differ: "
            f"{', '.join(parts)} (both will be identical in output)")


def process_file(filepath: Path) -> tuple[dict, bool]:
    """Process a single session JSON file.

    Returns (data, stats, modified) where modified indicates whether any
    utterances were changed.  The returned data dict is always the full
    parsed JSON (possibly with edits applied).

    Fixes are applied to sessionInputs[0].dataItems (the active copy
    used by Foundry's evaluation engine).  The top-level dataItems
    (deprecated) is then replaced with a copy to keep them in sync.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {"cp1252_fixed": 0, "nbsp_fixed": 0, "stripped": 0}

    # Work on sessionInputs[0].dataItems — the active copy.
    session_inputs = data.get("sessionInputs", [])
    if session_inputs and session_inputs[0].get("dataItems"):
        si_items = session_inputs[0]["dataItems"]
        modified = _fix_data_items(si_items, stats)
        # Sync: always copy active → top-level to resolve any
        # pre-existing ordering/content mismatches as well.
        data["dataItems"] = json.loads(json.dumps(si_items))
    else:
        # Legacy file without sessionInputs — fix top-level directly.
        modified = _fix_data_items(data.get("dataItems", []), stats)

    return data, stats, modified


def main():
    parser = argparse.ArgumentParser(
        description="Fix CP1252 C1 control chars and strip utterance whitespace "
                    "in session JSON files.  Copies all files to --output-dir.",
    )
    parser.add_argument(
        "--sessions-dir", type=Path, required=True,
        help="Directory containing session JSON files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write output session files (mirrors input structure).",
    )
    args = parser.parse_args()

    sessions_dir = args.sessions_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not sessions_dir.is_dir():
        print(f"Error: {sessions_dir} is not a directory")
        return

    json_files = sorted(sessions_dir.rglob("*.json"))
    print(f"Scanning {len(json_files)} JSON files in {sessions_dir}")
    print(f"Output directory: {output_dir}\n")

    total_cp1252 = 0
    total_nbsp = 0
    total_stripped = 0
    files_changed = 0
    files_copied = 0
    input_warnings: list[str] = []

    for fp in json_files:
        rel = fp.relative_to(sessions_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Check input consistency before processing
        with open(fp, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        input_warn = _check_input_consistency(raw_data, str(rel))
        if input_warn:
            input_warnings.append(input_warn)

        data, stats, modified = process_file(fp)

        # The file needs writing if fixes were applied OR if the sync
        # inside process_file changed top-level to match sessionInputs.
        needs_write = modified or (input_warn is not None)

        if needs_write:
            # Write the fixed/synced JSON
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                f.write("\n")
            files_changed += 1
            if modified:
                print(f"  [fixed] {rel}: {stats['cp1252_fixed']} CP1252, "
                      f"{stats['nbsp_fixed']} NBSP, {stats['stripped']} stripped")
        else:
            # Copy unchanged file as-is
            shutil.copy2(fp, out_path)

        files_copied += 1
        total_cp1252 += stats["cp1252_fixed"]
        total_nbsp += stats["nbsp_fixed"]
        total_stripped += stats["stripped"]

    print(f"\nDone: {files_copied} files written to {output_dir}")
    if input_warnings:
        print(f"\n--- Input consistency warnings ({len(input_warnings)} files) ---")
        for w in input_warnings:
            print(w)
    print(f"\n  {files_changed} files modified "
          f"({total_cp1252} CP1252 fixes, {total_nbsp} NBSP fixes, "
          f"{total_stripped} strips)")
    print(f"  {files_copied - files_changed} files copied unchanged")

    # --- Verification: check consistency between copies and residual issues ---
    print("\n=== Verification ===\n")

    files_inconsistent = 0
    files_residual = 0
    for fp in json_files:
        rel = fp.relative_to(sessions_dir)
        out_path = output_dir / rel

        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract utterances from both copies
        def _get_utts(items):
            result = []
            for item in items:
                try:
                    obj = json.loads(item.get("input", ""))
                    result.append(obj.get("utterance", ""))
                except (json.JSONDecodeError, TypeError):
                    result.append("")
            return result

        top_di = data.get("dataItems", [])
        si = data.get("sessionInputs", [])
        si_di = si[0].get("dataItems", []) if si else []
        top_utts = _get_utts(top_di)
        si_utts = _get_utts(si_di)

        # 1) Consistency check
        if top_utts != si_utts:
            files_inconsistent += 1
            diffs = sum(1 for a, b in zip(top_utts, si_utts) if a != b)
            print(f"  INCONSISTENT: {rel} ({diffs} utterances differ)")

        # 2) Residual C1 / NBSP-outside-URL check on the active copy
        active = si_utts if si_di else top_utts
        has_issue = False
        for i, u in enumerate(active):
            if any(ord(c) in C1_RANGE for c in u):
                if not has_issue:
                    print(f"  RESIDUAL: {rel}")
                    has_issue = True
                print(f"    [{i}] C1 control char in: {u[:60]!r}")
            if '\xa0' in u:
                url_spans = [(m.start(), m.end())
                             for m in re.finditer(r'https?://(?:\S|\xa0)+', u)]
                for j, ch in enumerate(u):
                    if ch == '\xa0' and not any(
                            s <= j < e for s, e in url_spans):
                        if not has_issue:
                            print(f"  RESIDUAL: {rel}")
                            has_issue = True
                        print(f"    [{i}] NBSP outside URL in: {u[:60]!r}")
                        break
        if has_issue:
            files_residual += 1

    if files_inconsistent == 0 and files_residual == 0:
        print(f"All {len(json_files)} files: copies consistent, "
              f"no residual C1/NBSP issues. \u2713")
    else:
        if files_inconsistent:
            print(f"\n{files_inconsistent} file(s) have inconsistent copies!")
        if files_residual:
            print(f"{files_residual} file(s) have residual C1/NBSP issues!")


if __name__ == "__main__":
    main()
