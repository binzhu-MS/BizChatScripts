"""
Apply Cartesian-product punctuation-only dedup to session JSON files.

Walks all *.json session files under --input-dir (recursively), extracts
utterances from dataItems[i].input.utterance, finds duplicates WITHIN
each session file, and applies the dedup scheme to make every utterance
unique within its file. Writes modified session files to --output-dir
preserving directory structure.

Identical utterances ACROSS different session files are encouraged — they
will be deduped identically (same base + same suffix order) so they merge
naturally when generating a queryset for SEVAL.

Duplicate detection:
  Two utterances are considered duplicates (and dedup is applied) when
  they produce the same base text after extract_base() normalization.
  This means all of the following are treated as duplicates of each other:

  - Exact same text:     "What is the weather"  ==  "What is the weather"
  - Trailing space:      "What is the weather "  ->  base "What is the weather"
  - Leading space:       " What is the weather"  ->  base "What is the weather"
  - Trailing punctuation:"What is the weather."  ->  base "What is the weather"
  - Trailing marks:      "What is the weather?"  ->  base "What is the weather"
  - Paired marks:        "¿What is the weather?" ->  base "What is the weather"
  - Combined:            " What is the weather. " -> base "What is the weather"

  Whitespace, paired marks, and trailing punctuation are stripped
  iteratively until stable. Utterances that differ only in these
  elements share the same base and are grouped for dedup.

  NOT considered duplicates:
  - Different interior text: "What is weather" vs "What is the weather"
  - Different interior spacing: "What  is" vs "What is"

The dedup scheme:
  1. Extract the base text by iteratively stripping whitespace, paired
     marks (e.g. ¿?, ¡!, "", 「」), and single trailing punctuation
     marks (both Latin and CJK).
  2. Detect CJK script at the end of the base text to choose suffix marks.
  3. Determine 4 compatible marks: period, comma, semicolon, plus
     question or exclamation mark (in the appropriate script).
  4. Enumerate suffixes as Cartesian products of marks, ordered by length.
  5. Assign each copy the next suffix in order.

Usage:
    python dedup_sessions.py --input-dir <DIR> --output-dir <DIR>

Arguments:
    --input-dir   Root directory containing session JSON files (searched
                  recursively). Each file must have a top-level "dataItems"
                  array where each item has an "input" field (JSON string)
                  containing an "utterance" key.

    --output-dir  Directory where deduped session files are written.
                  Subdirectory structure from --input-dir is preserved.
                  Created automatically if it does not exist.

Examples:
    # Dedup all sessions under sessions/ and write to local/sessions_deduped/
    python dedup_sessions.py \\
        --input-dir sessions \\
        --output-dir local/sessions_deduped

    # Dedup only base-sessions
    python dedup_sessions.py \\
        --input-dir sessions/base-sessions \\
        --output-dir local/sessions_deduped/base-sessions

Output:
    - Modified JSON files in --output-dir (same structure as --input-dir)
    - Console statistics: files processed, items changed, copy distribution
    - Verification pass confirming no per-file duplicates remain
"""

import argparse
import itertools
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

# --- Trailing marks (Latin + CJK) ---
TRAILING_MARKS = set(".,;?!。？！、；")

# --- Paired marks: opener -> closer ---
PAIRED_MARKS = {
    "¿": "?",
    "¡": "!",
    '"': '"',
    "'": "'",
    "\u201c": "\u201d",  # " "
    "\u2018": "\u2019",  # ' '
    "«": "»",
    "「": "」",
    "『": "』",
    "（": "）",
    "【": "】",
    "《": "》",
    "〈": "〉",
    "〔": "〕",
}

QUESTION_WORDS = {
    "what", "when", "where", "who", "whom", "which", "whose",
    "how", "why", "did", "do", "does", "is", "are", "was", "were",
    "can", "could", "will", "would", "should", "shall", "have", "has", "had",
}


def _is_cjk_char(ch: str) -> bool:
    """Return True if ch is a CJK ideograph, hiragana, katakana, or hangul."""
    cp = ord(ch)
    # CJK Unified Ideographs
    if 0x4E00 <= cp <= 0x9FFF:
        return True
    # CJK Unified Ideographs Extension A
    if 0x3400 <= cp <= 0x4DBF:
        return True
    # CJK Unified Ideographs Extension B
    if 0x20000 <= cp <= 0x2A6DF:
        return True
    # CJK Compatibility Ideographs
    if 0xF900 <= cp <= 0xFAFF:
        return True
    # Hiragana
    if 0x3040 <= cp <= 0x309F:
        return True
    # Katakana
    if 0x30A0 <= cp <= 0x30FF:
        return True
    # Katakana Phonetic Extensions
    if 0x31F0 <= cp <= 0x31FF:
        return True
    # Hangul Syllables
    if 0xAC00 <= cp <= 0xD7AF:
        return True
    # Hangul Jamo
    if 0x1100 <= cp <= 0x11FF:
        return True
    # Hangul Compatibility Jamo
    if 0x3130 <= cp <= 0x318F:
        return True
    return False


def _is_hangul_char(ch: str) -> bool:
    """Return True if ch is a Hangul character."""
    cp = ord(ch)
    # Hangul Syllables
    if 0xAC00 <= cp <= 0xD7AF:
        return True
    # Hangul Jamo
    if 0x1100 <= cp <= 0x11FF:
        return True
    # Hangul Compatibility Jamo
    if 0x3130 <= cp <= 0x318F:
        return True
    return False


# Characters to skip when scanning for the last content character
_SKIP_CHARS = (
    set("'\"")
    | set(PAIRED_MARKS.keys())
    | set(PAIRED_MARKS.values())
    | {"\u201c", "\u201d", "\u2018", "\u2019"}
    | TRAILING_MARKS
)


def _strip_trailing_url(text: str) -> str:
    """Remove a trailing URL from text for script detection purposes.

    If the text ends with a URL (https://... or http://... running to the
    end with no spaces), strip it so that the script of the preceding
    content is used instead.  E.g. `中文内容 https://example.com` should
    detect as CJK, not Latin.
    """
    idx = max(text.rfind("https://"), text.rfind("http://"))
    if idx == -1:
        return text
    # URL runs to end if no space after the scheme
    if text.find(" ", idx) == -1:
        return text[:idx]
    return text


def _last_content_char(text: str):
    """Find the last content character, skipping quotes/marks and trailing URLs."""
    text = _strip_trailing_url(text)
    for ch in reversed(text):
        if ch in _SKIP_CHARS or ch.isspace():
            continue
        return ch
    return None


def is_cjk_text(base: str) -> bool:
    """Return True if the base text's last content character is CJK.

    Skips trailing URLs, quote marks, paired mark characters, and
    trailing punctuation to find the actual last content character.
    """
    ch = _last_content_char(base)
    if ch is None:
        return False
    return _is_cjk_char(ch)


def is_hangul_text(base: str) -> bool:
    """Return True if the base text's last content character is Hangul.

    Same logic as is_cjk_text, but checks specifically for Hangul.
    Modern Korean uses Latin punctuation marks, not CJK ideographic marks.
    """
    ch = _last_content_char(base)
    if ch is None:
        return False
    return _is_hangul_char(ch)


def _ends_with_url(text: str) -> bool:
    """Return True if text ends within a URL.

    Finds the last http:// or https:// in the text and checks whether
    the URL extends to the end of the string (no space between the
    URL scheme and the end). If so, trailing characters like ? and .
    are part of the URL and must not be stripped.
    """
    idx = max(text.rfind("https://"), text.rfind("http://"))
    if idx == -1:
        return False
    # URLs don't contain spaces — if no space after the scheme,
    # the URL runs to the end of the string.
    return text.find(" ", idx) == -1


def extract_base(text: str) -> str:
    """Extract the base text by iteratively stripping whitespace, paired
    marks, and a single trailing punctuation mark until stable."""
    while True:
        prev = text

        # Step a: strip whitespace from both ends
        text = text.strip()

        # URL guard: if text ends within a URL, stop stripping —
        # characters like ? . are valid URL components.
        if _ends_with_url(text):
            break

        # Step b: strip one pair of matching marks (opener at start, closer at end)
        if len(text) >= 2:
            first = text[0]
            last = text[-1]
            if first in PAIRED_MARKS and PAIRED_MARKS[first] == last:
                text = text[1:-1]
                # Restart the loop after pair removal
                continue

        # Step c: strip a single trailing mark
        if text and text[-1] in TRAILING_MARKS:
            text = text[:-1]

        # Check for stability
        if text == prev:
            break

    return text


def is_question(original: str, base: str) -> bool:
    """Determine if the utterance is a question."""
    stripped = original.rstrip()
    if stripped.endswith("?") or stripped.endswith("？"):
        return True
    stripped_left = original.lstrip()
    if stripped_left.startswith("¿"):
        return True
    words = base.split()
    if words:
        first_word = words[0].lower()
        return first_word in QUESTION_WORDS
    return False


def generate_suffixes(marks: list[str], count: int) -> list[str]:
    """
    Generate `count` suffixes as Cartesian products of `marks`,
    ordered by length (0, 1, 2, ...) then lexicographically.
    """
    suffixes = []
    length = 0
    while len(suffixes) < count:
        if length == 0:
            suffixes.append("")
        else:
            for combo in itertools.product(marks, repeat=length):
                suffixes.append("".join(combo))
                if len(suffixes) >= count:
                    break
        length += 1
    return suffixes[:count]


def load_session(path: str) -> dict:
    """Load a session JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_session(data: dict, path: str) -> None:
    """Save a session JSON file, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def extract_utterance(input_field) -> str | None:
    """Parse the input field and extract the utterance."""
    if isinstance(input_field, str):
        try:
            obj = json.loads(input_field)
        except (json.JSONDecodeError, TypeError):
            return None
    elif isinstance(input_field, dict):
        obj = input_field
    else:
        return None
    return obj.get("utterance")


def set_utterance(input_field, new_utterance: str):
    """Return a new input field with the utterance replaced."""
    if isinstance(input_field, str):
        obj = json.loads(input_field)
        obj["utterance"] = new_utterance
        return json.dumps(obj, ensure_ascii=False)
    elif isinstance(input_field, dict):
        input_field["utterance"] = new_utterance
        return input_field
    return input_field


def _check_input_consistency(
    data: dict, rel: str, extract_utterance_fn
) -> str | None:
    """Check whether top-level and sessionInputs[0] dataItems match.

    Returns None if identical (or only one copy exists), otherwise a
    warning string describing the mismatch.
    """
    top_di = data.get("dataItems", [])
    si = data.get("sessionInputs", [])
    si_di = si[0].get("dataItems", []) if si else []
    if not top_di or not si_di:
        return None

    top_utts = [extract_utterance_fn(d.get("input")) or "" for d in top_di]
    si_utts = [extract_utterance_fn(d.get("input")) or "" for d in si_di]

    if top_utts == si_utts:
        return None

    # Classify the mismatch
    from collections import Counter
    top_counts = Counter(top_utts)
    si_counts = Counter(si_utts)

    if top_counts == si_counts:
        return (f"  INPUT WARNING: {rel} — copies have same utterances "
                f"but different ordering (both will be identical in output)")

    # Different content
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


def main():
    parser = argparse.ArgumentParser(
        description="Apply Cartesian-product punctuation dedup to session files"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Root directory containing session JSON files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for deduped session files")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all JSON files
    json_files: list[tuple[str, str]] = []  # (abs_path, rel_path)
    for root, _dirs, files in os.walk(input_dir):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            filepath = os.path.join(root, fn)
            rel = os.path.relpath(filepath, input_dir).replace("\\", "/")
            json_files.append((filepath, rel))

    print(f"Found {len(json_files)} session files\n")

    # --- Process each file independently ---
    total_files = 0
    files_changed = 0
    files_unchanged = 0
    total_items_all = 0
    total_items_changed = 0
    total_dup_bases = 0
    global_max_copies = 0
    global_max_suffix_len = 0
    copy_distribution: dict[int, int] = defaultdict(int)  # copies -> base count
    input_warnings: list[str] = []

    for filepath, rel in json_files:
        data = load_session(filepath)
        total_files += 1

        # Check whether the two copies already match in the input
        input_warn = _check_input_consistency(data, rel, extract_utterance)
        if input_warn:
            input_warnings.append(input_warn)

        # Use sessionInputs[0].dataItems (the active copy used by
        # Foundry's evaluation engine).  Fall back to top-level
        # dataItems for legacy files without sessionInputs.
        session_inputs = data.get("sessionInputs", [])
        if session_inputs and session_inputs[0].get("dataItems"):
            data_items = session_inputs[0]["dataItems"]
            has_session_inputs = True
        else:
            data_items = data.get("dataItems", [])
            has_session_inputs = False

        # Group items by base text within this file
        base_groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for idx, item in enumerate(data_items):
            input_field = item.get("input")
            if input_field is None:
                continue
            utterance = extract_utterance(input_field)
            if not utterance:
                continue
            total_items_all += 1
            base = extract_base(utterance)
            base_groups[base].append((idx, utterance))

        # Dedup within this file
        file_changes: dict[int, str] = {}  # item_idx -> new utterance

        for base, entries in base_groups.items():
            count = len(entries)
            copy_distribution[count] += 1

            if count == 1:
                continue

            total_dup_bases += 1
            if count > global_max_copies:
                global_max_copies = count

            # Determine question status from first entry
            first_original = entries[0][1]
            question = is_question(first_original, base)

            # Choose script-appropriate marks
            # Korean uses Latin marks; Chinese/Japanese use CJK marks
            cjk = is_cjk_text(base)
            hangul = is_hangul_text(base)
            if cjk and not hangul:
                type_mark = "？" if question else "！"
                marks = ["。", "、", "；", type_mark]
            else:
                type_mark = "?" if question else "!"
                marks = [".", ",", ";", type_mark]

            suffixes = generate_suffixes(marks, count)

            for s in suffixes:
                if len(s) > global_max_suffix_len:
                    global_max_suffix_len = len(s)

            for i, (item_idx, original_utt) in enumerate(entries):
                new_utt = base + suffixes[i]
                if new_utt != original_utt:
                    file_changes[item_idx] = new_utt

        # Apply changes
        if file_changes:
            files_changed += 1
            total_items_changed += len(file_changes)
            for idx, new_utt in file_changes.items():
                data_items[idx]["input"] = set_utterance(
                    data_items[idx]["input"], new_utt
                )

        # Sync: ensure both copies are identical (handles both dedup
        # changes and pre-existing ordering/content mismatches).
        if has_session_inputs:
            data["dataItems"] = json.loads(json.dumps(data_items))
        else:
            if session_inputs:
                session_inputs[0]["dataItems"] = json.loads(
                    json.dumps(data_items))

        if not file_changes:
            files_unchanged += 1

        # Write output
        out_path = os.path.join(output_dir, rel)
        save_session(data, out_path)

    # --- Summary ---
    print("=== Summary ===\n")

    if input_warnings:
        print(f"--- Input consistency warnings ({len(input_warnings)} files) ---")
        for w in input_warnings:
            print(w)
        print()

    print(f"Files processed:         {total_files}")
    print(f"Files changed:           {files_changed}")
    print(f"Files unchanged:         {files_unchanged}")
    print(f"Total dataItems:         {total_items_all}")
    print(f"Items changed:           {total_items_changed}")
    print(f"Bases with dups (total): {total_dup_bases}")
    print(f"Max copies in one file:  {global_max_copies}")
    print(f"Max suffix length used:  {global_max_suffix_len}")

    print("\n--- Per-file copy count distribution (across all files) ---")
    print(f"{'Copies':>8}  {'Bases':>8}")
    for copies in sorted(copy_distribution.keys()):
        print(f"{copies:>8}  {copy_distribution[copies]:>8}")

    # --- Verification: check per-file uniqueness ---
    print("\n=== Verification ===\n")

    files_with_dups = 0
    files_inconsistent = 0
    for filepath, rel in json_files:
        out_path = os.path.join(output_dir, rel)
        data = load_session(out_path)

        # Check sessionInputs[0].dataItems (active), fall back to top-level
        si = data.get("sessionInputs", [])
        if si and si[0].get("dataItems"):
            check_items = si[0]["dataItems"]
        else:
            check_items = data.get("dataItems", [])

        utterances = []
        for item in check_items:
            utt = extract_utterance(item.get("input"))
            if utt:
                utterances.append(utt)
        unique_count = len(set(utterances))
        if unique_count < len(utterances):
            dup_count = len(utterances) - unique_count
            print(f"  WARNING: {rel} has {dup_count} remaining duplicates!")
            files_with_dups += 1

        # Consistency check: compare both copies
        top_di = data.get("dataItems", [])
        si_di = si[0].get("dataItems", []) if si else []
        if top_di and si_di:
            top_utts = [extract_utterance(d.get("input")) or "" for d in top_di]
            si_utts = [extract_utterance(d.get("input")) or "" for d in si_di]
            if top_utts != si_utts:
                diffs = sum(1 for a, b in zip(top_utts, si_utts) if a != b)
                print(f"  INCONSISTENT: {rel} "
                      f"({diffs} utterances differ between copies)")
                files_inconsistent += 1

    if files_with_dups == 0:
        print(f"All {total_files} files have unique utterances within each file. \u2713")
    else:
        print(f"\n{files_with_dups} file(s) still have duplicates!")

    if files_inconsistent == 0:
        print(f"All {total_files} files have consistent copies "
              f"(dataItems == sessionInputs[0].dataItems). \u2713")
    else:
        print(f"{files_inconsistent} file(s) have inconsistent copies!")


if __name__ == "__main__":
    main()
