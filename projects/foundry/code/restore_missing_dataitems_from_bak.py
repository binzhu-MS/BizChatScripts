"""
Restore missing dataItems from sessions/**/*.json.bak into the active
sessions/**/*.json files.

For each session that has a sibling .bak:
  - Load both files.
  - Build the set of utterances currently in the active session's
    sessionInputs[0].dataItems (or top-level dataItems if no sessionInputs).
  - From the .bak's sessionInputs[0].dataItems (or top-level fallback),
    find dataItems whose utterance is missing from the active file.
  - Append the missing items to the active file's sessionInputs[0].dataItems
    AND top-level dataItems, assigning new sequential indices that continue
    from the current max index.

Items already present (by utterance) are NOT touched. No existing dataItems
are removed or reordered. dataItemsOutputs entries are unaffected (new
indices will simply have no scraper output, which build_foundry_eval_sessions
handles gracefully).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


SESSIONS_DIR = Path(__file__).resolve().parents[2] / "sessions"


def _utt(di: dict) -> str:
    try:
        obj = json.loads(di.get("input", "") or "")
    except (json.JSONDecodeError, TypeError):
        return ""
    if isinstance(obj, dict):
        return (obj.get("utterance") or "").strip()
    return ""


def _active_items(session: dict) -> tuple[list[dict], list[dict] | None]:
    """Return (sessionInputs0_items, top_level_items_or_None).

    sessionInputs0_items is a list (possibly the same object as the one
    stored on the session); when sessionInputs is absent it is the
    top-level list. top_level_items is None when no top-level dataItems
    key exists.
    """
    si = session.get("sessionInputs")
    si_items = None
    if isinstance(si, list) and si and isinstance(si[0], dict):
        si_items = si[0].get("dataItems")
    top = session.get("dataItems") if "dataItems" in session else None
    if si_items is None and top is None:
        return [], None
    if si_items is None:
        return top, top  # only top-level; treat as same
    return si_items, top


def _bak_source_items(bak: dict) -> list[dict]:
    si = bak.get("sessionInputs")
    if isinstance(si, list) and si and isinstance(si[0], dict):
        items = si[0].get("dataItems")
        if items:
            return items
    return bak.get("dataItems", []) or []


def restore_one(active_path: Path, bak_path: Path) -> int:
    with bak_path.open("r", encoding="utf-8") as fh:
        bak = json.load(fh)
    with active_path.open("r", encoding="utf-8") as fh:
        active = json.load(fh)

    si_items, top_items = _active_items(active)
    existing_utts = {u for u in (_utt(di) for di in si_items) if u}

    bak_items = _bak_source_items(bak)
    missing: list[dict] = []
    seen_in_missing: set[str] = set()
    for di in bak_items:
        u = _utt(di)
        if not u or u in existing_utts or u in seen_in_missing:
            continue
        seen_in_missing.add(u)
        missing.append(di)

    if not missing:
        return 0

    # Determine next index. Indices are 1-based and sequential in source
    # files; use max+1 to be safe even if there are gaps.
    def _max_index(items: list[dict]) -> int:
        return max((int(d.get("index", 0)) for d in items), default=0)

    next_idx = _max_index(si_items) + 1
    if top_items is not None and top_items is not si_items:
        next_idx = max(next_idx, _max_index(top_items) + 1)

    appended = 0
    for src in missing:
        new_item = {
            "input": src.get("input", ""),
            "pinned": src.get("pinned", False),
            "index": next_idx,
        }
        si_items.append(dict(new_item))
        if top_items is not None and top_items is not si_items:
            top_items.append(dict(new_item))
        next_idx += 1
        appended += 1

    # Write back with the same indentation style (4 spaces, UTF-8, LF).
    text = json.dumps(active, indent=4, ensure_ascii=False)
    with active_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.write("\n")
    return appended


def main() -> int:
    if not SESSIONS_DIR.exists():
        print(f"ERROR: sessions directory not found: {SESSIONS_DIR}")
        return 1

    total_files = 0
    total_appended = 0
    for bak in sorted(SESSIONS_DIR.rglob("*.json.bak")):
        active = bak.with_suffix("")  # strips .bak -> .json
        if not active.exists():
            print(f"SKIP (no active .json): {bak}")
            continue
        try:
            n = restore_one(active, bak)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR processing {active.relative_to(SESSIONS_DIR)}: {e}")
            continue
        rel = active.relative_to(SESSIONS_DIR)
        if n:
            print(f"{rel}: appended {n} dataItem(s) from .bak")
            total_appended += n
        else:
            print(f"{rel}: no missing utterances")
        total_files += 1

    print(f"\nProcessed {total_files} pair(s); appended {total_appended} dataItem(s) total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
