"""Audit Foundry criterion `script` fields that consume `{{assessment}}`.

Flags scripts that parse the assessment with json.loads / safe_json_loads
without first stripping Markdown code fences (the bug behind O-G).
"""
from __future__ import annotations

import glob
import json
import os
import re

ROOT = r"C:\working\Sydney\services\TuringBot\tools\ReasoningPythonScript\sessions"

PARSE_RE = re.compile(
    r"(json\.loads|safe_json_loads|json_loads|ast\.literal_eval|literal_eval)\s*\(",
)
FENCE_INDICATOR_RE = re.compile(r"```|backtick|rstrip\([^)]*`|strip\([^)]*`")


def main() -> None:
    files = sorted(glob.glob(os.path.join(ROOT, "**", "*.json"), recursive=True))
    rows = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fp:
                s = json.load(fp)
        except Exception as exc:  # noqa: BLE001
            rows.append((os.path.basename(f), "<load error>", "-", "-", "-", str(exc)))
            continue
        crits = (s.get("evaluationStrategy") or {}).get("criteriaList") or []
        for c in crits:
            script = c.get("script") or ""
            if "{{assessment}}" not in script:
                continue
            parses = bool(PARSE_RE.search(script))
            if not parses:
                continue
            strips_fence = bool(FENCE_INDICATOR_RE.search(script))
            has_try = ("try:" in script) and ("except" in script)
            rows.append((
                os.path.basename(f),
                (c.get("name") or "")[:60],
                "Y" if parses else "n",
                "Y" if strips_fence else "n",
                "Y" if has_try else "n",
                "",
            ))

    # Print
    print(
        f"{'FILE':<60} {'CRIT':<60} {'PARSE':<5} {'FENCE':<5} {'TRY':<3}"
    )
    print("-" * 140)
    risky = 0
    for r in rows:
        marker = "  <-- UNSAFE" if (r[2] == "Y" and r[3] != "Y") else ""
        if marker:
            risky += 1
        print(f"{r[0]:<60} {r[1]:<60} {r[2]:<5} {r[3]:<5} {r[4]:<3}{marker}")
    print()
    print(f"Total criteria parsing {{assessment}}: {len(rows)}")
    print(f"Without fence-stripping (UNSAFE): {risky}")


if __name__ == "__main__":
    main()
