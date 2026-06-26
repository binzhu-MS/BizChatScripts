# seval_reasoning_checklist_scorecard.py
#
# Copyright (c) Microsoft Corporation. All rights reserved.

"""
Seval Reasoning-Checklist scorecard runner.

Description:
    Walks a Seval job's raw Sydney scrape directory, pairs control/treatment
    JSON files by **utterance text** (UUIDs in the filenames are different
    between the two variants and cannot be used), looks up the per-query
    session/criteria from the Seval queryset TSV, and runs the CoMet
    ``bizchat_reasoning_checklist`` metric on each side.  Emits per-query
    results (JSONL) and a flat per-criterion CSV with control + treatment
    scores side by side.

Inputs:
    --raw-dir    (required) folder containing
                 control_sydney_response_<uuid>.json and
                 treatment_sydney_response_<uuid>.json files.
    --queryset   (required) TSV with two columns (query_id_json, metadata_json),
                 where metadata_json carries
                 personalization_metadata.metricsInput[] describing which
                 session/criteriaList/customDimensions to run.
    --output-dir (required) output folder (created if missing).
    --workers    (optional) thread-pool size for parallel metric runs.
                 Default: 8.
    --limit      (optional) process only the first N utterances (sorted)
                 for quick debugging. Default: 0 = process all.
    --pairs-cache (optional) path to a JSON file caching the
                 utterance -> {control: path, treatment: path} pairing. If
                 the file exists it is loaded (skipping the slow rescan of
                 raw-dir); otherwise the pairing is computed and written
                 to this path for reuse. Default: none (always rescan).
    --sessions   (optional) comma-separated list of session names (or
                 substrings) to include. metricsInput entries whose
                 session does not match any token are dropped. Useful for
                 reruns scoped to a single Foundry session, e.g.
                 --sessions "retain_i_me_my_in_query,time_range_checking".
                 Default: none = all sessions.
    --criteria   (optional) comma-separated list of criterion names to
                 include. Within each surviving session, criteriaList is
                 filtered to these names; sessions whose criteriaList
                 becomes empty are dropped. Default: none = all criteria.
    --debug      (optional) enable verbose output including progress and
                 CoMet INFO logs. Default: off (only warnings/errors shown).

Output behavior:
    All output files (``results.jsonl``, ``scores.csv``, ``failures.jsonl``,
    ``sessions/<group>/<name>.tsv``, ``results.md``) are OVERWRITTEN on
    every run — they are never merged with prior content. When you scope a
    rerun with ``--sessions`` / ``--criteria``, the rewritten files contain
    ONLY the filtered subset; rows for other sessions/criteria from a
    previous full run will be lost.

    Recommendation: for a partial rerun (e.g. only the criteria impacted
    by a fix), point ``--output-dir`` at a NEW directory and diff or merge
    against the prior full run externally. Use the same ``--pairs-cache``
    file to skip the slow scrape rescan.

Usage:
    conda run -n comet python seval_reasoning_checklist_scorecard.py ^
        --raw-dir   "C:\\working\\Sydney\\...\\Sydney_Raw_Responses" ^
        --queryset  "C:\\working\\Sydney\\...\\test_full_cklist_queryset.tsv" ^
        --output-dir ".\\out" ^
        --workers 8

Notes:
    Each Sydney scrape JSON contains an ``EvaluationData`` message inside
    ``requests[0].response_body.messages``; its ``evaluationData`` payload is
    fed verbatim to BizChatReasoningChecklistMetric via from_raw().
    LLM auth is patched to use ``get_token_local`` (interactive MSAL) the same
    way ``cometdefinition/scripts/local_scorecard.py`` does.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from contextlib import ExitStack
from unittest.mock import patch

# Default to WARNING to suppress noisy INFO lines from cometdefinition imports.
# Overridden to DEBUG/INFO when --debug is passed (handled in main()).
logging.basicConfig(level=logging.WARNING)

# ── sys.path: make cometdefinition importable + reuse scripts/token_utils ────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_METRIC_DEFINITION_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SCRIPTS_DIR = os.path.join(_METRIC_DEFINITION_DIR, "cometdefinition", "scripts")
for p in (_METRIC_DEFINITION_DIR, _SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from google.protobuf.json_format import MessageToDict  # noqa: E402

from cometdefinition.metrics.bizchat_reasoning_checklist.logic.metric_logic import (  # noqa: E402
    BizChatReasoningChecklistMetric,
)
from token_utils import get_token_local  # noqa: E402

# Filename pattern: "<variant>_sydney_response_<anything>.json".
# We only need the variant; the trailing token is NOT used for pairing because
# control and treatment runs assign different UUIDs to the same utterance.
_FNAME_RE = re.compile(
    r"^(?P<variant>control|treatment)_sydney_response_.+\.json$",
    re.IGNORECASE,
)

# Regex to grab the first "utterance": "..." literal in a scrape JSON,
# without parsing the entire (potentially multi-MB) document. Captures the
# raw inner string with JSON escapes still in place.
_UTTERANCE_RE = re.compile(r'"utterance"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _utterance_id(utterance: str) -> str:
    """Return a short, stable id for a record, derived from the utterance text."""
    h = hashlib.sha1(utterance.encode("utf-8")).hexdigest()
    return h[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Queryset loading
# ─────────────────────────────────────────────────────────────────────────────


def load_queryset(tsv_path: str) -> dict:
    """Parse the Seval queryset TSV.

    Each row is two tab-separated JSON blobs: (query_id_json, metadata_json).
    The metadata's ``personalization_metadata.metricsInput`` is a list of
    {session, criteriaList, customDimensions} entries.

    Returns:
        dict: utterance_text -> list of metricsInput entries.
              Falls back to keying by full query_id_json when utterance
              cannot be extracted.
    """
    out: dict = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"[queryset] WARN line {line_no}: expected 2 tab-separated columns, got {len(parts)}")
                continue
            query_col, meta_col = parts[0], parts[1]
            try:
                query_msgs = json.loads(query_col)
            except json.JSONDecodeError:
                query_msgs = None
            try:
                meta = json.loads(meta_col)
            except json.JSONDecodeError as e:
                print(f"[queryset] WARN line {line_no}: bad metadata JSON: {e}")
                continue

            metrics_input = (
                meta.get("personalization_metadata", {}).get("metricsInput") or []
            )
            if not metrics_input:
                continue

            # Prefer keying by the user utterance text (last user msg).
            utterance = None
            if isinstance(query_msgs, list):
                user_msgs = [
                    m.get("text")
                    for m in query_msgs
                    if isinstance(m, dict) and m.get("author") == "user"
                ]
                if user_msgs:
                    utterance = user_msgs[-1]

            keys = []
            if utterance:
                keys.append(("utterance", utterance))
            keys.append(("query_id", query_col))

            for kind, key in keys:
                bucket = out.setdefault((kind, key), [])
                bucket.append(metrics_input)
    return out


def lookup_metrics_input(qmap: dict, utterance: str | None, query_id_json: str | None):
    """Resolve metricsInput for a scrape file using utterance first, query_id as fallback."""
    if utterance:
        bucket = qmap.get(("utterance", utterance))
        if bucket:
            # If multiple rows share the same utterance, merge their metricsInput entries.
            merged: list = []
            for m in bucket:
                merged.extend(m)
            return merged
    if query_id_json:
        bucket = qmap.get(("query_id", query_id_json))
        if bucket:
            merged = []
            for m in bucket:
                merged.extend(m)
            return merged
    return None


def filter_metrics_input(
    metrics_input: list,
    session_filters: list[str] | None,
    criteria_filters: set[str] | None,
) -> list:
    """Drop metricsInput entries that do not match the requested sessions/criteria.

    - ``session_filters``: list of substrings; an entry is kept if its
      ``session`` field contains any token (case-sensitive substring match).
      ``None`` or empty list disables session filtering.
    - ``criteria_filters``: set of exact criterion names; within each
      surviving entry, ``criteriaList`` is filtered to these names. Entries
      left with an empty ``criteriaList`` are dropped. ``None`` or empty
      set disables criterion filtering.
    """
    if not session_filters and not criteria_filters:
        return metrics_input
    out: list = []
    for entry in metrics_input:
        sname = entry.get("session") or ""
        if session_filters and not any(tok in sname for tok in session_filters):
            continue
        if criteria_filters:
            kept = [c for c in (entry.get("criteriaList") or []) if c in criteria_filters]
            if not kept:
                continue
            entry = dict(entry)
            entry["criteriaList"] = kept
        out.append(entry)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scrape parsing
# ─────────────────────────────────────────────────────────────────────────────


def extract_scrape_fields(path: str) -> dict | None:
    """Pull (utterance, query_id_json, evaluation_data) out of a Sydney scrape file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[scrape] FAILED to read {os.path.basename(path)}: {e}")
        return None

    requests = doc.get("requests") or []
    if not requests:
        return None
    req0 = requests[0]
    utterance = req0.get("utterance")
    query_id_json = None
    qobj = doc.get("query")
    if isinstance(qobj, dict):
        query_id_json = qobj.get("id")

    eval_data = None
    for msg in (req0.get("response_body") or {}).get("messages") or []:
        if msg.get("messageType") == "EvaluationData":
            eval_data = msg.get("evaluationData")
            break

    if eval_data is None:
        return None

    return {
        "utterance": utterance,
        "query_id_json": query_id_json,
        "evaluation_data": eval_data,
    }


def build_sessions(metrics_input: list) -> list:
    """Map Seval metricsInput entries to ReasoningChecklist SessionEntry dicts."""
    sessions: list = []
    for entry in metrics_input:
        session_name = entry.get("session") or ""
        criteria_list = entry.get("criteriaList") or []
        custom_dims = entry.get("customDimensions") or {}
        # SessionEntry.custom_dimensions is map<string,string> — coerce values.
        # Use json.dumps for dict/list so criterion code can json.loads() round-trip
        # (e.g. expected_label for GC_* criteria). str(dict) would emit Python repr,
        # which is not valid JSON.
        coerced: dict = {}
        for k, v in custom_dims.items():
            if v is None:
                coerced[str(k)] = ""
            elif isinstance(v, (dict, list)):
                coerced[str(k)] = json.dumps(v)
            else:
                coerced[str(k)] = str(v)
        sessions.append(
            {
                "session_name": session_name,
                "criteria_list": list(criteria_list),
                "custom_dimensions": coerced,
            }
        )
    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# Metric execution
# ─────────────────────────────────────────────────────────────────────────────


def run_metric(uid: str, eval_data: dict, sessions: list) -> dict:
    """Run BizChatReasoningChecklistMetric.from_raw(...).calculate_metric() and return dict."""
    metric = BizChatReasoningChecklistMetric.from_raw(
        id=uid,
        signal={"evaluation_data": eval_data},
        eval_config={"sessions": sessions},
    )
    result = metric.calculate_metric()
    try:
        return MessageToDict(
            result,
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )
    except TypeError:
        try:
            return MessageToDict(
                result,
                preserving_proto_field_name=True,
                including_default_value_fields=True,
            )
        except TypeError:
            return MessageToDict(result, preserving_proto_field_name=True)


def flatten_scores(result_dict: dict) -> list[dict]:
    """Pull (session_name, criteria_name, score, message) rows from a metric output dict."""
    rows: list = []
    scores = (
        (result_dict or {}).get("result", {}).get("scores", {}).get("session_results")
        or []
    )
    for sess in scores:
        sname = sess.get("session_name", "")
        for cr in sess.get("criteria_results") or []:
            rows.append(
                {
                    "session_name": sname,
                    "criteria_name": cr.get("criteria_name", ""),
                    "score": cr.get("score"),
                    "message": cr.get("message", ""),
                }
            )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Pairing + worker
# ─────────────────────────────────────────────────────────────────────────────


def _extract_utterance_fast(path: str) -> str | None:
    """Cheaply extract ``requests[0].utterance`` from a Sydney scrape file.

    Avoids parsing the entire JSON document (scrapes can be megabytes). The
    field appears very near the top of the file in practice, so we read just
    the first 16 KB and regex for the first ``"utterance": "..."`` literal.
    Falls back to a full-file read if that misses.
    """
    pat = _UTTERANCE_RE
    try:
        with open(path, "rb") as f:
            head = f.read(16384)
        text = head.decode("utf-8", errors="replace")
        m = pat.search(text)
        if m is None:
            with open(path, "rb") as f:
                data = f.read()
            text = data.decode("utf-8", errors="replace")
            m = pat.search(text)
        if m is None:
            return None
        raw = m.group(1)
        try:
            return json.loads('"' + raw + '"')
        except Exception:  # noqa: BLE001
            return raw
    except Exception:  # noqa: BLE001
        return None


def discover_pairs(raw_dir: str, workers: int = 8, progress: bool = True) -> dict:
    """Pair control/treatment scrape files by utterance text (parallel).

    Algorithm:

    1. List filenames in ``raw_dir`` and split into two arms (control vs.
       treatment) using the filename prefix. UUIDs are NOT used.
    2. For every file, in parallel across ``workers`` threads, read only the
       utterance string (first 16 KB regex extraction; full-file fallback).
    3. Build per-arm ``{utterance: path}`` dicts (assumes utterances are
       unique within an arm, which they are in Seval querysets).
    4. Merge the two arm dicts by utterance string into the final
       ``{utterance: {control: path, treatment: path}}`` mapping.

    Returns:
        dict: {utterance: {"control": path, "treatment": path}}
    """
    names = [n for n in os.listdir(raw_dir) if _FNAME_RE.match(n)]
    total = len(names)
    if progress:
        print(f"  Found {total} scrape files; reading utterances with {workers} worker(s)...")

    # (variant, path) work items.
    items: list = []
    for n in names:
        m = _FNAME_RE.match(n)
        items.append((m.group("variant").lower(), os.path.join(raw_dir, n)))

    by_variant: dict = {"control": {}, "treatment": {}}
    counters = {"completed": 0, "duplicates": 0, "no_utterance": 0}
    lock = threading.Lock()
    started = time.perf_counter()
    step = max(1000, total // 20) if total else 1

    def task(item):
        variant, path = item
        utt = _extract_utterance_fast(path)
        with lock:
            counters["completed"] += 1
            if utt:
                arm = by_variant[variant]
                if utt in arm:
                    counters["duplicates"] += 1
                else:
                    arm[utt] = path
            else:
                counters["no_utterance"] += 1
            done = counters["completed"]
            if progress and (done % step == 0 or done == total):
                elapsed = time.perf_counter() - started
                print(f"  Pairing: {done}/{total} ({100*done//total}%) - {elapsed:.0f}s elapsed")

    if workers <= 1 or total == 0:
        for it in items:
            task(it)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(task, items))

    # Merge the two arm dicts by utterance string. Dict lookup is O(1), so
    # no sorting is needed here or on the on-disk cache.
    pairs: dict = {}
    for utt, p in by_variant["control"].items():
        pairs.setdefault(utt, {})["control"] = p
    for utt, p in by_variant["treatment"].items():
        pairs.setdefault(utt, {})["treatment"] = p

    if counters["duplicates"]:
        print(f"[scrape] WARN {counters['duplicates']} duplicate (utterance, variant) files; first kept")
    if counters["no_utterance"]:
        print(f"[scrape] WARN {counters['no_utterance']} files had no readable utterance; skipped")
    return pairs


def process_pair(
    utterance: str,
    files: dict,
    qmap: dict,
    session_filters: list[str] | None = None,
    criteria_filters: set[str] | None = None,
) -> dict:
    """Run the metric for one utterance (control + treatment) and return a combined record.

    The record is keyed by a stable hash of the utterance text. UUIDs from
    raw filenames are not used anywhere in the script.
    """
    rid = _utterance_id(utterance)
    record: dict = {"id": rid, "utterance": utterance, "control": None, "treatment": None, "errors": {}}
    sessions = None
    for variant in ("control", "treatment"):
        path = files.get(variant)
        if not path:
            record["errors"][variant] = "scrape file missing"
            continue
        scrape = extract_scrape_fields(path)
        if scrape is None:
            record["errors"][variant] = "no EvaluationData in scrape"
            continue
        if sessions is None:
            metrics_input = lookup_metrics_input(
                qmap,
                utterance,
                scrape["query_id_json"],
            )
            if not metrics_input:
                record["errors"][variant] = "no queryset match"
                continue
            metrics_input = filter_metrics_input(
                metrics_input, session_filters, criteria_filters
            )
            if not metrics_input:
                record["errors"][variant] = "no sessions/criteria after filter"
                continue
            sessions = build_sessions(metrics_input)
            if not sessions:
                record["errors"][variant] = "queryset entry produced no sessions"
                continue
        try:
            record[variant] = run_metric(f"{rid}-{variant}", scrape["evaluation_data"], sessions)
        except Exception as e:  # noqa: BLE001
            record["errors"][variant] = f"metric failed: {e}"
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Per-session TSV + results.md summary writers
# ─────────────────────────────────────────────────────────────────────────────

# TSV cell tokens used when a numeric score is unavailable.
#   "NA"   - no scrape file for that arm (or no EvaluationData / no queryset
#            match), so the metric was never invoked for this side.
#   "ERR"  - the metric ran but failed for this criterion (LLM error,
#            evaluation exception, unknown criterion, etc.).
#   "SKIP" - the metric ran and explicitly returned -1 (criterion not
#            applicable to this label/segment).
#   "-"    - criterion absent from the arm's output for an unknown reason
#            (should not normally happen).
_NA, _ERR, _SKIP, _MISSING = "NA", "ERR", "SKIP", "-"


def _cell_token(record: dict, variant: str, criterion_row: dict | None) -> str:
    """Return the TSV cell string for one (record, arm, criterion) tuple.

    Distinguishes 'no scrape file' from 'LLM/eval error' from 'not
    applicable (SKIPPED)' so consumers of the TSV can tell why a cell is
    blank. See the module-level token constants for the legend.
    """
    if criterion_row is not None:
        score = criterion_row.get("score")
        msg = criterion_row.get("message") or ""
        if isinstance(score, int) and score >= 0:
            return str(score)
        # score == -1: distinguish error from skip via message prefix.
        if msg.startswith(("Evaluation error", "Unknown criterion")):
            return _ERR
        return _SKIP
    err = (record.get("errors") or {}).get(variant) or ""
    if not err:
        return _MISSING
    if (
        "scrape file missing" in err
        or "no EvaluationData" in err
        or "no queryset match" in err
        or "produced no sessions" in err
    ):
        return _NA
    return _ERR


def _split_session_name(session_name: str) -> tuple[str, str]:
    """Return (group, name) for a session_name like 'base-sessions/triggering-files.json'.

    - ``group`` is the directory prefix (everything before the last '/'), or
      ``""`` if the name has no slash.
    - ``name`` is the file basename with any trailing ``.json`` stripped.
    """
    norm = session_name.replace("\\", "/")
    if "/" in norm:
        group, base = norm.rsplit("/", 1)
    else:
        group, base = "", norm
    if base.lower().endswith(".json"):
        base = base[: -len(".json")]
    return group, base


def write_session_tsvs(session_data: dict, sessions_root: str) -> None:
    """Write one TSV per session under ``sessions_root/<group>/<name>.tsv``.

    Each row is one query (one utterance); columns are ``id``, ``utterance``,
    then for every criterion of the session a pair of columns
    ``<criterion>_control`` and ``<criterion>_treatment``.

    Empty score cells are written when the metric did not produce a score on
    a side (e.g. the variant scrape was missing or errored).
    """
    if not session_data:
        return
    os.makedirs(sessions_root, exist_ok=True)
    for sname, sess in session_data.items():
        group, base = _split_session_name(sname)
        out_dir = os.path.join(sessions_root, group) if group else sessions_root
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}.tsv")
        criteria = list(sess["criteria"])  # preserve discovery order
        header = ["id", "utterance"]
        for cname in criteria:
            header.append(f"{cname}_control")
            header.append(f"{cname}_treatment")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(header)
            # Sort rows by id for deterministic output.
            for row in sorted(sess["rows"], key=lambda r: r["id"]):
                line: list = [row["id"], row["utterance"]]
                for cname in criteria:
                    s = row["scores"].get(cname) or {}
                    line.append(s.get("control", _MISSING))
                    line.append(s.get("treatment", _MISSING))
                w.writerow(line)


def _aggregate(values: list) -> tuple:
    """Return (mean_percent, n) over numeric score tokens.

    ``values`` is a list of TSV cell tokens (numeric strings or one of the
    non-numeric markers NA/ERR/SKIP/-). Only numeric tokens contribute to
    the mean; everything else is dropped. ``n`` is the number of rows that
    contributed (i.e. were applicable on that side).
    """
    valid: list = []
    for v in values:
        if v is None:
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv >= 0:
            valid.append(iv)
    n = len(valid)
    if n == 0:
        return None, 0
    return round(sum(valid) / n, 2), n


def write_results_md(session_data: dict, md_path: str) -> None:
    """Write a results.md summary grouped by session directory.

    Output mirrors the Foundry-style scorecard markdown: one ``# Directory:
    <group>`` heading per group, with a table whose columns are
    ``Name | Criteria | control | treatment``. Cell values are
    ``<mean>.XX (<n> rows)``.
    """
    if not session_data:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Reasoning Checklist Results\n\n_No data._\n")
        return

    # Bucket sessions by group, preserve criterion order within each session.
    groups: dict = {}
    for sname, sess in session_data.items():
        group, base = _split_session_name(sname)
        groups.setdefault(group, []).append((base, sname, sess))
    for entries in groups.values():
        entries.sort(key=lambda e: e[0].lower())

    lines: list = ["# Reasoning Checklist Results", ""]
    for group in sorted(groups.keys(), key=lambda g: (g == "", g.lower())):
        heading = group if group else "(root)"
        lines.append(f"# Directory: {heading}")
        lines.append("")
        lines.append("| Name | Criteria | control | treatment |")
        lines.append("|---|---|---|---|")
        for base, _sname, sess in groups[group]:
            for cname in sess["criteria"]:
                ctrl = [r["scores"].get(cname, {}).get("control") for r in sess["rows"]]
                trt = [r["scores"].get(cname, {}).get("treatment") for r in sess["rows"]]
                cm, cn = _aggregate(ctrl)
                tm, tn = _aggregate(trt)
                ccell = f"{cm:.2f} ({cn} rows)" if cm is not None else f"— ({cn} rows)"
                tcell = f"{tm:.2f} ({tn} rows)" if tm is not None else f"— ({tn} rows)"
                lines.append(f"| {base} | {cname} | {ccell} | {tcell} |")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True, help="Folder with Sydney_Raw_Responses JSON files.")
    p.add_argument("--queryset", required=True, help="Seval queryset TSV file.")
    p.add_argument("--output-dir", required=True, help="Output directory (created if missing).")
    p.add_argument("--workers", type=int, default=8, help="Parallel metric workers (default: 8).")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N utterances for debugging. 0 (default) = process all.",
    )
    p.add_argument(
        "--pairs-cache",
        default="",
        help=(
            "Optional JSON file caching the utterance -> {control, treatment} "
            "pairing. If the file exists, the slow scan of --raw-dir is "
            "skipped; otherwise pairs are computed and written here for reuse."
        ),
    )
    p.add_argument(
        "--sessions",
        default="",
        help=(
            "Comma-separated session names (or substrings) to include. "
            "metricsInput entries whose session does not match any token "
            "are dropped. Default: all sessions."
        ),
    )
    p.add_argument(
        "--criteria",
        default="",
        help=(
            "Comma-separated criterion names (exact match) to include. "
            "Within surviving sessions, criteriaList is filtered to these "
            "names; sessions left with no criteria are dropped. Default: "
            "all criteria."
        ),
    )
    p.add_argument("--debug", action="store_true", help="Enable verbose/debug output.")
    return p.parse_args()


def _log(msg: str, *, debug: bool) -> None:
    """Print a message only in debug mode."""
    if debug:
        print(msg)


def main() -> None:
    """Driver: load queryset, discover pairs, run metric, write outputs."""
    args = parse_args()

    # In debug mode, lower the log level so CoMet INFO messages are shown.
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Explicitly silence the cometdefinition logger which configures its
        # own handlers and ignores the root level.
        logging.getLogger("cometdefinition").setLevel(logging.WARNING)
        # Silence the Azure Monitor OpenTelemetry exporter, which logs
        # every successful telemetry upload ("Transmission succeeded: Item
        # received: N. Items accepted: N") at INFO and attaches its own
        # handler that ignores the root level.
        for name in (
            "azure.monitor.opentelemetry.exporter",
            "azure.core.pipeline.policies.http_logging_policy",
            "opentelemetry",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.jsonl")
    csv_path = os.path.join(args.output_dir, "scores.csv")
    failures_path = os.path.join(args.output_dir, "failures.jsonl")
    sessions_root = os.path.join(args.output_dir, "sessions")
    summary_md_path = os.path.join(args.output_dir, "results.md")

    _log(f"[setup] queryset: {args.queryset}", debug=args.debug)
    qmap = load_queryset(args.queryset)
    n_queryset_utts = sum(1 for k in qmap if isinstance(k, tuple) and k[0] == "utterance")
    print(f"Queryset: {n_queryset_utts} utterances")

    session_filters = [tok.strip() for tok in args.sessions.split(",") if tok.strip()] or None
    criteria_filters = {tok.strip() for tok in args.criteria.split(",") if tok.strip()} or None
    if session_filters:
        print(f"Session filter: {session_filters}")
    if criteria_filters:
        print(f"Criteria filter: {sorted(criteria_filters)}")

    _log(f"[setup] raw-dir : {args.raw_dir}", debug=args.debug)
    pairs = None
    cache_path = args.pairs_cache.strip() if args.pairs_cache else ""
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                pairs = json.load(f)
            print(f"Loaded pairing cache: {cache_path} ({len(pairs)} utterances)")
        except Exception as e:  # noqa: BLE001
            print(f"[cache] WARN failed to load {cache_path}: {e}; will rescan raw-dir.")
            pairs = None
    if pairs is None:
        print("Scanning scrape files and pairing by utterance...")
        pairs = discover_pairs(args.raw_dir, workers=args.workers)
        if cache_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
                # No sort: the cache is consumed by O(1) dict lookups keyed
                # on utterance, so insertion order is irrelevant on read.
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(pairs, f, ensure_ascii=False)
                print(f"Saved pairing cache: {cache_path}")
            except Exception as e:  # noqa: BLE001
                print(f"[cache] WARN failed to write {cache_path}: {e}")
    # ``pairs`` is a plain dict keyed by utterance. Pairing uses O(1) dict
    # lookups, so no sorting is needed (in memory or on disk).
    utterances = list(pairs.keys())
    incomplete = sum(1 for p in pairs.values() if not ("control" in p and "treatment" in p))
    msg = f"Pairs: {len(utterances)} utterances"
    if incomplete:
        msg += f" ({incomplete} missing one side)"
    print(msg)

    if args.limit and args.limit > 0 and args.limit < len(utterances):
        utterances = utterances[: args.limit]
        print(f"Limit: processing only the first {len(utterances)} utterances (--limit {args.limit})")

    # Pre-acquire the LLM API token interactively on the main thread so the
    # browser popup happens once. Concurrent worker threads then reuse the
    # silent-cache path inside MSAL.
    _log("[auth] acquiring LLM API token...", debug=args.debug)
    try:
        get_token_local()
        _log("[auth] token acquired.", debug=args.debug)
    except Exception as e:  # noqa: BLE001
        print(f"[auth] WARN: pre-acquire failed: {e}. Workers will retry.")

    # Patch auth at the top level (same approach as local_scorecard.py).
    auth_patches = ExitStack()
    auth_patches.enter_context(
        patch("cometdefinition.llm_api.llm_api.LLMAPI.get_token", staticmethod(get_token_local))
    )
    auth_patches.enter_context(
        patch(
            "cometdefinition.llm_api.llm_api_interface.LLMAPIInterface._get_token",
            lambda self: get_token_local(),
        )
    )

    results_lock = threading.Lock()
    completed = [0]
    started = time.perf_counter()

    # Per-session aggregator for TSV output and results.md summary.
    # Shape: { session_name: {
    #             "criteria":  ordered list of criterion names (insertion order),
    #             "criteria_set": set of criterion names (membership test),
    #             "rows":     list of {id, utterance, scores: {crit: {"control": s, "treatment": s}}},
    #         } }
    session_data: dict = {}

    with auth_patches, \
         open(results_path, "w", encoding="utf-8") as results_f, \
         open(failures_path, "w", encoding="utf-8") as fail_f, \
         open(csv_path, "w", encoding="utf-8", newline="") as csv_f:

        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(
            [
                "id",
                "utterance",
                "session_name",
                "criteria_name",
                "control_score",
                "treatment_score",
                "control_message",
                "treatment_message",
            ]
        )

        def write_record(record: dict) -> None:
            with results_lock:
                results_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                results_f.flush()
                if record.get("errors"):
                    fail_f.write(
                        json.dumps(
                            record["errors"] | {"id": record["id"], "utterance": record.get("utterance", "")},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    fail_f.flush()

                ctrl_rows = {
                    (r["session_name"], r["criteria_name"]): r
                    for r in flatten_scores(record.get("control") or {})
                }
                trt_rows = {
                    (r["session_name"], r["criteria_name"]): r
                    for r in flatten_scores(record.get("treatment") or {})
                }
                all_keys = sorted(set(ctrl_rows) | set(trt_rows))
                # Group by session for per-session TSV output and the
                # results.md summary. Cell values are TSV tokens (numeric
                # string, or NA/ERR/SKIP/-) that distinguish the reason a
                # score is missing on each arm.
                per_session: dict = {}
                for sname, cname in all_keys:
                    c = ctrl_rows.get((sname, cname))
                    t = trt_rows.get((sname, cname))
                    c_token = _cell_token(record, "control", c)
                    t_token = _cell_token(record, "treatment", t)
                    csv_writer.writerow(
                        [
                            record["id"],
                            record.get("utterance") or "",
                            sname,
                            cname,
                            c_token,
                            t_token,
                            (c or {}).get("message", ""),
                            (t or {}).get("message", ""),
                        ]
                    )
                    bucket = per_session.setdefault(sname, {})
                    bucket[cname] = {"control": c_token, "treatment": t_token}
                for sname, crit_scores in per_session.items():
                    sess = session_data.setdefault(
                        sname,
                        {"criteria": [], "criteria_set": set(), "rows": []},
                    )
                    for cname in crit_scores:
                        if cname not in sess["criteria_set"]:
                            sess["criteria_set"].add(cname)
                            sess["criteria"].append(cname)
                    sess["rows"].append(
                        {
                            "id": record["id"],
                            "utterance": record.get("utterance") or "",
                            "scores": crit_scores,
                        }
                    )
                csv_f.flush()

        def worker(utterance: str) -> None:
            t0 = time.perf_counter()
            try:
                record = process_pair(
                    utterance,
                    pairs[utterance],
                    qmap,
                    session_filters=session_filters,
                    criteria_filters=criteria_filters,
                )
            except Exception as e:  # noqa: BLE001
                record = {
                    "id": _utterance_id(utterance),
                    "utterance": utterance,
                    "errors": {"_": f"unhandled: {e}"},
                }
            write_record(record)
            with results_lock:
                completed[0] += 1
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if completed[0] % 50 == 0 or completed[0] == len(utterances):
                    overall = time.perf_counter() - started
                    print(f"  Progress: {completed[0]}/{len(utterances)} ({100*completed[0]//len(utterances)}%) - {overall:.0f}s elapsed")

        if args.workers <= 1:
            for u in utterances:
                worker(u)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(ex.map(worker, utterances))

    # ── Per-session TSV files + results.md summary ────────────────────────
    write_session_tsvs(session_data, sessions_root)
    write_results_md(session_data, summary_md_path)

    total_time = time.perf_counter() - started
    n_failures = sum(1 for s in session_data.values() for r in s["rows"] if not r["scores"])

    # Console summary
    print("")
    print(f"Done: {len(utterances)} utterances processed in {total_time:.1f}s")
    print(f"Sessions: {len(session_data)} | Failures: {failures_path}")
    print(f"Output: {args.output_dir}")
    print("")
    # Print abbreviated per-session aggregation to console
    for sname, sess in sorted(session_data.items(), key=lambda x: x[0].lower()):
        for cname in sess["criteria"]:
            ctrl = [r["scores"].get(cname, {}).get("control") for r in sess["rows"]]
            trt = [r["scores"].get(cname, {}).get("treatment") for r in sess["rows"]]
            cm, cn = _aggregate(ctrl)
            tm, tn = _aggregate(trt)
            c_str = f"{cm:.2f}" if cm is not None else "—"
            t_str = f"{tm:.2f}" if tm is not None else "—"
            print(f"  {sname}/{cname}: control={c_str} treatment={t_str} (n={cn}/{tn})")

    _log(f"  results : {results_path}", debug=args.debug)
    _log(f"  csv     : {csv_path}", debug=args.debug)
    _log(f"  failures: {failures_path}", debug=args.debug)
    _log(f"  sessions: {sessions_root}", debug=args.debug)
    _log(f"  summary : {summary_md_path}", debug=args.debug)


if __name__ == "__main__":
    main()
