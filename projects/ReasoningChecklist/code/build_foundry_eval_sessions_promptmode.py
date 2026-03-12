"""
build_foundry_eval_sessions_promptmode.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build Foundry evaluation session JSON files from Sydney scraper outputs,
storing the trimmed scraper results as prompt-based inference so that
all utterances are visible in Playground and evaluation scores are
extracted by the standard (unmodified) check script.

Three-step pipeline:

1. **Parse queryset** — read queryset.tsv to map each utterance to
   its session file(s) and collect the set of needed utterances.
2. **Stage scraper outputs** — load scraper JSON files in parallel,
   trim each response body, and write per-session staging files.
3. **Pack sessions** — for each base session, match staged results
   to dataItem indices and store them as prompt-based inference
   keyed by experiment arm name.  Write packed session JSONs.

The packed sessions are ready for parse-only evaluation runs
(runType 4 — ParseInference).

Differences from build_foundry_eval_sessions.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Scraper results are stored in the **prompt** inference slot
   (``inference[]`` with ``promptId``) instead of the Sydney slot
   (``sydneyInference[]`` with ``sydneyId``).
2. Empty prompts are created for each experiment arm so Foundry
   recognises the prompt-based structure.
3. No ``sydneyDetails`` or ``--config`` injection — not needed
   in prompt mode.
4. All utterances are visible in Playground (Sydney mode caps at 100).
5. Evaluation scores land in ``evaluation`` (not ``sydneyEvaluation``),
   so the standard check script works without modification.

Arguments
~~~~~~~~~
Required:
    --queryset       Path to queryset.tsv.
    --sessions-dir   Root directory containing base session JSON files.
                     Session files may be in subdirectories; paths in
                     the queryset and scraper segment fields are relative
                     to this directory.
    --scraper-dir    Directory containing Sydney scraper output JSON files.
    --output-dir     Output directory for packed session JSON files.

Optional:
    --exp-prompt-map  Comma-separated exp_name:index pairs.
                      Default: "control:0,experiment:1".
    --trim-level      2 or 3 (default: 3).
    --dry-run         Show stats without writing files.
    --workers         Threads for loading scraper files (default: 8).
    --staging-dir     Fixed staging directory (reuse if non-empty).
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Queryset parsing
# ---------------------------------------------------------------------------

def parse_queryset(queryset_path):
    """Parse queryset.tsv into a per-session validation lookup.

    Returns:
        queryset_check : dict
            ``session_file → {utterance → input_data}``
        needed_utterances : set
            All unique utterance strings.
        query_count : int
            Total (session_file, utterance) entries.
    """
    queryset_check = defaultdict(dict)
    needed_utterances = set()
    query_count = 0

    with open(queryset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            query_text = row["query"]

            try:
                inputs_list = json.loads(row["inputs"])
            except (json.JSONDecodeError, KeyError):
                print(f"WARNING: bad inputs JSON for query "
                      f"'{query_text[:50]}\u2026'")
                continue

            for inp in inputs_list:
                session_file = inp.get("file", "").replace("\\", "/")
                input_data = inp.get("input", {})
                utterance = input_data.get("utterance", query_text).strip()

                queryset_check[session_file][utterance] = input_data
                needed_utterances.add(utterance)
                query_count += 1

    return dict(queryset_check), needed_utterances, query_count


# ---------------------------------------------------------------------------
# Scraper output staging
# ---------------------------------------------------------------------------

def _load_one_scraper_file(filepath, needed_utterances, trim_level):
    """Load, trim, and pre-serialise a single scraper JSON file.

    Returns a dict with routing info and the trimmed response,
    ``"skipped"`` if not needed, or ``None`` on error.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None

    query = data.get("query", {})
    segment = query.get("segment", "")
    utterance = query.get("id", "").strip()

    # query.id may be a JSON array like [{"text": "...", "author": "user"}]
    # (CWC scraper format) — extract plain text for matching.
    # Try JSON-parsing unconditionally; only accept if it yields a non-empty
    # list of dicts whose first element has a "text" key.
    try:
        parsed = json.loads(utterance)
        if (isinstance(parsed, list) and parsed
                and isinstance(parsed[0], dict) and "text" in parsed[0]):
            utterance = parsed[0]["text"].strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        pass  # not JSON — keep original utterance as-is

    if not segment or not utterance:
        return None

    if needed_utterances is not None and utterance not in needed_utterances:
        return "skipped"

    requests = data.get("requests", [])
    if not requests:
        return None

    # exp_name may live at top level (old Seval) or in requests[0] (new Seval)
    exp_name = data.get("exp_name") or requests[0].get("exp_name", "")

    response_body = requests[0].get("response_body", {})

    if trim_level is not None:
        response_body = trim_sydney_response(response_body, trim_level)

    session_files = [
        s.strip().replace("\\", "/") for s in segment.split(",") if s.strip()
    ]

    return {
        "exp_name": exp_name,
        "utterance": utterance,
        "session_files": session_files,
        "response_body_json": json.dumps(response_body),
    }


def stage_scraper_outputs(scraper_dir, staging_dir, needed_utterances=None,
                          trim_level=None, num_workers=8):
    """Load scraper files in parallel, trim, and stage per-session.

    Staging layout::

        staging_dir/{exp_name}/{session_path}.tsv

    where session_path may include subdirectories (e.g.
    ``subdir/session-file.json``).

    Returns:
        (sessions_with_data, stats)
    """
    # Resolve the actual directory containing scraper JSON files.
    # CWC scraper nests responses under sydney_raw_responses/.
    sydney_sub = os.path.join(scraper_dir, "sydney_raw_responses")
    if os.path.isdir(sydney_sub):
        scraper_dir = sydney_sub
        print(f"  Using subdirectory: sydney_raw_responses/")

    pattern = os.path.join(scraper_dir, "*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"WARNING: No JSON files found in {scraper_dir}")
        return set(), {"loaded": 0, "skipped": 0, "errors": 0}

    staging_handles = {}
    sessions_with_data = set()
    skipped = loaded = errors = done = 0

    print(f"  Staging {len(files)} scraper files "
          f"with {num_workers} threads...")

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _load_one_scraper_file, f, needed_utterances, trim_level
                ): f
                for f in files
            }
            for future in as_completed(futures):
                done += 1
                if done % 2000 == 0 or done == len(files):
                    print(f"  Processed: {done}/{len(files)} "
                          f"(staged {loaded}, skipped {skipped})...",
                          flush=True)

                result = future.result()
                if result is None:
                    errors += 1
                elif result == "skipped":
                    skipped += 1
                else:
                    exp_name = result["exp_name"]
                    line = (result["utterance"] + "\t"
                            + result["response_body_json"] + "\n")
                    for sf in result["session_files"]:
                        key = (exp_name, sf)
                        if key not in staging_handles:
                            path = os.path.join(
                                staging_dir, exp_name, sf + ".tsv")
                            os.makedirs(
                                os.path.dirname(path), exist_ok=True)
                            staging_handles[key] = open(
                                path, "w", encoding="utf-8")
                        staging_handles[key].write(line)
                        sessions_with_data.add(sf)
                    loaded += 1
    finally:
        for fh in staging_handles.values():
            fh.close()

    if skipped:
        print(f"  Skipped {skipped} scraper files not in queryset")
    if errors:
        print(f"  {errors} scraper files had parse errors")
    print(f"  Staged {loaded} scraper results -> "
          f"{len(staging_handles)} session/arm files, "
          f"{len(sessions_with_data)} sessions")

    return sessions_with_data, {
        "loaded": loaded, "skipped": skipped, "errors": errors,
    }


# ---------------------------------------------------------------------------
# Sydney response trimming
# ---------------------------------------------------------------------------

def trim_sydney_response(response_body, trim_level=3):
    """Trim a Sydney response_body to the minimum needed by pre-eval.

    Args:
        response_body: dict with telemetry.metrics
        trim_level:    2 = filter metrics only, 3 = also trim output JSON

    Note:
        Only ``trim_level == 2`` is checked explicitly; any other value
        (including the default 3) falls through to the full trim path.
        The CLI ``--trim-level`` argument restricts choices to 2 or 3,
        and the caller ``_load_one_scraper_file`` skips this function
        entirely when ``trim_level is None``.
    """
    metrics = response_body.get("telemetry", {}).get("metrics", [])

    relevant = [
        m for m in metrics
        if m.get("serviceName") == "DeepLeoImprovedNetworking"
        and "fluxv3:invokingfunction" in m.get("output", "")
    ]

    if trim_level == 2:
        return {"telemetry": {"metrics": relevant}}

    trimmed_metrics = []
    for m in relevant:
        output_str = m.get("output", "")
        json_start = output_str.find("{")
        output_json = None
        prefix = ""

        while json_start != -1:
            try:
                parsed = json.loads(output_str[json_start:])
                output_json = parsed
                prefix = output_str[:json_start]
                break
            except json.JSONDecodeError:
                json_start = output_str.find("{", json_start + 1)

        if output_json:
            trimmed_json = {
                "promptTokenCount": output_json.get(
                    "promptTokenCount", 0),
                "completionTokenCount": output_json.get(
                    "completionTokenCount", 0),
                "toolInvocations": output_json.get(
                    "toolInvocations", []),
            }
            trimmed_metrics.append({
                "serviceName": m["serviceName"],
                "output": prefix + json.dumps(trimmed_json),
            })
        else:
            trimmed_metrics.append(m)

    return {"telemetry": {"metrics": trimmed_metrics}}


# ---------------------------------------------------------------------------
# Session processing — prompt-based output
# ---------------------------------------------------------------------------

def _process_one_session(session_file, queryset_utterances, sessions_dir,
                         exp_prompt_map, staging_dir, output_dir, dry_run):
    """Pack scraper results as prompt-based inference into a session.

    Creates empty prompts for each experiment arm and stores scraper
    responses in ``resultsv2[].inference`` (prompt mode) so that *all*
    utterances are visible in the Playground data browser.

    Iterates through dataItems sequentially.  For each item, looks up
    its utterance in the staged scraper results and assigns the next
    available result via a per-utterance round-robin counter.  This
    naturally handles duplicate utterances (for variance reduction)
    without a separate grouping step.

    Args:
        session_file:        Base session filename.
        queryset_utterances: ``{utterance → input_data}`` for this
                             session, or ``None``.
        sessions_dir:        Directory containing base session JSONs.
        exp_prompt_map:      ``{exp_name → prompt_index}``.
        staging_dir:         Staging directory from Step 2.
        output_dir:          Where to write the packed session JSON.
        dry_run:             If ``True``, skip writing files.

    Returns:
        ``(session_file, stats_or_None, log_lines)``

        *stats* keys: ``matched``, ``total``, ``no_scraper``,
        ``empty_response``, ``unique_utterances``, ``data_items``,
        ``packed_size_kb``.
    """
    log = []

    # Load base session
    session_path = os.path.join(sessions_dir, session_file)
    if not os.path.isfile(session_path):
        log.append(f"  SKIP {session_file}: base session not found")
        return session_file, None, log

    try:
        with open(session_path, "r", encoding="utf-8") as fh:
            session = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        log.append(f"  WARNING: Could not parse {session_file}: {e}")
        return session_file, None, log

    session_inputs = session.get("sessionInputs", [])
    if not session_inputs:
        log.append(f"  SKIP {session_file}: no sessionInputs")
        return session_file, None, log

    data_items = session_inputs[0].get("dataItems", [])

    # Parse each dataItem's utterance up-front (shared across arms).
    # item_utts: list of (index, utterance) for items with valid input.
    item_utts = []
    all_utts = set()
    for di in data_items:
        idx = di.get("index")
        if idx is None:
            continue
        try:
            input_json = json.loads(di.get("input", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(input_json, dict):
            utt = input_json.get("utterance", "").strip()
            if utt:
                item_utts.append((idx, utt))
                all_utts.add(utt)

    # Validate queryset utterances against session dataItems
    qs_utts = queryset_utterances or {}
    mismatches = 0
    for utt in qs_utts:
        if utt not in all_utts:
            mismatches += 1
            if mismatches <= 3:
                log.append(f"  WARNING: queryset utterance not in "
                           f"{session_file}: '{utt[:60]}...'")
    if mismatches > 3:
        log.append(f"  WARNING: ... and {mismatches - 3} more queryset "
                   f"mismatches in {session_file}")

    # Build prompt-based inference results per experiment arm
    inference_list = []
    total_matched = 0
    total_no_scraper = 0
    total_empty_response = 0
    total_size_kb = 0.0

    for exp_name in sorted(exp_prompt_map.keys()):
        staging_path = os.path.join(
            staging_dir, exp_name, session_file + ".tsv")
        staged = defaultdict(list)   # utterance → [resp_json, …]
        if os.path.isfile(staging_path):
            with open(staging_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if "\t" not in line:
                        continue
                    utt, resp_json = line.split("\t", 1)
                    staged[utt].append(resp_json)

        if not staged:
            log.append(f"  WARNING: No staged scraper data for "
                       f"exp_name='{exp_name}' in {session_file}")
            continue

        # Pre-filter staged results: keep only non-empty responses.
        staged_non_empty = {}   # utterance → [non-empty resp_json, …]
        for utt, results in staged.items():
            non_empty = [
                r for r in results
                if r and r not in (
                    '{}', '{"telemetry":{"metrics":[]}}',
                )
            ]
            if non_empty:
                staged_non_empty[utt] = non_empty

        # Iterate dataItems sequentially with per-utterance round-robin.
        data_items_outputs = {}
        matched = no_scraper = empty_response = 0
        utt_counter = defaultdict(int)   # utterance → next index

        for idx, utt in item_utts:
            if utt in staged_non_empty:
                results = staged_non_empty[utt]
                ci = utt_counter[utt]
                data_items_outputs[str(idx)] = results[ci % len(results)]
                utt_counter[utt] = ci + 1
                matched += 1
            elif utt in staged:
                empty_response += 1
            elif utt in qs_utts:
                no_scraper += 1

        # Prompt-based inference result (not Sydney)
        inference_list.append({
            "promptId": exp_name,
            "dataItemsOutputs": data_items_outputs,
            "dataItemsUsage": {},
        })
        total_matched += matched
        total_no_scraper += no_scraper
        total_empty_response += empty_response
        total_size_kb += sum(
            len(v) for v in data_items_outputs.values()) / 1024

    if not inference_list:
        log.append(f"  SKIP {session_file}: no valid inference arms")
        return session_file, None, log

    num_arms = len(inference_list)

    if total_matched == 0:
        log.append(
            f"  SKIP {session_file}: 0/{len(item_utts) * num_arms} matched"
            f" (no_scraper={total_no_scraper},"
            f" empty_response={total_empty_response})")
        return session_file, None, log

    # Build packed session — prompt-based (not Sydney)
    packed = dict(session)      # shallow copy

    # Create empty prompts for each experiment arm
    packed["prompts"] = [
        {
            "id": exp_name,
            "title": exp_name,
            "prompt": "",
            "promptTemplate": "",
        }
        for exp_name in sorted(exp_prompt_map.keys())
        if any(ir["promptId"] == exp_name for ir in inference_list)
    ]

    packed["resultsv2"] = [{
        "sessionInputIndex": 0,
        "inference": inference_list,
        "evaluation": [],
    }]

    # Remove Sydney-specific fields (not needed in prompt mode)
    packed.pop("sydneyDetails", None)

    stats = {
        "matched": total_matched,
        "total": len(item_utts) * num_arms,
        "no_scraper": total_no_scraper,
        "empty_response": total_empty_response,
        "unique_utterances": len(all_utts),
        "data_items": len(item_utts),
        "packed_size_kb": total_size_kb,
    }

    detail = (f"  {session_file}: {stats['matched']}/{stats['total']} "
              f"matched ({stats['unique_utterances']} unique -> "
              f"{stats['data_items']} dataItems), "
              f"{stats['packed_size_kb']:.1f} KB")
    extras = []
    if stats["no_scraper"]:
        extras.append(f"no_scraper={stats['no_scraper']}")
    if stats["empty_response"]:
        extras.append(f"empty_response={stats['empty_response']}")
    if extras:
        detail += f" ({', '.join(extras)})"
    log.append(detail)

    if not dry_run:
        output_path = os.path.join(output_dir, session_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(packed, fh, indent=2, ensure_ascii=False)

    return session_file, stats, log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pack Sydney scraper results as prompt-based "
                    "inference into Foundry session JSONs for "
                    "runType 4 (ParseInference).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--queryset", required=True,
        help="Path to queryset.tsv (query->session mapping)",
    )
    parser.add_argument(
        "--sessions-dir", required=True,
        help="Directory containing base session JSON files",
    )
    parser.add_argument(
        "--scraper-dir", required=True,
        help="Directory containing Sydney scraper output JSON files",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for packed session JSON files",
    )
    parser.add_argument(
        "--exp-prompt-map", default="control:0,experiment:1",
        help="Comma-separated exp_name:index pairs.  Only exp_name is "
             "used to identify experiment arms (indices are kept for "
             "backward compat).  Default: 'control:0,experiment:1'.",
    )
    parser.add_argument(
        "--trim-level", type=int, default=3, choices=[2, 3],
        help="Trimming level: 2=relevant metrics only, "
             "3=trimmed output fields (default: 3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without writing files",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel threads for loading scraper files "
             "(default: 8)",
    )
    parser.add_argument(
        "--staging-dir", default=None,
        help="Fixed staging directory.  Reused if non-empty.  "
             "When not set, a temp directory is created and cleaned up.",
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.isfile(args.queryset):
        print(f"ERROR: Queryset not found: {args.queryset}")
        sys.exit(1)
    if not os.path.isdir(args.sessions_dir):
        print(f"ERROR: Sessions directory not found: {args.sessions_dir}")
        sys.exit(1)
    if not os.path.isdir(args.scraper_dir):
        print(f"ERROR: Scraper directory not found: {args.scraper_dir}")
        sys.exit(1)

    # Create output directory
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    # Parse exp-prompt-map
    exp_prompt_map = {}
    for pair in args.exp_prompt_map.split(","):
        pair = pair.strip()
        if ":" not in pair:
            print(f"ERROR: Invalid exp-prompt-map entry: '{pair}' "
                  f"(expected 'exp_name:prompt_index')")
            sys.exit(1)
        ename, pidx = pair.split(":", 1)
        exp_prompt_map[ename.strip()] = int(pidx.strip())
    print(f"Exp-prompt map: {exp_prompt_map}")
    print("Output mode: prompt-based inference (not Sydney)")

    # ---- Step 1. Parse queryset ----
    print("\n--- Step 1: Parse queryset ---")
    queryset_check, needed_utterances, query_count = parse_queryset(
        args.queryset)
    print(f"Queryset: {query_count} entries across "
          f"{len(queryset_check)} session files, "
          f"{len(needed_utterances)} unique utterances")

    # ---- Step 2. Stage scraper outputs to disk ----
    print("\n--- Step 2: Stage scraper outputs ---")

    cleanup_staging = False
    reuse_staging = False
    if args.staging_dir:
        staging_dir = args.staging_dir
        if os.path.isdir(staging_dir) and any(os.scandir(staging_dir)):
            reuse_staging = True
            print(f"  Reusing existing staging directory: {staging_dir}")
        else:
            os.makedirs(staging_dir, exist_ok=True)
            print(f"  Staging directory (persistent): {staging_dir}")
    else:
        staging_dir = tempfile.mkdtemp(prefix="foundry_staging_")
        cleanup_staging = True
        print(f"  Staging directory (temp): {staging_dir}")

    # Wrap Steps 2–3 in try/finally to guarantee the temp staging
    # directory is cleaned up even if an error occurs mid-pipeline.
    try:
        if reuse_staging:
            sessions_with_data = set()
            for exp_name in exp_prompt_map:
                exp_dir = os.path.join(staging_dir, exp_name)
                if os.path.isdir(exp_dir):
                    for tsv_file in glob.glob(
                            os.path.join(exp_dir, "**", "*.tsv"),
                            recursive=True):
                        sf = os.path.relpath(
                            tsv_file, exp_dir).replace("\\", "/")
                        if sf.endswith(".tsv"):
                            sf = sf[:-4]
                        sessions_with_data.add(sf)
            scraper_stats = {
                "loaded": "reused", "skipped": "\u2013",
                "errors": "\u2013",
            }
            print(f"  Found staging data for "
                  f"{len(sessions_with_data)} sessions")
        else:
            sessions_with_data, scraper_stats = stage_scraper_outputs(
                args.scraper_dir,
                staging_dir,
                needed_utterances=needed_utterances,
                trim_level=args.trim_level,
                num_workers=args.workers,
            )

        if not sessions_with_data:
            print("ERROR: No scraper outputs staged. Check --scraper-dir.")
            sys.exit(1)

        # ---- Step 3. Pack each session ----
        print(f"\n--- Step 3: Pack {len(sessions_with_data)} sessions "
              f"(prompt mode) ---")

        results = []
        for sf in sorted(sessions_with_data):
            qs_utts = queryset_check.get(sf)
            result = _process_one_session(
                sf, qs_utts, args.sessions_dir, exp_prompt_map,
                staging_dir, args.output_dir, args.dry_run,
            )
            results.append(result)

        # Print results and accumulate totals
        total_matched = 0
        total_queries = 0
        total_sessions = 0
        total_size_kb = 0.0
        total_no_scraper = 0
        total_empty_response = 0

        for session_file, stats, log_lines in sorted(
                results, key=lambda r: r[0]):
            for line in log_lines:
                print(line)
            if stats is not None:
                total_sessions += 1
                total_matched += stats["matched"]
                total_queries += stats["total"]
                total_size_kb += stats["packed_size_kb"]
                total_no_scraper += stats["no_scraper"]
                total_empty_response += stats["empty_response"]

        # ---- Summary ----
        print(f"\n{'=== DRY RUN ===' if args.dry_run else '=== Summary ==='}")
        print(f"Output mode: prompt-based inference")
        print(f"Experiment arms: {list(exp_prompt_map.keys())}")
        print(f"Sessions packed: {total_sessions}/{len(sessions_with_data)}")
        print(f"Scraper files: loaded={scraper_stats['loaded']}, "
              f"skipped={scraper_stats['skipped']}, "
              f"errors={scraper_stats['errors']}")
        print(f"Queries matched (with data): {total_matched}/{total_queries}")
        if total_no_scraper:
            print(f"In queryset but no scraper result: {total_no_scraper}")
        if total_empty_response:
            print(f"Scraper result empty/no-data: {total_empty_response}")
        print(f"Total packed size: {total_size_kb:.1f} KB")
        print(f"Trim level: {args.trim_level}")
        if not args.dry_run:
            print(f"Output directory: {args.output_dir}")

    finally:
        if cleanup_staging:
            shutil.rmtree(staging_dir, ignore_errors=True)
            print(f"Cleaned up staging directory")
        else:
            print(f"Staging directory preserved: {staging_dir}")


if __name__ == "__main__":
    main()
