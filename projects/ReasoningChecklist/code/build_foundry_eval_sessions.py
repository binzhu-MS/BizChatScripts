"""
build_foundry_eval_sessions.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build Foundry evaluation session JSON files from Sydney scraper outputs,
ready for parse-only evaluation runs (runType 4 — ParseInference).

Given a queryset.tsv (source of truth for query→session mapping), a set of
base Foundry session templates, and a directory of Sydney scraper output
files, this script:

  1. Parses the queryset to determine which queries belong to which session.
  2. Loads and trims the scraper outputs to the minimum data the pre-eval
     script needs (DeepLeoImprovedNetworking metrics with
     fluxv3:invokingfunction — ~2.4 KB per item at trim level 3).
  3. Maps trimmed results to the correct data-item indices in each base
     session, creating one inference entry per experiment arm per query.
  4. Optionally injects ``sydneyDetails`` from config.json so the Foundry
     session records the exact Sydney configuration used during scraping.
  5. Writes packed session files to the output directory.

Mapping chain
~~~~~~~~~~~~~
queryset.tsv
  segment column ─────────────────┐   (raw value, may be comma-separated)
  inputs[].file  ─── session file │
  inputs[].input.utterance ───┐   │
                              │   │
scraper output                │   │
  query.segment  ─────────────┼───┘   (matches segment column exactly)
  query.id       ─────────────┘       (matches utterance)

base session JSON
  sessionInputs[0].dataItems[i].input  →  {"utterance": "..."} → match

Arguments:
    Required:
        --queryset      Path to queryset.tsv — the source of truth that maps
                        each query (utterance) to its session file(s) and the
                        scraper segment key.  Columns: query, segment, inputs.
        --sessions-dir  Directory containing base session JSON files (the
                        Foundry session templates with prompts, dataItems,
                        evaluationStrategy, etc.).
        --scraper-dir   Directory containing Sydney scraper output JSON files.
                        Each file has query.segment, query.id (utterance),
                        exp_name, and requests[0].response_body.
        --output-dir    Directory where packed session JSON files will be
                        written (created automatically if it doesn't exist).

    Optional:
        --config        Path to config.json from the scraper config dataset.
                        If provided, the Sydney settings from each exp_config
                        entry are injected into the generated session files as
                        ``sydneyDetails`` — documenting the exact Sydney
                        configuration that produced the scraped results.
                        Maps config fields to Foundry's SydneyDetail schema:
                        sydney.url → endpointType, sydney.option_sets →
                        configuration.optionsSets, sydney.plugins →
                        configuration.plugins, sydney.variants → variants,
                        sydney.extra_params.mockAppId → mockAppId, etc.
                        Default: not set (no injection).
        --exp-prompt-map Comma-separated exp_name:prompt_index pairs that map
                        each experiment arm to a prompt in the session.
                        Use a single entry (e.g. "control:0") to pack only
                        one arm.
                        Default: "control:0,experiment:1".
        --trim-level    How aggressively to trim Sydney response data.
                        2 = keep only DeepLeoImprovedNetworking metric entries
                            whose output contains "fluxv3:invokingfunction"
                            (~274 KB per item).
                        3 = also parse the output JSON and keep only
                            promptTokenCount, completionTokenCount, and
                            toolInvocations (~2.4 KB per item).
                        Default: 3.
        --dry-run       Show matching statistics without writing any files.
                        Default: off.
        --workers       Number of parallel threads for loading scraper files
                        and packing sessions.  Default: 8.

Usage:
    python build_foundry_eval_sessions.py \\
        --queryset <queryset.tsv> \\
        --sessions-dir <base-sessions-dir> \\
        --scraper-dir <scraper-output-dir> \\
        --output-dir <output-dir> \\
        [--config <config.json>] \\
        [--exp-prompt-map control:0,experiment:1] \\
        [--trim-level 3] \\
        [--dry-run] \\
        [--workers 8]

Examples:
    # Pack both control and experiment (default)
    python build_foundry_eval_sessions.py \\
        --queryset ../../local/Foundry_Data/config_dataset/queryset.tsv \\
        --sessions-dir sessions/base-sessions \\
        --scraper-dir ../../local/Foundry_Data/scraper_output_dataset \\
        --output-dir sessions/packed-sessions \\
        --config ../../local/Foundry_Data/config_dataset/config.json

    # Pack only control arm
    python build_foundry_eval_sessions.py \\
        --queryset ../../local/Foundry_Data/config_dataset/queryset.tsv \\
        --sessions-dir sessions/base-sessions \\
        --scraper-dir ../../local/Foundry_Data/scraper_output_dataset \\
        --output-dir sessions/packed-sessions \\
        --config ../../local/Foundry_Data/config_dataset/config.json \\
        --exp-prompt-map control:0
"""

import argparse
import copy
import csv
import glob
import json
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Queryset parsing
# ---------------------------------------------------------------------------

def parse_queryset(queryset_path):
    """
    Parse queryset.tsv and build two mappings.

    Returns:
        session_queries : dict  session_file → list of
            {"utterance": str, "segment_key": str, "input_data": dict}

            *segment_key* is the raw ``segment`` column value from
            queryset.tsv – it may be comma-separated for multi-session
            queries and must match the scraper output's ``query.segment``
            exactly.

        query_count : int   total number of (session_file, utterance) entries
    """
    session_queries = defaultdict(list)
    query_count = 0

    with open(queryset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_segment = row["segment"]           # e.g. "a.json, b.json"
            query_text = row["query"]

            try:
                inputs_list = json.loads(row["inputs"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"WARNING: bad inputs JSON for query "
                      f"'{query_text[:50]}…': {e}")
                continue

            for inp in inputs_list:
                session_file = inp.get("file", "")
                input_data = inp.get("input", {})
                utterance = input_data.get("utterance", query_text).strip()

                session_queries[session_file].append({
                    "utterance": utterance,
                    "segment_key": raw_segment,    # for scraper lookup
                    "input_data": input_data,
                })
                query_count += 1

    return dict(session_queries), query_count


# ---------------------------------------------------------------------------
# Scraper output loading
# ---------------------------------------------------------------------------

def _load_one_scraper_file(filepath, needed_keys, trim_level):
    """Load and optionally trim a single scraper JSON file.

    Returns a dict with parsed data, the string ``"skipped"`` if the
    file's key is not in *needed_keys*, or ``None`` on error.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    file_exp = data.get("exp_name", "")
    query = data.get("query", {})
    segment = query.get("segment", "")
    utterance = query.get("id", "").strip()

    if not segment or not utterance:
        return None

    if needed_keys is not None and (segment, utterance) not in needed_keys:
        return "skipped"

    requests = data.get("requests", [])
    if not requests:
        return None

    response_body = requests[0].get("response_body", {})

    pre_trimmed = False
    if trim_level is not None:
        response_body = trim_sydney_response(response_body, trim_level)
        pre_trimmed = True

    return {
        "exp_name": file_exp,
        "key": (segment, utterance),
        "entry": {
            "response_body": response_body,
            "file": os.path.basename(filepath),
            "exp_name": file_exp,
            "pre_trimmed": pre_trimmed,
        },
    }


def load_scraper_outputs(scraper_dir, needed_keys=None, trim_level=None,
                         num_workers=8):
    """
    Load scraper output files in parallel and group them by exp_name.

    Uses a thread pool to read and parse files concurrently, which
    dramatically speeds up I/O-bound loading of thousands of JSON files.

    Args:
        scraper_dir:  Directory containing scraper output JSON files
        needed_keys:  Optional set of ``(segment, utterance)`` tuples.
                      When provided, only scraper entries whose key is in
                      this set are kept — dramatically reducing memory for
                      large datasets.
        trim_level:   When set (2 or 3), trim each ``response_body``
                      immediately after loading so that only the compact
                      trimmed form is stored in memory.
        num_workers:  Number of threads in the pool (default 8).

    Returns:
        Dict mapping exp_name → {(segment, utterance) → {"response_body": …, …}}
    """
    outputs = defaultdict(dict)
    pattern = os.path.join(scraper_dir, "*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"WARNING: No JSON files found in {scraper_dir}")
        return dict(outputs)

    skipped = 0
    loaded = 0
    errors = 0
    done = 0

    print(f"  Loading {len(files)} scraper files with "
          f"{num_workers} threads…")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _load_one_scraper_file, f, needed_keys, trim_level
            ): f
            for f in files
        }
        for future in as_completed(futures):
            done += 1
            if done % 2000 == 0 or done == len(files):
                print(f"  Processed scraper files: {done}/{len(files)} "
                      f"(kept {loaded}, skipped {skipped})…",
                      flush=True)

            result = future.result()
            if result is None:
                errors += 1
            elif result == "skipped":
                skipped += 1
            else:
                outputs[result["exp_name"]][result["key"]] = result["entry"]
                loaded += 1

    if skipped:
        print(f"  Skipped {skipped} scraper files not matching queryset")
    if errors:
        print(f"  {errors} scraper files had parse errors")
    for ename in sorted(outputs):
        print(f"Loaded {len(outputs[ename])} scraper outputs "
              f"(exp_name={ename})")
    return dict(outputs)


# ---------------------------------------------------------------------------
# config.json → sydneyDetails mapping
# ---------------------------------------------------------------------------

# Map known Sydney endpoint URLs to Foundry SydneyEndpointType enum values.
# The backend resolves these to appsettings config section names matching
# SYDNEY_AVALON_CAFE, SYDNEY_FRONTIER_SDF_WESTUS, etc.
_URL_TO_ENDPOINT_TYPE = {
    "https://substrate.office.com/m365copilot":
        "SydneyAvalonCafe",
    "https://substrate-sdf.office.com/m365copilot":
        "SydneyAvalonSdfCafe",
}


def _resolve_endpoint_type(url):
    """Map a Sydney URL from config.json to a Foundry SydneyEndpointType.

    Falls back to ``"SydneyAvalonCafe"`` for unknown URLs.
    """
    normalised = url.rstrip("/").lower()
    return _URL_TO_ENDPOINT_TYPE.get(normalised, "SydneyAvalonCafe")


def load_sydney_configs(config_path, exp_prompt_map):
    """Load config.json and build a list of SydneyDetail dicts.

    Only the exp_configs whose ``exp_name`` appears in *exp_prompt_map* are
    included.  The returned list is ordered identically to ``sydneyDetails``
    in the Foundry session schema and matches the ``SydneyDetail`` C# class:

        id, owners, title, configuration (SydneyPostBody), endpointType,
        variants, mockAppId, customHeaders, agentId, runtimeId

    ``configuration`` is a Foundry ``SydneyPostBody``.  We only populate the
    fields that the scraper config explicitly specifies (optionsSets, plugins,
    scenario, tone) and leave the rest null so the defaults apply.

    Args:
        config_path:    Path to config.json from the scraper config dataset.
        exp_prompt_map: dict mapping exp_name → prompt_index (only these
                        exp_configs are included).

    Returns:
        List of ``SydneyDetail``-shaped dicts (one per matched exp_config),
        or an empty list if loading fails.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        print(f"WARNING: Could not load config {config_path}: {e}")
        return []

    conversations = config.get("conversations", {})
    exp_configs = conversations.get("exp_configs", [])
    if not exp_configs:
        print("WARNING: No exp_configs found in config.json")
        return []

    sydney_details = []
    for ec in exp_configs:
        ename = ec.get("exp_name", "")
        if ename not in exp_prompt_map:
            continue

        syd = ec.get("sydney", {})
        if not syd:
            print(f"WARNING: No sydney settings for exp_name='{ename}'")
            continue

        # Build option_sets → optionsSets list
        options_sets_str = syd.get("option_sets", "")
        options_sets = [
            s.strip() for s in options_sets_str.split(",") if s.strip()
        ] if options_sets_str else []

        # Extra params
        extra_params = syd.get("extra_params", {})
        scenario = extra_params.get("scenario")
        mock_app_id = extra_params.get("mockAppId", "")

        # chat_request_override (tone, etc.)
        chat_override = ec.get("chat_request_override", {})
        tone = chat_override.get("tone")

        # Build the SydneyPostBody (configuration).
        # Only populate fields present in the scraper config; leave
        # everything else null so Foundry uses its defaults.
        configuration = {
            "message": None,
            "optionsSets": options_sets or None,
            "plugins": syd.get("plugins") or None,
            "scenario": scenario,
            "tone": tone,
            "sliceIds": None,
            "gpts": None,
        }

        # Resolve endpoint type from URL
        url = syd.get("url", "")
        endpoint_type = _resolve_endpoint_type(url)

        # Variants string
        variants = syd.get("variants", "") or None

        # Build SydneyDetail
        sydney_detail = {
            "id": ename,
            "owners": [],
            "title": ename,
            "configuration": configuration,
            "endpointType": endpoint_type,
            "variants": variants,
            "mockAppId": mock_app_id,
        }
        sydney_details.append(sydney_detail)
        print(f"  Loaded Sydney config for exp_name='{ename}' "
              f"(endpoint={endpoint_type}, "
              f"optionsSets={len(options_sets)}, "
              f"variants={'yes' if variants else 'none'})")

    return sydney_details


# ---------------------------------------------------------------------------
# Sydney response trimming
# ---------------------------------------------------------------------------

def trim_sydney_response(response_body, trim_level=3):
    """
    Trim a Sydney response_body to the minimum needed by the pre-eval script.

    The pre-eval script needs ``telemetry.metrics`` entries where
    ``serviceName == "DeepLeoImprovedNetworking"`` and ``output`` contains
    ``"fluxv3:invokingfunction"``.  From those entries' output JSON it reads
    ``promptTokenCount``, ``completionTokenCount``, and ``toolInvocations``.

    Args:
        response_body: dict with telemetry.metrics
        trim_level:    2 = filter metrics only, 3 = also trim output JSON
    """
    metrics = response_body.get("telemetry", {}).get("metrics", [])

    relevant = [
        m for m in metrics
        if m.get("serviceName") == "DeepLeoImprovedNetworking"
        and "fluxv3:invokingfunction" in m.get("output", "")
    ]

    if trim_level == 2:
        return {"telemetry": {"metrics": relevant}}

    # Level 3: also trim the output JSON within each metric entry
    trimmed_metrics = []
    for m in relevant:
        output_str = m.get("output", "")

        # Parse JSON part after the "CallTags: ..., " prefix
        json_start = output_str.find("{")
        output_json = None
        prefix = ""

        while json_start != -1:
            try:
                output_json = json.loads(output_str[json_start:])
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
# Data-item index lookup
# ---------------------------------------------------------------------------

def build_utterance_index(data_items):
    """
    Build a dict mapping utterance text → data-item index for fast lookup.

    Data items store the utterance as a JSON string in their ``input`` field:
    ``'{"utterance": "...", ...}'``.
    """
    idx_map = {}
    for di in data_items:
        idx = di.get("index")
        if idx is None:
            continue
        input_str = di.get("input", "")
        if input_str:
            try:
                input_json = json.loads(input_str)
                if isinstance(input_json, dict):
                    utt = input_json.get("utterance", "").strip()
                    if utt:
                        idx_map[utt] = idx
            except (json.JSONDecodeError, TypeError):
                pass
        # Fallback: variables array
        if input_str not in idx_map:
            for var in di.get("variables", []):
                if var.get("name", "").lower() in (
                        "utterance", "query", "input"):
                    idx_map[var.get("value", "")] = idx
                    break
    return idx_map


# ---------------------------------------------------------------------------
# Session packing
# ---------------------------------------------------------------------------

def pack_session(session, queries, inference_arms, trim_level=3):
    """
    Pack trimmed Sydney results into a session using queryset-driven mapping.

    For each inference arm (exp_name → prompt), for each query entry from
    queryset.tsv that references this session:
      1. Find the matching data-item index by utterance
      2. Look up the scraper output by (segment_key, utterance)
      3. Trim and store in dataItemsOutputs

    This produces one inference entry per arm in ``resultsv2``.

    Args:
        session:         base session dict (deep-copied internally)
        queries:         list of queryset entries for this session
                         [{"utterance", "segment_key", "input_data"}, ...]
        inference_arms:  list of dicts, each with:
                         {"prompt_id": str,
                          "scraper_outputs": dict (segment, utterance) → data}
        trim_level:      2 or 3

    Returns:
        (packed_session, stats)
    """
    packed = copy.deepcopy(session)

    session_inputs = packed.get("sessionInputs", [])
    if not session_inputs:
        return packed, {"matched": 0, "total": len(queries),
                        "missing_in_session": 0,
                        "missing_in_scraper": 0,
                        "packed_size_kb": 0}

    data_items = session_inputs[0].get("dataItems", [])
    utt_index = build_utterance_index(data_items)

    inference_list = []
    total_matched = 0
    total_missing_session = 0
    total_missing_scraper = 0
    total_size_kb = 0

    for arm in inference_arms:
        prompt_id = arm["prompt_id"]
        scraper_outputs = arm["scraper_outputs"]

        data_items_outputs = {}
        matched = 0
        missing_in_session = 0
        missing_in_scraper = 0

        for qentry in queries:
            utterance = qentry["utterance"]
            segment_key = qentry["segment_key"]

            # 1. Find data-item index
            idx = utt_index.get(utterance)
            if idx is None:
                missing_in_session += 1
                continue

            # 2. Scraper lookup using (raw segment, utterance)
            scraper_key = (segment_key, utterance)
            if scraper_key not in scraper_outputs:
                missing_in_scraper += 1
                continue

            # 3. Trim (if not already pre-trimmed) and pack
            response_body = scraper_outputs[scraper_key]["response_body"]
            if scraper_outputs[scraper_key].get("pre_trimmed"):
                trimmed = response_body
            else:
                trimmed = trim_sydney_response(response_body, trim_level)
            data_items_outputs[str(idx)] = json.dumps(trimmed)
            matched += 1

        inference_list.append({
            "promptId": prompt_id,
            "dataItemsOutputs": data_items_outputs,
            "dataItemsUsage": {},
        })
        total_matched += matched
        total_missing_session += missing_in_session
        total_missing_scraper += missing_in_scraper
        total_size_kb += len(json.dumps(data_items_outputs)) / 1024

    # Build resultsv2
    packed["resultsv2"] = [{
        "sessionInputIndex": 0,
        "inference": inference_list,
        "evaluation": [],
        "sydneyInference": None,
        "sydneyEvaluation": None,
    }]

    num_arms = len(inference_arms)
    stats = {
        "matched": total_matched,
        "total": len(queries) * num_arms,
        "missing_in_session": total_missing_session,
        "missing_in_scraper": total_missing_scraper,
        "packed_size_kb": total_size_kb,
    }
    return packed, stats


# ---------------------------------------------------------------------------
# Session processing (thread worker)
# ---------------------------------------------------------------------------

def _process_one_session(session_file, queries, sessions_dir, exp_prompt_map,
                         scraper_outputs, trim_level, sydney_details,
                         output_dir, dry_run):
    """Load a base session, pack scraper results into it, and write output.

    Designed to run in a thread pool.  All output is collected into a list
    of log lines (instead of printing directly) so the caller can present
    results in deterministic order.

    Returns:
        (session_file, stats_or_None, log_lines)
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

    prompts = session.get("prompts", [])

    # Build inference arms from exp-prompt-map
    inference_arms = []
    for ename in sorted(exp_prompt_map.keys()):
        pidx = exp_prompt_map[ename]
        if pidx >= len(prompts):
            log.append(
                f"  WARNING: Prompt index {pidx} (for {ename}) "
                f"out of range in {session_file} "
                f"({len(prompts)} prompts)")
            continue
        prompt_id = prompts[pidx].get("id", "")
        if not prompt_id:
            log.append(
                f"  WARNING: No prompt ID at index "
                f"{pidx} in {session_file}")
            continue
        if ename not in scraper_outputs:
            log.append(
                f"  WARNING: No scraper outputs for "
                f"exp_name='{ename}'")
            continue
        inference_arms.append({
            "prompt_id": prompt_id,
            "scraper_outputs": scraper_outputs[ename],
        })

    if not inference_arms:
        log.append(f"  SKIP {session_file}: no valid inference arms")
        return session_file, None, log

    # Pack
    packed_session, stats = pack_session(
        session, queries, inference_arms, trim_level,
    )

    # Inject sydneyDetails from config.json (if provided)
    if sydney_details:
        packed_session["sydneyDetails"] = copy.deepcopy(sydney_details)

    if stats["matched"] == 0:
        log.append(
            f"  SKIP {session_file}: 0/{stats['total']} matched"
            f" (session={stats['missing_in_session']}"
            f" scraper={stats['missing_in_scraper']})")
        return session_file, None, log

    # Per-session summary line
    detail = (f"  {session_file}: {stats['matched']}/{stats['total']} "
              f"matched, {stats['packed_size_kb']:.1f} KB")
    if stats["missing_in_session"] or stats["missing_in_scraper"]:
        detail += (f" (miss: session={stats['missing_in_session']}"
                   f" scraper={stats['missing_in_scraper']})")
    log.append(detail)

    # Write output file
    if not dry_run:
        output_path = os.path.join(output_dir, session_file)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(packed_session, fh, indent=2, ensure_ascii=False)

    return session_file, stats, log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pack trimmed Sydney scraper results into Foundry "
                    "session JSONs for runType 4 (ParseInference).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--queryset", required=True,
        help="Path to queryset.tsv (source of truth for query→session mapping)",
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
        "--config", default=None,
        help="Path to config.json from the scraper config dataset.  "
             "When provided, Sydney settings from each exp_config are "
             "injected into packed sessions as sydneyDetails.",
    )
    parser.add_argument(
        "--exp-prompt-map", default="control:0,experiment:1",
        help="Comma-separated exp_name:prompt_index pairs that map each "
             "experiment arm to a prompt in the session. "
             "Use a single entry (e.g. 'control:0') to pack only one arm. "
             "Default: 'control:0,experiment:1'.",
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
             "and packing sessions (default: 8)",
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

    # ---- 1. Parse queryset.tsv ----
    session_queries, query_count = parse_queryset(args.queryset)
    print(f"Queryset: {query_count} entries across "
          f"{len(session_queries)} session files")

    # ---- 2. Load scraper outputs ----
    # Build the set of needed (segment, utterance) keys so that
    # load_scraper_outputs can skip irrelevant files and trim eagerly.
    needed_keys = set()
    for queries in session_queries.values():
        for q in queries:
            needed_keys.add((q["segment_key"], q["utterance"]))
    print(f"Unique scraper keys needed: {len(needed_keys)}")

    scraper_outputs = load_scraper_outputs(
        args.scraper_dir,
        needed_keys=needed_keys,
        trim_level=args.trim_level,
        num_workers=args.workers,
    )
    if not scraper_outputs:
        print("ERROR: No scraper outputs loaded. Check --scraper-dir.")
        sys.exit(1)

    # ---- 2b. Parse exp-prompt-map ----
    exp_prompt_map = {}  # exp_name → prompt_index
    for pair in args.exp_prompt_map.split(","):
        pair = pair.strip()
        if ":" not in pair:
            print(f"ERROR: Invalid exp-prompt-map entry: '{pair}' "
                  f"(expected 'exp_name:prompt_index')")
            sys.exit(1)
        ename, pidx = pair.split(":", 1)
        exp_prompt_map[ename.strip()] = int(pidx.strip())
    print(f"Exp-prompt map: {exp_prompt_map}")

    # ---- 2c. Load Sydney config (optional) ----
    sydney_details = []
    if args.config:
        if not os.path.isfile(args.config):
            print(f"ERROR: Config file not found: {args.config}")
            sys.exit(1)
        sydney_details = load_sydney_configs(args.config, exp_prompt_map)
        if sydney_details:
            print(f"Sydney configs loaded: {len(sydney_details)} arm(s)")
        else:
            print("WARNING: --config provided but no Sydney configs matched")

    # ---- 3. Pack each session (parallel) ----
    session_files = sorted(session_queries.keys())
    num_session_workers = min(args.workers, len(session_files))
    print(f"Packing {len(session_files)} sessions with "
          f"{num_session_workers} threads…")

    results = []
    with ThreadPoolExecutor(max_workers=num_session_workers) as executor:
        futures = {
            executor.submit(
                _process_one_session, sf, session_queries[sf],
                args.sessions_dir, exp_prompt_map, scraper_outputs,
                args.trim_level, sydney_details, args.output_dir,
                args.dry_run,
            ): sf
            for sf in session_files
        }
        for future in as_completed(futures):
            results.append(future.result())

    # Print results in sorted order and accumulate totals
    total_matched = 0
    total_queries = 0
    total_sessions = 0
    total_size_kb = 0
    total_missing_session = 0
    total_missing_scraper = 0

    for session_file, stats, log_lines in sorted(results, key=lambda r: r[0]):
        for line in log_lines:
            print(line)
        if stats is not None:
            total_sessions += 1
            total_matched += stats["matched"]
            total_queries += stats["total"]
            total_size_kb += stats["packed_size_kb"]
            total_missing_session += stats["missing_in_session"]
            total_missing_scraper += stats["missing_in_scraper"]

    # ---- Summary ----
    print(f"\n{'=== DRY RUN ===' if args.dry_run else '=== Summary ===' }")
    print(f"Exp-prompt map: {exp_prompt_map}")
    print(f"Queries matched: {total_matched}/{total_queries}")
    if total_missing_session or total_missing_scraper:
        print(f"Missing: {total_missing_session} not in session, "
              f"{total_missing_scraper} not in scraper")
    print(f"Total packed size: {total_size_kb:.1f} KB")
    print(f"Trim level: {args.trim_level}")
    if sydney_details:
        print(f"Sydney config injected: {len(sydney_details)} arm(s) "
              f"({', '.join(d['id'] for d in sydney_details)})")
    else:
        print("Sydney config: not injected (no --config)")
    if not args.dry_run:
        print(f"Output directory: {args.output_dir}")
        print(f"\nNext step: submit with run_reasoning_checklist.py "
              f"-source {args.output_dir} -run-type 4")


if __name__ == "__main__":
    main()
