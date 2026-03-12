"""
build_foundry_eval_sessions.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build Foundry evaluation session JSON files from Sydney scraper outputs,
ready for parse-only evaluation runs (runType 4 — ParseInference).

Three-step pipeline
~~~~~~~~~~~~~~~~~~~
1. **Parse queryset** (single-threaded, in-memory)

   Read queryset.tsv to build a per-session validation lookup mapping
   ``session_file → {utterance → input_data}``.  Queryset queries are
   a **subset** of each session's dataItems — queries not in the
   queryset simply have no scraper result.  Also collects the set of
   needed utterances for filtering in Step 2.

2. **Stage scraper outputs** (multi-threaded read, single-threaded write)

   Load scraper JSON files in a thread pool, trim each response body,
   then write per-session staging files to a temporary directory on
   disk.  Each scraper file's ``query.segment`` is split by comma to
   route the trimmed result to every target session.  Disk staging
   keeps memory usage constant regardless of dataset size or trim
   level.

   Staging layout::

       staging_dir/{exp_name}/{session_path}.tsv

   where session_path may include subdirectories (e.g.
   ``subdir/session-file.json``).

3. **Pack sessions** (sequential, one session at a time)

   For each base session JSON that has staging data:

   a. Iterate ``dataItems`` (source of truth for indices and inputs).
      A single utterance may appear multiple times (duplicated for
      variance reduction).  All indices per utterance are collected.
   b. Validate queryset entries against the session's dataItems.
   c. Read the session's staging file, match utterances to dataItem
      indices, and populate ``sydneyInference`` entries keyed by
      ``sydneyId`` (experiment arm name).  Duplicate utterances
      receive the same scraper response at every copy's index.
   d. Remove original GPT prompts (Sydney-only output).
   e. Inject ``sydneyDetails`` from config.json.
   f. Write the packed session JSON to the output directory.

Mapping chain
~~~~~~~~~~~~~
queryset.tsv
  inputs[].file  --- session file path (relative to sessions-dir)
  inputs[].input.utterance --- utterance text

scraper output
  query.segment  --- comma-separated session file paths
  query.id       --- utterance text
  exp_name       --- experiment arm

base session JSON
  sessionInputs[0].dataItems[i].input  →  {"utterance": "..."} → match
  sessionInputs[0].dataItems[i].index  →  key for dataItemsOutputs

Arguments
~~~~~~~~~
Required:
    --queryset      Path to queryset.tsv — maps queries to session files.
                    Queryset queries are a subset of each session's
                    dataItems.  Columns: query, segment, inputs.
    --sessions-dir  Root directory containing base session JSON files
                    (Foundry session templates with prompts, dataItems,
                    and evaluationStrategy).  Session files may be in
                    subdirectories; paths in the queryset and scraper
                    segment fields are relative to this directory.
    --scraper-dir   Directory containing Sydney scraper output JSON files.
                    Each file has query.segment, query.id (utterance),
                    exp_name, and requests[0].response_body.
    --output-dir    Directory where packed session JSON files will be
                    written (created automatically if it doesn't exist).
                    Subdirectory structure from sessions-dir is
                    preserved.

Optional:
    --config        Path to config.json from the scraper config dataset.
                    When provided, Sydney settings from each exp_config
                    are injected into packed sessions as ``sydneyDetails``.
                    Default: not set (no injection).
    --exp-prompt-map Comma-separated exp_name:index pairs.  Only the
                    exp_name part is used (indices are ignored — kept
                    for backward compatibility).
                    Default: "control:0,experiment:1".
    --trim-level    2 = keep relevant metrics only (~274 KB/item).
                    3 = also trim output JSON fields (~2.4 KB/item).
                    Default: 3.
    --dry-run       Show stats without writing output files.
    --workers       Threads for loading scraper files (default: 8).
    --staging-dir   Fixed directory for staging files.  When set and
                    the directory already has data, Step 2 is skipped
                    (reuse previous staging) — useful for debugging.
                    When not set, a unique temp directory is created
                    and cleaned up after use.
"""

import argparse
import copy
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
    """
    Parse queryset.tsv into a per-session validation lookup.

    The queryset records which queries (utterances) were scraped and for
    which session files.  It is a **subset** of the dataItems in each
    base session JSON — queries not in the queryset simply have no
    scraper result.

    Returns:
        queryset_check : dict
            ``session_file → {utterance → input_data}``
            Used in Step 3 to validate that queryset entries actually
            appear in the base session's dataItems.

        needed_utterances : set
            All unique utterance strings.  Passed to Step 2 so the
            scraper loader can skip files for queries outside the
            queryset.

        query_count : int
            Total number of (session_file, utterance) entries parsed.
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
    """Load, trim, and pre-serialize a single scraper JSON file.

    Returns a dict with session routing info and the trimmed response,
    the string ``"skipped"`` if the utterance is not in
    *needed_utterances*, or ``None`` on error.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None

    query = data.get("query", {})
    segment = query.get("segment", "")
    utterance = query.get("id", "").strip()

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

    # Split segment to get individual session file paths
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
    """
    Load scraper files in parallel, trim, and stage to per-session files.

    Thread pool reads and trims scraper files concurrently (I/O-bound).
    The main thread writes trimmed results to per-session staging files
    on disk, keeping memory usage constant regardless of dataset size.

    Staging layout::

        staging_dir/
            {exp_name}/
                {session_path}.tsv   # tab-separated: utterance\\tresponse_json

    where session_path may include subdirectories (e.g.
    ``subdir/session-file.json``).

    Args:
        scraper_dir:       Directory containing scraper output JSON files.
        staging_dir:       Directory where staging files will be written.
        needed_utterances: Optional set of utterance strings.  When
                           provided, scraper files whose utterance is not
                           in this set are skipped.
        trim_level:        When set (2 or 3), trim each response_body
                           immediately after loading.
        num_workers:       Number of threads in the pool (default 8).

    Returns:
        (sessions_with_data, stats)

        sessions_with_data: set of session filenames that received at
                            least one staging entry.
        stats: dict with ``loaded``, ``skipped``, ``errors`` counts.
    """
    pattern = os.path.join(scraper_dir, "*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"WARNING: No JSON files found in {scraper_dir}")
        return set(), {"loaded": 0, "skipped": 0, "errors": 0}

    staging_handles = {}      # (exp_name, session_file) → file handle
    sessions_with_data = set()
    skipped = 0
    loaded = 0
    errors = 0
    done = 0

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
                    utterance = result["utterance"]
                    resp_json = result["response_body_json"]

                    # Append to each target session's staging file
                    line = utterance + "\t" + resp_json + "\n"
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

    ``configuration`` is a Foundry ``SydneyPostBody``.  We populate all
    fields present in the scraper config (optionsSets, plugins, scenario,
    tone, options) so users can load the session into Playground and
    re-scrape with identical Sydney settings.  Fields not present in the
    scraper config are left null so Foundry uses its defaults.

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
    except (json.JSONDecodeError, OSError) as e:
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

        # sydney.options → configuration.options
        # (e.g. IsConfigOptionsListsMergeable, ModelClassificationOverride)
        options = syd.get("options") or None

        # Build the SydneyPostBody (configuration).
        # Faithfully copy all fields from the scraper config so users
        # can load the session into Playground and re-scrape with
        # identical Sydney settings.
        configuration = {
            "message": None,
            "optionsSets": options_sets or None,
            "plugins": syd.get("plugins") or None,
            "scenario": scenario,
            "tone": tone,
            "options": options,
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
# Session processing
# ---------------------------------------------------------------------------

def _process_one_session(session_file, queryset_utterances, sessions_dir,
                         exp_prompt_map, staging_dir, sydney_details,
                         output_dir, dry_run):
    """Load a base session, validate, pack scraper results, and write output.

    Iterates through dataItems sequentially.  For each item, looks up
    its utterance in the staged scraper results and assigns the next
    available result via a per-utterance round-robin counter.  This
    naturally handles duplicate utterances (for variance reduction)
    without a separate grouping step.  DataItems not in the queryset
    are expected to have no scraper result and are silently skipped.
    Original GPT prompts are removed — the output contains only
    Sydney arms.

    Args:
        session_file:        Base session filename.
        queryset_utterances: ``{utterance → input_data}`` from queryset
                             for this session, or ``None``.
        sessions_dir:        Directory containing base session JSON files.
        exp_prompt_map:      ``{exp_name → prompt_index}``.
        staging_dir:         Directory with staging TSV files from Step 2.
        sydney_details:      List of SydneyDetail dicts, or ``[]``.
        output_dir:          Where to write the packed session JSON.
        dry_run:             If ``True``, skip writing files.

    Returns:
        ``(session_file, stats_or_None, log_lines)``

        *stats* (when not ``None``) is a dict with keys:
            ``matched``           — dataItem×arm slots that received data
            ``total``             — total dataItem copies × arms
            ``no_scraper``        — in queryset but scraper file missing
            ``empty_response``    — scraper returned empty/trivial JSON
            ``unique_utterances`` — distinct utterance strings
            ``data_items``        — total dataItem copies (across dups)
            ``packed_size_kb``    — approximate packed output size in KB
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
        input_str = di.get("input", "")
        if not input_str:
            continue
        try:
            input_json = json.loads(input_str)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(input_json, dict):
            utt = input_json.get("utterance", "").strip()
            if utt:
                item_utts.append((idx, utt))
                all_utts.add(utt)

    # Validate: queryset utterances should appear in session dataItems
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

    # Build sydneyInference results per experiment arm
    sydney_inference_list = []
    total_matched = 0
    total_no_scraper = 0
    total_empty_response = 0
    total_size_kb = 0.0

    for exp_name in sorted(exp_prompt_map.keys()):
        # Read staging file for this arm + session
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
        matched = 0
        no_scraper = 0
        empty_response = 0
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
                # In queryset but scraper result missing
                no_scraper += 1
            # else: not in queryset — not scraped, expected

        sydney_inference_list.append({
            "sydneyId": exp_name,
            "dataItemsOutputs": data_items_outputs,
            "conversationIds": {},
            "fileLocations": {},
        })
        total_matched += matched
        total_no_scraper += no_scraper
        total_empty_response += empty_response
        total_size_kb += sum(
            len(v) for v in data_items_outputs.values()) / 1024

    if not sydney_inference_list:
        log.append(f"  SKIP {session_file}: no valid inference arms")
        return session_file, None, log

    # total_data_items across arms = total dataItem copies × arms
    num_arms = len(sydney_inference_list) if sydney_inference_list else 1

    if total_matched == 0:
        log.append(
            f"  SKIP {session_file}: 0/{len(item_utts) * num_arms} matched"
            f" (no_scraper={total_no_scraper},"
            f" empty_response={total_empty_response})")
        return session_file, None, log

    # Build packed session — Sydney-only (remove GPT prompts)
    packed = dict(session)      # shallow copy
    packed["prompts"] = []

    packed["resultsv2"] = [{
        "sessionInputIndex": 0,
        "inference": [],
        "evaluation": [],
        "sydneyInference": sydney_inference_list,
        "sydneyEvaluation": None,
    }]

    if sydney_details:
        packed["sydneyDetails"] = copy.deepcopy(sydney_details)

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
        description="Pack trimmed Sydney scraper results into Foundry "
                    "session JSONs for runType 4 (ParseInference).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--queryset", required=True,
        help="Path to queryset.tsv (query->session mapping for validation)",
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
        help="Comma-separated exp_name:index pairs.  Only the exp_name "
             "part is used to identify experiment arms (indices are "
             "ignored — kept for backward compat).  "
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
             "(default: 8)",
    )
    parser.add_argument(
        "--staging-dir", default=None,
        help="Fixed directory for scraper staging files.  When set "
             "and the directory already contains data, Step 2 is "
             "skipped (reuse previous staging).  Useful for "
             "debugging.  When not set, a unique temp directory is "
             "created and cleaned up automatically.",
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

    # Parse exp-prompt-map early (needed by config loading)
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

    # ---- Step 1. Parse queryset ----
    print("\n--- Step 1: Parse queryset ---")
    queryset_check, needed_utterances, query_count = parse_queryset(
        args.queryset)
    print(f"Queryset: {query_count} entries across "
          f"{len(queryset_check)} session files, "
          f"{len(needed_utterances)} unique utterances")

    # Load Sydney config (optional)
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

    # ---- Step 2. Stage scraper outputs to disk ----
    print("\n--- Step 2: Stage scraper outputs ---")

    # Determine staging directory
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
            # Discover sessions from existing staging files
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
                "loaded": "reused", "skipped": "\u2013", "errors": "\u2013",
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
        print(f"\n--- Step 3: Pack {len(sessions_with_data)} sessions ---")

        results = []
        for sf in sorted(sessions_with_data):
            qs_utts = queryset_check.get(sf)
            result = _process_one_session(
                sf, qs_utts, args.sessions_dir, exp_prompt_map,
                staging_dir, sydney_details, args.output_dir, args.dry_run,
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
        if sydney_details:
            print(f"Sydney details injected: {len(sydney_details)} arm(s) "
                  f"({', '.join(d['id'] for d in sydney_details)})")
        else:
            print("Sydney details: not injected (no --config)")
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
