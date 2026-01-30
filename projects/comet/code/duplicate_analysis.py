# duplicate_analysis.py
#
# Copyright (c) Microsoft Corporation. All rights reserved.

"""
Duplicate detection analysis tool for LLM NDCG results.

This script uses the new deduplication module to ensure the duplicate
detection logic is identical to production.

Commands:
    analyze  - Analyze duplicate detection for a specific utterance
    batch    - Analyze multiple utterances quickly
    top      - Show utterances with highest duplicate counts

Usage:
    python duplicate_analysis.py analyze --input tests/144683_top200.jsonl --utterance "Show my"
    python duplicate_analysis.py analyze --input tests/144683_top200.jsonl --index 2 --arm control
    python duplicate_analysis.py analyze --input tests/144683_top200.jsonl --utterance "Show my" --verbose
    python duplicate_analysis.py top --input tests/144683_top200.jsonl --topk 20
"""

import json
import sys
from pathlib import Path

import fire

# Add the cometdefinition to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cometdefinition.common import utils
from cometdefinition.common.entity_type_utils import resolve_full_entity_type
from cometdefinition.metrics.llm_ndcg.logic.deduplication import (
    DeduplicationManager,
    get_all_dedup_keys,
)
from cometdefinition.metrics.llm_ndcg.logic.verification_data_extractor import (
    extract_verification_data,
    get_domain_specific_id,
)


def group_results_with_debug(all_search_results: str | dict) -> dict:
    """
    Group results by iteration with detailed debug output.

    Uses the new deduplication module to ensure
    duplicate detection matches the production metric.
    """
    manager = DeduplicationManager()
    results_by_iteration: dict[int, dict] = {}

    # Track all results for building duplicate groups - use GLOBAL index
    all_results: dict[tuple[int, int], dict] = {}  # (iter, global_idx) -> result_info

    debug_info = {
        "total_results": 0,
        "duplicates": [],
        "unique_results": [],
        "by_iteration": {},
        "duplicate_groups": {},  # (orig_iter, orig_global_idx) -> [list of duplicate result_infos]
    }

    # Track global index across all batches within an iteration
    global_idx_by_iteration: dict[int, int] = {}

    for search_result, iteration, plugin, batch_idx in utils.yield_enterprise_search_results(all_search_results):
        if "Results" not in search_result:
            continue

        if iteration not in results_by_iteration:
            results_by_iteration[iteration] = {"results": [], "duplicates": []}
            debug_info["by_iteration"][iteration] = {"results": [], "duplicates": []}
            global_idx_by_iteration[iteration] = 0

        # Extract tool call info from search_result
        plugin_invocation = search_result.get("PluginInvocation", "")
        result_type = search_result.get("ResultType", "")

        for idx, entity in enumerate(search_result["Results"]):
            # Use global index for this iteration
            global_idx = global_idx_by_iteration[iteration]
            global_idx_by_iteration[iteration] += 1
            debug_info["total_results"] += 1

            # Use new deduplication module functions
            verification_data = extract_verification_data(entity)
            dedup_keys = get_all_dedup_keys(entity, verification_data)
            entity_type = resolve_full_entity_type(entity)
            domain_id = get_domain_specific_id(entity)

            # Get all entity_id_keys and url_key from dedup_keys for display
            entity_id_keys = []
            url_key = None
            for key in dedup_keys:
                if key.startswith("id:"):
                    entity_id_keys.append(key)
                elif key.startswith("url:"):
                    url_key = key

            # Extract content snippet for debugging
            content_body = entity.get("ContentBody") or verification_data.get("content") or ""
            content_snippet = content_body[:80].replace("\n", " ").strip() if content_body else ""

            # Build result info
            doc_type = entity.get("DocType", "")  # Available in test data format
            result_info = {
                "iteration": iteration,
                "index": global_idx,
                "batch_idx": batch_idx,
                "local_idx": idx,
                "plugin": plugin,
                "plugin_invocation": plugin_invocation,
                "result_type": result_type,
                "entity_type": entity_type,
                "doc_type": doc_type,
                "label": entity.get("Label", 0) or 0,
                "title": (verification_data.get("title") or "")[:60],
                "url": (verification_data.get("url") or "")[:80],
                "content_snippet": content_snippet,
                "entity_id_keys": entity_id_keys,
                "url_key": url_key,
                "domain_specific_id": domain_id,
            }

            # Store for later lookup
            all_results[(iteration, global_idx)] = result_info

            # Use DeduplicationManager to check for duplicates
            is_duplicate, matched_with = manager.check_duplicate(verification_data, iteration)

            match_reason = None
            if is_duplicate and matched_with:
                # Determine match reason by checking verification steps
                orig_key = (matched_with["iteration"], matched_with["index"])
                orig_info = all_results.get(orig_key, {})
                match_reason = _determine_match_reason_from_info(result_info, orig_info)

            # Classify
            if is_duplicate:
                result_info["matched_with"] = matched_with
                result_info["match_reason"] = match_reason
                debug_info["duplicates"].append(result_info)
                debug_info["by_iteration"][iteration]["duplicates"].append(result_info)

                # Add to duplicate groups (grouped by original)
                orig_key = (matched_with["iteration"], matched_with["index"])
                if orig_key not in debug_info["duplicate_groups"]:
                    debug_info["duplicate_groups"][orig_key] = []
                debug_info["duplicate_groups"][orig_key].append(result_info)
            else:
                debug_info["unique_results"].append(result_info)
                debug_info["by_iteration"][iteration]["results"].append(result_info)

            target = "duplicates" if is_duplicate else "results"
            results_by_iteration[iteration][target].append({
                "label": entity.get("Label", 0) or 0,
                "plugin": plugin,
            })

    # Store all_results for looking up originals
    debug_info["all_results"] = all_results

    return debug_info


def _determine_match_reason_from_info(
    dup_info: dict,
    orig_info: dict,
) -> str:
    """Determine match reason based on result info."""
    # Check if any entity_id_key matches
    dup_id_keys = set(dup_info.get("entity_id_keys") or [])
    orig_id_keys = set(orig_info.get("entity_id_keys") or [])
    if dup_id_keys and orig_id_keys and dup_id_keys & orig_id_keys:
        return "entity_id_match"

    # Check if url_key matches
    if dup_info.get("url_key") and dup_info.get("url_key") == orig_info.get("url_key"):
        return "url_match"

    # Check if domain_specific_id matches
    if dup_info.get("domain_specific_id") and dup_info.get("domain_specific_id") == orig_info.get("domain_specific_id"):
        return "domain_specific_id_match"

    # Check entity type specific
    entity_type = dup_info.get("entity_type") or dup_info.get("doc_type")
    if entity_type == "Email":
        return "email_type_specific"
    elif entity_type == "Event":
        return "event_type_specific"
    elif entity_type in ("People", "PeopleInferenceAnswer"):
        return "people_type_specific"

    # Default to content/snippet match
    return "content_match"


def find_record_by_utterance(input_path: Path, utterance_prefix: str) -> tuple[int, dict] | None:
    """
    Find a record by matching utterance prefix.

    Args:
        input_path: Path to input JSONL file
        utterance_prefix: Prefix to match against utterances (case-insensitive)

    Returns:
        Tuple of (index, record) if found, None otherwise
    """
    prefix_lower = utterance_prefix.lower().strip()
    matches = []

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            # Support both formats: root-level or nested under 'input'
            utterance = record.get("utterance") or record.get("input", {}).get("utterance", "")
            if utterance.lower().startswith(prefix_lower):
                matches.append((i, record, utterance))

    if not matches:
        print(f"Error: No utterance found starting with: '{utterance_prefix[:50]}...'")
        return None

    if len(matches) > 1:
        print(f"Warning: Multiple matches found ({len(matches)}). Using first match.")
        print("Matches:")
        for i, (idx, _, utt) in enumerate(matches[:5]):
            print(f"  [{idx}] {utt[:80]}...")
        if len(matches) > 5:
            print(f"  ... and {len(matches) - 5} more")
        print()

    return matches[0][0], matches[0][1]


def analyze(
    input: str,
    index: int | None = None,
    utterance: str | None = None,
    arm: str = "both",
    verbose: bool = False,
    show_all: bool = False,
    output: str | None = None,
):
    """
    Analyze duplicate detection for a specific utterance.

    Args:
        input: Path to input JSONL file
        index: 0-based index of the record to analyze (use this OR utterance)
        utterance: Prefix of utterance to match (case-insensitive, use this OR index)
        arm: Which arm to analyze: 'control', 'treatment', or 'both' (default)
        verbose: Show verbose output including all unique results
        show_all: Show all duplicates (not just first 20)
        output: Output file path (markdown format). If not specified, prints to console.
    """
    input_path = Path(input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input}")
        return

    # Validate arguments
    if index is None and utterance is None:
        print("Error: Must specify either --index or --utterance")
        return

    if index is not None and utterance is not None:
        print("Warning: Both --index and --utterance specified. Using --utterance.")

    # Find the record
    record = None
    record_index = None

    if utterance is not None:
        result = find_record_by_utterance(input_path, utterance)
        if result is None:
            return
        record_index, record = result
    else:
        # Load by index
        with open(input_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == index:
                    record = json.loads(line)
                    record_index = i
                    break
            else:
                print(f"Error: Index {index} out of range")
                return

    # Support both formats: root-level or nested under 'input'
    utterance_text = record.get("utterance") or record.get("input", {}).get("utterance", "N/A")

    # Support both field naming conventions
    arms_to_analyze = []
    if arm in ["control", "both"]:
        ctrl_results = (
            record.get("all_search_results_control")
            or record.get("input", {}).get("control_all_search_results")
        )
        arms_to_analyze.append(("control", ctrl_results))
    if arm in ["treatment", "both"]:
        treat_results = (
            record.get("all_search_results_treatment")
            or record.get("input", {}).get("treatment_all_search_results")
        )
        arms_to_analyze.append(("treatment", treat_results))

    # Build output lines
    lines = []

    def out(text: str = ""):
        lines.append(text)

    out(f"# Duplicate Analysis Report")
    out()
    out(f"**Record #**: {record_index}")
    out(f"**Utterance**: {utterance_text[:200]}...")
    out(f"**Input File**: `{input_path.name}`")
    out()

    for arm_name, all_search_results in arms_to_analyze:
        if not all_search_results:
            out(f"## {arm_name.upper()} ARM")
            out()
            out("No search results found.")
            out()
            continue

        out(f"## {arm_name.upper()} ARM ANALYSIS")
        out()

        debug_info = group_results_with_debug(all_search_results)

        total = debug_info["total_results"]
        num_dups = len(debug_info["duplicates"])
        num_unique = len(debug_info["unique_results"])
        dup_rate = (num_dups / total * 100) if total > 0 else 0

        out(f"### Summary")
        out()
        out(f"| Metric | Value |")
        out(f"|--------|-------|")
        out(f"| Total results | {total} |")
        out(f"| Unique results | {num_unique} |")
        out(f"| Duplicates | {num_dups} ({dup_rate:.1f}%) |")
        out()

        # Show by iteration
        out(f"### By Iteration")
        out()
        out(f"| Iteration | Unique | Duplicates |")
        out(f"|-----------|--------|------------|")
        for iteration in sorted(debug_info["by_iteration"].keys()):
            iter_data = debug_info["by_iteration"][iteration]
            n_res = len(iter_data["results"])
            n_dup = len(iter_data["duplicates"])
            out(f"| {iteration} | {n_res} | {n_dup} |")
        out()

        # Show duplicates GROUPED BY ORIGINAL
        duplicate_groups = debug_info.get("duplicate_groups", {})
        all_results = debug_info.get("all_results", {})

        if duplicate_groups:
            # Sort groups by number of duplicates (descending)
            sorted_groups = sorted(duplicate_groups.items(), key=lambda x: len(x[1]), reverse=True)

            # Filter to only groups with duplicates
            groups_with_dups = [(k, v) for k, v in sorted_groups if len(v) > 0]
            total_groups = len(groups_with_dups)

            out(f"### Duplicate Groups ({total_groups} unique results have duplicates)")
            out()

            max_groups = total_groups if show_all else min(10, total_groups)
            for group_num, (orig_key, dups) in enumerate(groups_with_dups[:max_groups], 1):
                orig_iter, orig_idx = orig_key
                orig = all_results.get(orig_key, {})

                out(f"#### Group {group_num}: Original at Iter {orig_iter}, Idx {orig_idx} ({len(dups)} duplicates)")
                out()
                out(f"**Original Result:**")
                out()
                out(f"| Field | Value |")
                out(f"|-------|-------|")
                out(f"| Location | Iter {orig_iter}, Idx {orig_idx} |")
                out(f"| **Tool Call** | `{orig.get('plugin_invocation') or '(none)'}` |")
                # Show entity_type (resolved from Type field), fallback to doc_type for test data
                orig_type = orig.get('entity_type')
                if not orig_type or orig_type == 'Unknown':
                    orig_type = orig.get('doc_type') or 'Unknown'
                out(f"| **Entity Type** | {orig_type} |")
                out(f"| Label | {orig.get('label', 0)} |")
                out(f"| Title | {orig.get('title') or '(no title)'} |")
                content = orig.get('content_snippet', '')
                out(f"| Content | {content}... |" if content else "| Content | (empty) |")
                out(f"| URL | {orig.get('url') or '(no URL)'} |")
                # Show all Entity ID Keys
                orig_id_keys = orig.get('entity_id_keys') or []
                if orig_id_keys:
                    out(f"| Entity ID Keys | `{'`, `'.join(orig_id_keys)}` |")
                else:
                    out(f"| Entity ID Keys | (none) |")
                out(f"| URL Key | `{orig.get('url_key') or '(none)'}` |")
                out(f"| Domain-specific ID | {orig.get('domain_specific_id') or '(none)'} |")
                out()

                out(f"**Duplicates:**")
                out()
                for i, dup in enumerate(dups, 1):
                    out(f"**Duplicate {i}:** (Match Reason: `{dup.get('match_reason', 'unknown')}`)")
                    out()
                    out(f"| Field | Original (Idx {orig_idx}) | Duplicate (Idx {dup['index']}) |")
                    out(f"|-------|---------------------------|-------------------------------|")
                    out(f"| Location | Iter {orig_iter}, Idx {orig_idx} | Iter {dup['iteration']}, Idx {dup['index']} |")
                    out(f"| **Tool Call** | `{orig.get('plugin_invocation') or '(none)'}` | `{dup.get('plugin_invocation') or '(none)'}` |")
                    # Show entity_type (resolved from Type field), fallback to doc_type for test data
                    orig_type = orig.get('entity_type')
                    if not orig_type or orig_type == 'Unknown':
                        orig_type = orig.get('doc_type') or 'Unknown'
                    dup_type = dup.get('entity_type')
                    if not dup_type or dup_type == 'Unknown':
                        dup_type = dup.get('doc_type') or 'Unknown'
                    out(f"| **Entity Type** | {orig_type} | {dup_type} |")
                    out(f"| Label | {orig.get('label', 0)} | {dup.get('label', 0)} |")
                    # Show all Entity ID Keys
                    orig_id_keys = orig.get('entity_id_keys') or []
                    dup_id_keys = dup.get('entity_id_keys') or []
                    orig_id_keys_str = ', '.join(f'`{k}`' for k in orig_id_keys) if orig_id_keys else '(none)'
                    dup_id_keys_str = ', '.join(f'`{k}`' for k in dup_id_keys) if dup_id_keys else '(none)'
                    out(f"| **Entity ID Keys** | {orig_id_keys_str} | {dup_id_keys_str} |")
                    out(f"| **URL Key** | `{orig.get('url_key') or '(none)'}` | `{dup.get('url_key') or '(none)'}` |")
                    out(f"| URL | {orig.get('url') or '(no URL)'} | {dup.get('url') or '(no URL)'} |")
                    out(f"| Domain-specific ID | {orig.get('domain_specific_id') or '(none)'} | {dup.get('domain_specific_id') or '(none)'} |")
                    out(f"| Title | {orig.get('title') or '(no title)'} | {dup.get('title') or '(no title)'} |")
                    orig_content = orig.get('content_snippet', '') or '(empty)'
                    dup_content = dup.get('content_snippet', '') or '(empty)'
                    out(f"| Content | {orig_content}... | {dup_content}... |")
                    out()

                out(f"---")
                out()

            if total_groups > max_groups:
                out(f"*... and {total_groups - max_groups} more groups (use --show-all to see all)*")
                out()

        # Show match reason breakdown
        duplicates = debug_info["duplicates"]
        if duplicates:
            out(f"### Match Reason Breakdown")
            out()
            out(f"| Reason | Count | Percentage |")
            out(f"|--------|-------|------------|")
            reasons = {}
            for dup in duplicates:
                reason = dup.get("match_reason", "unknown")
                reasons[reason] = reasons.get(reason, 0) + 1
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                out(f"| {reason} | {count} | {count/len(duplicates)*100:.1f}% |")
            out()

        # Show unique results if verbose
        if verbose:
            out(f"### Unique Results (first 10)")
            out()
            out(f"| # | Location | Type | Label | Title |")
            out(f"|---|----------|------|-------|-------|")
            for i, res in enumerate(debug_info["unique_results"][:10], 1):
                loc = f"Iter {res['iteration']}, Idx {res['index']}"
                title = (res.get('title') or '(no title)')[:40]
                out(f"| {i} | {loc} | {res['entity_type']} | {res['label']} | {title} |")
            out()

    # Output results
    output_text = "\n".join(lines)

    if output:
        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Report written to: {output_path}")
    else:
        print(output_text)


def batch_analyze(
    input: str,
    indices: str = "2,135,99,33,103",
    arm: str = "control",
):
    """
    Analyze multiple utterances quickly.

    Args:
        input: Path to input JSONL file
        indices: Comma-separated list of indices to analyze
        arm: Which arm to analyze: 'control' or 'treatment'
    """
    idx_list = [int(x.strip()) for x in indices.split(",")]
    
    for idx in idx_list:
        analyze(input, idx, arm=arm, verbose=False, show_all=False)
        print("\n" + "=" * 100 + "\n")


def count_duplicates_for_utterance(
    all_search_results: str | dict,
) -> tuple[int, int, int]:
    """
    Count total results, unique results, and duplicates for an utterance.

    Uses the new deduplication module.

    Args:
        all_search_results: Search results JSON

    Returns:
        Tuple of (total_count, unique_count, duplicate_count)
    """
    manager = DeduplicationManager()
    total_count = 0
    dup_count = 0

    for search_result, iteration, plugin, _ in utils.yield_enterprise_search_results(all_search_results):
        if "Results" not in search_result:
            continue

        for entity in search_result["Results"]:
            total_count += 1

            # Use new deduplication module
            verification_data = extract_verification_data(entity)
            is_duplicate, _ = manager.check_duplicate(verification_data, iteration)

            if is_duplicate:
                dup_count += 1

    return total_count, total_count - dup_count, dup_count


def top(
    input: str,
    topk: int = 20,
    arm: str = "both",
    min_results: int = 10,
    output: str | None = None,
):
    """
    Show utterances with highest duplicate counts using new dedup module.

    Args:
        input: Path to input JSONL file
        topk: Number of top utterances to show (default: 20)
        arm: Which arm to analyze: 'control', 'treatment', or 'both' (default)
        min_results: Minimum results count to include (default: 10)
        output: Output file path for results (optional, CSV format)
    """
    input_path = Path(input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input}")
        return

    # Collect stats per utterance
    control_stats: list[tuple[int, str, int, int, int, float]] = []  # (idx, utt, total, unique, dups, rate)
    treatment_stats: list[tuple[int, str, int, int, int, float]] = []

    print(f"Analyzing {input_path.name}...")
    total_records = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            total_records += 1

            # Support both formats: root-level or nested under 'input'
            utterance_text = record.get("utterance") or record.get("input", {}).get("utterance", "N/A")
            utt_display = (utterance_text[:50] + "...") if len(utterance_text) > 50 else utterance_text

            # Analyze control arm
            if arm in ["control", "both"]:
                ctrl_results = (
                    record.get("all_search_results_control")
                    or record.get("input", {}).get("control_all_search_results")
                )
                if ctrl_results:
                    total, unique, dups = count_duplicates_for_utterance(ctrl_results)
                    if total >= min_results:
                        rate = (dups / total * 100) if total > 0 else 0
                        control_stats.append((i, utt_display, total, unique, dups, rate))

            # Analyze treatment arm
            if arm in ["treatment", "both"]:
                treat_results = (
                    record.get("all_search_results_treatment")
                    or record.get("input", {}).get("treatment_all_search_results")
                )
                if treat_results:
                    total, unique, dups = count_duplicates_for_utterance(treat_results)
                    if total >= min_results:
                        rate = (dups / total * 100) if total > 0 else 0
                        treatment_stats.append((i, utt_display, total, unique, dups, rate))

            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1} records...")

    print(f"\nTotal records analyzed: {total_records}")
    print(f"Min results threshold: {min_results}")

    # Sort by duplicate count descending
    control_stats.sort(key=lambda x: x[4], reverse=True)
    treatment_stats.sort(key=lambda x: x[4], reverse=True)

    # Print control results
    if arm in ["control", "both"] and control_stats:
        print(f"\n{'=' * 100}")
        print(f"TOP {topk} CONTROL UTTERANCES BY DUPLICATE COUNT")
        print(f"{'=' * 100}")
        print(f"{'Rank':<6}{'Idx':<8}{'Total':<8}{'Unique':<8}{'Dups':<8}{'Rate':<10}{'Utterance'}")
        print("-" * 100)

        for rank, (idx, utt, total, unique, dups, rate) in enumerate(control_stats[:topk], 1):
            print(f"{rank:<6}{idx:<8}{total:<8}{unique:<8}{dups:<8}{rate:>5.1f}%    {utt}")

        # Summary stats
        all_dups = [s[4] for s in control_stats]
        all_rates = [s[5] for s in control_stats]
        all_totals = [s[2] for s in control_stats]
        print(f"\nCONTROL SUMMARY ({len(control_stats)} utterances with >= {min_results} results):")
        print(f"  Total duplicates: {sum(all_dups)}")
        print(f"  Avg duplicates/utterance: {sum(all_dups)/len(all_dups):.1f}")
        print(f"  Avg duplicate rate: {sum(all_rates)/len(all_rates):.1f}%")
        print(f"  Max duplicates: {max(all_dups)} (rate: {all_rates[all_dups.index(max(all_dups))]:.1f}%)")
        print(f"  Avg total results: {sum(all_totals)/len(all_totals):.1f}")

    # Print treatment results
    if arm in ["treatment", "both"] and treatment_stats:
        print(f"\n{'=' * 100}")
        print(f"TOP {topk} TREATMENT UTTERANCES BY DUPLICATE COUNT")
        print(f"{'=' * 100}")
        print(f"{'Rank':<6}{'Idx':<8}{'Total':<8}{'Unique':<8}{'Dups':<8}{'Rate':<10}{'Utterance'}")
        print("-" * 100)

        for rank, (idx, utt, total, unique, dups, rate) in enumerate(treatment_stats[:topk], 1):
            print(f"{rank:<6}{idx:<8}{total:<8}{unique:<8}{dups:<8}{rate:>5.1f}%    {utt}")

        # Summary stats
        all_dups = [s[4] for s in treatment_stats]
        all_rates = [s[5] for s in treatment_stats]
        all_totals = [s[2] for s in treatment_stats]
        print(f"\nTREATMENT SUMMARY ({len(treatment_stats)} utterances with >= {min_results} results):")
        print(f"  Total duplicates: {sum(all_dups)}")
        print(f"  Avg duplicates/utterance: {sum(all_dups)/len(all_dups):.1f}")
        print(f"  Avg duplicate rate: {sum(all_rates)/len(all_rates):.1f}%")
        print(f"  Max duplicates: {max(all_dups)} (rate: {all_rates[all_dups.index(max(all_dups))]:.1f}%)")
        print(f"  Avg total results: {sum(all_totals)/len(all_totals):.1f}")

    # Write CSV output if requested
    if output:
        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("arm,rank,index,utterance,total,unique,duplicates,rate\n")
            for rank, (idx, utt, total, unique, dups, rate) in enumerate(control_stats[:topk], 1):
                utt_escaped = utt.replace('"', '""')
                f.write(f'control,{rank},{idx},"{utt_escaped}",{total},{unique},{dups},{rate:.2f}\n')
            for rank, (idx, utt, total, unique, dups, rate) in enumerate(treatment_stats[:topk], 1):
                utt_escaped = utt.replace('"', '""')
                f.write(f'treatment,{rank},{idx},"{utt_escaped}",{total},{unique},{dups},{rate:.2f}\n')
        print(f"\nResults written to: {output_path}")


def merge_with_new_extraction(
    old_file: str,
    new_file: str,
    output: str | None = None,
    utterance_field: str = "utterance",
):
    """
    Merge utterances from old file with newly extracted data from new file.

    This function:
    1. Reads all utterances from the old file (preserving order)
    2. Reads the new file and builds an index by utterance
    3. For each utterance in the old file, finds the matching record in the new file
    4. Outputs a new JSONL file with the same utterances in the same order,
       but with the new extraction data

    Args:
        old_file: Path to old JSONL file (e.g., 135056_top200.jsonl)
        new_file: Path to new JSONL file with updated extractions
        output: Output file path. If not specified, uses old_file with '_new_extraction' suffix
        utterance_field: Field name for utterance (default: 'utterance')
    """
    old_path = Path(old_file)
    new_path = Path(new_file)

    if not old_path.exists():
        print(f"Error: Old file not found: {old_file}")
        return

    if not new_path.exists():
        print(f"Error: New file not found: {new_file}")
        return

    # Default output path
    if output is None:
        output = str(old_path.parent / f"{old_path.stem}_new_extraction.jsonl")
    output_path = Path(output)

    print(f"Reading old file: {old_path.name}")
    print(f"Reading new file: {new_path.name}")

    # Step 1: Read old file and extract utterances (preserving order)
    old_records = []
    old_utterances = []

    with open(old_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            # Support both formats: root-level or nested under 'input'
            utt = record.get(utterance_field) or record.get("input", {}).get(utterance_field, "")
            old_records.append(record)
            old_utterances.append(utt)

    print(f"  Found {len(old_utterances)} utterances in old file")

    # Step 2: Read new file and build index by utterance
    new_records_by_utterance: dict[str, dict] = {}
    new_total = 0

    with open(new_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            new_total += 1
            # Support both formats: root-level or nested under 'input'
            utt = record.get(utterance_field) or record.get("input", {}).get(utterance_field, "")
            if utt:
                # Store by exact utterance text
                new_records_by_utterance[utt] = record

    print(f"  Found {new_total} records in new file")
    print(f"  Indexed {len(new_records_by_utterance)} unique utterances")

    # Step 3: Match and merge
    matched_records = []
    not_found = []

    for i, (old_record, utt) in enumerate(zip(old_records, old_utterances)):
        if utt in new_records_by_utterance:
            matched_records.append(new_records_by_utterance[utt])
        else:
            # Try fuzzy match by stripping whitespace
            utt_stripped = utt.strip()
            found = False
            for new_utt, new_record in new_records_by_utterance.items():
                if new_utt.strip() == utt_stripped:
                    matched_records.append(new_record)
                    found = True
                    break
            if not found:
                not_found.append((i, utt[:80] if len(utt) > 80 else utt))
                # Keep old record as fallback
                matched_records.append(old_record)

    print(f"\nMatching Results:")
    print(f"  Matched: {len(matched_records) - len(not_found)}")
    print(f"  Not found (kept old): {len(not_found)}")

    if not_found:
        print(f"\n  Utterances not found in new file (first 10):")
        for idx, utt in not_found[:10]:
            print(f"    [{idx}] {utt}...")
        if len(not_found) > 10:
            print(f"    ... and {len(not_found) - 10} more")

    # Step 4: Write output file
    with open(output_path, "w", encoding="utf-8") as f:
        for record in matched_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nOutput written to: {output_path}")
    print(f"  Total records: {len(matched_records)}")


def extract_top200_with_new_data(
    old_top200: str = "tests/135056_top200.jsonl",
    new_full: str = "tests/135056_llm_ndcg_input_new.jsonl",
    output: str | None = None,
):
    """
    Convenience function to extract the top 200 utterances with new extraction data.

    This is a wrapper around merge_with_new_extraction specifically for the
    135056 dataset migration.

    Paths are relative to the current working directory (like other commands).

    Args:
        old_top200: Path to old top 200 file (default: tests/135056_top200.jsonl)
        new_full: Path to new full extraction file (default: tests/135056_llm_ndcg_input_new.jsonl)
        output: Output file path. If not specified, uses '<old_file>_new.jsonl'
    """
    # Use paths relative to current working directory (consistent with other commands)
    old_path = Path(old_top200)
    new_path = Path(new_full)

    if output is None:
        output = str(old_path.parent / f"{old_path.stem}_new.jsonl")

    merge_with_new_extraction(
        old_file=str(old_path),
        new_file=str(new_path),
        output=output,
    )


if __name__ == "__main__":
    fire.Fire({
        "analyze": analyze,
        "batch": batch_analyze,
        "top": top,
        "merge": merge_with_new_extraction,
        "extract_top200": extract_top200_with_new_data,
    })
