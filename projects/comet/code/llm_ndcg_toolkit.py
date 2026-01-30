# llm_ndcg_toolkit.py
#
# Copyright (c) Microsoft Corporation. All rights reserved.

"""
LLM NDCG Analysis Toolkit.

Description: This toolkit provides various analysis functions for LLM NDCG input/output files.
             Supports multiple commands via python-fire for command-line usage.

Usage:
    # Select top quality utterances based on iterations and results
    python llm_ndcg_toolkit.py select_top --input data.jsonl --output top200.jsonl --count 200

    # Analyze iteration distributions between control and treatment
    python llm_ndcg_toolkit.py iteration_distribution --input data.jsonl

    # Compare good gain metrics between control and treatment
    python llm_ndcg_toolkit.py compare_good_gain --input output.jsonl --output report.md

    # Show help for all commands
    python llm_ndcg_toolkit.py --help
"""

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import fire


# =============================================================================
# SEVAL Metrics Integration
# =============================================================================

# Default SEVAL metrics to include when seval_path is provided
DEFAULT_SEVAL_METRICS = ["citedcg_one_centric", "citedcg_num_enterprise_cites"]


def _load_seval_metrics(seval_path: str, metric_names: list[str]) -> tuple[dict[str, dict], set[str]]:
    """
    Load SEVAL metrics from the all_metrics_paired.csv file.

    Args:
        seval_path: Path to the SEVAL job folder (e.g., "144683_metrics")
        metric_names: List of metric base names to load (e.g., ["citedcg_one_centric"])

    Returns:
        Tuple of:
        - Dict mapping utterance -> {metric_name: {"control": value, "treatment": value}}
        - Set of duplicate utterances (to be excluded from comparison)
    """
    csv_path = Path(seval_path) / "offline_scorecard_generator_output" / "all_metrics_paired.csv"

    if not csv_path.exists():
        print(f"Warning: SEVAL metrics file not found: {csv_path}")
        return {}, set()

    # Track utterance occurrences for duplicate detection
    utterance_counts: dict[str, int] = {}
    seval_data: dict[str, dict] = {}
    available_metrics: list[str] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Find utterance column
        utterance_col = None
        for col in ["query", "Query", "utterance", "Utterance", "question", "Question"]:
            if col in headers:
                utterance_col = col
                break

        if not utterance_col:
            print("Warning: No utterance column found in SEVAL CSV")
            return {}, set()

        # Verify metric columns exist
        for metric in metric_names:
            ctrl_col = f"{metric}_control"
            treat_col = f"{metric}_treatment"
            if ctrl_col in headers and treat_col in headers:
                available_metrics.append(metric)
            else:
                print(f"Warning: Metric '{metric}' not found in SEVAL CSV (missing {ctrl_col} or {treat_col})")

        if not available_metrics:
            print("Warning: No requested SEVAL metrics found in CSV")
            return {}, set()

        # First pass: count utterances to detect duplicates
        rows_data = []
        for row in reader:
            utterance = row.get(utterance_col, "")  # No stripping - use full utterance
            if not utterance:
                continue
            utterance_counts[utterance] = utterance_counts.get(utterance, 0) + 1
            rows_data.append((utterance, row))

    # Identify duplicates
    duplicate_utterances = {u for u, count in utterance_counts.items() if count > 1}

    # Second pass: load data for unique utterances only
    for utterance, row in rows_data:
        if utterance in duplicate_utterances:
            continue  # Skip duplicates

        seval_data[utterance] = {}
        for metric in available_metrics:
            ctrl_val = row.get(f"{metric}_control", "")
            treat_val = row.get(f"{metric}_treatment", "")

            # Convert to float, handling empty/non-numeric values
            try:
                ctrl_val = float(ctrl_val) if ctrl_val else None
            except ValueError:
                ctrl_val = None
            try:
                treat_val = float(treat_val) if treat_val else None
            except ValueError:
                treat_val = None

            seval_data[utterance][metric] = {"control": ctrl_val, "treatment": treat_val}

    total_rows = len(rows_data)
    unique_count = len(seval_data)
    dup_count = len(duplicate_utterances)
    print(f"Loaded SEVAL metrics: {total_rows} total rows, {unique_count} unique utterances, {dup_count} duplicate utterances excluded")
    return seval_data, duplicate_utterances


def _compute_seval_comparison(
    results: list[dict], seval_data: dict[str, dict], metric_names: list[str],
    seval_duplicates: set[str]
) -> dict:
    """
    Compute SEVAL metrics comparison matched by utterance.

    Only includes utterances that:
    1. Have good gain metrics in the LLM NDCG output
    2. Exist in SEVAL data (unique, non-duplicate)
    3. Are not duplicated in the LLM NDCG output

    Args:
        results: LLM NDCG output records (only those with good gain metrics)
        seval_data: SEVAL metrics data keyed by utterance (unique only)
        metric_names: List of metric names to compare
        seval_duplicates: Set of duplicate utterances in SEVAL (for reporting)

    Returns:
        Comparison statistics dict
    """
    # First pass: detect duplicates in LLM NDCG results
    ndcg_utterance_counts: dict[str, int] = {}
    for record in results:
        utterance = record.get("utterance", "")  # No stripping - use full utterance
        if utterance:
            ndcg_utterance_counts[utterance] = ndcg_utterance_counts.get(utterance, 0) + 1

    ndcg_duplicates = {u for u, count in ndcg_utterance_counts.items() if count > 1}

    # Second pass: compute comparison for unique utterances only
    comparison = {metric: {"ctrl": [], "treat": [], "delta": []} for metric in metric_names}
    matched_count = 0
    unmatched_in_seval = 0
    skipped_ndcg_dup = 0
    skipped_seval_dup = 0

    for record in results:
        utterance = record.get("utterance", "")  # No stripping - use full utterance
        if not utterance:
            continue

        # Skip if duplicate in LLM NDCG output
        if utterance in ndcg_duplicates:
            skipped_ndcg_dup += 1
            continue

        # Skip if duplicate in SEVAL
        if utterance in seval_duplicates:
            skipped_seval_dup += 1
            continue

        # Skip if not found in SEVAL
        if utterance not in seval_data:
            unmatched_in_seval += 1
            continue

        matched_count += 1
        seval_record = seval_data[utterance]

        for metric in metric_names:
            if metric not in seval_record:
                continue

            ctrl_val = seval_record[metric].get("control")
            treat_val = seval_record[metric].get("treatment")

            if ctrl_val is not None and treat_val is not None:
                comparison[metric]["ctrl"].append(ctrl_val)
                comparison[metric]["treat"].append(treat_val)
                comparison[metric]["delta"].append(treat_val - ctrl_val)

    # Compute statistics
    stats = {
        "total_good_gain_utterances": len(results),
        "matched_utterances": matched_count,
        "unmatched_in_seval": unmatched_in_seval,
        "skipped_ndcg_duplicates": len(ndcg_duplicates),
        "skipped_seval_duplicates": len(seval_duplicates),
        "metrics": {},
    }

    for metric in metric_names:
        ctrl_vals = comparison[metric]["ctrl"]
        treat_vals = comparison[metric]["treat"]
        delta_vals = comparison[metric]["delta"]

        if not ctrl_vals:
            continue

        ctrl_wins = sum(1 for c, t in zip(ctrl_vals, treat_vals) if c > t)
        treat_wins = sum(1 for c, t in zip(ctrl_vals, treat_vals) if t > c)
        ties = len(ctrl_vals) - ctrl_wins - treat_wins

        stats["metrics"][metric] = {
            "ctrl_avg": statistics.mean(ctrl_vals),
            "treat_avg": statistics.mean(treat_vals),
            "delta_avg": statistics.mean(delta_vals),
            "ctrl_wins": ctrl_wins,
            "treat_wins": treat_wins,
            "ties": ties,
            "treat_win_rate": (treat_wins / len(ctrl_vals) * 100) if ctrl_vals else 0,
            "count": len(ctrl_vals),
        }

    return stats


def _format_seval_table(stats: dict) -> str:
    """Format the SEVAL metrics comparison table in markdown."""
    lines = []
    lines.append("")
    lines.append("## SEVAL Metrics Comparison (CiteDCG)")
    lines.append("")

    # Statistics summary
    lines.append("**Utterance Statistics:**")
    lines.append(f"- Total utterances with Good Gain metrics: {stats['total_good_gain_utterances']}")
    lines.append(f"- Matched in SEVAL (unique): {stats['matched_utterances']}")
    lines.append(f"- Not found in SEVAL: {stats['unmatched_in_seval']}")
    if stats['skipped_ndcg_duplicates'] > 0:
        lines.append(f"- Skipped (duplicate in Good Gain data): {stats['skipped_ndcg_duplicates']}")
    if stats['skipped_seval_duplicates'] > 0:
        lines.append(f"- Skipped (duplicate in SEVAL data): {stats['skipped_seval_duplicates']}")
    lines.append("")

    if not stats["metrics"]:
        lines.append("*No SEVAL metrics data available for matched utterances.*")
        return "\n".join(lines)

    lines.append("| Metric | Ctrl Avg | Treat Avg | Delta | Ctrl Wins | Treat Wins | Ties | Treat Win% |")
    lines.append("|--------|----------|-----------|-------|-----------|------------|------|------------|")

    for metric, m in stats["metrics"].items():
        lines.append(
            f"| {metric} | {m['ctrl_avg']:.4f} | {m['treat_avg']:.4f} | "
            f"{m['delta_avg']:+.4f} | {m['ctrl_wins']} | {m['treat_wins']} | "
            f"{m['ties']} | {m['treat_win_rate']:.1f}% |"
        )

    return "\n".join(lines)


# =============================================================================
# Helper Functions
# =============================================================================


def analyze_side(asr_data: dict) -> dict:
    """
    Analyze one side (control or treatment) of AllSearchResults.

    Args:
        asr_data: AllSearchResults dict (supports both wrapped and direct formats)

    Returns:
        Analysis dict with iteration count, total results, etc.
    """
    # Handle both formats:
    # Format 1: {"AllSearchResults": {"1": {"plugin": [...]}}}
    # Format 2: {"1": {"plugin": [...]}}  (direct iteration keys)
    all_search_results = asr_data.get("AllSearchResults", asr_data)

    iterations_with_results = 0
    total_invocations = 0
    total_results = 0
    max_iteration = 0

    for iteration_str, plugins in all_search_results.items():
        if not isinstance(plugins, dict):
            continue

        try:
            iteration = int(iteration_str)
            max_iteration = max(max_iteration, iteration)
        except ValueError:
            continue

        iteration_has_results = False

        for plugin_name, search_results in plugins.items():
            if not isinstance(search_results, list):
                continue

            for sr in search_results:
                pi = sr.get("PluginInvocation")
                results = sr.get("Results", [])

                if pi:
                    total_invocations += 1
                if results:
                    total_results += len(results)
                    iteration_has_results = True

        if iteration_has_results:
            iterations_with_results += 1

    return {
        "max_iteration": max_iteration,
        "iterations_with_results": iterations_with_results,
        "total_invocations": total_invocations,
        "total_results": total_results,
    }


def parse_record(record: dict) -> tuple[dict, dict]:
    """
    Parse a record and return control and treatment analysis.

    Args:
        record: A single record from the JSONL file

    Returns:
        Tuple of (ctrl_analysis, treat_analysis)
    """
    ctrl_data = record.get("all_search_results_control", {})
    treat_data = record.get("all_search_results_treatment", {})

    # Handle string format
    if isinstance(ctrl_data, str):
        ctrl_data = json.loads(ctrl_data)
    if isinstance(treat_data, str):
        treat_data = json.loads(treat_data)

    ctrl_analysis = analyze_side(ctrl_data)
    treat_analysis = analyze_side(treat_data)

    return ctrl_analysis, treat_analysis


# =============================================================================
# Command: select_top
# =============================================================================


def calculate_quality_score(ctrl_analysis: dict, treat_analysis: dict) -> float:
    """
    Calculate quality score for an utterance.

    Higher score = better quality candidate for iteration-based good gain comparison.

    Criteria:
    - Both sides must have invocations and results (required)
    - More iterations on both sides = higher score (primary)
    - Balanced iterations between control and treatment (secondary)
    - Penalize highly imbalanced iteration counts

    Score formula balances two goals:
    1. More iterations = better (for richer comparison data)
    2. Balanced iterations = better (for fair per-iteration comparison)

    Examples: (5,4)=86.4 > (3,3)=72 > (1,5)=16.8
    """
    # Must have both sides with data
    if ctrl_analysis["total_invocations"] == 0 or ctrl_analysis["total_results"] == 0:
        return 0
    if treat_analysis["total_invocations"] == 0 or treat_analysis["total_results"] == 0:
        return 0

    ctrl_iter = ctrl_analysis["iterations_with_results"]
    treat_iter = treat_analysis["iterations_with_results"]

    # Minimum iterations (ensures both sides have meaningful data)
    min_iter = min(ctrl_iter, treat_iter)
    max_iter = max(ctrl_iter, treat_iter)

    # Balance ratio: 1.0 when perfectly balanced, lower when imbalanced
    balance_ratio = min_iter / max_iter if max_iter > 0 else 0

    # Primary: reward total iterations (both sides contribute)
    # This ensures (5,4)=9 beats (3,3)=6 in raw iteration count
    total_iter_score = (ctrl_iter + treat_iter) * 8

    # Secondary: reward balanced iterations via min_iter
    # min_iter ensures both sides participate meaningfully
    min_iter_score = min_iter * 4

    # Balance penalty: reduce score for highly imbalanced cases
    # Use sqrt to soften the penalty: sqrt(0.8)=0.89, sqrt(0.5)=0.71, sqrt(0.2)=0.45
    import math
    balance_factor = math.sqrt(balance_ratio)

    # Combine: total iterations weighted more, balance as a soft penalty
    iteration_score = (total_iter_score + min_iter_score) * balance_factor

    # Small bonus for total results
    result_score = (
        min(ctrl_analysis["total_results"], 100) + min(treat_analysis["total_results"], 100)
    ) / 20

    return iteration_score + result_score


def extract_lines(input_file: str, line_numbers: list[int], output_file: str) -> None:
    """
    Extract specific lines from input file and save to output file.

    Args:
        input_file: Path to input JSONL file
        line_numbers: List of line numbers to extract (1-based)
        output_file: Path to output file
    """
    line_set = set(line_numbers)
    extracted = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line_num in line_set:
                extracted.append(line)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(extracted)

    print(f"Extracted {len(extracted)} lines to: {output_file}")


def select_top(
    input: str,
    output: str = None,
    count: int = 200,
    show: int = 20,
) -> None:
    """
    Select top quality utterances from llm_ndcg input files.

    Selects utterances based on quality criteria:
    - Both control and treatment have tool invocations and results
    - More iterations with tool invocations and results is better
    - More total results = higher score

    Args:
        input: Input JSONL file path
        output: Output file to save selected utterances (optional)
        count: Number of top utterances to select (default: 200)
        show: Number of top utterances to show in summary (default: 20)
    """
    print(f"Analyzing: {input}")
    print()

    analyses = []

    with open(input, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                record_id = record.get("id", f"line_{line_num}")
                utterance = record.get("utterance", "")

                ctrl_analysis, treat_analysis = parse_record(record)
                score = calculate_quality_score(ctrl_analysis, treat_analysis)

                if score > 0:  # Only include valid candidates
                    ctrl_iter = ctrl_analysis["iterations_with_results"]
                    treat_iter = treat_analysis["iterations_with_results"]
                    balance = min(ctrl_iter, treat_iter) / max(ctrl_iter, treat_iter) if max(ctrl_iter, treat_iter) > 0 else 0

                    analyses.append({
                        "line": line_num,
                        "id": record_id,
                        "utterance": utterance[:100],
                        "score": score,
                        "ctrl_iterations": ctrl_iter,
                        "ctrl_results": ctrl_analysis["total_results"],
                        "treat_iterations": treat_iter,
                        "treat_results": treat_analysis["total_results"],
                        "balance": balance,
                    })

            except Exception as e:
                print(f"Warning: Error parsing line {line_num}: {e}")

    # Sort by score descending
    analyses.sort(key=lambda x: x["score"], reverse=True)

    print(f"Found {len(analyses)} valid candidates (both sides have invocations and results)")
    print()

    # Show top utterances
    print(f"=== TOP {show} UTTERANCES (Balanced Iterations) ===")
    print(f"{'#':<4} {'Score':<8} {'Bal%':<6} {'Ctrl':<12} {'Treat':<12} {'Utterance':<50}")
    print("-" * 100)

    for i, a in enumerate(analyses[:show], 1):
        ctrl_info = f"{a['ctrl_iterations']}it/{a['ctrl_results']}r"
        treat_info = f"{a['treat_iterations']}it/{a['treat_results']}r"
        balance_pct = a['balance'] * 100
        utterance = a["utterance"][:45] + "..." if len(a["utterance"]) > 45 else a["utterance"]
        print(f"{i:<4} {a['score']:<8.1f} {balance_pct:<5.0f}% {ctrl_info:<12} {treat_info:<12} {utterance:<50}")

    print()

    # Score distribution
    if analyses:
        scores = [a["score"] for a in analyses]
        print(f"Score distribution:")
        print(f"  Min: {min(scores):.1f}, Max: {max(scores):.1f}, Avg: {sum(scores)/len(scores):.1f}")
        print()

    # Extract to output file
    if output:
        top_n = analyses[:count]
        if top_n:
            line_numbers = [a["line"] for a in top_n]
            extract_lines(input, line_numbers, output)
            print(f"\nSelected {len(top_n)} utterances with scores ranging from {top_n[-1]['score']:.1f} to {top_n[0]['score']:.1f}")
        else:
            print(f"\nNo utterances to select - no valid data found in input file.")
            print("This may happen if LLM labeling failed for all results.")


# =============================================================================
# Command: iteration_distribution
# =============================================================================


def iteration_distribution(
    input: str,
    show_details: bool = False,
) -> None:
    """
    Analyze and compare iteration distributions between control and treatment.

    Shows how many search iterations each side typically uses. This helps understand
    if treatment (with better search) needs fewer iterations than control.

    Args:
        input: Input JSONL file path
        show_details: Show per-utterance details (default: False)
    """
    print(f"Analyzing iteration distributions: {input}")
    print()

    ctrl_iterations = []
    treat_iterations = []
    ctrl_results = []
    treat_results = []
    details = []

    total_records = 0
    valid_records = 0

    with open(input, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            total_records += 1

            try:
                record = json.loads(line)
                ctrl_analysis, treat_analysis = parse_record(record)

                # Only count records where both sides have data
                if ctrl_analysis["total_results"] > 0 and treat_analysis["total_results"] > 0:
                    valid_records += 1
                    ctrl_iter = ctrl_analysis["iterations_with_results"]
                    treat_iter = treat_analysis["iterations_with_results"]
                    ctrl_res = ctrl_analysis["total_results"]
                    treat_res = treat_analysis["total_results"]

                    ctrl_iterations.append(ctrl_iter)
                    treat_iterations.append(treat_iter)
                    ctrl_results.append(ctrl_res)
                    treat_results.append(treat_res)

                    if show_details:
                        details.append({
                            "line": line_num,
                            "utterance": record.get("utterance", "")[:50],
                            "ctrl_iter": ctrl_iter,
                            "treat_iter": treat_iter,
                            "ctrl_res": ctrl_res,
                            "treat_res": treat_res,
                            "iter_delta": ctrl_iter - treat_iter,
                        })

            except Exception as e:
                print(f"Warning: Error parsing line {line_num}: {e}")

    print(f"Total records: {total_records}")
    print(f"Valid records (both sides have results): {valid_records}")
    print()

    if not ctrl_iterations:
        print("No valid records found.")
        return

    # Calculate distributions
    ctrl_counter = Counter(ctrl_iterations)
    treat_counter = Counter(treat_iterations)

    # Get all possible iteration counts
    all_iters = sorted(set(ctrl_counter.keys()) | set(treat_counter.keys()))

    # Print distribution comparison
    print("=" * 70)
    print("ITERATION DISTRIBUTION COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Iterations':<12} {'Control':<15} {'Treatment':<15} {'Ctrl %':<10} {'Treat %':<10}")
    print("-" * 70)

    for iter_count in all_iters:
        ctrl_count = ctrl_counter.get(iter_count, 0)
        treat_count = treat_counter.get(iter_count, 0)
        ctrl_pct = (ctrl_count / valid_records) * 100
        treat_pct = (treat_count / valid_records) * 100
        print(f"{iter_count:<12} {ctrl_count:<15} {treat_count:<15} {ctrl_pct:<10.1f} {treat_pct:<10.1f}")

    print("-" * 70)
    print(f"{'Total':<12} {sum(ctrl_counter.values()):<15} {sum(treat_counter.values()):<15}")
    print()

    # Iteration combination distribution
    combo_counter = Counter(zip(ctrl_iterations, treat_iterations))
    print("=" * 70)
    print("ITERATION COMBINATION DISTRIBUTION (Control, Treatment)")
    print("=" * 70)
    print()
    print(f"{'(Ctrl, Treat)':<15} {'Count':<10} {'%':<10} {'Balance':<10}")
    print("-" * 50)

    # Sort by count descending, then by combination
    sorted_combos = sorted(combo_counter.items(), key=lambda x: (-x[1], x[0]))
    for (ctrl_i, treat_i), count in sorted_combos:
        pct = count / valid_records * 100
        balance = min(ctrl_i, treat_i) / max(ctrl_i, treat_i) if max(ctrl_i, treat_i) > 0 else 0
        balance_str = f"{balance*100:.0f}%"
        print(f"({ctrl_i}, {treat_i}){'':<8} {count:<10} {pct:<9.1f}% {balance_str:<10}")

    # Show balanced vs imbalanced summary
    balanced_count = sum(count for (c, t), count in combo_counter.items() if c == t)
    off_by_one = sum(count for (c, t), count in combo_counter.items() if abs(c - t) == 1)
    imbalanced = valid_records - balanced_count - off_by_one

    print("-" * 50)
    print(f"Perfectly balanced (C=T):  {balanced_count:>6} ({balanced_count/valid_records*100:.1f}%)")
    print(f"Off by one (|C-T|=1):      {off_by_one:>6} ({off_by_one/valid_records*100:.1f}%)")
    print(f"Imbalanced (|C-T|>1):      {imbalanced:>6} ({imbalanced/valid_records*100:.1f}%)")
    print()

    # Summary statistics
    ctrl_avg = sum(ctrl_iterations) / len(ctrl_iterations)
    treat_avg = sum(treat_iterations) / len(treat_iterations)
    ctrl_max = max(ctrl_iterations)
    treat_max = max(treat_iterations)

    # Results statistics
    ctrl_res_avg = sum(ctrl_results) / len(ctrl_results)
    treat_res_avg = sum(treat_results) / len(treat_results)
    ctrl_res_total = sum(ctrl_results)
    treat_res_total = sum(treat_results)

    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"{'Metric':<25} {'Control':<15} {'Treatment':<15}")
    print("-" * 70)
    print(f"{'Average iterations':<25} {ctrl_avg:<15.2f} {treat_avg:<15.2f}")
    print(f"{'Max iterations':<25} {ctrl_max:<15} {treat_max:<15}")
    print(f"{'Single iteration %':<25} {(ctrl_counter.get(1, 0) / valid_records * 100):<14.1f}% {(treat_counter.get(1, 0) / valid_records * 100):<14.1f}%")
    print(f"{'Multi-iteration %':<25} {((valid_records - ctrl_counter.get(1, 0)) / valid_records * 100):<14.1f}% {((valid_records - treat_counter.get(1, 0)) / valid_records * 100):<14.1f}%")
    print()
    print(f"{'Average results':<25} {ctrl_res_avg:<15.1f} {treat_res_avg:<15.1f}")
    print(f"{'Total results':<25} {ctrl_res_total:<15} {treat_res_total:<15}")
    print(f"{'Results per iteration':<25} {ctrl_res_total/sum(ctrl_iterations):<15.1f} {treat_res_total/sum(treat_iterations):<15.1f}")
    print()

    # Comparison: cases where treatment uses fewer iterations
    fewer_count = sum(1 for c, t in zip(ctrl_iterations, treat_iterations) if t < c)
    same_count = sum(1 for c, t in zip(ctrl_iterations, treat_iterations) if t == c)
    more_count = sum(1 for c, t in zip(ctrl_iterations, treat_iterations) if t > c)

    print("=" * 70)
    print("ITERATION COMPARISON (Treatment vs Control)")
    print("=" * 70)
    print(f"Treatment uses FEWER iterations: {fewer_count:>6} ({fewer_count/valid_records*100:.1f}%)")
    print(f"Treatment uses SAME iterations:  {same_count:>6} ({same_count/valid_records*100:.1f}%)")
    print(f"Treatment uses MORE iterations:  {more_count:>6} ({more_count/valid_records*100:.1f}%)")
    print()

    # Show details if requested
    if show_details and details:
        print("=" * 80)
        print("PER-UTTERANCE DETAILS")
        print("=" * 80)
        print(f"{'Line':<6} {'C_Iter':<7} {'T_Iter':<7} {'C_Res':<7} {'T_Res':<7} {'Utterance':<45}")
        print("-" * 80)
        for d in details[:50]:  # Limit to 50 for readability
            print(f"{d['line']:<6} {d['ctrl_iter']:<7} {d['treat_iter']:<7} {d['ctrl_res']:<7} {d['treat_res']:<7} {d['utterance']:<45}")


# =============================================================================
# Command: compare_good_gain
# =============================================================================

# Summary metrics to compare
SUMMARY_METRICS = [
    "final_CG",
    "final_RG",
    "final_DCG",
    "final_DRG",
    "final_RAG",
    "final_DRAG",
    "final_SRE",
    "final_SRR",
]

# Per-iteration metrics (cumulative at iteration i)
ITERATION_METRICS = [
    "CG_at_i",
    "RG_at_i",
    "DCG_at_i",
    "DRG_at_i",
    "RAG_at_i",
    "DRAG_at_i",
    "SRE_at_i",
    "SRR_at_i",
]

# Efficiency metrics
EFFICIENCY_METRICS = [
    "search_iteration_count",
    "total_results",
    "total_good_results",
    "total_duplicates",
]


def _load_results(input_file: str) -> list[dict]:
    """Load results from JSONL output file."""
    results = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def _extract_good_gain_metrics(record: dict) -> tuple[Optional[dict], Optional[dict]]:
    """Extract control and treatment good gain metrics from a record."""
    ctrl = record.get("control", {}).get("retrieved_good_gain")
    treat = record.get("treatment", {}).get("retrieved_good_gain")
    return ctrl, treat


def _compute_summary_comparison(results: list[dict]) -> dict:
    """Compute summary metrics comparison across all utterances."""
    comparison = {metric: {"ctrl": [], "treat": [], "delta": []} for metric in SUMMARY_METRICS}

    valid_count = 0
    for record in results:
        ctrl, treat = _extract_good_gain_metrics(record)
        if not ctrl or not treat:
            continue

        ctrl_summary = ctrl.get("summary", {})
        treat_summary = treat.get("summary", {})

        valid_count += 1
        for metric in SUMMARY_METRICS:
            ctrl_val = ctrl_summary.get(metric, 0) or 0
            treat_val = treat_summary.get(metric, 0) or 0
            comparison[metric]["ctrl"].append(ctrl_val)
            comparison[metric]["treat"].append(treat_val)
            comparison[metric]["delta"].append(treat_val - ctrl_val)

    stats = {"valid_utterances": valid_count, "metrics": {}}
    for metric in SUMMARY_METRICS:
        ctrl_vals = comparison[metric]["ctrl"]
        treat_vals = comparison[metric]["treat"]
        deltas = comparison[metric]["delta"]

        if not ctrl_vals:
            continue

        treat_wins = sum(1 for d in deltas if d > 0.001)
        ctrl_wins = sum(1 for d in deltas if d < -0.001)
        ties = len(deltas) - treat_wins - ctrl_wins

        stats["metrics"][metric] = {
            "ctrl_avg": statistics.mean(ctrl_vals),
            "treat_avg": statistics.mean(treat_vals),
            "delta_avg": statistics.mean(deltas),
            "ctrl_std": statistics.stdev(ctrl_vals) if len(ctrl_vals) > 1 else 0,
            "treat_std": statistics.stdev(treat_vals) if len(treat_vals) > 1 else 0,
            "treat_wins": treat_wins,
            "ctrl_wins": ctrl_wins,
            "ties": ties,
            "treat_win_rate": treat_wins / len(deltas) * 100 if deltas else 0,
        }

    return stats


def _compute_iteration_comparison(results: list[dict], max_iterations: int = 10) -> dict:
    """Compute per-iteration metrics comparison."""
    iteration_data = {
        i: {metric: {"ctrl": [], "treat": []} for metric in ITERATION_METRICS}
        for i in range(1, max_iterations + 1)
    }

    for record in results:
        ctrl, treat = _extract_good_gain_metrics(record)
        if not ctrl or not treat:
            continue

        ctrl_iterations = {it["iteration"]: it for it in ctrl.get("iterations", [])}
        treat_iterations = {it["iteration"]: it for it in treat.get("iterations", [])}

        common_iterations = set(ctrl_iterations.keys()) & set(treat_iterations.keys())

        for i in common_iterations:
            if i > max_iterations:
                continue

            ctrl_it = ctrl_iterations[i]
            treat_it = treat_iterations[i]

            for metric in ITERATION_METRICS:
                ctrl_val = ctrl_it.get(metric, 0) or 0
                treat_val = treat_it.get(metric, 0) or 0
                iteration_data[i][metric]["ctrl"].append(ctrl_val)
                iteration_data[i][metric]["treat"].append(treat_val)

    stats = {}
    for i in range(1, max_iterations + 1):
        sample_metric = ITERATION_METRICS[0]
        if not iteration_data[i][sample_metric]["ctrl"]:
            continue

        count = len(iteration_data[i][sample_metric]["ctrl"])
        stats[i] = {"utterance_count": count, "metrics": {}}

        for metric in ITERATION_METRICS:
            ctrl_vals = iteration_data[i][metric]["ctrl"]
            treat_vals = iteration_data[i][metric]["treat"]

            if not ctrl_vals:
                continue

            deltas = [t - c for c, t in zip(ctrl_vals, treat_vals)]
            treat_wins = sum(1 for d in deltas if d > 0.001)
            ctrl_wins = sum(1 for d in deltas if d < -0.001)

            stats[i]["metrics"][metric] = {
                "ctrl_avg": statistics.mean(ctrl_vals),
                "treat_avg": statistics.mean(treat_vals),
                "delta_avg": statistics.mean(deltas),
                "treat_wins": treat_wins,
                "ctrl_wins": ctrl_wins,
                "treat_win_rate": treat_wins / len(deltas) * 100 if deltas else 0,
            }

    return stats


def _compute_efficiency_comparison(results: list[dict]) -> dict:
    """Compute efficiency metrics comparison."""
    comparison = {metric: {"ctrl": [], "treat": []} for metric in EFFICIENCY_METRICS}

    valid_count = 0
    for record in results:
        ctrl, treat = _extract_good_gain_metrics(record)
        if not ctrl or not treat:
            continue

        ctrl_summary = ctrl.get("summary", {})
        treat_summary = treat.get("summary", {})

        valid_count += 1
        for metric in EFFICIENCY_METRICS:
            ctrl_val = ctrl_summary.get(metric, 0) or 0
            treat_val = treat_summary.get(metric, 0) or 0
            comparison[metric]["ctrl"].append(ctrl_val)
            comparison[metric]["treat"].append(treat_val)

    stats = {"valid_utterances": valid_count, "metrics": {}}
    for metric in EFFICIENCY_METRICS:
        ctrl_vals = comparison[metric]["ctrl"]
        treat_vals = comparison[metric]["treat"]

        if not ctrl_vals:
            continue

        stats["metrics"][metric] = {
            "ctrl_avg": statistics.mean(ctrl_vals),
            "treat_avg": statistics.mean(treat_vals),
            "ctrl_total": sum(ctrl_vals),
            "treat_total": sum(treat_vals),
        }

    if "search_iteration_count" in stats["metrics"] and valid_count > 0:
        ctrl_rag_sum = 0
        treat_rag_sum = 0
        for record in results:
            ctrl, treat = _extract_good_gain_metrics(record)
            if not ctrl or not treat:
                continue
            ctrl_summary = ctrl.get("summary", {})
            treat_summary = treat.get("summary", {})
            ctrl_iter = ctrl_summary.get("search_iteration_count", 1) or 1
            treat_iter = treat_summary.get("search_iteration_count", 1) or 1
            ctrl_rag_sum += (ctrl_summary.get("final_RAG", 0) or 0) / ctrl_iter
            treat_rag_sum += (treat_summary.get("final_RAG", 0) or 0) / treat_iter

        stats["metrics"]["RAG_per_iteration"] = {
            "ctrl_avg": ctrl_rag_sum / valid_count,
            "treat_avg": treat_rag_sum / valid_count,
        }

    return stats


def _format_summary_table(stats: dict) -> str:
    """Format the primary summary comparison table in markdown."""
    lines = []
    lines.append("# Good Gain Metrics Comparison Report")
    lines.append("")
    lines.append("## Primary Table: Summary Metrics (final_* metrics)")
    lines.append("")
    lines.append(f"**Valid utterances with both control and treatment data:** {stats['valid_utterances']}")
    lines.append("")

    lines.append("| Metric | Ctrl Avg | Treat Avg | Delta | Ctrl Wins | Treat Wins | Ties | Treat Win% |")
    lines.append("|--------|----------|-----------|-------|-----------|------------|------|------------|")

    for metric in SUMMARY_METRICS:
        if metric not in stats["metrics"]:
            continue
        m = stats["metrics"][metric]
        lines.append(
            f"| {metric} | {m['ctrl_avg']:.4f} | {m['treat_avg']:.4f} | "
            f"{m['delta_avg']:+.4f} | {m['ctrl_wins']} | {m['treat_wins']} | "
            f"{m['ties']} | {m['treat_win_rate']:.1f}% |"
        )

    return "\n".join(lines)


def _format_iteration_table(stats: dict) -> str:
    """Format the secondary iteration comparison table in markdown."""
    lines = []
    lines.append("")
    lines.append("## Secondary Table: Per-Iteration Comparison")
    lines.append("")
    lines.append("*Only includes iterations where both control and treatment have data.*")
    lines.append("")

    if not stats:
        lines.append("No common iterations found between control and treatment.")
        return "\n".join(lines)

    for iteration in sorted(stats.keys()):
        it_stats = stats[iteration]
        n = it_stats['utterance_count']
        lines.append(f"### Iteration @{iteration} (n={n} utterances)")
        lines.append("")

        lines.append("| Metric | Ctrl Avg | Treat Avg | Delta | Ctrl Wins | Treat Wins | Ties | Treat Win% |")
        lines.append("|--------|----------|-----------|-------|-----------|------------|------|------------|")

        for metric in ITERATION_METRICS:
            if metric not in it_stats["metrics"]:
                continue
            m = it_stats["metrics"][metric]
            delta = m["treat_avg"] - m["ctrl_avg"]
            ties = n - m["ctrl_wins"] - m["treat_wins"]
            lines.append(
                f"| {metric} | {m['ctrl_avg']:.4f} | {m['treat_avg']:.4f} | "
                f"{delta:+.4f} | {m['ctrl_wins']} | {m['treat_wins']} | "
                f"{ties} | {m['treat_win_rate']:.1f}% |"
            )
        lines.append("")

    return "\n".join(lines)


def _format_efficiency_table(stats: dict) -> str:
    """Format the efficiency comparison table in markdown."""
    lines = []
    lines.append("")
    lines.append("## Efficiency Table: Search Efficiency Comparison")
    lines.append("")

    lines.append("| Metric | Ctrl Avg | Treat Avg | Ctrl Total | Treat Total |")
    lines.append("|--------|----------|-----------|------------|-------------|")

    for metric in EFFICIENCY_METRICS:
        if metric not in stats["metrics"]:
            continue
        m = stats["metrics"][metric]
        ctrl_total = m.get("ctrl_total", "")
        treat_total = m.get("treat_total", "")
        lines.append(
            f"| {metric} | {m['ctrl_avg']:.2f} | {m['treat_avg']:.2f} | "
            f"{ctrl_total} | {treat_total} |"
        )

    if "RAG_per_iteration" in stats["metrics"]:
        m = stats["metrics"]["RAG_per_iteration"]
        lines.append(f"| RAG_per_iteration | {m['ctrl_avg']:.4f} | {m['treat_avg']:.4f} | - | - |")

    return "\n".join(lines)


def _find_top_bottom_cases(results: list[dict], metric: str = "final_RAG", top_k: int = 5) -> tuple[list, list]:
    """Find top and bottom k cases by delta (treatment - control)."""
    cases = []

    for idx, record in enumerate(results):
        ctrl, treat = _extract_good_gain_metrics(record)
        if not ctrl or not treat:
            continue

        ctrl_summary = ctrl.get("summary", {})
        treat_summary = treat.get("summary", {})

        ctrl_val = ctrl_summary.get(metric, 0) or 0
        treat_val = treat_summary.get(metric, 0) or 0
        delta = treat_val - ctrl_val

        cases.append({
            "index": idx + 1,
            "id": record.get("Id", "")[:16],
            "utterance": record.get("utterance", "")[:60],
            "ctrl_val": ctrl_val,
            "treat_val": treat_val,
            "delta": delta,
            "ctrl_results": ctrl_summary.get("total_results", 0),
            "treat_results": treat_summary.get("total_results", 0),
            "ctrl_good": ctrl_summary.get("total_good_results", 0),
            "treat_good": treat_summary.get("total_good_results", 0),
        })

    cases.sort(key=lambda x: x["delta"], reverse=True)

    top_cases = cases[:top_k]
    bottom_cases = cases[-top_k:][::-1]

    return top_cases, bottom_cases


def _format_top_bottom_cases(top_cases: list, bottom_cases: list, metric: str) -> str:
    """Format top and bottom cases in markdown."""
    lines = []
    lines.append("")
    lines.append(f"## Top/Bottom Cases by {metric} Delta")
    lines.append("")
    lines.append("*Delta = Treatment - Control*")
    lines.append("")

    lines.append(f"### Top {len(top_cases)} Cases (Treatment Better)")
    lines.append("")
    lines.append("| # | Delta | Ctrl | Treat | C_Res | T_Res | C_Good | T_Good | Utterance |")
    lines.append("|---|-------|------|-------|-------|-------|--------|--------|-----------|")

    for case in top_cases:
        utterance = case["utterance"][:40] + "..." if len(case["utterance"]) > 40 else case["utterance"]
        lines.append(
            f"| {case['index']} | {case['delta']:+.4f} | {case['ctrl_val']:.4f} | {case['treat_val']:.4f} | "
            f"{case['ctrl_results']} | {case['treat_results']} | "
            f"{case['ctrl_good']} | {case['treat_good']} | {utterance} |"
        )

    lines.append("")
    lines.append(f"### Bottom {len(bottom_cases)} Cases (Control Better)")
    lines.append("")
    lines.append("| # | Delta | Ctrl | Treat | C_Res | T_Res | C_Good | T_Good | Utterance |")
    lines.append("|---|-------|------|-------|-------|-------|--------|--------|-----------|")

    for case in bottom_cases:
        utterance = case["utterance"][:40] + "..." if len(case["utterance"]) > 40 else case["utterance"]
        lines.append(
            f"| {case['index']} | {case['delta']:+.4f} | {case['ctrl_val']:.4f} | {case['treat_val']:.4f} | "
            f"{case['ctrl_results']} | {case['treat_results']} | "
            f"{case['ctrl_good']} | {case['treat_good']} | {utterance} |"
        )

    return "\n".join(lines)


def _format_interpretation_guide() -> str:
    """Format the interpretation guide in markdown."""
    lines = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- **Delta > 0**: Treatment is better")
    lines.append("- **Treat Win% > 50%**: Treatment wins more utterances")
    lines.append("- **RAG/DRAG/SRE**: Higher is better (0-1 scale, normalized)")
    lines.append("- **CG/DCG/RG/DRG**: Higher is better (absolute gain)")
    lines.append("- **RAG_per_iteration**: Higher means more efficient search")
    return "\n".join(lines)


def _print_console_summary(stats: dict, efficiency_stats: dict, seval_stats: dict = None) -> None:
    """Print brief summary to console."""
    print("\n--- Quick Summary ---")
    valid = stats['valid_utterances']
    print(f"Valid utterances: {valid}")

    if "final_RAG" in stats["metrics"]:
        m = stats["metrics"]["final_RAG"]
        print(f"final_RAG: Ctrl={m['ctrl_avg']:.4f}, Treat={m['treat_avg']:.4f}, Delta={m['delta_avg']:+.4f}, Treat Win%={m['treat_win_rate']:.1f}%")

    if "final_DRAG" in stats["metrics"]:
        m = stats["metrics"]["final_DRAG"]
        print(f"final_DRAG: Ctrl={m['ctrl_avg']:.4f}, Treat={m['treat_avg']:.4f}, Delta={m['delta_avg']:+.4f}, Treat Win%={m['treat_win_rate']:.1f}%")

    if "final_SRE" in stats["metrics"]:
        m = stats["metrics"]["final_SRE"]
        print(f"final_SRE: Ctrl={m['ctrl_avg']:.4f}, Treat={m['treat_avg']:.4f}, Delta={m['delta_avg']:+.4f}, Treat Win%={m['treat_win_rate']:.1f}%")

    if "RAG_per_iteration" in efficiency_stats["metrics"]:
        m = efficiency_stats["metrics"]["RAG_per_iteration"]
        print(f"RAG_per_iteration: Ctrl={m['ctrl_avg']:.4f}, Treat={m['treat_avg']:.4f}")

    # Print SEVAL metrics summary if available
    if seval_stats and seval_stats.get("metrics"):
        print(f"\n--- SEVAL Metrics (matched: {seval_stats['matched_utterances']}) ---")
        for metric, m in seval_stats["metrics"].items():
            print(f"{metric}: Ctrl={m['ctrl_avg']:.4f}, Treat={m['treat_avg']:.4f}, Delta={m['delta_avg']:+.4f}, Treat Win%={m['treat_win_rate']:.1f}%")


def compare_good_gain(
    input: str,
    output: str = None,
    output_json: str = None,
    max_iterations: int = 10,
    top_k: int = 5,
    case_metric: str = "final_RAG",
    verbose: bool = False,
    seval_path: str = None,
    seval_metrics: str = None,
) -> None:
    """
    Compare Retrieved Good Gain metrics between control and treatment.

    Generates comparison tables for all good gain metrics including:
    - Primary table: Summary metrics (final_* metrics)
    - Secondary table: Per-iteration comparison
    - Efficiency table: Search efficiency metrics
    - Top/bottom cases by delta
    - SEVAL metrics comparison (optional, when seval_path is provided)

    Args:
        input: Input JSONL file (llm_ndcg output)
        output: Output markdown file for detailed report (optional)
        output_json: Output JSON file for structured results (optional)
        max_iterations: Maximum iterations to compare (default: 10)
        top_k: Number of top/bottom cases to show (default: 5)
        case_metric: Metric for top/bottom case selection (default: final_RAG)
        verbose: Print full report to console (default: summary only)
        seval_path: Path to SEVAL job folder for CiteDCG comparison (optional)
        seval_metrics: Comma-separated SEVAL metric names (default: citedcg_one_centric,citedcg_num_enterprise_cites)
    """
    print(f"Loading results from: {input}")
    results = _load_results(input)
    print(f"Total records: {len(results)}")

    print("Computing comparisons...")
    summary_stats = _compute_summary_comparison(results)
    iteration_stats = _compute_iteration_comparison(results, max_iterations)
    efficiency_stats = _compute_efficiency_comparison(results)

    # Load and compute SEVAL metrics if path provided
    seval_stats = None
    if seval_path:
        metric_list = seval_metrics.split(",") if seval_metrics else DEFAULT_SEVAL_METRICS
        metric_list = [m.strip() for m in metric_list]
        print(f"Loading SEVAL metrics from: {seval_path}")
        seval_data, seval_duplicates = _load_seval_metrics(seval_path, metric_list)
        if seval_data:
            seval_stats = _compute_seval_comparison(results, seval_data, metric_list, seval_duplicates)

    top_cases, bottom_cases = [], []
    if top_k > 0:
        top_cases, bottom_cases = _find_top_bottom_cases(results, case_metric, top_k)

    report_sections = [
        _format_summary_table(summary_stats),
    ]

    # Insert SEVAL table after summary if available
    if seval_stats:
        report_sections.append(_format_seval_table(seval_stats))

    report_sections.extend([
        _format_iteration_table(iteration_stats),
        _format_efficiency_table(efficiency_stats),
    ])

    if top_k > 0:
        report_sections.append(_format_top_bottom_cases(top_cases, bottom_cases, case_metric))
    report_sections.append(_format_interpretation_guide())

    full_report = "\n".join(report_sections)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"Full report saved to: {output}")
        _print_console_summary(summary_stats, efficiency_stats, seval_stats)
    elif verbose:
        print(full_report)
    else:
        _print_console_summary(summary_stats, efficiency_stats, seval_stats)

    if output_json:
        output_data = {
            "summary_comparison": summary_stats,
            "iteration_comparison": iteration_stats,
            "efficiency_comparison": efficiency_stats,
        }
        if seval_stats:
            output_data["seval_comparison"] = seval_stats
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"JSON results saved to: {output_json}")

    print("\nDone.")


if __name__ == "__main__":
    fire.Fire({
        "select_top": select_top,
        "iteration_distribution": iteration_distribution,
        "compare_good_gain": compare_good_gain,
        # NOTE: top_duplicates moved to duplicate_analysis.py (uses actual metric dedup logic)
    })
