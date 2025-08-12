#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Selected Utterance Statistics Analyzer - A tool to analyze utterance selection statistics from selected_utterances.json files.

This script analyzes the output of the utterance selector and provides:
1. Number of utterances in each selected_round & total number of rounds
2. Total number of utterances
3. Additional statistics about category distribution and selection patterns
4. Visualization plots showing complete category distribution

Example usage:
    # Basic analysis with console output only
    python utterance_statistics.py data/results/selected_utterances.json

    # Save markdown report
    python utterance_statistics.py data/results/selected_utterances.json --output_path=report.md

    # Save JSON report
    python utterance_statistics.py data/results/selected_utterances.json --output_path=report.json

    # Generate plots and save report
    python utterance_statistics.py data/results/selected_utterances.json --output_path=report.md --create_plots=True

    # Generate plots with custom directory
    python utterance_statistics.py data/results/selected_utterances.json --create_plots=True --plot_output_dir=visualizations
"""

import os
import sys
import json
import fire
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
import logging

# Import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.style as mplstyle
    import numpy as np

    PLOTTING_AVAILABLE = True
except ImportError as e:
    print(
        f"Warning: Plotting libraries not available. Please install matplotlib: pip install matplotlib"
    )
    print(f"Error: {e}")
    PLOTTING_AVAILABLE = False

# Set up logger properly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PLOTS_DIRNAME = "plots"


def escape_markdown_html(text: str) -> str:
    """
    Escape HTML/XML-like tags in text for proper markdown display.

    Args:
        text: Text that may contain HTML/XML-like tags

    Returns:
        Text with angle brackets escaped
    """
    if not text:
        return text
    return text.replace("<", "&lt;").replace(">", "&gt;")


def read_selected_utterances_file(file_path: str) -> Dict[str, Any]:
    """
    Read a selected_utterances.json file and return the parsed data.

    Args:
        file_path: Path to the selected_utterances.json file

    Returns:
        Dictionary containing the parsed JSON data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded data from: {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise


def analyze_utterance_statistics(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze utterance selection statistics from the loaded data.

    Args:
        data: Dictionary containing selected_utterances data

    Returns:
        Dictionary containing comprehensive statistics
    """
    selected_utterances = data.get("selected_utterances", [])

    if not selected_utterances:
        logger.warning("No selected_utterances found in the data")
        return {
            "total_utterances": 0,
            "total_rounds": 0,
            "utterances_per_round": {},
            "categories": {},
            "prompt_length_distribution": {},
            "duplication_statistics": {},
            "error": "No selected_utterances found",
        }

    # Initialize counters
    round_counts = Counter()
    category_counts = Counter()
    round_category_breakdown = defaultdict(lambda: defaultdict(int))

    # New counters for prompt length and duplication
    utterance_texts = []
    prompt_lengths = []
    utterance_counter = Counter()

    # Analyze each utterance
    for utterance in selected_utterances:
        selected_round = utterance.get("selected_round", "unknown")
        selected_category = utterance.get("selected_category", "unknown")
        utterance_text = utterance.get("utterance", "")

        # Count utterances per round
        round_counts[selected_round] += 1

        # Count utterances per category
        category_counts[selected_category] += 1

        # Count utterances per round per category
        round_category_breakdown[selected_round][selected_category] += 1

        # Collect utterance texts for duplication analysis
        if utterance_text:
            utterance_texts.append(utterance_text)
            utterance_counter[utterance_text] += 1
            prompt_lengths.append(len(utterance_text))

    # Calculate basic statistics
    total_utterances = len(selected_utterances)
    total_rounds = len(round_counts) if round_counts else 0

    # Convert defaultdict to regular dict for JSON serialization
    round_category_breakdown = {
        str(round_num): dict(categories)
        for round_num, categories in round_category_breakdown.items()
    }

    # Sort rounds for better readability
    utterances_per_round = dict(sorted(round_counts.items()))

    # Get top categories
    top_categories = dict(category_counts.most_common())

    # Analyze prompt length distribution
    prompt_length_stats = {}
    if prompt_lengths:
        prompt_lengths.sort()
        total_prompts = len(prompt_lengths)

        prompt_length_stats = {
            "total_prompts_analyzed": total_prompts,
            "min_length": min(prompt_lengths),
            "max_length": max(prompt_lengths),
            "avg_length": sum(prompt_lengths) / total_prompts,
            "median_length": (
                prompt_lengths[total_prompts // 2] if total_prompts > 0 else 0
            ),
            "length_distribution": {
                "0-50": sum(1 for l in prompt_lengths if 0 <= l <= 50),
                "51-100": sum(1 for l in prompt_lengths if 51 <= l <= 100),
                "101-200": sum(1 for l in prompt_lengths if 101 <= l <= 200),
                "201-300": sum(1 for l in prompt_lengths if 201 <= l <= 300),
                "301-500": sum(1 for l in prompt_lengths if 301 <= l <= 500),
                "501+": sum(1 for l in prompt_lengths if l > 500),
            },
        }

    # Analyze duplication statistics
    duplicates = {text: count for text, count in utterance_counter.items() if count > 1}
    unique_utterances = len(set(utterance_texts))
    total_with_text = len(utterance_texts)

    duplication_stats = {
        "total_utterances_with_text": total_with_text,
        "unique_utterances": unique_utterances,
        "duplicate_utterances": total_with_text - unique_utterances,
        "duplication_rate": (
            (total_with_text - unique_utterances) / total_with_text * 100
            if total_with_text > 0
            else 0
        ),
        "most_duplicated": utterance_counter.most_common(5),
        "duplicate_groups": len(duplicates),
        "duplicate_breakdown": {
            "2_copies": sum(1 for count in duplicates.values() if count == 2),
            "3_copies": sum(1 for count in duplicates.values() if count == 3),
            "4_copies": sum(1 for count in duplicates.values() if count == 4),
            "5+_copies": sum(1 for count in duplicates.values() if count >= 5),
        },
    }

    # Analyze category frequency distribution (how many categories have X utterances)
    category_frequency_stats = {}
    if category_counts:
        category_utterance_counts = list(category_counts.values())
        category_utterance_counts.sort(reverse=True)  # Sort in descending order
        total_categories = len(category_utterance_counts)

        category_frequency_stats = {
            "total_categories_analyzed": total_categories,
            "min_utterances_per_category": min(category_utterance_counts),
            "max_utterances_per_category": max(category_utterance_counts),
            "avg_utterances_per_category": sum(category_utterance_counts)
            / total_categories,
            "median_utterances_per_category": (
                category_utterance_counts[total_categories // 2]
                if total_categories > 0
                else 0
            ),
            "category_frequency_distribution": {
                "1": sum(1 for count in category_utterance_counts if count == 1),
                "2-5": sum(1 for count in category_utterance_counts if 2 <= count <= 5),
                "6-10": sum(
                    1 for count in category_utterance_counts if 6 <= count <= 10
                ),
                "11-20": sum(
                    1 for count in category_utterance_counts if 11 <= count <= 20
                ),
                "21-50": sum(
                    1 for count in category_utterance_counts if 21 <= count <= 50
                ),
                "51+": sum(1 for count in category_utterance_counts if count > 50),
            },
        }

    statistics = {
        "total_utterances": total_utterances,
        "total_rounds": total_rounds,
        "utterances_per_round": utterances_per_round,
        "round_category_breakdown": round_category_breakdown,
        "categories": {
            "total_categories": len(category_counts),
            "category_distribution": top_categories,
        },
        "prompt_length_distribution": prompt_length_stats,
        "category_frequency_distribution": category_frequency_stats,
        "duplication_statistics": duplication_stats,
        "summary": {
            "avg_utterances_per_round": (
                total_utterances / total_rounds if total_rounds > 0 else 0
            ),
            "most_productive_round": (
                max(round_counts.keys(), key=lambda x: round_counts[x])
                if round_counts
                else None
            ),
            "most_common_category": (
                category_counts.most_common(1)[0] if category_counts else None
            ),
        },
    }

    return statistics


def print_statistics_summary(stats: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the statistics to the console.

    Args:
        stats: Dictionary containing the statistics
    """
    print("\n" + "=" * 60)
    print("UTTERANCE SELECTION STATISTICS")
    print("=" * 60)

    print(f"📊 Total Utterances: {stats['total_utterances']}")
    print(f"🔄 Total Rounds: {stats['total_rounds']}")

    if stats["total_rounds"] > 0:
        print(
            f"📈 Average per Round: {stats['summary']['avg_utterances_per_round']:.1f}"
        )
        print(
            f"🎯 Most Productive Round: {stats['summary']['most_productive_round']} ({stats['utterances_per_round'][stats['summary']['most_productive_round']]} utterances)"
        )

    print(f"\n📂 Categories: {stats['categories']['total_categories']} total")
    if stats["summary"]["most_common_category"]:
        category, count = stats["summary"]["most_common_category"]
        print(f"🏆 Most Common Category: {category} ({count} utterances)")

    print(f"\n🔢 Utterances per Round:")
    for round_num, count in stats["utterances_per_round"].items():
        print(f"   Round {round_num}: {count} utterances")

    # Display prompt length distribution
    if stats.get("prompt_length_distribution"):
        length_stats = stats["prompt_length_distribution"]
        if length_stats and length_stats.get("length_distribution"):
            print(f"\n📏 Prompt Length Distribution:")
            print(
                f"   Average Length: {length_stats.get('avg_length', 0):.1f} characters"
            )
            print(
                f"   Length Range: {length_stats.get('min_length', 0)} - {length_stats.get('max_length', 0)} characters"
            )
            print(
                f"   Median Length: {length_stats.get('median_length', 0)} characters"
            )

            # Show distribution ranges and counts
            length_dist = length_stats.get("length_distribution", {})
            for range_name, count in length_dist.items():
                if count > 0:
                    print(f"   {range_name} chars: {count} utterances")

    # Display category frequency distribution
    if stats.get("category_frequency_distribution"):
        cat_freq_stats = stats["category_frequency_distribution"]
        if cat_freq_stats and cat_freq_stats.get("category_frequency_distribution"):
            print(f"\n📊 Category Frequency Distribution:")
            print(
                f"   Average per Category: {cat_freq_stats.get('avg_utterances_per_category', 0):.1f} utterances"
            )
            print(
                f"   Range: {cat_freq_stats.get('min_utterances_per_category', 0)} - {cat_freq_stats.get('max_utterances_per_category', 0)} utterances"
            )
            print(
                f"   Median per Category: {cat_freq_stats.get('median_utterances_per_category', 0)} utterances"
            )

            # Show distribution ranges and counts
            cat_freq_dist = cat_freq_stats.get("category_frequency_distribution", {})
            for range_name, count in cat_freq_dist.items():
                if count > 0:
                    print(f"   {range_name} utterances: {count} categories")

    # Display duplication statistics
    if stats.get("duplication_statistics"):
        dup_stats = stats["duplication_statistics"]
        print(f"\n🔄 Duplication Statistics:")
        print(f"   Unique Utterances: {dup_stats.get('unique_utterances', 0)}")
        print(f"   Duplicate Utterances: {dup_stats.get('duplicate_utterances', 0)}")
        print(f"   Duplication Rate: {dup_stats.get('duplication_rate', 0):.1f}%")
        print(f"   Duplicate Groups: {dup_stats.get('duplicate_groups', 0)}")

        # Show duplication breakdown
        if dup_stats.get("duplicate_breakdown"):
            dup_breakdown = dup_stats["duplicate_breakdown"]
            for copies_type, count in dup_breakdown.items():
                if count > 0:
                    if copies_type == "2_copies":
                        print(f"   Duplicated 2x: {count} groups")
                    elif copies_type == "3_copies":
                        print(f"   Duplicated 3x: {count} groups")
                    elif copies_type == "4_copies":
                        print(f"   Duplicated 4x: {count} groups")
                    elif copies_type == "5+_copies":
                        print(f"   Duplicated 5+x: {count} groups")

    print("=" * 60 + "\n")


def generate_markdown_report(
    stats: Dict[str, Any], input_file: str, timestamp: str
) -> str:
    """
    Generate a markdown report from statistics.

    Args:
        stats: Dictionary containing the statistics
        input_file: Path to the input file
        timestamp: Analysis timestamp

    Returns:
        Markdown formatted report as a string
    """
    md_lines = []
    md_lines.append("# Utterance Selection Statistics Report\n")
    md_lines.append(f"**Input File:** `{input_file}`\n")
    md_lines.append(f"**Analysis Date:** {timestamp}\n")
    md_lines.append("---\n")

    # Basic Statistics
    md_lines.append("## Summary Statistics\n")
    md_lines.append(f"- **Total Utterances:** {stats['total_utterances']}")
    md_lines.append(f"- **Total Rounds:** {stats['total_rounds']}")

    if stats["total_rounds"] > 0:
        md_lines.append(
            f"- **Average per Round:** {stats['summary']['avg_utterances_per_round']:.1f}"
        )
        md_lines.append(
            f"- **Most Productive Round:** {stats['summary']['most_productive_round']} ({stats['utterances_per_round'][stats['summary']['most_productive_round']]} utterances)"
        )

    md_lines.append(
        f"- **Categories:** {stats['categories']['total_categories']} total"
    )
    if stats["summary"]["most_common_category"]:
        category, count = stats["summary"]["most_common_category"]
        escaped_category = escape_markdown_html(category)
        md_lines.append(
            f"- **Most Common Category:** `{escaped_category}` ({count} utterances)"
        )

    md_lines.append("")

    # Round Distribution
    md_lines.append("## Utterances per Round\n")
    for round_num, count in stats["utterances_per_round"].items():
        md_lines.append(f"- **Round {round_num}:** {count} utterances")
    md_lines.append("")

    # Prompt Length Distribution
    if stats.get("prompt_length_distribution"):
        length_stats = stats["prompt_length_distribution"]
        if length_stats:
            md_lines.append("## Prompt Length Distribution\n")
            md_lines.append(
                f"- **Total Analyzed:** {length_stats.get('total_prompts_analyzed', 0)}"
            )
            md_lines.append(
                f"- **Average Length:** {length_stats.get('avg_length', 0):.1f} characters"
            )
            md_lines.append(
                f"- **Length Range:** {length_stats.get('min_length', 0)} - {length_stats.get('max_length', 0)} characters"
            )
            md_lines.append(
                f"- **Median Length:** {length_stats.get('median_length', 0)} characters"
            )
            md_lines.append("")

            if length_stats.get("length_distribution"):
                md_lines.append("### Length Breakdown\n")
                for range_name, count in length_stats["length_distribution"].items():
                    percentage = (
                        count / length_stats.get("total_prompts_analyzed", 1)
                    ) * 100
                    md_lines.append(
                        f"- **{range_name} chars:** {count} ({percentage:.1f}%)"
                    )
                md_lines.append("")

    # Category Frequency Distribution
    if stats.get("category_frequency_distribution"):
        cat_freq_stats = stats["category_frequency_distribution"]
        if cat_freq_stats:
            md_lines.append("## Category Frequency Distribution\n")
            md_lines.append(
                f"- **Total Categories Analyzed:** {cat_freq_stats.get('total_categories_analyzed', 0)}"
            )
            md_lines.append(
                f"- **Average per Category:** {cat_freq_stats.get('avg_utterances_per_category', 0):.1f} utterances"
            )
            md_lines.append(
                f"- **Range:** {cat_freq_stats.get('min_utterances_per_category', 0)} - {cat_freq_stats.get('max_utterances_per_category', 0)} utterances"
            )
            md_lines.append(
                f"- **Median per Category:** {cat_freq_stats.get('median_utterances_per_category', 0)} utterances"
            )
            md_lines.append("")

            if cat_freq_stats.get("category_frequency_distribution"):
                md_lines.append("### Category Frequency Breakdown\n")
                for range_name, count in cat_freq_stats[
                    "category_frequency_distribution"
                ].items():
                    percentage = (
                        count / cat_freq_stats.get("total_categories_analyzed", 1)
                    ) * 100
                    md_lines.append(
                        f"- **{range_name} utterances:** {count} categories ({percentage:.1f}%)"
                    )
                md_lines.append("")

    # Duplication Statistics
    if stats.get("duplication_statistics"):
        dup_stats = stats["duplication_statistics"]
        md_lines.append("## Duplication Statistics\n")
        md_lines.append(
            f"- **Unique Utterances:** {dup_stats.get('unique_utterances', 0)}"
        )
        md_lines.append(
            f"- **Duplicate Utterances:** {dup_stats.get('duplicate_utterances', 0)}"
        )
        md_lines.append(
            f"- **Duplication Rate:** {dup_stats.get('duplication_rate', 0):.1f}%"
        )
        md_lines.append(
            f"- **Duplicate Groups:** {dup_stats.get('duplicate_groups', 0)}"
        )
        md_lines.append("")

        if dup_stats.get("most_duplicated"):
            md_lines.append("### Most Duplicated Utterances\n")
            for utterance, count in dup_stats["most_duplicated"][:3]:
                truncated = utterance[:80] + "..." if len(utterance) > 80 else utterance
                md_lines.append(f"- **{count}x:** {truncated}")
            md_lines.append("")

        if dup_stats.get("duplicate_breakdown"):
            md_lines.append("### Duplicate Breakdown\n")
            for copies, count in dup_stats["duplicate_breakdown"].items():
                if count > 0:
                    md_lines.append(f"- **{copies}:** {count} groups")
            md_lines.append("")

    # All Categories (complete list for detailed report)
    if stats["categories"]["category_distribution"]:
        md_lines.append("## All Categories\n")
        for category, count in stats["categories"]["category_distribution"].items():
            escaped_category = escape_markdown_html(category)
            md_lines.append(f"- **{escaped_category}:** {count}")
        md_lines.append("")

    return "\n".join(md_lines)


def plot_round_distribution(
    stats: Dict[str, Any],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Create a bar plot of round distribution.

    Args:
        stats: Dictionary containing the statistics
        output_path: Path to save the plot (optional)
        figsize: Figure size (width, height)
    """
    if not PLOTTING_AVAILABLE:
        logger.warning(
            "Plotting libraries not available. Skipping round distribution plot."
        )
        return

    utterances_per_round = stats.get("utterances_per_round", {})
    if not utterances_per_round:
        logger.warning("No round data available for plotting")
        return

    # Extract rounds and counts
    rounds = list(utterances_per_round.keys())
    counts = list(utterances_per_round.values())

    # Set up the plot
    plt.figure(figsize=figsize)

    # Create the bar plot
    bars = plt.bar(
        range(len(rounds)),
        counts,
        color="skyblue",
        alpha=0.8,
        edgecolor="navy",
        linewidth=1,
    )

    # Customize the plot
    plt.title("Utterance Distribution by Round", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Selection Round", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Utterances", fontsize=12, fontweight="bold")

    # Set x-axis labels
    plt.xticks(range(len(rounds)), [f"Round {r}" for r in rounds])

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(counts) * 0.01,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Add grid for better readability
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    # Add summary text
    total_utterances = sum(counts)
    avg_per_round = total_utterances / len(rounds) if rounds else 0
    summary_text = (
        f"Total: {total_utterances} utterances\nAverage: {avg_per_round:.1f} per round"
    )

    plt.text(
        0.02,
        0.98,
        summary_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        fontsize=9,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_category_distribution(
    stats: Dict[str, Any],
    output_path: Optional[str] = None,
    top_n: int = 15,
    figsize: Tuple[int, int] = (16, 8),
    show_all: bool = False,
) -> None:
    """
    Create a bar plot of category distribution.

    Args:
        stats: Dictionary containing the statistics
        output_path: Path to save the plot (optional)
        top_n: Number of top categories to show (ignored if show_all=True)
        figsize: Figure size (width, height)
        show_all: If True, show all categories without names for complete distribution view
    """
    if not PLOTTING_AVAILABLE:
        logger.warning(
            "Plotting libraries not available. Skipping category distribution plot."
        )
        return

    category_distribution = stats.get("categories", {}).get("category_distribution", {})
    if not category_distribution:
        logger.warning("No category data available for plotting")
        return

    # Determine what to plot
    if show_all:
        # Show all categories without names for complete distribution view
        categories_to_plot = list(category_distribution.items())
        title = (
            f"Complete Category Distribution - All {len(categories_to_plot)} Categories"
        )
        use_category_labels = False
        figsize = (20, 8)  # Wider figure for all categories
    else:
        # Show top N categories with names
        categories_to_plot = list(category_distribution.items())[:top_n]
        title = f"Top {len(categories_to_plot)} Categories - Utterance Distribution"
        use_category_labels = True

    if not categories_to_plot:
        logger.warning("No categories to plot")
        return

    # Extract categories and counts
    categories, counts = zip(*categories_to_plot)

    # Set up the plot
    plt.figure(figsize=figsize)

    # Create the bar plot
    if show_all:
        # For all categories, use a more condensed bar plot without individual labels
        bars = plt.bar(
            range(len(categories)),
            counts,
            color="lightcoral",
            alpha=0.8,
            edgecolor="darkred",
            linewidth=0.2,  # Thinner edges for dense plot
            width=0.8,  # Slightly thinner bars for better density
        )

        # No individual value labels on bars (too many to be readable)
        # Only show x-axis as category indices
        plt.xlabel(
            "Category Index (sorted by utterance count)", fontsize=12, fontweight="bold"
        )

        # Set x-axis ticks to show every nth category for reference
        num_ticks = min(20, len(categories))  # Show at most 20 tick marks
        tick_positions = [
            int(i * len(categories) / num_ticks) for i in range(num_ticks + 1)
        ]
        tick_labels = [
            str(i + 1) for i in tick_positions
        ]  # 1-based indexing for display
        plt.xticks(tick_positions, tick_labels, fontsize=9)

    else:
        # For top categories, show with names and individual labels
        bars = plt.bar(
            range(len(categories)),
            counts,
            color="lightcoral",
            alpha=0.8,
            edgecolor="darkred",
            linewidth=0.5,
        )

        # Format category names for display (truncate long names)
        formatted_categories = []
        for cat in categories:
            if len(cat) > 50:
                if "|" in cat:
                    parts = cat.split("|")
                    if len(parts[0]) <= 50:
                        formatted_categories.append(parts[0] + "|...")
                    else:
                        formatted_categories.append(cat[:47] + "...")
                else:
                    formatted_categories.append(cat[:47] + "...")
            else:
                formatted_categories.append(cat)

        # Set x-axis labels with rotation
        plt.xticks(
            range(len(categories)),
            formatted_categories,
            rotation=45,
            ha="right",
            fontsize=9,
        )
        plt.xlabel("Categories", fontsize=12, fontweight="bold")

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(counts) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    # Customize the plot
    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.ylabel("Number of Utterances", fontsize=12, fontweight="bold")

    # Add grid for better readability
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    # Add summary text
    total_categories = stats.get("categories", {}).get("total_categories", 0)
    total_utterances = sum(category_distribution.values())
    shown_categories = len(categories_to_plot)

    if show_all:
        # Summary for complete distribution
        max_count = max(counts) if counts else 0
        min_count = min(counts) if counts else 0
        avg_count = sum(counts) / len(counts) if counts else 0

        summary_text = f"All {shown_categories} categories shown\nRange: {min_count}-{max_count} utterances\nAverage: {avg_count:.1f} utterances/category\nTotal: {total_utterances} utterances"
    else:
        # Summary for top categories
        summary_text = f"Showing top {shown_categories} of {total_categories} total categories\nTotal utterances: {total_utterances}"

    plt.text(
        0.02,
        0.98,
        summary_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        fontsize=9,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_length_distribution(
    stats: Dict[str, Any],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Create a histogram of prompt length distribution.

    Args:
        stats: Dictionary containing the statistics
        output_path: Path to save the plot (optional)
        figsize: Figure size (width, height)
    """
    if not PLOTTING_AVAILABLE:
        logger.warning(
            "Plotting libraries not available. Skipping length distribution plot."
        )
        return

    length_stats = stats.get("prompt_length_distribution", {})
    if not length_stats or not length_stats.get("length_distribution"):
        logger.warning("No length distribution data available for plotting")
        return

    length_distribution = length_stats["length_distribution"]

    # Extract ranges and counts
    ranges = list(length_distribution.keys())
    counts = list(length_distribution.values())

    # Set up the plot
    plt.figure(figsize=figsize)

    # Create the bar plot
    bars = plt.bar(
        range(len(ranges)),
        counts,
        color="lightgreen",
        alpha=0.8,
        edgecolor="darkgreen",
        linewidth=1,
    )

    # Customize the plot
    plt.title("Prompt Length Distribution", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Character Length Range", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Utterances", fontsize=12, fontweight="bold")

    # Set x-axis labels
    plt.xticks(range(len(ranges)), ranges)

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        if count > 0:  # Only show label if count > 0
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(counts) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Add grid for better readability
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    # Add summary text
    total_analyzed = length_stats.get("total_prompts_analyzed", 0)
    avg_length = length_stats.get("avg_length", 0)
    median_length = length_stats.get("median_length", 0)

    summary_text = f"Total analyzed: {total_analyzed}\nAvg: {avg_length:.1f} chars\nMedian: {median_length} chars"

    plt.text(
        0.02,
        0.98,
        summary_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8),
        fontsize=9,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def create_all_plots(
    stats: Dict[str, Any],
    output_dir: str,
    top_categories: int = 15,
) -> None:
    """
    Create all available plots for the statistics.

    Args:
        stats: Dictionary containing the statistics
        output_dir: Directory to save plots
        top_categories: Number of top categories to show in comparison plot
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting libraries not available. Cannot create plots.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get absolute path for better logging
    abs_output_dir = os.path.abspath(output_dir)

    logger.info("Creating visualization plots...")

    # Plot round distribution
    round_output = os.path.join(output_dir, "round_distribution.png")
    plot_round_distribution(stats, output_path=round_output)

    # Plot complete category distribution (all categories)
    category_output = os.path.join(output_dir, "category_distribution_complete.png")
    plot_category_distribution(stats, output_path=category_output, show_all=True)

    # Also create the top categories plot for comparison
    category_top_output = os.path.join(output_dir, "category_distribution_top.png")
    plot_category_distribution(
        stats, output_path=category_top_output, top_n=top_categories, show_all=False
    )

    # Plot length distribution
    length_output = os.path.join(output_dir, "length_distribution.png")
    plot_length_distribution(stats, output_path=length_output)

    logger.info(f"All plots saved to directory: {abs_output_dir}")


def main(
    input_path: str,
    output_path: Optional[str] = None,
    indent: int = 2,
    create_plots: bool = True,
    plot_output_dir: Optional[str] = None,
) -> None:
    """
    Analyze utterance selection statistics from a selected_utterances.json file.

    Args:
        input_path: Path to the selected_utterances.json file (required)
        output_path: Path to save the statistics report (.json/.md) (optional). If not specified, no output is generated.
        indent: Indentation level for JSON output (default: 2)
        create_plots: Whether to create visualization plots (default: True)
        plot_output_dir: Directory to save plots (default: None). If None, plots are saved to output_path directory + 'plots' if output_path specified, otherwise input_path directory + 'plots'.
    """
    try:
        # Validate input file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Read the selected utterances file
        logger.info(f"Reading utterance data from: {input_path}")
        data = read_selected_utterances_file(input_path)

        # Analyze statistics
        logger.info("Analyzing utterance statistics...")
        stats = analyze_utterance_statistics(data)

        # Print summary to console
        print_statistics_summary(stats)

        # Create plots if requested
        if create_plots:
            # Determine plot output directory based on the new logic
            if plot_output_dir is not None:
                # Use explicitly specified plot_output_dir
                plots_dir = plot_output_dir
            elif output_path is not None:
                # Use output_path directory + 'plots'
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    plots_dir = os.path.join(output_dir, DEFAULT_PLOTS_DIRNAME)
                else:
                    plots_dir = DEFAULT_PLOTS_DIRNAME
            else:
                # Use input_path directory + 'plots'
                input_dir = os.path.dirname(input_path)
                if input_dir:
                    plots_dir = os.path.join(input_dir, DEFAULT_PLOTS_DIRNAME)
                else:
                    plots_dir = DEFAULT_PLOTS_DIRNAME

            create_all_plots(
                stats,
                output_dir=plots_dir,
            )

        # Save detailed report if output path is provided
        if output_path:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Determine output format based on file extension
            file_ext = os.path.splitext(output_path)[1].lower()

            if file_ext == ".md":
                # Generate markdown report
                timestamp = data.get("timestamp", "unknown")
                markdown_content = generate_markdown_report(
                    stats, input_path, timestamp
                )

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(markdown_content)

                logger.info(f"Detailed markdown report saved to: {output_path}")

            else:
                # Default to JSON format
                report = {
                    "input_file": input_path,
                    "analysis_timestamp": data.get("timestamp", "unknown"),
                    "statistics": stats,
                }

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=indent, ensure_ascii=False)

                logger.info(f"Detailed JSON report saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Use fire for command line arguments
    fire.Fire(main)
