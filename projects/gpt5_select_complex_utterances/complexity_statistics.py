"""
Utterance Complexity Classification Statistics Analyzer

This program analyzes the results from utterance complexity classification
and provides comprehensive statistics including:
- Category-level statistics
- Classification distribution
- Confidence metrics
- Overall summaries
- Graphical visualizations

Author: Generated for BizChatScripts project
Date: August 6, 2025
"""

import json
import logging
import os
import sys
import fire
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

# Add parent directories to path to import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from utils.json_utils import read_json_file, write_json_file

# Try to import matplotlib for plotting, with graceful fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print(
        "Warning: matplotlib not available. Install matplotlib for plotting features: pip install matplotlib"
    )


def safe_round(value: float, digits: int = 2) -> float:
    """Safely round a float value to specified decimal places."""
    try:
        return round(float(value), digits)
    except (ValueError, TypeError):
        return 0.0


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplexityStatisticsAnalyzer:
    """
    Analyzer for utterance complexity classification statistics.
    """

    def __init__(self):
        self.categories = {}
        self.total_utterances = 0
        self.classification_counts = defaultdict(int)
        self.confidence_stats = defaultdict(list)
        self.error_counts = defaultdict(int)

    def load_and_analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Load classification results and perform comprehensive analysis.

        Args:
            file_path: Path to the JSON file containing classification results

        Returns:
            Dictionary containing all statistics
        """
        logger.info(f"Loading classification results from: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = read_json_file(file_path)
        logger.info(f"Loaded data for {len(data)} categories")

        # Handle both dict and list formats
        if isinstance(data, dict):
            # Data is a dictionary with category names as keys
            for category_name, utterances in data.items():
                self._analyze_category(category_name, utterances)
        elif isinstance(data, list):
            # Data is a list of results, create a single category
            self._analyze_category("default_category", data)
        else:
            raise ValueError(
                f"Unexpected data format. Expected dict or list, got {type(data)}"
            )

        # Generate comprehensive statistics
        stats = self._generate_statistics()

        return stats

    def _analyze_category(self, category_name: str, utterances: List[Dict]) -> None:
        """
        Analyze a single category's utterances.

        Args:
            category_name: Name of the category
            utterances: List of utterance classification results
        """
        if not utterances:
            logger.warning(f"Category '{category_name}' has no utterances")
            self.categories[category_name] = {
                "total_utterances": 0,
                "chat_count": 0,
                "cot_count": 0,
                "error_count": 0,
                "unknown_count": 0,
                "avg_confidence": 0.0,
                "confidence_range": (0.0, 0.0),
                "classification_distribution": {},
            }
            return

        category_stats = {
            "total_utterances": len(utterances),
            "chat_count": 0,
            "cot_count": 0,
            "error_count": 0,
            "unknown_count": 0,
            "confidences": [],
            "classification_distribution": defaultdict(int),
        }

        # Analyze each utterance in the category
        for utterance_data in utterances:
            if not isinstance(utterance_data, dict):
                logger.warning(
                    f"Invalid utterance data in category '{category_name}': {utterance_data}"
                )
                continue

            classification = utterance_data.get("classification", "unknown").lower()
            confidence = utterance_data.get("confidence", 0.0)

            # Count classifications
            if classification == "chat":
                category_stats["chat_count"] += 1
                self.classification_counts["chat"] += 1
            elif classification == "cot":
                category_stats["cot_count"] += 1
                self.classification_counts["cot"] += 1
            elif classification == "error":
                category_stats["error_count"] += 1
                self.classification_counts["error"] += 1
                self.error_counts[category_name] += 1
            else:
                category_stats["unknown_count"] += 1
                self.classification_counts["unknown"] += 1

            # Track confidence scores
            if isinstance(confidence, (int, float)) and confidence > 0:
                category_stats["confidences"].append(confidence)
                self.confidence_stats[classification].append(confidence)

            # Count distribution
            category_stats["classification_distribution"][classification] += 1

            self.total_utterances += 1

        # Calculate confidence statistics
        if category_stats["confidences"]:
            confidences = category_stats["confidences"]
            category_stats["avg_confidence"] = safe_round(
                sum(confidences) / len(confidences), 3
            )
            category_stats["confidence_range"] = (
                safe_round(min(confidences), 3),
                safe_round(max(confidences), 3),
            )
        else:
            category_stats["avg_confidence"] = 0.0
            category_stats["confidence_range"] = (0.0, 0.0)

        # Remove raw confidence list (not needed in final stats)
        del category_stats["confidences"]

        # Convert defaultdict to regular dict
        category_stats["classification_distribution"] = dict(
            category_stats["classification_distribution"]
        )

        self.categories[category_name] = category_stats

    def _generate_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistics from analyzed data.

        Returns:
            Dictionary containing all statistics
        """
        # Overall classification statistics
        total_classified = sum(self.classification_counts.values())

        overall_stats = {
            "total_categories": len(self.categories),
            "total_utterances": self.total_utterances,
            "total_classified": total_classified,
            "classification_summary": {
                "chat_model": {
                    "count": self.classification_counts["chat"],
                    "percentage": safe_round(
                        (self.classification_counts["chat"] / max(1, total_classified))
                        * 100,
                        1,
                    ),
                },
                "reasoning_model": {
                    "count": self.classification_counts["cot"],
                    "percentage": safe_round(
                        (self.classification_counts["cot"] / max(1, total_classified))
                        * 100,
                        1,
                    ),
                },
                "errors": {
                    "count": self.classification_counts["error"],
                    "percentage": safe_round(
                        (self.classification_counts["error"] / max(1, total_classified))
                        * 100,
                        1,
                    ),
                },
                "unknown": {
                    "count": self.classification_counts["unknown"],
                    "percentage": safe_round(
                        (
                            self.classification_counts["unknown"]
                            / max(1, total_classified)
                        )
                        * 100,
                        1,
                    ),
                },
            },
        }

        # Confidence statistics by classification type
        confidence_analysis = {}
        for classification_type, confidences in self.confidence_stats.items():
            if confidences:
                confidence_analysis[classification_type] = {
                    "count": len(confidences),
                    "average": safe_round(sum(confidences) / len(confidences), 3),
                    "min": safe_round(min(confidences), 3),
                    "max": safe_round(max(confidences), 3),
                    "median": safe_round(sorted(confidences)[len(confidences) // 2], 3),
                }
            else:
                confidence_analysis[classification_type] = {
                    "count": 0,
                    "average": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "median": 0.0,
                }

        # Category rankings
        categories_by_size = sorted(
            self.categories.items(),
            key=lambda x: x[1]["total_utterances"],
            reverse=True,
        )

        categories_by_complexity = sorted(
            self.categories.items(),
            key=lambda x: x[1]["cot_count"] / max(1, x[1]["total_utterances"]),
            reverse=True,
        )

        return {
            "file_analysis": {
                "overall_statistics": overall_stats,
                "confidence_analysis": confidence_analysis,
                "category_statistics": self.categories,
                "category_rankings": {
                    "by_utterance_count": [
                        (name, stats["total_utterances"])
                        for name, stats in categories_by_size[:10]
                    ],
                    "by_complexity_ratio": [
                        (
                            name,
                            safe_round(
                                stats["cot_count"] / max(1, stats["total_utterances"]),
                                3,
                            ),
                        )
                        for name, stats in categories_by_complexity[:10]
                    ],
                },
                "error_analysis": dict(self.error_counts) if self.error_counts else {},
            }
        }

    def create_distribution_plots(
        self, stats: Dict[str, Any], output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Create graphical distribution plots showing Chat vs CoT utterances across categories.

        Args:
            stats: Statistics dictionary from analyze method
            output_dir: Directory to save plots (defaults to same dir as script)

        Returns:
            List of paths to saved plot files
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available. Cannot create plots.")
            return []

        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))

        os.makedirs(output_dir, exist_ok=True)

        analysis = stats["file_analysis"]
        category_stats = analysis["category_statistics"]

        saved_plots = []

        # Plot 1: Overall Distribution Pie Chart
        saved_plots.append(self._create_overall_distribution_pie(analysis, output_dir))

        # Plot 2: Top Categories Bar Chart
        saved_plots.append(
            self._create_top_categories_bar_chart(category_stats, output_dir)
        )

        # Plot 3: Complexity Ratio Distribution
        saved_plots.append(
            self._create_complexity_ratio_distribution(category_stats, output_dir)
        )

        # Plot 4: Category Comparison Stacked Bar Chart
        saved_plots.append(
            self._create_category_comparison_chart(category_stats, output_dir)
        )

        logger.info(f"Created {len(saved_plots)} visualization plots in: {output_dir}")
        return saved_plots

    def _create_overall_distribution_pie(
        self, analysis: Dict[str, Any], output_dir: str
    ) -> str:
        """Create overall Chat vs CoT distribution pie chart."""
        overall = analysis["overall_statistics"]
        classification_summary = overall["classification_summary"]

        # Prepare data
        labels = []
        sizes = []
        colors = []

        chat_count = classification_summary["chat_model"]["count"]
        cot_count = classification_summary["reasoning_model"]["count"]
        error_count = classification_summary["errors"]["count"]
        unknown_count = classification_summary["unknown"]["count"]

        if chat_count > 0:
            labels.append(
                f'Chat Model\n{chat_count:,} ({classification_summary["chat_model"]["percentage"]}%)'
            )
            sizes.append(chat_count)
            colors.append("#4CAF50")  # Green

        if cot_count > 0:
            labels.append(
                f'Reasoning Model\n{cot_count:,} ({classification_summary["reasoning_model"]["percentage"]}%)'
            )
            sizes.append(cot_count)
            colors.append("#FF9800")  # Orange

        if error_count > 0:
            labels.append(
                f'Errors\n{error_count:,} ({classification_summary["errors"]["percentage"]}%)'
            )
            sizes.append(error_count)
            colors.append("#F44336")  # Red

        if unknown_count > 0:
            labels.append(
                f'Unknown\n{unknown_count:,} ({classification_summary["unknown"]["percentage"]}%)'
            )
            sizes.append(unknown_count)
            colors.append("#9E9E9E")  # Gray

        # Create pie chart
        plt.figure(figsize=(10, 8))
        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="",
            startangle=90,
            textprops={"fontsize": 10},
        )

        plt.title(
            "Overall Classification Distribution",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Add total count
        total_utterances = sum(sizes)
        plt.figtext(
            0.5,
            0.02,
            f"Total Utterances: {total_utterances:,}",
            ha="center",
            fontsize=12,
            style="italic",
        )

        plt.axis("equal")

        # Save plot
        filename = os.path.join(output_dir, "overall_distribution.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return filename

    def _create_top_categories_bar_chart(
        self, category_stats: Dict[str, Any], output_dir: str, top_n: int = 15
    ) -> str:
        """Create horizontal bar chart of top N categories by total utterances."""
        # Sort categories by total utterances
        sorted_categories = sorted(
            [
                (name, stats)
                for name, stats in category_stats.items()
                if stats["total_utterances"] > 0
            ],
            key=lambda x: x[1]["total_utterances"],
            reverse=True,
        )[:top_n]

        if not sorted_categories:
            return ""

        # Prepare data
        category_names = [
            name[:50] + ("..." if len(name) > 50 else "")
            for name, _ in sorted_categories
        ]
        chat_counts = [stats["chat_count"] for _, stats in sorted_categories]
        cot_counts = [stats["cot_count"] for _, stats in sorted_categories]
        error_counts = [stats["error_count"] for _, stats in sorted_categories]

        # Create stacked horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, max(8, len(category_names) * 0.4)))

        y_pos = np.arange(len(category_names))

        # Create stacked bars
        p1 = ax.barh(y_pos, chat_counts, color="#4CAF50", label="Chat Model", alpha=0.8)
        p2 = ax.barh(
            y_pos,
            cot_counts,
            left=chat_counts,
            color="#FF9800",
            label="Reasoning Model",
            alpha=0.8,
        )

        # Add error bars if there are any errors
        if any(error_counts):
            chat_cot_total = [c + r for c, r in zip(chat_counts, cot_counts)]
            p3 = ax.barh(
                y_pos,
                error_counts,
                left=chat_cot_total,
                color="#F44336",
                label="Errors",
                alpha=0.8,
            )

        # Customize chart
        ax.set_yticks(y_pos)
        ax.set_yticklabels(category_names)
        ax.set_xlabel("Number of Utterances", fontsize=12)
        ax.set_title(
            f"Top {len(category_names)} Categories by Utterance Count\n(Chat vs Reasoning Model Distribution)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Add legend
        ax.legend(loc="lower right")

        # Add value labels on bars
        for i, (chat, cot, error) in enumerate(
            zip(chat_counts, cot_counts, error_counts)
        ):
            total = chat + cot + error
            if total > 0:
                ax.text(
                    total + max(chat_counts + cot_counts) * 0.01,
                    i,
                    f"{total:,}",
                    va="center",
                    fontsize=9,
                )

        plt.tight_layout()

        # Save plot
        filename = os.path.join(output_dir, "top_categories_distribution.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return filename

    def _create_complexity_ratio_distribution(
        self, category_stats: Dict[str, Any], output_dir: str
    ) -> str:
        """Create histogram showing distribution of complexity ratios across categories."""
        # Calculate complexity ratios
        complexity_ratios = []
        category_names = []

        for name, stats in category_stats.items():
            total = stats["total_utterances"]
            if total > 0:  # Only include categories with utterances
                ratio = stats["cot_count"] / total
                complexity_ratios.append(ratio)
                category_names.append(name)

        if not complexity_ratios:
            return ""

        # Create histogram
        plt.figure(figsize=(12, 6))

        # Create histogram with bins
        n_bins = (
            min(20, len(complexity_ratios) // 2) if len(complexity_ratios) > 10 else 10
        )
        n, bins, patches = plt.hist(
            complexity_ratios,
            bins=n_bins,
            color="#2196F3",
            alpha=0.7,
            edgecolor="black",
        )

        # Color bars based on complexity level
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i + 1]) / 2
            if bin_center < 0.2:
                patch.set_facecolor("#4CAF50")  # Green for low complexity
            elif bin_center < 0.5:
                patch.set_facecolor("#FF9800")  # Orange for medium complexity
            else:
                patch.set_facecolor("#F44336")  # Red for high complexity

        plt.xlabel("Complexity Ratio (CoT utterances / Total utterances)", fontsize=12)
        plt.ylabel("Number of Categories", fontsize=12)
        plt.title(
            "Distribution of Category Complexity Ratios",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.grid(True, alpha=0.3)

        # Add statistics text
        avg_ratio = np.mean(complexity_ratios)
        median_ratio = np.median(complexity_ratios)
        plt.axvline(
            avg_ratio,
            color="red",
            linestyle="--",
            alpha=0.8,
            linewidth=2,
            label=f"Mean: {avg_ratio:.3f}",
        )
        plt.axvline(
            median_ratio,
            color="orange",
            linestyle="--",
            alpha=0.8,
            linewidth=2,
            label=f"Median: {median_ratio:.3f}",
        )

        plt.legend()
        plt.tight_layout()

        # Save plot
        filename = os.path.join(output_dir, "complexity_ratio_distribution.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return filename

    def _create_category_comparison_chart(
        self, category_stats: Dict[str, Any], output_dir: str, top_n: int = 20
    ) -> str:
        """Create scatter plot comparing total utterances vs complexity ratio."""
        # Prepare data
        total_utterances = []
        complexity_ratios = []
        category_names = []
        colors = []

        for name, stats in category_stats.items():
            total = stats["total_utterances"]
            if total > 0:  # Only include categories with utterances
                ratio = stats["cot_count"] / total
                total_utterances.append(total)
                complexity_ratios.append(ratio)
                category_names.append(name)

                # Color based on complexity ratio
                if ratio < 0.2:
                    colors.append("#4CAF50")  # Green for low complexity
                elif ratio < 0.5:
                    colors.append("#FF9800")  # Orange for medium complexity
                else:
                    colors.append("#F44336")  # Red for high complexity

        if not total_utterances:
            return ""

        # Create scatter plot
        plt.figure(figsize=(14, 8))

        # Create scatter plot with variable sizes
        sizes = [
            max(50, min(500, total * 2)) for total in total_utterances
        ]  # Scale point sizes
        scatter = plt.scatter(
            total_utterances,
            complexity_ratios,
            c=colors,
            s=sizes,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )

        plt.xlabel("Total Utterances (log scale)", fontsize=12)
        plt.ylabel("Complexity Ratio (CoT / Total)", fontsize=12)
        plt.title(
            "Category Comparison: Total Utterances vs Complexity Ratio",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xscale("log")
        plt.grid(True, alpha=0.3)

        # Add legend for colors
        green_patch = mpatches.Patch(color="#4CAF50", label="Low Complexity (< 20%)")
        orange_patch = mpatches.Patch(
            color="#FF9800", label="Medium Complexity (20-50%)"
        )
        red_patch = mpatches.Patch(color="#F44336", label="High Complexity (> 50%)")
        plt.legend(handles=[green_patch, orange_patch, red_patch], loc="upper right")

        # Annotate top categories by size and highest complexity
        sorted_by_size = sorted(
            zip(category_names, total_utterances, complexity_ratios),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_by_complexity = sorted(
            zip(category_names, total_utterances, complexity_ratios),
            key=lambda x: x[2],
            reverse=True,
        )

        # Annotate top 3 by size
        for i, (name, total, ratio) in enumerate(sorted_by_size[:3]):
            plt.annotate(
                name[:30] + ("..." if len(name) > 30 else ""),
                (total, ratio),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
                weight="bold",
            )

        # Annotate top 3 by complexity (if different from size)
        for i, (name, total, ratio) in enumerate(sorted_by_complexity[:3]):
            if ratio > 0.7:  # Only annotate very high complexity ones
                plt.annotate(
                    name[:30] + ("..." if len(name) > 30 else ""),
                    (total, ratio),
                    xytext=(5, -15),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.8,
                    style="italic",
                )

        plt.tight_layout()

        # Save plot
        filename = os.path.join(output_dir, "category_comparison_scatter.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return filename

    def print_summary(self, stats: Dict[str, Any]) -> None:
        """
        Print a formatted summary of the statistics.

        Args:
            stats: Statistics dictionary from analyze method
        """
        analysis = stats["file_analysis"]
        overall = analysis["overall_statistics"]

        print("\n" + "=" * 80)
        print("UTTERANCE COMPLEXITY CLASSIFICATION STATISTICS")
        print("=" * 80)

        # Overall summary
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"  Total Categories: {overall['total_categories']}")
        print(f"  Total Utterances: {overall['total_utterances']}")
        print(f"  Successfully Classified: {overall['total_classified']}")

        # Classification distribution
        classification_summary = overall["classification_summary"]
        print(f"\n🎯 CLASSIFICATION DISTRIBUTION:")
        print(
            f"  Chat Model: {classification_summary['chat_model']['count']} ({classification_summary['chat_model']['percentage']}%)"
        )
        print(
            f"  Reasoning Model: {classification_summary['reasoning_model']['count']} ({classification_summary['reasoning_model']['percentage']}%)"
        )
        if classification_summary["errors"]["count"] > 0:
            print(
                f"  Errors: {classification_summary['errors']['count']} ({classification_summary['errors']['percentage']}%)"
            )
        if classification_summary["unknown"]["count"] > 0:
            print(
                f"  Unknown: {classification_summary['unknown']['count']} ({classification_summary['unknown']['percentage']}%)"
            )

        # Confidence analysis
        if "confidence_analysis" in analysis:
            print(f"\n📈 CONFIDENCE ANALYSIS:")
            conf_analysis = analysis["confidence_analysis"]
            for class_type, conf_stats in conf_analysis.items():
                if conf_stats["count"] > 0:
                    print(
                        f"  {class_type.upper()}: avg={conf_stats['average']}, range=[{conf_stats['min']}-{conf_stats['max']}], median={conf_stats['median']}"
                    )

        # Top categories by size
        if "category_rankings" in analysis:
            rankings = analysis["category_rankings"]
            print(f"\n📋 TOP 10 CATEGORIES BY UTTERANCE COUNT:")
            for i, (category, count) in enumerate(
                rankings["by_utterance_count"][:10], 1
            ):
                print(f"  {i:2d}. {category[:60]:<60} ({count} utterances)")

            print(f"\n🧠 TOP 10 CATEGORIES BY COMPLEXITY RATIO:")
            for i, (category, ratio) in enumerate(
                rankings["by_complexity_ratio"][:10], 1
            ):
                print(f"  {i:2d}. {category[:60]:<60} ({ratio:.3f})")

        # Error analysis
        if analysis.get("error_analysis"):
            print(f"\n⚠️  CATEGORIES WITH ERRORS:")
            for category, error_count in analysis["error_analysis"].items():
                print(f"  {category}: {error_count} errors")

        print("\n" + "=" * 80)


def analyze_complexity_statistics(
    input_file: str,
    output_file: Optional[str] = None,
    show_detailed_stats: bool = True,
    create_plots: bool = False,
    plot_output_dir: Optional[str] = None,
) -> None:
    """
    Analyze utterance complexity classification statistics from a JSON file.

    Args:
        input_file: Path to the JSON file containing classification results
        output_file: Optional path to save detailed statistics as JSON
        show_detailed_stats: Whether to show detailed category-by-category statistics
        create_plots: Whether to create graphical visualizations
        plot_output_dir: Directory to save plot files (defaults to same dir as input file)

    Examples:
        # Basic analysis with summary
        analyze_complexity_statistics("results.json")

        # Analysis with JSON output
        analyze_complexity_statistics("results.json", "stats.json")

        # Analysis with plots
        analyze_complexity_statistics("results.json", create_plots=True)

        # Analysis without detailed stats but with plots
        analyze_complexity_statistics("results.json", show_detailed_stats=False, create_plots=True)
    """
    try:
        analyzer = ComplexityStatisticsAnalyzer()
        stats = analyzer.load_and_analyze(input_file)

        # Print summary
        analyzer.print_summary(stats)

        # Print detailed category statistics if requested
        if show_detailed_stats:
            print(f"\n📂 DETAILED CATEGORY STATISTICS:")
            print("-" * 80)

            analysis = stats["file_analysis"]
            for category_name, category_stats in analysis[
                "category_statistics"
            ].items():
                total = category_stats["total_utterances"]
                if total == 0:
                    continue

                chat_pct = safe_round((category_stats["chat_count"] / total) * 100, 1)
                cot_pct = safe_round((category_stats["cot_count"] / total) * 100, 1)

                print(f"\nCategory: {category_name}")
                print(f"  Total Utterances: {total}")
                print(f"  Chat Model: {category_stats['chat_count']} ({chat_pct}%)")
                print(f"  Reasoning Model: {category_stats['cot_count']} ({cot_pct}%)")
                if category_stats["error_count"] > 0:
                    error_pct = safe_round(
                        (category_stats["error_count"] / total) * 100, 1
                    )
                    print(f"  Errors: {category_stats['error_count']} ({error_pct}%)")
                if category_stats["unknown_count"] > 0:
                    unknown_pct = safe_round(
                        (category_stats["unknown_count"] / total) * 100, 1
                    )
                    print(
                        f"  Unknown: {category_stats['unknown_count']} ({unknown_pct}%)"
                    )

                if category_stats["avg_confidence"] > 0:
                    print(f"  Avg Confidence: {category_stats['avg_confidence']}")
                    print(
                        f"  Confidence Range: {category_stats['confidence_range'][0]} - {category_stats['confidence_range'][1]}"
                    )

        # Create graphical plots if requested
        if create_plots:
            if plot_output_dir is None:
                plot_output_dir = os.path.dirname(os.path.abspath(input_file))

            saved_plots = analyzer.create_distribution_plots(stats, plot_output_dir)
            if saved_plots:
                print(f"\n📊 GRAPHICAL VISUALIZATIONS:")
                print("-" * 80)
                for plot_file in saved_plots:
                    print(f"  Created: {os.path.basename(plot_file)}")
                print(f"\nPlots saved in: {plot_output_dir}")

        # Save detailed statistics to file if requested
        if output_file:
            write_json_file(stats, output_file, indent=2)
            logger.info(f"Detailed statistics saved to: {output_file}")

        logger.info("Analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error analyzing statistics: {e}")
        raise


def main():
    """Main entry point for the statistics analyzer."""
    fire.Fire(analyze_complexity_statistics)


if __name__ == "__main__":
    main()
