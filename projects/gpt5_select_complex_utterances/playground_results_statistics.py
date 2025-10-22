"""
Playground Results Statistics Analyzer

This program analyzes the results from playground v11 evaluator output
and provides comprehensive statistics grouped by:
- Segment name (e.g., "cwc - generate_response")
- Switching class (e.g., "work_requests_2_default")
- Joint groupings of both

Includes confidence metrics, classification distribution, and visualizations.

Author: Generated for BizChatScripts project
Date: August 6, 2025
"""

import json
import logging
import os
import sys
import fire
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter

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
    import seaborn as sns

    PLOTTING_AVAILABLE = True
    # Set style for better plots
    plt.style.use(
        "seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default"
    )
except ImportError:
    PLOTTING_AVAILABLE = False
    print(
        "Warning: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn"
    )


def safe_round(value: float, digits: int = 2) -> float:
    """Safely round a float value to specified decimal places."""
    try:
        return round(float(value), digits)
    except (ValueError, TypeError):
        return 0.0


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlaygroundResultsStatisticsAnalyzer:
    """
    Analyzer for playground v11 evaluator results statistics.
    """

    def __init__(self):
        self.segment_stats = {}
        self.switching_class_stats = {}
        self.joint_stats = {}
        self.total_utterances = 0
        self.total_segments = 0
        self.confidence_stats = defaultdict(list)
        self.complexity_indicators_stats = defaultdict(int)

    def load_and_analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Load playground results and perform comprehensive analysis.

        Args:
            file_path: Path to the JSON file containing playground results

        Returns:
            Dictionary containing all statistics
        """
        logger.info(f"Loading playground results from: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = read_json_file(file_path)
        logger.info(f"Loaded data for {len(data)} segments")

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected dict format with segment names as keys, got {type(data)}"
            )

        self.total_segments = len(data)

        # Analyze each segment
        for segment_name, utterances in data.items():
            self._analyze_segment(segment_name, utterances)

        # Generate comprehensive statistics
        stats = self._generate_statistics()
        return stats

    def _analyze_segment(self, segment_name: str, utterances: List[Dict]) -> None:
        """
        Analyze a single segment's utterances.

        Args:
            segment_name: Name of the segment (e.g., "cwc - generate_response")
            utterances: List of utterance evaluation results
        """
        if not utterances:
            logger.warning(f"Segment '{segment_name}' has no utterances")
            return

        segment_stats = {
            "total_utterances": len(utterances),
            "switching_class_distribution": defaultdict(int),
            "confidence_scores": [],
            "complexity_indicators": defaultdict(int),
        }

        # Process each utterance in the segment
        for utterance_data in utterances:
            if not isinstance(utterance_data, dict):
                logger.warning(
                    f"Invalid utterance data in segment '{segment_name}': {utterance_data}"
                )
                continue

            switching_class = utterance_data.get("switching_class", "unknown")
            confidence = utterance_data.get("confidence", 0.0)
            complexity_indicators = utterance_data.get("complexity_indicators", [])

            # Update segment-level stats
            segment_stats["switching_class_distribution"][switching_class] += 1

            if isinstance(confidence, (int, float)) and confidence > 0:
                segment_stats["confidence_scores"].append(confidence)
                self.confidence_stats[switching_class].append(confidence)

            # Count complexity indicators
            if isinstance(complexity_indicators, list):
                for indicator in complexity_indicators:
                    segment_stats["complexity_indicators"][indicator] += 1
                    self.complexity_indicators_stats[indicator] += 1

            # Update switching class global stats
            if switching_class not in self.switching_class_stats:
                self.switching_class_stats[switching_class] = {
                    "total_utterances": 0,
                    "segments": set(),
                    "confidence_scores": [],
                }

            self.switching_class_stats[switching_class]["total_utterances"] += 1
            self.switching_class_stats[switching_class]["segments"].add(segment_name)
            if isinstance(confidence, (int, float)) and confidence > 0:
                self.switching_class_stats[switching_class]["confidence_scores"].append(
                    confidence
                )

            # Update joint stats (segment + switching_class combination)
            joint_key = f"{segment_name}::{switching_class}"
            if joint_key not in self.joint_stats:
                self.joint_stats[joint_key] = {
                    "segment_name": segment_name,
                    "switching_class": switching_class,
                    "utterance_count": 0,
                    "confidence_scores": [],
                }

            self.joint_stats[joint_key]["utterance_count"] += 1
            if isinstance(confidence, (int, float)) and confidence > 0:
                self.joint_stats[joint_key]["confidence_scores"].append(confidence)

            self.total_utterances += 1

        # Calculate segment-level confidence statistics
        if segment_stats["confidence_scores"]:
            confidences = segment_stats["confidence_scores"]
            segment_stats["avg_confidence"] = safe_round(
                sum(confidences) / len(confidences), 3
            )
            segment_stats["confidence_range"] = (
                safe_round(min(confidences), 3),
                safe_round(max(confidences), 3),
            )
            segment_stats["confidence_std"] = safe_round(
                np.std(confidences) if PLOTTING_AVAILABLE else 0, 3
            )
        else:
            segment_stats["avg_confidence"] = 0.0
            segment_stats["confidence_range"] = (0.0, 0.0)
            segment_stats["confidence_std"] = 0.0

        # Convert defaultdict to regular dict
        segment_stats["switching_class_distribution"] = dict(
            segment_stats["switching_class_distribution"]
        )
        segment_stats["complexity_indicators"] = dict(
            segment_stats["complexity_indicators"]
        )

        # Remove raw scores (not needed in final stats)
        del segment_stats["confidence_scores"]

        self.segment_stats[segment_name] = segment_stats

    def _generate_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistics from analyzed data.
        """
        # Process switching class stats (convert sets to counts)
        for switching_class in self.switching_class_stats:
            stats = self.switching_class_stats[switching_class]
            stats["segment_count"] = len(stats["segments"])
            stats["segments"] = list(stats["segments"])  # Convert set to list

            # Calculate confidence statistics
            if stats["confidence_scores"]:
                confidences = stats["confidence_scores"]
                stats["avg_confidence"] = safe_round(
                    sum(confidences) / len(confidences), 3
                )
                stats["confidence_range"] = (
                    safe_round(min(confidences), 3),
                    safe_round(max(confidences), 3),
                )
                stats["confidence_std"] = safe_round(
                    np.std(confidences) if PLOTTING_AVAILABLE else 0, 3
                )
            else:
                stats["avg_confidence"] = 0.0
                stats["confidence_range"] = (0.0, 0.0)
                stats["confidence_std"] = 0.0

            del stats["confidence_scores"]  # Remove raw scores

        # Process joint stats confidence
        for joint_key in self.joint_stats:
            stats = self.joint_stats[joint_key]
            if stats["confidence_scores"]:
                confidences = stats["confidence_scores"]
                stats["avg_confidence"] = safe_round(
                    sum(confidences) / len(confidences), 3
                )
                stats["confidence_range"] = (
                    safe_round(min(confidences), 3),
                    safe_round(max(confidences), 3),
                )
            else:
                stats["avg_confidence"] = 0.0
                stats["confidence_range"] = (0.0, 0.0)

            del stats["confidence_scores"]  # Remove raw scores

        # Generate overall statistics
        switching_class_counts = Counter()
        for segment_stats in self.segment_stats.values():
            for switching_class, count in segment_stats[
                "switching_class_distribution"
            ].items():
                switching_class_counts[switching_class] += count

        # Create rankings
        segments_by_size = sorted(
            self.segment_stats.items(),
            key=lambda x: x[1]["total_utterances"],
            reverse=True,
        )

        switching_classes_by_size = sorted(
            switching_class_counts.items(), key=lambda x: x[1], reverse=True
        )

        joint_stats_by_size = sorted(
            self.joint_stats.items(),
            key=lambda x: x[1]["utterance_count"],
            reverse=True,
        )

        return {
            "file_analysis": {
                "overall_statistics": {
                    "total_segments": self.total_segments,
                    "total_utterances": self.total_utterances,
                    "total_switching_classes": len(self.switching_class_stats),
                    "total_joint_combinations": len(self.joint_stats),
                    "avg_utterances_per_segment": safe_round(
                        self.total_utterances / max(1, self.total_segments), 1
                    ),
                },
                "segment_statistics": self.segment_stats,
                "switching_class_statistics": self.switching_class_stats,
                "joint_statistics": self.joint_stats,
                "complexity_indicators_global": dict(self.complexity_indicators_stats),
                "rankings": {
                    "segments_by_utterance_count": [
                        (name, stats["total_utterances"])
                        for name, stats in segments_by_size[:20]
                    ],
                    "switching_classes_by_utterance_count": [
                        (switching_class, count)
                        for switching_class, count in switching_classes_by_size[:20]
                    ],
                    "joint_combinations_by_utterance_count": [
                        (
                            joint_key.split("::")[0],
                            joint_key.split("::")[1],
                            stats["utterance_count"],
                        )
                        for joint_key, stats in joint_stats_by_size[:30]
                    ],
                },
            }
        }

    def create_visualizations(
        self, stats: Dict[str, Any], output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Create comprehensive visualizations of the playground results.
        """
        if not PLOTTING_AVAILABLE:
            logger.warning(
                "Matplotlib/seaborn not available. Cannot create visualizations."
            )
            return []

        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))

        os.makedirs(output_dir, exist_ok=True)
        analysis = stats["file_analysis"]
        saved_plots = []

        # 1. Switching Class Distribution Pie Chart
        saved_plots.append(self._create_switching_class_pie(analysis, output_dir))

        # 2. Top Segments Bar Chart
        saved_plots.append(self._create_segments_bar_chart(analysis, output_dir))

        # 3. Segment vs Switching Class Heatmap
        saved_plots.append(self._create_segment_switching_heatmap(analysis, output_dir))

        # 4. Confidence Distribution by Switching Class
        saved_plots.append(self._create_confidence_distribution(analysis, output_dir))

        # 5. Top Complexity Indicators
        saved_plots.append(
            self._create_complexity_indicators_chart(analysis, output_dir)
        )

        logger.info(f"Created {len(saved_plots)} visualization plots in: {output_dir}")
        return saved_plots

    def _create_switching_class_pie(
        self, analysis: Dict[str, Any], output_dir: str
    ) -> str:
        """Create pie chart showing distribution of switching classes."""
        switching_class_stats = analysis["switching_class_statistics"]

        # Prepare data
        labels = []
        sizes = []
        colors = plt.cm.Set3(np.linspace(0, 1, min(len(switching_class_stats), 12)))

        for switching_class, stats in switching_class_stats.items():
            count = stats["total_utterances"]
            percentage = (
                count / analysis["overall_statistics"]["total_utterances"]
            ) * 100

            # Truncate long labels
            display_label = (
                switching_class[:40] + "..."
                if len(switching_class) > 40
                else switching_class
            )
            labels.append(f"{display_label}\n{count} ({percentage:.1f}%)")
            sizes.append(count)

        # Create pie chart
        plt.figure(figsize=(14, 10))
        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=labels,
            colors=colors[: len(sizes)],
            autopct="",
            startangle=90,
            textprops={"fontsize": 8},
        )

        plt.title(
            "Distribution of Switching Classes\n(Playground v11 Evaluator Results)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Add total count
        total_utterances = sum(sizes)
        plt.figtext(
            0.5,
            0.02,
            f"Total Utterances: {total_utterances:,} | Total Classes: {len(sizes)}",
            ha="center",
            fontsize=12,
            style="italic",
        )

        plt.axis("equal")
        plt.tight_layout()

        filename = os.path.join(output_dir, "switching_class_distribution.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return filename

    def _create_segments_bar_chart(
        self, analysis: Dict[str, Any], output_dir: str
    ) -> str:
        """Create horizontal bar chart of top segments by utterance count."""
        segment_stats = analysis["segment_statistics"]

        # Sort segments by utterance count
        sorted_segments = sorted(
            segment_stats.items(), key=lambda x: x[1]["total_utterances"], reverse=True
        )[
            :15
        ]  # Top 15

        # Prepare data
        segment_names = [name for name, _ in sorted_segments]
        utterance_counts = [stats["total_utterances"] for _, stats in sorted_segments]

        # Create bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.barh(
            range(len(segment_names)), utterance_counts, color="skyblue", alpha=0.8
        )

        # Customize
        plt.yticks(range(len(segment_names)), segment_names)
        plt.xlabel("Number of Utterances", fontsize=12)
        plt.title(
            "Top 15 Segments by Utterance Count", fontsize=14, fontweight="bold", pad=20
        )
        plt.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, utterance_counts)):
            plt.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{count}",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()

        filename = os.path.join(output_dir, "top_segments_by_count.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return filename

    def _create_segment_switching_heatmap(
        self, analysis: Dict[str, Any], output_dir: str
    ) -> str:
        """Create heatmap showing segment vs switching class combinations."""
        joint_stats = analysis["joint_statistics"]

        # Create matrix data
        segments = set()
        switching_classes = set()

        for joint_key, stats in joint_stats.items():
            segments.add(stats["segment_name"])
            switching_classes.add(stats["switching_class"])

        # Limit to top segments and classes for readability
        top_segments = sorted(
            segments,
            key=lambda s: sum(
                stats["utterance_count"]
                for joint_key, stats in joint_stats.items()
                if stats["segment_name"] == s
            ),
            reverse=True,
        )[:10]

        top_switching_classes = sorted(
            switching_classes,
            key=lambda sc: sum(
                stats["utterance_count"]
                for joint_key, stats in joint_stats.items()
                if stats["switching_class"] == sc
            ),
            reverse=True,
        )[:12]

        # Create matrix
        matrix = np.zeros((len(top_segments), len(top_switching_classes)))

        for i, segment in enumerate(top_segments):
            for j, switching_class in enumerate(top_switching_classes):
                joint_key = f"{segment}::{switching_class}"
                if joint_key in joint_stats:
                    matrix[i, j] = joint_stats[joint_key]["utterance_count"]

        # Create heatmap
        plt.figure(figsize=(16, 10))
        sns.heatmap(
            matrix,
            xticklabels=[
                sc[:30] + "..." if len(sc) > 30 else sc for sc in top_switching_classes
            ],
            yticklabels=top_segments,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            cbar_kws={"label": "Utterance Count"},
        )

        plt.title(
            "Segment vs Switching Class Heatmap\n(Top 10 Segments × Top 12 Classes)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Switching Class", fontsize=12)
        plt.ylabel("Segment Name", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        filename = os.path.join(output_dir, "segment_switching_class_heatmap.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return filename

    def _create_confidence_distribution(
        self, analysis: Dict[str, Any], output_dir: str
    ) -> str:
        """Create box plot showing confidence distribution by switching class."""
        switching_class_stats = analysis["switching_class_statistics"]

        # Collect confidence data
        confidence_data = []
        switching_classes = []

        for switching_class, stats in switching_class_stats.items():
            if stats["avg_confidence"] > 0:  # Only include classes with confidence data
                # Recreate confidence scores for box plot (approximate from stats)
                avg_conf = stats["avg_confidence"]
                conf_range = stats["confidence_range"]
                count = stats["total_utterances"]

                # Generate approximate confidence scores (for visualization)
                conf_scores = np.random.normal(
                    avg_conf, stats.get("confidence_std", 0.1), min(count, 100)
                )
                conf_scores = np.clip(conf_scores, conf_range[0], conf_range[1])

                confidence_data.extend(conf_scores)
                switching_classes.extend([switching_class] * len(conf_scores))

        if not confidence_data:
            logger.warning("No confidence data available for visualization")
            return ""

        # Create box plot
        plt.figure(figsize=(14, 8))

        # Group data for seaborn
        import pandas as pd

        df = pd.DataFrame(
            {"Switching Class": switching_classes, "Confidence": confidence_data}
        )

        sns.boxplot(data=df, x="Confidence", y="Switching Class", orient="h")
        plt.title(
            "Confidence Score Distribution by Switching Class",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Confidence Score", fontsize=12)
        plt.ylabel("Switching Class", fontsize=12)
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        filename = os.path.join(output_dir, "confidence_distribution_by_class.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return filename

    def _create_complexity_indicators_chart(
        self, analysis: Dict[str, Any], output_dir: str
    ) -> str:
        """Create bar chart of top complexity indicators."""
        complexity_indicators = analysis["complexity_indicators_global"]

        if not complexity_indicators:
            logger.warning("No complexity indicators data available")
            return ""

        # Sort by count and take top 15
        sorted_indicators = sorted(
            complexity_indicators.items(), key=lambda x: x[1], reverse=True
        )[:15]

        indicators = [indicator for indicator, _ in sorted_indicators]
        counts = [count for _, count in sorted_indicators]

        # Create bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(indicators)), counts, color="lightcoral", alpha=0.8)

        plt.yticks(range(len(indicators)), indicators)
        plt.xlabel("Frequency", fontsize=12)
        plt.title(
            "Top 15 Complexity Indicators", fontsize=14, fontweight="bold", pad=20
        )
        plt.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{count}",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()

        filename = os.path.join(output_dir, "top_complexity_indicators.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return filename

    def print_summary(self, stats: Dict[str, Any]) -> None:
        """Print a formatted summary of the statistics."""
        analysis = stats["file_analysis"]
        overall = analysis["overall_statistics"]

        print("\n" + "=" * 80)
        print("PLAYGROUND v11 EVALUATOR RESULTS STATISTICS")
        print("=" * 80)

        # Overall summary
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"  Total Segments: {overall['total_segments']}")
        print(f"  Total Utterances: {overall['total_utterances']}")
        print(f"  Total Switching Classes: {overall['total_switching_classes']}")
        print(f"  Total Joint Combinations: {overall['total_joint_combinations']}")
        print(f"  Avg Utterances per Segment: {overall['avg_utterances_per_segment']}")

        # Top segments
        rankings = analysis["rankings"]
        print(f"\n📋 TOP 10 SEGMENTS BY UTTERANCE COUNT:")
        for i, (segment, count) in enumerate(
            rankings["segments_by_utterance_count"][:10], 1
        ):
            print(f"  {i:2d}. {segment}: {count:,} utterances")

        # Top switching classes
        print(f"\n🎯 TOP 10 SWITCHING CLASSES BY UTTERANCE COUNT:")
        for i, (switching_class, count) in enumerate(
            rankings["switching_classes_by_utterance_count"][:10], 1
        ):
            display_class = (
                switching_class[:50] + "..."
                if len(switching_class) > 50
                else switching_class
            )
            print(f"  {i:2d}. {display_class}: {count:,} utterances")

        # Top joint combinations
        print(f"\n🔗 TOP 10 JOINT COMBINATIONS:")
        for i, (segment, switching_class, count) in enumerate(
            rankings["joint_combinations_by_utterance_count"][:10], 1
        ):
            display_class = (
                switching_class[:40] + "..."
                if len(switching_class) > 40
                else switching_class
            )
            print(f"  {i:2d}. {segment} → {display_class}: {count:,} utterances")

        # Top complexity indicators
        complexity_indicators = analysis["complexity_indicators_global"]
        if complexity_indicators:
            print(f"\n🧠 TOP 10 COMPLEXITY INDICATORS:")
            sorted_indicators = sorted(
                complexity_indicators.items(), key=lambda x: x[1], reverse=True
            )[:10]
            for i, (indicator, count) in enumerate(sorted_indicators, 1):
                print(f"  {i:2d}. {indicator}: {count:,} occurrences")

        print("\n" + "=" * 80)

    def generate_markdown_report(self, stats: Dict[str, Any], output_file: str) -> str:
        """
        Generate a comprehensive markdown report of the playground results statistics.

        Args:
            stats: Statistics dictionary from analyze method
            output_file: Path to save the markdown report

        Returns:
            Path to the saved markdown file
        """
        analysis = stats["file_analysis"]
        overall = analysis["overall_statistics"]

        # Generate markdown content
        markdown_content = []

        # Title and overview
        markdown_content.extend(
            [
                "# Playground v11 Evaluator Results Statistics Report",
                "",
                f"**Generated on:** {os.path.basename(output_file).replace('.md', '')}",
                f"**Analysis Date:** August 6, 2025",
                "",
                "## 📊 Executive Summary",
                "",
                f"- **Total Segments:** {overall['total_segments']:,}",
                f"- **Total Utterances:** {overall['total_utterances']:,}",
                f"- **Total Switching Classes:** {overall['total_switching_classes']:,}",
                f"- **Total Joint Combinations:** {overall['total_joint_combinations']:,}",
                f"- **Average Utterances per Segment:** {overall['avg_utterances_per_segment']:.1f}",
                "",
            ]
        )

        # Top segments
        rankings = analysis["rankings"]
        markdown_content.extend(
            [
                "## 📋 Top Segments by Utterance Count",
                "",
                "| Rank | Segment Name | Utterance Count |",
                "|------|-------------|----------------|",
            ]
        )

        for i, (segment, count) in enumerate(
            rankings["segments_by_utterance_count"][:15], 1
        ):
            markdown_content.append(f"| {i:2d} | {segment} | {count:,} |")

        markdown_content.extend(["", ""])

        # Top switching classes
        markdown_content.extend(
            [
                "## 🎯 Top Switching Classes by Utterance Count",
                "",
                "| Rank | Switching Class | Utterance Count |",
                "|------|----------------|----------------|",
            ]
        )

        for i, (switching_class, count) in enumerate(
            rankings["switching_classes_by_utterance_count"][:15], 1
        ):
            display_class = (
                switching_class[:60] + "..."
                if len(switching_class) > 60
                else switching_class
            )
            markdown_content.append(f"| {i:2d} | {display_class} | {count:,} |")

        markdown_content.extend(["", ""])

        # Top joint combinations
        markdown_content.extend(
            [
                "## 🔗 Top Joint Combinations (Segment + Switching Class)",
                "",
                "| Rank | Segment | Switching Class | Count |",
                "|------|---------|----------------|-------|",
            ]
        )

        for i, (segment, switching_class, count) in enumerate(
            rankings["joint_combinations_by_utterance_count"][:15], 1
        ):
            display_segment = segment[:40] + "..." if len(segment) > 40 else segment
            display_class = (
                switching_class[:40] + "..."
                if len(switching_class) > 40
                else switching_class
            )
            markdown_content.append(
                f"| {i:2d} | {display_segment} | {display_class} | {count:,} |"
            )

        markdown_content.extend(["", ""])

        # Top complexity indicators
        complexity_indicators = analysis["complexity_indicators_global"]
        if complexity_indicators:
            markdown_content.extend(
                [
                    "## 🧠 Top Complexity Indicators",
                    "",
                    "| Rank | Complexity Indicator | Frequency |",
                    "|------|---------------------|-----------|",
                ]
            )

            sorted_indicators = sorted(
                complexity_indicators.items(), key=lambda x: x[1], reverse=True
            )[:15]
            for i, (indicator, count) in enumerate(sorted_indicators, 1):
                display_indicator = (
                    indicator[:80] + "..." if len(indicator) > 80 else indicator
                )
                markdown_content.append(f"| {i:2d} | {display_indicator} | {count:,} |")

            markdown_content.extend(["", ""])

        # Detailed segment analysis (top 20 segments only for readability)
        segment_stats = analysis["segment_statistics"]
        markdown_content.extend(
            [
                "## 📂 Detailed Segment Analysis (Top 20 Segments)",
                "",
            ]
        )

        # Sort segments by utterance count for detailed analysis
        sorted_segments = sorted(
            segment_stats.items(), key=lambda x: x[1]["total_utterances"], reverse=True
        )

        for segment_name, segment_data in sorted_segments[:20]:
            markdown_content.extend(
                [
                    f"### {segment_name}",
                    "",
                    f"**Total Utterances:** {segment_data['total_utterances']:,}",
                    f"**Average Confidence:** {segment_data['avg_confidence']:.3f}",
                    f"**Confidence Range:** {segment_data['confidence_range'][0]:.3f} - {segment_data['confidence_range'][1]:.3f}",
                    "",
                    "**Top Switching Classes:**",
                    "",
                ]
            )

            # Show top switching classes for this segment
            sorted_classes = sorted(
                segment_data["switching_class_distribution"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            for j, (switching_class, count) in enumerate(sorted_classes, 1):
                percentage = (count / segment_data["total_utterances"]) * 100
                markdown_content.append(
                    f"{j:2d}. **{switching_class}**: {count:,} utterances ({percentage:.1f}%)"
                )

            # Show top complexity indicators for this segment if available
            if segment_data.get("complexity_indicators"):
                markdown_content.extend(
                    [
                        "",
                        "**Top Complexity Indicators:**",
                        "",
                    ]
                )

                sorted_complexity = sorted(
                    segment_data["complexity_indicators"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]

                for indicator, count in sorted_complexity:
                    markdown_content.append(f"- **{indicator}**: {count:,} occurrences")

            markdown_content.extend(["", "---", ""])

        # Switching class detailed analysis (top 15 classes)
        switching_class_stats = analysis["switching_class_statistics"]
        markdown_content.extend(
            [
                "## 🎯 Detailed Switching Class Analysis (Top 15 Classes)",
                "",
            ]
        )

        # Sort switching classes by utterance count
        sorted_switching_classes = sorted(
            switching_class_stats.items(),
            key=lambda x: x[1]["total_utterances"],
            reverse=True,
        )

        for switching_class, class_data in sorted_switching_classes[:15]:
            markdown_content.extend(
                [
                    f"### {switching_class}",
                    "",
                    f"**Total Utterances:** {class_data['total_utterances']:,}",
                    f"**Segments Involved:** {class_data['segment_count']:,}",
                    f"**Average Confidence:** {class_data['avg_confidence']:.3f}",
                    f"**Confidence Range:** {class_data['confidence_range'][0]:.3f} - {class_data['confidence_range'][1]:.3f}",
                    "",
                    "**Top Segments:**",
                    "",
                ]
            )

            # Show top segments for this switching class (limit to top 8)
            segments_list = class_data["segments"][:8]
            for segment in segments_list:
                # Get count for this specific combination
                joint_key = f"{segment}::{switching_class}"
                joint_stats = analysis["joint_statistics"].get(joint_key, {})
                count = joint_stats.get("utterance_count", 0)
                markdown_content.append(f"- **{segment}**: {count:,} utterances")

            if len(class_data["segments"]) > 8:
                markdown_content.append(
                    f"- *...and {len(class_data['segments']) - 8} more segments*"
                )

            markdown_content.extend(["", "---", ""])

        # Summary statistics
        markdown_content.extend(
            [
                "## 📈 Statistical Summary",
                "",
                "### Distribution Overview",
                "",
            ]
        )

        # Calculate some additional insights
        total_utterances = overall["total_utterances"]
        top_5_segments = sum(
            count for _, count in rankings["segments_by_utterance_count"][:5]
        )
        top_5_classes = sum(
            count for _, count in rankings["switching_classes_by_utterance_count"][:5]
        )

        markdown_content.extend(
            [
                f"- **Top 5 segments** account for {top_5_segments:,} utterances ({(top_5_segments/total_utterances*100):.1f}% of total)",
                f"- **Top 5 switching classes** account for {top_5_classes:,} utterances ({(top_5_classes/total_utterances*100):.1f}% of total)",
                f"- **Average segment size**: {total_utterances/overall['total_segments']:.1f} utterances per segment",
                f"- **Joint combinations density**: {overall['total_joint_combinations']:.0f} combinations across {overall['total_segments']} segments and {overall['total_switching_classes']} classes",
                "",
                "### Confidence Analysis",
                "",
            ]
        )

        # Calculate overall confidence stats
        all_confidences = []
        for segment_data in segment_stats.values():
            if segment_data["avg_confidence"] > 0:
                all_confidences.append(segment_data["avg_confidence"])

        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            min_confidence = min(all_confidences)
            max_confidence = max(all_confidences)

            markdown_content.extend(
                [
                    f"- **Overall average confidence**: {avg_confidence:.3f}",
                    f"- **Confidence range across segments**: {min_confidence:.3f} - {max_confidence:.3f}",
                    f"- **Segments with confidence data**: {len(all_confidences):,} out of {overall['total_segments']:,}",
                    "",
                ]
            )

        # Report generation info
        markdown_content.extend(
            [
                "---",
                "",
                "## 📝 Report Generation Info",
                "",
                f"- **Generated by**: Playground Results Statistics Analyzer",
                f"- **Generated on**: August 6, 2025",
                f"- **Source data**: Playground v11 evaluator results",
                f"- **Analysis type**: Comprehensive statistical analysis with segment and switching class groupings",
                "",
                "*This report was automatically generated from playground evaluation results to provide insights into utterance classification patterns, confidence metrics, and complexity indicators.*",
            ]
        )

        # Write markdown file
        markdown_text = "\n".join(markdown_content)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        return output_file


def analyze_playground_results(
    input_file: str,
    output_prefix: Optional[str] = None,
    show_detailed_stats: bool = False,
    create_plots: bool = False,
    plot_output_dir: Optional[str] = None,
) -> None:
    """
    Analyze playground v11 evaluator results statistics.

    Args:
        input_file: Path to the JSON file containing playground results
        output_prefix: Optional path prefix for saving reports (generates both .json and .md files)
        show_detailed_stats: Whether to show detailed segment and class statistics
        create_plots: Whether to create graphical visualizations
        plot_output_dir: Directory to save plot files (defaults to same dir as input file)

    Examples:
        # Basic analysis with summary
        analyze_playground_results("parsed_results.json")

        # Analysis with both JSON and markdown output
        analyze_playground_results("parsed_results.json", "results_report")

        # Analysis with plots
        analyze_playground_results("parsed_results.json", "results_report", create_plots=True)
    """
    try:
        analyzer = PlaygroundResultsStatisticsAnalyzer()
        stats = analyzer.load_and_analyze(input_file)

        # Print summary
        analyzer.print_summary(stats)

        # Print detailed statistics if requested
        if show_detailed_stats:
            print(f"\n📂 DETAILED SEGMENT STATISTICS:")
            print("-" * 80)

            analysis = stats["file_analysis"]
            for segment_name, segment_stats in analysis["segment_statistics"].items():
                print(f"\n🏷️  Segment: {segment_name}")
                print(f"   Total Utterances: {segment_stats['total_utterances']}")
                print(f"   Avg Confidence: {segment_stats['avg_confidence']}")
                print(f"   Confidence Range: {segment_stats['confidence_range']}")
                print(f"   Top Switching Classes:")

                # Show top 5 switching classes for this segment
                sorted_classes = sorted(
                    segment_stats["switching_class_distribution"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]

                for j, (switching_class, count) in enumerate(sorted_classes, 1):
                    display_class = (
                        switching_class[:50] + "..."
                        if len(switching_class) > 50
                        else switching_class
                    )
                    percentage = (count / segment_stats["total_utterances"]) * 100
                    print(f"     {j}. {display_class}: {count} ({percentage:.1f}%)")

        # Create graphical plots if requested
        if create_plots:
            if plot_output_dir is None:
                plot_output_dir = os.path.dirname(os.path.abspath(input_file))

            saved_plots = analyzer.create_visualizations(stats, plot_output_dir)
            if saved_plots:
                print(f"\n📈 VISUALIZATIONS CREATED:")
                for plot_file in saved_plots:
                    print(f"   📊 {os.path.basename(plot_file)}")

        # Generate reports if output prefix is provided
        if output_prefix:
            # Generate both JSON and markdown filenames from prefix
            json_output_file = f"{output_prefix}.json"
            markdown_output_file = f"{output_prefix}.md"

            # Generate default prefix if not provided
            if output_prefix is None:
                input_path = Path(input_file)
                output_prefix = str(input_path.parent / f"{input_path.stem}_report")
                json_output_file = f"{output_prefix}.json"
                markdown_output_file = f"{output_prefix}.md"

            # Save JSON statistics
            write_json_file(stats, json_output_file, indent=2)

            # Generate markdown report
            saved_report = analyzer.generate_markdown_report(
                stats, markdown_output_file
            )

            # Consolidated output message
            print(f"\n📄 REPORTS GENERATED:")
            print(f"   📊 {json_output_file} (JSON statistics)")
            print(f"   📝 {markdown_output_file} (Human-readable report)")

        print(f"\n✅ Analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error analyzing playground results: {e}")
        raise


def main():
    """Main entry point for the playground results analyzer."""
    fire.Fire(analyze_playground_results)


if __name__ == "__main__":
    main()
