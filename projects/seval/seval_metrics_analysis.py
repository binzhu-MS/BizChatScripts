#!/usr/bin/env python3
"""
SEVAL Metrics Analysis Tool

A comprehensive statistical analysis tool for comparing A/B test metrics between control and treatment groups.
This script reads SEVAL metrics CSV files and performs detailed statistical comparisons, paired analysis,
win/loss analysis, and generates comprehensive reports.

Features:
- Statistical significance testing (paired/independent t-tests, p-values)
- Missing data pattern analysis
- Win/loss analysis (query-level comparison of treatment vs control)
- Score difference distribution analysis with percentiles and histograms
- Segment-level breakdowns for all metrics
- Markdown report generation with detailed statistics
- Multiple analysis modes: comprehensive analysis, paired analysis, and metric listing

Main Functions:
1. comprehensive_analysis: Full A/B testing analysis with statistical tests and reports
2. paired_analysis: Detailed paired comparison with win/loss and distribution analysis
3. list_metrics: Display all available metrics in the dataset

Usage Examples:
    # List all available metrics
    python seval_metrics_analysis.py list_metrics "path/to/job_folder"

    # Comprehensive analysis of all metrics
    python seval_metrics_analysis.py comprehensive_analysis "path/to/job_folder" --output_file="report.md"

    # Analyze specific metrics only
    python seval_metrics_analysis.py comprehensive_analysis "path/to/job_folder" --metrics="citedcg_one_centric,another_metric"

    # Segment-level analysis
    python seval_metrics_analysis.py comprehensive_analysis "path/to/job_folder" --segment_column="segment 2"

    # Detailed paired analysis for one metric
    python seval_metrics_analysis.py paired_analysis "path/to/job_folder" --metric="citedcg_one_centric" --output_file="paired_report.md"

Input Requirements:
- Job folder containing 'offline_scorecard_generator_output/all_metrics_paired.csv'
- CSV file with columns formatted as '{metric_name}_control' and '{metric_name}_treatment'
- Optional segment column for breakdown analysis
"""

import pandas as pd
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import fire
from fire.core import FireExit
import json

# Add parent directory to path to import utils
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent  # Go up to BizChatScripts root
sys.path.insert(0, str(parent_dir))

# Import from the utils package
from utils.statistics_utils import tdiff

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsComparison:
    """A class for comparing control vs experiment metrics."""

    def __init__(
        self,
        job_path: str,
        metrics_file_name: str = "all_metrics_paired.csv",
        metrics_relative_path: str = "offline_scorecard_generator_output",
    ):
        """Initialize the metrics comparison tool."""
        self.job_path = Path(job_path)
        self.metrics_file_name = metrics_file_name
        self.metrics_relative_path = metrics_relative_path
        self.df = None
        self._load_data()

    def _load_data(self) -> None:
        """Load the metrics CSV file."""
        metrics_file = (
            self.job_path / self.metrics_relative_path / self.metrics_file_name
        )

        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

        try:
            self.df = pd.read_csv(metrics_file)
            # Print absolute path
            logger.info(
                f"Loaded metrics data of size: {self.df.shape} from {metrics_file.absolute()}"
            )
        except Exception as e:
            raise Exception(f"Error reading metrics file: {e}")

    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics that have both control and treatment columns."""
        if self.df is None:
            return []

        metrics = []
        columns = list(self.df.columns)

        # Find metrics with both control and treatment variants
        for col in columns:
            if col.endswith("_control"):
                base_metric = col.replace("_control", "")
                treatment_col = base_metric + "_treatment"
                if treatment_col in columns:
                    metrics.append(base_metric)

        return sorted(metrics)

    def compare_single_metric(
        self,
        metric_base_name: str,
        segment_column: Optional[str] = None,
        paired_test: bool = True,
    ) -> Dict[str, Any]:
        """Compare a single metric between control and treatment."""
        # Ensure data is loaded
        if self.df is None:
            self._load_data()

        # Type assertion for Pylance
        assert self.df is not None, "DataFrame should be loaded after _load_data()"

        control_col = f"{metric_base_name}_control"
        treatment_col = f"{metric_base_name}_treatment"

        if control_col not in self.df.columns or treatment_col not in self.df.columns:
            raise ValueError(
                f"Metric columns not found: {control_col}, {treatment_col}"
            )

        # Overall comparison
        control_data = pd.to_numeric(self.df[control_col], errors="coerce").dropna()
        treatment_data = pd.to_numeric(self.df[treatment_col], errors="coerce").dropna()

        overall_stats = self._calculate_comparison_stats(
            control_data, treatment_data, paired_test
        )

        # Overall paired analysis
        overall_paired = self._calculate_paired_analysis(metric_base_name)

        result = {
            "metric_name": metric_base_name,
            "overall": overall_stats,
            "overall_paired_analysis": overall_paired,
        }

        # Segment-level comparison if requested
        if segment_column and segment_column in self.df.columns:
            segments = {}
            segment_paired = {}
            unique_segments = self.df[segment_column].dropna().unique()

            for segment in unique_segments:
                segment_df = self.df[self.df[segment_column] == segment]
                seg_control = pd.to_numeric(
                    segment_df[control_col], errors="coerce"
                ).dropna()
                seg_treatment = pd.to_numeric(
                    segment_df[treatment_col], errors="coerce"
                ).dropna()

                if len(seg_control) > 0 and len(seg_treatment) > 0:
                    segments[str(segment)] = self._calculate_comparison_stats(
                        seg_control, seg_treatment, paired_test
                    )
                    segment_paired[str(segment)] = self._calculate_paired_analysis(
                        metric_base_name, segment_df
                    )

            result["segments"] = segments
            result["segment_paired_analysis"] = segment_paired

        return result

    def _calculate_comparison_stats(
        self,
        control_data: pd.Series,
        treatment_data: pd.Series,
        paired_test: bool = True,
    ) -> Dict[str, Any]:
        """Calculate statistical comparison between control and treatment."""
        if len(control_data) == 0 or len(treatment_data) == 0:
            return {
                "control_mean": 0,
                "treatment_mean": 0,
                "difference": 0,
                "percent_change": 0,
                "p_value": 1.0,
                "significant": False,
                "control_count": len(control_data),
                "treatment_count": len(treatment_data),
            }

        control_mean = control_data.mean()
        treatment_mean = treatment_data.mean()
        difference = treatment_mean - control_mean

        # Calculate percent change
        percent_change = 0
        if control_mean != 0:
            percent_change = (difference / control_mean) * 100

        # Statistical test
        try:
            if paired_test and len(control_data) == len(treatment_data):
                # Paired t-test
                test_result = tdiff(
                    control_data.values, treatment_data.values, paired=True
                )
            else:
                # Independent t-test
                test_result = tdiff(
                    control_data.values, treatment_data.values, paired=False
                )

            logger.debug(f"Statistical test result: {test_result}")
            p_value = test_result.get("pval", 1.0)
            significant = p_value < 0.05

        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            p_value = 1.0
            significant = False

        return {
            "control_mean": float(control_mean),
            "treatment_mean": float(treatment_mean),
            "difference": float(difference),
            "percent_change": float(percent_change),
            "p_value": float(p_value),
            "significant": significant,
            "control_count": len(control_data),
            "treatment_count": len(treatment_data),
        }

    def _calculate_paired_analysis(
        self, metric_base_name: str, segment_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Calculate detailed paired analysis for a metric."""
        import numpy as np

        # Use provided segment or full dataframe
        if segment_df is not None:
            df = segment_df
        else:
            df = self.df

        if df is None:
            return {"error": "No data available"}

        control_col = f"{metric_base_name}_control"
        treatment_col = f"{metric_base_name}_treatment"

        # Convert to numeric, keeping NaN for missing values
        control_vals = pd.to_numeric(df[control_col], errors="coerce")
        treatment_vals = pd.to_numeric(df[treatment_col], errors="coerce")

        total_queries = len(df)

        # Missing data analysis
        both_present = (~control_vals.isna() & ~treatment_vals.isna()).sum()
        only_control = (~control_vals.isna() & treatment_vals.isna()).sum()
        only_treatment = (control_vals.isna() & ~treatment_vals.isna()).sum()
        both_missing = (control_vals.isna() & treatment_vals.isna()).sum()

        missing_analysis = {
            "total_queries": total_queries,
            "both_present": int(both_present),
            "only_control": int(only_control),
            "only_treatment": int(only_treatment),
            "both_missing": int(both_missing),
            "both_present_pct": float(both_present / total_queries * 100),
            "only_control_pct": float(only_control / total_queries * 100),
            "only_treatment_pct": float(only_treatment / total_queries * 100),
            "both_missing_pct": float(both_missing / total_queries * 100),
        }

        # Difference analysis for valid pairs only
        valid_pairs_mask = ~control_vals.isna() & ~treatment_vals.isna()
        if valid_pairs_mask.sum() == 0:
            return {
                "missing_analysis": missing_analysis,
                "paired_analysis": {"error": "No valid pairs found"},
            }

        valid_control = control_vals[valid_pairs_mask]
        valid_treatment = treatment_vals[valid_pairs_mask]
        differences = valid_treatment - valid_control

        # Win/Loss analysis
        treatment_wins = (differences > 0).sum()
        ties = (differences == 0).sum()
        control_wins = (differences < 0).sum()

        # Histogram analysis - create bins for difference distribution
        n_bins = min(20, max(5, int(len(differences) / 50)))  # Adaptive bin count
        hist_counts, bin_edges = np.histogram(differences, bins=n_bins)

        # Create histogram data structure
        histogram_data = {"bins": [], "total_queries": len(differences)}

        for i in range(len(hist_counts)):
            bin_start = float(bin_edges[i])
            bin_end = float(bin_edges[i + 1])
            count = int(hist_counts[i])
            percentage = float(count / len(differences) * 100)

            histogram_data["bins"].append(
                {
                    "range": f"[{bin_start:.1f}, {bin_end:.1f})",
                    "count": count,
                    "percentage": percentage,
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                }
            )

        # Distribution statistics
        paired_analysis = {
            "valid_pairs": int(valid_pairs_mask.sum()),
            "mean_difference": float(differences.mean()),
            "median_difference": float(differences.median()),
            "std_difference": float(differences.std()),
            "min_difference": float(differences.min()),
            "max_difference": float(differences.max()),
            "win_loss": {
                "treatment_wins": int(treatment_wins),
                "ties": int(ties),
                "control_wins": int(control_wins),
                "treatment_win_rate": float(treatment_wins / len(differences) * 100),
            },
            "percentiles": {
                f"p{p}": float(np.percentile(differences, p))
                for p in [5, 10, 25, 50, 75, 90, 95]
            },
            "histogram": histogram_data,
            "control_stats": {
                "mean": float(valid_control.mean()),
                "std": float(valid_control.std()),
                "min": float(valid_control.min()),
                "max": float(valid_control.max()),
            },
            "treatment_stats": {
                "mean": float(valid_treatment.mean()),
                "std": float(valid_treatment.std()),
                "min": float(valid_treatment.min()),
                "max": float(valid_treatment.max()),
            },
        }

        # Add top and bottom 10 queries analysis
        valid_df = df[valid_pairs_mask].copy()
        valid_df["score_difference"] = differences
        valid_df["control_score"] = valid_control
        valid_df["treatment_score"] = valid_treatment

        # Sort by score difference for top/bottom analysis
        sorted_by_diff = valid_df.sort_values("score_difference", ascending=False)

        # Get top 10 (highest positive differences - treatment much better)
        top_10 = sorted_by_diff.head(10)
        # Get bottom 10 (lowest negative differences - control much better)
        bottom_10 = sorted_by_diff.tail(10)

        # Function to extract query info
        def extract_query_info(query_df):
            queries = []
            for idx, row in query_df.iterrows():
                query_info = {
                    "index": int(idx),
                    "control_score": float(row["control_score"]),
                    "treatment_score": float(row["treatment_score"]),
                    "score_difference": float(row["score_difference"]),
                }
                # Add query text - look for common query column names
                query_text = None
                for col_name in [
                    "query",
                    "Query",
                    "question",
                    "Question",
                    "utterance",
                    "Utterance",
                ]:
                    if col_name in row and pd.notna(row[col_name]):
                        query_text = str(row[col_name]).strip()
                        break

                if query_text:
                    # Truncate if too long (keep first 80 characters)
                    if len(query_text) > 80:
                        query_text = query_text[:77] + "..."
                    query_info["query_text"] = query_text
                else:
                    # Fallback to index/ID if no query text found
                    query_info["query_text"] = f"Index: {idx}"

                # Also keep ID info for reference
                if "query_id" in row:
                    query_info["query_id"] = str(row["query_id"])
                if "id" in row:
                    query_info["id"] = str(row["id"])
                queries.append(query_info)
            return queries

        paired_analysis["top_10_queries"] = {
            "description": "Top 10 queries where treatment performed much better than control",
            "queries": extract_query_info(top_10),
        }

        paired_analysis["bottom_10_queries"] = {
            "description": "Bottom 10 queries where control performed much better than treatment",
            "queries": extract_query_info(bottom_10),
        }

        return {
            "missing_analysis": missing_analysis,
            "paired_analysis": paired_analysis,
        }

    def compare_multiple_metrics(
        self,
        metric_names: Optional[List[str]] = None,
        segment_column: Optional[str] = None,
        paired_test: bool = True,
    ) -> Dict[str, Any]:
        """Compare multiple metrics between control and treatment."""
        if metric_names is None:
            metric_names = self.get_available_metrics()

        results = {}
        for metric in metric_names:
            try:
                results[metric] = self.compare_single_metric(
                    metric, segment_column, paired_test
                )
            except Exception as e:
                logger.error(f"Error comparing metric {metric}: {e}")
                continue

        return results

    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """Generate a markdown report from comparison results."""
        lines = []
        lines.append("# SEVAL Metrics Comparison Report")
        lines.append("")
        lines.append(f"**Job Path:** {self.job_path.name}")
        lines.append(f"**Total Metrics:** {len(results)}")
        lines.append("")

        # Overall summary table
        lines.append("## Overall Summary")
        lines.append("")
        lines.append(
            "| Metric | Control Mean | Treatment Mean | Change | Change % | P-Value | Significant |"
        )
        lines.append(
            "|--------|--------------|----------------|--------|----------|---------|-------------|"
        )

        for metric_name, metric_data in results.items():
            overall = metric_data["overall"]
            control_mean = overall["control_mean"]
            treatment_mean = overall["treatment_mean"]
            change = overall["difference"]
            change_pct = overall["percent_change"]
            p_value = overall["p_value"]
            significant = "✓" if overall["significant"] else "✗"

            lines.append(
                f"| {metric_name} | {control_mean:.4f} | {treatment_mean:.4f} | "
                f"{change:+.4f} | {change_pct:+.2f}% | {p_value:.4f} | {significant} |"
            )

        lines.append("")

        # Detailed results for each metric
        lines.append("## Detailed Results")
        lines.append("")

        for metric_name, metric_data in results.items():
            lines.append(f"### {metric_name}")
            lines.append("")

            overall = metric_data["overall"]
            lines.append(f"**Overall Comparison:**")
            lines.append(
                f"- Control: {overall['control_mean']:.4f} (n={overall['control_count']})"
            )
            lines.append(
                f"- Treatment: {overall['treatment_mean']:.4f} (n={overall['treatment_count']})"
            )
            lines.append(f"- Difference: {overall['difference']:+.4f}")
            lines.append(f"- Change: {overall['percent_change']:+.2f}%")
            lines.append(f"- P-value: {overall['p_value']:.4f}")
            lines.append(f"- Significant: {'Yes' if overall['significant'] else 'No'}")
            lines.append("")

            # Add paired analysis results if available
            if "overall_paired_analysis" in metric_data:
                paired_data = metric_data["overall_paired_analysis"]

                # Missing data analysis
                if "missing_analysis" in paired_data:
                    missing = paired_data["missing_analysis"]
                    lines.append("**Missing Data Analysis:**")
                    lines.append(f"- Total queries: {missing['total_queries']}")
                    lines.append(
                        f"- Valid pairs: {missing['both_present']} ({missing['both_present_pct']:.1f}%)"
                    )
                    lines.append(
                        f"- Control only: {missing['only_control']} ({missing['only_control_pct']:.1f}%)"
                    )
                    lines.append(
                        f"- Treatment only: {missing['only_treatment']} ({missing['only_treatment_pct']:.1f}%)"
                    )
                    lines.append(
                        f"- Both missing: {missing['both_missing']} ({missing['both_missing_pct']:.1f}%)"
                    )
                    lines.append("")

                # Win/Loss analysis
                if (
                    "paired_analysis" in paired_data
                    and "error" not in paired_data["paired_analysis"]
                ):
                    paired = paired_data["paired_analysis"]
                    lines.append("**Win/Loss Analysis (Valid Pairs Only):**")
                    lines.append(
                        f"- Treatment wins: {paired['win_loss']['treatment_wins']}"
                    )
                    lines.append(f"- Ties: {paired['win_loss']['ties']}")
                    lines.append(
                        f"- Control wins: {paired['win_loss']['control_wins']}"
                    )
                    lines.append(
                        f"- Treatment win rate: {paired['win_loss']['treatment_win_rate']:.1f}%"
                    )
                    lines.append("")

                    lines.append("**Score Difference Distribution:**")
                    lines.append(f"- Mean difference: {paired['mean_difference']:.4f}")
                    lines.append(
                        f"- Median difference: {paired['median_difference']:.4f}"
                    )
                    lines.append(f"- Std deviation: {paired['std_difference']:.4f}")
                    lines.append(f"- Min difference: {paired['min_difference']:.4f}")
                    lines.append(f"- Max difference: {paired['max_difference']:.4f}")
                    lines.append("")

                    lines.append("**Percentiles:**")
                    for p in [5, 25, 50, 75, 95]:
                        lines.append(f"- P{p}: {paired['percentiles'][f'p{p}']:.2f}")
                    lines.append("")

            # Segment results if available
            if "segments" in metric_data:
                lines.append("**By Segment:**")
                lines.append("")

                for segment_name, segment_data in metric_data["segments"].items():
                    lines.append(f"- **{segment_name}:**")
                    lines.append(
                        f"  - Control: {segment_data['control_mean']:.4f} (n={segment_data['control_count']})"
                    )
                    lines.append(
                        f"  - Treatment: {segment_data['treatment_mean']:.4f} (n={segment_data['treatment_count']})"
                    )
                    lines.append(f"  - Change: {segment_data['percent_change']:+.2f}%")
                    lines.append(f"  - P-value: {segment_data['p_value']:.4f}")

                    # Add segment paired analysis if available
                    if (
                        "segment_paired_analysis" in metric_data
                        and segment_name in metric_data["segment_paired_analysis"]
                    ):
                        seg_paired = metric_data["segment_paired_analysis"][
                            segment_name
                        ]
                        if (
                            "paired_analysis" in seg_paired
                            and "error" not in seg_paired["paired_analysis"]
                        ):
                            seg_stats = seg_paired["paired_analysis"]
                            lines.append(
                                f"  - Valid pairs: {seg_paired['missing_analysis']['both_present']}"
                            )
                            lines.append(
                                f"  - Win rate: {seg_stats['win_loss']['treatment_win_rate']:.1f}%"
                            )
                            lines.append(
                                f"  - Mean diff: {seg_stats['mean_difference']:.4f}"
                            )

                lines.append("")

        return "\n".join(lines)


def list_metrics(job_path: str) -> List[str]:
    """
    List all available metrics in the dataset.

    Args:
        job_path: Path to the job folder containing metrics

    Returns:
        List of available metric names
    """
    comparator = MetricsComparison(job_path)
    return comparator.get_available_metrics()


def comprehensive_metrics_analysis(
    job_path: str,
    metrics: Optional[str] = None,
    segment_column: Optional[str] = None,
    paired_test: bool = True,
    output_file: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Perform comprehensive statistical analysis comparing metrics between control and treatment groups.

    This function provides a complete A/B testing analysis including:
    - Statistical significance testing (t-tests, p-values)
    - Paired analysis with missing data patterns
    - Win/loss analysis (how many queries treatment beats control)
    - Score difference distribution analysis (percentiles, std dev, etc.)
    - Segment-level breakdowns for all above metrics
    - Markdown report generation

    Args:
        job_path: Path to the job folder containing metrics
        metrics: Comma-separated list of metrics to analyze (default: all available)
        segment_column: Column name to segment analysis by (e.g., 'segment 2')
        paired_test: Use paired t-test (True) or independent t-test (False)
        output_file: Output file path for the comprehensive report (optional)

    Returns:
        Dictionary containing comprehensive analysis results including:
        - Statistical comparison (means, p-values, significance)
        - Paired analysis (missing data patterns, win/loss counts)
        - Distribution statistics (percentiles, std dev, min/max)
        - Segment-level breakdowns of all above
    """
    comparator = MetricsComparison(job_path)

    # Parse metrics list
    metric_names = None
    if metrics:
        metric_names = [m.strip() for m in metrics.split(",")]

    # Run comparison
    results = comparator.compare_multiple_metrics(
        metric_names=metric_names,
        segment_column=segment_column,
        paired_test=paired_test,
    )

    if output_file and results:
        # Generate report
        report = comparator.generate_comparison_report(results)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to: {output_file}")
        # Don't return results when outputting to file to avoid console JSON spam
        return None

    # Return results when no output file specified (for programmatic use)
    return results


def paired_analysis(
    job_path: str,
    metric: str,
    segment_column: Optional[str] = None,
    output_file: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Perform detailed paired analysis for a specific metric.

    Args:
        job_path: Path to the job folder containing metrics
        metric: The metric base name to analyze
        segment_column: Column name to segment analysis by (e.g., 'segment 2')
        output_file: Output file path for the report (optional)

    Returns:
        Dictionary containing paired analysis results
    """
    comparator = MetricsComparison(job_path)

    # Get overall paired analysis
    overall_results = comparator._calculate_paired_analysis(metric)

    results = {"metric": metric, "overall": overall_results}

    # Add segment analysis if requested
    if (
        segment_column
        and comparator.df is not None
        and segment_column in comparator.df.columns
    ):
        segments = {}
        unique_segments = comparator.df[segment_column].dropna().unique()

        for segment in unique_segments:
            segment_df = comparator.df[comparator.df[segment_column] == segment]
            segments[str(segment)] = comparator._calculate_paired_analysis(
                metric, segment_df
            )

        results["segments"] = segments

    if output_file:
        # Generate detailed report
        report_lines = []
        report_lines.append(f"# Paired Analysis Report: {metric}")
        report_lines.append("")

        # Overall analysis
        if "missing_analysis" in overall_results:
            missing = overall_results["missing_analysis"]
            report_lines.append("## Overall Missing Data Analysis")
            report_lines.append(f"- Total queries: {missing['total_queries']}")
            report_lines.append(
                f"- Valid pairs: {missing['both_present']} ({missing['both_present_pct']:.1f}%)"
            )
            report_lines.append(
                f"- Control only: {missing['only_control']} ({missing['only_control_pct']:.1f}%)"
            )
            report_lines.append(
                f"- Treatment only: {missing['only_treatment']} ({missing['only_treatment_pct']:.1f}%)"
            )
            report_lines.append(
                f"- Both missing: {missing['both_missing']} ({missing['both_missing_pct']:.1f}%)"
            )
            report_lines.append("")

        if (
            "paired_analysis" in overall_results
            and "error" not in overall_results["paired_analysis"]
        ):
            paired = overall_results["paired_analysis"]
            total_pairs = paired["valid_pairs"]
            treatment_wins = paired["win_loss"]["treatment_wins"]
            ties = paired["win_loss"]["ties"]
            control_wins = paired["win_loss"]["control_wins"]

            report_lines.append("## Overall Win/Loss Analysis")
            report_lines.append(f"- Total valid pairs: {total_pairs}")
            report_lines.append(
                f"- Treatment wins: {treatment_wins} ({treatment_wins/total_pairs*100:.1f}%)"
            )
            report_lines.append(f"- Ties: {ties} ({ties/total_pairs*100:.1f}%)")
            report_lines.append(
                f"- Control wins: {control_wins} ({control_wins/total_pairs*100:.1f}%)"
            )
            report_lines.append("")

            report_lines.append("## Score Difference Distribution")
            report_lines.append(f"- Mean difference: {paired['mean_difference']:.4f}")
            report_lines.append(
                f"- Median difference: {paired['median_difference']:.4f}"
            )
            report_lines.append(f"- Std deviation: {paired['std_difference']:.4f}")
            report_lines.append(f"- Min difference: {paired['min_difference']:.4f}")
            report_lines.append(f"- Max difference: {paired['max_difference']:.4f}")
            report_lines.append("")

            report_lines.append("## Percentiles")
            for p in [5, 10, 25, 50, 75, 90, 95]:
                report_lines.append(f"- P{p}: {paired['percentiles'][f'p{p}']:.2f}")
            report_lines.append("")

            # Add histogram section
            if "histogram" in paired:
                histogram = paired["histogram"]
                report_lines.append("## Score Difference Histogram")
                report_lines.append(
                    f"Distribution of score differences across {len(histogram['bins'])} bins:"
                )
                report_lines.append("")

                for bin_data in histogram["bins"]:
                    count = bin_data["count"]
                    percentage = bin_data["percentage"]
                    range_str = bin_data["range"]
                    # Create a simple ASCII bar chart
                    bar_length = int(percentage / 2)  # Scale down for readability
                    bar = "█" * bar_length
                    report_lines.append(
                        f"- {range_str:<15} {count:>4} queries ({percentage:>5.1f}%) {bar}"
                    )
                report_lines.append("")

            # Add top and bottom 10 queries if available
            if "top_10_queries" in paired:
                report_lines.append("## Top 10 Queries (Treatment >> Control)")
                report_lines.append(paired["top_10_queries"]["description"])
                report_lines.append("")
                report_lines.append(
                    "| Rank | Control Score | Treatment Score | Difference | Query |"
                )
                report_lines.append(
                    "|------|---------------|-----------------|------------|-------|"
                )

                for i, query in enumerate(paired["top_10_queries"]["queries"], 1):
                    control_score = query["control_score"]
                    treatment_score = query["treatment_score"]
                    difference = query["score_difference"]
                    query_text = query.get("query_text", f"Index: {query['index']}")

                    report_lines.append(
                        f"| {i} | {control_score:.3f} | {treatment_score:.3f} | {difference:+.3f} | {query_text} |"
                    )
                report_lines.append("")

            if "bottom_10_queries" in paired:
                report_lines.append("## Bottom 10 Queries (Control >> Treatment)")
                report_lines.append(paired["bottom_10_queries"]["description"])
                report_lines.append("")
                report_lines.append(
                    "| Rank | Control Score | Treatment Score | Difference | Query |"
                )
                report_lines.append(
                    "|------|---------------|-----------------|------------|-------|"
                )

                for i, query in enumerate(paired["bottom_10_queries"]["queries"], 1):
                    control_score = query["control_score"]
                    treatment_score = query["treatment_score"]
                    difference = query["score_difference"]
                    query_text = query.get("query_text", f"Index: {query['index']}")

                    report_lines.append(
                        f"| {i} | {control_score:.3f} | {treatment_score:.3f} | {difference:+.3f} | {query_text} |"
                    )
                report_lines.append("")

        # Segment analysis
        if "segments" in results:
            report_lines.append("## Segment Analysis")
            for segment_name, segment_data in results["segments"].items():
                report_lines.append(f"### {segment_name}")

                # Missing data analysis for segment
                if "missing_analysis" in segment_data:
                    missing = segment_data["missing_analysis"]
                    report_lines.append("**Missing Data Analysis:**")
                    report_lines.append(f"- Total queries: {missing['total_queries']}")
                    report_lines.append(
                        f"- Valid pairs: {missing['both_present']} ({missing['both_present_pct']:.1f}%)"
                    )
                    report_lines.append(
                        f"- Control only: {missing['only_control']} ({missing['only_control_pct']:.1f}%)"
                    )
                    report_lines.append(
                        f"- Treatment only: {missing['only_treatment']} ({missing['only_treatment_pct']:.1f}%)"
                    )
                    report_lines.append(
                        f"- Both missing: {missing['both_missing']} ({missing['both_missing_pct']:.1f}%)"
                    )
                    report_lines.append("")

                # Detailed paired analysis for segment
                if (
                    "paired_analysis" in segment_data
                    and "error" not in segment_data["paired_analysis"]
                ):
                    paired = segment_data["paired_analysis"]
                    total_pairs = paired["valid_pairs"]
                    treatment_wins = paired["win_loss"]["treatment_wins"]
                    ties = paired["win_loss"]["ties"]
                    control_wins = paired["win_loss"]["control_wins"]

                    # Win/Loss Analysis
                    report_lines.append("**Win/Loss Analysis:**")
                    report_lines.append(f"- Total valid pairs: {total_pairs}")
                    report_lines.append(
                        f"- Treatment wins: {treatment_wins} ({treatment_wins/total_pairs*100:.1f}%)"
                    )
                    report_lines.append(f"- Ties: {ties} ({ties/total_pairs*100:.1f}%)")
                    report_lines.append(
                        f"- Control wins: {control_wins} ({control_wins/total_pairs*100:.1f}%)"
                    )
                    report_lines.append("")  # Score Difference Distribution
                    report_lines.append("**Score Difference Distribution:**")
                    report_lines.append(
                        f"- Mean difference: {paired['mean_difference']:.4f}"
                    )
                    report_lines.append(
                        f"- Median difference: {paired['median_difference']:.4f}"
                    )
                    report_lines.append(
                        f"- Std deviation: {paired['std_difference']:.4f}"
                    )
                    report_lines.append(
                        f"- Min difference: {paired['min_difference']:.4f}"
                    )
                    report_lines.append(
                        f"- Max difference: {paired['max_difference']:.4f}"
                    )
                    report_lines.append("")

                    # Percentiles
                    report_lines.append("**Percentiles:**")
                    for p in [5, 10, 25, 50, 75, 90, 95]:
                        report_lines.append(
                            f"- P{p}: {paired['percentiles'][f'p{p}']:.2f}"
                        )
                    report_lines.append("")

                    # Histogram for segment
                    if "histogram" in paired:
                        histogram = paired["histogram"]
                        report_lines.append("**Score Difference Histogram:**")
                        report_lines.append(
                            f"Distribution of score differences across {len(histogram['bins'])} bins:"
                        )
                        report_lines.append("")

                        for bin_data in histogram["bins"]:
                            count = bin_data["count"]
                            percentage = bin_data["percentage"]
                            range_str = bin_data["range"]
                            # Create a simple ASCII bar chart
                            bar_length = int(
                                percentage / 2
                            )  # Scale down for readability
                            bar = "█" * bar_length
                            report_lines.append(
                                f"- {range_str:<15} {count:>4} queries ({percentage:>5.1f}%) {bar}"
                            )
                        report_lines.append("")

                    # Add top and bottom 10 queries for segments
                    if "top_10_queries" in paired:
                        report_lines.append(
                            "**Top 10 Queries (Treatment >> Control):**"
                        )
                        report_lines.append("")
                        report_lines.append(
                            "| Rank | Control | Treatment | Diff | Query |"
                        )
                        report_lines.append(
                            "|------|---------|-----------|------|-------|"
                        )

                        for i, query in enumerate(
                            paired["top_10_queries"]["queries"], 1
                        ):
                            control_score = query["control_score"]
                            treatment_score = query["treatment_score"]
                            difference = query["score_difference"]
                            query_text = query.get(
                                "query_text", f"Idx:{query['index']}"
                            )
                            # Further truncate for segment tables (50 chars)
                            if len(query_text) > 50:
                                query_text = query_text[:47] + "..."

                            report_lines.append(
                                f"| {i} | {control_score:.2f} | {treatment_score:.2f} | {difference:+.2f} | {query_text} |"
                            )
                        report_lines.append("")

                    if "bottom_10_queries" in paired:
                        report_lines.append(
                            "**Bottom 10 Queries (Control >> Treatment):**"
                        )
                        report_lines.append("")
                        report_lines.append(
                            "| Rank | Control | Treatment | Diff | Query |"
                        )
                        report_lines.append(
                            "|------|---------|-----------|------|-------|"
                        )

                        for i, query in enumerate(
                            paired["bottom_10_queries"]["queries"], 1
                        ):
                            control_score = query["control_score"]
                            treatment_score = query["treatment_score"]
                            difference = query["score_difference"]
                            query_text = query.get(
                                "query_text", f"Idx:{query['index']}"
                            )
                            # Further truncate for segment tables (50 chars)
                            if len(query_text) > 50:
                                query_text = query_text[:47] + "..."

                            report_lines.append(
                                f"| {i} | {control_score:.2f} | {treatment_score:.2f} | {difference:+.2f} | {query_text} |"
                            )
                        report_lines.append("")

                report_lines.append("")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"Paired analysis report saved to: {output_file}")
        # Don't return results when outputting to file to avoid console JSON spam
        return None

    # Return results when no output file specified (for programmatic use)
    return results


def export_win_loss_utterances(
    job_path: str,
    metric: str,
    output_dir: str = "results",
    segment_column: Optional[str] = None,
    min_difference: float = 0.0,
) -> Dict[str, Any]:
    """
    Export utterances where treatment outperforms control and vice versa to separate files.

    This function analyzes a specific metric and exports:
    1. Utterances where treatment > control (treatment wins)
    2. Utterances where control > treatment (control wins)
    3. Utterances where treatment == control (ties)

    Note on Missing Data Handling:
        - Only utterances with BOTH control AND treatment scores present are included
        - Utterances with missing control score, missing treatment score, or both missing
          are EXCLUDED from all output files (not treated as ties)
        - This ensures fair comparison - we only export cases where both variants
          successfully produced a score

    Args:
        job_path: Path to the job folder containing metrics
        metric: The metric base name to analyze
        output_dir: Directory to save the output files (default: "results")
        segment_column: Optional column name to include segment information
        min_difference: Minimum absolute difference to include (default: 0.0)
                       Use this to filter out small differences

    Returns:
        Dictionary with statistics about exported utterances
    """
    import os
    from pathlib import Path

    comparator = MetricsComparison(job_path)

    if comparator.df is None:
        raise ValueError("Failed to load metrics data")

    control_col = f"{metric}_control"
    treatment_col = f"{metric}_treatment"

    if (
        control_col not in comparator.df.columns
        or treatment_col not in comparator.df.columns
    ):
        raise ValueError(f"Metric columns not found: {control_col}, {treatment_col}")

    # Convert to numeric
    control_vals = pd.to_numeric(comparator.df[control_col], errors="coerce")
    treatment_vals = pd.to_numeric(comparator.df[treatment_col], errors="coerce")

    # Calculate differences
    valid_pairs_mask = ~control_vals.isna() & ~treatment_vals.isna()
    df_valid = comparator.df[valid_pairs_mask].copy()
    df_valid["control_score"] = control_vals[valid_pairs_mask]
    df_valid["treatment_score"] = treatment_vals[valid_pairs_mask]
    df_valid["score_difference"] = (
        df_valid["treatment_score"] - df_valid["control_score"]
    )
    df_valid["abs_difference"] = df_valid["score_difference"].abs()

    # Filter by minimum difference
    df_valid = df_valid[df_valid["abs_difference"] >= min_difference]

    # Split into three categories
    treatment_wins = df_valid[df_valid["score_difference"] > 0].copy()
    control_wins = df_valid[df_valid["score_difference"] < 0].copy()
    ties = df_valid[df_valid["score_difference"] == 0].copy()

    # Sort by absolute difference (largest differences first)
    treatment_wins = treatment_wins.sort_values("score_difference", ascending=False)
    control_wins = control_wins.sort_values("score_difference", ascending=True)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract job ID from job_path (e.g., "seval_data/232361_metrics" -> "232361")
    job_path_obj = Path(job_path)
    job_folder_name = (
        job_path_obj.name
    )  # Gets last part of path (e.g., "232361_metrics")
    # Extract job ID (everything before first underscore, or whole name if no underscore)
    job_id = (
        job_folder_name.split("_")[0] if "_" in job_folder_name else job_folder_name
    )

    # Determine which columns to export
    export_columns = []

    # Find query/utterance column
    query_col = None
    for col_name in [
        "query",
        "Query",
        "question",
        "Question",
        "utterance",
        "Utterance",
    ]:
        if col_name in df_valid.columns:
            query_col = col_name
            export_columns.append(col_name)
            break

    # Add ID columns if available
    for id_col in ["query_id", "id", "ID"]:
        if id_col in df_valid.columns and id_col not in export_columns:
            export_columns.append(id_col)

    # Add segment column if requested
    if segment_column and segment_column in df_valid.columns:
        export_columns.append(segment_column)

    # Add score columns
    export_columns.extend(
        ["control_score", "treatment_score", "score_difference", "abs_difference"]
    )

    # Export treatment wins
    treatment_wins_file = output_path / f"{job_id}_{metric}_treatment_wins.tsv"
    if len(treatment_wins) > 0:
        treatment_wins[export_columns].to_csv(
            treatment_wins_file, sep="\t", index=False, encoding="utf-8"
        )
        print(
            f"✓ Exported {len(treatment_wins)} treatment wins to: {treatment_wins_file}"
        )
    else:
        print(f"⚠ No treatment wins found (with min_difference >= {min_difference})")

    # Export control wins
    control_wins_file = output_path / f"{job_id}_{metric}_control_wins.tsv"
    if len(control_wins) > 0:
        control_wins[export_columns].to_csv(
            control_wins_file, sep="\t", index=False, encoding="utf-8"
        )
        print(f"✓ Exported {len(control_wins)} control wins to: {control_wins_file}")
    else:
        print(f"⚠ No control wins found (with min_difference >= {min_difference})")

    # Export ties (optional, only if there are any)
    ties_file = output_path / f"{job_id}_{metric}_ties.tsv"
    if len(ties) > 0:
        ties[export_columns].to_csv(ties_file, sep="\t", index=False, encoding="utf-8")
        print(f"✓ Exported {len(ties)} ties to: {ties_file}")

    # Generate summary statistics
    stats = {
        "metric": metric,
        "total_valid_pairs": len(df_valid),
        "min_difference_threshold": min_difference,
        "treatment_wins": {
            "count": len(treatment_wins),
            "percentage": (
                len(treatment_wins) / len(df_valid) * 100 if len(df_valid) > 0 else 0
            ),
            "mean_difference": (
                float(treatment_wins["score_difference"].mean())
                if len(treatment_wins) > 0
                else 0
            ),
            "max_difference": (
                float(treatment_wins["score_difference"].max())
                if len(treatment_wins) > 0
                else 0
            ),
            "file": str(treatment_wins_file),
        },
        "control_wins": {
            "count": len(control_wins),
            "percentage": (
                len(control_wins) / len(df_valid) * 100 if len(df_valid) > 0 else 0
            ),
            "mean_difference": (
                float(control_wins["score_difference"].mean())
                if len(control_wins) > 0
                else 0
            ),
            "min_difference": (
                float(control_wins["score_difference"].min())
                if len(control_wins) > 0
                else 0
            ),
            "file": str(control_wins_file),
        },
        "ties": {
            "count": len(ties),
            "percentage": len(ties) / len(df_valid) * 100 if len(df_valid) > 0 else 0,
            "file": str(ties_file) if len(ties) > 0 else None,
        },
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"Win/Loss Export Summary for: {metric}")
    print("=" * 60)
    print(f"Total valid pairs: {stats['total_valid_pairs']}")
    print(f"Minimum difference threshold: {min_difference}")
    print(
        f"\nTreatment wins: {stats['treatment_wins']['count']} ({stats['treatment_wins']['percentage']:.1f}%)"
    )
    if stats["treatment_wins"]["count"] > 0:
        print(f"  - Mean difference: {stats['treatment_wins']['mean_difference']:.4f}")
        print(f"  - Max difference: {stats['treatment_wins']['max_difference']:.4f}")
    print(
        f"\nControl wins: {stats['control_wins']['count']} ({stats['control_wins']['percentage']:.1f}%)"
    )
    if stats["control_wins"]["count"] > 0:
        print(f"  - Mean difference: {stats['control_wins']['mean_difference']:.4f}")
        print(f"  - Min difference: {stats['control_wins']['min_difference']:.4f}")
    print(f"\nTies: {stats['ties']['count']} ({stats['ties']['percentage']:.1f}%)")
    print("=" * 60)

    return stats


def export_two_jobs_utterances(
    job1_metrics_path: str,
    job2_metrics_path: str,
    metric: str,
    output_file: str,
    job1_id: Optional[str] = None,
    job2_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Combine ALL utterances from both SEVAL jobs into a single file.

    This function loads the full metrics CSV from both jobs and combines them, showing:
    - Score values and gains (treatment - control) for both jobs
    - Combined gain (sum of both job gains)
    - Utterances are kept in the original order from the first job
    - All utterances are included, even those with missing scores

    Args:
        job1_metrics_path: Path to job1 metrics folder (e.g., "seval_data/123665_metrics")
        job2_metrics_path: Path to job2 metrics folder (e.g., "seval_data/232361_metrics")
        metric: Metric name (e.g., "citedcg_one_centric")
        output_file: Path for combined output TSV file
        job1_id: Optional identifier for first job (auto-extracted from path if not provided)
        job2_id: Optional identifier for second job (auto-extracted from path if not provided)

    Returns:
        Dictionary with statistics

    Example:
        export_two_jobs_utterances(
            job1_metrics_path="seval_data/123665_metrics",
            job2_metrics_path="seval_data/232361_metrics",
            metric="citedcg_one_centric",
            output_file="results/combined_all_utterances.tsv"
        )
    """
    # Auto-extract job IDs from paths if not provided
    if job1_id is None:
        # Extract from path like "seval_data/123665_metrics" -> "123665"
        job1_folder = Path(job1_metrics_path).name
        job1_id = job1_folder.split("_")[0]
        logger.info(f"Auto-extracted job1_id: {job1_id}")

    if job2_id is None:
        # Extract from path like "seval_data/232361_metrics" -> "232361"
        job2_folder = Path(job2_metrics_path).name
        job2_id = job2_folder.split("_")[0]
        logger.info(f"Auto-extracted job2_id: {job2_id}")

    # Load both jobs' data
    logger.info(f"Loading job {job1_id} from: {job1_metrics_path}")
    comparator1 = MetricsComparison(job1_metrics_path)

    logger.info(f"Loading job {job2_id} from: {job2_metrics_path}")
    comparator2 = MetricsComparison(job2_metrics_path)

    # Get dataframes
    df1 = comparator1.df
    df2 = comparator2.df

    if df1 is None or df2 is None:
        raise ValueError("Failed to load metrics data from one or both jobs")

    # Get the metric columns
    control_col1 = f"{metric}_control"
    treatment_col1 = f"{metric}_treatment"
    control_col2 = f"{metric}_control"
    treatment_col2 = f"{metric}_treatment"

    # Verify columns exist
    for col, job in [(control_col1, job1_id), (treatment_col1, job1_id)]:
        if col not in df1.columns:
            raise ValueError(f"Column '{col}' not found in job {job}")

    for col, job in [(control_col2, job2_id), (treatment_col2, job2_id)]:
        if col not in df2.columns:
            raise ValueError(f"Column '{col}' not found in job {job}")

    # Find common query/utterance column
    match_column = None
    for candidate in [
        "query",
        "Query",
        "utterance",
        "Utterance",
        "query_id",
        "id",
        "ID",
    ]:
        if candidate in df1.columns and candidate in df2.columns:
            match_column = candidate
            logger.info(f"Using match column: {match_column}")
            break

    if match_column is None:
        raise ValueError("Could not find common query/utterance column in both jobs")

    # Prepare job1 data with metric name in column headers
    df1_subset = df1[[match_column, control_col1, treatment_col1]].copy()
    df1_subset[f"{metric}_job{job1_id}_control"] = pd.to_numeric(
        df1_subset[control_col1], errors="coerce"
    )
    df1_subset[f"{metric}_job{job1_id}_treatment"] = pd.to_numeric(
        df1_subset[treatment_col1], errors="coerce"
    )
    df1_subset[f"{metric}_job{job1_id}_gain"] = (
        df1_subset[f"{metric}_job{job1_id}_treatment"]
        - df1_subset[f"{metric}_job{job1_id}_control"]
    )

    # Add segment column if it exists
    segment_col = None
    for seg_candidate in ["segment 2", "Segment", "segment"]:
        if seg_candidate in df1.columns:
            df1_subset[seg_candidate] = df1[seg_candidate]
            segment_col = seg_candidate
            break

    # Prepare job2 data with metric name in column headers
    df2_subset = df2[[match_column, control_col2, treatment_col2]].copy()
    df2_subset[f"{metric}_job{job2_id}_control"] = pd.to_numeric(
        df2_subset[control_col2], errors="coerce"
    )
    df2_subset[f"{metric}_job{job2_id}_treatment"] = pd.to_numeric(
        df2_subset[treatment_col2], errors="coerce"
    )
    df2_subset[f"{metric}_job{job2_id}_gain"] = (
        df2_subset[f"{metric}_job{job2_id}_treatment"]
        - df2_subset[f"{metric}_job{job2_id}_control"]
    )

    # Merge on match column (outer join to include all utterances)
    merged = pd.merge(
        df1_subset, df2_subset, on=match_column, how="outer", suffixes=("", "_dup")
    )

    logger.info(f"Total utterances in combined output: {len(merged)}")

    # Calculate combined gain
    merged[f"{metric}_combined_gain"] = (
        merged[f"{metric}_job{job1_id}_gain"] + merged[f"{metric}_job{job2_id}_gain"]
    )

    # Keep original order from first job (no sorting)
    # Note: Outer join preserves order from left dataframe (job1)

    # Prepare export columns
    export_columns = [match_column]

    if segment_col and segment_col in merged.columns:
        export_columns.append(segment_col)

    # Add all metric columns with metric name in headers
    export_columns.extend(
        [
            f"{metric}_job{job1_id}_control",
            f"{metric}_job{job1_id}_treatment",
            f"{metric}_job{job1_id}_gain",
            f"{metric}_job{job2_id}_control",
            f"{metric}_job{job2_id}_treatment",
            f"{metric}_job{job2_id}_gain",
            f"{metric}_combined_gain",
        ]
    )

    # Export to TSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged[export_columns].to_csv(output_path, sep="\t", index=False, encoding="utf-8")

    # Calculate statistics
    combined_gain_col = f"{metric}_combined_gain"
    job1_gain_col = f"{metric}_job{job1_id}_gain"
    job2_gain_col = f"{metric}_job{job2_id}_gain"

    # Count utterances with valid scores in both jobs
    valid_both = (~merged[job1_gain_col].isna() & ~merged[job2_gain_col].isna()).sum()

    # Count utterances with missing scores
    missing_job1 = merged[job1_gain_col].isna().sum()
    missing_job2 = merged[job2_gain_col].isna().sum()

    # Win/Loss/Tie analysis for each job (among all utterances)
    job1_wins = (merged[job1_gain_col] > 0).sum()
    job1_losses = (merged[job1_gain_col] < 0).sum()
    job1_ties = (merged[job1_gain_col] == 0).sum()

    job2_wins = (merged[job2_gain_col] > 0).sum()
    job2_losses = (merged[job2_gain_col] < 0).sum()
    job2_ties = (merged[job2_gain_col] == 0).sum()

    # Cross-job analysis (only for utterances with valid scores in both jobs)
    valid_mask = ~merged[job1_gain_col].isna() & ~merged[job2_gain_col].isna()

    # Both win (treatment > control in both jobs)
    both_win = ((merged[job1_gain_col] > 0) & (merged[job2_gain_col] > 0)).sum()

    # Both lose (treatment < control in both jobs)
    both_lose = ((merged[job1_gain_col] < 0) & (merged[job2_gain_col] < 0)).sum()

    # Both tie (treatment == control in both jobs)
    both_tie = ((merged[job1_gain_col] == 0) & (merged[job2_gain_col] == 0)).sum()

    # Conflict (win in one, lose in other)
    conflict = (
        ((merged[job1_gain_col] > 0) & (merged[job2_gain_col] < 0))
        | ((merged[job1_gain_col] < 0) & (merged[job2_gain_col] > 0))
    ).sum()

    # Mixed with tie/missing (win or lose in one job, tie or missing in the other)
    job1_win_job2_tie_or_missing = (
        (merged[job1_gain_col] > 0)
        & ((merged[job2_gain_col] == 0) | merged[job2_gain_col].isna())
    ).sum()

    job1_lose_job2_tie_or_missing = (
        (merged[job1_gain_col] < 0)
        & ((merged[job2_gain_col] == 0) | merged[job2_gain_col].isna())
    ).sum()

    job2_win_job1_tie_or_missing = (
        (merged[job2_gain_col] > 0)
        & ((merged[job1_gain_col] == 0) | merged[job1_gain_col].isna())
    ).sum()

    job2_lose_job1_tie_or_missing = (
        (merged[job2_gain_col] < 0)
        & ((merged[job1_gain_col] == 0) | merged[job1_gain_col].isna())
    ).sum()

    total = len(merged)

    stats = {
        "job1_id": job1_id,
        "job2_id": job2_id,
        "metric": metric,
        "total_utterances": total,
        "valid_in_both_jobs": int(valid_both),
        "missing_in_job1": int(missing_job1),
        "missing_in_job2": int(missing_job2),
        "job1_performance": {
            "wins": int(job1_wins),
            "losses": int(job1_losses),
            "ties": int(job1_ties),
        },
        "job2_performance": {
            "wins": int(job2_wins),
            "losses": int(job2_losses),
            "ties": int(job2_ties),
        },
        "cross_job_analysis": {
            "both_win": int(both_win),
            "both_lose": int(both_lose),
            "both_tie": int(both_tie),
            "conflict": int(conflict),
            "job1_win_job2_tie_or_missing": int(job1_win_job2_tie_or_missing),
            "job1_lose_job2_tie_or_missing": int(job1_lose_job2_tie_or_missing),
            "job2_win_job1_tie_or_missing": int(job2_win_job1_tie_or_missing),
            "job2_lose_job1_tie_or_missing": int(job2_lose_job1_tie_or_missing),
        },
        "combined_gain_stats": {
            "mean": float(merged[combined_gain_col].mean()) if valid_both > 0 else None,
            "median": (
                float(merged[combined_gain_col].median()) if valid_both > 0 else None
            ),
            "min": float(merged[combined_gain_col].min()) if valid_both > 0 else None,
            "max": float(merged[combined_gain_col].max()) if valid_both > 0 else None,
        },
        "output_file": str(output_path),
    }

    # Print summary
    print("\n" + "=" * 80)
    print(f"Export Two Jobs Utterances: Job {job1_id} + Job {job2_id}")
    print(f"Metric: {metric}")
    print("=" * 80)
    print(f"Total utterances: {total}")
    print(f"Valid scores in both jobs: {valid_both} ({valid_both/total*100:.1f}%)")
    print(f"Missing in job {job1_id}: {missing_job1} ({missing_job1/total*100:.1f}%)")
    print(f"Missing in job {job2_id}: {missing_job2} ({missing_job2/total*100:.1f}%)")

    print()
    print("-" * 80)
    print(f"Job {job1_id} Performance (Treatment vs Control):")
    print(f"  - Treatment Wins:   {job1_wins:>5} ({job1_wins/total*100:>5.1f}%)")
    print(f"  - Treatment Losses: {job1_losses:>5} ({job1_losses/total*100:>5.1f}%)")
    print(f"  - Ties:             {job1_ties:>5} ({job1_ties/total*100:>5.1f}%)")

    print()
    print(f"Job {job2_id} Performance (Treatment vs Control):")
    print(f"  - Treatment Wins:   {job2_wins:>5} ({job2_wins/total*100:>5.1f}%)")
    print(f"  - Treatment Losses: {job2_losses:>5} ({job2_losses/total*100:>5.1f}%)")
    print(f"  - Ties:             {job2_ties:>5} ({job2_ties/total*100:>5.1f}%)")

    print()
    print("-" * 80)
    print("Cross-Job Comparison:")
    print(f"  - Both Win (treatment > control in both jobs):")
    print(f"    {both_win:>5} ({both_win/total*100:>5.1f}%)")

    print(f"  - Both Lose (treatment < control in both jobs):")
    print(f"    {both_lose:>5} ({both_lose/total*100:>5.1f}%)")

    print(f"  - Both Tie (treatment == control in both jobs):")
    print(f"    {both_tie:>5} ({both_tie/total*100:>5.1f}%)")

    print(f"  - Conflict (win in one job, lose in other):")
    print(f"    {conflict:>5} ({conflict/total*100:>5.1f}%)")

    print()
    print(f"  - Job {job1_id} wins, Job {job2_id} tie/missing:")
    print(
        f"    {job1_win_job2_tie_or_missing:>5} ({job1_win_job2_tie_or_missing/total*100:>5.1f}%)"
    )

    print(f"  - Job {job1_id} loses, Job {job2_id} tie/missing:")
    print(
        f"    {job1_lose_job2_tie_or_missing:>5} ({job1_lose_job2_tie_or_missing/total*100:>5.1f}%)"
    )

    print(f"  - Job {job2_id} wins, Job {job1_id} tie/missing:")
    print(
        f"    {job2_win_job1_tie_or_missing:>5} ({job2_win_job1_tie_or_missing/total*100:>5.1f}%)"
    )

    print(f"  - Job {job2_id} loses, Job {job1_id} tie/missing:")
    print(
        f"    {job2_lose_job1_tie_or_missing:>5} ({job2_lose_job1_tie_or_missing/total*100:>5.1f}%)"
    )

    if valid_both > 0:
        print()
        print("-" * 80)
        print("Combined Gain Statistics (valid pairs only):")
        print(f"  - Mean:   {stats['combined_gain_stats']['mean']:>8.4f}")
        print(f"  - Median: {stats['combined_gain_stats']['median']:>8.4f}")
        print(f"  - Min:    {stats['combined_gain_stats']['min']:>8.4f}")
        print(f"  - Max:    {stats['combined_gain_stats']['max']:>8.4f}")

    print()
    print("=" * 80)
    print(f"✓ Combined results exported to: {output_path}")
    print(f"  Output maintains original order from job {job1_id}")
    print("=" * 80)

    return stats


def export_two_jobs_utterances_cli(
    job1_metrics_path: str,
    job2_metrics_path: str,
    metric: str,
    output_file: str,
    job1_id: Optional[str] = None,
    job2_id: Optional[str] = None,
) -> None:
    """
    CLI wrapper for export_two_jobs_utterances that doesn't return anything.

    This prevents Fire from outputting JSON to console.
    See export_two_jobs_utterances() for full documentation.
    """
    export_two_jobs_utterances(
        job1_metrics_path=job1_metrics_path,
        job2_metrics_path=job2_metrics_path,
        metric=metric,
        output_file=output_file,
        job1_id=job1_id,
        job2_id=job2_id,
    )


def export_win_loss_utterances_cli(
    job_path: str,
    metric: str,
    output_dir: str = "results",
    segment_column: Optional[str] = None,
    min_difference: float = 0.0,
) -> None:
    """
    CLI wrapper for export_win_loss_utterances that doesn't return anything.

    This prevents Fire from outputting JSON to console.
    See export_win_loss_utterances() for full documentation.
    """
    export_win_loss_utterances(
        job_path=job_path,
        metric=metric,
        output_dir=output_dir,
        segment_column=segment_column,
        min_difference=min_difference,
    )


if __name__ == "__main__":
    # Create a dictionary of available functions
    functions = {
        "comprehensive_analysis": comprehensive_metrics_analysis,
        "paired_analysis": paired_analysis,
        "list_metrics": list_metrics,
        "export_win_loss_utterances": export_win_loss_utterances_cli,
        "export_two_jobs_utterances": export_two_jobs_utterances_cli,
    }

    fire.Fire(functions)
