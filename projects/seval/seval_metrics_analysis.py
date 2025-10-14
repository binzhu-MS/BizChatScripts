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


if __name__ == "__main__":
    # Create a dictionary of available functions
    functions = {
        "comprehensive_analysis": comprehensive_metrics_analysis,
        "paired_analysis": paired_analysis,
        "list_metrics": list_metrics,
    }

    fire.Fire(functions)
