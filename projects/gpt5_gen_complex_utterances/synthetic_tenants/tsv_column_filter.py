#!/usr/bin/env python3
"""
TSV Filter For Seval Data Files

This program filters TSV Seval data files to select rows where a specified column equals a
specific value (similar to SQL SELECT * FROM table WHERE column = value). It reads from the
input TSV file and creates a new TSV file containing only the matching rows, preserving the
original column structure and headers.

Features:
- SQL-like filtering: specify column name and value to match
- Case-insensitive column name matching (works with "Segment" or "segment")
- Preserves TSV format and column headers
- Provides detailed statistics on filtering results
- Handles various file encodings
- Error handling for missing files or malformed data

Usage:
    python seval_data_email_segment_filter.py filter_by_column Segment email
    python seval_data_email_segment_filter.py filter_by_column Environment production --input_file="custom.tsv"
    python seval_data_email_segment_filter.py analyze_column Segment --input_file="results/utterances_seval_timestamped.tsv"
"""

import csv
import os
import logging
import fire
from typing import Dict, List, Optional, Sequence
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TSVFilter:
    """Filters TSV files to extract rows where specified column matches specified values"""

    def __init__(self):
        """Initialize the TSV filter"""
        self.stats = {
            "total_rows": 0,
            "filtered_rows": 0,
            "other_values": {},
            "processing_errors": 0,
        }

    def _find_column_name(
        self, fieldnames: Optional[Sequence[str]], target_column: str
    ) -> Optional[str]:
        """Find the actual column name in fieldnames (case-insensitive)

        Args:
            fieldnames: List of column names from CSV header
            target_column: The target column name to find

        Returns:
            The actual column name if found, None otherwise
        """
        if not fieldnames:
            return None
        target_lower = target_column.lower().strip()
        for field in fieldnames:
            if field.lower().strip() == target_lower:
                return field
        return None

    def analyze_column(
        self,
        column_name: str = "Segment",
        input_file: str = "results/utterances_seval_timestamped.tsv",
        encoding: str = "utf-8",
    ) -> Dict:
        """Analyze all values in the specified column without filtering

        Args:
            column_name: Name of the column to analyze (case-insensitive)
            input_file: Path to the input TSV file
            encoding: File encoding to use

        Returns:
            Dictionary containing column analysis
        """
        logger.info(f"🔍 Analyzing column '{column_name}' in: {input_file}")

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        value_stats = {}
        total_rows = 0
        actual_column = None

        try:
            with open(input_file, "r", encoding=encoding) as infile:
                reader = csv.DictReader(infile, delimiter="\t")

                if not reader.fieldnames:
                    raise ValueError("Input file has no headers")

                # Find the actual column name (case-insensitive)
                actual_column = self._find_column_name(reader.fieldnames, column_name)
                if not actual_column:
                    available_columns = ", ".join(reader.fieldnames)
                    raise ValueError(
                        f"Column '{column_name}' not found. Available columns: {available_columns}"
                    )

                for row in reader:
                    total_rows += 1
                    value = row.get(actual_column, "").strip().lower()

                    if value:
                        value_stats[value] = value_stats.get(value, 0) + 1

        except Exception as e:
            logger.error(f"❌ Failed to analyze column '{column_name}': {str(e)}")
            raise

        # Display analysis results
        logger.info(f"\n📊 Column '{actual_column}' Analysis Results:")
        logger.info(f"📄 Total rows: {total_rows:,}")
        logger.info(f"📋 Values found:")

        for value, count in sorted(
            value_stats.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / total_rows) * 100
            logger.info(f"  - {value}: {count:,} rows ({percentage:.1f}%)")

        return {
            "total_rows": total_rows,
            "column_name": actual_column,
            "values": value_stats,
            "analysis_timestamp": datetime.now().isoformat(),
            "input_file": input_file,
        }

    def filter_by_column(
        self,
        column_name: str,
        column_value: str,
        input_file: str = "results/utterances_seval_timestamped.tsv",
        output_file: Optional[str] = None,
        show_stats: bool = True,
        encoding: str = "utf-8",
    ) -> Dict:
        """Filter TSV file by any column name and value (SQL-like WHERE clause)

        Args:
            column_name: The column name to filter by (case-insensitive)
            column_value: The value to match in the specified column
            input_file: Path to the input TSV file
            output_file: Path to the output TSV file (auto-generated if None)
            show_stats: Whether to display filtering statistics
            encoding: File encoding to use

        Returns:
            Dictionary containing filtering statistics
        """
        # Auto-generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            safe_column = column_name.lower().replace(" ", "_")
            safe_value = column_value.lower().replace(" ", "_")
            output_file = f"{base_name}_{safe_column}_{safe_value}.tsv"

        logger.info(f"🔍 Filtering by {column_name}='{column_value}'")

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Reset statistics
        self.stats = {
            "total_rows": 0,
            "filtered_rows": 0,
            "other_values": {},
            "processing_errors": 0,
            "target_column": column_name,
            "target_value": column_value,
        }

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        actual_column = None

        try:
            with open(input_file, "r", encoding=encoding) as infile, open(
                output_file, "w", encoding=encoding, newline=""
            ) as outfile:

                reader = csv.DictReader(infile, delimiter="\t")
                fieldnames = reader.fieldnames

                if not fieldnames:
                    raise ValueError("Input file has no headers")

                # Find the actual column name (case-insensitive)
                actual_column = self._find_column_name(fieldnames, column_name)
                if not actual_column:
                    available_columns = ", ".join(fieldnames)
                    raise ValueError(
                        f"Column '{column_name}' not found. Available columns: {available_columns}"
                    )

                writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()

                for row_num, row in enumerate(reader, start=2):
                    try:
                        self.stats["total_rows"] += 1

                        row_value = row.get(actual_column, "").strip().lower()
                        target_value = column_value.strip().lower()

                        if row_value == target_value:
                            writer.writerow(row)
                            self.stats["filtered_rows"] += 1
                        else:
                            if row_value:
                                self.stats["other_values"][row_value] = (
                                    self.stats["other_values"].get(row_value, 0) + 1
                                )

                    except Exception as e:
                        logger.warning(f"⚠️ Error processing row {row_num}: {str(e)}")
                        self.stats["processing_errors"] += 1

        except Exception as e:
            logger.error(f"❌ Failed to process files: {str(e)}")
            raise

        if show_stats:
            logger.info(
                f"\n📊 Column '{actual_column}' = '{column_value}' Filtering Statistics:"
            )
            logger.info(f"📄 Total rows processed: {self.stats['total_rows']:,}")
            logger.info(f"🎯 Matching rows found: {self.stats['filtered_rows']:,}")
            logger.info(
                f"🔢 Other rows: {self.stats['total_rows'] - self.stats['filtered_rows']:,}"
            )

        logger.info(f"✅ Filtering completed successfully!")
        logger.info(f"💾 Output saved to: {output_file}")

        return self.stats


# Standalone functions for Fire CLI
def analyze_column(
    column_name: str = "Segment",
    input_file: str = "results/utterances_seval_timestamped.tsv",
    encoding: str = "utf-8",
) -> None:
    """Analyze all values in the specified column

    Args:
        column_name: Name of the column to analyze (case-insensitive)
        input_file: Path to the input TSV file
        encoding: File encoding to use
    """
    filter_obj = TSVFilter()
    filter_obj.analyze_column(
        column_name=column_name, input_file=input_file, encoding=encoding
    )


def filter_by_column(
    column_name: str,
    column_value: str,
    input_file: str = "results/utterances_seval_timestamped.tsv",
    output_file: Optional[str] = None,
    show_stats: bool = True,
    encoding: str = "utf-8",
) -> None:
    """Filter TSV file by any column name and value (SQL-like WHERE clause)

    Args:
        column_name: The column name to filter by (case-insensitive)
        column_value: The value to match in the specified column
        input_file: Path to the input TSV file
        output_file: Path to the output TSV file (auto-generated if None)
        show_stats: Whether to display filtering statistics
        encoding: File encoding to use
    """
    filter_obj = TSVFilter()
    filter_obj.filter_by_column(
        column_name=column_name,
        column_value=column_value,
        input_file=input_file,
        output_file=output_file,
        show_stats=show_stats,
        encoding=encoding,
    )


if __name__ == "__main__":
    fire.Fire()
