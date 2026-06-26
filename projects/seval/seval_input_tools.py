#!/usr/bin/env python3
"""
SEVAL Input Tools

A collection of utility functions for analyzing and processing SEVAL input data files.
This script reads TSV/CSV input files and provides various analysis and extraction capabilities.

Features:
- List distinct values for any column in the input file
- Column statistics and data profiling
- Input data validation and summary

Main Functions:
1. list_distinct_values: List all unique values for a specified column
2. list_columns: Display all available columns in the dataset
3. column_statistics: Get basic statistics for a column

Usage Examples:
    # List all columns in a TSV file
    python seval_input_tools.py list_columns "path/to/file.tsv"

    # List distinct values for 'user_id' column
    python seval_input_tools.py list_distinct_values "path/to/file.tsv" --column="user_id"

    # List distinct values with counts
    python seval_input_tools.py list_distinct_values "path/to/file.tsv" --column="user_id" --show_counts=True

    # Get column statistics
    python seval_input_tools.py column_statistics "path/to/file.tsv" --column="user_id"

Input Requirements:
- TSV or CSV file with headers
- Delimiter auto-detected based on file extension (.tsv -> tab, .csv -> comma)
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import fire
import pandas as pd

# Add parent directory to path to import utils
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent  # Go up to BizChatScripts root
sys.path.insert(0, str(parent_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputDataAnalyzer:
    """A class for analyzing SEVAL input data files."""

    def __init__(self, file_path: str):
        """
        Initialize the input data analyzer.

        Args:
            file_path: Path to the TSV or CSV file to analyze
        """
        self.file_path = Path(file_path)
        self.df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        """Load the data file (TSV or CSV)."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Determine delimiter based on file extension
        suffix = self.file_path.suffix.lower()
        if suffix == ".tsv":
            delimiter = "\t"
        elif suffix == ".csv":
            delimiter = ","
        else:
            # Try to auto-detect
            logger.warning(
                f"Unknown file extension '{suffix}', attempting to auto-detect delimiter"
            )
            delimiter = None

        try:
            self.df = pd.read_csv(
                self.file_path,
                sep=delimiter,
                engine="python" if delimiter is None else None,
            )
            logger.info(
                f"Loaded data: {self.df.shape[0]} rows x {self.df.shape[1]} columns from {self.file_path.absolute()}"
            )
        except Exception as e:
            raise Exception(f"Error reading file: {e}")

    def get_columns(self) -> List[str]:
        """Get list of all column names."""
        if self.df is None:
            return []
        return list(self.df.columns)

    def get_distinct_values(
        self, column: str, sort: bool = True
    ) -> List[Any]:
        """
        Get distinct values for a specified column.

        Args:
            column: Column name to get distinct values for
            sort: Whether to sort the values (default: True)

        Returns:
            List of distinct values
        """
        if self.df is None:
            raise ValueError("Data not loaded")

        if column not in self.df.columns:
            raise ValueError(
                f"Column '{column}' not found. Available columns: {list(self.df.columns)}"
            )

        values = self.df[column].dropna().unique().tolist()

        if sort:
            try:
                values = sorted(values)
            except TypeError:
                # Can't sort mixed types
                pass

        return values

    def get_value_counts(self, column: str) -> Dict[Any, int]:
        """
        Get value counts for a specified column.

        Args:
            column: Column name to get value counts for

        Returns:
            Dictionary mapping values to their counts
        """
        if self.df is None:
            raise ValueError("Data not loaded")

        if column not in self.df.columns:
            raise ValueError(
                f"Column '{column}' not found. Available columns: {list(self.df.columns)}"
            )

        counts = self.df[column].value_counts(dropna=False)
        return counts.to_dict()

    def get_column_statistics(self, column: str) -> Dict[str, Any]:
        """
        Get statistics for a specified column.

        Args:
            column: Column name to analyze

        Returns:
            Dictionary with column statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded")

        if column not in self.df.columns:
            raise ValueError(
                f"Column '{column}' not found. Available columns: {list(self.df.columns)}"
            )

        col_data = self.df[column]
        stats: Dict[str, Any] = {
            "column_name": column,
            "total_rows": len(col_data),
            "non_null_count": col_data.notna().sum(),
            "null_count": col_data.isna().sum(),
            "distinct_count": col_data.nunique(dropna=True),
            "dtype": str(col_data.dtype),
        }

        # Add numeric statistics if applicable
        numeric_col = pd.to_numeric(col_data, errors="coerce")
        if numeric_col.notna().sum() > 0:
            stats["numeric_stats"] = {
                "mean": float(numeric_col.mean()),
                "median": float(numeric_col.median()),
                "min": float(numeric_col.min()),
                "max": float(numeric_col.max()),
                "std": float(numeric_col.std()),
            }

        return stats


def list_columns(file_path: str) -> List[str]:
    """
    List all columns in the input file.

    Args:
        file_path: Path to the TSV or CSV file

    Returns:
        List of column names
    """
    analyzer = InputDataAnalyzer(file_path)
    columns = analyzer.get_columns()

    print("\n" + "=" * 60)
    print(f"Columns in: {file_path}")
    print("=" * 60)
    print(f"Total columns: {len(columns)}\n")

    for i, col in enumerate(columns, 1):
        print(f"  {i:3}. {col}")

    print("=" * 60)

    return columns


def list_distinct_values(
    file_path: str,
    column: str,
    show_counts: bool = False,
    output_file: Optional[str] = None,
) -> List[Any]:
    """
    List all distinct values for a specified column.

    Args:
        file_path: Path to the TSV or CSV file
        column: Column name to get distinct values for
        show_counts: If True, also show the count for each value
        output_file: Optional file path to save results (TSV format)

    Returns:
        List of distinct values
    """
    analyzer = InputDataAnalyzer(file_path)

    if show_counts:
        value_counts = analyzer.get_value_counts(column)
        distinct_values = list(value_counts.keys())

        print("\n" + "=" * 60)
        print(f"Distinct values for column: '{column}'")
        print(f"File: {file_path}")
        print("=" * 60)
        print(f"Total distinct values: {len(distinct_values)}\n")

        # Sort by count descending
        sorted_items = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)

        for value, count in sorted_items:
            display_value = "<NULL>" if pd.isna(value) else value
            print(f"  {display_value}: {count}")

        print("=" * 60)

        # Export to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            export_df = pd.DataFrame(
                [(v, c) for v, c in sorted_items],
                columns=[column, "count"],
            )
            export_df.to_csv(output_path, sep="\t", index=False)
            print(f"\n✓ Results exported to: {output_path}")

        return distinct_values
    else:
        distinct_values = analyzer.get_distinct_values(column)

        print("\n" + "=" * 60)
        print(f"Distinct values for column: '{column}'")
        print(f"File: {file_path}")
        print("=" * 60)
        print(f"Total distinct values: {len(distinct_values)}\n")

        for i, value in enumerate(distinct_values, 1):
            display_value = "<NULL>" if pd.isna(value) else value
            print(f"  {i:3}. {display_value}")

        print("=" * 60)

        # Export to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            export_df = pd.DataFrame({column: distinct_values})
            export_df.to_csv(output_path, sep="\t", index=False)
            print(f"\n✓ Results exported to: {output_path}")

        return distinct_values


def column_statistics(file_path: str, column: str) -> Dict[str, Any]:
    """
    Get statistics for a specified column.

    Args:
        file_path: Path to the TSV or CSV file
        column: Column name to analyze

    Returns:
        Dictionary with column statistics
    """
    analyzer = InputDataAnalyzer(file_path)
    stats = analyzer.get_column_statistics(column)

    print("\n" + "=" * 60)
    print(f"Statistics for column: '{column}'")
    print(f"File: {file_path}")
    print("=" * 60)

    print(f"  Data type:       {stats['dtype']}")
    print(f"  Total rows:      {stats['total_rows']}")
    print(f"  Non-null count:  {stats['non_null_count']}")
    print(f"  Null count:      {stats['null_count']}")
    print(f"  Distinct values: {stats['distinct_count']}")

    if "numeric_stats" in stats:
        print("\n  Numeric Statistics:")
        ns = stats["numeric_stats"]
        print(f"    Mean:   {ns['mean']:.4f}")
        print(f"    Median: {ns['median']:.4f}")
        print(f"    Min:    {ns['min']:.4f}")
        print(f"    Max:    {ns['max']:.4f}")
        print(f"    Std:    {ns['std']:.4f}")

    print("=" * 60)

    return stats


def data_summary(file_path: str) -> Dict[str, Any]:
    """
    Get a summary of the input data file.

    Args:
        file_path: Path to the TSV or CSV file

    Returns:
        Dictionary with data summary
    """
    analyzer = InputDataAnalyzer(file_path)

    if analyzer.df is None:
        raise ValueError("Data not loaded")

    summary: Dict[str, Any] = {
        "file_path": str(analyzer.file_path.absolute()),
        "rows": len(analyzer.df),
        "columns": len(analyzer.df.columns),
        "column_info": {},
    }

    print("\n" + "=" * 70)
    print(f"Data Summary: {file_path}")
    print("=" * 70)
    print(f"  Total rows:    {summary['rows']}")
    print(f"  Total columns: {summary['columns']}")
    print("\n  Column Details:")
    print("  " + "-" * 66)
    print(f"  {'Column':<30} {'Non-Null':<12} {'Distinct':<12} {'Type':<12}")
    print("  " + "-" * 66)

    for col in analyzer.df.columns:
        non_null = analyzer.df[col].notna().sum()
        distinct = analyzer.df[col].nunique(dropna=True)
        dtype = str(analyzer.df[col].dtype)

        summary["column_info"][col] = {
            "non_null": int(non_null),
            "distinct": int(distinct),
            "dtype": dtype,
        }

        col_display = col[:28] + ".." if len(col) > 30 else col
        print(f"  {col_display:<30} {non_null:<12} {distinct:<12} {dtype:<12}")

    print("  " + "-" * 66)
    print("=" * 70)

    return summary


def clear_column_values(
    file_path: str,
    column: str,
    output_file: Optional[str] = None,
    backup: bool = True,
) -> Dict[str, Any]:
    """
    Clear all values for a specified column, setting them to empty/null.

    This function reads the input file, clears all values in the specified column,
    and writes the result to an output file. By default, the original file is
    backed up before any changes.

    Args:
        file_path: Path to the TSV or CSV file
        column: Column name to clear values for
        output_file: Optional output file path. If not provided, overwrites input file.
        backup: If True and output_file is not provided, create a backup of the
                original file with .bak extension (default: True)

    Returns:
        Dictionary with operation statistics

    Example:
        # Clear user_id column, save to new file
        python seval_input_tools.py clear_column_values "data.tsv" --column="user_id" --output_file="data_cleared.tsv"

        # Clear user_id column in-place (with backup)
        python seval_input_tools.py clear_column_values "data.tsv" --column="user_id"

        # Clear user_id column in-place (no backup)
        python seval_input_tools.py clear_column_values "data.tsv" --column="user_id" --backup=False
    """
    input_path = Path(file_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    # Determine delimiter based on file extension
    suffix = input_path.suffix.lower()
    if suffix == ".tsv":
        delimiter = "\t"
    elif suffix == ".csv":
        delimiter = ","
    else:
        logger.warning(
            f"Unknown file extension '{suffix}', defaulting to tab delimiter"
        )
        delimiter = "\t"

    # Load data
    df = pd.read_csv(input_path, sep=delimiter)
    logger.info(
        f"Loaded data: {df.shape[0]} rows x {df.shape[1]} columns "
        f"from {input_path.absolute()}"
    )

    # Validate column exists
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. Available columns: {list(df.columns)}"
        )

    # Count non-null values before clearing
    non_null_before = df[column].notna().sum()
    distinct_before = df[column].nunique(dropna=True)

    # Clear the column values (set to empty string for string columns, NaN otherwise)
    df[column] = ""

    # Determine output path
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
        # Create backup if requested
        if backup:
            backup_path = input_path.with_suffix(input_path.suffix + ".bak")
            import shutil

            shutil.copy2(input_path, backup_path)
            logger.info(f"Created backup: {backup_path}")

    # Write output
    df.to_csv(output_path, sep=delimiter, index=False)

    stats: Dict[str, Any] = {
        "input_file": str(input_path.absolute()),
        "output_file": str(output_path.absolute()),
        "column_cleared": column,
        "total_rows": len(df),
        "values_cleared": int(non_null_before),
        "distinct_values_removed": int(distinct_before),
    }

    print("\n" + "=" * 60)
    print(f"Clear Column Values: '{column}'")
    print("=" * 60)
    print(f"  Input file:              {input_path}")
    print(f"  Output file:             {output_path}")
    print(f"  Total rows:              {stats['total_rows']}")
    print(f"  Values cleared:          {stats['values_cleared']}")
    print(f"  Distinct values removed: {stats['distinct_values_removed']}")
    print("=" * 60)
    print(f"✓ Column '{column}' cleared successfully!")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    # Create a dictionary of available functions
    functions = {
        "list_columns": list_columns,
        "list_distinct_values": list_distinct_values,
        "column_statistics": column_statistics,
        "data_summary": data_summary,
        "clear_column_values": clear_column_values,
    }

    fire.Fire(functions)
