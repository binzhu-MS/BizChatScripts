#!/usr/bin/env python3
"""
TSV Processing Suite
====================

This module provides comprehensive TSV file processing capabilities including:

1. TSV Transformation (process_file):
   - Remove the last column (Complexity_Reason)
   - Add a 'Segment' column after 'Utterance' with user-specified value (converted to lowercase)
   - Convert email_account by appending "@VertexEdgeLabs@SyntheticTenant"
   - Add a new "Source" column with value "synthetic" for each utterance
   - Supports append mode for incremental processing

2. LLM TSV to SEVAL Format Conversion (llm_tsv_to_seval_tsv):
   - Direct conversion from LLM output format to SEVAL format in one step
   - Input columns: Utterance, Email_Account, Query_Timestamp, Complexity_Reason (optional)
   - Output columns: Utterance, Segment, Environment, Grounding Data Source, user_id, timestamp
   - Transformations: Email_Account → user_id (+suffix), Query_Timestamp → timestamp
   - Adds metadata fields: Segment (from parameter), Environment="Test Tenant", Grounding Data Source="Web+Work"

3. TSV Cleanup Utility (cleanup_tsv):
   - Removes enclosing quotes from utterances and adjusts internal quotes
   - Identifies and replaces non-ASCII characters with ASCII equivalents
   - Validates that quoted content and non-ASCII characters only appear in utterance column
   - Reports all modifications and validation errors with immediate output during processing

4. TSV Splitting by Segment and Email (split_utterances):
   - Splits TSV files into separate files based on segment and email account
   - Groups utterances by segment (e.g., 'email', 'file') and email prefix
   - Creates separate files with naming pattern: <segment>_<email_prefix>_utterances.tsv
   - Provides detailed statistics about splitting process

5. TSV Column Filtering (filter_by_column, analyze_column):
   - SQL-like filtering: SELECT * FROM table WHERE column_name = 'value'
   - analyze_column: Analyze value distribution in any column
   - filter_by_column: Filter rows where column matches specified value
   - Case-insensitive column and value matching
   - Automatic output file naming or custom paths

6. TSV Field Validation (validate_file, validate_specific_fields):
   - Validates all fields in TSV files for completeness
   - Identifies rows with missing or empty fields
   - Generates detailed validation reports
   - Supports custom field validation rules
   - Provides statistics on field coverage
   - Exports validation results to various formats

7. TSV Duplicate Detection and Removal (check_duplicates, remove_duplicates):
   - Detects duplicate utterances in TSV files
   - Checks if entire rows are identical (all fields match)
   - Reports all duplicate occurrences with row numbers
   - Optionally removes duplicate rows, keeping only the first occurrence
   - Generates detailed duplicate analysis reports

Usage:
    # Transform TSV file
    python tsv_processor.py process_file --input_file="utterances_curr.tsv" --output_file="utterances_processed.tsv" --segment="email"
    python tsv_processor.py process_file --append_mode=True  # Use append-mode defaults

    # Convert LLM output directly to SEVAL format
    python tsv_processor.py llm_tsv_to_seval_tsv --input_file="utterances_curr.tsv" --output_file="utterances_seval.tsv" --segment="email"
    python tsv_processor.py llm_tsv_to_seval_tsv --append_mode=True  # Use results/ path defaults

    # Clean up TSV file
    python tsv_processor.py cleanup_tsv --input_file="utterances_seval.tsv" --output_file="utterances_cleaned.tsv"

    # Split TSV file by segment and email
    python tsv_processor.py split_utterances
    python tsv_processor.py split_utterances --input_file="results/utterances_all.tsv" --output_dir="results/split_results"

    # Analyze column values
    python tsv_processor.py analyze_column Segment --input_file="results/utterances_seval_timestamped.tsv"

    # Filter by column value
    python tsv_processor.py filter_by_column Segment email --input_file="results/utterances_seval_timestamped.tsv"
    python tsv_processor.py filter_by_column Segment email --output_file="results/utterances_email_only.tsv" --show_stats=True

    # Validate TSV file (full details)
    python tsv_processor.py validate_file "results/utterances_seval_timestamped.tsv" --show_details=True --max_issues=20
    python tsv_processor.py validate_file "file.tsv" --output_report="validation_report.json" --show_details=True --max_issues=10

    # Validate TSV file (summary only, no details)
    python tsv_processor.py validate_file "results/utterances_seval_timestamped.tsv"

    # Validate specific fields
    python tsv_processor.py validate_specific_fields "file.tsv" "Utterance,Segment,user_id"

    # Check for duplicate utterances (report only)
    python tsv_processor.py check_duplicates "results/utterances_seval_timestamped.tsv"
    python tsv_processor.py check_duplicates "file.tsv" --show_details=True --max_duplicates=20

    # Remove duplicate utterances (keep first occurrence)
    python tsv_processor.py remove_duplicates "file.tsv" "file_no_duplicates.tsv"
    python tsv_processor.py remove_duplicates "file.tsv" --output_file="cleaned.tsv" --show_stats=True
"""

import csv
import os
import sys
import json
import logging
import fire
import re
import unicodedata
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# Data classes for validation results
@dataclass
class FieldIssue:
    """Represents a field validation issue"""

    row_number: int
    field_name: str
    field_value: str
    issue_type: str
    severity: str


@dataclass
class ValidationResult:
    """Represents the complete validation result"""

    file_path: str
    total_rows: int
    total_fields: int
    field_names: List[str]
    issues: List[FieldIssue]
    field_coverage: Dict[str, Dict[str, int]]
    validation_timestamp: str
    is_valid: bool


@dataclass
class DuplicateGroup:
    """Represents a group of duplicate rows"""

    utterance: str
    row_numbers: List[int]
    is_fully_identical: bool
    first_occurrence: int
    duplicate_count: int
    sample_row: Dict[str, str]


@dataclass
class DuplicateResult:
    """Represents the complete duplicate detection result"""

    file_path: str
    total_rows: int
    unique_utterances: int
    duplicate_utterances: int
    duplicate_groups: List[DuplicateGroup]
    fully_identical_rows: int
    partial_duplicates: int
    analysis_timestamp: str


class TSVProcessor:
    """
    Comprehensive TSV file processor for transformation, SEVAL conversion, cleanup, splitting, filtering, validation, and duplicate detection.

    This class provides seven main functionalities:

    1. TSV Transformation (process_file): Transform utterance TSV files with column manipulations and email account modifications
    2. LLM to SEVAL Conversion (llm_tsv_to_seval_tsv): Direct conversion from LLM output format to SEVAL evaluation format
    3. TSV Cleanup (cleanup_tsv): Remove quotes and normalize non-ASCII characters in TSV files
    4. TSV Splitting (split_utterances): Split TSV files by segment and email account into separate files
    5. TSV Column Filtering (filter_by_column, analyze_column): SQL-like filtering and column analysis
    6. TSV Field Validation (validate_file, validate_specific_fields): Validate TSV files for missing/empty fields
    7. TSV Duplicate Detection (check_duplicates, remove_duplicates): Detect and remove duplicate utterances
    """

    def __init__(self):
        """Initialize the TSV processor"""
        # Validation rules for field validation
        self.validation_rules = {
            "empty": self._check_empty,
            "whitespace_only": self._check_whitespace_only,
            "null_values": self._check_null_values,
        }

    # ============================================
    # PART 1: TSV TRANSFORMATION
    # ============================================

    def _validate_input_file(self, input_file: str) -> None:
        """Validate that the input file exists and is readable

        Args:
            input_file: Path to the input TSV file

        Raises:
            FileNotFoundError: If the input file doesn't exist
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"✗ Error: Input file '{input_file}' not found! "
                f"Please ensure the file exists in the specified location."
            )

        logger.info(f"✓ Input file validated: {input_file}")

    def _process_header_row(self, header: List[str]) -> List[str]:
        """Process the header row according to transformation requirements

        Args:
            header: Original header row from TSV file

        Returns:
            List[str]: Transformed header row

        Raises:
            ValueError: If header doesn't have enough columns
        """
        if len(header) < 2:
            raise ValueError(
                f"✗ Error: Header row must have at least 2 columns, got: {header}"
            )

        # Remove last column and add Segment and Source columns
        # Expected format: [Utterance, Email_Account, ...] -> [Utterance, Segment, Email_Account, Source]
        new_header = [header[0], "Segment", header[1], "Source"]
        logger.info(f"📝 Header transformed: {header} -> {new_header}")
        return new_header

    def _process_data_row(
        self, row: List[str], empty_row_counter: dict, segment: str = "email"
    ) -> Optional[List[str]]:
        """Process a single data row according to transformation requirements

        Args:
            row: Original data row from TSV file
            empty_row_counter: Dictionary to track empty row counts
            segment: The segment value to use (will be converted to lowercase)

        Returns:
            Optional[List[str]]: Transformed row, or None if row is invalid
        """
        # Skip completely empty rows
        if not row:
            empty_row_counter["empty"] += 1
            return None

        # Skip rows with only whitespace (but be careful about empty lists)
        if all(cell.strip() == "" for cell in row if cell is not None):
            empty_row_counter["whitespace"] += 1
            return None

        if len(row) < 2:  # Ensure we have at least Utterance and Email_Account
            logger.warning(f"⚠️ Skipping invalid row (insufficient columns): {row}")
            return None

        try:
            utterance = row[0].strip() if row[0] is not None else ""
            email_account = row[1].strip() if row[1] is not None else ""
        except IndexError as e:
            logger.warning(f"⚠️ Skipping row with index error: {row}, error: {e}")
            return None

        # Skip rows where utterance or email_account is empty after stripping
        if not utterance or not email_account:
            empty_row_counter["empty_content"] += 1
            return None

        # Convert email account by appending the specified suffix
        modified_email = email_account + "@VertexEdgeLabs@SyntheticTenant"

        # Create new row: Utterance, Segment (lowercase), Modified Email, Source="synthetic"
        segment_value = segment.lower().strip()
        new_row = [utterance, segment_value, modified_email, "synthetic"]
        return new_row

    def process_tsv_file(
        self, input_file: str, output_file: str, segment: str = "email"
    ) -> None:
        """Process the TSV file and append to output file, creating header if file is empty

        Args:
            input_file: Path to the input TSV file
            output_file: Path to the output TSV file (will append or create with header)
            segment: The segment value to use for all rows (will be converted to lowercase)

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If file contains invalid data
        """
        logger.info(f"🚀 Starting TSV processing with append mode...")
        logger.info(f"📁 Input file: {input_file}")
        logger.info(f"📁 Output file: {output_file}")

        # Validate input file
        self._validate_input_file(input_file)

        processed_rows = []
        skipped_rows = 0
        empty_row_counter = {"empty": 0, "whitespace": 0, "empty_content": 0}
        output_file_exists = (
            os.path.exists(output_file) and os.path.getsize(output_file) > 0
        )

        try:
            # Read the input TSV file
            with open(input_file, "r", encoding="utf-8") as infile:
                reader = csv.reader(infile, delimiter="\t")

                # Process first row - determine if it's a header or data
                try:
                    first_row = next(reader)

                    # Check if first row looks like a header (contains expected header terms)
                    is_header_row = len(first_row) >= 2 and (
                        "Utterance" in first_row[0]
                        or "Email_Account"
                        in (first_row[1] if len(first_row) > 1 else "")
                        or "Complexity_Reason"
                        in (first_row[2] if len(first_row) > 2 else "")
                    )

                    if is_header_row:
                        # It's a proper header, process it
                        new_header = self._process_header_row(first_row)

                        # Only add header if output file doesn't exist or is empty
                        if not output_file_exists:
                            processed_rows.append(new_header)
                            logger.info(
                                f"📝 Adding header to new/empty file: {new_header}"
                            )
                        else:
                            logger.info(
                                f"📝 File exists, skipping header. Will append data only."
                            )
                    else:
                        # It's actually data, not a header - process as data and add our own header
                        logger.info(
                            f"📝 First row appears to be data, not header. Processing as data row."
                        )

                        # Add header if output file doesn't exist
                        if not output_file_exists:
                            default_header = [
                                "Utterance",
                                "Segment",
                                "Email_Account",
                                "Source",
                            ]
                            processed_rows.append(default_header)
                            logger.info(
                                f"📝 Adding default header to new file: {default_header}"
                            )

                        # Process the first row as data
                        processed_row = self._process_data_row(
                            first_row, empty_row_counter, segment
                        )
                        if processed_row:
                            processed_rows.append(processed_row)
                        else:
                            skipped_rows += 1

                except StopIteration:
                    raise ValueError(f"✗ Error: Input file '{input_file}' is empty!")

                # Process data rows
                for row_num, row in enumerate(
                    reader, start=2
                ):  # Start at 2 (after header)
                    processed_row = self._process_data_row(
                        row, empty_row_counter, segment
                    )
                    if processed_row:
                        processed_rows.append(processed_row)
                    else:
                        skipped_rows += 1

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"📁 Created output directory: {output_dir}")

            # Append the processed data to output file
            mode = "a" if output_file_exists else "w"
            logger.info(
                f"📁 Writing to file with mode: {mode}, rows to write: {len(processed_rows)}"
            )
            with open(output_file, mode, encoding="utf-8", newline="") as outfile:
                writer = csv.writer(outfile, delimiter="\t")
                writer.writerows(processed_rows)

            # Calculate statistics
            data_rows_processed = len(processed_rows) - (
                0 if output_file_exists else 1
            )  # Exclude header if added
            total_input_rows = data_rows_processed + skipped_rows

            logger.info(f"✅ Processing complete!")
            logger.info(f"📊 Input rows processed: {total_input_rows}")
            logger.info(f"📊 Data rows transformed: {data_rows_processed}")
            logger.info(f"📊 Rows skipped: {skipped_rows}")

            # Log empty row counts
            total_empty_rows = sum(empty_row_counter.values())
            if total_empty_rows > 0:
                logger.info(f"📊 Empty rows skipped: {total_empty_rows}")

            logger.info(
                f"💾 Output {'appended to' if output_file_exists else 'saved to'}: {output_file}"
            )

            # Don't return anything to avoid Fire printing the result
            return None

        except Exception as e:
            logger.error(f"✗ Error processing file: {e}")
            raise

    def llm_tsv_to_seval_tsv(
        self,
        input_file: str = "utterances_curr.tsv",
        output_file: str = "utterances_seval.tsv",
        segment: str = "email",
        append_mode: bool = False,
    ) -> None:
        """Process LLM output TSV file directly to SEVAL format

        Converts from LLM output format to SEVAL format in one step:
        - Input columns: Utterance, Email_Account, Query_Timestamp, Complexity_Reason (optional)
        - Output columns: Utterance, Segment, Environment, Grounding Data Source, user_id, timestamp

        Transformations:
        - Email_Account → user_id (with @VertexEdgeLabs@SyntheticTenant suffix)
        - Query_Timestamp → timestamp
        - Adds fixed values: Segment (from parameter), Environment="Test Tenant", Grounding Data Source="Web+Work"

        Args:
            input_file: Path to the input TSV file (default: "utterances_curr.tsv")
            output_file: Path to the output TSV file (default: "utterances_seval.tsv")
            segment: The segment value to use for all rows (default: "email", will be converted to lowercase)
            append_mode: If True, use results/ path defaults AND append to output file if it exists (no header written)
        """
        # Set default paths based on append_mode
        if append_mode and input_file == "utterances_curr.tsv":
            input_file = "results/utterances_curr.tsv"
        if append_mode and output_file == "utterances_seval.tsv":
            output_file = "results/utterances_seval.tsv"

        logger.info("🚀 Starting LLM TSV to SEVAL conversion...")
        logger.info(f"📁 Input file: {input_file}")
        logger.info(f"📄 Output file: {output_file}")
        logger.info(f"📋 Segment: {segment}")

        try:
            # Validate input file exists
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            # Read input TSV
            with open(input_file, "r", encoding="utf-8") as infile:
                reader = csv.DictReader(infile, delimiter="\t")

                # Validate required columns
                required_columns = ["Utterance", "Email_Account", "Query_Timestamp"]
                fieldnames = reader.fieldnames or []
                if not all(col in fieldnames for col in required_columns):
                    missing = [col for col in required_columns if col not in fieldnames]
                    raise ValueError(
                        f"Missing required columns in input file: {missing}"
                    )

                # Process rows
                seval_rows = []
                for row_num, row in enumerate(reader, start=1):
                    utterance = row.get("Utterance", "").strip()
                    email_account = row.get("Email_Account", "").strip()
                    query_timestamp = row.get("Query_Timestamp", "").strip()

                    # Skip empty rows
                    if not utterance or not email_account:
                        logger.warning(
                            f"⚠️ Skipping row {row_num}: empty utterance or email_account"
                        )
                        continue

                    # Transform to SEVAL format
                    seval_row = {
                        "Utterance": utterance,
                        "Segment": segment.lower().strip(),
                        "Environment": "Test Tenant",
                        "Grounding Data Source": "Web+Work",
                        "user_id": email_account + "@VertexEdgeLabs@SyntheticTenant",
                        "timestamp": query_timestamp,
                    }
                    seval_rows.append(seval_row)

            logger.info(f"✅ Processed {len(seval_rows)} rows")

            # Create output directory if needed
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Write SEVAL TSV
            seval_fieldnames = [
                "Utterance",
                "Segment",
                "Environment",
                "Grounding Data Source",
                "user_id",
                "timestamp",
            ]

            # Check if output file exists and has content for append mode
            output_file_exists = (
                os.path.exists(output_file) and os.path.getsize(output_file) > 0
            )
            write_header = not output_file_exists

            # Determine file mode: append if file exists and append_mode is True, otherwise write
            file_mode = "a" if (append_mode and output_file_exists) else "w"

            logger.info(
                f"📁 Writing to file with mode: {file_mode}, write_header: {write_header}"
            )

            with open(output_file, file_mode, newline="", encoding="utf-8") as outfile:
                writer = csv.DictWriter(
                    outfile, fieldnames=seval_fieldnames, delimiter="\t"
                )
                if write_header:
                    writer.writeheader()
                writer.writerows(seval_rows)

            logger.info(
                f"💾 Output {'appended to' if (append_mode and output_file_exists) else 'saved to'}: {output_file}"
            )
            logger.info("🎉 LLM TSV to SEVAL conversion complete!")
            logger.info("=" * 50)
            logger.info(f"📊 Total rows converted: {len(seval_rows)}")

        except Exception as e:
            logger.error(f"❌ Failed to process file: {e}")
            raise

        # Don't return anything to avoid Fire printing the result
        return None

    # ============================================
    # PART 2: TSV CLEANUP UTILITY
    # ============================================

    def _normalize_non_ascii_chars(self, text: str) -> Tuple[str, List[str]]:
        """
        Replace non-ASCII characters with ASCII equivalents using direct character mapping only.

        Args:
            text: Input text that may contain non-ASCII characters

        Returns:
            Tuple of (normalized_text, list_of_replacements_made)
        """
        replacements = []
        result = text

        # non-ASCII character replacement table
        char_replacements = {
            # Quotes (using Unicode escape codes to ensure correctness)
            "\u201c": '"',  # " → " - Left double quotation mark (ord 8220)
            "\u201d": '"',  # " → " - Right double quotation mark (ord 8221)
            "\u2018": "'",  # ' → ' - Left single quotation mark (ord 8216)
            "\u2019": "'",  # ' → ' - Right single quotation mark (ord 8217)
            "\u201a": "'",  # ‚ → ' - Single low-9 quotation mark (ord 8218)
            "\u201b": "'",  # ‛ → ' - Single high-reversed-9 quotation mark (ord 8219)
            "\u201e": '"',  # „ → " - Double low-9 quotation mark (ord 8222)
            "\u201f": '"',  # ‟ → " - Double high-reversed-9 quotation mark (ord 8223)
            "\u2039": "'",  # ‹ → ' - Single left-pointing angle quotation mark (ord 8249)
            "\u203a": "'",  # › → ' - Single right-pointing angle quotation mark (ord 8250)
            "\u00ab": '"',  # « → " - Left-pointing double angle quotation mark (ord 171)
            "\u00bb": '"',  # » → " - Right-pointing double angle quotation mark (ord 187)
            # Dashes and hyphens
            "\u2013": "-",  # – → - - En dash (ord 8211)
            "\u2014": "-",  # — → - - Em dash (ord 8212)
            "\u2011": "-",  # ‑ → - - Non-breaking hyphen (ord 8209)
            "\u2012": "-",  # ‒ → - - Figure dash (ord 8210)
            "\u2015": "-",  # ― → - - Horizontal bar (ord 8213)
            "\u2212": "-",  # − → - - Minus sign (ord 8722)
            # Spaces
            "\u00a0": " ",  # non-breaking space → space (ord 160)
            "\u2002": " ",  # en space → space (ord 8194)
            "\u2003": " ",  # em space → space (ord 8195)
            "\u2009": " ",  # thin space → space (ord 8201)
            "\u200a": " ",  # hair space → space (ord 8202)
            "\u202f": " ",  # narrow no-break space → space (ord 8239)
            # Ellipsis
            "\u2026": "...",  # … → ... - Horizontal ellipsis (ord 8230)
            # Bullets and symbols
            "\u2022": "*",  # • → * - Bullet (ord 8226)
            "\u2023": ">",  # ‣ → > - Triangular bullet (ord 8227)
            "\u2024": ".",  # ․ → . - One dot leader (ord 8228)
            "\u2025": "..",  # ‥ → .. - Two dot leader (ord 8229)
            "\u00b7": "*",  # · → * - Middle dot (ord 183)
        }

        # Apply direct replacements only
        for non_ascii, ascii_replacement in char_replacements.items():
            if non_ascii in result:
                result = result.replace(non_ascii, ascii_replacement)
                replacements.append(f"'{non_ascii}' → '{ascii_replacement}'")

        return result, replacements

    def _clean_quoted_utterance(self, utterance: str) -> Tuple[str, bool]:
        """
        Remove enclosing quotes from utterances and adjust internal quotes.

        Args:
            utterance: The utterance text to clean

        Returns:
            Tuple of (cleaned_utterance, was_modified)
        """
        original = utterance.strip()

        # Check if the utterance is enclosed in quotes
        if len(original) >= 2 and original.startswith('"') and original.endswith('"'):
            # Remove the enclosing quotes
            content = original[1:-1]

            # Replace internal double quotes with single quotes
            # Handle escaped quotes properly
            content = content.replace('""', '"')  # Fix double-escaped quotes first

            return content, True

        return original, False

    def _detect_non_ascii_chars(self, text: str) -> List[str]:
        """
        Detect non-ASCII characters in text.

        Args:
            text: Text to analyze

        Returns:
            List of non-ASCII characters found
        """
        non_ascii_chars = []
        for char in text:
            if ord(char) > 127:
                if char not in non_ascii_chars:
                    non_ascii_chars.append(char)
        return non_ascii_chars

    def _get_unreplaced_non_ascii_chars(self, text: str) -> List[Dict[str, str]]:
        """
        Get non-ASCII characters that are not in the char_replacements mapping.

        Args:
            text: Text to analyze

        Returns:
            List of dictionaries with character info for unreplaced non-ASCII chars
        """
        char_replacements = {
            "\u201c": '"',  # " → " - Left double quotation mark (ord 8220)
            "\u201d": '"',  # " → " - Right double quotation mark (ord 8221)
            "\u2018": "'",  # ' → ' - Left single quotation mark (ord 8216)
            "\u2019": "'",  # ' → ' - Right single quotation mark (ord 8217)
            "\u2013": "-",  # – → - - En dash (ord 8211)
            "\u2014": "-",  # — → - - Em dash (ord 8212)
            "\u2011": "-",  # ‑ → - - Non-breaking hyphen (ord 8209)
        }

        unreplaced_chars = []
        for char in text:
            if not char.isascii() and char not in char_replacements:
                # Check if we already have this character
                if not any(c["char"] == char for c in unreplaced_chars):
                    unreplaced_chars.append(
                        {
                            "char": char,
                            "unicode_name": unicodedata.name(
                                char, f"U+{ord(char):04X}"
                            ),
                            "ord": ord(char),
                            "hex": f"U+{ord(char):04X}",
                        }
                    )
        return unreplaced_chars

    def cleanup_tsv(
        self,
        input_file: str,
        output_file: str,
        max_items: Optional[int] = None,
    ) -> None:
        """
        Clean up TSV file by removing quotes and normalizing non-ASCII characters.

        This function:
        1. Removes enclosing quotes from utterances and adjusts internal quotes
        2. Identifies and replaces non-ASCII characters with ASCII equivalents
        3. Validates that quoted content and non-ASCII chars only appear in utterance column
        4. Reports all modifications and validation errors

        Args:
            input_file: Path to the input TSV file to clean
            output_file: Path to the cleaned output TSV file
            max_items: Maximum number of items to process (None for all)
        """
        logger.info("🧹 Starting TSV cleanup process...")
        logger.info(f"📁 Input file: {input_file}")
        logger.info(f"📄 Output file: {output_file}")
        if max_items:
            logger.info(f"🔢 Max items: {max_items}")

        try:
            # Load input TSV file
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            rows = []
            errors = []

            # Detailed statistics tracking
            stats = {
                "total_rows": 0,
                "rows_with_changes": 0,
                "quote_removals": 0,
                "non_ascii_normalizations": 0,
                "validation_errors": 0,
                "character_replacements": {},  # Track specific character replacements with row numbers
                "rows_with_quotes": set(),  # Track which rows had quotes
                "rows_with_non_ascii": set(),  # Track which rows had non-ASCII chars
                "unreplaced_utterances": [],  # Track utterances with unreplaced non-ASCII chars
            }

            # Load TSV file manually to preserve quotes
            with open(input_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # First line is headers
            header_line = lines[0].strip()
            fieldnames = header_line.split("\t")

            if not fieldnames:
                raise ValueError("No columns found in TSV file")

            logger.info(f"📊 Found columns: {fieldnames}")

            # Process each data line manually to preserve quotes
            for row_num, line in enumerate(lines[1:], 1):
                if max_items and row_num > max_items:
                    break

                stats["total_rows"] += 1
                cleaned_row = {}
                row_modified = False
                row_errors = []

                line = line.rstrip("\n\r")  # Remove line endings only
                fields = line.split("\t")

                # Pad fields if line has fewer than expected columns
                while len(fields) < len(fieldnames):
                    fields.append("")

                # Create row dictionary
                row = {}
                for i, header in enumerate(fieldnames):
                    row[header] = fields[i] if i < len(fields) else ""

                for field_name, field_value in row.items():
                    if field_value is None:
                        field_value = ""

                    # Check for non-ASCII characters in all fields
                    non_ascii_chars = self._detect_non_ascii_chars(field_value)

                    if field_name.lower() == "utterance":
                        # Handle utterance column - both quote cleaning and non-ASCII normalization

                        # Step 1: First normalize non-ASCII characters (including smart quotes)
                        cleaned_value = field_value
                        if non_ascii_chars:
                            stats["rows_with_non_ascii"].add(row_num)

                            # Apply only direct character replacements (no Unicode normalization)
                            normalized_value, replacements = (
                                self._normalize_non_ascii_chars(cleaned_value)
                            )

                            if replacements:
                                stats["non_ascii_normalizations"] += 1
                                # Track specific character replacements with row numbers
                                for replacement in replacements:
                                    if (
                                        replacement
                                        not in stats["character_replacements"]
                                    ):
                                        stats["character_replacements"][replacement] = {
                                            "count": 0,
                                            "rows": [],
                                        }
                                    stats["character_replacements"][replacement][
                                        "count"
                                    ] += 1
                                    stats["character_replacements"][replacement][
                                        "rows"
                                    ].append(row_num)

                                cleaned_value = normalized_value
                                row_modified = True

                            # Check for unreplaced non-ASCII characters
                            unreplaced_chars = self._get_unreplaced_non_ascii_chars(
                                cleaned_value
                            )
                            if unreplaced_chars:
                                # Store for final statistics (no real-time printing)
                                stats["unreplaced_utterances"].append(
                                    {
                                        "row": row_num,
                                        "utterance": cleaned_value,
                                        "unreplaced_chars": unreplaced_chars,
                                    }
                                )

                        # Step 2: Now clean quotes (after smart quotes have been converted to ASCII)
                        final_value, quote_modified = self._clean_quoted_utterance(
                            cleaned_value
                        )
                        if quote_modified:
                            row_modified = True
                            stats["quote_removals"] += 1
                            stats["rows_with_quotes"].add(row_num)
                            cleaned_value = final_value

                        cleaned_row[field_name] = cleaned_value

                    else:
                        # For non-utterance columns, flag errors if quotes or non-ASCII found

                        # Check for enclosing quotes (error condition)
                        field_value_stripped = field_value.strip()
                        if (
                            len(field_value_stripped) >= 2
                            and field_value_stripped.startswith('"')
                            and field_value_stripped.endswith('"')
                            and field_value_stripped.count('"') == 2
                        ):
                            row_errors.append(
                                f"Field '{field_name}' has enclosing quotes: {field_value}"
                            )

                        # Check for non-ASCII characters (error condition)
                        if non_ascii_chars:
                            row_errors.append(
                                f"Field '{field_name}' has non-ASCII characters {non_ascii_chars}: {field_value}"
                            )

                        # For non-utterance fields, use the original value (no cleaning)
                        cleaned_row[field_name] = field_value

                # Record any errors for this row
                if row_errors:
                    stats["validation_errors"] += len(row_errors)
                    errors.append(
                        {"row": row_num, "errors": row_errors, "data": dict(row)}
                    )

                # Track if this row had any changes
                if row_modified:
                    stats["rows_with_changes"] += 1

                rows.append(cleaned_row)

            # Create output directory if needed
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write cleaned TSV manually to preserve quote handling
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                # Write header
                f.write("\t".join(fieldnames) + "\n")

                # Write each row manually
                for row in rows:
                    row_values = []
                    for field_name in fieldnames:
                        value = row.get(field_name, "")
                        row_values.append(value)
                    f.write("\t".join(row_values) + "\n")

            # Show summary statistics
            print("\n" + "=" * 70)
            print("🎉 TSV CLEANUP COMPLETE!")
            print("=" * 70)
            print(f"📁 Input file: {input_file}")
            print(f"📄 Output file saved: {output_file}")
            print("\n📊 PROCESSING STATISTICS:")
            print(f"   • Total rows in file: {stats['total_rows']}")
            print(
                f"   • Rows with changes: {stats['rows_with_changes']} ({stats['rows_with_changes']/stats['total_rows']*100:.1f}%)"
            )
            print(
                f"   • Unchanged rows: {stats['total_rows'] - stats['rows_with_changes']}"
            )

            print("\n🔧 QUOTE PROCESSING:")
            print(f"   • Quote removals: {stats['quote_removals']}")
            if stats["rows_with_quotes"]:
                rows_list = sorted(list(stats["rows_with_quotes"]))
                if len(rows_list) <= 10:
                    row_display = f" (rows: {', '.join(map(str, rows_list))})"
                else:
                    first_10 = ", ".join(map(str, rows_list[:10]))
                    row_display = (
                        f" (rows: {first_10}, ... and {len(rows_list) - 10} more)"
                    )
                print(
                    f"   • Rows with quotes: {len(stats['rows_with_quotes'])}{row_display}"
                )
            else:
                print(f"   • Rows with quotes: 0")

            print("\n🌐 NON-ASCII CHARACTER PROCESSING:")
            print(f"   • Rows with replacements: {stats['non_ascii_normalizations']}")
            print(
                f"   • Rows with non-ASCII chars detected: {len(stats['rows_with_non_ascii'])}"
            )

            if stats["character_replacements"]:
                print("\n📈 CHARACTER REPLACEMENT BREAKDOWN:")
                sorted_replacements = sorted(
                    stats["character_replacements"].items(),
                    key=lambda x: x[1]["count"],
                    reverse=True,
                )
                for char_mapping, data in sorted_replacements:
                    count = data["count"]
                    rows = data["rows"]
                    # Show first 10 row numbers
                    if len(rows) <= 10:
                        row_display = f" (rows: {', '.join(map(str, rows))})"
                    else:
                        first_10 = ", ".join(map(str, rows[:10]))
                        row_display = (
                            f" (rows: {first_10}, ... and {len(rows) - 10} more)"
                        )
                    print(f"   • {char_mapping}: {count} replacement(s){row_display}")

            # Show unreplaced non-ASCII characters statistics
            if stats["unreplaced_utterances"]:
                print(f"\n⚠️  UNREPLACED NON-ASCII CHARACTERS SUMMARY:")
                print(
                    f"   • Total utterances with unreplaced characters: {len(stats['unreplaced_utterances'])}"
                )

                # Collect all unique unreplaced characters
                all_unreplaced_chars = {}
                for utterance_info in stats["unreplaced_utterances"]:
                    for char_info in utterance_info["unreplaced_chars"]:
                        char = char_info["char"]
                        if char not in all_unreplaced_chars:
                            all_unreplaced_chars[char] = {
                                "info": char_info,
                                "count": 0,
                                "rows": [],
                            }
                        all_unreplaced_chars[char]["count"] += 1
                        all_unreplaced_chars[char]["rows"].append(utterance_info["row"])

                print("   • Unique unreplaced characters found:")
                for char, data in sorted(
                    all_unreplaced_chars.items(),
                    key=lambda x: x[1]["count"],
                    reverse=True,
                ):
                    char_info = data["info"]
                    rows = data["rows"]
                    # Show first 10 row numbers
                    if len(rows) <= 10:
                        row_display = f"rows {rows}"
                    else:
                        first_10 = rows[:10]
                        row_display = f"rows {first_10} ... and {len(rows) - 10} more"
                    print(
                        f"     - '{char}' ({char_info['hex']}) - {char_info['unicode_name']}: {data['count']} occurrence(s) in {row_display}"
                    )
            else:
                print("\n✅ No unreplaced non-ASCII characters found")

            if errors:
                print(f"\n❌ Validation errors found: {len(errors)}")
                for error in errors:
                    print(f"   • Row {error['row']}:")
                    for err_msg in error["errors"]:
                        print(f"     - {err_msg}")
                    print(f"     Data: {error['data']}")
            else:
                print("✅ No validation errors found")

            print("=" * 70)

        except Exception as e:
            logger.error(f"❌ Critical error in TSV cleanup: {e}")
            raise

    # ============================================
    # PART 4: TSV SPLITTING BY SEGMENT AND EMAIL
    # ============================================

    def _extract_email_prefix(self, email_account: str) -> str:
        """Extract the email prefix (part before @) from email account

        Args:
            email_account: Full email account string

        Returns:
            str: Email prefix (part before first @)
        """
        # Split by @ and take the first part
        email_parts = email_account.split("@")
        return email_parts[0] if email_parts else email_account

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing/replacing invalid characters

        Args:
            filename: Original filename

        Returns:
            str: Sanitized filename safe for filesystem
        """
        # Replace common problematic characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")

        # Remove any leading/trailing dots or spaces
        filename = filename.strip(". ")

        return filename

    def _group_utterances_by_segment_and_email(
        self, input_file: str
    ) -> Tuple[List[str], Dict[str, List[List[str]]]]:
        """Group utterances by segment and email account

        Args:
            input_file: Path to the input TSV file

        Returns:
            tuple: (header, grouped_data) where grouped_data is keyed by "segment_email_prefix"
        """
        from collections import defaultdict

        grouped_data = defaultdict(list)
        header = None

        with open(input_file, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")

            # Read header
            header = next(reader)
            logger.info(f"📝 Header found: {header}")

            # Expected columns: Utterance, Segment, Email_Account, Source (or similar)
            if len(header) < 3:
                raise ValueError(
                    f"✗ Expected at least 3 columns (Utterance, Segment, Email_Account), got {len(header)}: {header}"
                )

            # Process data rows
            for row_num, row in enumerate(reader, start=2):
                if len(row) < 3:
                    logger.warning(
                        f"⚠️ Skipping row {row_num} (insufficient columns): {row}"
                    )
                    continue

                utterance = row[0].strip()
                segment = row[1].strip().lower()
                email_account = row[2].strip()

                # Skip empty rows
                if not utterance or not email_account:
                    logger.warning(
                        f"⚠️ Skipping row {row_num} (empty utterance or email): {row}"
                    )
                    continue

                # Extract email prefix
                email_prefix = self._extract_email_prefix(email_account)

                # Create group key: segment_email_prefix
                group_key = f"{segment}_{email_prefix}"

                # Add row to group
                grouped_data[group_key].append(row)

        return header, grouped_data

    def split_utterances(
        self,
        input_file: str = "results/utterances_all.tsv",
        output_dir: str = "results/split_results",
    ) -> None:
        """Split a TSV file containing utterances by segment and email account

        Creates separate TSV files for each unique combination of segment and email account.
        Each output file follows the naming pattern: <segment>_<email_prefix>_utterances.tsv

        Args:
            input_file: Path to the input TSV file (default: "results/utterances_all.tsv")
            output_dir: Directory to save split files (default: "results/split_results")

        Example:
            processor = TSVProcessor()
            processor.split_utterances(
                input_file="results/utterances_all.tsv",
                output_dir="results/split_results"
            )
        """
        logger.info(f"🚀 Starting TSV splitting process...")
        logger.info(f"📁 Input file: {input_file}")
        logger.info(f"📁 Output directory: {output_dir}")

        # Validate input file
        self._validate_input_file(input_file)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"📁 Created/verified output directory: {output_dir}")

        # Group utterances
        header, grouped_data = self._group_utterances_by_segment_and_email(input_file)

        if not grouped_data:
            logger.warning("⚠️ No data groups found to split!")
            return None

        # Write each group to separate files
        groups_created = 0
        total_utterances = 0
        file_stats = {}

        for group_key, rows in grouped_data.items():
            # Create filename: <segment>_<email_prefix>_utterances.tsv
            # Group key format: segment_email_prefix
            parts = group_key.split("_", 1)  # Split only on first underscore
            if len(parts) == 2:
                segment, email_prefix = parts
                sanitized_email_prefix = self._sanitize_filename(email_prefix)
                filename = f"{segment}_{sanitized_email_prefix}_utterances.tsv"
            else:
                # Fallback if parsing fails
                sanitized_group_key = self._sanitize_filename(group_key)
                filename = f"{sanitized_group_key}_utterances.tsv"

            output_path = os.path.join(output_dir, filename)

            # Write the group to file
            with open(output_path, "w", encoding="utf-8", newline="") as outfile:
                writer = csv.writer(outfile, delimiter="\t")

                # Write header
                writer.writerow(header)

                # Write data rows
                writer.writerows(rows)

            groups_created += 1
            group_utterance_count = len(rows)
            total_utterances += group_utterance_count
            file_stats[filename] = group_utterance_count

            logger.info(f"📄 Created: {filename} ({group_utterance_count} utterances)")

        # Log summary
        logger.info(f"✅ Splitting complete!")
        logger.info(f"📊 Groups created: {groups_created}")
        logger.info(f"📊 Total utterances: {total_utterances}")
        logger.info(f"📁 Files saved to: {output_dir}")

        # Log detailed file statistics
        logger.info(f"📋 File breakdown:")
        for filename, count in sorted(file_stats.items()):
            logger.info(f"  - {filename}: {count} utterances")

        # Don't return anything to avoid Fire printing the result
        return None

    # ============================================
    # PART 5: TSV COLUMN FILTERING (SQL-LIKE)
    # ============================================

    def _find_column_name(self, fieldnames: Any, target_column: str) -> Optional[str]:
        """Find the actual column name in fieldnames (case-insensitive)

        Args:
            fieldnames: List or sequence of column names from CSV header
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
    ) -> None:
        """Analyze all values in the specified column without filtering

        Provides statistics about the distribution of values in a specified column,
        similar to SQL SELECT column_name, COUNT(*) FROM table GROUP BY column_name.

        Args:
            column_name: Name of the column to analyze (case-insensitive)
            input_file: Path to the input TSV file
            encoding: File encoding to use (default: "utf-8")

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If column not found or file has no headers

        Example:
            processor = TSVProcessor()
            processor.analyze_column("Segment", "results/data.tsv")
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
            percentage = (count / total_rows) * 100 if total_rows > 0 else 0
            logger.info(f"  - {value}: {count:,} rows ({percentage:.1f}%)")

        # Don't return anything to avoid Fire printing the result
        return None

    def filter_by_column(
        self,
        column_name: str,
        column_value: str,
        input_file: str = "results/utterances_seval_timestamped.tsv",
        output_file: Optional[str] = None,
        show_stats: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        """Filter TSV file by any column name and value (SQL-like WHERE clause)

        Filters rows where the specified column matches the specified value,
        similar to SQL SELECT * FROM table WHERE column_name = 'column_value'.

        Args:
            column_name: The column name to filter by (case-insensitive)
            column_value: The value to match in the specified column (case-insensitive comparison)
            input_file: Path to the input TSV file
            output_file: Path to the output TSV file (auto-generated if None)
            show_stats: Whether to display filtering statistics (default: True)
            encoding: File encoding to use (default: "utf-8")

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If column not found or file has no headers

        Example:
            processor = TSVProcessor()
            processor.filter_by_column(
                "Segment", "email",
                input_file="results/utterances_seval_timestamped.tsv",
                output_file="results/utterances_email_only.tsv"
            )
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

        # Initialize statistics
        stats = {
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
                        stats["total_rows"] += 1

                        row_value = row.get(actual_column, "").strip().lower()
                        target_value = column_value.strip().lower()

                        if row_value == target_value:
                            writer.writerow(row)
                            stats["filtered_rows"] += 1
                        else:
                            if row_value:
                                stats["other_values"][row_value] = (
                                    stats["other_values"].get(row_value, 0) + 1
                                )

                    except Exception as e:
                        logger.warning(f"⚠️ Error processing row {row_num}: {str(e)}")
                        stats["processing_errors"] += 1

        except Exception as e:
            logger.error(f"❌ Failed to process files: {str(e)}")
            raise

        if show_stats:
            logger.info(
                f"\n📊 Column '{actual_column}' = '{column_value}' Filtering Statistics:"
            )
            logger.info(f"📄 Total rows processed: {stats['total_rows']:,}")
            logger.info(f"🎯 Matching rows found: {stats['filtered_rows']:,}")
            logger.info(
                f"🔢 Other rows: {stats['total_rows'] - stats['filtered_rows']:,}"
            )

        logger.info(f"✅ Filtering completed successfully!")
        logger.info(f"💾 Output saved to: {output_file}")

        # Don't return anything to avoid Fire printing the result
        return None

    # ============================================
    # PART 6: TSV FIELD VALIDATION
    # ============================================

    def _check_empty(self, value: str) -> bool:
        """Check if value is empty"""
        return value == ""

    def _check_whitespace_only(self, value: str) -> bool:
        """Check if value contains only whitespace"""
        return value.strip() == ""

    def _check_null_values(self, value: str) -> bool:
        """Check if value represents null/none"""
        null_values = {"null", "none", "n/a", "na", "-", "undefined", "nil"}
        return value.lower().strip() in null_values

    def _perform_validation(
        self,
        file_path: str,
        max_issues: Optional[int] = None,
        validation_rules: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Internal method to perform validation and return result object

        Args:
            file_path: Path to the TSV file to validate
            max_issues: Maximum number of issues to display/store
            validation_rules: List of validation rules to apply

        Returns:
            ValidationResult object containing validation details
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Set default validation rules
        if validation_rules is None:
            validation_rules = ["empty", "whitespace_only"]

        # Read and validate the TSV file
        issues = []
        field_names = []
        total_rows = 0
        field_coverage = {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                field_names = list(reader.fieldnames or [])

                # Initialize field coverage tracking
                for field_name in field_names:
                    field_coverage[field_name] = {
                        "total": 0,
                        "valid": 0,
                        "empty": 0,
                        "whitespace_only": 0,
                        "null_values": 0,
                    }

                # Process each row
                for row_num, row in enumerate(
                    reader, start=2
                ):  # Start at 2 (header is row 1)
                    total_rows += 1

                    # Check each field in the row
                    for field_name in field_names:
                        field_value = row.get(field_name, "")
                        field_coverage[field_name]["total"] += 1

                        # Apply validation rules
                        field_valid = True
                        for rule_name in validation_rules:
                            if rule_name in self.validation_rules:
                                rule_func = self.validation_rules[rule_name]
                                if rule_func(field_value):
                                    # Field failed this rule
                                    issue = FieldIssue(
                                        row_number=row_num,
                                        field_name=field_name,
                                        field_value=field_value,
                                        issue_type=rule_name,
                                        severity=(
                                            "warning"
                                            if rule_name == "whitespace_only"
                                            else "error"
                                        ),
                                    )
                                    issues.append(issue)
                                    field_coverage[field_name][rule_name] += 1
                                    field_valid = False

                        if field_valid:
                            field_coverage[field_name]["valid"] += 1

                        # Stop if we've reached max issues
                        if max_issues and len(issues) >= max_issues:
                            break

                    if max_issues and len(issues) >= max_issues:
                        logger.warning(
                            f"⚠️ Reached maximum issue limit ({max_issues}), stopping validation"
                        )
                        break

        except Exception as e:
            logger.error(f"❌ Error reading TSV file: {str(e)}")
            raise

        # Create validation result
        validation_result = ValidationResult(
            file_path=file_path,
            total_rows=total_rows,
            total_fields=len(field_names),
            field_names=field_names,
            issues=issues,
            field_coverage=field_coverage,
            validation_timestamp=datetime.now().isoformat(),
            is_valid=len(issues) == 0,
        )

        return validation_result

    def validate_file(
        self,
        file_path: str,
        output_report: Optional[str] = None,
        show_details: bool = False,
        max_issues: Optional[int] = None,
        validation_rules: Optional[List[str]] = None,
    ) -> None:
        """Validate a TSV file for missing fields

        Args:
            file_path: Path to the TSV file to validate
            output_report: Optional path to save validation report
            show_details: Whether to show detailed issue information
            max_issues: Maximum number of issues to display/store
            validation_rules: List of validation rules to apply
        """
        logger.info(f"🔍 Starting TSV field validation for: {file_path}")

        # Perform validation
        validation_result = self._perform_validation(
            file_path, max_issues, validation_rules
        )

        # Display results
        self._display_validation_results(validation_result, show_details, max_issues)

        # Save report if requested
        if output_report:
            self._save_validation_report(validation_result, output_report)

        # Don't return anything to avoid Fire printing the result
        return None

    def _display_validation_results(
        self, result: ValidationResult, show_details: bool, max_issues: Optional[int]
    ) -> None:
        """Display validation results to console"""
        logger.info(f"📊 Validation Results for: {result.file_path}")
        logger.info(f"📄 Total rows: {result.total_rows}")
        logger.info(f"📋 Total fields: {result.total_fields}")
        logger.info(f"🏷️ Field names: {', '.join(result.field_names)}")
        logger.info(f"❗ Total issues found: {len(result.issues)}")
        logger.info(f"✅ File is valid: {result.is_valid}")

        # Display field coverage summary
        logger.info(f"\n📈 Field Coverage Summary:")
        for field_name, coverage in result.field_coverage.items():
            total = coverage["total"]
            valid = coverage["valid"]
            empty = coverage["empty"]
            whitespace = coverage["whitespace_only"]
            null_vals = coverage["null_values"]

            logger.info(f"  📌 {field_name}:")
            logger.info(f"     ✅ Valid: {valid}/{total} ({(valid/total*100):.1f}%)")
            if empty > 0:
                logger.info(f"     🚫 Empty: {empty} ({(empty/total*100):.1f}%)")
            if whitespace > 0:
                logger.info(
                    f"     ⚪ Whitespace only: {whitespace} ({(whitespace/total*100):.1f}%)"
                )
            if null_vals > 0:
                logger.info(
                    f"     ❌ Null values: {null_vals} ({(null_vals/total*100):.1f}%)"
                )

        # Display detailed issues if requested
        if show_details and result.issues:
            logger.info(f"\n🔍 Detailed Issues (showing up to {max_issues or 'all'}):")
            display_issues = result.issues[:max_issues] if max_issues else result.issues

            for issue in display_issues:
                severity_icon = "🚨" if issue.severity == "error" else "⚠️"
                logger.info(
                    f"  {severity_icon} Row {issue.row_number}, Field '{issue.field_name}': "
                    f"{issue.issue_type} (Value: '{issue.field_value}')"
                )

            if max_issues and len(result.issues) > max_issues:
                logger.info(
                    f"     ... and {len(result.issues) - max_issues} more issues"
                )

        # Summary
        if result.is_valid:
            logger.info(f"\n🎉 All fields are properly filled! No issues detected.")
        else:
            error_count = sum(1 for issue in result.issues if issue.severity == "error")
            warning_count = len(result.issues) - error_count
            logger.info(f"\n⚠️ Found {error_count} errors and {warning_count} warnings")

    def _save_validation_report(
        self, result: ValidationResult, output_path: str
    ) -> None:
        """Save validation report to file"""
        try:
            # Convert result to dictionary for JSON serialization
            report_data = asdict(result)

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Validation report saved to: {output_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save validation report: {str(e)}")
            raise

    def validate_specific_fields(
        self,
        file_path: str,
        required_fields: str,
        output_report: Optional[str] = None,
    ) -> None:
        """Validate specific fields in a TSV file

        Args:
            file_path: Path to the TSV file to validate
            required_fields: Comma-separated list of required field names
            output_report: Optional path to save validation report
        """
        # Parse required fields from comma-separated string
        fields = [field.strip() for field in required_fields.split(",")]

        logger.info(f"🔍 Validating specific fields: {', '.join(fields)}")

        # First, do general validation
        result = self._perform_validation(
            file_path, max_issues=None, validation_rules=None
        )

        # Check if all required fields exist
        missing_fields = [field for field in fields if field not in result.field_names]
        if missing_fields:
            logger.error(f"❌ Missing required fields: {', '.join(missing_fields)}")

        # Filter issues to only required fields
        filtered_issues = [
            issue for issue in result.issues if issue.field_name in fields
        ]

        # Create new result with filtered issues
        filtered_result = ValidationResult(
            file_path=result.file_path,
            total_rows=result.total_rows,
            total_fields=len(fields),
            field_names=fields,
            issues=filtered_issues,
            field_coverage={
                field: result.field_coverage[field]
                for field in fields
                if field in result.field_coverage
            },
            validation_timestamp=result.validation_timestamp,
            is_valid=len(filtered_issues) == 0 and len(missing_fields) == 0,
        )

        # Display results
        self._display_validation_results(
            filtered_result, show_details=True, max_issues=None
        )

        # Save report if requested
        if output_report:
            self._save_validation_report(filtered_result, output_report)

        # Don't return anything to avoid Fire printing the result
        return None

    # ============================================
    # PART 7: TSV DUPLICATE DETECTION AND REMOVAL
    # ============================================

    def _analyze_duplicates(self, file_path: str) -> DuplicateResult:
        """Internal method to analyze duplicates and return result object

        Args:
            file_path: Path to the TSV file to check

        Returns:
            DuplicateResult object containing duplicate analysis details
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Track utterances and their row information
        utterance_tracker = {}  # utterance -> list of (row_num, row_dict)
        total_rows = 0
        field_names = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                field_names = list(reader.fieldnames or [])

                if not field_names:
                    raise ValueError("File has no headers")

                # Find the utterance column (case-insensitive)
                utterance_col = self._find_column_name(field_names, "Utterance")
                if not utterance_col:
                    raise ValueError(
                        f"No 'Utterance' column found. Available columns: {', '.join(field_names)}"
                    )

                for row_num, row in enumerate(
                    reader, start=2
                ):  # Start at 2 (after header)
                    total_rows += 1
                    utterance = row.get(utterance_col, "").strip()

                    if not utterance:
                        logger.warning(f"⚠️ Row {row_num}: Empty utterance, skipping")
                        continue

                    # Store row information
                    if utterance not in utterance_tracker:
                        utterance_tracker[utterance] = []
                    utterance_tracker[utterance].append((row_num, dict(row)))

        except Exception as e:
            logger.error(f"❌ Error reading file: {e}")
            raise

        # Analyze duplicates
        duplicate_groups = []
        fully_identical_count = 0
        partial_duplicate_count = 0

        for utterance, occurrences in utterance_tracker.items():
            if len(occurrences) > 1:
                # Found duplicate utterance
                row_numbers = [row_num for row_num, _ in occurrences]
                first_row_num, first_row_data = occurrences[0]

                # Check if all rows are fully identical
                is_fully_identical = True
                for row_num, row_data in occurrences[1:]:
                    if row_data != first_row_data:
                        is_fully_identical = False
                        break

                if is_fully_identical:
                    fully_identical_count += (
                        len(occurrences) - 1
                    )  # Don't count first occurrence
                else:
                    partial_duplicate_count += len(occurrences)

                duplicate_group = DuplicateGroup(
                    utterance=utterance,
                    row_numbers=row_numbers,
                    is_fully_identical=is_fully_identical,
                    first_occurrence=first_row_num,
                    duplicate_count=len(occurrences) - 1,
                    sample_row=first_row_data,
                )
                duplicate_groups.append(duplicate_group)

        # Sort by number of duplicates (descending)
        duplicate_groups.sort(key=lambda g: g.duplicate_count, reverse=True)

        # Create result
        result = DuplicateResult(
            file_path=file_path,
            total_rows=total_rows,
            unique_utterances=len(utterance_tracker) - len(duplicate_groups),
            duplicate_utterances=len(duplicate_groups),
            duplicate_groups=duplicate_groups,
            fully_identical_rows=fully_identical_count,
            partial_duplicates=partial_duplicate_count,
            analysis_timestamp=datetime.now().isoformat(),
        )

        return result

    def check_duplicates(
        self,
        file_path: str,
        show_details: bool = False,
        max_duplicates: Optional[int] = None,
        output_report: Optional[str] = None,
    ) -> None:
        """Check for duplicate utterances in a TSV file

        This method detects duplicate utterances and checks if entire rows are identical.
        - Duplicate utterances: Same utterance text in different rows
        - Fully identical rows: Same utterance AND all other fields are identical
        - Partial duplicates: Same utterance but different values in other fields

        Args:
            file_path: Path to the TSV file to check
            show_details: Whether to show detailed duplicate information
            max_duplicates: Maximum number of duplicate groups to display
            output_report: Optional path to save duplicate report (JSON format)
        """
        logger.info(f"🔍 Starting duplicate detection for: {file_path}")

        # Analyze duplicates
        result = self._analyze_duplicates(file_path)

        # Display results
        self._display_duplicate_results(result, show_details, max_duplicates)

        # Save report if requested
        if output_report:
            self._save_duplicate_report(result, output_report)

        # Don't return anything to avoid Fire printing the result
        return None

    def _display_duplicate_results(
        self, result: DuplicateResult, show_details: bool, max_duplicates: Optional[int]
    ) -> None:
        """Display duplicate detection results to console"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 DUPLICATE DETECTION RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"📁 File: {result.file_path}")
        logger.info(f"📄 Total rows: {result.total_rows}")
        logger.info(f"✅ Unique utterances: {result.unique_utterances}")
        logger.info(f"🔁 Duplicate utterance groups: {result.duplicate_utterances}")

        if result.duplicate_utterances > 0:
            total_duplicate_rows = sum(
                g.duplicate_count for g in result.duplicate_groups
            )
            logger.info(f"📊 Total duplicate rows: {total_duplicate_rows}")
            logger.info(f"🎯 Fully identical rows: {result.fully_identical_rows}")
            logger.info(
                f"⚠️  Partial duplicates (same utterance, different fields): {result.partial_duplicates}"
            )

            if show_details and result.duplicate_groups:
                logger.info(f"\n{'='*70}")
                logger.info(f"📋 DUPLICATE GROUPS DETAILS:")
                logger.info(f"{'='*70}")

                display_count = (
                    min(max_duplicates, len(result.duplicate_groups))
                    if max_duplicates
                    else len(result.duplicate_groups)
                )

                for idx, group in enumerate(result.duplicate_groups[:display_count], 1):
                    logger.info(f"\n🔁 Duplicate Group #{idx}:")
                    logger.info(
                        f"   Utterance: {group.utterance[:100]}{'...' if len(group.utterance) > 100 else ''}"
                    )
                    logger.info(f"   First occurrence: Row {group.first_occurrence}")
                    logger.info(f"   Duplicate count: {group.duplicate_count}")
                    logger.info(
                        f"   All occurrences: {', '.join(map(str, group.row_numbers))}"
                    )
                    logger.info(
                        f"   Status: {'✅ Fully identical' if group.is_fully_identical else '⚠️  Partial duplicate (fields differ)'}"
                    )

                    if not group.is_fully_identical:
                        logger.info(
                            f"   Note: Same utterance but different values in other fields"
                        )

                if len(result.duplicate_groups) > display_count:
                    logger.info(
                        f"\n... and {len(result.duplicate_groups) - display_count} more duplicate groups"
                    )
                    logger.info(
                        f"💡 Use --show_details=True --max_duplicates={len(result.duplicate_groups)} to see all"
                    )
        else:
            logger.info(f"\n✅ No duplicate utterances found!")

        logger.info(f"{'='*70}\n")

    def _save_duplicate_report(self, result: DuplicateResult, output_path: str) -> None:
        """Save duplicate detection report to JSON file"""
        try:
            report_data = {
                "file_path": result.file_path,
                "analysis_timestamp": result.analysis_timestamp,
                "summary": {
                    "total_rows": result.total_rows,
                    "unique_utterances": result.unique_utterances,
                    "duplicate_utterances": result.duplicate_utterances,
                    "fully_identical_rows": result.fully_identical_rows,
                    "partial_duplicates": result.partial_duplicates,
                },
                "duplicate_groups": [
                    {
                        "utterance": g.utterance,
                        "first_occurrence": g.first_occurrence,
                        "duplicate_count": g.duplicate_count,
                        "row_numbers": g.row_numbers,
                        "is_fully_identical": g.is_fully_identical,
                        "sample_row": g.sample_row,
                    }
                    for g in result.duplicate_groups
                ],
            }

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Duplicate report saved to: {output_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save duplicate report: {e}")
            raise

    def remove_duplicates(
        self,
        input_file: str,
        output_file: str,
        show_stats: bool = True,
    ) -> None:
        """Remove duplicate utterances from TSV file, keeping only the first occurrence

        This method removes rows with duplicate utterances, keeping only the first
        occurrence of each unique utterance. Only fully identical rows are removed.

        Args:
            input_file: Path to the input TSV file
            output_file: Path to the output TSV file (deduplicated)
            show_stats: Whether to display removal statistics
        """
        logger.info(f"🧹 Starting duplicate removal for: {input_file}")
        logger.info(f"📄 Output file: {output_file}")

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Track which rows to keep
        seen_utterances = {}  # utterance -> (row_num, row_data)
        rows_to_write = []
        removed_rows = []
        total_rows = 0
        field_names = []

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                field_names = list(reader.fieldnames or [])

                if not field_names:
                    raise ValueError("File has no headers")

                # Find the utterance column
                utterance_col = self._find_column_name(field_names, "Utterance")
                if not utterance_col:
                    raise ValueError(
                        f"No 'Utterance' column found. Available columns: {', '.join(field_names)}"
                    )

                for row_num, row in enumerate(reader, start=2):
                    total_rows += 1
                    utterance = row.get(utterance_col, "").strip()

                    if not utterance:
                        # Keep rows with empty utterances
                        rows_to_write.append(dict(row))
                        continue

                    # Check if we've seen this utterance before
                    if utterance in seen_utterances:
                        # Found duplicate - check if fully identical
                        first_row_num, first_row_data = seen_utterances[utterance]

                        if dict(row) == first_row_data:
                            # Fully identical - remove this duplicate
                            removed_rows.append(
                                {
                                    "row_number": row_num,
                                    "utterance": utterance,
                                    "first_occurrence": first_row_num,
                                }
                            )
                        else:
                            # Partial duplicate - keep it
                            rows_to_write.append(dict(row))
                            logger.warning(
                                f"⚠️  Row {row_num}: Duplicate utterance but different fields - KEEPING"
                            )
                    else:
                        # First occurrence - keep it
                        seen_utterances[utterance] = (row_num, dict(row))
                        rows_to_write.append(dict(row))

            # Write deduplicated file
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=field_names, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows_to_write)

            # Calculate statistics
            stats = {
                "input_file": input_file,
                "output_file": output_file,
                "total_rows": total_rows,
                "unique_rows": len(rows_to_write),
                "removed_rows": len(removed_rows),
                "processing_timestamp": datetime.now().isoformat(),
            }

            if show_stats:
                logger.info(f"\n{'='*70}")
                logger.info(f"🎉 DUPLICATE REMOVAL COMPLETE")
                logger.info(f"{'='*70}")
                logger.info(f"📁 Input file: {input_file}")
                logger.info(f"📄 Output file: {output_file}")
                logger.info(f"📊 Total rows processed: {total_rows}")
                logger.info(f"✅ Unique rows kept: {len(rows_to_write)}")
                logger.info(f"🗑️  Duplicate rows removed: {len(removed_rows)}")

                if removed_rows:
                    logger.info(f"\n🗑️  Removed duplicate rows:")
                    for removal in removed_rows[:10]:  # Show first 10
                        logger.info(
                            f"   Row {removal['row_number']}: Duplicate of row {removal['first_occurrence']}"
                        )
                    if len(removed_rows) > 10:
                        logger.info(f"   ... and {len(removed_rows) - 10} more")

                logger.info(f"{'='*70}\n")

            # Don't return anything to avoid Fire printing the result
            return None

        except Exception as e:
            logger.error(f"❌ Error removing duplicates: {e}")
            raise

    # ============================================
    # PART 8: TSV FILE MERGING/APPENDING
    # ============================================

    def merge_tsv_files(
        self,
        file1: str,
        file2: str,
        output_file: str,
        encoding: str = "utf-8",
    ) -> None:
        """Merge/append two TSV files after validating identical headers

        This function validates that both input files have identical headers (same columns in same order),
        then merges them by appending all data rows from file2 to file1, writing the result to output_file.

        Args:
            file1: Path to the first TSV file (base file)
            file2: Path to the second TSV file (file to append)
            output_file: Path to the output merged TSV file
            encoding: File encoding to use (default: "utf-8")

        Raises:
            FileNotFoundError: If either input file doesn't exist
            ValueError: If headers don't match or files are invalid

        Example:
            processor = TSVProcessor()
            processor.merge_tsv_files(
                "results/data1.tsv",
                "results/data2.tsv",
                "results/merged_data.tsv"
            )
        """
        logger.info("🔄 Starting TSV file merge operation...")
        logger.info(f"📁 File 1 (base): {file1}")
        logger.info(f"📁 File 2 (append): {file2}")
        logger.info(f"📄 Output file: {output_file}")

        # Validate both input files exist
        if not os.path.exists(file1):
            raise FileNotFoundError(f"File 1 not found: {file1}")
        if not os.path.exists(file2):
            raise FileNotFoundError(f"File 2 not found: {file2}")

        try:
            # Read headers from both files
            with open(file1, "r", encoding=encoding) as f:
                file1_lines = f.readlines()
                if not file1_lines:
                    raise ValueError(f"File 1 is empty: {file1}")
                file1_header = file1_lines[0].strip().split("\t")
                file1_data_rows = file1_lines[1:]

            with open(file2, "r", encoding=encoding) as f:
                file2_lines = f.readlines()
                if not file2_lines:
                    raise ValueError(f"File 2 is empty: {file2}")
                file2_header = file2_lines[0].strip().split("\t")
                file2_data_rows = file2_lines[1:]

            logger.info(f"📊 File 1 header: {file1_header}")
            logger.info(f"📊 File 2 header: {file2_header}")

            # Validate headers are identical (same columns in same order)
            if file1_header != file2_header:
                logger.error("❌ Header mismatch detected!")
                logger.error(f"   File 1 columns: {file1_header}")
                logger.error(f"   File 2 columns: {file2_header}")

                # Provide detailed mismatch information
                if len(file1_header) != len(file2_header):
                    logger.error(
                        f"   Column count mismatch: {len(file1_header)} vs {len(file2_header)}"
                    )
                else:
                    mismatches = []
                    for i, (col1, col2) in enumerate(zip(file1_header, file2_header)):
                        if col1 != col2:
                            mismatches.append(f"Position {i}: '{col1}' != '{col2}'")
                    if mismatches:
                        logger.error("   Mismatched columns:")
                        for mismatch in mismatches:
                            logger.error(f"     - {mismatch}")

                raise ValueError(
                    "Headers do not match! Both files must have identical columns in the same order."
                )

            logger.info("✅ Headers match! Proceeding with merge...")

            # Count rows
            file1_row_count = len(file1_data_rows)
            file2_row_count = len(file2_data_rows)
            total_rows = file1_row_count + file2_row_count

            logger.info(f"📊 File 1 data rows: {file1_row_count}")
            logger.info(f"📊 File 2 data rows: {file2_row_count}")
            logger.info(f"📊 Total merged rows: {total_rows}")

            # Create output directory if needed
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"📁 Created output directory: {output_dir}")

            # Write merged file
            with open(output_file, "w", encoding=encoding, newline="") as f:
                # Write header
                f.write("\t".join(file1_header) + "\n")

                # Write all data rows from file1
                for line in file1_data_rows:
                    f.write(line)

                # Write all data rows from file2
                for line in file2_data_rows:
                    f.write(line)

            logger.info("✅ Merge completed successfully!")
            logger.info(f"💾 Output saved to: {output_file}")
            logger.info(f"📊 Final statistics:")
            logger.info(f"   • Total columns: {len(file1_header)}")
            logger.info(f"   • Total data rows: {total_rows}")
            logger.info(f"   • Rows from file 1: {file1_row_count}")
            logger.info(f"   • Rows from file 2: {file2_row_count}")

            # Don't return anything to avoid Fire printing the result
            return None

        except Exception as e:
            logger.error(f"❌ Failed to merge TSV files: {e}")
            raise


def main():
    """Main entry point for Fire CLI"""
    fire.Fire(TSVProcessor)


if __name__ == "__main__":
    main()
