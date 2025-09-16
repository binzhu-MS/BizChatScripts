#!/usr/bin/env python3
"""
TSV Processing Module

This module processes TSV files containing complex utterances and transforms them according to
specific business requirements. It performs the following transformations:

1. Remove the last column (Complexity_Reason)
2. Add a 'Segment' column after 'Utterance' with user-specified value (converted to lowercase)
3. Convert email_account by appending "@VertexEdgeLabs@SyntheticTenant"
4. Add a new "Source" column with value "synthetic" for each utterance

Key Features:
- Handles UTF-8 encoded TSV files with proper error handling
- Maintains data integrity during transformations
- Provides detailed logging of processing steps
- Validates input file existence before processing

Usage:
    python process_tsv.py process_file
    python process_tsv.py process_file --input_file="custom_input.tsv"
    python process_tsv.py process_file --output_file="custom_output.tsv"
    python process_tsv.py process_file --segment="file"
    python process_tsv.py process_file --segment="EMAIL" --input_file="data.tsv"
    python process_tsv.py process_file --append_mode=True  # Use append-mode defaults
    python process_tsv.py process_file --append_mode=True --show_sample=False  # Append without sample output
"""

import csv
import os
import logging
import fire
from typing import Dict, List, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TSVProcessor:
    """Process TSV files"""

    def __init__(self):
        """Initialize the TSV processor"""
        pass

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
    ) -> Dict[str, Any]:
        """Process the TSV file and append to output file, creating header if file is empty

        Args:
            input_file: Path to the input TSV file
            output_file: Path to the output TSV file (will append or create with header)
            segment: The segment value to use for all rows (will be converted to lowercase)

        Returns:
            Dict[str, any]: Processing results and statistics

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

            return {
                "input_file": input_file,
                "output_file": output_file,
                "total_input_rows": total_input_rows,
                "data_rows_processed": data_rows_processed,
                "rows_skipped": skipped_rows,
                "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "append_mode": output_file_exists,
            }

        except Exception as e:
            logger.error(f"✗ Error processing file: {e}")
            raise


def process_file(
    input_file: str = "utterances_curr.tsv",
    output_file: str = "utterances_processed.tsv",
    segment: str = "email",
    append_mode: bool = False,
) -> None:
    """Process a TSV file with complex utterance transformations

    Args:
        input_file: Path to the input TSV file (default: "utterances_curr.tsv")
        output_file: Path to the output TSV file (default: "utterances_processed.tsv")
        segment: The segment value to use for all rows (default: "email", will be converted to lowercase)
        append_mode: If True, use append-mode defaults (results/ paths), otherwise use single-file defaults
        show_sample: If True, display sample output for verification (default: True)
    """
    processor = TSVProcessor()

    # Set default paths based on append_mode
    if append_mode and input_file == "utterances_curr.tsv":
        input_file = "results/utterances_curr.tsv"
    if append_mode and output_file == "utterances_processed.tsv":
        output_file = "results/utterances_all.tsv"

    try:
        # Process the file
        result = processor.process_tsv_file(input_file, output_file, segment)

        # Log completion message
        if append_mode:
            logger.info(
                f"🎉 File successfully processed and {'appended to' if result['append_mode'] else 'saved as'} '{output_file}'"
            )
        else:
            logger.info(f"🎉 File successfully processed and saved as '{output_file}'")

    except Exception as e:
        logger.error(f"✗ Failed to process file: {e}")
        return None

    # Don't return anything to avoid Fire printing the result
    return None


if __name__ == "__main__":
    fire.Fire()
