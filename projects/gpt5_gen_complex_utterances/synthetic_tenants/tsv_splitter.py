#!/usr/bin/env python3
"""
TSV Splitter Module

This module reads a TSV file containing utterances and splits them into separate files
based on segment (email/file) and email account. Each group is saved to a file with
the naming pattern: <segment>_<email_account_prefix>_utterances.tsv

Key Features:
- Groups utterances by segment and email account
- Extracts email prefix (part before @) for filename
- Maintains TSV format with proper headers
- Provides detailed logging of splitting process
- Handles UTF-8 encoding and proper error handling

Usage:
    python tsv_splitter.py split_utterances
    python tsv_splitter.py split_utterances --input_file="custom_input.tsv"
    python tsv_splitter.py split_utterances --output_dir="custom_output_directory"
"""

import csv
import os
import logging
import fire
from typing import Dict, List, DefaultDict, Any
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TSVSplitter:
    """Split TSV files by segment and email account"""

    def __init__(self):
        """Initialize the TSV splitter"""
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

    def _group_utterances(
        self, input_file: str
    ) -> tuple[List[str], DefaultDict[str, List[List[str]]]]:
        """Group utterances by segment and email account

        Args:
            input_file: Path to the input TSV file

        Returns:
            tuple: (header, grouped_data) where grouped_data is keyed by "segment_email_prefix"
        """
        grouped_data = defaultdict(list)
        header = None

        with open(input_file, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")

            # Read header
            header = next(reader)
            logger.info(f"📝 Header found: {header}")

            # Expected columns: Utterance, Segment, Email_Account, Source
            if len(header) < 4:
                raise ValueError(
                    f"✗ Expected at least 4 columns, got {len(header)}: {header}"
                )

            # Process data rows
            for row_num, row in enumerate(reader, start=2):
                if len(row) < 4:
                    logger.warning(
                        f"⚠️ Skipping row {row_num} (insufficient columns): {row}"
                    )
                    continue

                utterance = row[0].strip()
                segment = row[1].strip().lower()
                email_account = row[2].strip()
                source = row[3].strip()

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

    def split_tsv_file(
        self, input_file: str, output_dir: str = "split_results"
    ) -> Dict[str, Any]:
        """Split the TSV file by segment and email account

        Args:
            input_file: Path to the input TSV file
            output_dir: Directory to save split files (default: "split_results")

        Returns:
            Dict[str, int]: Statistics about the splitting process
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
        header, grouped_data = self._group_utterances(input_file)

        if not grouped_data:
            logger.warning("⚠️ No data groups found to split!")
            return {"groups_created": 0, "total_utterances": 0}

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

        return {
            "groups_created": groups_created,
            "total_utterances": total_utterances,
            "output_directory": output_dir,
            "file_statistics": file_stats,
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


def split_utterances(
    input_file: str = "results/utterances_all.tsv",
    output_dir: str = "results/split_results",
) -> None:
    """Split a TSV file containing utterances by segment and email account

    Args:
        input_file: Path to the input TSV file (default: "results/utterances_all.tsv")
        output_dir: Directory to save split files (default: "split_results")
    """
    splitter = TSVSplitter()

    try:
        # Split the file
        result = splitter.split_tsv_file(input_file, output_dir)

        logger.info(
            f"🎉 TSV file successfully split into {result['groups_created']} files!"
        )

    except Exception as e:
        logger.error(f"✗ Failed to split TSV file: {e}")
        return None

    # Don't return anything to avoid Fire printing the result
    return None


if __name__ == "__main__":
    fire.Fire()
