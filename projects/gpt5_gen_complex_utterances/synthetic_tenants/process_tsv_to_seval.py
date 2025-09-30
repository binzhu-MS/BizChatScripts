#!/usr/bin/env python3
"""
TSV to SEVAL Format Processor and TSV Cleanup Utility
=====================================================

This module provides two main functionalities:

1. TSV to SEVAL Format Conversion:
   - Processes utterances_all.tsv and converts it to SEVAL format
   - Maps columns and adds metadata fields

2. TSV Cleanup Utility:
   This utility provides comprehensive TSV file cleaning capabilities with immediate output during processing.
   - Removes enclosing quotes from utterances and adjusts internal quotes
   - Identifies and replaces non-ASCII characters with ASCII equivalents
   - Validates that quoted content and non-ASCII characters only appear in utterance column
   - Reports all modifications and validation errors with immediate output during processing

Usage:
    # Convert to SEVAL format
    python process_tsv_to_seval.py process_to_seval --input_file="utterances_all.tsv" --output_file="utterances_seval.tsv"

    # Clean up TSV file
    python process_tsv_to_seval.py cleanup_tsv --input_file="utterances_seval.tsv" --output_file="utterances_cleaned.tsv"
"""

import os
import sys
import csv
import fire
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class TSVToSEvalProcessor:
    """
    Processes TSV files for SEVAL format conversion and cleanup operations.

    This class provides two main functionalities:

    1. SEVAL Format Conversion:
       - Input format: Utterance, Segment, Email_Account, Source
       - Output format: Utterance, Segment, Environment, Grounding Data Source, user_id, timestamp

    2. TSV Cleanup Operations:
       - Removes enclosing quotes from utterances and adjusts internal quotes
       - Identifies and replaces non-ASCII characters with ASCII equivalents
       - Validates that problematic content only appears in utterance column
       - Reports all modifications and validation errors
    """

    def __init__(self):
        """Initialize the TSV to SEVAL processor."""
        pass

    def _load_input_tsv(self, input_file: str) -> list:
        """Load and parse the input TSV file

        Args:
            input_file: Path to the input TSV file

        Returns:
            list: List of dictionaries containing the row data
        """
        logger.info(f"📖 Loading input TSV file: {input_file}")

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        rows = []
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # First line is headers
            header_line = lines[0].strip()
            headers = header_line.split("\t")

            # Validate required columns
            required_columns = ["Utterance", "Segment", "Email_Account", "Source"]
            if not all(col in headers for col in required_columns):
                missing = [col for col in required_columns if col not in headers]
                raise ValueError(f"Missing required columns: {missing}")

            # Process each data line manually to preserve quotes
            for row_num, line in enumerate(lines[1:], 1):
                line = line.rstrip("\n\r")  # Remove line endings only
                fields = line.split("\t")

                # Pad fields if line has fewer than expected columns
                while len(fields) < len(headers):
                    fields.append("")

                # Create row dictionary
                row_data = {}
                for i, header in enumerate(headers):
                    row_data[header] = fields[i] if i < len(fields) else ""

                rows.append(
                    {
                        "row_number": row_num,
                        "utterance": row_data.get(
                            "Utterance", ""
                        ),  # Don't strip here to preserve quotes
                        "segment": row_data.get("Segment", "").strip(),
                        "email_account": row_data.get("Email_Account", "").strip(),
                        "source": row_data.get("Source", "").strip(),
                    }
                )

            logger.info(f"✅ Successfully loaded {len(rows)} rows from input file")
            return rows

        except Exception as e:
            error_msg = f"❌ Error loading input TSV file: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _transform_to_seval_format(self, input_rows: list) -> list:
        """Transform input rows to SEVAL format

        Args:
            input_rows: List of input row dictionaries

        Returns:
            list: List of transformed row dictionaries in SEVAL format
        """
        logger.info(f"🔄 Transforming {len(input_rows)} rows to SEVAL format...")

        seval_rows = []
        for row in input_rows:
            seval_row = {
                "Utterance": row["utterance"],
                "Segment": row["segment"],
                "Environment": "Test Tenant",
                "Grounding Data Source": "Web+Work",
                "user_id": row["email_account"],
                "timestamp": "",  # Empty as requested
            }
            seval_rows.append(seval_row)

        logger.info(f"✅ Successfully transformed {len(seval_rows)} rows")
        return seval_rows

    def _save_output_tsv(self, output_rows: list, output_file: str) -> None:
        """Save the transformed rows to output TSV file

        Args:
            output_rows: List of transformed row dictionaries
            output_file: Path to the output TSV file
        """
        logger.info(f"💾 Saving {len(output_rows)} rows to: {output_file}")

        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Define the column order
        fieldnames = [
            "Utterance",
            "Segment",
            "Environment",
            "Grounding Data Source",
            "user_id",
            "timestamp",
        ]

        try:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(output_rows)

            logger.info(f"✅ Successfully saved output file: {output_file}")

        except Exception as e:
            error_msg = f"❌ Error saving output TSV file: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def process_to_seval(
        self,
        input_file: str = "utterances_all.tsv",
        output_file: str = "utterances_seval.tsv",
        max_items: Optional[int] = None,
    ) -> None:
        """Process TSV file from utterances_all format to SEVAL format

        Args:
            input_file: Path to the input TSV file
            output_file: Path to the output TSV file
            max_items: Maximum number of items to process (None for all)
        """
        logger.info("🚀 Starting TSV to SEVAL conversion...")
        logger.info(f"📁 Input file: {input_file}")
        logger.info(f"📄 Output file: {output_file}")
        if max_items:
            logger.info(f"🔢 Max items: {max_items}")

        try:
            # Load input data
            input_rows = self._load_input_tsv(input_file)

            # Limit items if specified
            if max_items and max_items > 0:
                original_count = len(input_rows)
                input_rows = input_rows[:max_items]
                logger.info(
                    f"🔢 Limited to {len(input_rows)} rows (max_items={max_items}, original={original_count})"
                )

            if not input_rows:
                logger.warning("⚠️ No rows to process!")
                return

            # Transform to SEVAL format
            seval_rows = self._transform_to_seval_format(input_rows)

            # Save output
            self._save_output_tsv(seval_rows, output_file)

            # Print summary
            logger.info("🎉 TSV to SEVAL Conversion Complete!")
            logger.info("=" * 50)
            logger.info(f"📊 Input rows processed: {len(input_rows)}")
            logger.info(f"📊 Output rows generated: {len(seval_rows)}")
            logger.info(f"📁 Input file: {input_file}")
            logger.info(f"📄 Output file: {output_file}")

        except Exception as e:
            logger.error(f"❌ Critical error in TSV to SEVAL conversion: {e}")
            raise

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

        # Common non-ASCII character replacements
        char_replacements = {
            # Quotes (using Unicode escape codes to ensure correctness)
            "\u201c": '"',  # Left double quotation mark (ord 8220)
            "\u201d": '"',  # Right double quotation mark (ord 8221)
            "\u2018": "'",  # Left single quotation mark (ord 8216)
            "\u2019": "'",  # Right single quotation mark (ord 8217)
            # Dashes
            "\u2013": "-",  # En dash (ord 8211)
            "\u2014": "-",  # Em dash (ord 8212)
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
            "\u201c": '"',  # Left double quotation mark (ord 8220)
            "\u201d": '"',  # Right double quotation mark (ord 8221)
            "\u2018": "'",  # Left single quotation mark (ord 8216)
            "\u2019": "'",  # Right single quotation mark (ord 8217)
            "\u2013": "-",  # En dash (ord 8211)
            "\u2014": "-",  # Em dash (ord 8212)
            "\u2011": "-",  # Non-breaking hyphen (ord 8209)
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
                "character_replacements": {},  # Track specific character replacements
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
                                # Print replacement immediately
                                print(
                                    f"Row {row_num}: Non-ASCII character replacements:"
                                )
                                print(f"   Original: {cleaned_value}")
                                print(f"   Cleaned:  {normalized_value}")
                                for replacement in replacements:
                                    print(f"   Change:   {replacement}")

                                stats["non_ascii_normalizations"] += 1
                                # Track specific character replacements
                                for replacement in replacements:
                                    stats["character_replacements"][replacement] = (
                                        stats["character_replacements"].get(
                                            replacement, 0
                                        )
                                        + 1
                                    )

                                cleaned_value = normalized_value
                                row_modified = True

                            # Check for unreplaced non-ASCII characters
                            unreplaced_chars = self._get_unreplaced_non_ascii_chars(
                                cleaned_value
                            )
                            if unreplaced_chars:
                                print(
                                    f"Row {row_num}: Non-ASCII characters found - NO REPLACEMENT OCCURRED:"
                                )
                                print(f"   Utterance: {cleaned_value}")
                                print("   Unreplaced Unicode characters:")
                                for char_info in unreplaced_chars:
                                    print(
                                        f"     • '{char_info['char']}' - {char_info['unicode_name']} ({char_info['hex']}, ord {char_info['ord']})"
                                    )

                                # Store for final statistics
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
                            print(
                                f"Row {row_num}: Removed enclosing quotes from utterance"
                            )
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
            print(f"   • Total rows processed: {stats['total_rows']}")
            print(
                f"   • Rows with changes: {stats['rows_with_changes']} ({stats['rows_with_changes']/stats['total_rows']*100:.1f}%)"
            )
            print(
                f"   • Unchanged rows: {stats['total_rows'] - stats['rows_with_changes']}"
            )

            print("\n🔧 QUOTE PROCESSING:")
            print(f"   • Quote removals: {stats['quote_removals']}")
            print(f"   • Rows with quotes: {len(stats['rows_with_quotes'])}")

            print("\n🌐 NON-ASCII CHARACTER PROCESSING:")
            print(f"   • Non-ASCII normalizations: {stats['non_ascii_normalizations']}")
            print(
                f"   • Rows with non-ASCII chars: {len(stats['rows_with_non_ascii'])}"
            )

            if stats["character_replacements"]:
                print("\n📈 CHARACTER REPLACEMENT BREAKDOWN:")
                sorted_replacements = sorted(
                    stats["character_replacements"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for char_mapping, count in sorted_replacements:
                    print(f"   • {char_mapping}: {count} replacement(s)")

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
                    print(
                        f"     - '{char}' ({char_info['hex']}) - {char_info['unicode_name']}: {data['count']} occurrence(s) in rows {data['rows']}"
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


def main():
    """Main entry point for Fire CLI"""
    fire.Fire(TSVToSEvalProcessor)


if __name__ == "__main__":
    main()
