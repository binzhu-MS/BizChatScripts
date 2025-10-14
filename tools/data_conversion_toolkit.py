#!/usr/bin/env python3
"""
Data Conversion Toolkit

This program provides comprehensive data conversion utilities for various formats including
text to JSON strings, JSON formatting, and JSON to TSV conversion for SEVAL data.

Usage:
    # Text to JSON string conversion
    python data_conversion_toolkit.py text_to_json --input_file "input.txt" --output_file "output.txt"

    # JSON string to text conversion
    python data_conversion_toolkit.py json_to_text --input_file "input.txt" --output_file "output.txt"

    # JSON formatting (pretty print or minify)
    python data_conversion_toolkit.py format_json --input_file "input.json" --output_file "output.json" --pretty
    python data_conversion_toolkit.py format_json --input_file "input.json" --output_file "output.json" --minify

    # JSON of Playground Output to TSV conversion for SEVAL format
    python data_conversion_toolkit.py playground_json_to_tsv --input_file "input.json" --output_file "output.tsv" --num_repeat 1
"""

import os
import json
import pandas as pd
import csv
import fire
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataConversionToolkit:
    """A comprehensive toolkit for various data conversion operations."""

    def text_to_json(
        self,
        input_file: str,
        output_file: Optional[str] = None,
    ):
        """
        Convert a plain text string to a JSON string.

        Args:
            input_file: Path to the input text file
            output_file: Path to the output file (optional, defaults to input_file with different extension)
        """
        logger.info(f"Converting text to JSON string: {input_file}")

        # Set default output file if not provided
        if output_file is None:
            output_file = os.path.splitext(input_file)[0] + "_json_string.txt"

        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return

        try:
            # Read input text file
            with open(input_file, "r", encoding="utf-8") as f:
                text_content = f.read()

            # Convert text to JSON string
            json_string = json.dumps(text_content)

            # Ensure output directory exists
            os.makedirs(
                os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
                exist_ok=True,
            )

            # Write JSON string to output file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_string)

            logger.info(f"Successfully converted text to JSON string: {output_file}")
            logger.info(f"Original text length: {len(text_content)} characters")
            logger.info(f"JSON string length: {len(json_string)} characters")

        except Exception as e:
            logger.error(f"Error converting text to JSON string: {e}")

    def json_to_markdown(
        self,
        input_file: str,
        output_file: Optional[str] = None,
    ):
        """
        Convert JSON data to markdown format.

        If the JSON contains conversation data, it will be formatted as markdown with proper headers.
        If the JSON contains other data types, it will be formatted as readable markdown with code blocks.

        Args:
            input_file: Path to the input file containing JSON data
            output_file: Path to the output markdown file (optional, defaults to input_file with .md extension)
        """
        logger.info(f"Converting JSON to markdown: {input_file}")

        # Set default output file if not provided
        if output_file is None:
            output_file = os.path.splitext(input_file)[0] + ".md"

        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return

        try:
            # Read input JSON string file
            with open(input_file, "r", encoding="utf-8") as f:
                json_string = f.read()

            # Convert JSON string back to text
            text_content = json.loads(json_string)

            # Ensure output directory exists
            os.makedirs(
                os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
                exist_ok=True,
            )

            # Write text to output file
            with open(output_file, "w", encoding="utf-8") as f:
                # Handle case where decoded JSON is not a string
                if isinstance(text_content, str):
                    f.write(text_content)
                else:
                    # Extract markdown-formatted text from structured data
                    markdown_content = self._extract_markdown_content(text_content)
                    f.write(markdown_content)

            logger.info(f"Successfully converted JSON to markdown: {output_file}")
            logger.info(f"Input JSON length: {len(json_string)} characters")
            if isinstance(text_content, str):
                logger.info(f"Extracted text length: {len(text_content)} characters")
                logger.info("JSON contained a string value")
            else:
                logger.info(
                    f"JSON contained a {type(text_content).__name__} - converted to markdown"
                )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON string in input file: {e}")
        except Exception as e:
            logger.error(f": {e}")

    def _extract_markdown_content(self, data) -> str:
        """
        Extract markdown-formatted content from JSON data structures.

        Handles common patterns like conversation messages, content fields, etc.
        """
        if isinstance(data, str):
            return data

        text_parts = []

        if isinstance(data, list):
            # Handle list of items (like conversation messages)
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    # Look for common text fields
                    if "content" in item and "role" in item:
                        # Conversation message format - use markdown headers
                        role = item.get("role", "").title()
                        content = item.get("content", "")
                        text_parts.append(f"## {role}\n\n{content}\n")
                    elif "content" in item:
                        # Generic content field
                        text_parts.append(f"### Item {i+1}\n\n{item['content']}\n")
                    elif "text" in item:
                        # Generic text field
                        text_parts.append(f"### Item {i+1}\n\n{item['text']}\n")
                    elif "message" in item:
                        # Message field
                        text_parts.append(f"### Item {i+1}\n\n{item['message']}\n")
                    else:
                        # Extract all string values
                        text_content = self._extract_all_strings(item)
                        if text_content:
                            text_parts.append(f"### Item {i+1}\n\n{text_content}\n")
                elif isinstance(item, str):
                    text_parts.append(f"### Item {i+1}\n\n{item}\n")

        elif isinstance(data, dict):
            # Handle single dictionary
            if "content" in data and "role" in data:
                # Single conversation message
                role = data.get("role", "").title()
                content = data.get("content", "")
                text_parts.append(f"## {role}\n\n{content}")
            elif "content" in data:
                text_parts.append(data["content"])
            elif "text" in data:
                text_parts.append(data["text"])
            elif "message" in data:
                text_parts.append(data["message"])
            else:
                # Extract all string values
                text_content = self._extract_all_strings(data)
                if text_content:
                    text_parts.append(text_content)

        # If no text was extracted, fallback to JSON code block
        if not text_parts:
            return f"```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"

        return "\n".join(text_parts)

    def _extract_readable_text(self, data) -> str:
        """
        Extract human-readable text from JSON data structures (plain text format).

        Handles common patterns like conversation messages, content fields, etc.
        """
        if isinstance(data, str):
            return data

        text_parts = []

        if isinstance(data, list):
            # Handle list of items (like conversation messages)
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    # Look for common text fields
                    if "content" in item and "role" in item:
                        # Conversation message format
                        role = item.get("role", "").title()
                        content = item.get("content", "")
                        text_parts.append(f"=== {role} ===\n{content}\n")
                    elif "content" in item:
                        # Generic content field
                        text_parts.append(f"--- Item {i+1} ---\n{item['content']}\n")
                    elif "text" in item:
                        # Generic text field
                        text_parts.append(f"--- Item {i+1} ---\n{item['text']}\n")
                    elif "message" in item:
                        # Message field
                        text_parts.append(f"--- Item {i+1} ---\n{item['message']}\n")
                    else:
                        # Extract all string values
                        text_content = self._extract_all_strings(item)
                        if text_content:
                            text_parts.append(f"--- Item {i+1} ---\n{text_content}\n")
                elif isinstance(item, str):
                    text_parts.append(f"--- Item {i+1} ---\n{item}\n")

        elif isinstance(data, dict):
            # Handle single dictionary
            if "content" in data and "role" in data:
                # Single conversation message
                role = data.get("role", "").title()
                content = data.get("content", "")
                text_parts.append(f"=== {role} ===\n{content}")
            elif "content" in data:
                text_parts.append(data["content"])
            elif "text" in data:
                text_parts.append(data["text"])
            elif "message" in data:
                text_parts.append(data["message"])
            else:
                # Extract all string values
                text_content = self._extract_all_strings(data)
                if text_content:
                    text_parts.append(text_content)

        # If no text was extracted, fallback to pretty JSON
        if not text_parts:
            return json.dumps(data, indent=2, ensure_ascii=False)

        return "\n".join(text_parts)

    def _extract_all_strings(self, obj) -> str:
        """Extract all string values from a nested object."""
        strings = []

        if isinstance(obj, str):
            strings.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, str) and len(value.strip()) > 0:
                    strings.append(value)
                elif isinstance(value, (dict, list)):
                    nested = self._extract_all_strings(value)
                    if nested:
                        strings.append(nested)
        elif isinstance(obj, list):
            for item in obj:
                nested = self._extract_all_strings(item)
                if nested:
                    strings.append(nested)

        return "\n".join(strings)

    def format_json(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        pretty: bool = False,
        minify: bool = False,
    ):
        """
        Format JSON file as pretty-printed or minified.

        Args:
            input_file: Path to the input JSON file
            output_file: Path to the output JSON file (optional, defaults to input_file with suffix)
            pretty: Format as pretty-printed JSON with indentation
            minify: Format as minified JSON without spaces
        """
        if pretty and minify:
            logger.error("Cannot specify both --pretty and --minify options")
            return

        if not pretty and not minify:
            pretty = True  # Default to pretty printing

        operation = "pretty-printing" if pretty else "minifying"
        logger.info(f"JSON formatting ({operation}): {input_file}")

        # Set default output file if not provided
        if output_file is None:
            suffix = "_pretty" if pretty else "_minified"
            output_file = os.path.splitext(input_file)[0] + suffix + ".json"

        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return

        try:
            # Read input JSON file
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Ensure output directory exists
            os.makedirs(
                os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
                exist_ok=True,
            )

            # Write formatted JSON to output file
            with open(output_file, "w", encoding="utf-8") as f:
                if pretty:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                else:  # minify
                    json.dump(data, f, separators=(",", ":"), ensure_ascii=False)

            logger.info(f"Successfully formatted JSON ({operation}): {output_file}")

            # Show file size comparison
            input_size = os.path.getsize(input_file)
            output_size = os.path.getsize(output_file)
            logger.info(f"File size: {input_size} bytes → {output_size} bytes")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in input file: {e}")
        except Exception as e:
            logger.error(f"Error formatting JSON: {e}")

    def playground_json_to_tsv(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        num_repeat: int = 1,
    ):
        """
        Extract results from JSON data and save to TSV file in SEVAL data format.

        Expected JSON format: Array of objects with 'input.parameters' containing 'utterance' field.

        Args:
            input_file: Path to the input JSON file
            output_file: Path to the output TSV file (optional, defaults to input_file with .tsv extension)
            num_repeat: Number of times to repeat each utterance in the output file
        """
        logger.info(f"Converting JSON to TSV (SEVAL format): {input_file}")
        logger.info(f"Repetition count: {num_repeat}")

        # Set default output file if not provided
        if output_file is None:
            output_file = os.path.splitext(input_file)[0] + ".tsv"

        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return

        try:
            # Read input JSON file
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.error(f"Expected JSON array, got {type(data).__name__}")
                return

            # Initialize DataFrame columns for SEVAL format
            columns = ["query", "segment"]
            df = pd.DataFrame(columns=columns)

            processed_count = 0

            # Process each item in the JSON data
            for item in data:
                try:
                    # Parse the 'parameters' field to extract 'utterance'
                    if "input" not in item or "parameters" not in item["input"]:
                        logger.warning(
                            f"Skipping item missing 'input.parameters': {item}"
                        )
                        continue

                    parameters = json.loads(item["input"]["parameters"])

                    if "utterance" not in parameters:
                        logger.warning(
                            f"Skipping item missing 'utterance' in parameters: {parameters}"
                        )
                        continue

                    utterance = parameters["utterance"]

                    # Repeat utterance as specified
                    for i in range(num_repeat):
                        row_data = {
                            "query": utterance,
                            "segment": "en",  # Default segment
                        }

                        df = pd.concat(
                            [df, pd.DataFrame([row_data])], ignore_index=True
                        )
                        processed_count += 1

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error processing item: {e}")
                    continue

            # Ensure output directory exists
            os.makedirs(
                os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
                exist_ok=True,
            )

            # Save DataFrame to TSV file
            df.to_csv(output_file, sep="\t", index=False, encoding="utf-8")

            logger.info(f"Successfully converted JSON to TSV: {output_file}")
            logger.info(f"Processed {len(data)} input items")
            logger.info(
                f"Generated {processed_count} TSV rows ({len(df)} total with repetitions)"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in input file: {e}")
        except Exception as e:
            logger.error(f"Error converting JSON to TSV: {e}")


def main():
    """Main entry point using Fire for CLI interface."""
    fire.Fire(DataConversionToolkit)


if __name__ == "__main__":
    main()
