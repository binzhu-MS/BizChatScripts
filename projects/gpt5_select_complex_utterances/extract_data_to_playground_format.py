"""
Classification Data to Playground Format Converter

This program reads classification data and converts it to playground input format.
It supports filtering by classification type (chat, cot, error, etc.) and maintains
all original information including category names.

Author: Bin Zhu
Date: August 6, 2025
"""

import json
import logging
import os
import sys
import fire
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directories to path to import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from utils.json_utils import read_json_file, write_json_file
from playground_converter import (
    PlaygroundDataConverter,
    convert_classification_to_playground,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationToPlaygroundConverter:
    """
    Converter to transform classification data to playground format with filtering options.
    """

    def __init__(self):
        self.supported_classifications = ["chat", "cot", "error", "unknown"]

    def convert_and_save(
        self,
        input_file: str,
        output_file: str,
        classification_filter: str = "cot",
        source_name: str = "classification-data",
        include_metadata: bool = True,
        preserve_all_fields: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert classification data to playground format and save it.

        Args:
            input_file: Path to input classification JSON file
            output_file: Path to save playground JSON file
            classification_filter: Which classification to filter by (chat, cot, error, all)
            source_name: Source identifier for playground entries
            include_metadata: Whether to include classification metadata
            preserve_all_fields: Whether to preserve all original fields in metadata

        Returns:
            Dictionary with conversion statistics
        """
        logger.info(f"Loading classification data from: {input_file}")

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Load classification data
        classification_data = read_json_file(input_file)

        # Handle both dict and list formats
        if isinstance(classification_data, list):
            # Convert list format to dict format with a default category
            classification_data = {"default_category": classification_data}

        logger.info(f"Loaded data with {len(classification_data)} categories")

        # Initialize converter with new simplified constructor
        from playground_converter import FieldMapping, ConversionConfig

        field_mapping = FieldMapping(utterance_field="utterance")
        config = ConversionConfig(default_source=source_name)
        converter = PlaygroundDataConverter(field_mapping, config)

        # Convert to playground format with filtering
        filter_value = (
            None if classification_filter.lower() == "all" else classification_filter
        )
        playground_data = converter.convert_classification_data_to_playground(
            classification_data=classification_data,
            filter_classification=filter_value,
            include_metadata=include_metadata,
        )

        # Enhance with additional metadata if preserve_all_fields is True
        if preserve_all_fields:
            playground_data = self._enhance_with_all_fields(
                playground_data, classification_data, filter_value
            )

        # Save playground data
        converter.save_playground_data(playground_data, output_file)

        # Generate statistics
        stats = self._generate_conversion_stats(
            classification_data, playground_data, filter_value
        )

        return stats

    def _enhance_with_all_fields(
        self,
        playground_data: List[Dict[str, Any]],
        original_data: Dict[str, List[Dict]],
        classification_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Enhance playground data with all fields from original data.

        Args:
            playground_data: Converted playground data
            original_data: Original classification data
            classification_filter: Applied classification filter

        Returns:
            Enhanced playground data
        """
        enhanced_data = []
        original_index = 0

        for category_name, utterances in original_data.items():
            for utterance_data in utterances:
                # Skip if doesn't match filter
                if classification_filter:
                    classification = utterance_data.get("classification", "").lower()
                    if classification != classification_filter.lower():
                        continue

                # Find corresponding playground entry
                if original_index < len(playground_data):
                    playground_entry = playground_data[original_index].copy()

                    # Add all original fields to metadata
                    if "results" in playground_entry and playground_entry["results"]:
                        metadata_result = playground_entry["results"][0]

                        # Add all original fields as evaluations
                        for field, value in utterance_data.items():
                            if field not in [
                                "utterance",
                                "text",
                                "content",
                                "query",
                                "input",
                            ]:
                                field_name = field.replace("_", " ").title()
                                metadata_result["evaluations"][field_name] = {
                                    "score": 100,
                                    "assessment": f"{field_name}: {value}",
                                }

                    enhanced_data.append(playground_entry)
                    original_index += 1

        return enhanced_data

    def _generate_conversion_stats(
        self,
        original_data: Dict[str, List[Dict]],
        playground_data: List[Dict[str, Any]],
        classification_filter: Optional[str],
    ) -> Dict[str, Any]:
        """Generate statistics about the conversion process."""

        # Count original data by classification
        original_counts = {
            "total_categories": len(original_data),
            "total_utterances": 0,
            "by_classification": {"chat": 0, "cot": 0, "error": 0, "unknown": 0},
        }

        for category_name, utterances in original_data.items():
            original_counts["total_utterances"] += len(utterances)

            for utterance_data in utterances:
                classification = utterance_data.get("classification", "unknown").lower()
                if classification in original_counts["by_classification"]:
                    original_counts["by_classification"][classification] += 1
                else:
                    original_counts["by_classification"]["unknown"] += 1

        # Calculate conversion stats
        converted_count = len(playground_data)
        filter_applied = classification_filter or "all"

        stats = {
            "conversion_summary": {
                "input_file_stats": original_counts,
                "filter_applied": filter_applied,
                "converted_entries": converted_count,
                "conversion_rate": round(
                    converted_count / max(1, original_counts["total_utterances"]) * 100,
                    2,
                ),
            }
        }

        return stats


def convert_to_playground(
    input_file: str,
    output_file: str,
    classification: str = "cot",
    source: str = "classification-data",
    include_metadata: bool = True,
    show_stats: bool = True,
) -> None:
    """
    Convert classification data to playground format.

    Args:
        input_file: Path to the JSON file containing classification results
        output_file: Path to save playground-formatted JSON file
        classification: Classification type to filter by (chat, cot, error, all)
        source: Source identifier for playground entries
        include_metadata: Whether to include classification metadata in results
        show_stats: Whether to display conversion statistics

    Examples:
        # Extract CoT utterances (default)
        convert_to_playground("results.json", "playground_cot.json")

        # Extract Chat utterances
        convert_to_playground("results.json", "playground_chat.json", classification="chat")

        # Extract all utterances
        convert_to_playground("results.json", "playground_all.json", classification="all")

        # Extract with custom source name
        convert_to_playground("results.json", "playground.json", source="my-experiment")
    """
    try:
        # Validate classification parameter
        valid_classifications = ["chat", "cot", "error", "unknown", "all"]
        if classification.lower() not in valid_classifications:
            raise ValueError(
                f"Invalid classification '{classification}'. Must be one of: {valid_classifications}"
            )

        # Create converter and convert data
        converter = ClassificationToPlaygroundConverter()
        stats = converter.convert_and_save(
            input_file=input_file,
            output_file=output_file,
            classification_filter=classification.lower(),
            source_name=source,
            include_metadata=include_metadata,
        )

        # Display statistics if requested
        if show_stats:
            print("\n" + "=" * 80)
            print("CLASSIFICATION TO PLAYGROUND CONVERSION STATISTICS")
            print("=" * 80)

            conversion_summary = stats["conversion_summary"]
            input_stats = conversion_summary["input_file_stats"]

            print(f"\n📊 INPUT DATA SUMMARY:")
            print(f"  Total Categories: {input_stats['total_categories']}")
            print(f"  Total Utterances: {input_stats['total_utterances']}")

            print(f"\n🎯 CLASSIFICATION DISTRIBUTION:")
            by_class = input_stats["by_classification"]
            for class_type, count in by_class.items():
                if count > 0:
                    percentage = round(
                        count / max(1, input_stats["total_utterances"]) * 100, 1
                    )
                    print(f"  {class_type.upper()}: {count} ({percentage}%)")

            print(f"\n✅ CONVERSION RESULTS:")
            print(f"  Filter Applied: {conversion_summary['filter_applied'].upper()}")
            print(f"  Converted Entries: {conversion_summary['converted_entries']}")
            print(f"  Conversion Rate: {conversion_summary['conversion_rate']}%")

            # Show filter-specific stats
            filter_applied = conversion_summary["filter_applied"]
            if filter_applied != "all" and filter_applied in by_class:
                original_filtered_count = by_class[filter_applied]
                print(
                    f"  Original {filter_applied.upper()} Entries: {original_filtered_count}"
                )
                if original_filtered_count > 0:
                    retention_rate = round(
                        conversion_summary["converted_entries"]
                        / original_filtered_count
                        * 100,
                        1,
                    )
                    print(f"  Retention Rate: {retention_rate}%")

            print(f"\n📁 OUTPUT:")
            print(f"  Playground File: {output_file}")
            print(f"  Format: Playground Input JSON")
            print("=" * 80)

        logger.info("Conversion completed successfully!")

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        raise


def main():
    """Main entry point for the converter."""
    fire.Fire(convert_to_playground)


if __name__ == "__main__":
    main()
