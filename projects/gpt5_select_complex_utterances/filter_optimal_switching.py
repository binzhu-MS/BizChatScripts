#!/usr/bin/env python3
"""
Optimal Switching Class Filter

This program filters utterances from parsed playground results based on the switching
classes defined in Merged_Optimal_Switching.json. It maintains the same structure
as the input file while only including utterances with matching switching classes.

Author: Bin Zhu
Date: August 6, 2025
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Set, Any
import fire

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


class OptimalSwitchingFilter:
    """
    Filters playground results to include only utterances with optimal switching classes.
    """

    def __init__(self):
        """Initialize the filter."""
        self.optimal_classes: Set[str] = set()

    def load_optimal_classes(self, optimal_switching_file: str) -> None:
        """
        Load the optimal switching classes from the configuration file.

        Args:
            optimal_switching_file: Path to the Merged_Optimal_Switching.json file
        """
        logger.info(f"Loading optimal switching classes from: {optimal_switching_file}")

        try:
            with open(optimal_switching_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert title case to lowercase with underscores to match data format
            reasoning_classes = data.get("ReasoningClasses", [])

            for class_name in reasoning_classes:
                # Convert from "How To Advice 2 Default" to "how_to_advice_2_default"
                normalized_class = class_name.lower().replace(" ", "_")
                self.optimal_classes.add(normalized_class)

            logger.info(f"Loaded {len(self.optimal_classes)} optimal switching classes")
            logger.debug(f"Optimal classes: {sorted(self.optimal_classes)}")

        except FileNotFoundError:
            logger.error(f"Optimal switching file not found: {optimal_switching_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in optimal switching file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading optimal switching file: {e}")
            raise

    def filter_utterances(
        self, input_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Filter utterances to include only those with optimal switching classes.

        Args:
            input_data: The parsed playground data grouped by segments

        Returns:
            Filtered data with same structure but only optimal switching class utterances
        """
        filtered_data = {}
        total_utterances = 0
        filtered_utterances = 0

        for segment_name, utterances in input_data.items():
            filtered_segment_utterances = []

            for utterance_data in utterances:
                total_utterances += 1
                switching_class = utterance_data.get("switching_class", "").lower()

                if switching_class in self.optimal_classes:
                    filtered_segment_utterances.append(utterance_data)
                    filtered_utterances += 1

            # Only include segments that have at least one matching utterance
            if filtered_segment_utterances:
                filtered_data[segment_name] = filtered_segment_utterances

        logger.info(
            f"Filtered {filtered_utterances:,} out of {total_utterances:,} utterances "
            f"({filtered_utterances/total_utterances*100:.1f}%)"
        )
        logger.info(
            f"Segments with optimal utterances: {len(filtered_data):,} out of {len(input_data):,}"
        )

        return filtered_data

    def print_summary(
        self,
        original_data: Dict[str, List[Dict[str, Any]]],
        filtered_data: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Print a summary of the filtering results."""

        # Count original statistics
        orig_segments = len(original_data)
        orig_utterances = sum(len(utterances) for utterances in original_data.values())

        # Count filtered statistics
        filt_segments = len(filtered_data)
        filt_utterances = sum(len(utterances) for utterances in filtered_data.values())

        # Get switching class distribution in filtered data
        switching_class_counts = {}
        for utterances in filtered_data.values():
            for utterance_data in utterances:
                switching_class = utterance_data.get("switching_class", "unknown")
                switching_class_counts[switching_class] = (
                    switching_class_counts.get(switching_class, 0) + 1
                )

        print("=" * 80)
        print("OPTIMAL SWITCHING CLASS FILTER RESULTS")
        print("=" * 80)
        print(f"📊 FILTERING SUMMARY:")
        print(f"  Original Segments: {orig_segments:,}")
        print(
            f"  Filtered Segments: {filt_segments:,} ({filt_segments/orig_segments*100:.1f}%)"
        )
        print(f"  Original Utterances: {orig_utterances:,}")
        print(
            f"  Filtered Utterances: {filt_utterances:,} ({filt_utterances/orig_utterances*100:.1f}%)"
        )

        print(f"\n🎯 TOP 10 OPTIMAL SWITCHING CLASSES:")
        for i, (switching_class, count) in enumerate(
            sorted(switching_class_counts.items(), key=lambda x: x[1], reverse=True)[
                :10
            ],
            1,
        ):
            print(f"   {i}. {switching_class}: {count:,} utterances")

        print(f"\n📋 TOP 10 SEGMENTS BY FILTERED UTTERANCE COUNT:")
        segment_counts = [
            (name, len(utterances)) for name, utterances in filtered_data.items()
        ]
        segment_counts.sort(key=lambda x: x[1], reverse=True)

        for i, (segment_name, count) in enumerate(segment_counts[:10], 1):
            # Truncate long segment names for display
            display_name = (
                segment_name[:60] + "..." if len(segment_name) > 60 else segment_name
            )
            print(f"   {i}. {display_name}: {count:,} utterances")

        print("=" * 80)


def filter_optimal_switching_utterances(
    input_file: str,
    optimal_switching_file: str,
    output_file: str,
    show_summary: bool = True,
) -> None:
    """
    Filter playground results to include only utterances with optimal switching classes.

    Args:
        input_file: Path to the parsed playground results JSON file
        optimal_switching_file: Path to the Merged_Optimal_Switching.json file
        output_file: Path to save the filtered results
        show_summary: Whether to print a summary of results

    Examples:
        # Basic filtering
        filter_optimal_switching_utterances("parsed_results.json", "Merged_Optimal_Switching.json", "filtered_results.json")

        # With summary disabled
        filter_optimal_switching_utterances("parsed_results.json", "Merged_Optimal_Switching.json", "filtered_results.json", False)
    """
    try:
        # Initialize the filter
        filter_obj = OptimalSwitchingFilter()

        # Load optimal switching classes
        filter_obj.load_optimal_classes(optimal_switching_file)

        # Load input data
        logger.info(f"Loading input data from: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        # Filter the data
        logger.info("Filtering utterances based on optimal switching classes...")
        filtered_data = filter_obj.filter_utterances(input_data)

        # Save filtered data
        logger.info(f"Saving filtered data to: {output_file}")

        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)

        # Print summary if requested
        if show_summary:
            filter_obj.print_summary(input_data, filtered_data)

        logger.info("Filtering completed successfully!")

    except Exception as e:
        logger.error(f"Error during filtering: {e}")
        raise


def main():
    """Main entry point for the optimal switching filter."""
    fire.Fire(filter_optimal_switching_utterances)


if __name__ == "__main__":
    main()
