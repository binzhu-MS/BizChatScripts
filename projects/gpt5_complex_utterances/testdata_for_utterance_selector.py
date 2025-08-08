#!/usr/bin/env python3
"""
Create a test dataset for utterance_selector.py by selecting the most popular categories from the input file.
This is useful for testing the utterance selector with a smaller, representative dataset.
"""

import json
import logging
import fire
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_categories(data: Dict[str, List[Dict]]) -> Dict[str, int]:
    """
    Analyze categories (segment|switching_class combinations) and count utterances.

    Args:
        data: Input data with segments as keys and utterance lists as values

    Returns:
        Dictionary mapping category keys to utterance counts
    """
    category_counts = defaultdict(int)

    for segment_name, utterances in data.items():
        for utterance in utterances:
            switching_class = utterance.get("switching_class", "unknown")
            category_key = f"{segment_name}|{switching_class}"
            category_counts[category_key] += 1

    return dict(category_counts)


def select_top_categories(
    data: Dict[str, List[Dict]], top_n: int = 3
) -> Dict[str, List[Dict]]:
    """
    Select utterances from the top N most popular categories.

    Args:
        data: Input data with segments as keys and utterance lists as values
        top_n: Number of top categories to select (default: 3)

    Returns:
        Dictionary with selected categories and their utterances
    """
    # Analyze categories
    category_counts = analyze_categories(data)

    # Sort by count (descending) and take top N
    sorted_categories = sorted(
        category_counts.items(), key=lambda x: x[1], reverse=True
    )
    top_categories = sorted_categories[:top_n]

    # Build filtered dataset
    selected_data = {}
    total_selected = 0

    for segment_name, utterances in data.items():
        # Group utterances by switching class within this segment
        switching_class_groups = defaultdict(list)
        for utterance in utterances:
            switching_class = utterance.get("switching_class", "unknown")
            switching_class_groups[switching_class].append(utterance)

        # Check each switching class group
        for switching_class, class_utterances in switching_class_groups.items():
            category_key = f"{segment_name}|{switching_class}"

            # If this category is in our top N, include utterances
            top_category_keys = [cat for cat, _ in top_categories]
            if category_key in top_category_keys:
                # Select all utterances from this category
                selected_utterances = class_utterances

                # Use the full category key as the key in output
                selected_data[category_key] = selected_utterances
                total_selected += len(selected_utterances)

    return selected_data


def create_test_data(
    input_file: str,
    output_file: str,
    top_categories: int = 3,
) -> None:
    """
    Create a test dataset from the most popular categories.

    Args:
        input_file: Path to input JSON file
        output_file: Path to save the test dataset
        top_categories: Number of top categories to select (default: 3)
    """
    logger.info("Creating Test Dataset from Most Popular Categories")
    logger.info("=" * 60)

    # Load input data
    try:
        logger.info(f"Reading from input file: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        raise

    # Select top categories
    selected_data = select_top_categories(data, top_categories)

    # Final statistics
    total_selected = sum(len(utterances) for utterances in selected_data.values())
    logger.info(f"Selected categories: {len(selected_data)}")
    logger.info(f"Total selected utterances: {total_selected}")
    for category, utterances in selected_data.items():
        logger.info(f"  {category}: {len(utterances)} utterances")

    # Save results
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(selected_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Test dataset saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save output file: {e}")
        raise

    logger.info("✅ Test dataset creation completed successfully!")


def main(
    input_file: str = "c:\\working\\BizChatScripts\\projects\\gpt5_complex_utterances\\data\\results\\filtered_optimal_switching_results.json",
    output_file: str = "test_data_top3_categories.json",
    top_categories: int = 3,
):
    """
    Main function to create test dataset.

    Args:
        input_file: Path to input JSON file (default: filtered_optimal_switching_results.json)
        output_file: Path to save test dataset (default: test_data_top3_categories.json)
        top_categories: Number of top categories to select (default: 3)

    Example usage:
        python create_test_data.py
        python create_test_data.py --top_categories=5
        python create_test_data.py --output_file=small_test_data.json --top_categories=2
    """
    create_test_data(input_file, output_file, top_categories)


if __name__ == "__main__":
    fire.Fire(main)
