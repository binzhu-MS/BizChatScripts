"""
Utterance selector using iterative approach for balanced category selection.
This replaces the complex analysis system with a simple, fast iterative selection.
"""

import json
import logging
import sys
import os
import traceback
import fire
from typing import Dict, Any, List, Union
from collections import defaultdict

# Add parent directory to path to import llms
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from using_llms.utterance_selector_core import IterativeUtteranceSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UtteranceSelector:
    """Utterance selector using iterative approach for balanced selection."""

    def __init__(
        self,
        increment_per_category: int = 2,
    ):
        """
        Initialize the utterance selector.

        Args:
            increment_per_category: Number of utterances to select per category per round (default: 2).
                                  Keep this small (1-3) for better balance across categories.
        """
        self.increment_per_category = increment_per_category

    def select_utterances(
        self,
        input_file: str,
        output_file: str,
        test: bool = False,
        test_utterances: int = 12,
        test_categories: int = 3,
        target_utterances: int = 2000,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Select utterances from input file and save to output file.

        Args:
            input_file: Path to input JSON file with utterance data
            output_file: Path to save selected utterances
            test: Whether to run in test mode (selects from top N categories)
            test_utterances: Number of utterances to select in test mode (default: 12)
            test_categories: Number of top categories to use in test mode (default: 3)
            resume: Whether to attempt resuming from existing output (default: True).
                   If False, will overwrite any existing output file.

        Returns:
            Selection results dictionary
        """
        logger.info(f"Loading data from {input_file}")

        # Load input data
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load input file: {e}")
            raise

        logger.info(
            f"Loaded data with {len(data) if isinstance(data, (list, dict)) else 'unknown'} items"
        )

        # Use the proper LLM-based selector for complexity analysis
        iterative_selector = IterativeUtteranceSelector(
            increment_per_category=self.increment_per_category
        )

        # Determine target count and categories based on mode
        if test:
            target_count = test_utterances
            max_categories = test_categories
            logger.info(
                f"Running in TEST MODE: {test_utterances} utterances from top {test_categories} categories"
            )
            # Limit data to top categories BEFORE processing for accurate estimation
            data = self._limit_data_to_top_categories(data, max_categories)
        else:
            target_count = target_utterances
            max_categories = None  # Use all categories
            logger.info(
                f"Running in PRODUCTION MODE: targeting {target_utterances} utterances from all categories"
            )

        # Run iterative selection with LLM analysis enabled
        # Note: Results are automatically saved after each round for safety (LLM selection is slow)
        results = iterative_selector.select_utterances_iteratively(
            data=data,
            target_count=target_count,
            use_llm_analysis=True,  # Explicitly enable LLM analysis
            output_file=output_file,
            resume=True,  # Always attempt to resume if output file exists
            verbose=test,  # Show detailed info in test mode
        )

        # Results are already saved by the simple selector
        logger.info(
            f"Selection completed: {results['total_selected']} selected utterances saved to {output_file}"
        )

        # Return the results for any further processing
        return results

    def _limit_data_to_top_categories(
        self, data: Union[List[Dict], Dict], max_categories: int
    ) -> Union[List[Dict], Dict]:
        """
        Limit input data to top N categories by utterance count before processing.
        This is more efficient than limiting after processing.

        Args:
            data: Input data (list of utterances or dictionary with categories as keys)
            max_categories: Maximum number of categories to keep

        Returns:
            Filtered data with same structure as input
        """
        from collections import Counter

        # Handle dictionary format (segment -> utterances)
        if isinstance(data, dict):
            # First, organize by actual categories (segment|switching_class)
            category_data = defaultdict(list)

            for segment_name, utterances in data.items():
                for utterance in utterances:
                    switching_class = utterance.get("switching_class", "unknown")
                    category_key = f"{segment_name}|{switching_class}"
                    category_data[category_key].append(utterance)

            # Count utterances by actual category
            category_counts = []
            for category_key, utterances in category_data.items():
                category_counts.append((category_key, len(utterances)))

            # Sort by count (descending) and take top categories
            category_counts.sort(key=lambda x: x[1], reverse=True)
            top_categories = category_counts[:max_categories]

            logger.info(f"Limiting to top {max_categories} categories:")
            for i, (category, count) in enumerate(top_categories, 1):
                logger.info(f"  {i}. {category}: {count} utterances")

            # Create filtered dictionary using full category keys
            # instead of grouping back by segment
            filtered_data = {}
            top_category_names = {category for category, _ in top_categories}

            for segment_name, utterances in data.items():
                for utterance in utterances:
                    switching_class = utterance.get("switching_class", "unknown")
                    category_key = f"{segment_name}|{switching_class}"
                    if category_key in top_category_names:
                        if category_key not in filtered_data:
                            filtered_data[category_key] = []
                        filtered_data[category_key].append(utterance)

            return filtered_data

        # Handle list format (for backward compatibility)
        elif isinstance(data, list):
            utterances = data

            # Count utterances by category (using complexity/optimal_switching fields)
            category_counts = Counter()

            for utterance in utterances:
                if isinstance(utterance, dict):
                    # Extract category key (same logic as in core)
                    complexity = utterance.get("complexity_raw", "unknown")
                    optimal_switching = utterance.get(
                        "optimal_switching_classification", "default"
                    )
                    category_key = f"{complexity}|{optimal_switching}"
                    category_counts[category_key] += 1

            # Get top categories
            top_categories = dict(category_counts.most_common(max_categories))
            top_category_names = set(top_categories.keys())

            logger.info(
                f"Limiting to top {max_categories} categories: {list(top_category_names)}"
            )
            logger.info(
                f"Category counts: {dict(list(top_categories.items())[:5])}..."
            )  # Show first 5

            # Filter utterances to only include top categories
            filtered_utterances = []
            for utterance in utterances:
                if isinstance(utterance, dict):
                    complexity = utterance.get("complexity_raw", "unknown")
                    optimal_switching = utterance.get(
                        "optimal_switching_classification", "default"
                    )
                    category_key = f"{complexity}|{optimal_switching}"

                    if category_key in top_category_names:
                        filtered_utterances.append(utterance)

            return filtered_utterances
        else:
            # Unknown format, return as-is
            return data


def main(
    input_file: str,
    output_file: str = "selected_utterances.json",
    test: bool = False,
    test_utterances: int = 12,
    test_categories: int = 3,
    target_utterances: int = 2000,
    increment_per_category: int = 2,
    resume: bool = True,
):
    """
    Main function for utterance selection.

    Args:
        input_file: Path to input JSON file with utterance data
        output_file: Path to save selected utterances (default: selected_utterances.json)
        test: Whether to run in test mode (default: False)
        test_utterances: Number of utterances to select in test mode (default: 12)
        test_categories: Number of top categories to use in test mode (default: 3)
        target_utterances: Target number of utterances for production mode (default: 2000)
        increment_per_category: Utterances to select per category per round (default: 2).
                               Keep small (1-3) for better category balance.
        resume: Whether to attempt resuming from existing output (default: True).
               If False, will overwrite any existing output file.

    Example usage:
        # Test mode
        python utterance_selector.py input_data.json --test=True --test_utterances=20 --test_categories=4
        # Production mode
        python utterance_selector.py input_data.json --output_file results.json
    """

    logger.info("Iterative Utterance Selector")
    logger.info("=" * 50)
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    if test:
        logger.info(
            f"Mode: TEST ({test_utterances} utterances from top {test_categories} categories)"
        )
    else:
        logger.info("Mode: PRODUCTION (all utterances from all categories)")
    logger.info(f"Increment per category: {increment_per_category}")
    logger.info("")

    # Create selector
    selector = UtteranceSelector(
        increment_per_category=increment_per_category,
    )

    # Run selection
    try:
        results = selector.select_utterances(
            input_file=input_file,
            output_file=output_file,
            test=test,
            test_utterances=test_utterances,
            test_categories=test_categories,
            target_utterances=target_utterances,
            resume=resume,
        )

        # Check completion status
        target = target_utterances if not test else test_utterances
        if results["total_selected"] >= target:
            logger.info(f"\n✅ Selection completed successfully!")
            logger.info(
                f"🎯 Target reached: {results['total_selected']}/{target} utterances"
            )
        else:
            logger.warning(f"\n⚠️  Selection completed with shortfall!")
            logger.warning(
                f"🎯 Target not reached: {results['total_selected']}/{target} utterances"
            )
            completion_rate = (results["total_selected"] / target) * 100
            logger.warning(f"📊 Completion rate: {completion_rate:.1f}%")

        logger.info(
            f"Selected {results['total_selected']} utterances in {results['rounds_completed']} rounds"
        )

        # Show final file summary
        logger.info("\n📄 File Created:")
        logger.info(f"   • Selection results: {output_file}")

        # Check if the selector created any additional files and mention them
        if hasattr(results, "output_files") and results.get("output_files"):
            for file_path in results["output_files"]:
                logger.info(f"   • Additional output: {file_path}")
        else:
            logger.info(
                f"   • Contains {results['total_selected']} selected "
                f"utterances with full metadata"
            )

    except Exception as e:
        logger.error(f"\n❌ Selection failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire(main)
