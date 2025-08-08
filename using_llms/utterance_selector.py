"""
Utterance selector using LLM-based iterative approach for balanced category selection.
In each round, a small number of utterances are selected from each category based on complexity,
diversity, enterprise relevance, and personalization possibilities. This process is repeated
until a target number of utterances is reached or all categories are exhausted.
"""

import json
import logging
import sys
import os
import traceback
import fire
from typing import Dict, Any, List, Union

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
        target_utterances: int,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Select utterances from input file and save to output file.

        Args:
            input_file: Path to input JSON file with utterance data
            output_file: Path to save selected utterances
            target_utterances: Number of utterances to select
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
            f"Loaded data with {len(data) if isinstance(data, (list, dict)) else 'unknown'} categories"
        )

        # Use the proper LLM-based selector for complexity analysis
        iterative_selector = IterativeUtteranceSelector(
            increment_per_category=self.increment_per_category
        )

        logger.info(f"Targeting {target_utterances} utterances from all categories")
        logger.info("")

        # Run iterative selection with LLM analysis enabled
        # Note: Results are automatically saved after each round for safety (LLM selection is slow)
        results = iterative_selector.select_utterances_iteratively(
            data=data,
            target_count=target_utterances,
            use_llm_analysis=True,  # Explicitly enable LLM analysis
            output_file=output_file,
            resume=resume,  # Attempt to resume if output file exists
            verbose=False,  # No verbose output needed
        )

        # Return the results for any further processing
        return results


def main(
    input_file: str,
    output_file: str,
    target_utterances: int,
    increment_per_category: int = 2,
    resume: bool = True,
):
    """
    Main function for utterance selection.

    Args:
        input_file: Path to input JSON file with utterance data
        output_file: Path to save selected utterances
        target_utterances: Number of utterances to select
        increment_per_category: Utterances to select per category per round (default: 2).
                               Keep small (1-3) for better category balance.
        resume: Whether to attempt resuming from existing output (default: True).
               If False, will overwrite any existing output file.

    Example usage:
        python utterance_selector.py input_data.json results.json 20
        python utterance_selector.py input_data.json results.json 2000
    """

    logger.info("Iterative Utterance Selector")
    logger.info("=" * 50)
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Target utterances: {target_utterances}")
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
            target_utterances=target_utterances,
            resume=resume,
        )

        # Check completion status and handle LLM failures
        target = target_utterances

        # Handle LLM failure status
        if results.get("status") in ["llm_failure", "llm_failure_during_resume"]:
            logger.error(f"\n🚨 SELECTION TERMINATED DUE TO LLM FAILURE")
            failure_info = results.get("failure_info", {})
            logger.error(
                f"🎯 Partial progress: {results['total_selected']}/{target} utterances selected"
            )
            logger.error(f"📁 Progress saved to: {output_file}")
            logger.error(f"🔄 Resume by running the same command again")
            sys.exit(1)  # Exit with error code

        elif results["total_selected"] >= target:
            logger.info(f"✅ Selection completed successfully!")
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

        # Show rounds information only if actual work was done
        if results.get("rounds_completed", 0) > 0:
            logger.info(
                f"Selected {results['total_selected']} utterances in {results['rounds_completed']} rounds"
            )

        # Show final file summary
        logger.info(f"📄 Selected utterances are saved into: {output_file}")

    except Exception as e:
        logger.error(f"\n❌ Selection failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire(main)
