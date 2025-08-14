"""
RSP-style scoring example using the BizChatScripts framework.
This demonstrates how to build RSP-compatible applications using the framework's
prompt management and base classes.
"""

import json
import logging
import sys
import os
import fire
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path to import llms
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llms import ChatCompletionLLMApplier, ApplicationModes, with_retries, prompts, util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RSPStyleScorer(ChatCompletionLLMApplier):
    """
    Example RSP-compatible scorer that evaluates text quality on multiple dimensions.
    This demonstrates the RSP pattern used in Microsoft's internal scoring systems.
    """

    DEFAULT_PROMPT = prompts.get("example_rsp_scorer", "0.1.0")
    DEFAULT_MODEL_CONFIG = {
        "model": "dev-gpt-41-longco-2025-04-14",
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    DEFAULT_THREADS = 3
    DEFAULT_RETRIES = 3
    APPLICATION_MODE = ApplicationModes.PerItem

    def __init__(
        self,
        threads=None,
        retries=None,
        client_type=None,
        **kwargs,
    ):
        """Initialize with optional custom thread count, retries, and client type."""
        self.threads = threads if threads is not None else self.DEFAULT_THREADS
        self.retries = retries if retries is not None else self.DEFAULT_RETRIES

        # Pass client_type to parent class
        if client_type is not None:
            kwargs["client_type"] = client_type

        super().__init__(**kwargs)

    def process_item(self, item, i):
        """Process a single item for RSP-style scoring."""
        text_to_score = item.get("text", "")
        scoring_criteria = item.get("criteria", ["clarity", "helpfulness", "accuracy"])

        # Validate input
        if not text_to_score.strip():
            logger.warning(f"Empty text for item {i}")
            item["error"] = "No text provided for scoring"
            return item

        # Score the text
        try:
            scores = self.score_text(text_to_score, scoring_criteria)
            if scores:
                item.update(scores)
                logger.info(f"Scored item {i} on {len(scoring_criteria)} criteria")
            else:
                item["error"] = "Failed to generate scores"
        except Exception as e:
            logger.error(f"Error scoring item {i}: {e}")
            item["error"] = str(e)

        return item

    @with_retries
    def score_text(self, text, criteria=None):
        """
        Score text using RSP-compatible multi-dimensional evaluation.

        Args:
            text (str): The text to score
            criteria (list): List of scoring criteria (e.g., ["clarity", "helpfulness"])

        Returns:
            dict: Dictionary with scores and explanations for each criterion
        """
        if criteria is None:
            criteria = ["clarity", "helpfulness", "accuracy"]

        prompt_context = {
            "text_to_score": text,
            "criteria": ", ".join(criteria),
            "criteria_count": len(criteria),
        }

        # Format the RSP-style prompt
        formatted_prompt = prompts.formatting.render_messages(
            self.prompt, prompt_context
        )

        # Call the LLM
        completion = self.llmapi.chat_completion(self.model_config, formatted_prompt)

        # Extract and parse response
        response_text = completion["choices"][0]["message"]["content"].strip()

        if response_text:
            return self._parse_rsp_scores(response_text, criteria)

        return None

    def _parse_rsp_scores(self, response_text, criteria):
        """
        Parse the LLM response into structured scores following RSP format.

        Args:
            response_text (str): Raw response from LLM
            criteria (list): Expected scoring criteria

        Returns:
            dict: Parsed scores and explanations
        """
        scores = {}

        try:
            # Split response into lines for parsing
            lines = response_text.split("\n")
            current_criterion = None
            current_explanation = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if line contains a criterion and score
                for criterion in criteria:
                    if line.lower().startswith(criterion.lower()):
                        # Save previous criterion if exists
                        if current_criterion:
                            scores[f"{current_criterion}_explanation"] = " ".join(
                                current_explanation
                            )

                        # Extract score (look for pattern like "Clarity: 4/5" or "Clarity: 4")
                        if ":" in line:
                            score_part = line.split(":", 1)[1].strip()
                            # Extract numeric score
                            import re

                            score_match = re.search(r"(\d+(?:\.\d+)?)", score_part)
                            if score_match:
                                scores[f"{criterion}_score"] = float(
                                    score_match.group(1)
                                )

                        current_criterion = criterion
                        current_explanation = []
                        break
                else:
                    # This line is part of explanation
                    if current_criterion:
                        current_explanation.append(line)

            # Save the last criterion
            if current_criterion:
                scores[f"{current_criterion}_explanation"] = " ".join(
                    current_explanation
                )

            # Calculate overall score if multiple criteria
            if len(criteria) > 1:
                individual_scores = [
                    scores.get(f"{c}_score", 0)
                    for c in criteria
                    if f"{c}_score" in scores
                ]
                if individual_scores:
                    scores["overall_score"] = sum(individual_scores) / len(
                        individual_scores
                    )

        except Exception as e:
            logger.warning(f"Error parsing RSP scores: {e}, using fallback parsing")
            # Fallback: just extract any numbers found
            import re

            numbers = re.findall(r"(\d+(?:\.\d+)?)", response_text)
            if numbers:
                scores["overall_score"] = float(numbers[0])
                scores["raw_response"] = response_text

        return scores


def process_texts(
    input_file: str,
    output_file: Optional[str] = None,
    threads: int = 3,
    retries: int = 3,
    max_items: int = -1,
    client_type: Optional[str] = None,
):
    """
    Process a JSON file with RSP-style scoring requests.

    Args:
        input_file: Path to input JSON file with text to score
        output_file: Path to output JSON file (default: input file with _output suffix)
        threads: Number of worker threads to use for processing (default: 3)
        retries: Number of retries for failed LLM calls (default: 3)
        max_items: Maximum number of items to process (-1 for all)

    Examples:
        # Basic usage with defaults
        process_texts("examples/example_scoring_data.json")

        # With custom parameters
        process_texts("data.json", "results.json", threads=5, retries=2, max_items=100)
    """
    logger.info(f"Loading data from {input_file}")

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Set default output file if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(
            input_path.parent / f"{input_path.stem}_output{input_path.suffix}"
        )

    # Load input data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            logger.error(f"No data loaded from {input_file}")
            return False

        logger.info(f"Loaded {len(data)} items from {input_file}")
    except Exception as e:
        logger.error(f"Error loading input file {input_file}: {e}")
        return False

    # Limit items if specified
    if max_items > 0 and len(data) > max_items:
        data = data[:max_items]
        logger.info(f"Limited to {max_items} items for processing")

    # Initialize scorer with configuration
    scorer = RSPStyleScorer(threads=threads, retries=retries, client_type=client_type)

    # Log the model configuration and client information being used
    logger.info(f"Model configuration: {scorer.model_config}")

    # Determine and log the functional capabilities being used
    client_name = type(scorer.llmapi).__name__
    if "RSP" in client_name or "LLMAPI" == client_name:
        logger.info(f"LLM Service: RSP endpoint (limited models, dedicated processing)")
    elif "MSLLMAPIClientAdapter" in client_name:
        logger.info(
            f"LLM Service: Microsoft LLM API (all internal models, automatic routing)"
        )
    elif "UnifiedLLMAPI" in client_name:
        logger.info(f"LLM Service: Unified client (intelligent model routing)")
    else:
        # Only show implementation details if it's not one of the standard types
        logger.info(f"Client implementation: {client_name}")
        logger.info(f"LLM Service: Custom implementation")

    logger.info(f"Processing configuration: threads={threads}, retries={retries}")

    # Process the data
    logger.info("Processing texts...")
    try:
        results = list(scorer.apply(data, show_progress=True))

        # Count successful and failed items
        successful = sum(1 for item in results if "error" not in item)
        failed = len(results) - successful

        if failed > 0:
            logger.warning(
                f"Processing Summary: {successful}/{len(results)} successful, {failed} failed"
            )
            logger.warning("Failed items contain 'error' field for easy identification")
        else:
            logger.info(
                f"Processing Summary: {successful}/{len(results)} items processed successfully"
            )

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved processed data to {output_file}")

        return failed == 0

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return False


def main(
    input_file: str = "examples/example_rsp_scoring_input.json",
    output_file: Optional[str] = None,
    threads: int = 3,
    retries: int = 3,
    max_items: int = -1,
    client_type: Optional[str] = None,
):
    """
    Main function for command line usage.

    Args:
        input_file: Path to input JSON file with text to score (default: examples/example_rsp_scoring_input.json)
        output_file: Path to output JSON file (default: input file with _output suffix)
        threads: Number of worker threads (default: 3)
        retries: Number of retries for failed calls (default: 3)
        max_items: Maximum number of items to process (-1 for all)

    Returns:
        bool: True if all items processed successfully, False if any failed
    """
    success = process_texts(
        input_file,
        output_file,
        threads=threads,
        retries=retries,
        max_items=max_items,
        client_type=client_type,
    )
    return success


if __name__ == "__main__":
    fire.Fire(main)
