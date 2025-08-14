"""
Simple text scoring example using the SimpleLLM framework.
This shows how to create a basic LLM applier that scores text relevance.
"""

import logging
import sys
import os
import fire
from typing import Optional

# Add parent directory to path to import simplellm
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llms import ChatCompletionLLMApplier, ApplicationModes, with_retries, prompts, util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTextScorer(ChatCompletionLLMApplier):
    """
    Example LLM applier that scores how well text addresses a user request.
    """

    DEFAULT_PROMPT = prompts.get("example_simple_scorer", "0.1.0")
    DEFAULT_MODEL_CONFIG = {
        "model": "dev-gpt-41-longco-2025-04-14",
        "temperature": 0.1,
        "max_tokens": 500,
    }
    DEFAULT_THREADS = 2
    DEFAULT_RETRIES = 3
    APPLICATION_MODE = ApplicationModes.PerItem

    def __init__(self, threads=None, retries=None, client_type="rsp"):
        """Initialize the Simple Text Scorer.

        Args:
            threads (int): Number of threads for parallel processing
            retries (int): Number of retry attempts for failed requests
            client_type (str): Type of LLM client to use ('rsp', 'ms_llm_client', 'unified')
        """
        super().__init__(threads=threads, retries=retries, client_type=client_type)

    def process_item(self, item, i):
        """Process a single item for scoring."""
        user_request = item.get("user_request", "")
        text_to_evaluate = item.get("text_to_evaluate", "")

        # Score the text
        try:
            score_result = self.score_text(user_request, text_to_evaluate)
            if score_result:
                item["score_result"] = score_result
                logger.info(f"Scored item {i}: {score_result.get('score', 'N/A')}")
        except Exception as e:
            logger.error(f"Error scoring item {i}: {e}")

        return item

    @with_retries
    def score_text(self, user_request, text_to_evaluate):
        """Score how well text addresses the user request."""
        variables = {"user_request": user_request, "text_to_evaluate": text_to_evaluate}

        # Format the prompt with variables
        formatted_prompt = prompts.formatting.render_messages(self.prompt, variables)

        # Call the LLM
        completion = self.llmapi.chat_completion(self.model_config, formatted_prompt)

        # Extract response
        response_text = completion["choices"][0]["message"]["content"]

        # Clean and parse JSON response
        cleaned_response = util.clean_json_response(response_text)

        if not cleaned_response:
            logger.warning("Empty response after cleaning")
            return None

        try:
            return util.safe_json_loads(cleaned_response)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response_text}")
            return None


def process_texts(
    input_file: str = "examples/example_simple_scorer_input.json",
    output_file: Optional[str] = None,
    threads=2,
    retries=3,
    client_type="rsp",
):
    """
    Process texts from input file with scoring.

    Args:
        input_file: Path to input JSON file with text to score (default: examples/example_simple_scorer_input.json)
        output_file: Path to output JSON file (default: input file with _output suffix)
        threads (int): Number of threads for parallel processing
        retries (int): Number of retry attempts for failed requests
        client_type (str): Type of LLM client to use ('rsp', 'ms_llm_client', 'unified')
    """
    import json
    from pathlib import Path

    # Set default output file if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.with_stem(input_path.stem + "_output"))

    logger.info(f"Loading data from {input_file}")

    # Load data from input file
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            test_items = json.load(f)

        if not test_items:
            logger.error(f"No data loaded from {input_file}")
            return

        logger.info(f"Loaded {len(test_items)} items from {input_file}")
    except Exception as e:
        logger.error(f"Error loading input file {input_file}: {e}")
        return

    logger.info("Starting Simple Text Scorer...")
    logger.info(f"Input items: {len(test_items)}")

    # Initialize scorer with configuration
    scorer = SimpleTextScorer(threads=threads, retries=retries, client_type=client_type)

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

    print("Running Simple Text Scorer Example...")
    print("=" * 50)

    # Process items
    results = list(scorer.apply(test_items))

    # Display results
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Request: {result['user_request']}")
        print(f"Text: {result['text_to_evaluate'][:100]}...")

        if "score_result" in result:
            score_info = result["score_result"]
            print(f"Score: {score_info.get('score', 'N/A')}/10")
            print(f"Reasoning: {score_info.get('reasoning', 'N/A')}")
        else:
            print("Score: Failed to score")
        print("-" * 30)

    logger.info(f"Completed processing {len(results)} items")

    # Save results to output file
    logger.info(f"Saving results to {output_file}")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")

    return results


def main(
    input_file: str = "examples/example_simple_scorer_input.json",
    output_file: Optional[str] = None,
    threads=2,
    retries=3,
    client_type="rsp",
):
    """
    Run the simple scorer example.

    Args:
        input_file: Path to input JSON file with text to score (default: examples/example_simple_scorer_input.json)
        output_file: Path to output JSON file (default: input file with _output suffix)
        threads (int): Number of threads for parallel processing
        retries (int): Number of retry attempts for failed requests
        client_type (str): Type of LLM client to use ('rsp', 'ms_llm_client', 'unified')
    """
    return process_texts(
        input_file=input_file,
        output_file=output_file,
        threads=threads,
        retries=retries,
        client_type=client_type,
    )


if __name__ == "__main__":
    fire.Fire(main)
