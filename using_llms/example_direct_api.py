"""
Direct LLM API usage example using the BizChatScripts framework.
This demonstrates how to use the LLM API client directly for custom applications
that need fine-grained control over API interactions.
"""

import json
import logging
import sys
import os
import fire
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path to import llms
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llms import LLMAPI, util
from llms.auth import LLMAPI as LLMAuthAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DirectLLMExample:
    """
    Example demonstrating direct LLM API usage without the applier framework.
    This is useful when you need maximum control over the API interactions.
    """

    def __init__(self, model="dev-gpt-41-longco-2025-04-14"):
        """Initialize with authentication and model configuration."""
        self.llm_api = LLMAPI()
        self.model = model

        # Default model configuration
        self.model_config = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 1500,
            "top_p": 0.9,
        }

    def simple_completion(self, prompt, **kwargs):
        """
        Get a simple completion from the LLM.

        Args:
            prompt (str): The user prompt
            **kwargs: Additional model configuration parameters

        Returns:
            str: The LLM response text
        """
        # Build messages in OpenAI format
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # Merge any custom parameters
        config = {**self.model_config, **kwargs}

        try:
            # Make the API call
            response = self.llm_api.chat_completion(config, messages)

            # Extract the response text
            return response["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Error in simple_completion: {e}")
            raise

    def structured_conversation(self, conversation_history):
        """
        Conduct a structured conversation with the LLM.

        Args:
            conversation_history (list): List of message dictionaries

        Returns:
            dict: Full API response including usage statistics
        """
        try:
            # Make the API call with conversation history
            response = self.llm_api.chat_completion(
                self.model_config, conversation_history
            )
            return response

        except Exception as e:
            logger.error(f"Error in structured_conversation: {e}")
            raise

    def batch_completions(self, prompts, batch_size=5):
        """
        Process multiple prompts in batches.

        Args:
            prompts (list): List of prompt strings
            batch_size (int): Number of concurrent requests

        Returns:
            list: List of responses corresponding to each prompt
        """
        results = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            batch_results = []

            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}"
            )

            for prompt in batch:
                try:
                    result = self.simple_completion(prompt)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing prompt: {e}")
                    batch_results.append(f"Error: {str(e)}")

            results.extend(batch_results)

        return results

    def custom_parameters_example(self):
        """
        Demonstrate using custom model parameters for different use cases.
        """
        examples = []

        # Creative writing (high temperature)
        creative_prompt = "Write a short, creative story about a robot discovering art."
        creative_result = self.simple_completion(
            creative_prompt, temperature=0.8, max_tokens=200
        )
        examples.append(("Creative Writing", creative_result))

        # Analytical task (low temperature)
        analytical_prompt = (
            "Analyze the pros and cons of remote work in exactly 3 bullet points."
        )
        analytical_result = self.simple_completion(
            analytical_prompt, temperature=0.1, max_tokens=300
        )
        examples.append(("Analytical Task", analytical_result))

        # Code generation (medium temperature)
        code_prompt = "Write a Python function that calculates the factorial of a number with error handling."
        code_result = self.simple_completion(
            code_prompt, temperature=0.3, max_tokens=400
        )
        examples.append(("Code Generation", code_result))

        return examples

    def conversation_with_context(self):
        """
        Demonstrate a multi-turn conversation with maintained context.
        """
        # Build conversation history
        conversation = [
            {"role": "system", "content": "You are a knowledgeable science tutor."},
            {"role": "user", "content": "Explain photosynthesis in simple terms."},
        ]

        # First response
        response1 = self.structured_conversation(conversation)
        assistant_reply1 = response1["choices"][0]["message"]["content"]
        conversation.append({"role": "assistant", "content": assistant_reply1})

        # Follow-up question
        conversation.append(
            {
                "role": "user",
                "content": "Now explain how this relates to the oxygen we breathe.",
            }
        )

        # Second response with context
        response2 = self.structured_conversation(conversation)
        assistant_reply2 = response2["choices"][0]["message"]["content"]

        return [
            ("Initial Question", "Explain photosynthesis in simple terms."),
            ("Initial Response", assistant_reply1),
            (
                "Follow-up Question",
                "Now explain how this relates to the oxygen we breathe.",
            ),
            ("Contextual Response", assistant_reply2),
            (
                "Token Usage",
                f"Total tokens: {response2.get('usage', {}).get('total_tokens', 'N/A')}",
            ),
        ]


def process_prompts(
    input_file: str,
    output_file: Optional[str] = None,
    model: str = "dev-gpt-41-longco-2025-04-14",
    temperature: float = 0.2,
    max_tokens: int = 1500,
):
    """
    Process a JSON file with direct LLM API requests.

    Args:
        input_file: Path to input JSON file with prompts to process
        output_file: Path to output JSON file (default: input file with _output suffix)
        model: Model name to use (default: dev-gpt-41-longco-2025-04-14)
        temperature: Temperature parameter (default: 0.2)
        max_tokens: Maximum tokens to generate (default: 1500)

    Examples:
        # Basic usage with defaults
        process_prompts("examples/example_direct_api_input.json")

        # With custom parameters
        process_prompts("data.json", "results.json", temperature=0.5, max_tokens=2000)
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

    # Initialize direct LLM example with custom model
    llm_example = DirectLLMExample(model=model)

    # Override model config with custom parameters
    llm_example.model_config.update(
        {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    )

    # Log the model configuration and client information being used
    logger.info(f"Model configuration: {llm_example.model_config}")
    logger.info(f"Client implementation: {type(llm_example.llm_api).__name__}")

    # Determine and log the functional capabilities being used
    client_name = type(llm_example.llm_api).__name__
    if "RSP" in client_name or "LLMAPI" == client_name:
        logger.info(f"LLM Service: RSP endpoint (limited models, dedicated processing)")
    elif "MSLLMAPIClientAdapter" in client_name:
        logger.info(
            f"LLM Service: Microsoft LLM API (all internal models, automatic routing)"
        )
    elif "UnifiedLLMAPI" in client_name:
        logger.info(f"LLM Service: Unified client (intelligent model routing)")
    else:
        logger.info(f"LLM Service: {client_name}")

    logger.info(
        f"Direct API usage: bypassing framework's threading and retry mechanisms"
    )

    # Process the data
    logger.info("Processing prompts...")
    results = []
    error_count = 0

    for i, item in enumerate(data):
        prompt = item.get("prompt", "")
        item_id = item.get("id", f"item_{i}")

        if not prompt.strip():
            logger.warning(f"Empty prompt for item {i}")
            item["error"] = "No prompt provided"
            results.append(item)
            error_count += 1
            continue

        try:
            # Process the prompt using simple completion
            response = llm_example.simple_completion(prompt)
            item["response"] = response
            logger.info(f"Processed item {i}: {len(response)} characters")
        except Exception as e:
            logger.error(f"Error processing item {i}: {e}")
            item["error"] = str(e)
            error_count += 1

        results.append(item)

    # Log processing summary
    successful = len(results) - error_count
    if error_count > 0:
        logger.warning(
            f"Processing Summary: {successful}/{len(results)} successful, {error_count} failed"
        )
        logger.warning("Failed items contain 'error' field for easy identification")
    else:
        logger.info(
            f"Processing Summary: {successful}/{len(results)} items processed successfully"
        )

    # Save results
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved processed data to {output_file}")
        return error_count == 0

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False


def main(
    input_file: str = "examples/example_direct_api_input.json",
    output_file: Optional[str] = None,
    model: str = "dev-gpt-41-longco-2025-04-14",
    temperature: float = 0.2,
    max_tokens: int = 1500,
):
    """
    Main function for command line usage.

    Args:
        input_file: Path to input JSON file with prompts (default: examples/example_direct_api_input.json)
        output_file: Path to output JSON file (default: input file with _output suffix)
        model: Model name to use (default: dev-gpt-41-longco-2025-04-14)
        temperature: Temperature parameter (default: 0.2)
        max_tokens: Maximum tokens to generate (default: 1500)

    Returns:
        bool: True if all items processed successfully, False if any failed
    """
    success = process_prompts(
        input_file,
        output_file,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return success


if __name__ == "__main__":
    fire.Fire(main)
