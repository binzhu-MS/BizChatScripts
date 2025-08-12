"""
Utterance personalizer using LLM to generate templates with entity variables.
Reads utterances from a TSV file and outputs personalized templates.
"""

import json
import logging
import sys
import os
import fire
import pandas as pd
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add parent directory to path to import llms
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llms import ChatCompletionLLMApplier, ApplicationModes, with_retries, prompts, util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UtterancePersonalizer(ChatCompletionLLMApplier):
    DEFAULT_PROMPT = prompts.get("utterance_personalizer", "0.1.0")
    DEFAULT_MODEL_CONFIG = {
        "model": "dev-gpt-41-longco-2025-04-14",
        "temperature": 0.1,
        "max_tokens": 1000,
    }
    DEFAULT_THREADS = 3
    DEFAULT_RETRIES = 3
    APPLICATION_MODE = ApplicationModes.PerItem

    def __init__(
        self,
        threads=None,
        retries=None,
        **kwargs,
    ):
        """Initialize with optional custom thread count and retries."""
        self.threads = threads if threads is not None else self.DEFAULT_THREADS
        self.retries = retries if retries is not None else self.DEFAULT_RETRIES

        super().__init__(**kwargs)

    @with_retries
    def personalize_utterance(self, utterance: str) -> str:
        """
        Personalize a single utterance and return only the template string.
        """
        variables = {
            "utterance": utterance,
        }
        formatted_prompt = prompts.formatting.render_messages(self.prompt, variables)
        completion = self.llmapi.chat_completion(self.model_config, formatted_prompt)
        response = completion["choices"][0]["message"]["content"].strip()

        # Debug: log the raw response
        logger.debug(f"Raw LLM response for '{utterance[:50]}...': {response}")

        try:
            # Handle responses wrapped in markdown code blocks
            json_content = response
            if response.startswith("```json"):
                # Extract content between ```json and ```
                lines = response.split("\n")
                start_idx = None
                end_idx = len(lines)

                for i, line in enumerate(lines):
                    if line.strip() == "```json":
                        start_idx = i + 1
                    elif (
                        line.strip() == "```"
                        and start_idx is not None
                        and i > start_idx
                    ):
                        end_idx = i
                        break

                if start_idx is not None:
                    json_content = "\n".join(lines[start_idx:end_idx])
                else:
                    # Fallback if no proper start marker found
                    json_content = response

            template_data = json.loads(json_content)
            return template_data.get("template", utterance)
        except Exception as e:
            logger.warning(
                f"Failed to parse LLM response for '{utterance[:50]}...': {e}"
            )
            logger.warning(f"Raw response was: {response[:200]}...")
            raise  # Re-raise the exception so with_retries can catch it

    def personalize_utterance_with_error_handling(self, utterance: str) -> str:
        """
        Wrapper method that handles errors and returns <error> indicator for failed cases.
        """
        try:
            return self.personalize_utterance(utterance)
        except Exception as e:
            logger.error(
                f"Failed to personalize utterance after {self.retries} retries: '{utterance[:100]}...' - Error: {e}"
            )
            return (
                f"<error>{utterance}</error>"  # Error indicator with original utterance
            )


def process_utterances(
    input_file: str,
    output_file: str,
    threads: int = 3,
    retries: int = 3,
):
    """
    Process a TSV file, replacing the second column with templated utterances.

    Args:
        input_file: Path to input TSV file
        output_file: Path to output TSV file
        threads: Number of worker threads to use for processing (default: 3)
        retries: Number of retries for failed LLM calls (default: 3)

    Examples:
        # Basic usage with defaults
        process_utterances("input.tsv", "output.tsv")

        # With custom threading and retry settings
        process_utterances("input.tsv", "output.tsv", threads=8, retries=5)
    """
    logger.info(f"Loading data from {input_file}")

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read the TSV file
    df = pd.read_csv(input_file, sep="\t")
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    # Ensure we have at least 2 columns
    if len(df.columns) < 2:
        logger.error("Input file must have at least 2 columns")
        return

    # Rename columns to match expected format
    # First column becomes "query", second becomes "raw_query_pattern"
    new_columns = list(df.columns)
    new_columns[0] = "query"
    new_columns[1] = "raw_query_pattern"

    # Convert all other column names to lowercase
    for i in range(2, len(new_columns)):
        new_columns[i] = new_columns[i].lower()

    df.columns = new_columns

    # Ensure the target column is string type to avoid dtype warnings
    df["raw_query_pattern"] = df["raw_query_pattern"].astype(str)

    # Initialize personalizer with configuration
    personalizer = UtterancePersonalizer(threads=threads, retries=retries)

    # Log the model configuration being used (actual config from the instance)
    logger.info(f"Model configuration: {personalizer.model_config}")
    logger.info(f"Configuration: threads={threads}, retries={retries}")

    # Process each utterance in the first column and update directly
    logger.info("Processing utterances...")
    error_count = 0
    total_processed = 0
    format_error_count = 0

    for i, utterance in enumerate(df["query"]):
        if pd.isna(utterance) or utterance.strip() == "":
            # Handle empty/NaN values - keep as is
            df.loc[i, "raw_query_pattern"] = str(utterance)
        else:
            # Validate that utterance is a string
            if not isinstance(utterance, str):
                logger.error(
                    f"Row {i+1}: Wrong input format - utterance must be string, got {type(utterance).__name__}: {utterance}"
                )
                df.loc[i, "raw_query_pattern"] = (
                    f"<error>Wrong input format: expected string, got {type(utterance).__name__}</error>"
                )
                format_error_count += 1
                total_processed += 1
                error_count += 1
            else:
                template = personalizer.personalize_utterance_with_error_handling(
                    str(utterance).strip()
                )
                df.loc[i, "raw_query_pattern"] = template
                total_processed += 1

                # Count errors
                if template.startswith("<error>") and template.endswith("</error>"):
                    error_count += 1

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(df)} utterances")

    # Print summary first
    success_count = total_processed - error_count

    if format_error_count > 0:
        logger.error(
            f"Input Format Errors: {format_error_count} rows had non-string utterances"
        )

    if error_count > 0:
        processing_error_count = error_count - format_error_count
        logger.warning(
            f"Processing Summary: {success_count}/{total_processed} successful, {error_count} failed"
        )
        if format_error_count > 0:
            logger.warning(
                f"  - {format_error_count} input format errors (non-string values)"
            )
        if processing_error_count > 0:
            logger.warning(f"  - {processing_error_count} LLM processing errors")
        logger.warning(
            f"Failed utterances marked with <error>...</error> for easy identification"
        )
    else:
        logger.info(
            f"Processing Summary: {success_count}/{total_processed} utterances processed successfully"
        )

    # Save to output file
    df.to_csv(output_file, sep="\t", index=False)
    logger.info(f"Saved processed data to {output_file}")
    logger.info(f"Output columns: {list(df.columns)}")

    # Return success/failure status
    return error_count == 0


def main(input_file: str, output_file: str, threads: int = 3, retries: int = 3):
    """
    Main function for command line usage and debugging.

    Args:
        input_file: Path to input TSV file
        output_file: Path to output TSV file
        threads: Number of worker threads (default: 3)
        retries: Number of retries for failed calls (default: 3)

    Returns:
        bool: True if all utterances processed successfully, False if any failed
    """
    success = process_utterances(
        input_file, output_file, threads=threads, retries=retries
    )
    return success


if __name__ == "__main__":
    fire.Fire(main)
