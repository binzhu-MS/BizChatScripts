"""
Utterance complexity classifier using LLM.
This classifies utterances as requiring either chat model (simple)
or reasoning model (complex/COT) processing.
"""

import logging
import sys
import os
import json
import fire
import threading
from pathlib import Path

# Add parent directory to path to import llms
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llms import ChatCompletionLLMApplier, ApplicationModes, with_retries, prompts, util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UtteranceComplexityClassifier(ChatCompletionLLMApplier):
    """
    LLM applier that classifies utterances as requiring chat model (simple)
    or reasoning model (complex/COT) processing.
    """

    DEFAULT_PROMPT = prompts.get("utterance_complexity_classifier", "0.1.0")
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
        save_batch_size=100,
        incremental_save=True,
        output_file_path=None,
        total_utterances=0,
        retries=None,
        **kwargs,
    ):
        """Initialize with optional custom thread count, incremental saving, and retries."""
        self.save_batch_size = save_batch_size
        self.incremental_save = incremental_save
        self.output_file_path = output_file_path
        self.total_utterances = total_utterances
        self.processed_count = 0
        self.processed_by_thread = {}  # Track processed count per thread
        self.file_lock = threading.Lock()
        self.all_results = {}  # Store all results for incremental saving

        # Pass threads and retries to parent constructor
        kwargs["threads"] = threads
        kwargs["retries"] = retries

        super().__init__(**kwargs)

    def process_item(self, item, i):
        """Process a single utterance for complexity classification."""
        category = item.get("category", "")
        utterances = item.get("utterances", [])

        logger.info(f"Processing Category: {category} ({len(utterances)} utterances)")

        # Process each utterance in this category
        classified_utterances = []
        thread_id = threading.current_thread().ident

        # Initialize thread count if not exists
        if thread_id not in self.processed_by_thread:
            self.processed_by_thread[thread_id] = 0

        utterances_per_save = max(1, self.save_batch_size // self.DEFAULT_THREADS)

        for idx, utterance in enumerate(utterances):
            logger.debug(
                f"  Processing utterance #{idx + 1}/{len(utterances)}: {utterance[:80]}..."
            )
            try:
                classification = self.classify_utterance(utterance, category)
                if classification:
                    classified_utterances.append(
                        {
                            "utterance": utterance,
                            "classification": classification.get(
                                "classification", "unknown"
                            ),
                            "confidence": classification.get("confidence", 0.0),
                            "reasoning": classification.get("reasoning", ""),
                            "complexity_indicators": classification.get(
                                "complexity_indicators", []
                            ),
                        }
                    )
                    logger.debug(
                        f"Classified utterance #{idx + 1} in category {category}: {classification.get('classification', 'unknown')}"
                    )
                else:
                    logger.warning(
                        f"Failed to classify utterance #{idx + 1} in category {category}: {utterance[:100]}..."
                    )
                    classified_utterances.append(
                        {
                            "utterance": utterance,
                            "classification": "unknown",
                            "confidence": 0.0,
                            "reasoning": "Failed to classify",
                            "complexity_indicators": [],
                        }
                    )
            except Exception as e:
                logger.error(
                    f"Error classifying utterance #{idx + 1} in category {category}: {e}"
                )
                classified_utterances.append(
                    {
                        "utterance": utterance,
                        "classification": "error",
                        "confidence": 0.0,
                        "reasoning": f"Error: {str(e)}",
                        "complexity_indicators": [],
                    }
                )

            # Check if we should save incrementally
            if self.incremental_save:
                self.processed_by_thread[thread_id] += 1

                # Update global progress counter
                with self.file_lock:
                    self.processed_count += 1
                    current_progress = self.processed_count

                # Show progress every 10 utterances or at save points
                if (
                    current_progress % 10 == 0
                    or self.processed_by_thread[thread_id] >= utterances_per_save
                ):
                    progress_pct = (
                        (current_progress / self.total_utterances * 100)
                        if self.total_utterances > 0
                        else 0
                    )
                    logger.info(
                        f"📊 Progress: {current_progress}/{self.total_utterances} ({progress_pct:.1f}%) utterances processed"
                    )

                if self.processed_by_thread[thread_id] >= utterances_per_save:
                    self._save_incremental_results(
                        category, classified_utterances[: idx + 1]
                    )
                    self.processed_by_thread[thread_id] = 0  # Reset counter

        item["classified_utterances"] = classified_utterances

        # Log category completion summary
        successful_count = len(
            [
                u
                for u in classified_utterances
                if u["classification"] not in ["unknown", "error"]
            ]
        )
        failed_count = len(utterances) - successful_count
        if failed_count > 0:
            logger.warning(
                f"Category '{category}' completed: {successful_count}/{len(utterances)} successful, {failed_count} failed"
            )
        else:
            logger.info(
                f"Category '{category}' completed successfully: {successful_count}/{len(utterances)} utterances"
            )

        # Store results for final save if not incremental
        if not self.incremental_save:
            with self.file_lock:
                self.all_results[category] = classified_utterances

        return item

    def _save_incremental_results(self, category, classified_utterances):
        """Save results incrementally with file locking."""
        if not self.output_file_path:
            return

        with self.file_lock:
            try:
                # Read existing data if file exists
                existing_data = {}
                if os.path.exists(self.output_file_path):
                    try:
                        with open(self.output_file_path, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        existing_data = {}

                # Update with new results
                if category not in existing_data:
                    existing_data[category] = []

                # Add new utterances (avoiding duplicates by checking utterance text)
                existing_utterances = {
                    u.get("utterance", "") for u in existing_data[category]
                }
                for utterance_data in classified_utterances:
                    if utterance_data["utterance"] not in existing_utterances:
                        existing_data[category].append(utterance_data)

                # Save updated data
                with open(self.output_file_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)

                # Count total utterances saved so far
                total_saved = sum(
                    len(utterances) for utterances in existing_data.values()
                )
                new_count = len(
                    [
                        u
                        for u in classified_utterances
                        if u["utterance"] not in existing_utterances
                    ]
                )

                logger.info(
                    f"💾 Batch saved: +{new_count} utterances from '{category}' → Total saved: {total_saved} utterances"
                )

            except Exception as e:
                logger.error(f"Error in incremental save: {e}")

    @with_retries
    def classify_utterance(self, utterance, category):
        """Classify a single utterance's complexity."""
        variables = {"utterance": utterance, "category": category}

        # Format the prompt with variables
        formatted_prompt = prompts.formatting.render_messages(self.prompt, variables)

        # Call the LLM
        completion = self.llmapi.chat_completion(self.model_config, formatted_prompt)

        # Extract and parse response
        response_text = completion["choices"][0]["message"]["content"].strip()

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


def load_utterance_data(json_file_path, max_utterances=20):
    """Load utterance data from JSON file, limiting to first max_utterances for testing."""
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert the data structure to our expected format
        processed_items = []
        total_utterances = 0

        for category, utterances in data.items():
            # If max_utterances is -1, process all utterances
            if max_utterances == -1:
                processed_items.append({"category": category, "utterances": utterances})
                total_utterances += len(utterances)
            else:
                if total_utterances >= max_utterances:
                    break

                # Take only the utterances we need to reach the limit
                remaining_slots = max_utterances - total_utterances
                limited_utterances = utterances[:remaining_slots]

                if limited_utterances:  # Only add if there are utterances
                    processed_items.append(
                        {"category": category, "utterances": limited_utterances}
                    )
                    total_utterances += len(limited_utterances)

        return processed_items
    except Exception as e:
        logger.error(f"Error loading JSON file {json_file_path}: {e}")
        return []


def save_results_with_original_data(results, input_file_path, output_file_path):
    """Save classification results merged with original data to the specified output file."""
    try:
        output_path = Path(output_file_path)

        # Prepare output data in the original format but with added classification columns
        output_data = {}

        for result in results:
            category = result["category"]
            classified_utterances = result.get("classified_utterances", [])

            # Create the enhanced utterance list with original data + classification
            enhanced_utterances = []
            for utterance_data in classified_utterances:
                # Create enhanced utterance entry with original text + classification columns
                enhanced_entry = {
                    "utterance": utterance_data["utterance"],
                    "classification": utterance_data["classification"],
                    "confidence": utterance_data["confidence"],
                    "reasoning": utterance_data["reasoning"],
                    "complexity_indicators": utterance_data["complexity_indicators"],
                }
                enhanced_utterances.append(enhanced_entry)

            output_data[category] = enhanced_utterances

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Labeled results saved to {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")
        return None


def save_results(results, output_file_path):
    """Save classification results to JSON file (legacy format for backward compatibility)."""
    try:
        # Prepare output data
        output_data = {}

        for result in results:
            category = result["category"]
            classified_utterances = result.get("classified_utterances", [])

            output_data[category] = {
                "total_utterances": len(classified_utterances),
                "chat_model_count": len(
                    [u for u in classified_utterances if u["classification"] == "chat"]
                ),
                "reasoning_model_count": len(
                    [u for u in classified_utterances if u["classification"] == "cot"]
                ),
                "utterances": classified_utterances,
            }

        # Save to file
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving results to {output_file_path}: {e}")
        return False


def print_summary(results):
    """Print a summary of classification results."""
    total_utterances = 0
    total_chat = 0
    total_cot = 0
    total_errors = 0

    print("\nClassification Summary by Category:")
    print("=" * 80)

    for result in results:
        category = result["category"]
        classified_utterances = result.get("classified_utterances", [])

        chat_count = len(
            [u for u in classified_utterances if u["classification"] == "chat"]
        )
        cot_count = len(
            [u for u in classified_utterances if u["classification"] == "cot"]
        )
        error_count = len(
            [
                u
                for u in classified_utterances
                if u["classification"] in ["unknown", "error"]
            ]
        )

        total_utterances += len(classified_utterances)
        total_chat += chat_count
        total_cot += cot_count
        total_errors += error_count

        print(f"\nCategory: {category}")
        print(f"  Total: {len(classified_utterances)}")

        # Handle empty categories to avoid division by zero
        if len(classified_utterances) > 0:
            print(
                f"  Chat Model: {chat_count} ({chat_count/len(classified_utterances)*100:.1f}%)"
            )
            print(
                f"  Reasoning Model: {cot_count} ({cot_count/len(classified_utterances)*100:.1f}%)"
            )
            if error_count > 0:
                print(
                    f"  Errors: {error_count} ({error_count/len(classified_utterances)*100:.1f}%)"
                )
        else:
            print("  ⚠️  No utterances were successfully classified in this category")
            print("  Check logs for processing errors or authentication issues")

    print(f"\nOverall Summary:")
    print(f"  Total Utterances: {total_utterances}")

    # Handle case where no utterances were successfully processed
    if total_utterances > 0:
        print(f"  Chat Model: {total_chat} ({total_chat/total_utterances*100:.1f}%)")
        print(f"  Reasoning Model: {total_cot} ({total_cot/total_utterances*100:.1f}%)")
        if total_errors > 0:
            print(
                f"  Errors: {total_errors} ({total_errors/total_utterances*100:.1f}%)"
            )
    else:
        print("  ⚠️  No utterances were successfully processed!")
        print(
            "  This may indicate authentication issues, network problems, or input file format issues."
        )
        print("  Check the logs above for specific error messages.")


def classify_utterances(
    input_file,
    max_utterances=20,
    output_file=None,
    threads=3,
    retries=3,
    save_batch_size=100,
    incremental_save=True,
):
    """
    Run the utterance complexity classifier with customizable parameters.

    Args:
        input_file: Path to input JSON file (required)
        max_utterances: Maximum number of utterances to process (default: 20 for testing)
                       Set to -1 or a very large number to process ALL utterances in the file
        output_file: Output file path (default: input file name with "_labeled" suffix)
        threads: Number of worker threads to use for processing (default: 3)
        retries: Number of retries for each LLM API call in case of failure (default: 3)
        save_batch_size: Number of utterances to process before saving incrementally (default: 100)
        incremental_save: Whether to save results incrementally during processing (default: True)
                         Set to False to save only at the end (faster for small datasets)

    Examples:
        # Process only first 20 utterances (default for testing)
        classify_utterances("path/to/input.json")

        # Process ALL utterances in the file with incremental saving every 50 utterances
        classify_utterances("path/to/input.json", max_utterances=-1, save_batch_size=50)

        # Process first 1000 utterances with 8 threads, save at end only
        classify_utterances("path/to/input.json", max_utterances=1000, threads=8, incremental_save=False)

        # Process all utterances with 5 retries per API call
        classify_utterances("path/to/input.json", max_utterances=-1, retries=5)

        # Process all utterances with custom output file and save every 200 utterances
        classify_utterances("path/to/input.json", max_utterances=-1, output_file="results.json", save_batch_size=200)
    """

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logger.info(f"Input file: {input_file}")
    logger.info(f"Max utterances: {max_utterances}")
    logger.info(f"Worker threads: {threads}")
    logger.info(f"Retries per API call: {retries}")
    logger.info(f"Save batch size: {save_batch_size}")
    logger.info(f"Incremental save: {incremental_save}")

    # Prepare output file path
    if output_file is None:
        # Default: input file name with "_labeled" suffix
        input_path = Path(input_file)
        output_file_path = (
            input_path.parent / f"{input_path.stem}_labeled{input_path.suffix}"
        )
    else:
        # Use provided output file path
        output_file_path = Path(output_file)

    logger.info(f"Output file: {output_file_path}")

    # Load utterance data
    if max_utterances == -1:
        logger.info(f"Loading ALL utterances from the dataset...")
    else:
        logger.info(f"Loading utterance data (first {max_utterances} utterances)...")
    utterance_data = load_utterance_data(input_file, max_utterances=max_utterances)

    if not utterance_data:
        logger.error("No data loaded. Please check the input file path and format.")
        return

    total_utterances = sum(len(item["utterances"]) for item in utterance_data)
    logger.info(
        f"Loaded {len(utterance_data)} categories with {total_utterances} utterances total"
    )

    # Create and run the classifier
    classifier_kwargs = {
        "threads": threads,
        "retries": retries,
        "save_batch_size": save_batch_size,
        "incremental_save": incremental_save,
        "output_file_path": str(output_file_path),
        "total_utterances": total_utterances,
    }

    classifier = UtteranceComplexityClassifier(**classifier_kwargs)

    # Log the model configuration being used
    final_model_config = getattr(
        classifier, "model_config", classifier.DEFAULT_MODEL_CONFIG
    )
    logger.info(f"Model configuration: {final_model_config}")

    logger.info("Running Utterance Complexity Classifier...")
    if incremental_save:
        logger.info(
            f"Results will be saved incrementally every ~{save_batch_size // threads} utterances per thread"
        )
        # Initialize empty output file
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
    else:
        logger.info("Results will be saved only at the end")
    logger.info("=" * 60)

    # Process items
    results = list(classifier.apply(utterance_data))

    # Print summary
    print_summary(results)

    # Save results (final save for non-incremental, or final cleanup for incremental)
    if not incremental_save:
        save_results_with_original_data(results, input_file, output_file_path)
    else:
        logger.info(f"Final results are in: {output_file_path}")

    # Return results only when called programmatically, not from Fire
    return None  # Don't return results to avoid Fire printing them


def main():
    """Main entry point that doesn't return results to avoid Fire printing them."""
    fire.Fire(classify_utterances)


if __name__ == "__main__":
    main()
