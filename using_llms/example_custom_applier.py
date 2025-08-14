"""
Custom LLM applier example showing how to create your own LLM application.
This demonstrates the flexibility of the framework for various use cases.
"""

import logging
import sys
import os
import json
import fire
from pathlib import Path
from typing import Optional

from llms import ChatCompletionLLMApplier, ApplicationModes, with_retries, prompts, util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer(ChatCompletionLLMApplier):
    """
    Custom LLM applier that analyzes sentiment of text.
    Shows how to create your own prompts and processing logic.
    """

    DEFAULT_PROMPT = prompts.get("example_custom_applier", "0.1.0")

    DEFAULT_MODEL_CONFIG = {
        "model": "dev-gpt-41-longco-2025-04-14",
        "temperature": 0.1,
        "max_tokens": 300,
    }
    DEFAULT_THREADS = 3
    DEFAULT_RETRIES = 3
    APPLICATION_MODE = ApplicationModes.PerItem

    def process_item(self, item, i):
        """Process a single item for sentiment analysis."""
        text = item.get("text", "")

        try:
            sentiment_result = self.analyze_sentiment(text)
            if sentiment_result:
                item["sentiment_analysis"] = sentiment_result
                logger.info(
                    f"Analyzed item {i}: {sentiment_result.get('sentiment', 'N/A')}"
                )
        except Exception as e:
            logger.error(f"Error analyzing sentiment for item {i}: {e}")

        return item

    @with_retries
    def analyze_sentiment(self, text):
        """Analyze sentiment of the given text."""
        variables = {"text": text}

        # Format prompt
        formatted_prompt = prompts.formatting.render_messages(self.prompt, variables)

        # Call LLM
        completion = self.llmapi.chat_completion(self.model_config, formatted_prompt)
        response_text = completion["choices"][0]["message"]["content"]

        # Parse response
        cleaned_response = util.clean_json_response(response_text)

        try:
            return util.safe_json_loads(cleaned_response)
        except Exception as e:
            logger.warning(f"Failed to parse sentiment response: {e}")
            return None


class CodeReviewer(ChatCompletionLLMApplier):
    """
    Another custom applier that reviews code for potential issues.
    Uses inline prompt to demonstrate multiple prompt approaches in one example.
    """

    DEFAULT_PROMPT = {
        "messages": [
            {
                "role": "system",
                "content": "You are a senior software engineer conducting code reviews.",
            },
            {
                "role": "user",
                "content": """Review the following {{{language}}} code for potential issues:

```{{{language}}}
{{{code}}}
```

Provide your review in JSON format:
```json
{
  "overall_quality": "<excellent|good|fair|poor>",
  "issues": [
    {
      "type": "<bug|style|performance|security>",
      "severity": "<high|medium|low>", 
      "description": "<issue description>",
      "suggestion": "<how to fix>"
    }
  ],
  "positive_aspects": ["<aspect1>", "<aspect2>"],
  "summary": "<overall assessment>"
}
```""",
            },
        ]
    }

    DEFAULT_MODEL_CONFIG = {
        "model": "dev-gpt-41-longco-2025-04-14",
        "temperature": 0.2,
        "max_tokens": 1000,
    }
    DEFAULT_THREADS = 2
    DEFAULT_RETRIES = 3
    APPLICATION_MODE = ApplicationModes.PerItem

    def process_item(self, item, i):
        """Process a single code item for review."""
        code = item.get("code", "")
        language = item.get("language", "python")

        try:
            review_result = self.review_code(code, language)
            if review_result:
                item["code_review"] = review_result
                logger.info(
                    f"Reviewed item {i}: {review_result.get('overall_quality', 'N/A')}"
                )
        except Exception as e:
            logger.error(f"Error reviewing code for item {i}: {e}")

        return item

    @with_retries
    def review_code(self, code, language):
        """Review the given code."""
        variables = {"code": code, "language": language}

        # Format prompt
        formatted_prompt = prompts.formatting.render_messages(self.prompt, variables)

        # Call LLM
        completion = self.llmapi.chat_completion(self.model_config, formatted_prompt)
        response_text = completion["choices"][0]["message"]["content"]

        # Parse response
        cleaned_response = util.clean_json_response(response_text)

        try:
            return util.safe_json_loads(cleaned_response)
        except Exception as e:
            logger.warning(f"Failed to parse code review response: {e}")
            return None


def main():
    """Run the custom appliers example with hardcoded demo data."""

    print("Custom LLM Appliers Example (Demo Mode)")
    print("=" * 40)

    # Test sentiment analysis
    print("\n1. Sentiment Analysis Example:")
    print("-" * 30)

    sentiment_data = [
        {
            "type": "sentiment",
            "text": "I absolutely love this new feature! It's amazing and works perfectly.",
        },
        {
            "type": "sentiment",
            "text": "This is the worst experience I've ever had. Nothing works as expected.",
        },
        {
            "type": "sentiment",
            "text": "The weather today is partly cloudy with a chance of rain.",
        },
    ]

    sentiment_analyzer = SentimentAnalyzer()

    # Process only sentiment items
    sentiment_items = [
        item for item in sentiment_data if item.get("type") == "sentiment"
    ]
    sentiment_results = list(sentiment_analyzer.apply(sentiment_items))

    for result in sentiment_results:
        analysis = result.get("sentiment_analysis", {})
        print(f"Text: {result['text'][:50]}...")
        print(f"Sentiment: {analysis.get('sentiment', 'N/A')}")
        print(f"Confidence: {analysis.get('confidence', 'N/A')}")
        print()

    # Test code review
    print("\n2. Code Review Example:")
    print("-" * 30)

    code_data = [
        {
            "type": "code_review",
            "code": """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
""",
            "language": "python",
        },
        {
            "type": "code_review",
            "code": """
function getUserById(id) {
    var users = getAllUsers();
    for (var i = 0; i < users.length; i++) {
        if (users[i].id == id) {
            return users[i];
        }
    }
}
""",
            "language": "javascript",
        },
    ]

    code_reviewer = CodeReviewer()

    # Process only code review items
    code_items = [item for item in code_data if item.get("type") == "code_review"]
    review_results = list(code_reviewer.apply(code_items))

    for i, result in enumerate(review_results):
        review = result.get("code_review", {})
        print(f"Code Sample {i+1}:")
        print(f"Quality: {review.get('overall_quality', 'N/A')}")
        print(f"Issues: {len(review.get('issues', []))}")
        if review.get("summary"):
            print(f"Summary: {review['summary']}")
        print()


def process_file(
    input_file: str = "examples/example_custom_applier_input.json",
    output_file: Optional[str] = None,
    client_type: str = "rsp",
    threads: int = 2,
    retries: int = 3,
    max_items: Optional[int] = None,
):
    """
    Process custom applier tasks from file input.

    Args:
        input_file: Path to input JSON file with mixed sentiment and code review data (default: examples/example_custom_applier_input.json)
        output_file: Path to output JSON file (default: input file with _output suffix)
        client_type: LLM client type ('rsp', 'ms_llm_client', 'unified') (default: 'rsp')
        threads: Number of threads for parallel processing (default: 2)
        retries: Number of retries for failed requests (default: 3)
        max_items: Maximum number of items to process (optional)
    """

    # Set up output file path
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.with_stem(input_path.stem + "_output"))

    logger.info(f"Loading data from {input_file}")

    # Load input data
    try:
        if not os.path.exists(input_file):
            logger.error(f"Input file does not exist: {input_file}")
            raise FileNotFoundError(f"Input file not found: {input_file}")

        with open(input_file, "r", encoding="utf-8") as f:
            test_items = json.load(f)

        if not test_items:
            logger.error(f"No data loaded from {input_file}")
            return

        logger.info(f"Loaded {len(test_items)} items from {input_file}")

    except Exception as e:
        logger.error(f"Error loading input file {input_file}: {e}")
        raise

    # Apply max_items limit if specified
    if max_items and max_items > 0:
        test_items = test_items[:max_items]
        logger.info(f"Processing limited to {len(test_items)} items")

    # Separate items by type
    sentiment_items = [item for item in test_items if item.get("type") == "sentiment"]
    code_items = [item for item in test_items if item.get("type") == "code_review"]

    all_results = []

    # Process sentiment items
    if sentiment_items:
        print(f"Processing {len(sentiment_items)} sentiment analysis items...")
        logger.info(f"Client Type: {client_type}")

        sentiment_analyzer = SentimentAnalyzer(
            threads=threads, retries=retries, client_type=client_type
        )

        sentiment_results = list(sentiment_analyzer.apply(sentiment_items))
        all_results.extend(sentiment_results)

        logger.info(f"Completed sentiment analysis for {len(sentiment_results)} items")

    # Process code review items
    if code_items:
        print(f"Processing {len(code_items)} code review items...")
        logger.info(f"Client Type: {client_type}")

        code_reviewer = CodeReviewer(
            threads=threads, retries=retries, client_type=client_type
        )

        review_results = list(code_reviewer.apply(code_items))
        all_results.extend(review_results)

        logger.info(f"Completed code review for {len(review_results)} items")

    # Save results
    logger.info(f"Saving results to {output_file}")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")
        raise

    print(f"\nProcessing complete! Results saved to: {output_file}")
    return all_results


def demo(
    client_type: str = "rsp",
):
    """
    Run demo mode with hardcoded test data.

    Args:
        client_type: LLM client type ('rsp', 'ms_llm_client', 'unified') (default: 'rsp')
    """
    logger.info(f"Client Type: {client_type}")
    main()


if __name__ == "__main__":
    fire.Fire(
        {
            "demo": demo,
            "process": process_file,
        }
    )
