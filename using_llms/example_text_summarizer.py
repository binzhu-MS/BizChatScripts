"""
Text summarization example using the SimpleLLM framework.
This shows how to create a text summarizer that can generate summaries of various lengths.
"""

import logging
import sys
import os

# Add parent directory to path to import simplellm
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llms import ChatCompletionLLMApplier, ApplicationModes, with_retries, prompts, util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextSummarizer(ChatCompletionLLMApplier):
    """
    Example LLM applier that creates summaries of provided text.
    """

    DEFAULT_PROMPT = prompts.get("example_text_summarizer", "0.1.0")
    DEFAULT_MODEL_CONFIG = {
        "model": "dev-gpt-41-longco-2025-04-14",
        "temperature": 0.1,
        "max_tokens": 1000,
    }
    DEFAULT_THREADS = 2
    DEFAULT_RETRIES = 3
    APPLICATION_MODE = ApplicationModes.PerItem

    def process_item(self, item, i):
        """Process a single item for summarization."""
        input_text = item.get("input_text", "")
        max_length = item.get("max_length", "3 sentences")

        # Validate input
        if not input_text.strip():
            logger.warning(f"Empty input text for item {i}")
            item["error"] = "No input text provided"
            return item

        # Summarize the text
        try:
            summary_result = self.summarize_text(input_text, max_length)
            if summary_result:
                item["summary"] = summary_result
                logger.info(f"Summarized item {i}: {len(summary_result)} characters")
        except Exception as e:
            logger.error(f"Error summarizing item {i}: {e}")
            item["error"] = str(e)

        return item

    @with_retries
    def summarize_text(self, input_text, max_length="3 sentences"):
        """
        Summarize the given text to the specified length.

        Args:
            input_text (str): The text to summarize
            max_length (str): Length specification (e.g., "3 sentences", "100 words", "2 paragraphs")

        Returns:
            str: The generated summary
        """
        prompt_context = {"input_text": input_text, "max_length": max_length}

        # Format the prompt with variables
        formatted_prompt = prompts.formatting.render_messages(
            self.prompt, prompt_context
        )

        # Call the LLM
        completion = self.llmapi.chat_completion(self.model_config, formatted_prompt)

        # Extract response
        response_text = completion["choices"][0]["message"]["content"].strip()

        if response_text:
            # Clean up the response - remove any extra formatting
            summary = response_text

            # Remove any markdown formatting that might be present
            if summary.startswith("```") and summary.endswith("```"):
                lines = summary.split("\n")
                summary = "\n".join(lines[1:-1])

            return summary

        return None


def main():
    """
    Demonstration of the text summarizer with sample data.
    """
    # Sample data for demonstration
    sample_data = [
        {
            "input_text": """
            The field of artificial intelligence has undergone remarkable transformations over the past decade. 
            Machine learning algorithms, particularly deep neural networks, have achieved unprecedented success 
            in tasks ranging from image recognition to natural language processing. Large language models like 
            GPT and BERT have revolutionized how we approach text generation and understanding. Meanwhile, 
            computer vision systems can now accurately identify objects, faces, and scenes in real-time. 
            These advances have led to practical applications in healthcare, autonomous vehicles, finance, 
            and countless other domains. However, challenges remain in areas such as AI safety, interpretability, 
            and ensuring ethical deployment of these powerful technologies. As we move forward, the focus is 
            shifting toward creating more robust, reliable, and beneficial AI systems that can work alongside 
            humans to solve complex problems.
            """,
            "max_length": "2 sentences",
        },
        {
            "input_text": """
            Climate change represents one of the most pressing challenges of our time. Rising global temperatures, 
            caused primarily by human activities such as burning fossil fuels and deforestation, are leading to 
            widespread environmental impacts. These include melting ice caps, rising sea levels, more frequent 
            extreme weather events, and disruptions to ecosystems and biodiversity. The economic costs are 
            substantial, affecting agriculture, infrastructure, and human health. International efforts such as 
            the Paris Climate Agreement aim to limit global warming, but achieving these goals requires rapid 
            transitions to renewable energy, improved energy efficiency, and significant changes in consumption 
            patterns. Individual actions, while important, must be combined with systemic changes in policy, 
            technology, and business practices to address this global challenge effectively.
            """,
            "max_length": "50 words",
        },
        {
            "input_text": """
            The Renaissance was a period of cultural, artistic, and intellectual rebirth in Europe that began 
            in Italy during the 14th century and spread throughout Europe over the following centuries. 
            This era marked a transition from medieval to modern thinking and was characterized by renewed 
            interest in classical Greek and Roman knowledge, humanism, and scientific inquiry.
            """,
            "max_length": "1 sentence",
        },
    ]

    # Create and configure the summarizer
    summarizer = TextSummarizer()

    print("Text Summarization Demo")
    print("=" * 50)

    # Process each sample
    for i, item in enumerate(sample_data):
        print(f"\nExample {i + 1}:")
        print(f"Original text length: {len(item['input_text'])} characters")
        print(f"Summary length requirement: {item['max_length']}")

        # Process the item
        result = summarizer.process_item(item.copy(), i)

        if "summary" in result:
            print(f"Summary: {result['summary']}")
            print(f"Summary length: {len(result['summary'])} characters")
        elif "error" in result:
            print(f"Error: {result['error']}")

        print("-" * 30)


if __name__ == "__main__":
    # Check if running as a demo or as part of the framework
    if len(sys.argv) == 1:
        main()
    else:
        # This allows the file to be used with the LLM framework's command line interface
        import fire

        fire.Fire(TextSummarizer)
