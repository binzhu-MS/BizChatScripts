import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Dict

import tqdm

from .llm_api import LLMAPI

logger = logging.getLogger(__name__)


def with_retries(func):
    """Decorator to add retry functionality to methods."""

    @wraps(func)
    def wrapped_func(self, *args, **kwargs):
        errors = []
        for attempt in range(1, self.retries + 1):
            try:
                log_message = f"Submitting {self.__class__.__name__} (attempt {attempt} of {self.retries})"
                if attempt <= self.retries - 1:
                    logger.debug(log_message)
                else:
                    logger.warning(log_message)
                return func(self, *args, **kwargs)
            except Exception as e:
                errors.append((e, traceback.format_exc()))
                logger.debug(f"Error with {self.__class__.__name__}: {e}")
                if hasattr(e, "response"):
                    response = getattr(e, "response")
                    logger.debug(f"Response text: {getattr(response, 'text', None)}")

        raise errors[-1][0]

    return wrapped_func


class ApplicationModes:
    """Application mode constants."""

    PerItem = "PerItem"
    PerCondition = "PerCondition"


class ChatCompletionLLMApplier:
    """
    Base class for LLM applications using chat completion.
    Provides framework for batch processing with threading.
    """

    # These should be overridden by subclasses
    DEFAULT_PROMPT = NotImplemented
    DEFAULT_MODEL_CONFIG = NotImplemented
    DEFAULT_THREADS = NotImplemented
    DEFAULT_RETRIES = NotImplemented
    APPLICATION_MODE = NotImplemented

    def __init__(
        self,
        llmapi=None,
        prompt=None,
        threads=None,
        model_config=None,
        retries=None,
        client_type=None,
        ms_scenario_guid=None,
    ):
        """
        Initialize the LLM applier.

        Args:
            llmapi: LLMAPI instance (if None, creates based on client_type)
            prompt: Prompt template
            threads: Number of worker threads for parallel processing
            model_config: Model configuration
            retries: Number of retries on failures (framework-level)
            client_type: Type of client to use:
                - None or "rsp": RSP client (default, maintains compatibility)
                - "ms_llm_client": Microsoft LLM API Client (no RSP fallback)
                - "unified": Smart routing between RSP and LLM API Client
            ms_scenario_guid: Scenario GUID for Microsoft LLM API Client (when using ms_llm_client or unified)

        Note on Parameters when using Microsoft LLM API Client:
            - threads: Still used for parallel processing of multiple items
            - retries: Used for compatibility, but LLM API Client has internal retry mechanisms
        """
        if llmapi is None:
            # Create appropriate client based on client_type
            if client_type == "ms_llm_client":
                # No fallback - users must install required packages
                from .ms_llm_api_client_adapter import MSLLMAPIClientAdapter

                # Note: retries parameter is used for compatibility but LLM API Client
                # has its own internal retry mechanisms
                self.llmapi = MSLLMAPIClientAdapter(
                    scenario_guid=ms_scenario_guid,
                    retries=retries or self.__class__.DEFAULT_RETRIES,
                )
            elif client_type == "unified":
                # No fallback - users must install required packages
                from .llm_api_unified import UnifiedLLMAPI

                self.llmapi = UnifiedLLMAPI(
                    ms_scenario_guid=ms_scenario_guid,
                    rsp_retries=retries or self.__class__.DEFAULT_RETRIES,
                    ms_retries=retries or self.__class__.DEFAULT_RETRIES,
                )
            elif client_type == "rsp" or client_type is None:
                # Default to RSP client for backward compatibility and explicit RSP selection
                from .llm_api import LLMAPI

                self.llmapi = LLMAPI()
        else:
            self.llmapi = llmapi  # User-provided client

        self.prompt = prompt or self.__class__.DEFAULT_PROMPT
        self.threads = threads or self.__class__.DEFAULT_THREADS
        self.model_config = model_config or self.__class__.DEFAULT_MODEL_CONFIG
        self.retries = retries or self.__class__.DEFAULT_RETRIES

    def apply(self, items, show_progress=True):
        """
        Apply LLM processing to a list of items.

        Args:
            items: List of items to process
            show_progress: Whether to show progress bar

        Yields:
            Processed items
        """
        if len(items) > 0:
            logger.info(
                f"Starting {self.__class__.__name__} with {self.threads} workers"
            )
            if show_progress:
                progress = tqdm.tqdm(total=len(items))

            futures = []
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                for i, item in enumerate(items):
                    logger.debug(f"Submitting {self.__class__.__name__} for item {i}")
                    futures.append(executor.submit(self._apply_to_item, item, i))

                for future in as_completed(futures):
                    if show_progress:
                        progress.update(1)
                    item, error = future.result()
                    if error is None:
                        yield item
                    else:
                        e, tb = error
                        logger.warning(
                            f"Could not apply {self.__class__.__name__}: {e}"
                        )
                        logger.debug(tb)

    def _apply_to_item(self, item, i):
        """Apply processing to a single item."""
        logger.debug(f"Processing {self.__class__.__name__} for item {i}")
        if self.__class__.APPLICATION_MODE == ApplicationModes.PerCondition:
            return self._process_per_condition(item)
        elif self.__class__.APPLICATION_MODE == ApplicationModes.PerItem:
            return self._process_per_item(item, i)

    def _process_per_condition(self, item):
        """Process item with multiple conditions."""
        try:
            for exp_name, conversation in item.get("conditions", {}).items():
                self.process_query_conversation(
                    item.get("query", ""), exp_name, conversation
                )
            return item, None
        except Exception as e:
            return item, (e, traceback.format_exc())

    def _process_per_item(self, item, i):
        """Process single item."""
        try:
            self.process_item(item, i)
            return item, None
        except Exception as e:
            return item, (e, traceback.format_exc())

    def process_query_conversation(
        self, query: str, exp_name: str, conversation: Dict[str, Any]
    ) -> None:
        """
        Process a query-conversation pair. Override in subclasses.

        Args:
            query: User query
            exp_name: Experiment name
            conversation: Conversation data
        """
        raise NotImplementedError(
            "Subclasses must implement process_query_conversation"
        )

    def process_item(self, item: Dict[str, Any], i: int) -> None:
        """
        Process a single item. Override in subclasses.

        Args:
            item: Item to process
            i: Item index
        """
        raise NotImplementedError("Subclasses must implement process_item")
