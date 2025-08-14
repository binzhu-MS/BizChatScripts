"""
Unified LLM API Client for BizChatScripts
Provides intelligent routing between RSP and Microsoft LLM API Client approaches
"""

import logging
from typing import Dict, Any, Optional, Union
from .llm_api import LLMAPI  # Original RSP client

logger = logging.getLogger(__name__)

try:
    from .ms_llm_api_client_adapter import MSLLMAPIClientAdapter

    _HAS_MS_LLM_CLIENT = True
except ImportError:
    _HAS_MS_LLM_CLIENT = False
    logger.warning("Microsoft LLM API Client not available during import.")


class ClientType:
    """Constants for client type selection"""

    RSP = "rsp"
    MS_LLM_CLIENT = "ms_llm_client"
    AUTO = "auto"


class UnifiedLLMAPI:
    """
    Unified LLM API client that provides flexible routing between RSP and Microsoft LLM API Client.
    Maintains full compatibility with existing BizChatScripts code while adding enhanced capabilities.

    Users can choose their approach:
    1. Automatic routing based on model name
    2. Explicit client selection by name
    3. Fallback strategies for robustness
    """

    # Models that are specifically tested with RSP approach
    RSP_PREFERRED_MODELS = {
        "dev-gpt-41-longco-2025-04-14",  # Primary model tested with RSP
    }

    def __init__(
        self,
        # RSP client configuration
        rsp_endpoint=None,
        rsp_auth=None,
        rsp_retries=3,
        # Microsoft LLM API Client configuration
        ms_scenario_guid=None,
        ms_retries=3,
        ms_enable_async=False,
        ms_max_requests_per_minute=5,
        # Routing configuration
        default_client=ClientType.AUTO,
        fallback_enabled=True,
    ):
        """
        Initialize unified client with both RSP and Microsoft LLM API Client capabilities.

        Args:
            rsp_*: Parameters for RSP client (same as original LLMAPI)
            ms_*: Parameters for Microsoft LLM API Client adapter
            default_client: Default client to use - "rsp", "ms_llm_client", or "auto"
            fallback_enabled: If True, fallback to other client on failure
        """
        # Initialize RSP client (always available)
        self.rsp_client = LLMAPI(
            endpoint=rsp_endpoint, auth=rsp_auth, retries=rsp_retries
        )

        # Initialize Microsoft LLM API Client (required - no fallback)
        if not _HAS_MS_LLM_CLIENT:
            raise ImportError(
                "Microsoft LLM API Client is required for unified client but not installed. "
                "Install with: pip install llm-api-client --index-url "
                "https://o365exchange.pkgs.visualstudio.com/_packaging/O365PythonPackagesV2/pypi/simple/ "
                "See README.md for installation instructions."
            )

        self.ms_client = MSLLMAPIClientAdapter(
            scenario_guid=ms_scenario_guid,
            retries=ms_retries,
            enable_async=ms_enable_async,
            max_requests_per_minute=ms_max_requests_per_minute,
        )

        self.default_client = default_client
        self.fallback_enabled = fallback_enabled

    def _select_client(self, model_config, preferred_client=None) -> tuple:
        """
        Select appropriate client based on model name and user preferences.

        Args:
            model_config: Model configuration dict
            preferred_client: Explicit client preference ("rsp", "ms_llm_client", or None for auto)

        Returns:
            tuple: (client, client_name)
        """
        model_name = model_config.get("model_name") or model_config.get("model", "")

        # If user explicitly specifies client, honor that choice
        if preferred_client == ClientType.RSP:
            return self.rsp_client, "rsp"
        elif preferred_client == ClientType.MS_LLM_CLIENT:
            if self.ms_client:
                return self.ms_client, "ms_llm_client"
            else:
                logger.warning(
                    "Microsoft LLM API Client requested but not available. Falling back to RSP."
                )
                return self.rsp_client, "rsp"

        # Auto-routing logic based on model and availability
        if model_name in self.RSP_PREFERRED_MODELS:
            # Use RSP for specifically tested models
            return self.rsp_client, "rsp"
        elif self.ms_client and self.default_client != ClientType.RSP:
            # Use Microsoft LLM API Client for broader model support (if available)
            logger.debug(
                f"Using Microsoft LLM API Client for model: {model_name} (broader support)"
            )
            return self.ms_client, "ms_llm_client"
        else:
            # Fallback to RSP
            logger.debug(f"Using RSP client for model: {model_name}")
            return self.rsp_client, "rsp"

    def chat_completion(self, model_config, input_data, preferred_client=None):
        """
        Route chat completion to appropriate client with optional fallback.
        Maintains exact same interface as original LLMAPI for seamless integration.

        Args:
            model_config: Model configuration (supports both "model" and "model_name")
            input_data: Input messages or dict
            preferred_client: Optional explicit client choice ("rsp" or "ms_llm_client")
        """
        client, client_name = self._select_client(model_config, preferred_client)

        try:
            logger.debug(
                f"Routing to {client_name} client for model: {model_config.get('model', 'unknown')}"
            )
            return client.chat_completion(model_config, input_data)

        except Exception as e:
            error_message = str(e).lower()
            logger.error(f"Error in {client_name} client: {e}")

            # Check for rate limiting/throttling scenarios where switching makes sense
            is_rate_limited = any(
                keyword in error_message
                for keyword in [
                    "rate limit",
                    "throttle",
                    "429",
                    "too many requests",
                    "quota",
                ]
            )

            # Try fallback if enabled and we have a good reason to switch
            if self.fallback_enabled and is_rate_limited:
                fallback_client = (
                    self.ms_client if client_name == "rsp" else self.rsp_client
                )
                fallback_name = "ms_llm_client" if client_name == "rsp" else "rsp"

                logger.warning(
                    f"Rate limiting detected. Attempting fallback from {client_name} to {fallback_name}"
                )
                try:
                    return fallback_client.chat_completion(model_config, input_data)
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback to {fallback_name} also failed: {fallback_error}"
                    )
            elif self.fallback_enabled:
                # General fallback for other errors (less aggressive)
                fallback_client = (
                    self.ms_client if client_name == "rsp" else self.rsp_client
                )
                fallback_name = "ms_llm_client" if client_name == "rsp" else "rsp"

                logger.warning(
                    f"Attempting fallback from {client_name} to {fallback_name}"
                )
                try:
                    return fallback_client.chat_completion(model_config, input_data)
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback to {fallback_name} also failed: {fallback_error}"
                    )

            # Re-raise original error if no fallback or fallback failed
            raise

    def send_request(
        self, model_name, request_data, chat_completion=True, preferred_client=None
    ):
        """
        Route send_request to appropriate client.
        Maintains exact same interface as original LLMAPI.
        """
        model_config = {"model": model_name}
        client, client_name = self._select_client(model_config, preferred_client)

        try:
            return client.send_request(model_name, request_data, chat_completion)
        except Exception as e:
            logger.error(f"Error in {client_name} client: {e}")

            # Try fallback similar to chat_completion
            if self.fallback_enabled and self.ms_client and chat_completion:
                fallback_client = (
                    self.ms_client if client_name == "rsp" else self.rsp_client
                )
                fallback_name = "ms_llm_client" if client_name == "rsp" else "rsp"

                logger.warning(
                    f"Attempting fallback from {client_name} to {fallback_name}"
                )
                try:
                    return fallback_client.send_request(
                        model_name, request_data, chat_completion
                    )
                except Exception:
                    pass  # Continue to raise original error

            raise


# Convenience functions for explicit client selection
def create_rsp_client(**kwargs):
    """Create a UnifiedLLMAPI configured to prefer RSP client"""
    return UnifiedLLMAPI(default_client=ClientType.RSP, **kwargs)


def create_ms_llm_client(**kwargs):
    """Create a UnifiedLLMAPI configured to prefer Microsoft LLM API Client"""
    return UnifiedLLMAPI(default_client=ClientType.MS_LLM_CLIENT, **kwargs)


def create_auto_client(**kwargs):
    """Create a UnifiedLLMAPI with automatic routing (default)"""
    return UnifiedLLMAPI(default_client=ClientType.AUTO, **kwargs)
