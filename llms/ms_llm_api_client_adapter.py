"""
Microsoft LLM API Client Adapter for BizChatScripts
Provides compatibility layer between BizChatScripts and Microsoft's llm-api-client library
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

try:
    from llm_api_client.llm_call import llm_call, Tolerance
    from llm_api_client.structured_processing.post_process import (
        PassthroughResponseProcessorFactory,
    )
    from llm_api_client.structured_processing.prompt_data import PromptData, PromptSpec
    _HAS_LLM_API_CLIENT = True
except ImportError:
    _HAS_LLM_API_CLIENT = False
    logger.warning("Microsoft llm-api-client not available. Install with: pip install llm-api-client")


class MSLLMAPIClientAdapter:
    """
    Adapter to make Microsoft's llm-api-client compatible with BizChatScripts interface.
    Maintains the same method signatures as the original LLMAPI class for seamless integration.
    
    This adapter supports broader internal Microsoft model access compared to the RSP approach.
    """
    
    def __init__(self, scenario_guid=None, retries=3, enable_async=False, max_requests_per_minute=5):
        """
        Initialize the Microsoft LLM API Client adapter.
        
        Args:
            scenario_guid: Scenario GUID for llm-api-client
            retries: Number of retries (converted to Tolerance for error handling)
            enable_async: Whether to use async mode
            max_requests_per_minute: Rate limiting for requests
        """
        if not _HAS_LLM_API_CLIENT:
            raise ImportError(
                "Microsoft llm-api-client is required but not installed. "
                "Install with: pip install llm-api-client --index-url "
                "https://o365exchange.pkgs.visualstudio.com/_packaging/O365PythonPackagesV2/pypi/simple/"
            )
            
        self.scenario_guid = scenario_guid or "4d89af25-54b8-414a-807a-0c9186ff7539"
        self.retries = retries
        self.enable_async = enable_async
        self.max_requests_per_minute = max_requests_per_minute
        
    def chat_completion(self, model_config, input_data):
        """
        Send a chat completion request using Microsoft llm-api-client.
        Compatible with existing BizChatScripts LLMAPI interface.
        
        Args:
            model_config: Model configuration dict (same format as RSP)
                         Supports both "model" and "model_name" keys for compatibility
            input_data: Messages or input dict (same format as RSP)
            
        Returns:
            Response in same format as RSP for compatibility
        """
        try:
            # Extract model name - support both "model" and "model_name" for compatibility
            model_name = model_config.get("model_name") or model_config.get("model")
            if not model_name:
                raise ValueError("Model name not found in model_config. Use 'model' or 'model_name' key.")
            
            # Build payload in Microsoft LLM API Client format
            if isinstance(input_data, dict) and "messages" in input_data:
                payload = dict(input_data)
            else:
                payload = {"messages": input_data}
                
            # Add model config parameters with proper field names
            payload.update({
                "temperature": model_config.get("temperature", 0.7),
                "max_completion_tokens": model_config.get("max_tokens", 1000),
                "top_p": model_config.get("top_p", 1.0),
            })
            
            # Wrap in Microsoft LLM API Client structures
            prompt_spec = PromptSpec(PromptData(prompt=payload, metadata=None))
            
            # Configure tolerance based on retries
            # Allow some failures but not too many - this gives users more control than default (0 tolerance)
            tolerance = Tolerance(absolute=max(1, self.retries // 2))
            
            # Call Microsoft llm-api-client with controlled parameters
            results = list(llm_call(
                model=model_name,
                model_path="/chat/completions",
                prompts=[prompt_spec],
                response_processor_factory=PassthroughResponseProcessorFactory(),
                scenario_guid=self.scenario_guid,
                cache_path="",
                disable_cache=True,
                enable_async=self.enable_async,
                error_tolerance=tolerance,
                include_errors_in_output=True,
                backoff=1.5,  # Moderate backoff for balance between speed and reliability
                max_requests_per_minute=self.max_requests_per_minute,
            ))
            
            # Convert response to RSP-compatible format for seamless integration
            if results and not hasattr(results[0], 'error'):
                response_content = results[0].response
                
                # Parse the response to match RSP format
                if hasattr(response_content, 'choices') and hasattr(response_content, 'model'):
                    # Already in OpenAI format
                    return response_content
                else:
                    # Convert to OpenAI format for compatibility
                    return {
                        "choices": [{
                            "message": {
                                "content": str(response_content),
                                "role": "assistant"
                            },
                            "finish_reason": "stop"
                        }],
                        "model": model_name,
                        "usage": {}  # Basic structure for compatibility
                    }
            else:
                # Handle errors gracefully
                error_msg = str(results[0]) if results else "Unknown error from Microsoft LLM API Client"
                raise Exception(f"Microsoft LLM API Client error: {error_msg}")
                
        except Exception as e:
            logger.error(f"MSLLMAPIClientAdapter error: {e}")
            raise
            
    def send_request(self, model_name, request_data, chat_completion=True):
        """
        Compatibility method for direct send_request calls.
        Maintains compatibility with existing BizChatScripts usage patterns.
        """
        model_config = {"model": model_name}
        model_config.update(request_data)
        
        if chat_completion and "messages" in request_data:
            return self.chat_completion(model_config, request_data)
        else:
            # Handle completion mode if needed in future
            raise NotImplementedError("Completion mode not implemented in adapter - use chat_completion mode")


# Convenience function for quick setup
def create_ms_llm_client(scenario_guid=None, **kwargs):
    """
    Convenience function to create Microsoft LLM API Client adapter with sensible defaults.
    
    Args:
        scenario_guid: Your scenario GUID
        **kwargs: Additional configuration options
        
    Returns:
        Configured MSLLMAPIClientAdapter instance
    """
    return MSLLMAPIClientAdapter(scenario_guid=scenario_guid, **kwargs)
