import json
import logging
import requests
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0
DEFAULT_N = 1
DEFAULT_STOP = None

# Scenario ID from RSP project for increased capacity
SCENARIO_ID = "6b68a373-f1ca-4ee4-945b-0b4b2111f757"

class LLMAPI:
    """
    A simplified LLM API client for calling chat completion endpoints.
    Based on the RSP project's LLMAPI implementation.
    """
    
    _API_CHAT_COMPLETIONS = 'chat/completions'
    _ENDPOINT = 'https://fe-26.qas.bing.net/sdf/'  # Replace with your actual endpoint
    
    def __init__(self, endpoint=None, auth=None, retries=3):
        """
        Initialize the LLM API client.
        
        Args:
            endpoint: API endpoint URL (optional)
            auth: Authentication provider (optional) 
            retries: Number of retry attempts on failures
        """
        self.endpoint = endpoint or self._ENDPOINT
        self.auth = auth or self._get_default_auth()
        self.retries = retries
    
    def _get_default_auth(self):
        """Get default authentication. Override this method for custom auth."""
        try:
            # Import RSP auth directly since it's already working
            import sys
            import os
            rsp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Scripts', 'rsp')
            if rsp_path not in sys.path:
                sys.path.insert(0, rsp_path)
            import rsp.auth
            return rsp.auth.LLMAPI.get_instance()
        except ImportError as e:
            logger.warning(f"Failed to import RSP auth ({e}). Trying local auth.")
            try:
                from . import auth
                return auth.LLMAPI.get_instance()
            except ImportError:
                logger.warning("No authentication provider found. Using dummy auth.")
                return DummyAuth()
    
    def _build_headers(self, model_name):
        """Build request headers including authentication."""
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.auth.get_token()}',
            'X-ModelType': model_name,
            'X-ScenarioGUID': SCENARIO_ID
        }
    
    def _build_request(self, model_config, input_data):
        """
        Build request data from model config and input.
        
        Args:
            model_config: Dictionary containing model configuration
            input_data: Input data (messages or dict)
            
        Returns:
            Tuple of (model_name, request_data)
        """
        request_data = {
            "max_tokens": model_config.get("max_tokens", DEFAULT_MAX_TOKENS),
            "temperature": model_config.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": model_config.get("top_p", DEFAULT_TOP_P),
            "n": model_config.get("n", DEFAULT_N),
            "stop": model_config.get("stop", DEFAULT_STOP),
        }
        
        # Handle different input formats
        if isinstance(input_data, dict):
            request_data.update(input_data)
        else:
            request_data['messages'] = input_data
        
        model_name = model_config.get("model_name") or model_config.get("model")
        return model_name, request_data
    
    def _execute_request(self, endpoint, body, headers):
        """Execute HTTP request with error handling."""
        with requests.post(endpoint, data=body, headers=headers) as response:
            response.raise_for_status()
            return response.json()
    
    def _retry_with_backoff(self, method, *args, **kwargs):
        """Retry method with exponential backoff on failures."""
        last_exception = None
        
        for attempt in range(self.retries):
            try:
                return method(*args, **kwargs)
            except requests.exceptions.HTTPError as e:
                last_exception = e
                if e.response.status_code == 429:  # Rate limit
                    sleep_time = min(60, (2 ** attempt))
                    logger.warning(f"Rate limited. Retry {attempt+1}/{self.retries}. Sleeping {sleep_time}s.")
                    time.sleep(sleep_time)
                else:
                    raise
            except requests.RequestException as e:
                last_exception = e
                sleep_time = 5
                logger.warning(f"Network error. Retry {attempt+1}/{self.retries}. Sleeping {sleep_time}s.")
                time.sleep(sleep_time)
        
        logger.error(f"All {self.retries} retries failed.")
        raise last_exception
    
    def chat_completion(self, model_config, input_data):
        """
        Send a chat completion request.
        
        Args:
            model_config: Model configuration dict
            input_data: Messages or input dict
            
        Returns:
            API response dictionary
        """
        model_name, request_data = self._build_request(model_config, input_data)
        return self.send_request(model_name, request_data, chat_completion=True)
    
    def send_request(self, model_name, request_data, chat_completion=True):
        """
        Send request to LLM API.
        
        Args:
            model_name: Name of the model to use
            request_data: Request payload
            chat_completion: Whether to use chat completion endpoint
            
        Returns:
            API response dictionary
        """
        headers = self._build_headers(model_name)
        body = json.dumps(request_data).encode('utf-8')
        endpoint = self.endpoint + (self._API_CHAT_COMPLETIONS if chat_completion else 'completions')
        
        return self._retry_with_backoff(self._execute_request, endpoint, body, headers)
    
    def stream_chat_completion(self, model_config, input_data):
        """
        Send a streaming chat completion request.
        Note: This is a simplified version. Full streaming implementation would be more complex.
        """
        # For now, just return the regular completion
        return self.chat_completion(model_config, input_data)


class DummyAuth:
    """Dummy authentication for testing purposes."""
    
    def get_token(self):
        return "dummy-token"