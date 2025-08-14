"""
SimpleLLM - A Simple LLM Calling Framework
Enhanced with Microsoft LLM API Client support for broader internal model access
"""

from .llm_api import LLMAPI
from .base_applier import ChatCompletionLLMApplier, ApplicationModes, with_retries
from . import prompts
from . import util

# Import new clients if available
try:
    from .llm_api_unified import UnifiedLLMAPI
    from .ms_llm_api_client_adapter import MSLLMAPIClientAdapter
    _HAS_MS_LLM_CLIENT = True
except ImportError:
    # Fallback if Microsoft LLM API Client not installed
    UnifiedLLMAPI = None
    MSLLMAPIClientAdapter = None
    _HAS_MS_LLM_CLIENT = False

__version__ = "0.2.0"

# Base exports (always available)
__all__ = [
    'LLMAPI',
    'ChatCompletionLLMApplier', 
    'ApplicationModes',
    'with_retries',
    'prompts',
    'util'
]

# Add enhanced exports if available
if _HAS_MS_LLM_CLIENT:
    __all__.extend([
        'UnifiedLLMAPI',
        'MSLLMAPIClientAdapter'
    ])