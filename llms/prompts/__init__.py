"""
Prompts package for SimpleLLM.
Simple prompt loading from markdown files organized by project/function names.
"""

from . import formatting
from . import mirror
from . import templates
from .general_loader import get

# Simple, clean interface
__all__ = [
    'formatting',
    'mirror', 
    'templates',
    'get'
]