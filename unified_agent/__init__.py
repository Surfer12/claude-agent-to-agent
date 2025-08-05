"""
Unified Agent System

A provider-agnostic agent framework that supports both Claude and OpenAI backends,
with unified CLI and computer use capabilities.
"""

from .core import UnifiedAgent
from .types import AgentConfig, ProviderType
from .providers import ClaudeProvider, OpenAIProvider
from .tool_registry import ToolRegistry
from .computer_use import ComputerUseInterface

__version__ = "1.0.0"
__all__ = [
    "UnifiedAgent",
    "AgentConfig", 
    "ProviderType",
    "ClaudeProvider",
    "OpenAIProvider",
    "ToolRegistry",
    "ComputerUseInterface"
] 