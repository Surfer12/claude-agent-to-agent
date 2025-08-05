"""Unified Agent System - A provider-agnostic agent framework."""

from .core.agent import Agent
from .core.base import BaseAgent, BaseProvider
from .core.types import AgentConfig, Message, Tool, ToolResult, ProviderType, ComputerType
from .providers import ClaudeProvider, OpenAIProvider, MockProvider

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "BaseAgent", 
    "BaseProvider",
    "AgentConfig",
    "Message",
    "Tool", 
    "ToolResult",
    "ProviderType",
    "ComputerType",
    "ClaudeProvider",
    "OpenAIProvider",
    "MockProvider",
]
