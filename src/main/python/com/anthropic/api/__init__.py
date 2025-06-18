"""
Anthropic API Client Library for Python

This package provides a comprehensive interface to Anthropic's Claude API,
including support for various tools, streaming, and advanced features.

Package Structure:
- client: Core API client functionality
- tools: Tool implementations (bash, web_search, weather, etc.)
- streaming: Streaming response handling
- cli: Command-line interface
- utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "Anthropic API Team"

from .client import AnthropicClient
from .tools import (
    BashTool,
    WebSearchTool,
    WeatherTool,
    TextEditorTool,
    CodeExecutionTool,
    StreamingTools
)
from .cli import CognitiveAgentCLI

__all__ = [
    "AnthropicClient",
    "BashTool",
    "WebSearchTool", 
    "WeatherTool",
    "TextEditorTool",
    "CodeExecutionTool",
    "StreamingTools",
    "CognitiveAgentCLI"
] 