"""
Core Anthropic API Client for Python

This module provides the main client interface for interacting with Anthropic's Claude API,
including support for various tools, streaming, and advanced features.
"""

import os
import anthropic
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from collections.abc import Mapping
from frozendict import frozendict


@dataclass(frozen=True)
class ToolConfig:
    """Immutable configuration for API tools."""
    name: str
    tool_type: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    max_uses: Optional[int] = None
    display_config: Optional[Dict[str, Any]] = None


class AnthropicClient:
    """
    A comprehensive client for Anthropic's Claude API with support for
    various tools, streaming, and advanced features.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Anthropic client.
        
        Args:
            api_key: Anthropic API key. If not provided, will use ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in ANTHROPIC_API_KEY environment variable")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Immutable tool configurations
        self._tool_configs = frozendict({
            "bash": ToolConfig(
                name="bash",
                tool_type="bash_20250124",
                description="Execute bash commands in a secure environment"
            ),
            "web_search": ToolConfig(
                name="web_search", 
                tool_type="web_search_20250305",
                description="Search the web for current information",
                max_uses=5
            ),
            "weather": ToolConfig(
                name="get_weather",
                tool_type="weather_tool",
                description="Get current weather information for a location",
                input_schema={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            ),
            "text_editor": ToolConfig(
                name="str_replace_based_edit_tool",
                tool_type="text_editor_20250429",
                description="Edit text files with string replacement operations"
            ),
            "code_execution": ToolConfig(
                name="code_execution",
                tool_type="code_execution_20250522",
                description="Execute code in a secure environment"
            ),
            "computer": ToolConfig(
                name="computer",
                tool_type="computer_20250124",
                description="Interact with computer interface",
                display_config={
                    "display_width_px": 1024,
                    "display_height_px": 768,
                    "display_number": 1
                }
            )
        })
    
    def create_message(
        self,
        messages: List[Dict[str, Any]],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        tools: Optional[List[str]] = None,
        betas: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """
        Create a message using the Anthropic API.
        
        Args:
            messages: List of message dictionaries
            model: Claude model to use
            max_tokens: Maximum tokens for response
            tools: List of tool names to enable
            betas: Beta features to enable
            **kwargs: Additional parameters
            
        Returns:
            API response
        """
        # Convert tool names to tool configurations
        tool_configs = []
        if tools:
            for tool_name in tools:
                if tool_name in self._tool_configs:
                    config = self._tool_configs[tool_name]
                    tool_config = {
                        "type": config.tool_type,
                        "name": config.name
                    }
                    if config.description:
                        tool_config["description"] = config.description
                    if config.input_schema:
                        tool_config["input_schema"] = config.input_schema
                    if config.max_uses:
                        tool_config["max_uses"] = config.max_uses
                    if config.display_config:
                        tool_config.update(config.display_config)
                    tool_configs.append(tool_config)
        
        # Prepare request parameters
        params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs
        }
        
        if tool_configs:
            params["tools"] = tool_configs
        
        if betas:
            params["betas"] = betas
        
        return self.client.messages.create(**params)
    
    def create_streaming_message(
        self,
        messages: List[Dict[str, Any]],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 65536,
        tools: Optional[List[str]] = None,
        betas: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """
        Create a streaming message using the Anthropic API.
        
        Args:
            messages: List of message dictionaries
            model: Claude model to use
            max_tokens: Maximum tokens for response
            tools: List of tool names to enable
            betas: Beta features to enable
            **kwargs: Additional parameters
            
        Returns:
            Streaming response
        """
        # Convert tool names to tool configurations
        tool_configs = []
        if tools:
            for tool_name in tools:
                if tool_name in self._tool_configs:
                    config = self._tool_configs[tool_name]
                    tool_config = {
                        "type": config.tool_type,
                        "name": config.name
                    }
                    if config.description:
                        tool_config["description"] = config.description
                    if config.input_schema:
                        tool_config["input_schema"] = config.input_schema
                    if config.max_uses:
                        tool_config["max_uses"] = config.max_uses
                    if config.display_config:
                        tool_config.update(config.display_config)
                    tool_configs.append(tool_config)
        
        # Prepare request parameters
        params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        if tool_configs:
            params["tools"] = tool_configs
        
        if betas:
            params["betas"] = betas
        
        return self.client.messages.stream(**params)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._tool_configs.keys())
    
    def get_tool_config(self, tool_name: str) -> Optional[ToolConfig]:
        """Get configuration for a specific tool."""
        return self._tool_configs.get(tool_name)


# Convenience functions for common use cases
def create_basic_client() -> AnthropicClient:
    """Create a basic Anthropic client with default configuration."""
    return AnthropicClient()


def create_tool_enabled_client(tools: List[str]) -> AnthropicClient:
    """Create an Anthropic client with specific tools enabled."""
    client = AnthropicClient()
    # Validate tools
    available_tools = client.get_available_tools()
    for tool in tools:
        if tool not in available_tools:
            raise ValueError(f"Unknown tool: {tool}")
    return client 