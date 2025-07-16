"""
Core unified agent framework with provider abstraction.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .providers import ClaudeProvider, OpenAIProvider
from .tools import ToolRegistry


class ProviderType(Enum):
    """Supported AI providers."""
    CLAUDE = "claude"
    OPENAI = "openai"


@dataclass
class AgentConfig:
    """Configuration for the unified agent."""
    
    # Provider settings
    provider: ProviderType = ProviderType.CLAUDE
    model: str = "claude-3-5-sonnet-20241022"
    api_key: Optional[str] = None
    
    # Model parameters
    max_tokens: int = 4096
    temperature: float = 1.0
    context_window_tokens: int = 180000
    
    # Agent settings
    system_prompt: str = "You are a helpful AI assistant."
    verbose: bool = False
    
    # Tool settings
    enable_tools: bool = True
    enable_computer_use: bool = False
    enable_code_execution: bool = False
    
    # Computer use settings
    computer_type: str = "local-playwright"
    start_url: str = "https://bing.com"
    show_images: bool = False
    debug: bool = False
    
    def __post_init__(self):
        """Set default API key based on provider."""
        if self.api_key is None:
            if self.provider == ProviderType.CLAUDE:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif self.provider == ProviderType.OPENAI:
                self.api_key = os.environ.get("OPENAI_API_KEY")


class ProviderInterface(ABC):
    """Abstract interface for AI providers."""
    
    @abstractmethod
    async def create_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a message with the provider."""
        pass
    
    @abstractmethod
    def get_tool_schema(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert tools to provider-specific schema."""
        pass


class UnifiedAgent:
    """Unified agent that works with multiple AI providers."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the unified agent."""
        self.config = config
        self.tool_registry = ToolRegistry()
        self.provider = self._create_provider()
        self.message_history: List[Dict[str, Any]] = []
        
        # Initialize tools based on configuration
        if config.enable_tools:
            self._setup_tools()
    
    def _create_provider(self) -> ProviderInterface:
        """Create the appropriate provider based on configuration."""
        if self.config.provider == ProviderType.CLAUDE:
            return ClaudeProvider(self.config)
        elif self.config.provider == ProviderType.OPENAI:
            return OpenAIProvider(self.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _setup_tools(self):
        """Setup tools based on configuration."""
        if self.config.enable_code_execution:
            self.tool_registry.register_code_execution_tools()
        
        if self.config.enable_computer_use:
            self.tool_registry.register_computer_use_tools(
                computer_type=self.config.computer_type
            )
    
    async def run_async(self, user_input: str) -> Dict[str, Any]:
        """Run the agent asynchronously."""
        if self.config.verbose:
            print(f"[Agent] Processing: {user_input}")
        
        # Add user message to history
        self.message_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Get tools for this interaction
        tools = self.tool_registry.get_active_tools()
        tool_schema = self.provider.get_tool_schema(tools)
        
        # Create message with provider
        response = await self.provider.create_message(
            messages=self.message_history,
            tools=tool_schema,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self.config.system_prompt
        )
        
        # Add assistant response to history
        self.message_history.append({
            "role": "assistant",
            "content": response.get("content", [])
        })
        
        # Handle tool calls if present
        if "tool_calls" in response:
            await self._handle_tool_calls(response["tool_calls"])
        
        return response
    
    async def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]):
        """Handle tool calls and add results to history."""
        tool_results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]
            tool_id = tool_call.get("id")
            
            if self.config.verbose:
                print(f"[Agent] Executing tool: {tool_name}")
            
            # Execute tool
            result = await self.tool_registry.execute_tool(tool_name, tool_input)
            
            # Add tool result to history
            tool_results.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_id
            })
        
        # Add all tool results to history
        self.message_history.extend(tool_results)
    
    def run(self, user_input: str) -> Dict[str, Any]:
        """Run the agent synchronously."""
        return asyncio.run(self.run_async(user_input))
    
    def reset(self):
        """Reset the agent's message history."""
        self.message_history = []
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the current message history."""
        return self.message_history.copy() 