"""
Core types for the unified agent system.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

class ProviderType(Enum):
    """Supported AI providers."""
    CLAUDE = "claude"
    OPENAI = "openai"

@dataclass
class AgentConfig:
    """Configuration for the unified agent."""
    
    # Provider settings
    provider: ProviderType = ProviderType.CLAUDE
    model: str = "claude-3-5-sonnet-20240620"
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
