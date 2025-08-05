"""Core types and data structures for the unified agent system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import os


class ProviderType(Enum):
    """Supported AI providers."""
    CLAUDE = "claude"
    OPENAI = "openai"


class ComputerType(Enum):
    """Supported computer environments."""
    NONE = "none"
    LOCAL_PLAYWRIGHT = "local-playwright"
    BROWSERBASE = "browserbase"
    DOCKER = "docker"


@dataclass
class Message:
    """A message in the conversation."""
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None


@dataclass
class Tool:
    """Base tool interface."""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    async def execute(self, **kwargs) -> "ToolResult":
        """Execute the tool with given parameters."""
        # Default implementation - subclasses should override
        return ToolResult(
            success=False,
            content="Tool execution not implemented",
            error="This tool does not have an execute method implemented"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format for API calls."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    content: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


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


@dataclass
class ConversationState:
    """State of a conversation."""
    messages: List[Message] = field(default_factory=list)
    tools: List[Tool] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: Message):
        """Add a message to the conversation."""
        self.messages.append(message)
    
    def get_context(self, max_messages: Optional[int] = None) -> List[Message]:
        """Get conversation context, optionally limited to recent messages."""
        if max_messages is None:
            return self.messages
        return self.messages[-max_messages:] 