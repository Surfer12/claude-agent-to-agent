"""Core types and data structures for the unified agent system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


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
    
    @abstractmethod
    async def execute(self, **kwargs) -> "ToolResult":
        """Execute the tool with given parameters."""
        pass
    
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
    """Configuration for an agent."""
    provider: ProviderType
    model: str
    max_tokens: int = 4096
    temperature: float = 1.0
    system_prompt: str = ""
    tools: List[Tool] = field(default_factory=list)
    computer_type: ComputerType = ComputerType.NONE
    verbose: bool = False
    api_key: Optional[str] = None
    
    # Provider-specific settings
    claude_config: Optional[Dict[str, Any]] = None
    openai_config: Optional[Dict[str, Any]] = None


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