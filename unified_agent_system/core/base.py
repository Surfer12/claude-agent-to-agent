"""Base classes for agents and providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .types import AgentConfig, Message, Tool, ToolResult


class BaseProvider(ABC):
    """Base class for AI providers."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = self._create_client()
    
    @abstractmethod
    def _create_client(self) -> Any:
        """Create the provider-specific client."""
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        messages: List[Message], 
        tools: List[Tool]
    ) -> Dict[str, Any]:
        """Generate a response from the AI provider."""
        pass
    
    @abstractmethod
    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for the specific provider."""
        pass
    
    @abstractmethod
    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Format tools for the specific provider."""
        pass


class BaseAgent(ABC):
    """Base class for agents."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.provider = self._create_provider()
        self.conversation_state = self._create_conversation_state()
    
    @abstractmethod
    def _create_provider(self) -> BaseProvider:
        """Create the appropriate provider for this agent."""
        pass
    
    @abstractmethod
    def _create_conversation_state(self) -> Any:
        """Create the conversation state manager."""
        pass
    
    @abstractmethod
    async def process_message(self, user_input: str) -> str:
        """Process a user message and return the response."""
        pass
    
    @abstractmethod
    async def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute tool calls and return results."""
        pass
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        message = Message(role=role, content=content)
        self.conversation_state.add_message(message)
    
    def get_context(self, max_messages: Optional[int] = None) -> List[Message]:
        """Get conversation context."""
        return self.conversation_state.get_context(max_messages) 