"""Core agent implementation for the unified agent system."""

import asyncio
from typing import Any, Dict, List, Optional

from .base import BaseAgent, BaseProvider
from .types import AgentConfig, Message, ProviderType, ConversationState, ToolResult


class Agent(BaseAgent):
    """A unified agent that can be orchestrated by a swarm."""

    def __init__(self, config: AgentConfig):
        """Initialize the agent."""
        super().__init__(config)

    def _create_provider(self) -> BaseProvider:
        """Create the appropriate provider for this agent."""
        from ..providers import ClaudeProvider, OpenAIProvider, MockProvider

        if self.config.provider == ProviderType.CLAUDE:
            return ClaudeProvider(self.config)
        elif self.config.provider == ProviderType.OPENAI:
            return OpenAIProvider(self.config)
        elif self.config.provider == ProviderType.MOCK:
            return MockProvider(self.config)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _create_conversation_state(self) -> ConversationState:
        """Create the conversation state manager."""
        return ConversationState()

    async def process_message(self, user_input: str) -> str:
        """Process a user message and return the response."""
        # Add user message to conversation
        user_message = Message(role="user", content=user_input)
        self.conversation_state.add_message(user_message)
        
        # Get response from provider
        response = await self.provider.generate_response(
            messages=self.conversation_state.messages,
            tools=self.config.tools
        )
        
        # Extract text content from response
        text_content = ""
        for content_block in response.get("content", []):
            if content_block.get("type") == "text":
                text_content += content_block.get("text", "")
        
        # Add assistant response to conversation
        assistant_message = Message(
            role="assistant", 
            content=text_content,
            tool_calls=response.get("tool_calls", [])
        )
        self.conversation_state.add_message(assistant_message)
        
        # Handle tool calls if present
        if response.get("tool_calls"):
            tool_results = await self.execute_tools(response["tool_calls"])
            # Add tool results to conversation
            for result in tool_results:
                tool_message = Message(
                    role="tool",
                    content=str(result.content),
                    tool_results=[{
                        "tool_call_id": result.metadata.get("tool_call_id") if result.metadata else None,
                        "content": result.content
                    }]
                )
                self.conversation_state.add_message(tool_message)
        
        return text_content

    async def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute tool calls and return results."""
        results = []
        for tool_call in tool_calls:
            # For now, return a placeholder result
            # In a full implementation, this would execute the actual tools
            result = ToolResult(
                success=True,
                content=f"Tool {tool_call['name']} executed with input: {tool_call.get('input', {})}",
                metadata={"tool_call_id": tool_call.get("id")}
            )
            results.append(result)
        return results
