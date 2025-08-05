"""Provider implementations for Claude and OpenAI."""

import asyncio
import json
import os
from typing import Any, Dict, List

from .core.base import BaseProvider
from .core.types import AgentConfig, Message, Tool, ToolResult


class MockProvider(BaseProvider):
    """Mock provider for testing without API keys."""
    
    def _create_client(self) -> None:
        """Create a mock client."""
        return None
    
    async def generate_response(
        self, 
        messages: List[Message], 
        tools: List[Tool]
    ) -> Dict[str, Any]:
        """Generate a mock response."""
        last_message = messages[-1] if messages else None
        user_content = last_message.content if last_message else "No input"
        
        return {
            "content": [{
                "type": "text",
                "text": f"Mock response to: {user_content}"
            }],
            "tool_calls": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30
            }
        }
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for mock provider."""
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Format tools for mock provider."""
        return [tool.to_dict() for tool in tools]


class ClaudeProvider(BaseProvider):
    """Claude provider implementation."""
    
    def _create_client(self):
        """Create the Anthropic client."""
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or config")
        
        try:
            from anthropic import Anthropic
            return Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    async def generate_response(
        self, 
        messages: List[Message], 
        tools: List[Tool]
    ) -> Dict[str, Any]:
        """Generate a response from Claude."""
        # Format messages for Claude
        claude_messages = self.format_messages(messages)
        
        # Prepare parameters
        params = {
            "model": self.config.model or "claude-3-5-sonnet-20241022",
            "messages": claude_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        
        # Add system prompt if provided
        if self.config.system_prompt:
            params["system"] = self.config.system_prompt
        
        # Add tools if provided
        if tools:
            params["tools"] = self.format_tools(tools)
        
        # Create message
        response = await asyncio.to_thread(
            self.client.messages.create,
            **params
        )
        
        # Convert response to unified format
        return self._convert_claude_response(response)
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for Claude."""
        claude_messages = []
        
        for message in messages:
            if message.role == "system":
                # System messages are handled separately in Claude
                continue
            
            claude_message = {
                "role": message.role,
                "content": message.content
            }
            
            # Handle tool calls and results
            if message.tool_calls:
                # Convert tool calls to Claude format
                content_blocks = []
                if isinstance(message.content, str):
                    content_blocks.append({"type": "text", "text": message.content})
                
                for tool_call in message.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "input": tool_call.get("input", {})
                    })
                
                claude_message["content"] = content_blocks
            
            claude_messages.append(claude_message)
        
        return claude_messages
    
    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Format tools for Claude."""
        return [tool.to_dict() for tool in tools]
    
    def _convert_claude_response(self, response) -> Dict[str, Any]:
        """Convert Claude response to unified format."""
        content = []
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content.append({
                    "type": "text",
                    "text": block.text
                })
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        
        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        }


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""
    
    def _create_client(self):
        """Create the OpenAI client."""
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or config")
        
        try:
            from openai import AsyncOpenAI
            return AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    async def generate_response(
        self, 
        messages: List[Message], 
        tools: List[Tool]
    ) -> Dict[str, Any]:
        """Generate a response from OpenAI."""
        # Format messages for OpenAI
        openai_messages = self.format_messages(messages)
        
        # Prepare parameters
        params = {
            "model": self.config.model or "gpt-4o",
            "messages": openai_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        
        # Add tools if provided
        if tools:
            params["tools"] = self.format_tools(tools)
            params["tool_choice"] = "auto"
        
        # Create message
        response = await self.client.chat.completions.create(**params)
        
        # Convert response to unified format
        return self._convert_openai_response(response)
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI."""
        openai_messages = []
        
        for message in messages:
            openai_message = {
                "role": message.role,
                "content": message.content
            }
            
            # Handle tool calls
            if message.tool_calls:
                openai_message["tool_calls"] = []
                for tool_call in message.tool_calls:
                    openai_message["tool_calls"].append({
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call.get("input", {}))
                        }
                    })
            
            # Handle tool results
            if message.tool_results:
                for tool_result in message.tool_results:
                    openai_messages.append({
                        "role": "tool",
                        "content": str(tool_result.get("content", "")),
                        "tool_call_id": tool_result.get("tool_call_id")
                    })
            
            openai_messages.append(openai_message)
        
        return openai_messages
    
    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI."""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        return openai_tools
    
    def _convert_openai_response(self, response) -> Dict[str, Any]:
        """Convert OpenAI response to unified format."""
        content = []
        tool_calls = []
        
        message = response.choices[0].message
        
        if message.content:
            content.append({
                "type": "text",
                "text": message.content
            })
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                tool_calls.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": arguments
                })
                content.append({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": arguments
                })
        
        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
