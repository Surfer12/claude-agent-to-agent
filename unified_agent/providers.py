"""
Provider implementations for Claude and OpenAI.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from openai import AsyncOpenAI

from .core import AgentConfig, ProviderInterface


class ClaudeProvider(ProviderInterface):
    """Claude provider implementation."""
    
    def __init__(self, config: AgentConfig):
        """Initialize Claude provider."""
        self.config = config
        self.client = Anthropic(api_key=config.api_key)
    
    async def create_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a message with Claude."""
        # Convert messages to Claude format
        claude_messages = self._convert_messages_to_claude_format(messages)
        
        # Prepare parameters
        params = {
            "model": self.config.model,
            "messages": claude_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "system": kwargs.get("system", self.config.system_prompt),
        }
        
        # Add tools if provided
        if tools:
            params["tools"] = tools
            # Add beta headers for tools
            params["betas"] = self._get_beta_headers(tools)
        
        # Create message
        response = await asyncio.to_thread(
            self.client.messages.create,
            **params
        )
        
        # Convert response to unified format
        return self._convert_claude_response_to_unified(response)
    
    def get_tool_schema(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert tools to Claude schema."""
        return [tool.to_dict() for tool in tools]
    
    def _convert_messages_to_claude_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert unified message format to Claude format."""
        claude_messages = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "tool":
                # Handle tool results
                claude_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id"),
                        "content": content
                    }]
                })
            else:
                # Handle regular messages
                if isinstance(content, str):
                    claude_messages.append({
                        "role": role,
                        "content": content
                    })
                else:
                    claude_messages.append({
                        "role": role,
                        "content": content
                    })
        
        return claude_messages
    
    def _convert_claude_response_to_unified(self, response) -> Dict[str, Any]:
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
            "usage": response.usage
        }
    
    def _get_beta_headers(self, tools: List[Dict[str, Any]]) -> List[str]:
        """Get beta headers needed for tools."""
        betas = []
        
        # Check for computer use tools
        computer_tools = [tool for tool in tools if tool.get("name") == "computer"]
        if computer_tools:
            model = self.config.model.lower()
            if "claude-4" in model or "claude-sonnet-3.7" in model or "claude-sonnet-4" in model:
                betas.append("computer-use-2025-01-24")
            elif "claude-sonnet-3.5" in model:
                betas.append("computer-use-2024-10-22")
            else:
                betas.append("computer-use-2025-01-24")
        
        # Check for code execution tools
        code_execution_tools = [tool for tool in tools if "code_execution" in tool.get("tool_type", "")]
        if code_execution_tools:
            betas.append("code-execution-2025-05-22")
            
            # Check if any code execution tools support files
            for tool in code_execution_tools:
                if tool.get("supports_files", False):
                    betas.append("files-api-2025-04-14")
                    break
        
        return betas


class OpenAIProvider(ProviderInterface):
    """OpenAI provider implementation."""
    
    def __init__(self, config: AgentConfig):
        """Initialize OpenAI provider."""
        self.config = config
        self.client = AsyncOpenAI(api_key=config.api_key)
    
    async def create_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a message with OpenAI."""
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        # Prepare parameters
        params = {
            "model": self.config.model,
            "messages": openai_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        # Add tools if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        # Create message
        response = await self.client.chat.completions.create(**params)
        
        # Convert response to unified format
        return self._convert_openai_response_to_unified(response)
    
    def get_tool_schema(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI schema."""
        return [tool.to_openai_dict() for tool in tools]
    
    def _convert_messages_to_openai_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert unified message format to OpenAI format."""
        openai_messages = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "tool":
                # Handle tool results
                openai_messages.append({
                    "role": "tool",
                    "content": content,
                    "tool_call_id": message.get("tool_call_id")
                })
            else:
                # Handle regular messages
                if isinstance(content, str):
                    openai_messages.append({
                        "role": role,
                        "content": content
                    })
                else:
                    # Handle content blocks
                    text_content = []
                    for block in content:
                        if block["type"] == "text":
                            text_content.append(block["text"])
                    
                    openai_messages.append({
                        "role": role,
                        "content": " ".join(text_content)
                    })
        
        return openai_messages
    
    def _convert_openai_response_to_unified(self, response) -> Dict[str, Any]:
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
                tool_calls.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": tool_call.function.arguments
                })
                content.append({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": tool_call.function.arguments
                })
        
        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": response.usage
        } 