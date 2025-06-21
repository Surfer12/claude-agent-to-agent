"""Agent implementation with Claude API and tools."""

import asyncio
import os
from contextlib import AsyncExitStack
from typing import Any

from anthropic import Anthropic

from ..tools.base import Tool
from ..tools.mcp.connections import setup_mcp_connections
from .history import MessageHistory
from .config import AgentConfig, ModelConfig
from ..utils.tool_util import execute_tools


class Agent:
    """Claude-powered agent with tool use capabilities."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        tools: list[Tool] | None = None,
        client: Anthropic | None = None,
        message_params: dict[str, Any] | None = None,
    ):
        """Initialize an Agent.
        
        Args:
            config: Agent configuration with defaults
            tools: List of tools available to the agent (overrides config)
            client: Anthropic client instance
            message_params: Additional parameters for client.messages.create().
                           These override any conflicting parameters from config.
        """
        self.config = config or AgentConfig()
        self.name = self.config.name
        self.system = self.config.system_prompt
        self.verbose = self.config.verbose
        self.tools = list(tools or [])
        self.mcp_servers = [{"url": url} for url in self.config.mcp_servers]
        self.message_params = message_params or {}
        self.client = client or Anthropic(
            api_key=self.config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.history = MessageHistory(
            model=self.config.model_config.model,
            system=self.system,
            context_window_tokens=self.config.model_config.context_window_tokens,
            client=self.client,
        )

        if self.verbose:
            print(f"\n[{self.name}] Agent initialized")

    def _prepare_message_params(self) -> dict[str, Any]:
        """Prepare parameters for client.messages.create() call.
        
        Returns a dict with base parameters from config, with any
        message_params overriding conflicting keys.
        """
        params = {
            "model": self.config.model_config.model,
            "max_tokens": self.config.model_config.max_tokens,
            "temperature": self.config.model_config.temperature,
            "system": self.system,
            "messages": self.history.format_for_api(),
            "tools": [tool.to_dict() for tool in self.tools],
            **self.message_params,
        }
        
        # Collect beta headers needed for tools
        beta_headers = set()
        
        # Add beta headers for computer use tools
        computer_tools = [tool for tool in self.tools if hasattr(tool, 'tool_version') and tool.name == 'computer']
        if computer_tools:
            model = self.config.model_config.model.lower()
            tool_version = computer_tools[0].tool_version
            
            if "claude-4" in model or "claude-sonnet-3.7" in model or "claude-sonnet-4" in model:
                if tool_version == "computer_20250124":
                    beta_headers.add("computer-use-2025-01-24")
                else:
                    beta_headers.add("computer-use-2024-10-22")
            elif "claude-sonnet-3.5" in model:
                beta_headers.add("computer-use-2024-10-22")
            else:
                beta_headers.add("computer-use-2025-01-24")
        
        # Add beta headers for code execution tools
        code_execution_tools = [tool for tool in self.tools if hasattr(tool, 'tool_type') and 'code_execution' in tool.tool_type]
        if code_execution_tools:
            beta_headers.add("code-execution-2025-05-22")
            
            # Check if any code execution tools support files
            for tool in code_execution_tools:
                if hasattr(tool, 'supports_files') and tool.supports_files:
                    beta_headers.add("files-api-2025-04-14")
                    break
        
        # Add beta headers to params if any are needed
        if beta_headers:
            params["betas"] = list(beta_headers)
                
        return params

    async def _agent_loop(self, user_input: str) -> list[dict[str, Any]]:
        """Process user input and handle tool calls in a loop"""
        if self.verbose:
            print(f"\n[{self.name}] Received: {user_input}")
        await self.history.add_message("user", user_input, None)

        tool_dict = {tool.name: tool for tool in self.tools}

        while True:
            self.history.truncate()
            params = self._prepare_message_params()

            # Use beta client if beta tools are present
            beta_tools = [tool for tool in self.tools if 
                         (hasattr(tool, 'tool_version') and tool.name == 'computer') or
                         (hasattr(tool, 'tool_type') and 'code_execution' in tool.tool_type)]
            if beta_tools:
                response = self.client.beta.messages.create(**params)
            else:
                response = self.client.messages.create(**params)
            tool_calls = [
                block for block in response.content if block.type == "tool_use"
            ]

            if self.verbose:
                for block in response.content:
                    if block.type == "text":
                        print(f"\n[{self.name}] Output: {block.text}")
                    elif block.type == "tool_use":
                        params_str = ", ".join(
                            [f"{k}={v}" for k, v in block.input.items()]
                        )
                        print(
                            f"\n[{self.name}] Tool call: "
                            f"{block.name}({params_str})"
                        )

            await self.history.add_message(
                "assistant", response.content, response.usage
            )

            if tool_calls:
                tool_results = await execute_tools(
                    tool_calls,
                    tool_dict,
                )
                if self.verbose:
                    for block in tool_results:
                        print(
                            f"\n[{self.name}] Tool result: "
                            f"{block.get('content')}"
                        )
                await self.history.add_message("user", tool_results)
            else:
                return response

    async def run_async(self, user_input: str) -> list[dict[str, Any]]:
        """Run agent with MCP tools asynchronously."""
        async with AsyncExitStack() as stack:
            original_tools = list(self.tools)

            try:
                mcp_tools = await setup_mcp_connections(
                    self.mcp_servers, stack
                )
                self.tools.extend(mcp_tools)
                return await self._agent_loop(user_input)
            finally:
                self.tools = original_tools

    def run(self, user_input: str) -> list[dict[str, Any]]:
        """Run agent synchronously"""
        return asyncio.run(self.run_async(user_input))
