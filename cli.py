#!/usr/bin/env python
"""Claude Agent-to-Agent CLI"""

import argparse
import asyncio
import os
import sys
from typing import List, Optional

from anthropic import Anthropic

from agents.agent import Agent, ModelConfig
from agents.tools.think import ThinkTool
from agents.tools.file_tools import FileReadTool, FileWriteTool
from agents.utils.connections import setup_mcp_connections


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Claude Agent-to-Agent CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Basic configuration
    parser.add_argument("--name", default="claude-cli", help="Agent name for logging")
    parser.add_argument("--system", help="System prompt for the agent")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Input modes
    input_group = parser.add_argument_group("Input options")
    input_group.add_argument("--prompt", help="Single prompt to send to the agent")
    input_group.add_argument(
        "--interactive", action="store_true", help="Start interactive session"
    )
    input_group.add_argument(
        "--file", help="Read prompt from file (use - for stdin)"
    )
    
    # Tool configuration
    tool_group = parser.add_argument_group("Tool options")
    tool_group.add_argument(
        "--tools", 
        nargs="+", 
        choices=["think", "file_read", "file_write", "all"],
        default=["all"],
        help="Enable specific tools"
    )
    tool_group.add_argument(
        "--mcp-server", 
        action="append", 
        help="MCP server URL (can be specified multiple times)"
    )
    
    # API configuration
    api_group = parser.add_argument_group("API options")
    api_group.add_argument(
        "--api-key", 
        help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Validate input mode
    input_modes = sum(
        bool(mode) for mode in [args.prompt, args.interactive, args.file]
    )
    if input_modes != 1:
        parser.error("Exactly one input mode (--prompt, --interactive, or --file) is required")
    
    # Set system prompt if not provided
    if not args.system:
        args.system = "You are Claude, an AI assistant. Be concise and helpful."
    
    return args


def get_enabled_tools(tool_names: List[str]) -> List:
    """Get enabled tool instances based on names."""
    tools = []
    
    if "all" in tool_names or "think" in tool_names:
        tools.append(ThinkTool())
    
    if "all" in tool_names or "file_read" in tool_names:
        tools.append(FileReadTool())
    
    if "all" in tool_names or "file_write" in tool_names:
        tools.append(FileWriteTool())
    
    return tools


def setup_mcp_servers(server_urls: Optional[List[str]]) -> List[dict]:
    """Configure MCP server connections."""
    if not server_urls:
        return []
    
    return [
        {
            "url": url,
            "connection_type": "sse" if url.startswith("http") else "stdio",
        }
        for url in server_urls
    ]


def format_response(response):
    """Format agent response for display."""
    output = []
    
    for block in response.content:
        if block.type == "text":
            output.append(block.text)
    
    return "\n".join(output)


async def handle_interactive_session(agent: Agent):
    """Run an interactive session with the agent."""
    print(f"Starting interactive session with {agent.name}...")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'clear' to clear conversation history.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ("exit", "quit"):
                print("Ending session.")
                break
                
            if user_input.lower() == "clear":
                agent.history.clear()
                print("Conversation history cleared.")
                continue
                
            if not user_input.strip():
                continue
                
            response = await agent.run_async(user_input)
            print("\nClaude:", format_response(response))
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


async def handle_single_prompt(agent: Agent, prompt: str):
    """Run a single prompt through the agent."""
    try:
        response = await agent.run_async(prompt)
        print(format_response(response))
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


async def handle_file_input(agent: Agent, file_path: str):
    """Run agent with input from a file."""
    try:
        if file_path == "-":
            content = sys.stdin.read()
        else:
            with open(file_path, "r") as f:
                content = f.read()
                
        response = await agent.run_async(content)
        print(format_response(response))
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


async def main_async():
    """Async entry point for the CLI."""
    args = parse_args()
    
    # Configure API client
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: Anthropic API key not provided. Set ANTHROPIC_API_KEY environment "
            "variable or use --api-key",
            file=sys.stderr,
        )
        sys.exit(1)
    
    client = Anthropic(api_key=api_key)
    
    # Configure agent
    config = ModelConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    tools = get_enabled_tools(args.tools)
    mcp_servers = setup_mcp_servers(args.mcp_server)
    
    agent = Agent(
        name=args.name,
        system=args.system,
        tools=tools,
        mcp_servers=mcp_servers,
        config=config,
        verbose=args.verbose,
        client=client,
    )
    
    # Handle input mode
    if args.interactive:
        await handle_interactive_session(agent)
    elif args.prompt:
        await handle_single_prompt(agent, args.prompt)
    elif args.file:
        await handle_file_input(agent, args.file)


def main():
    """Entry point for the CLI."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()