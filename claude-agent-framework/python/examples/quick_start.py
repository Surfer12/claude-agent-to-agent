#!/usr/bin/env python3
"""Quick start example for Claude Agent Framework."""

import asyncio
import os
from claude_agent import Agent, AgentConfig, get_tool


async def main():
    """Quick start example."""
    print("Claude Agent Framework - Quick Start")
    print("=" * 40)
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # Create configuration
    config = AgentConfig(
        name="quick-start-agent",
        system_prompt="You are a helpful AI assistant. Be concise and friendly.",
        verbose=True
    )
    
    # Get some tools
    think_tool = get_tool("think")
    file_read_tool = get_tool("file_read")
    
    # Create agent with specific tools
    agent = Agent(config=config, tools=[think_tool, file_read_tool])
    
    print(f"Created agent: {agent.name}")
    print(f"Available tools: {[tool.name for tool in agent.tools]}")
    
    # Test a simple interaction
    print("\nTesting agent interaction...")
    try:
        response = await agent.run_async("Hello! Can you tell me what tools you have available?")
        
        # Print response
        for block in response.content:
            if block.type == "text":
                print(f"Agent: {block.text}")
                
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This requires a valid ANTHROPIC_API_KEY")


if __name__ == "__main__":
    asyncio.run(main())
