#!/usr/bin/env python3
"""Example of using the computer use tool with Claude."""

import asyncio
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from agents.tools.computer_use import ComputerUseTool


async def computer_use_example():
    """Example of using computer use tool."""
    print("Computer Use Tool Example")
    print("=" * 40)
    
    # Check if we have an API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return
    
    # Create the computer use tool
    computer_tool = ComputerUseTool(
        display_width_px=1024,
        display_height_px=768,
        tool_version="computer_20250124"
    )
    
    print(f"Created computer use tool: {computer_tool.name}")
    print(f"Tool version: {computer_tool.tool_version}")
    print(f"Display size: {computer_tool.width}x{computer_tool.height}")
    
    # Test basic tool functionality
    print("\nTesting screenshot capability...")
    try:
        result = await computer_tool.execute(action="screenshot")
        if "Screenshot captured successfully" in result:
            print("✓ Screenshot capability working")
        else:
            print(f"Screenshot result: {result}")
    except Exception as e:
        print(f"✗ Screenshot failed: {e}")
        print("Note: This is expected if running without a display environment")
    
    # Show tool schema
    print(f"\nTool API schema:")
    schema = computer_tool.to_dict()
    for key, value in schema.items():
        print(f"  {key}: {value}")
    
    print("\nExample completed!")
    print("\nTo use with Claude CLI:")
    print("claude-agent --interactive --tools computer_use --prompt 'Take a screenshot'")


if __name__ == "__main__":
    asyncio.run(computer_use_example())
