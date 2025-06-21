#!/usr/bin/env python3
"""Test script for computer use tool."""

import asyncio
import os
from agents.tools.computer_use import ComputerUseTool


async def test_computer_use():
    """Test basic computer use functionality."""
    print("Testing Computer Use Tool...")
    
    # Create tool instance
    tool = ComputerUseTool(
        display_width_px=1024,
        display_height_px=768,
        tool_version="computer_20250124"
    )
    
    print(f"Tool created: {tool.name}")
    print(f"Tool version: {tool.tool_version}")
    print(f"Display size: {tool.width}x{tool.height}")
    
    # Test screenshot
    print("\nTesting screenshot...")
    try:
        result = await tool.execute(action="screenshot")
        print(f"Screenshot result: {result}")
    except Exception as e:
        print(f"Screenshot failed: {e}")
        print("Note: This is expected if running without a display environment")
    
    # Test tool schema
    print(f"\nTool schema: {tool.to_dict()}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(test_computer_use())
