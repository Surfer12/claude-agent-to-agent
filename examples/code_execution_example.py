#!/usr/bin/env python3
"""Example of using the code execution tool with Claude."""

import asyncio
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.tools.code_execution import CodeExecutionTool, CodeExecutionWithFilesTool, get_supported_models, is_model_supported


async def code_execution_example():
    """Example of using code execution tool."""
    print("Code Execution Tool Example")
    print("=" * 40)
    
    # Check supported models
    print("Supported models for code execution:")
    for model in get_supported_models():
        print(f"  - {model}")
    
    print(f"\nModel support check:")
    test_models = [
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514", 
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219"
    ]
    
    for model in test_models:
        supported = "✓" if is_model_supported(model) else "✗"
        print(f"  {supported} {model}")
    
    # Create basic code execution tool
    print(f"\nCreating basic code execution tool...")
    basic_tool = CodeExecutionTool()
    print(f"Tool name: {basic_tool.name}")
    print(f"Tool type: {basic_tool.tool_type}")
    
    # Show tool schema
    print(f"\nBasic tool schema:")
    schema = basic_tool.to_dict()
    for key, value in schema.items():
        print(f"  {key}: {value}")
    
    # Create code execution tool with file support
    print(f"\nCreating code execution tool with file support...")
    file_tool = CodeExecutionWithFilesTool()
    print(f"Tool name: {file_tool.name}")
    print(f"Tool type: {file_tool.tool_type}")
    print(f"Supports files: {file_tool.supports_files}")
    
    # Show required beta headers
    print(f"\nRequired beta headers for file support:")
    for header in file_tool.get_beta_headers():
        print(f"  - {header}")
    
    print(f"\nFile tool schema:")
    schema = file_tool.to_dict()
    for key, value in schema.items():
        print(f"  {key}: {value}")
    
    print("\nExample completed!")
    print("\nTo use with Claude CLI:")
    print("# Basic code execution")
    print("claude-agent --interactive --tools code_execution --prompt 'Calculate the factorial of 10'")
    print("\n# With file support")
    print("claude-agent --interactive --tools code_execution --enable-file-support --prompt 'Create a data visualization'")


if __name__ == "__main__":
    asyncio.run(code_execution_example())
