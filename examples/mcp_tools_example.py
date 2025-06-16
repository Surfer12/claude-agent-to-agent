#!/usr/bin/env python
"""Example demonstrating Claude Agent CLI with MCP tools."""

import os
import sys
import json
import asyncio
import tempfile
import subprocess
from pathlib import Path

# Make sure parent directory is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent import Agent, ModelConfig
from agents.tools.think import ThinkTool
from agents.tools.file_tools import FileReadTool, FileWriteTool


async def main():
    """Run the MCP tools example."""
    
    # Check for API key
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Error: Please set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)
    
    # Create a calculator MCP tool config file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_file:
        calculator_mcp = {
            "name": "calculator",
            "description": "A simple calculator tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            }
        }
        json.dump(calculator_mcp, temp_file)
        mcp_config_path = temp_file.name
    
    try:
        # Start an MCP server (mock implementation for the example)
        mcp_process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"""
import json
import sys
import math

def calculator(expression):
    try:
        # Using eval is not secure for production use!
        # This is just for the example
        allowed_names = {
            'abs': abs, 'pow': pow, 'round': round,
            'int': int, 'float': float,
            'max': max, 'min': min,
            'sum': sum,
            # Add math functions
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log,
        }}
        
        result = eval(expression, {{"__builtins__": {{}}}}, allowed_names)
        return {{"content": str(result)}}
    except Exception as e:
        return {{"content": f"Error: {{str(e)}}"}}

# Read MCP config
with open("{mcp_config_path}") as f:
    tool_config = json.load(f)

# Simple MCP server
while True:
    try:
        line = input()
        request = json.loads(line)
        
        if request["type"] == "registration":
            # Send tool registration
            print(json.dumps({{"type": "registration_response", "tools": [tool_config]}}))
        elif request["type"] == "tool_call":
            # Handle tool call
            if request["name"] == "calculator":
                result = calculator(request["arguments"]["expression"])
                print(json.dumps({{"type": "tool_response", "id": request["id"], **result}}))
            else:
                print(json.dumps({{"type": "tool_response", "id": request["id"], "content": "Unknown tool"}}))
    except EOFError:
        break
    except Exception as e:
        print(json.dumps({{"type": "error", "content": str(e)}}), file=sys.stderr)
                """,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        
        # Allow time for the MCP server to start
        await asyncio.sleep(1)
        
        print("Starting CLI with MCP server...")
        
        # Run the CLI with the MCP server
        result = subprocess.run(
            [
                "python",
                "../cli.py",
                "--prompt",
                "Calculate the value of (5 * 7) + sqrt(16) - sin(0)",
                "--mcp-server",
                "stdio:subprocess",
                "--verbose",
            ],
            input=json.dumps({"type": "registration"}) + "\n",
            text=True,
            capture_output=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        
        print("\nCLI Output:")
        print("-" * 50)
        print(result.stdout)
        
        if result.stderr:
            print("\nErrors:")
            print("-" * 50)
            print(result.stderr)
            
    finally:
        # Clean up
        if 'mcp_process' in locals():
            mcp_process.terminate()
            try:
                mcp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mcp_process.kill()
                
        os.unlink(mcp_config_path)


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())