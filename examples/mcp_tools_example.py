#!/usr/bin/env python
"""Example demonstrating Claude Agent CLI with MCP tools."""

import os
import sys
import json
import asyncio
import tempfile
import subprocess
from pathlib import Path
import ast
import operator
import math

# Make sure parent directory is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent import Agent, ModelConfig
from agents.tools.think import ThinkTool
from agents.tools.file_tools import FileReadTool, FileWriteTool


# Safe mathematical expression evaluator
class SafeMathEvaluator:
    """Secure mathematical expression evaluator."""
    
    # Allowed operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
    }
    
    # Allowed functions
    functions = {
        'abs': abs,
        'round': round,
        'int': int,
        'float': float,
        'max': max,
        'min': min,
        'sum': sum,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'sqrt': math.sqrt,
        'exp': math.exp,
        'log': math.log,
        'pi': math.pi,
        'e': math.e,
    }
    
    def evaluate(self, expression: str) -> float:
        """Safely evaluate a mathematical expression."""
        try:
            # Parse the expression into an AST
            node = ast.parse(expression, mode='eval')
            return self._eval_node(node.body)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")
    
    def _eval_node(self, node):
        """Recursively evaluate AST nodes."""
        if isinstance(node, ast.Constant):  # Numbers
            return node.value
        elif isinstance(node, ast.Name):  # Variables/constants
            if node.id in self.functions:
                return self.functions[node.id]
            else:
                raise ValueError(f"Unknown identifier: {node.id}")
        elif isinstance(node, ast.BinOp):  # Binary operations
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            operand = self._eval_node(node.operand)
            op = self.operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op(operand)
        elif isinstance(node, ast.Call):  # Function calls
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name in self.functions:
                args = [self._eval_node(arg) for arg in node.args]
                return self.functions[func_name](*args)
            else:
                raise ValueError(f"Unknown function: {func_name}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")


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
        # Start an MCP server (secure implementation)
        mcp_process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"""
import json
import sys
import math
import ast
import operator

class SafeMathEvaluator:
    operators = {{
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }}
    
    functions = {{
        'abs': abs, 'round': round, 'int': int, 'float': float,
        'max': max, 'min': min, 'sum': sum,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log,
        'pi': math.pi, 'e': math.e,
    }}
    
    def evaluate(self, expression):
        try:
            node = ast.parse(expression, mode='eval')
            return self._eval_node(node.body)
        except Exception as e:
            raise ValueError(f"Invalid expression: {{e}}")
    
    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in self.functions:
                return self.functions[node.id]
            else:
                raise ValueError(f"Unknown identifier: {{node.id}}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {{type(node.op)}}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {{type(node.op)}}")
            return op(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name in self.functions:
                args = [self._eval_node(arg) for arg in node.args]
                return self.functions[func_name](*args)
            else:
                raise ValueError(f"Unknown function: {{func_name}}")
        else:
            raise ValueError(f"Unsupported node type: {{type(node)}}")

evaluator = SafeMathEvaluator()

def calculator(expression):
    try:
        # Use secure AST-based evaluator instead of eval()
        result = evaluator.evaluate(expression)
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