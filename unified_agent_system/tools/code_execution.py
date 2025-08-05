"""
Code execution tool for the unified agent system.
"""

import asyncio
from typing import Any, Dict

from .base import BaseTool


class CodeExecutionTool(BaseTool):
    """Tool for executing code."""
    
    def __init__(self):
        """Initialize code execution tool."""
        super().__init__(
            name="code_execution",
            description="Execute Python code in a sandboxed environment"
        )
        self.tool_type = "code_execution"
        self.supports_files = True
    
    async def execute(self, input_data: Dict[str, Any]) -> str:
        """Execute Python code."""
        code = input_data.get("code", "")
        if not code:
            return "No code provided"
        
        try:
            # Execute code in a safe environment
            result = await asyncio.to_thread(self._execute_code, code)
            return result
        except Exception as e:
            return f"Error executing code: {str(e)}"
    
    def _execute_code(self, code: str) -> str:
        """Execute code in a safe environment."""
        # This is a simplified implementation
        # In production, you'd want proper sandboxing
        try:
            # Create a restricted globals dict
            restricted_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                    'pow': pow,
                    'divmod': divmod,
                    'all': all,
                    'any': any,
                    'sorted': sorted,
                    'reversed': reversed,
                }
            }
            
            # Execute the code
            exec(code, restricted_globals)
            return "Code executed successfully"
        except Exception as e:
            return f"Code execution error: {str(e)}"
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema for code execution."""
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        } 