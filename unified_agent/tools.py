"""
Unified tool registry for integrating tools from different implementations.
"""

import asyncio
from typing import Any, Dict, List, Optional

from .tools.base import BaseTool
from .tools.computer_use import ComputerUseTool
from .tools.code_execution import CodeExecutionTool
from .tools.file_tools import FileTools


class ToolRegistry:
    """Registry for managing tools across different providers."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, BaseTool] = {}
        self.active_tools: List[BaseTool] = []
    
    def register_tool(self, tool: BaseTool):
        """Register a tool in the registry."""
        self.tools[tool.name] = tool
        self.active_tools.append(tool)
    
    def unregister_tool(self, tool_name: str):
        """Unregister a tool from the registry."""
        if tool_name in self.tools:
            tool = self.tools.pop(tool_name)
            if tool in self.active_tools:
                self.active_tools.remove(tool)
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def get_active_tools(self) -> List[BaseTool]:
        """Get all active tools."""
        return self.active_tools.copy()
    
    def register_code_execution_tools(self):
        """Register code execution tools."""
        code_tool = CodeExecutionTool()
        self.register_tool(code_tool)
    
    def register_computer_use_tools(self, computer_type: str = "local-playwright"):
        """Register computer use tools."""
        computer_tool = ComputerUseTool(computer_type=computer_type)
        self.register_tool(computer_tool)
    
    def register_file_tools(self):
        """Register file manipulation tools."""
        file_tools = FileTools()
        for tool in file_tools.get_tools():
            self.register_tool(tool)
    
    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool by name."""
        tool = self.get_tool(tool_name)
        if not tool:
            return f"Tool '{tool_name}' not found"
        
        try:
            result = await tool.execute(tool_input)
            return result
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())


class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self, name: str, description: str):
        """Initialize a tool."""
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> str:
        """Execute the tool with given input."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to Claude format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_input_schema()
        }
    
    def to_openai_dict(self) -> Dict[str, Any]:
        """Convert tool to OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_input_schema()
            }
        }
    
    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Get the input schema for the tool."""
        pass


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


class ComputerUseTool(BaseTool):
    """Tool for computer use capabilities."""
    
    def __init__(self, computer_type: str = "local-playwright"):
        """Initialize computer use tool."""
        super().__init__(
            name="computer",
            description="Interact with computer environment including web browsing, file operations, and system commands"
        )
        self.computer_type = computer_type
        self.tool_version = "computer_20250124"
        self._computer = None
    
    async def execute(self, input_data: Dict[str, Any]) -> str:
        """Execute computer use action."""
        action = input_data.get("action", "")
        params = input_data.get("params", {})
        
        if not action:
            return "No action specified"
        
        try:
            # Initialize computer if not already done
            if self._computer is None:
                await self._initialize_computer()
            
            # Execute the action
            result = await self._execute_action(action, params)
            return result
        except Exception as e:
            return f"Error executing computer action: {str(e)}"
    
    async def _initialize_computer(self):
        """Initialize the computer environment."""
        # This would integrate with the computer use implementations
        # For now, we'll create a placeholder
        self._computer = {"type": self.computer_type, "initialized": True}
    
    async def _execute_action(self, action: str, params: Dict[str, Any]) -> str:
        """Execute a specific computer action."""
        # This would integrate with the actual computer use implementations
        # For now, return a placeholder response
        return f"Executed {action} with params {params} on {self.computer_type}"
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema for computer use."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform (e.g., 'click', 'type', 'navigate', 'screenshot')"
                },
                "params": {
                    "type": "object",
                    "description": "Parameters for the action"
                }
            },
            "required": ["action"]
        }


class FileTools:
    """Collection of file manipulation tools."""
    
    def __init__(self):
        """Initialize file tools."""
        self.tools = [
            ReadFileTool(),
            WriteFileTool(),
            ListDirectoryTool(),
            DeleteFileTool()
        ]
    
    def get_tools(self) -> List[BaseTool]:
        """Get all file tools."""
        return self.tools


class ReadFileTool(BaseTool):
    """Tool for reading files."""
    
    def __init__(self):
        """Initialize read file tool."""
        super().__init__(
            name="read_file",
            description="Read the contents of a file"
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> str:
        """Read a file."""
        file_path = input_data.get("file_path", "")
        if not file_path:
            return "No file path provided"
        
        try:
            content = await asyncio.to_thread(self._read_file, file_path)
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _read_file(self, file_path: str) -> str:
        """Read file contents."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema for read file."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["file_path"]
        }


class WriteFileTool(BaseTool):
    """Tool for writing files."""
    
    def __init__(self):
        """Initialize write file tool."""
        super().__init__(
            name="write_file",
            description="Write content to a file"
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> str:
        """Write to a file."""
        file_path = input_data.get("file_path", "")
        content = input_data.get("content", "")
        
        if not file_path:
            return "No file path provided"
        
        try:
            await asyncio.to_thread(self._write_file, file_path, content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def _write_file(self, file_path: str, content: str):
        """Write content to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema for write file."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        }


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""
    
    def __init__(self):
        """Initialize list directory tool."""
        super().__init__(
            name="list_directory",
            description="List contents of a directory"
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> str:
        """List directory contents."""
        directory = input_data.get("directory", ".")
        
        try:
            contents = await asyncio.to_thread(self._list_directory, directory)
            return contents
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    def _list_directory(self, directory: str) -> str:
        """List directory contents."""
        import os
        try:
            items = os.listdir(directory)
            return "\n".join(items)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema for list directory."""
        return {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to list (default: current directory)"
                }
            }
        }


class DeleteFileTool(BaseTool):
    """Tool for deleting files."""
    
    def __init__(self):
        """Initialize delete file tool."""
        super().__init__(
            name="delete_file",
            description="Delete a file or directory"
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> str:
        """Delete a file."""
        file_path = input_data.get("file_path", "")
        if not file_path:
            return "No file path provided"
        
        try:
            await asyncio.to_thread(self._delete_file, file_path)
            return f"Successfully deleted {file_path}"
        except Exception as e:
            return f"Error deleting file: {str(e)}"
    
    def _delete_file(self, file_path: str):
        """Delete a file."""
        import os
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            import shutil
            shutil.rmtree(file_path)
        else:
            raise FileNotFoundError(f"File or directory not found: {file_path}")
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema for delete file."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file or directory to delete"
                }
            },
            "required": ["file_path"]
        } 