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