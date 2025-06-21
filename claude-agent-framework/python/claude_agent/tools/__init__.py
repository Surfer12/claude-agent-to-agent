"""Tool system for Claude Agent Framework"""

from .base import Tool
from .registry import ToolRegistry, get_tool_registry, get_available_tools, get_tool, register_tool

# Auto-discover tools on import
_registry = get_tool_registry()
_registry.discover_tools()

__all__ = [
    "Tool", 
    "ToolRegistry", 
    "get_tool_registry",
    "get_available_tools", 
    "get_tool",
    "register_tool"
]
