"""Tool registry system for automatic tool discovery and management."""

import importlib
import inspect
import pkgutil
from typing import Dict, List, Type, Optional, Any

from .base import Tool


class ToolRegistry:
    """Registry for managing and discovering tools."""
    
    def __init__(self):
        self._tools: Dict[str, Type[Tool]] = {}
        self._instances: Dict[str, Tool] = {}
        self._discovered = False
    
    def register_tool(self, tool_class: Type[Tool], name: Optional[str] = None):
        """Register a tool class.
        
        Args:
            tool_class: The tool class to register
            name: Optional name override (uses tool.name by default)
        """
        if not issubclass(tool_class, Tool):
            raise ValueError(f"Tool class must inherit from Tool: {tool_class}")
        
        # Get tool name from class or parameter
        if name:
            tool_name = name
        else:
            # Try to get name from class attribute or instantiate to get name
            if hasattr(tool_class, 'name') and isinstance(tool_class.name, str):
                tool_name = tool_class.name
            else:
                # Instantiate to get name (for tools that set name in __init__)
                try:
                    instance = tool_class()
                    tool_name = instance.name
                except Exception as e:
                    raise ValueError(f"Could not determine tool name for {tool_class}: {e}")
        
        self._tools[tool_name] = tool_class
        
    def discover_tools(self):
        """Automatically discover and register tools from builtin and beta packages."""
        if self._discovered:
            return
            
        # Discover builtin tools
        self._discover_package_tools('claude_agent.tools.builtin')
        
        # Discover beta tools
        self._discover_package_tools('claude_agent.tools.beta')
        
        # Discover MCP tools
        self._discover_package_tools('claude_agent.tools.mcp')
        
        self._discovered = True
    
    def _discover_package_tools(self, package_name: str):
        """Discover tools in a specific package."""
        try:
            package = importlib.import_module(package_name)
            package_path = package.__path__
            
            for _, module_name, _ in pkgutil.iter_modules(package_path):
                if module_name.startswith('_'):
                    continue
                    
                try:
                    module = importlib.import_module(f"{package_name}.{module_name}")
                    self._discover_module_tools(module)
                except Exception as e:
                    print(f"Warning: Could not import {package_name}.{module_name}: {e}")
                    
        except ImportError as e:
            print(f"Warning: Could not import package {package_name}: {e}")
    
    def _discover_module_tools(self, module):
        """Discover tool classes in a module."""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, Tool) and 
                obj is not Tool and
                not name.startswith('_')):
                try:
                    self.register_tool(obj)
                except Exception as e:
                    print(f"Warning: Could not register tool {name}: {e}")
    
    def get_tool(self, name: str, **kwargs) -> Tool:
        """Get a tool instance by name.
        
        Args:
            name: Tool name
            **kwargs: Arguments to pass to tool constructor
            
        Returns:
            Tool instance
        """
        if not self._discovered:
            self.discover_tools()
            
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}. Available tools: {list(self._tools.keys())}")
        
        # Create new instance with provided kwargs
        tool_class = self._tools[name]
        return tool_class(**kwargs)
    
    def get_cached_tool(self, name: str, **kwargs) -> Tool:
        """Get a cached tool instance by name.
        
        Args:
            name: Tool name
            **kwargs: Arguments to pass to tool constructor (only used if not cached)
            
        Returns:
            Cached tool instance
        """
        if name not in self._instances:
            self._instances[name] = self.get_tool(name, **kwargs)
        return self._instances[name]
    
    def list_tools(self) -> List[str]:
        """Get list of all registered tool names."""
        if not self._discovered:
            self.discover_tools()
        return list(self._tools.keys())
    
    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """Get information about a tool.
        
        Args:
            name: Tool name
            
        Returns:
            Dictionary with tool information
        """
        if not self._discovered:
            self.discover_tools()
            
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        
        tool_class = self._tools[name]
        
        # Try to get tool info
        try:
            instance = tool_class()
            return {
                'name': instance.name,
                'description': instance.description,
                'input_schema': instance.input_schema,
                'class': tool_class.__name__,
                'module': tool_class.__module__,
            }
        except Exception as e:
            return {
                'name': name,
                'description': f"Error getting description: {e}",
                'class': tool_class.__name__,
                'module': tool_class.__module__,
            }
    
    def clear_cache(self):
        """Clear cached tool instances."""
        self._instances.clear()
    
    def reset(self):
        """Reset the registry (clear all tools and cache)."""
        self._tools.clear()
        self._instances.clear()
        self._discovered = False


# Global registry instance
_global_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry


def get_available_tools() -> List[str]:
    """Get list of all available tools."""
    return _global_registry.list_tools()


def get_tool(name: str, **kwargs) -> Tool:
    """Get a tool instance by name."""
    return _global_registry.get_tool(name, **kwargs)


def register_tool(tool_class: Type[Tool], name: Optional[str] = None):
    """Register a tool class with the global registry."""
    _global_registry.register_tool(tool_class, name)
