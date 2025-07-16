"""
Base tool class for the unified agent system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


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