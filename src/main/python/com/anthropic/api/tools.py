"""
Tool implementations for Anthropic API

This module provides specialized tool classes for various Anthropic API features
including bash execution, web search, weather, text editing, and code execution.
"""

import anthropic
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections.abc import Mapping
from frozendict import frozendict


@dataclass(frozen=True)
class ToolDefinition:
    """Immutable tool definition for API requests."""
    name: str
    tool_type: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    max_uses: Optional[int] = None
    display_config: Optional[Dict[str, Any]] = None


class BaseTool:
    """Base class for all Anthropic API tools."""
    
    def __init__(self, definition: ToolDefinition):
        self.definition = definition
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool definition to dictionary for API request."""
        result = {
            "type": self.definition.tool_type,
            "name": self.definition.name,
            "description": self.definition.description
        }
        
        if self.definition.input_schema:
            result["input_schema"] = self.definition.input_schema
        if self.definition.max_uses:
            result["max_uses"] = self.definition.max_uses
        if self.definition.display_config:
            result.update(self.definition.display_config)
        
        return result


class BashTool(BaseTool):
    """Tool for executing bash commands."""
    
    def __init__(self):
        definition = ToolDefinition(
            name="bash",
            tool_type="bash_20250124",
            description="Execute bash commands in a secure environment"
        )
        super().__init__(definition)


class WebSearchTool(BaseTool):
    """Tool for web search functionality."""
    
    def __init__(self, max_uses: int = 5):
        definition = ToolDefinition(
            name="web_search",
            tool_type="web_search_20250305", 
            description="Search the web for current information",
            max_uses=max_uses
        )
        super().__init__(definition)


class WeatherTool(BaseTool):
    """Tool for weather information."""
    
    def __init__(self):
        definition = ToolDefinition(
            name="get_weather",
            tool_type="weather_tool",
            description="Get the current weather in a given location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        )
        super().__init__(definition)


class TextEditorTool(BaseTool):
    """Tool for text editing operations."""
    
    def __init__(self):
        definition = ToolDefinition(
            name="str_replace_based_edit_tool",
            tool_type="text_editor_20250429",
            description="Edit text files with string replacement operations"
        )
        super().__init__(definition)


class CodeExecutionTool(BaseTool):
    """Tool for code execution."""
    
    def __init__(self):
        definition = ToolDefinition(
            name="code_execution",
            tool_type="code_execution_20250522",
            description="Execute code in a secure environment"
        )
        super().__init__(definition)


class ComputerTool(BaseTool):
    """Tool for computer interface interaction."""
    
    def __init__(self, display_width: int = 1024, display_height: int = 768, display_number: int = 1):
        definition = ToolDefinition(
            name="computer",
            tool_type="computer_20250124",
            description="Interact with computer interface",
            display_config={
                "display_width_px": display_width,
                "display_height_px": display_height,
                "display_number": display_number
            }
        )
        super().__init__(definition)


class StreamingTools:
    """Collection of tools for streaming operations."""
    
    @staticmethod
    def make_file_tool() -> Dict[str, Any]:
        """Create a file creation tool for streaming."""
        return {
            "name": "make_file",
            "description": "Write text to a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename to write text to"
                    },
                    "lines_of_text": {
                        "type": "array",
                        "description": "An array of lines of text to write to the file"
                    }
                },
                "required": ["filename", "lines_of_text"]
            }
        }


# Tool factory functions
def create_bash_tool() -> BashTool:
    """Create a bash tool instance."""
    return BashTool()


def create_web_search_tool(max_uses: int = 5) -> WebSearchTool:
    """Create a web search tool instance."""
    return WebSearchTool(max_uses=max_uses)


def create_weather_tool() -> WeatherTool:
    """Create a weather tool instance."""
    return WeatherTool()


def create_text_editor_tool() -> TextEditorTool:
    """Create a text editor tool instance."""
    return TextEditorTool()


def create_code_execution_tool() -> CodeExecutionTool:
    """Create a code execution tool instance."""
    return CodeExecutionTool()


def create_computer_tool(display_width: int = 1024, display_height: int = 768) -> ComputerTool:
    """Create a computer tool instance."""
    return ComputerTool(display_width, display_height)


# Example usage functions
def example_bash_usage(client: anthropic.Anthropic) -> Any:
    """Example of using bash tool."""
    return client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[create_bash_tool().to_dict()],
        messages=[
            {"role": "user", "content": "List all Python files in the current directory."}
        ]
    )


def example_web_search_usage(client: anthropic.Anthropic) -> Any:
    """Example of using web search tool."""
    return client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "How do I update a web app to TypeScript 5.5?"}
        ],
        tools=[create_web_search_tool().to_dict()]
    )


def example_weather_usage(client: anthropic.Anthropic) -> Any:
    """Example of using weather tool."""
    return client.beta.messages.create(
        max_tokens=1024,
        model="claude-3-7-sonnet-20250219",
        tools=[create_weather_tool().to_dict()],
        messages=[
            {"role": "user", "content": "Tell me the weather in San Francisco."}
        ],
        betas=["token-efficient-tools-2025-02-19"]
    )


def example_code_execution_usage(client: anthropic.Anthropic) -> Any:
    """Example of using code execution tool."""
    return client.beta.messages.create(
        model="claude-opus-4-20250514",
        betas=["code-execution-2025-05-22"],
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": "Calculate the mean and standard deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
        }],
        tools=[create_code_execution_tool().to_dict()]
    )


def example_streaming_usage(client: anthropic.Anthropic) -> Any:
    """Example of using streaming with tools."""
    return client.messages.stream(
        max_tokens=65536,
        model="claude-sonnet-4-20250514",
        tools=[StreamingTools.make_file_tool()],
        messages=[{
            "role": "user",
            "content": "Can you write a long poem and make a file called poem.txt?"
        }],
        betas=["fine-grained-tool-streaming-2025-05-14"]
    ) 