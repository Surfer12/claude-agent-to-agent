"""
File manipulation tools for the unified agent system.
"""

import asyncio
import os
import shutil
from typing import Any, Dict, List

from .base import BaseTool


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
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
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