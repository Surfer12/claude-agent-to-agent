"""File operation tools for reading and writing files."""

import asyncio
import glob
import os
import re
from pathlib import Path
from typing import Optional

from .base import Tool

# Security constants
MAX_FILE_SIZE = 1024 * 1024  # 1MB
SAFE_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9_\-./]+$')
BLOCKED_EXTENSIONS = {'.exe', '.dll', '.so', '.dylib', '.sh', '.bat', '.cmd', '.ps1'}

def validate_path(path: str) -> Optional[str]:
    """Validate file path for security.
    
    Args:
        path: File path to validate
        
    Returns:
        Error message if validation fails, None if path is safe
    """
    # Check for path traversal
    try:
        path = os.path.abspath(path)
        if '..' in path or not SAFE_PATH_PATTERN.match(path):
            return "Invalid characters in path"
    except Exception:
        return "Invalid path format"
        
    # Check file extension
    ext = os.path.splitext(path)[1].lower()
    if ext in BLOCKED_EXTENSIONS:
        return f"File extension {ext} is not allowed"
        
    return None


class FileReadTool(Tool):
    """Tool for reading files and listing directories."""

    def __init__(self):
        super().__init__(
            name="file_read",
            description="""
            Read files or list directory contents.

            Operations:
            - read: Read the contents of a file
            - list: List files in a directory
            """,
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "list"],
                        "description": "File operation to perform",
                    },
                    "path": {
                        "type": "string",
                        "description": "File path for read or directory path",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum lines to read (0 means no limit)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "File pattern to match",
                    },
                },
                "required": ["operation", "path"],
            },
        )

    async def execute(
        self,
        operation: str,
        path: str,
        max_lines: int = 0,
        pattern: str = "*",
    ) -> str:
        """Execute a file read operation.

        Args:
            operation: The operation to perform (read or list)
            path: The file or directory path
            max_lines: Maximum lines to read (for read operation, 0 means no limit)
            pattern: File pattern to match (for list operation)

        Returns:
            Result of the operation as string
        """
        if operation == "read":
            return await self._read_file(path, max_lines)
        elif operation == "list":
            return await self._list_files(path, pattern)
        else:
            return f"Error: Unsupported operation '{operation}'"

    async def _read_file(self, path: str, max_lines: int = 0) -> str:
        """Read a file from disk securely."""
        try:
            # Validate path
            error = validate_path(path)
            if error:
                return f"Error: {error}"
                
            file_path = Path(path)
            
            if not file_path.exists():
                return f"Error: File not found at {path}"
            if not file_path.is_file():
                return f"Error: {path} is not a file"
            if not file_path.is_relative_to(Path.cwd()):
                return f"Error: Path {path} is outside working directory"
            if not os.access(file_path, os.R_OK):
                return f"Error: No read permission for {path}"
                
            # Check file size
            size = file_path.stat().st_size
            if size > MAX_FILE_SIZE:
                return f"Error: File too large (>{MAX_FILE_SIZE} bytes): {path}"
                
            def read_sync():
                with open(file_path, encoding='utf-8', errors='replace') as f:
                    if max_lines > 0:
                        lines = []
                        for i, line in enumerate(f):
                            if i >= max_lines:
                                break
                            lines.append(line)
                        return ''.join(lines)
                    return f.read()
                    
            return await asyncio.to_thread(read_sync)
        except Exception as e:
            return f"Error reading {path}: {str(e)}"

    async def _list_files(self, directory: str, pattern: str = "*") -> str:
        """List files in a directory."""
        try:
            dir_path = Path(directory)

            if not dir_path.exists():
                return f"Error: Directory not found at {directory}"
            if not dir_path.is_dir():
                return f"Error: {directory} is not a directory"

            def list_sync():
                search_pattern = f"{directory}/{pattern}"
                files = glob.glob(search_pattern)

                if not files:
                    return f"No files found matching {directory}/{pattern}"

                file_list = []
                for file_path in sorted(files):
                    path_obj = Path(file_path)
                    rel_path = str(file_path).replace(str(dir_path) + "/", "")

                    if path_obj.is_dir():
                        file_list.append(f"📁 {rel_path}/")
                    else:
                        file_list.append(f"📄 {rel_path}")

                return "\n".join(file_list)

            return await asyncio.to_thread(list_sync)
        except Exception as e:
            return f"Error listing files in {directory}: {str(e)}"


class FileWriteTool(Tool):
    """Tool for writing and editing files."""

    def __init__(self):
        super().__init__(
            name="file_write",
            description="""
            Write or edit files.

            Operations:
            - write: Create or completely replace a file
            - edit: Make targeted changes to parts of a file
            """,
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["write", "edit"],
                        "description": "File operation to perform",
                    },
                    "path": {
                        "type": "string",
                        "description": "File path to write to or edit",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Text to replace (for edit operation)",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text (for edit operation)",
                    },
                },
                "required": ["operation", "path"],
            },
        )

    async def execute(
        self,
        operation: str,
        path: str,
        content: str = "",
        old_text: str = "",
        new_text: str = "",
    ) -> str:
        """Execute a file write operation.

        Args:
            operation: The operation to perform (write or edit)
            path: The file path
            content: Content to write (for write operation)
            old_text: Text to replace (for edit operation)
            new_text: Replacement text (for edit operation)

        Returns:
            Result of the operation as string
        """
        if operation == "write":
            if not content:
                return "Error: content parameter is required"
            return await self._write_file(path, content)
        elif operation == "edit":
            if not old_text or not new_text:
                return (
                    "Error: both old_text and new_text parameters "
                    "are required for edit operation"
                )
            return await self._edit_file(path, old_text, new_text)
        else:
            return f"Error: Unsupported operation '{operation}'"

    async def _write_file(self, path: str, content: str) -> str:
        """Write content to a file securely."""
        try:
            # Validate path
            error = validate_path(path)
            if error:
                return f"Error: {error}"
                
            file_path = Path(path)
            
            # Create parent dirs if needed
            parent = file_path.parent
            if not parent.exists():
                os.makedirs(parent)
                
            if file_path.exists() and not os.access(file_path, os.W_OK):
                return f"Error: No write permission for {path}"
                
            def write_sync():
                # Write atomically using temporary file
                tmp_path = str(file_path) + '.tmp'
                try:
                    with open(tmp_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    os.replace(tmp_path, file_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                        
                return f"Successfully wrote {len(content)} characters to {path}"
                
            return await asyncio.to_thread(write_sync)
        except Exception as e:
            return f"Error writing to {path}: {str(e)}"

    async def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Make targeted changes to a file."""
        try:
            file_path = Path(path)

            if not file_path.exists():
                return f"Error: File not found at {path}"
            if not file_path.is_file():
                return f"Error: {path} is not a file"

            def edit_sync():
                try:
                    with open(
                        file_path, encoding="utf-8", errors="replace"
                    ) as f:
                        content = f.read()

                    if old_text not in content:
                        return (
                            f"Error: The specified text was not "
                            f"found in {path}"
                        )

                    # Count occurrences to warn about multiple matches
                    count = content.count(old_text)
                    if count > 1:
                        # Edit with warning about multiple occurrences
                        new_content = content.replace(old_text, new_text)
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        return (
                            f"Warning: Found {count} occurrences. "
                            f"All were replaced in {path}"
                        )
                    else:
                        # One occurrence, straightforward replacement
                        new_content = content.replace(old_text, new_text)
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        return f"Successfully edited {path}"
                except UnicodeDecodeError:
                    return f"Error: {path} appears to be a binary file"

            return await asyncio.to_thread(edit_sync)
        except Exception as e:
            return f"Error editing {path}: {str(e)}"
