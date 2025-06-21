"""Code execution tool for running Python code in a secure sandbox."""

from typing import Any, Dict, Optional

from ..base import Tool


class CodeExecutionTool(Tool):
    """Code execution tool that allows Claude to run Python code in a secure sandbox."""
    
    def __init__(self):
        """Initialize the code execution tool."""
        super().__init__(
            name="code_execution",
            description="Execute Python code in a secure, sandboxed environment. Can analyze data, create visualizations, perform calculations, and process files.",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute in the sandbox environment"
                    }
                },
                "required": ["code"]
            }
        )
        self.tool_type = "code_execution_20250522"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to Claude API format for code execution."""
        return {
            "type": self.tool_type,
            "name": self.name
        }
    
    async def execute(self, **kwargs) -> str:
        """
        Code execution is handled server-side by Claude's API.
        This method should not be called directly as the tool execution
        happens within Claude's secure sandbox environment.
        """
        return "Code execution is handled server-side by Claude's API. This tool should not be executed locally."


class CodeExecutionWithFilesTool(CodeExecutionTool):
    """Code execution tool with file upload support."""
    
    def __init__(self):
        """Initialize the code execution tool with file support."""
        super().__init__()
        self.description = "Execute Python code in a secure sandbox with access to uploaded files. Can analyze CSV, Excel, JSON, images, and other file formats."
        self.supports_files = True
    
    def get_beta_headers(self) -> list[str]:
        """Get required beta headers for code execution with files."""
        return ["code-execution-2025-05-22", "files-api-2025-04-14"]


def get_supported_models() -> list[str]:
    """Get list of models that support code execution."""
    return [
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514", 
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-latest"
    ]


def is_model_supported(model: str) -> bool:
    """Check if a model supports code execution."""
    return model in get_supported_models()
