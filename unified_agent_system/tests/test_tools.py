import pytest
import asyncio
from unittest.mock import AsyncMock

from ..tools.base import BaseTool
from ..tools.code_execution import CodeExecutionTool
from ..tools.computer_use import ComputerUseTool
from ..tools.file_tools import FileTools, ReadFileTool, WriteFileTool
from ..tool_registry import ToolRegistry

@pytest.fixture
def tool_registry():
    return ToolRegistry()

def test_base_tool():
    class TestTool(BaseTool):
        async def execute(self, input_data):
            return "executed"

        def get_input_schema(self):
            return {"type": "object"}

    tool = TestTool(name="test", description="test desc")
    assert tool.name == "test"
    assert tool.to_dict() == {
        "name": "test",
        "description": "test desc",
        "input_schema": {"type": "object"}
    }
    assert tool.to_openai_dict()["function"]["name"] == "test"

@pytest.mark.asyncio
async def test_code_execution_tool():
    tool = CodeExecutionTool()
    result = await tool.execute({"code": "print('hello')"})
    assert "Code executed successfully" in result

@pytest.mark.asyncio
async def test_computer_use_tool():
    tool = ComputerUseTool()
    result = await tool.execute({"action": "navigate", "params": {"url": "https://example.com"}})
    assert "Executed navigate" in result

def test_file_tools():
    file_tools = FileTools()
    tools = file_tools.get_tools()
    assert len(tools) == 4
    assert all(isinstance(t, BaseTool) for t in tools)
    assert {t.name for t in tools} == {"read_file", "write_file", "list_directory", "delete_file"}

@pytest.mark.asyncio
async def test_read_file_tool(tmp_path):
    tool = ReadFileTool()
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    result = await tool.execute({"file_path": str(test_file)})
    assert result == "hello"

@pytest.mark.asyncio
async def test_write_file_tool(tmp_path):
    tool = WriteFileTool()
    test_file = tmp_path / "test.txt"
    result = await tool.execute({"file_path": str(test_file), "content": "hello"})
    assert "Successfully wrote" in result
    assert test_file.read_text() == "hello"

def test_tool_registry(tool_registry):
    tool = CodeExecutionTool()
    tool_registry.register_tool(tool)
    assert tool_registry.get_tool("code_execution") == tool
    assert "code_execution" in tool_registry.list_tools()

    tool_registry.unregister_tool("code_execution")
    assert tool_registry.get_tool("code_execution") is None

def test_register_tools(tool_registry):
    tool_registry.register_code_execution_tools()
    assert "code_execution" in tool_registry.list_tools()

    tool_registry.register_computer_use_tools()
    assert "computer" in tool_registry.list_tools()

    tool_registry.register_file_tools()
    assert set(["read_file", "write_file", "list_directory", "delete_file"]).issubset(set(tool_registry.list_tools()))

@pytest.mark.asyncio
async def test_execute_tool(tool_registry):
    tool_registry.register_code_execution_tools()
    result = await tool_registry.execute_tool("code_execution", {"code": "print('test')"})
    assert "executed successfully" in result 