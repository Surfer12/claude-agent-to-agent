"""Tests for tool_util.py - leaf node utility functions."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from agents.utils.tool_util import execute_tools, _execute_single_tool


class MockTool:
    """Mock tool for testing."""
    
    def __init__(self, name, result="success", should_error=False):
        self.name = name
        self.result = result
        self.should_error = should_error
    
    async def execute(self, **kwargs):
        if self.should_error:
            raise ValueError(f"Mock error from {self.name}")
        return self.result


class MockToolCall:
    """Mock tool call for testing."""
    
    def __init__(self, id, name, input_data=None):
        self.id = id
        self.name = name
        self.input = input_data or {}


class TestExecuteSingleTool:
    """Test suite for _execute_single_tool function."""
    
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful tool execution."""
        tool = MockTool("test_tool", "test_result")
        call = MockToolCall("call_1", "test_tool", {"param": "value"})
        tool_dict = {"test_tool": tool}
        
        result = await _execute_single_tool(call, tool_dict)
        
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "call_1"
        assert result["content"] == "test_result"
        assert "is_error" not in result
    
    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        """Test handling of missing tool."""
        call = MockToolCall("call_1", "missing_tool")
        tool_dict = {}
        
        result = await _execute_single_tool(call, tool_dict)
        
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "call_1"
        assert "not found" in result["content"]
        assert result["is_error"] is True
    
    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """Test handling of tool execution error."""
        tool = MockTool("error_tool", should_error=True)
        call = MockToolCall("call_1", "error_tool")
        tool_dict = {"error_tool": tool}
        
        result = await _execute_single_tool(call, tool_dict)
        
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "call_1"
        assert "Error executing tool" in result["content"]
        assert "Mock error from error_tool" in result["content"]
        assert result["is_error"] is True
    
    @pytest.mark.asyncio
    async def test_with_complex_input(self):
        """Test execution with complex input parameters."""
        tool = MockTool("complex_tool", "complex_result")
        complex_input = {
            "text": "Hello world",
            "numbers": [1, 2, 3],
            "nested": {"key": "value"}
        }
        call = MockToolCall("call_1", "complex_tool", complex_input)
        tool_dict = {"complex_tool": tool}
        
        result = await _execute_single_tool(call, tool_dict)
        
        assert result["content"] == "complex_result"
        assert "is_error" not in result


class TestExecuteTools:
    """Test suite for execute_tools function."""
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel execution of multiple tools."""
        tools = {
            "tool1": MockTool("tool1", "result1"),
            "tool2": MockTool("tool2", "result2"),
            "tool3": MockTool("tool3", "result3")
        }
        calls = [
            MockToolCall("call_1", "tool1"),
            MockToolCall("call_2", "tool2"),
            MockToolCall("call_3", "tool3")
        ]
        
        results = await execute_tools(calls, tools, parallel=True)
        
        assert len(results) == 3
        assert all(r["type"] == "tool_result" for r in results)
        
        # Results might be in different order due to parallel execution
        contents = [r["content"] for r in results]
        assert "result1" in contents
        assert "result2" in contents
        assert "result3" in contents
    
    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Test sequential execution of multiple tools."""
        tools = {
            "tool1": MockTool("tool1", "result1"),
            "tool2": MockTool("tool2", "result2")
        }
        calls = [
            MockToolCall("call_1", "tool1"),
            MockToolCall("call_2", "tool2")
        ]
        
        results = await execute_tools(calls, tools, parallel=False)
        
        assert len(results) == 2
        assert results[0]["content"] == "result1"
        assert results[1]["content"] == "result2"
        assert results[0]["tool_use_id"] == "call_1"
        assert results[1]["tool_use_id"] == "call_2"
    
    @pytest.mark.asyncio
    async def test_mixed_success_and_error(self):
        """Test execution with mix of successful and failing tools."""
        tools = {
            "good_tool": MockTool("good_tool", "success"),
            "bad_tool": MockTool("bad_tool", should_error=True)
        }
        calls = [
            MockToolCall("call_1", "good_tool"),
            MockToolCall("call_2", "bad_tool"),
            MockToolCall("call_3", "missing_tool")
        ]
        
        results = await execute_tools(calls, tools, parallel=True)
        
        assert len(results) == 3
        
        # Find results by tool_use_id
        good_result = next(r for r in results if r["tool_use_id"] == "call_1")
        bad_result = next(r for r in results if r["tool_use_id"] == "call_2")
        missing_result = next(r for r in results if r["tool_use_id"] == "call_3")
        
        assert good_result["content"] == "success"
        assert "is_error" not in good_result
        
        assert "Error executing tool" in bad_result["content"]
        assert bad_result["is_error"] is True
        
        assert "not found" in missing_result["content"]
        assert missing_result["is_error"] is True
    
    @pytest.mark.asyncio
    async def test_empty_tool_calls(self):
        """Test execution with empty tool calls list."""
        results = await execute_tools([], {}, parallel=True)
        assert results == []
        
        results = await execute_tools([], {}, parallel=False)
        assert results == []
    
    @pytest.mark.asyncio
    async def test_performance_comparison(self):
        """Test that parallel execution is faster than sequential."""
        import time
        
        # Create slow tools
        class SlowTool:
            def __init__(self, name, delay=0.1):
                self.name = name
                self.delay = delay
            
            async def execute(self, **kwargs):
                await asyncio.sleep(self.delay)
                return f"result_{self.name}"
        
        tools = {f"tool{i}": SlowTool(f"tool{i}") for i in range(5)}
        calls = [MockToolCall(f"call_{i}", f"tool{i}") for i in range(5)]
        
        # Test parallel execution
        start_time = time.time()
        await execute_tools(calls, tools, parallel=True)
        parallel_time = time.time() - start_time
        
        # Test sequential execution
        start_time = time.time()
        await execute_tools(calls, tools, parallel=False)
        sequential_time = time.time() - start_time
        
        # Parallel should be significantly faster
        assert parallel_time < sequential_time * 0.8  # At least 20% faster


class TestToolUtilEdgeCases:
    """Edge case tests for tool utilities."""
    
    @pytest.mark.asyncio
    async def test_tool_returning_none(self):
        """Test tool that returns None."""
        class NoneTool:
            async def execute(self, **kwargs):
                return None
        
        tool = NoneTool()
        call = MockToolCall("call_1", "none_tool")
        tool_dict = {"none_tool": tool}
        
        result = await _execute_single_tool(call, tool_dict)
        assert result["content"] == "None"
    
    @pytest.mark.asyncio
    async def test_tool_returning_complex_object(self):
        """Test tool that returns complex object."""
        class ComplexTool:
            async def execute(self, **kwargs):
                return {"status": "success", "data": [1, 2, 3]}
        
        tool = ComplexTool()
        call = MockToolCall("call_1", "complex_tool")
        tool_dict = {"complex_tool": tool}
        
        result = await _execute_single_tool(call, tool_dict)
        assert "status" in result["content"]
        assert "success" in result["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
