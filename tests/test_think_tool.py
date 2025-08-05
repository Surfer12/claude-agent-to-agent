"""Tests for ThinkTool - a leaf node with minimal dependencies."""

import pytest
import asyncio
from agents.tools.think import ThinkTool


class TestThinkTool:
    """Test suite for ThinkTool leaf node."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.think_tool = ThinkTool()
    
    def test_initialization(self):
        """Test ThinkTool initialization."""
        assert self.think_tool.name == "think"
        assert "internal reasoning" in self.think_tool.description.lower()
        assert "thought" in self.think_tool.input_schema["properties"]
        assert self.think_tool.input_schema["required"] == ["thought"]
    
    def test_input_schema_structure(self):
        """Test input schema is properly structured."""
        schema = self.think_tool.input_schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "thought" in schema["properties"]
        assert schema["properties"]["thought"]["type"] == "string"
        assert "description" in schema["properties"]["thought"]
    
    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic execution of think tool."""
        result = await self.think_tool.execute(thought="Test thought")
        assert result == "Thinking complete!"
    
    @pytest.mark.asyncio
    async def test_execute_with_complex_thought(self):
        """Test execution with complex thought."""
        complex_thought = "This is a complex thought with multiple sentences. It includes reasoning about various topics."
        result = await self.think_tool.execute(thought=complex_thought)
        assert result == "Thinking complete!"
    
    @pytest.mark.asyncio
    async def test_execute_with_empty_thought(self):
        """Test execution with empty thought."""
        result = await self.think_tool.execute(thought="")
        assert result == "Thinking complete!"
    
    @pytest.mark.asyncio
    async def test_execute_with_unicode_thought(self):
        """Test execution with unicode characters."""
        unicode_thought = "思考中... 🤔 Thinking about émotions and café"
        result = await self.think_tool.execute(thought=unicode_thought)
        assert result == "Thinking complete!"
    
    def test_to_dict_format(self):
        """Test tool dictionary format for API compatibility."""
        tool_dict = self.think_tool.to_dict()
        assert tool_dict["name"] == "think"
        assert tool_dict["description"] == self.think_tool.description
        assert tool_dict["input_schema"] == self.think_tool.input_schema
    
    def test_inheritance_structure(self):
        """Test that ThinkTool properly inherits from Tool."""
        from agents.tools.base import Tool
        assert isinstance(self.think_tool, Tool)
        assert hasattr(self.think_tool, 'execute')
        assert hasattr(self.think_tool, 'to_dict')


# Performance and edge case tests
class TestThinkToolEdgeCases:
    """Edge case tests for ThinkTool."""
    
    def setup_method(self):
        self.think_tool = ThinkTool()
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test concurrent execution of think tool."""
        thoughts = [f"Thought {i}" for i in range(10)]
        tasks = [self.think_tool.execute(thought=thought) for thought in thoughts]
        results = await asyncio.gather(*tasks)
        assert all(result == "Thinking complete!" for result in results)
    
    @pytest.mark.asyncio
    async def test_large_thought_input(self):
        """Test with very large thought input."""
        large_thought = "A" * 10000  # 10KB of text
        result = await self.think_tool.execute(thought=large_thought)
        assert result == "Thinking complete!"
    
    def test_schema_validation_requirements(self):
        """Test that schema properly validates required fields."""
        schema = self.think_tool.input_schema
        assert "required" in schema
        assert "thought" in schema["required"]
        assert len(schema["required"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
