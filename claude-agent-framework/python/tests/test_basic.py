"""Basic tests for the Claude Agent Framework."""

import pytest
from claude_agent import Agent, AgentConfig, get_available_tools, get_tool


def test_import():
    """Test that basic imports work."""
    assert Agent is not None
    assert AgentConfig is not None


def test_agent_config():
    """Test agent configuration."""
    config = AgentConfig()
    assert config.name == "claude-agent"
    assert config.model_config.model == "claude-sonnet-4-20250514"
    assert config.model_config.max_tokens == 4096


def test_available_tools():
    """Test tool discovery."""
    tools = get_available_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
    
    # Check that basic tools are available
    expected_tools = ["think", "file_read", "file_write", "computer", "code_execution"]
    for tool_name in expected_tools:
        assert tool_name in tools, f"Tool {tool_name} not found in {tools}"


def test_get_tool():
    """Test getting tool instances."""
    # Test think tool
    think_tool = get_tool("think")
    assert think_tool.name == "think"
    assert "reasoning" in think_tool.description.lower()
    
    # Test file_read tool
    file_read_tool = get_tool("file_read")
    assert file_read_tool.name == "file_read"
    assert "read" in file_read_tool.description.lower()


def test_agent_creation():
    """Test agent creation."""
    config = AgentConfig(api_key="test-key")
    agent = Agent(config=config)
    
    assert agent.name == "claude-agent"
    assert agent.config.api_key == "test-key"
    assert isinstance(agent.tools, list)


def test_agent_with_tools():
    """Test agent creation with specific tools."""
    config = AgentConfig(api_key="test-key")
    think_tool = get_tool("think")
    
    agent = Agent(config=config, tools=[think_tool])
    
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "think"


if __name__ == "__main__":
    pytest.main([__file__])
