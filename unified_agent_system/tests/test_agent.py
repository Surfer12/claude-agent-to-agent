import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from ..core import UnifiedAgent
from ..core.types import AgentConfig, ProviderType
from ..tool_registry import ToolRegistry

@pytest.fixture
def mock_config():
    return AgentConfig(
        provider=ProviderType.CLAUDE,
        model="claude-3-5-sonnet-20240620",
        api_key="test_key",
        enable_tools=True,
        enable_code_execution=True,
        enable_computer_use=True
    )

@pytest.fixture
def unified_agent(mock_config):
    return UnifiedAgent(mock_config)

def test_initialization(unified_agent):
    assert unified_agent.config.provider == ProviderType.CLAUDE
    assert isinstance(unified_agent.tool_registry, ToolRegistry)
    assert len(unified_agent.message_history) == 0

def test_setup_tools(unified_agent):
    assert len(unified_agent.tool_registry.active_tools) > 0
    tool_names = unified_agent.tool_registry.list_tools()
    assert "code_execution" in tool_names
    assert "computer" in tool_names

@pytest.mark.asyncio
async def test_run_async(unified_agent):
    with patch.object(unified_agent.provider, 'generate_response', new_callable=AsyncMock) as mock_response:
        mock_response.return_value = {"content": [{"type": "text", "text": "Test response"}]}
        
        response = await unified_agent.run_async("Test input")
        
        assert "content" in response
        assert len(unified_agent.message_history) == 2  # user + assistant
        assert unified_agent.message_history[0]["role"] == "user"
        assert unified_agent.message_history[1]["role"] == "assistant"

@pytest.mark.asyncio
async def test_handle_tool_calls(unified_agent):
    tool_calls = [{"name": "code_execution", "input": {"code": "print('test')"}, "id": "test_id"}]
    
    with patch.object(unified_agent.tool_registry, 'execute_tool', new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "Code executed successfully"
        
        await unified_agent._handle_tool_calls(tool_calls)
        
        assert len(unified_agent.message_history) == 1
        assert unified_agent.message_history[0]["role"] == "tool"
        assert unified_agent.message_history[0]["tool_call_id"] == "test_id"

def test_reset(unified_agent):
    unified_agent.message_history = [{"role": "user", "content": "test"}]
    unified_agent.reset()
    assert len(unified_agent.message_history) == 0

def test_get_history(unified_agent):
    unified_agent.message_history = [{"role": "user", "content": "test"}]
    history = unified_agent.get_history()
    assert len(history) == 1
    assert history[0] == {"role": "user", "content": "test"} 