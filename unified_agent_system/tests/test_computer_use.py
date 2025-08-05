import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from ..computer_use import ComputerUseInterface, ComputerUseAgent
from ..core import UnifiedAgent
from ..core.types import AgentConfig

@pytest.fixture
def mock_agent():
    config = AgentConfig()
    return UnifiedAgent(config)

@pytest.fixture
def computer_interface(mock_agent):
    return ComputerUseInterface(mock_agent)

@pytest.mark.asyncio
async def test_initialize(computer_interface):
    await computer_interface.initialize()
    assert computer_interface.initialized
    assert computer_interface.computer["initialized"]

@pytest.mark.asyncio
async def test_navigate_to(computer_interface):
    await computer_interface.initialize()
    await computer_interface.navigate_to("https://example.com")
    assert computer_interface.computer["current_url"] == "https://example.com"

@pytest.mark.asyncio
async def test_take_screenshot(computer_interface):
    await computer_interface.initialize()
    result = await computer_interface.take_screenshot()
    assert result == "screenshot_placeholder.png"

@pytest.mark.asyncio
async def test_execute_action(computer_interface):
    await computer_interface.initialize()
    result = await computer_interface.execute_action("navigate", {"url": "test.com"})
    assert "Navigated to test.com" in result

    result = await computer_interface.execute_action("wait", {"seconds": 0.1})
    assert "Waited 0.1 seconds" in result

@pytest.mark.asyncio
async def test_cleanup(computer_interface):
    await computer_interface.initialize()
    await computer_interface.cleanup()
    assert not computer_interface.initialized
    assert computer_interface.computer is None

@pytest.mark.asyncio
async def test_computer_use_agent(mock_agent):
    config = AgentConfig(enable_computer_use=True)
    agent = ComputerUseAgent(config)
    agent.agent = mock_agent
    
    with patch.object(mock_agent, 'run_async', new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {"content": [{"type": "text", "text": "Test"}]}
        
        response = await agent.run("test input")
        assert "content" in response 