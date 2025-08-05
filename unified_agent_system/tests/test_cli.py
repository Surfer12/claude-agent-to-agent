import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from ..cli import CLIInterface
from ..config import ProviderType

@pytest.fixture
def cli_interface():
    return CLIInterface()

def test_create_parser(cli_interface):
    parser = cli_interface.create_parser()
    args = parser.parse_args([])
    assert args.provider == "claude"
    assert not args.enable_computer_use
    assert not args.interactive

@pytest.mark.asyncio
@patch('unified_agent_system.cli.UnifiedAgent')
async def test_run_single_input(mock_agent, cli_interface):
    cli_interface.agent = mock_agent.return_value
    cli_interface.agent.run_async = AsyncMock(return_value={"content": [{"type": "text", "text": "Test"}]})

    response = await cli_interface.run_single_input("test input")
    assert "content" in response
    cli_interface.agent.run_async.assert_called_with("test input")

@pytest.mark.asyncio
@patch('unified_agent_system.cli.UnifiedAgent')
@patch('builtins.input', side_effect=['test', 'exit'])
async def test_run_interactive(mock_input, mock_agent, cli_interface):
    cli_interface.agent = mock_agent.return_value
    cli_interface.agent.run_async = AsyncMock(return_value={"content": [{"type": "text", "text": "Test response"}]})

    await cli_interface.run_interactive()
    cli_interface.agent.run_async.assert_called_with('test')

@pytest.mark.asyncio
@patch('unified_agent_system.cli.Swarm')
@patch('builtins.input', side_effect=['test', 'exit'])
@patch('importlib.util.spec_from_file_location')
async def test_run_swarm_interactive(mock_spec, mock_input, mock_swarm, cli_interface):
    args = MagicMock()
    args.swarm_config = "test_config.py"
    args.initial_agent = "test_agent"

    mock_module = MagicMock()
    mock_module.test_agent = "mock_agent"
    mock_spec.return_value = MagicMock()
    mock_spec.return_value.loader.exec_module.return_value = mock_module

    mock_client = mock_swarm.return_value
    mock_client.run.return_value = MagicMock(messages=[{"content": "Swarm response"}])

    await cli_interface.run_swarm_interactive(args)
    mock_client.run.assert_called()

def test_create_config(cli_interface):
    class Args:
        provider = "openai"
        model = None
        api_key = "test"
        system_prompt = "test prompt"
        max_tokens = 100
        temperature = 0.5
        verbose = True
        enable_tools = True
        enable_code_execution = True
        enable_computer_use = True
        computer_type = "test_type"
        start_url = "test.com"
        show_images = True
        debug = True

    config = cli_interface.create_config(Args())
    assert config.provider == ProviderType.OPENAI
    assert config.model == "gpt-4o"
    assert config.api_key == "test"
    assert config.enable_tools
    assert config.enable_computer_use 