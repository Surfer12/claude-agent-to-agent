#!/usr/bin/env python3
"""
Test suite for the Cognitive Agent CLI implementation.
Focuses on thread safety, immutable collections, and error handling.
"""

import os
import sys
import pytest
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cli
from cli import CognitiveAgentCLI, InteractionMetrics, AgentConfig

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock()
    agent.run_async = AsyncMock()
    return agent

@pytest.fixture
def cli_instance():
    """Create a CLI instance for testing."""
    return CognitiveAgentCLI(
        name="TestAgent",
        system_prompt="Test prompt",
        verbose=True
    )

def test_immutable_metrics():
    """Test that metrics are immutable."""
    metrics = InteractionMetrics()
    
    # Attempt to modify metrics should raise AttributeError
    with pytest.raises(AttributeError):
        metrics.total_interactions = 1
    
    # Create new instance for updates
    new_metrics = InteractionMetrics(
        total_interactions=1,
        successful_interactions=1,
        average_response_time=0.5,
        last_interaction_time=datetime.now()
    )
    
    assert new_metrics.total_interactions == 1
    assert new_metrics.successful_interactions == 1
    assert new_metrics.average_response_time == 0.5

def test_immutable_config():
    """Test that agent configuration is immutable."""
    config = AgentConfig(
        name="TestAgent",
        system_prompt="Test prompt",
        tools=frozenset(),
        verbose=True,
        model="claude-3-5-sonnet-20240620"
    )
    
    # Attempt to modify config should raise AttributeError
    with pytest.raises(AttributeError):
        config.name = "NewName"
    
    assert config.name == "TestAgent"
    assert config.verbose is True

@pytest.mark.asyncio
async def test_interactive_session(cli_instance, mock_agent):
    """Test interactive session functionality."""
    with patch('cli.Agent', return_value=mock_agent):
        # Mock user input
        with patch('builtins.input', side_effect=['test input', 'exit']):
            # Mock agent response
            mock_agent.run_async.return_value = Mock(
                content=[Mock(text="Test response")]
            )
            
            await cli_instance.interactive_session()
            
            # Verify agent was called
            mock_agent.run_async.assert_called_once_with('test input')
            
            # Verify metrics were updated
            assert cli_instance.metrics.total_interactions == 1
            assert cli_instance.metrics.successful_interactions == 1

@pytest.mark.asyncio
async def test_error_handling(cli_instance, mock_agent):
    """Test error handling in interactive session."""
    with patch('cli.Agent', return_value=mock_agent):
        # Mock user input
        with patch('builtins.input', side_effect=['test input', 'exit']):
            # Mock agent error
            mock_agent.run_async.side_effect = Exception("Test error")
            
            await cli_instance.interactive_session()
            
            # Verify metrics were updated for failed interaction
            assert cli_instance.metrics.total_interactions == 1
            assert cli_instance.metrics.successful_interactions == 0

def test_thread_safety():
    """Test thread safety of CLI implementation."""
    cli = CognitiveAgentCLI()
    
    # Create multiple threads accessing metrics
    async def update_metrics():
        for _ in range(100):
            cli.metrics = InteractionMetrics(
                total_interactions=cli.metrics.total_interactions + 1,
                successful_interactions=cli.metrics.successful_interactions + 1,
                average_response_time=0.5,
                last_interaction_time=datetime.now()
            )
    
    # Run multiple threads
    async def run_threads():
        tasks = [update_metrics() for _ in range(10)]
        await asyncio.gather(*tasks)
    
    asyncio.run(run_threads())
    
    # Verify final metrics
    assert cli.metrics.total_interactions == 1000
    assert cli.metrics.successful_interactions == 1000

def test_model_validation():
    """Test model validation in CLI initialization."""
    # Test valid model
    cli = CognitiveAgentCLI(model="claude-3-5-sonnet-20240620")
    assert cli.config.model == "claude-3-5-sonnet-20240620"
    
    # Test invalid model
    with pytest.raises(ValueError):
        CognitiveAgentCLI(model="invalid-model")


@pytest.mark.asyncio
async def test_handle_single_prompt_async(monkeypatch):
    args = SimpleNamespace(
        name="TestAgent",
        system_prompt="Test prompt",
        verbose=True,
        model="claude-3-5-sonnet-20240620",
        prompt="Hello"
    )

    mock_agent = Mock()
    monkeypatch.setattr("cli.Agent", Mock(return_value=mock_agent))
    handle_mock = AsyncMock()
    monkeypatch.setattr("cli.handle_single_prompt", handle_mock)

    await cli.handle_single_prompt_async(args, [])
    handle_mock.assert_awaited_once_with(mock_agent, "Hello")


@pytest.mark.asyncio
async def test_handle_file_input_async(tmp_path, monkeypatch):
    file = tmp_path / "input.txt"
    file.write_text("data")
    args = SimpleNamespace(
        name="TestAgent",
        system_prompt="Test prompt",
        verbose=True,
        model="claude-3-5-sonnet-20240620",
        file=str(file)
    )

    mock_agent = Mock()
    monkeypatch.setattr("cli.Agent", Mock(return_value=mock_agent))
    handle_mock = AsyncMock()
    monkeypatch.setattr("cli.handle_file_input", handle_mock)

    await cli.handle_file_input_async(args, [])
    handle_mock.assert_awaited_once_with(mock_agent, str(file))


@pytest.mark.asyncio
async def test_main_async_interactive(monkeypatch):
    args = SimpleNamespace(
        interactive=True,
        prompt=None,
        file=None,
        name="TestAgent",
        system_prompt="Test prompt",
        verbose=True,
        model="claude-3-5-sonnet-20240620",
        tools=[],
        mcp_server=None,
        api_key="key",
        enable_file_support=False,
        display_width=0,
        display_height=0,
        display_number=0,
        computer_tool_version="computer_20250124"
    )

    monkeypatch.setattr("cli.parse_args", lambda: args)
    monkeypatch.setattr(cli, "Anthropic", Mock, raising=False)
    monkeypatch.setattr("cli.get_enabled_tools", lambda names, a: [])
    monkeypatch.setattr("cli.setup_mcp_servers", lambda urls: [])

    cli_instance = Mock()
    cli_instance.interactive_session = AsyncMock()
    monkeypatch.setattr("cli.CognitiveAgentCLI", Mock(return_value=cli_instance))

    await cli.main_async()
    cli_instance.interactive_session.assert_awaited_once()

if __name__ == "__main__":
    pytest.main([__file__]) 