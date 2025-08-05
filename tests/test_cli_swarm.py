import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock
import asyncio
from unified_agent.cli import CLIInterface

@pytest.mark.asyncio
async def test_run_swarm_interactive():
    """Test the run_swarm_interactive method of the CLIInterface."""
    cli = CLIInterface()
    args = MagicMock()
    args.swarm_config = "swarm/examples/airline/configs/agents.py"
    args.initial_agent = "triage_agent"

    with patch('builtins.input', side_effect=['hello', 'exit']), \
         patch('importlib.util.spec_from_file_location'), \
         patch('importlib.util.module_from_spec'), \
         patch('swarm.Swarm') as mock_swarm, \
         patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):

        # Mock the swarm instance and its run method
        mock_swarm_instance = mock_swarm.return_value
        mock_swarm_instance.run.return_value = MagicMock(messages=[{"role": "assistant", "content": "Test response"}])

        await cli.run_swarm_interactive(args)

        # Assert that the swarm was initialized and run
        mock_swarm.assert_called_once()
        mock_swarm_instance.run.assert_called_once()
