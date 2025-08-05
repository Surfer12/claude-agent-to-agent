#!/usr/bin/env python3
"""
Comprehensive test suite for CLI leaf nodes.

This module tests the core CLI components identified as leaf nodes:
- Argument parsing functions
- Configuration creation
- Environment validation
- CLI interface components
"""

import pytest
import asyncio
import os
import sys
import tempfile
import subprocess
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CLI components
try:
    from unified_agent.cli import CLIInterface
    from unified_agent.core import AgentConfig, ProviderType
    UNIFIED_AGENT_AVAILABLE = True
except ImportError:
    UNIFIED_AGENT_AVAILABLE = False

try:
    from quick_start import (
        check_python_version,
        check_dependencies,
        setup_environment,
        show_next_steps
    )
    QUICK_START_AVAILABLE = True
except ImportError:
    QUICK_START_AVAILABLE = False


class TestCLIArgumentParsing:
    """Test CLI argument parsing leaf node functionality."""
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    def test_create_parser_basic(self):
        """Test basic parser creation."""
        cli = CLIInterface()
        parser = cli.create_parser()
        
        assert parser is not None
        assert parser.prog is not None
        assert "Unified Agent System" in parser.description
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    def test_create_parser_arguments(self):
        """Test parser has required arguments."""
        cli = CLIInterface()
        parser = cli.create_parser()
        
        # Parse help to check arguments exist
        help_text = parser.format_help()
        
        # Check key arguments are present
        assert "--provider" in help_text
        assert "--model" in help_text
        assert "--interactive" in help_text
        assert "--enable-tools" in help_text
        assert "--enable-computer-use" in help_text
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    def test_parse_claude_arguments(self):
        """Test parsing Claude-specific arguments."""
        cli = CLIInterface()
        parser = cli.create_parser()
        
        args = parser.parse_args([
            "--provider", "claude",
            "--model", "claude-3-5-sonnet-20241022",
            "--interactive"
        ])
        
        assert args.provider == "claude"
        assert args.model == "claude-3-5-sonnet-20241022"
        assert args.interactive is True
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    def test_parse_openai_arguments(self):
        """Test parsing OpenAI-specific arguments."""
        cli = CLIInterface()
        parser = cli.create_parser()
        
        args = parser.parse_args([
            "--provider", "openai",
            "--model", "gpt-4o",
            "--enable-code-execution"
        ])
        
        assert args.provider == "openai"
        assert args.model == "gpt-4o"
        assert args.enable_code_execution is True
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    def test_parse_computer_use_arguments(self):
        """Test parsing computer use arguments."""
        cli = CLIInterface()
        parser = cli.create_parser()
        
        args = parser.parse_args([
            "--enable-computer-use",
            "--computer-type", "local-playwright",
            "--start-url", "https://example.com",
            "--show-images"
        ])
        
        assert args.enable_computer_use is True
        assert args.computer_type == "local-playwright"
        assert args.start_url == "https://example.com"
        assert args.show_images is True


class TestCLIConfiguration:
    """Test CLI configuration creation leaf node functionality."""
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    def test_create_config_claude_defaults(self):
        """Test creating Claude configuration with defaults."""
        cli = CLIInterface()
        
        # Mock arguments
        args = Mock()
        args.provider = "claude"
        args.model = None  # Test default model selection
        args.api_key = None
        args.max_tokens = 4096
        args.temperature = 1.0
        args.system_prompt = "Test prompt"
        args.verbose = False
        args.enable_tools = False
        args.enable_code_execution = False
        args.enable_computer_use = False
        args.computer_type = "local-playwright"
        args.start_url = "https://bing.com"
        args.show_images = False
        args.debug = False
        
        config = cli.create_config(args)
        
        assert config.provider == ProviderType.CLAUDE
        assert config.model == "claude-3-5-sonnet-20241022"  # Default Claude model
        assert config.max_tokens == 4096
        assert config.temperature == 1.0
        assert config.system_prompt == "Test prompt"
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    def test_create_config_openai_defaults(self):
        """Test creating OpenAI configuration with defaults."""
        cli = CLIInterface()
        
        # Mock arguments
        args = Mock()
        args.provider = "openai"
        args.model = None  # Test default model selection
        args.api_key = "test-key"
        args.max_tokens = 2048
        args.temperature = 0.7
        args.system_prompt = "Test prompt"
        args.verbose = True
        args.enable_tools = True
        args.enable_code_execution = False
        args.enable_computer_use = False
        args.computer_type = "local-playwright"
        args.start_url = "https://bing.com"
        args.show_images = False
        args.debug = False
        
        config = cli.create_config(args)
        
        assert config.provider == ProviderType.OPENAI
        assert config.model == "gpt-4o"  # Default OpenAI model
        assert config.api_key == "test-key"
        assert config.max_tokens == 2048
        assert config.temperature == 0.7
        assert config.verbose is True
        assert config.enable_tools is True
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    def test_create_config_computer_use_enabled(self):
        """Test configuration with computer use enabled."""
        cli = CLIInterface()
        
        # Mock arguments
        args = Mock()
        args.provider = "claude"
        args.model = "claude-3-5-sonnet-20241022"
        args.api_key = None
        args.max_tokens = 4096
        args.temperature = 1.0
        args.system_prompt = "Test prompt"
        args.verbose = False
        args.enable_tools = False
        args.enable_code_execution = False
        args.enable_computer_use = True
        args.computer_type = "browserbase"
        args.start_url = "https://google.com"
        args.show_images = True
        args.debug = True
        
        config = cli.create_config(args)
        
        assert config.enable_computer_use is True
        assert config.enable_tools is True  # Should be enabled when computer use is enabled
        assert config.computer_type == "browserbase"
        assert config.start_url == "https://google.com"
        assert config.show_images is True
        assert config.debug is True


class TestEnvironmentValidation:
    """Test environment validation leaf node functionality."""
    
    @pytest.mark.skipif(not QUICK_START_AVAILABLE, reason="quick_start not available")
    def test_check_python_version_success(self):
        """Test Python version check with compatible version."""
        # This should always pass since we're running the test
        result = check_python_version()
        assert result is True
    
    @pytest.mark.skipif(not QUICK_START_AVAILABLE, reason="quick_start not available")
    @patch('sys.version_info', (3, 7, 0))
    def test_check_python_version_failure(self):
        """Test Python version check with incompatible version."""
        result = check_python_version()
        assert result is False
    
    @pytest.mark.skipif(not QUICK_START_AVAILABLE, reason="quick_start not available")
    @patch('builtins.__import__')
    def test_check_dependencies_all_available(self, mock_import):
        """Test dependency check when all packages are available."""
        # Mock successful imports
        mock_import.return_value = Mock()
        
        result = check_dependencies()
        assert result is True
    
    @pytest.mark.skipif(not QUICK_START_AVAILABLE, reason="quick_start not available")
    @patch('builtins.__import__')
    @patch('subprocess.check_call')
    def test_check_dependencies_missing_packages(self, mock_subprocess, mock_import):
        """Test dependency check with missing packages."""
        # Mock ImportError for some packages
        def side_effect(name):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return Mock()
        
        mock_import.side_effect = side_effect
        mock_subprocess.return_value = None
        
        result = check_dependencies()
        assert result is True  # Should succeed after installation
        mock_subprocess.assert_called_once()
    
    @pytest.mark.skipif(not QUICK_START_AVAILABLE, reason="quick_start not available")
    @patch('os.environ.get')
    def test_setup_environment_with_keys(self, mock_env_get):
        """Test environment setup when API keys are present."""
        # Mock environment variables
        def env_side_effect(key):
            if key == "ANTHROPIC_API_KEY":
                return "test-claude-key"
            elif key == "OPENAI_API_KEY":
                return "test-openai-key"
            return None
        
        mock_env_get.side_effect = env_side_effect
        
        result = setup_environment()
        assert result is True
    
    @pytest.mark.skipif(not QUICK_START_AVAILABLE, reason="quick_start not available")
    @patch('os.environ.get')
    @patch('pathlib.Path.exists')
    def test_setup_environment_no_keys(self, mock_exists, mock_env_get):
        """Test environment setup when no API keys are present."""
        # Mock no environment variables
        mock_env_get.return_value = None
        mock_exists.return_value = False
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = Mock()
            
            result = setup_environment()
            assert result is False
            mock_open.assert_called_once()
    
    @pytest.mark.skipif(not QUICK_START_AVAILABLE, reason="quick_start not available")
    def test_show_next_steps(self, capsys):
        """Test show_next_steps function output."""
        show_next_steps()
        
        captured = capsys.readouterr()
        assert "Next Steps:" in captured.out
        assert "Set up your API keys" in captured.out
        assert "Documentation:" in captured.out


class TestCLIInterfaceComponents:
    """Test CLI interface component leaf nodes."""
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    def test_cli_interface_initialization(self):
        """Test CLI interface initialization."""
        cli = CLIInterface()
        assert cli.agent is None
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    @patch('unified_agent.cli.UnifiedAgent')
    async def test_run_single_input_success(self, mock_agent_class):
        """Test successful single input processing."""
        # Mock agent and response
        mock_agent = Mock()
        mock_response = {
            "content": [
                {"type": "text", "text": "Test response"}
            ]
        }
        mock_agent.run_async.return_value = mock_response
        mock_agent_class.return_value = mock_agent
        
        cli = CLIInterface()
        cli.agent = mock_agent
        
        result = await cli.run_single_input("Test input")
        
        assert result == mock_response
        mock_agent.run_async.assert_called_once_with("Test input")
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    async def test_run_single_input_no_agent(self):
        """Test single input processing without initialized agent."""
        cli = CLIInterface()
        
        with pytest.raises(RuntimeError, match="Agent not initialized"):
            await cli.run_single_input("Test input")
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    @patch('unified_agent.cli.UnifiedAgent')
    async def test_run_single_input_with_file_output(self, mock_agent_class):
        """Test single input processing with file output."""
        # Mock agent and response
        mock_agent = Mock()
        mock_response = {
            "content": [
                {"type": "text", "text": "Test response"}
            ]
        }
        mock_agent.run_async.return_value = mock_response
        mock_agent_class.return_value = mock_agent
        
        cli = CLIInterface()
        cli.agent = mock_agent
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            result = await cli.run_single_input("Test input", temp_path)
            
            # Check file was written
            with open(temp_path, 'r') as f:
                content = f.read()
            assert content == "Test response"
            
        finally:
            os.unlink(temp_path)


class TestCLIErrorHandling:
    """Test CLI error handling in leaf nodes."""
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    def test_invalid_provider_argument(self):
        """Test handling of invalid provider argument."""
        cli = CLIInterface()
        parser = cli.create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(["--provider", "invalid-provider"])
    
    @pytest.mark.skipif(not UNIFIED_AGENT_AVAILABLE, reason="unified_agent not available")
    def test_invalid_temperature_argument(self):
        """Test handling of invalid temperature argument."""
        cli = CLIInterface()
        parser = cli.create_parser()
        
        # This should not raise an error (argparse handles type conversion)
        args = parser.parse_args(["--temperature", "1.5"])
        assert args.temperature == 1.5
    
    @pytest.mark.skipif(not QUICK_START_AVAILABLE, reason="quick_start not available")
    @patch('subprocess.check_call')
    def test_check_dependencies_installation_failure(self, mock_subprocess):
        """Test dependency check when installation fails."""
        # Mock ImportError and subprocess failure
        with patch('builtins.__import__') as mock_import:
            mock_import.side_effect = ImportError("No module found")
            mock_subprocess.side_effect = subprocess.CalledProcessError(1, "pip")
            
            result = check_dependencies()
            assert result is False


# Integration test markers
pytestmark = pytest.mark.asyncio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
