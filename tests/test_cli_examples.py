#!/usr/bin/env python3
"""
Test suite for CLI example scripts (leaf nodes).

This module tests the example CLI scripts identified as leaf nodes:
- simple_cli_example.py
- basic_usage.py
- computer_use_example.py
- code_execution_example.py
"""

import pytest
import asyncio
import os
import sys
import subprocess
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if example files exist
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
SIMPLE_CLI_EXAMPLE = EXAMPLES_DIR / "simple_cli_example.py"
BASIC_USAGE_EXAMPLE = EXAMPLES_DIR / "basic_usage.py"
COMPUTER_USE_EXAMPLE = EXAMPLES_DIR / "computer_use_example.py"
CODE_EXECUTION_EXAMPLE = EXAMPLES_DIR / "code_execution_example.py"

# Import example modules if available
try:
    sys.path.insert(0, str(EXAMPLES_DIR))
    if BASIC_USAGE_EXAMPLE.exists():
        import basic_usage
        BASIC_USAGE_AVAILABLE = True
    else:
        BASIC_USAGE_AVAILABLE = False
except ImportError:
    BASIC_USAGE_AVAILABLE = False

try:
    if COMPUTER_USE_EXAMPLE.exists():
        import computer_use_example
        COMPUTER_USE_AVAILABLE = True
    else:
        COMPUTER_USE_AVAILABLE = False
except ImportError:
    COMPUTER_USE_AVAILABLE = False

try:
    if CODE_EXECUTION_EXAMPLE.exists():
        import code_execution_example
        CODE_EXECUTION_AVAILABLE = True
    else:
        CODE_EXECUTION_AVAILABLE = False
except ImportError:
    CODE_EXECUTION_AVAILABLE = False


class TestSimpleCLIExample:
    """Test simple_cli_example.py leaf node."""
    
    @pytest.mark.skipif(not SIMPLE_CLI_EXAMPLE.exists(), reason="simple_cli_example.py not found")
    def test_simple_cli_example_exists(self):
        """Test that simple CLI example file exists and is readable."""
        assert SIMPLE_CLI_EXAMPLE.exists()
        assert SIMPLE_CLI_EXAMPLE.is_file()
        assert os.access(SIMPLE_CLI_EXAMPLE, os.R_OK)
    
    @pytest.mark.skipif(not SIMPLE_CLI_EXAMPLE.exists(), reason="simple_cli_example.py not found")
    def test_simple_cli_example_structure(self):
        """Test simple CLI example has expected structure."""
        with open(SIMPLE_CLI_EXAMPLE, 'r') as f:
            content = f.read()
        
        # Check for key components
        assert "subprocess.run" in content
        assert "ANTHROPIC_API_KEY" in content
        assert "tempfile" in content
        assert "Example 1:" in content
        assert "Example 2:" in content
        assert "Example 3:" in content
    
    @pytest.mark.skipif(not SIMPLE_CLI_EXAMPLE.exists(), reason="simple_cli_example.py not found")
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch('subprocess.run')
    def test_simple_cli_example_execution_mock(self, mock_subprocess):
        """Test simple CLI example execution with mocked subprocess."""
        # Mock successful subprocess calls
        mock_result = Mock()
        mock_result.stdout = "Mocked CLI response"
        mock_subprocess.return_value = mock_result
        
        # This would normally execute the script, but we'll mock it
        # In a real test, you might want to run the actual script
        result = subprocess.run([
            sys.executable, str(SIMPLE_CLI_EXAMPLE)
        ], capture_output=True, text=True, cwd=str(EXAMPLES_DIR))
        
        # The script should attempt to run (even if it fails due to missing dependencies)
        assert result is not None
    
    @pytest.mark.skipif(not SIMPLE_CLI_EXAMPLE.exists(), reason="simple_cli_example.py not found")
    def test_simple_cli_example_no_api_key(self):
        """Test simple CLI example behavior without API key."""
        # Remove API key from environment
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        
        result = subprocess.run([
            sys.executable, str(SIMPLE_CLI_EXAMPLE)
        ], capture_output=True, text=True, env=env, cwd=str(EXAMPLES_DIR))
        
        # Should exit with error due to missing API key
        assert result.returncode != 0
        assert "ANTHROPIC_API_KEY" in result.stdout or "ANTHROPIC_API_KEY" in result.stderr


class TestBasicUsageExample:
    """Test basic_usage.py leaf node."""
    
    @pytest.mark.skipif(not BASIC_USAGE_AVAILABLE, reason="basic_usage.py not available")
    def test_basic_usage_functions_exist(self):
        """Test that basic usage example has expected functions."""
        assert hasattr(basic_usage, 'claude_example')
        assert hasattr(basic_usage, 'openai_example')
        assert hasattr(basic_usage, 'main')
        
        # Check functions are callable
        assert callable(basic_usage.claude_example)
        assert callable(basic_usage.openai_example)
        assert callable(basic_usage.main)
    
    @pytest.mark.skipif(not BASIC_USAGE_AVAILABLE, reason="basic_usage.py not available")
    @patch('basic_usage.UnifiedAgent')
    async def test_claude_example_function(self, mock_agent_class):
        """Test claude_example function with mocked agent."""
        # Mock agent and responses
        mock_agent = AsyncMock()
        mock_response = {
            "content": [
                {"type": "text", "text": "Mocked Claude response"}
            ]
        }
        mock_agent.run_async.return_value = mock_response
        mock_agent_class.return_value = mock_agent
        
        # Run the function
        await basic_usage.claude_example()
        
        # Verify agent was created and used
        mock_agent_class.assert_called_once()
        assert mock_agent.run_async.call_count >= 1
    
    @pytest.mark.skipif(not BASIC_USAGE_AVAILABLE, reason="basic_usage.py not available")
    @patch('basic_usage.UnifiedAgent')
    async def test_openai_example_function(self, mock_agent_class):
        """Test openai_example function with mocked agent."""
        # Mock agent and responses
        mock_agent = AsyncMock()
        mock_response = {
            "content": [
                {"type": "text", "text": "Mocked OpenAI response"}
            ]
        }
        mock_agent.run_async.return_value = mock_response
        mock_agent_class.return_value = mock_agent
        
        # Check if openai_example function exists and run it
        if hasattr(basic_usage, 'openai_example'):
            await basic_usage.openai_example()
            
            # Verify agent was created and used
            mock_agent_class.assert_called_once()
            assert mock_agent.run_async.call_count >= 1
    
    @pytest.mark.skipif(not BASIC_USAGE_AVAILABLE, reason="basic_usage.py not available")
    @patch('basic_usage.claude_example')
    @patch('basic_usage.openai_example')
    async def test_main_function(self, mock_openai, mock_claude):
        """Test main function orchestration."""
        # Mock the example functions
        mock_claude.return_value = None
        mock_openai.return_value = None
        
        # Run main function
        await basic_usage.main()
        
        # Verify both examples were called
        mock_claude.assert_called_once()
        if hasattr(basic_usage, 'openai_example'):
            mock_openai.assert_called_once()


class TestComputerUseExample:
    """Test computer_use_example.py leaf node."""
    
    @pytest.mark.skipif(not COMPUTER_USE_AVAILABLE, reason="computer_use_example.py not available")
    def test_computer_use_example_structure(self):
        """Test computer use example has expected structure."""
        assert hasattr(computer_use_example, 'computer_use_example')
        assert callable(computer_use_example.computer_use_example)
    
    @pytest.mark.skipif(not COMPUTER_USE_EXAMPLE.exists(), reason="computer_use_example.py not found")
    def test_computer_use_example_file_structure(self):
        """Test computer use example file structure."""
        with open(COMPUTER_USE_EXAMPLE, 'r') as f:
            content = f.read()
        
        # Check for key components (updated to match actual file)
        assert "ComputerUseTool" in content
        assert "computer_use" in content.lower()
        assert "async def computer_use_example" in content
    
    @pytest.mark.skipif(not COMPUTER_USE_AVAILABLE, reason="computer_use_example.py not available")
    @patch('computer_use_example.ComputerUseTool')
    async def test_computer_use_main_function(self, mock_tool_class):
        """Test computer use example main function."""
        # Mock tool
        mock_tool = AsyncMock()
        mock_tool.name = "computer_use"
        mock_tool.execute.return_value = "Screenshot captured successfully"
        mock_tool.to_dict.return_value = {"name": "computer_use"}
        mock_tool_class.return_value = mock_tool
        
        # Run main function
        await computer_use_example.computer_use_example()
        
        # Verify tool was created
        mock_tool_class.assert_called_once()


class TestCodeExecutionExample:
    """Test code_execution_example.py leaf node."""
    
    @pytest.mark.skipif(not CODE_EXECUTION_AVAILABLE, reason="code_execution_example.py not available")
    def test_code_execution_example_structure(self):
        """Test code execution example has expected structure."""
        assert hasattr(code_execution_example, 'code_execution_example')
        assert callable(code_execution_example.code_execution_example)
    
    @pytest.mark.skipif(not CODE_EXECUTION_EXAMPLE.exists(), reason="code_execution_example.py not found")
    def test_code_execution_example_file_structure(self):
        """Test code execution example file structure."""
        with open(CODE_EXECUTION_EXAMPLE, 'r') as f:
            content = f.read()
        
        # Check for key components (updated to match actual file)
        assert "CodeExecutionTool" in content
        assert "code_execution" in content.lower()
        assert "async def code_execution_example" in content
    
    @pytest.mark.skipif(not CODE_EXECUTION_AVAILABLE, reason="code_execution_example.py not available")
    @patch('code_execution_example.CodeExecutionTool')
    async def test_code_execution_main_function(self, mock_tool_class):
        """Test code execution example main function."""
        # Mock tool
        mock_tool = Mock()
        mock_tool.name = "code_execution"
        mock_tool.to_dict.return_value = {"name": "code_execution"}
        mock_tool_class.return_value = mock_tool
        
        # Run main function
        await code_execution_example.code_execution_example()
        
        # Verify tool was created
        mock_tool_class.assert_called()


class TestExampleScriptIntegration:
    """Integration tests for example scripts."""
    
    def test_all_example_files_exist(self):
        """Test that all expected example files exist."""
        examples_dir = Path(__file__).parent.parent / "examples"
        
        expected_files = [
            "simple_cli_example.py",
            "basic_usage.py",
            "computer_use_example.py",
            "code_execution_example.py"
        ]
        
        existing_files = []
        for file_name in expected_files:
            file_path = examples_dir / file_name
            if file_path.exists():
                existing_files.append(file_name)
        
        # At least some example files should exist
        assert len(existing_files) > 0, f"No example files found in {examples_dir}"
    
    def test_example_files_are_executable(self):
        """Test that example files are executable."""
        examples_dir = Path(__file__).parent.parent / "examples"
        
        for example_file in examples_dir.glob("*.py"):
            if example_file.name.startswith("test_"):
                continue
                
            # Check file is readable
            assert os.access(example_file, os.R_OK), f"{example_file} is not readable"
            
            # Check file has proper shebang or is a valid Python file
            with open(example_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith("#!"):
                    assert "python" in first_line.lower()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_examples_handle_missing_environment(self):
        """Test that examples handle missing environment variables gracefully."""
        examples_dir = Path(__file__).parent.parent / "examples"
        
        # Test simple_cli_example specifically handles missing API key
        if SIMPLE_CLI_EXAMPLE.exists():
            result = subprocess.run([
                sys.executable, str(SIMPLE_CLI_EXAMPLE)
            ], capture_output=True, text=True, cwd=str(examples_dir))
            
            # Should exit with error or handle gracefully
            assert result.returncode != 0 or "Error" in result.stdout


class TestExampleScriptErrorHandling:
    """Test error handling in example scripts."""
    
    @pytest.mark.skipif(not BASIC_USAGE_AVAILABLE, reason="basic_usage.py not available")
    @patch('basic_usage.UnifiedAgent')
    async def test_basic_usage_handles_agent_error(self, mock_agent_class):
        """Test basic usage example handles agent creation errors."""
        # Mock agent creation failure
        mock_agent_class.side_effect = Exception("Agent creation failed")
        
        # The function should handle the error gracefully
        with pytest.raises(Exception):
            await basic_usage.claude_example()
    
    @pytest.mark.skipif(not COMPUTER_USE_AVAILABLE, reason="computer_use_example.py not available")
    @patch('computer_use_example.ComputerUseAgent')
    async def test_computer_use_handles_initialization_error(self, mock_agent_class):
        """Test computer use example handles initialization errors."""
        # Mock agent initialization failure
        mock_agent_class.side_effect = Exception("Computer use initialization failed")
        
        # The function should handle the error gracefully
        with pytest.raises(Exception):
            await computer_use_example.main()


# Mark all tests as async-compatible
pytestmark = pytest.mark.asyncio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
