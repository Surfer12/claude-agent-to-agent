"""Tests for simple Anthropic API tool files - leaf nodes with minimal logic."""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAnthropicBashTool:
    """Test the anthropic_bash_tool.py leaf node."""
    
    @patch('anthropic.Anthropic')
    def test_bash_tool_client_creation(self, mock_anthropic):
        """Test that the bash tool creates an Anthropic client."""
        # Mock the client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        # Import and execute the module
        import anthropic_bash_tool
        
        # Verify client was created
        mock_anthropic.assert_called_once()
        
        # Verify messages.create was called with correct parameters
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        
        assert call_args[1]['model'] == "claude-sonnet-4-20250514"
        assert call_args[1]['max_tokens'] == 1024
        assert 'tools' in call_args[1]
        assert call_args[1]['tools'][0]['type'] == "bash_20250124"
        assert call_args[1]['tools'][0]['name'] == "bash"
    
    @patch('anthropic.Anthropic')
    def test_bash_tool_message_content(self, mock_anthropic):
        """Test the message content for bash tool."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        import anthropic_bash_tool
        
        call_args = mock_client.messages.create.call_args
        messages = call_args[1]['messages']
        
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert 'Python files' in messages[0]['content']
        assert 'current directory' in messages[0]['content']


class TestAnthropicWeatherTool:
    """Test the anthropic_weather_tool.py leaf node."""
    
    @patch('anthropic.Anthropic')
    def test_weather_tool_client_creation(self, mock_anthropic):
        """Test that the weather tool creates an Anthropic client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_client.beta.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        import anthropic_weather_tool
        
        mock_anthropic.assert_called_once()
        mock_client.beta.messages.create.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_weather_tool_configuration(self, mock_anthropic):
        """Test weather tool configuration."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_client.beta.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        import anthropic_weather_tool
        
        call_args = mock_client.beta.messages.create.call_args
        
        assert call_args[1]['model'] == "claude-3-7-sonnet-20250219"
        assert call_args[1]['max_tokens'] == 1024
        assert 'tools' in call_args[1]
        
        tool = call_args[1]['tools'][0]
        assert tool['name'] == 'get_weather'
        assert 'weather' in tool['description'].lower()
        assert 'input_schema' in tool
        assert 'location' in tool['input_schema']['properties']
        assert tool['input_schema']['required'] == ['location']
        
        assert call_args[1]['betas'] == ["token-efficient-tools-2025-02-19"]
    
    @patch('anthropic.Anthropic')
    def test_weather_tool_message(self, mock_anthropic):
        """Test weather tool message content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_client.beta.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        import anthropic_weather_tool
        
        call_args = mock_client.beta.messages.create.call_args
        messages = call_args[1]['messages']
        
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert 'San Francisco' in messages[0]['content']
        assert 'weather' in messages[0]['content'].lower()


class TestAnthropicTextEditor:
    """Test the anthropic_text_editor.py leaf node."""
    
    @patch('anthropic.Anthropic')
    def test_text_editor_client_creation(self, mock_anthropic):
        """Test that the text editor creates an Anthropic client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        import anthropic_text_editor
        
        mock_anthropic.assert_called_once()
        mock_client.messages.create.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_text_editor_configuration(self, mock_anthropic):
        """Test text editor tool configuration."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        import anthropic_text_editor
        
        call_args = mock_client.messages.create.call_args
        
        assert call_args[1]['model'] == "claude-opus-4-20250514"
        assert call_args[1]['max_tokens'] == 1024
        assert 'tools' in call_args[1]
        
        tool = call_args[1]['tools'][0]
        assert tool['type'] == 'text_editor_20250429'
        assert tool['name'] == 'str_replace_based_edit_tool'
    
    @patch('anthropic.Anthropic')
    def test_text_editor_message(self, mock_anthropic):
        """Test text editor message content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        import anthropic_text_editor
        
        call_args = mock_client.messages.create.call_args
        messages = call_args[1]['messages']
        
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert 'syntax error' in messages[0]['content'].lower()
        assert 'primes.py' in messages[0]['content']


class TestAnthropicWebSearch:
    """Test the anthropic_web_search.py leaf node."""
    
    @patch('anthropic.Anthropic')
    def test_web_search_client_creation(self, mock_anthropic):
        """Test that the web search creates an Anthropic client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        import anthropic_web_search
        
        mock_anthropic.assert_called_once()
        mock_client.messages.create.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_web_search_configuration(self, mock_anthropic):
        """Test web search tool configuration."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        import anthropic_web_search
        
        call_args = mock_client.messages.create.call_args
        
        assert call_args[1]['model'] == "claude-opus-4-20250514"
        assert call_args[1]['max_tokens'] == 1024
        assert 'tools' in call_args[1]
        
        tool = call_args[1]['tools'][0]
        assert tool['type'] == 'web_search_20250305'
        assert tool['name'] == 'web_search'
        assert tool['max_uses'] == 5
    
    @patch('anthropic.Anthropic')
    def test_web_search_message(self, mock_anthropic):
        """Test web search message content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        import anthropic_web_search
        
        call_args = mock_client.messages.create.call_args
        messages = call_args[1]['messages']
        
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert 'TypeScript 5.5' in messages[0]['content']
        assert 'web app' in messages[0]['content']


class TestAnthropicClientSimple:
    """Test the anthropic_client.py leaf node."""
    
    @patch('anthropic.Anthropic')
    def test_client_creation(self, mock_anthropic):
        """Test simple client creation."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        import anthropic_client
        
        mock_anthropic.assert_called_once()


# Integration tests for leaf node behavior
class TestLeafNodeIntegration:
    """Integration tests for leaf node behavior."""
    
    def test_all_tools_importable(self):
        """Test that all tool files can be imported without errors."""
        tool_files = [
            'anthropic_bash_tool',
            'anthropic_weather_tool', 
            'anthropic_text_editor',
            'anthropic_web_search',
            'anthropic_client'
        ]
        
        for tool_file in tool_files:
            try:
                with patch('anthropic.Anthropic'):
                    __import__(tool_file)
            except ImportError as e:
                pytest.fail(f"Failed to import {tool_file}: {e}")
    
    @patch('anthropic.Anthropic')
    def test_tools_use_different_models(self, mock_anthropic):
        """Test that different tools use different models."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_client.beta.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        # Import all tools
        import anthropic_bash_tool
        import anthropic_weather_tool
        import anthropic_text_editor
        import anthropic_web_search
        
        # Check that different models are used
        calls = mock_client.messages.create.call_args_list + mock_client.beta.messages.create.call_args_list
        models_used = [call[1]['model'] for call in calls if 'model' in call[1]]
        
        # Should have multiple different models
        unique_models = set(models_used)
        assert len(unique_models) > 1
        
        # Check for specific models
        assert any('sonnet' in model.lower() for model in models_used)
        assert any('opus' in model.lower() for model in models_used)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
