import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from ..providers import ClaudeProvider, OpenAIProvider
from ..core.types import AgentConfig, ProviderType

@pytest.fixture
def mock_config():
    return AgentConfig(
        provider=ProviderType.CLAUDE,
        model="test-model",
        api_key="test-key"
    )

@pytest.mark.asyncio
async def test_claude_provider_create_message(mock_config):
    provider = ClaudeProvider(mock_config)
    
    with patch.object(provider.client.messages, 'create', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = MagicMock(
            content=[MagicMock(type="text", text="Test response")]
        )
        
        response = await provider.create_message(
            messages=[{"role": "user", "content": "test"}]
        )
        
        assert "content" in response
        assert response["content"][0]["text"] == "Test response"

@pytest.mark.asyncio
async def test_openai_provider_create_message(mock_config):
    mock_config.provider = ProviderType.OPENAI
    provider = OpenAIProvider(mock_config)
    
    with patch.object(provider.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))]
        )
        
        response = await provider.create_message(
            messages=[{"role": "user", "content": "test"}]
        )
        
        assert "content" in response
        assert response["content"][0]["text"] == "Test response"

def test_claude_get_tool_schema(mock_config):
    provider = ClaudeProvider(mock_config)
    mock_tool = MagicMock()
    mock_tool.to_dict.return_value = {"name": "test"}
    schema = provider.get_tool_schema([mock_tool])
    assert schema[0]["name"] == "test"

def test_openai_get_tool_schema(mock_config):
    provider = OpenAIProvider(mock_config)
    mock_tool = MagicMock()
    mock_tool.to_openai_dict.return_value = {"type": "function"}
    schema = provider.get_tool_schema([mock_tool])
    assert schema[0]["type"] == "function"

@pytest.mark.asyncio
async def test_claude_beta_headers(mock_config):
    provider = ClaudeProvider(mock_config)
    tools = [{"name": "computer"}, {"tool_type": "code_execution", "supports_files": True}]
    betas = provider._get_beta_headers(tools)
    assert "computer-use-2025-01-24" in betas or "computer-use-2024-10-22" in betas
    assert "code-execution-2025-05-22" in betas
    assert "files-api-2025-04-14" in betas 