#!/usr/bin/env python3
"""Test script for the unified agent system."""

import asyncio
from unified_agent_system import Agent, AgentConfig, ProviderType

async def test_mock_provider():
    """Test the mock provider functionality."""
    print("Testing Mock Provider...")
    
    config = AgentConfig(
        provider=ProviderType.MOCK,
        model="test-model",
        system_prompt="You are a test assistant.",
        max_tokens=100,
        temperature=0.7
    )
    
    agent = Agent(config)
    
    # Test basic conversation
    response1 = await agent.process_message("Hello, can you introduce yourself?")
    print(f"Response 1: {response1}")
    
    # Test follow-up message
    response2 = await agent.process_message("What can you help me with?")
    print(f"Response 2: {response2}")
    
    # Check conversation state
    context = agent.get_context()
    print(f"Conversation has {len(context)} messages")
    
    print("Mock provider test completed successfully!\n")

def test_provider_types():
    """Test that all provider types are available."""
    print("Testing Provider Types...")
    
    providers = [ProviderType.CLAUDE, ProviderType.OPENAI, ProviderType.MOCK]
    for provider in providers:
        print(f"Provider {provider.value} is available")
    
    print("Provider types test completed successfully!\n")

def test_agent_config():
    """Test agent configuration."""
    print("Testing Agent Configuration...")
    
    config = AgentConfig(
        provider=ProviderType.MOCK,
        model="test-model",
        max_tokens=2048,
        temperature=0.5,
        system_prompt="Custom system prompt",
        verbose=True
    )
    
    print(f"Provider: {config.provider}")
    print(f"Model: {config.model}")
    print(f"Max tokens: {config.max_tokens}")
    print(f"Temperature: {config.temperature}")
    print(f"System prompt: {config.system_prompt}")
    print(f"Verbose: {config.verbose}")
    
    print("Agent configuration test completed successfully!\n")

async def main():
    """Run all tests."""
    print("=== Unified Agent System Tests ===\n")
    
    test_provider_types()
    test_agent_config()
    await test_mock_provider()
    
    print("=== All tests completed successfully! ===")

if __name__ == "__main__":
    asyncio.run(main())
