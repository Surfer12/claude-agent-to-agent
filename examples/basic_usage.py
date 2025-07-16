#!/usr/bin/env python3
"""
Basic usage example for the Unified Agent System.

This example demonstrates how to use the unified agent with both Claude and OpenAI providers.
"""

import asyncio
import os
from unified_agent import UnifiedAgent, AgentConfig, ProviderType


async def claude_example():
    """Example using Claude provider."""
    print("\n=== Claude Example ===")
    
    # Create configuration for Claude
    config = AgentConfig(
        provider=ProviderType.CLAUDE,
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a helpful AI assistant. Be concise and clear.",
        enable_tools=True,
        enable_code_execution=True,
        verbose=True
    )
    
    # Create agent
    agent = UnifiedAgent(config)
    
    # Run some examples
    examples = [
        "What is the capital of France?",
        "Calculate the factorial of 5 using Python code.",
        "List the files in the current directory."
    ]
    
    for example in examples:
        print(f"\nUser: {example}")
        response = await agent.run_async(example)
        
        # Extract text content
        text_content = []
        for block in response.get("content", []):
            if block.get("type") == "text":
                text_content.append(block.get("text", ""))
        
        print(f"Claude: {' '.join(text_content)}")


async def openai_example():
    """Example using OpenAI provider."""
    print("\n=== OpenAI Example ===")
    
    # Create configuration for OpenAI
    config = AgentConfig(
        provider=ProviderType.OPENAI,
        model="gpt-4o",
        system_prompt="You are a helpful AI assistant. Be concise and clear.",
        enable_tools=True,
        enable_code_execution=True,
        verbose=True
    )
    
    # Create agent
    agent = UnifiedAgent(config)
    
    # Run some examples
    examples = [
        "What is the largest planet in our solar system?",
        "Write a Python function to check if a number is prime.",
        "Create a simple text file with 'Hello, World!' content."
    ]
    
    for example in examples:
        print(f"\nUser: {example}")
        response = await agent.run_async(example)
        
        # Extract text content
        text_content = []
        for block in response.get("content", []):
            if block.get("type") == "text":
                text_content.append(block.get("text", ""))
        
        print(f"OpenAI: {' '.join(text_content)}")


async def computer_use_example():
    """Example using computer use capabilities."""
    print("\n=== Computer Use Example ===")
    
    from unified_agent import ComputerUseAgent
    
    # Create configuration for computer use
    config = AgentConfig(
        provider=ProviderType.OPENAI,
        model="gpt-4o",
        enable_computer_use=True,
        computer_type="local-playwright",
        start_url="https://bing.com",
        verbose=True
    )
    
    # Create computer use agent
    agent = ComputerUseAgent(config)
    
    # Example computer use tasks
    examples = [
        "Navigate to the search page",
        "Take a screenshot of the current page",
        "Type 'unified agent system' into the search box"
    ]
    
    for example in examples:
        print(f"\nUser: {example}")
        response = await agent.run(example)
        
        # Extract text content
        text_content = []
        for block in response.get("content", []):
            if block.get("type") == "text":
                text_content.append(block.get("text", ""))
        
        print(f"Computer Agent: {' '.join(text_content)}")
    
    # Clean up
    await agent.cleanup()


async def interactive_example():
    """Interactive example with user input."""
    print("\n=== Interactive Example ===")
    
    # Ask user for provider preference
    provider_choice = input("Choose provider (claude/openai): ").lower().strip()
    
    if provider_choice == "claude":
        provider = ProviderType.CLAUDE
        model = "claude-3-5-sonnet-20241022"
    elif provider_choice == "openai":
        provider = ProviderType.OPENAI
        model = "gpt-4o"
    else:
        print("Invalid choice, using Claude")
        provider = ProviderType.CLAUDE
        model = "claude-3-5-sonnet-20241022"
    
    # Create configuration
    config = AgentConfig(
        provider=provider,
        model=model,
        enable_tools=True,
        enable_code_execution=True,
        verbose=True
    )
    
    # Create agent
    agent = UnifiedAgent(config)
    
    print(f"\nInteractive session with {provider.value} ({model})")
    print("Type 'exit' to quit, 'reset' to clear history")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'reset':
                agent.reset()
                print("History cleared.")
                continue
            elif not user_input:
                continue
            
            response = await agent.run_async(user_input)
            
            # Extract and display text content
            text_content = []
            for block in response.get("content", []):
                if block.get("type") == "text":
                    text_content.append(block.get("text", ""))
            
            if text_content:
                print(f"\n{provider.value.title()}: {' '.join(text_content)}")
            
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\nEnd of input. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def check_api_keys():
    """Check if required API keys are set."""
    claude_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    print("API Key Status:")
    print(f"  Claude: {'✓ Set' if claude_key else '✗ Not set'}")
    print(f"  OpenAI: {'✓ Set' if openai_key else '✗ Not set'}")
    
    return claude_key, openai_key


async def main():
    """Main function to run examples."""
    print("Unified Agent System - Basic Usage Examples")
    print("=" * 50)
    
    # Check API keys
    claude_key, openai_key = check_api_keys()
    
    # Run examples based on available API keys
    if claude_key:
        await claude_example()
    else:
        print("\nSkipping Claude example - API key not set")
    
    if openai_key:
        await openai_example()
    else:
        print("\nSkipping OpenAI example - API key not set")
    
    # Computer use example (requires OpenAI for now)
    if openai_key:
        try:
            await computer_use_example()
        except Exception as e:
            print(f"Computer use example failed: {str(e)}")
    else:
        print("\nSkipping computer use example - OpenAI API key required")
    
    # Interactive example
    if claude_key or openai_key:
        await interactive_example()
    else:
        print("\nSkipping interactive example - No API keys available")


if __name__ == "__main__":
    asyncio.run(main()) 