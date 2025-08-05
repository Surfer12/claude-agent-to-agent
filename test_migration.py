#!/usr/bin/env python3
"""
Test script to verify OpenAI Agents SDK migration
This script tests the key features of the new SDK
"""

import asyncio
import os
import sys

def test_import():
    """Test that we can import the Agents SDK."""
    try:
        from agents import Agent, Runner, function_tool
        print("✅ Successfully imported OpenAI Agents SDK")
        return True
    except ImportError as e:
        print(f"❌ Failed to import OpenAI Agents SDK: {e}")
        print("Please install with: pip install openai-agents")
        return False

def test_basic_agent():
    """Test basic agent creation and execution."""
    try:
        from agents import Agent, Runner, function_tool
        
        @function_tool
        def simple_calc(x: int, y: int) -> str:
            return f"Result: {x + y}"
        
        agent = Agent(
            name="Test Agent",
            instructions="I am a test agent that can add numbers.",
            tools=[simple_calc]
        )
        
        print("✅ Successfully created test agent")
        return agent
    except Exception as e:
        print(f"❌ Failed to create test agent: {e}")
        return None

async def test_async_execution(agent):
    """Test async execution of the agent."""
    try:
        result = await Runner.run(agent, "What is 5 + 3?")
        print(f"✅ Async execution successful: {result.final_output}")
        return True
    except Exception as e:
        print(f"❌ Async execution failed: {e}")
        return False

def test_sync_execution(agent):
    """Test synchronous execution of the agent."""
    try:
        result = Runner.run_sync(agent, "What is 2 + 2?")
        print(f"✅ Sync execution successful: {result.final_output}")
        return True
    except Exception as e:
        print(f"❌ Sync execution failed: {e}")
        return False

def test_environment():
    """Test environment setup."""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OPENAI_API_KEY is set")
        return True
    else:
        print("⚠️  OPENAI_API_KEY not set (will use default if available)")
        return True

async def main():
    """Run all tests."""
    print("🧪 Testing OpenAI Agents SDK Migration")
    print("=" * 40)
    
    # Test 1: Import
    if not test_import():
        return False
    
    # Test 2: Environment
    test_environment()
    
    # Test 3: Agent Creation
    agent = test_basic_agent()
    if not agent:
        return False
    
    # Test 4: Sync Execution
    test_sync_execution(agent)
    
    # Test 5: Async Execution
    await test_async_execution(agent)
    
    print("\n" + "=" * 40)
    print("🎉 Migration test completed!")
    print("\nNext steps:")
    print("1. Set your OPENAI_API_KEY environment variable")
    print("2. Run the full migration demo: python3 migration_to_agents_sdk.py")
    print("3. Check the migration guide: MIGRATION_GUIDE_AGENTS_SDK.md")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)