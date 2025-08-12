#!/usr/bin/env python3
"""
OpenAI Agents SDK - Hello World Example

This is the simplest example of using the new OpenAI Agents SDK.
It replaces the old Swarm framework with a more production-ready solution.
"""

import os
from agents import Agent, Runner

def main():
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create a simple agent
    agent = Agent(
        name="Assistant", 
        instructions="You are a helpful assistant. Be concise but friendly."
    )
    
    # Run the agent synchronously
    try:
        result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
        print("🤖 Agent Response:")
        print(result.final_output)
        
        # Example output:
        # Code within the code,
        # Functions calling themselves,
        # Infinite loop's dance.
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your OPENAI_API_KEY is set correctly.")

if __name__ == "__main__":
    main()