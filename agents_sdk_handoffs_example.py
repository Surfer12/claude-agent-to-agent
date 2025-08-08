#!/usr/bin/env python3
"""
OpenAI Agents SDK - Handoffs Example

This demonstrates how to use handoffs for agent-to-agent transfers.
This replaces the Swarm's agent transfer functionality.
"""

import os
import asyncio
from agents import Agent, Runner

def main():
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create specialized agents
    spanish_agent = Agent(
        name="Spanish Agent",
        instructions="You only speak Spanish. Respond to everything in Spanish only.",
    )

    english_agent = Agent(
        name="English Agent", 
        instructions="You only speak English. Respond to everything in English only.",
    )

    french_agent = Agent(
        name="French Agent",
        instructions="You only speak French. Respond to everything in French only.",
    )

    # Create triage agent that can handoff to others
    triage_agent = Agent(
        name="Triage Agent",
        instructions="""You are a language triage agent. Based on the language of the user's request, 
        handoff to the appropriate language specialist:
        - For Spanish text, handoff to Spanish Agent
        - For French text, handoff to French Agent  
        - For English text, handoff to English Agent
        - If unsure, ask the user to clarify the language they want to use.""",
        handoffs=[spanish_agent, english_agent, french_agent],
    )

    async def run_examples():
        print("🤖 Testing Handoffs Example")
        print("=" * 50)
        
        # Test Spanish handoff
        print("\n1. Testing Spanish input:")
        result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
        print(f"Response: {result.final_output}")
        
        # Test English handoff
        print("\n2. Testing English input:")
        result = await Runner.run(triage_agent, input="Hello, how are you today?")
        print(f"Response: {result.final_output}")
        
        # Test French handoff
        print("\n3. Testing French input:")
        result = await Runner.run(triage_agent, input="Bonjour, comment allez-vous?")
        print(f"Response: {result.final_output}")

    try:
        asyncio.run(run_examples())
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your OPENAI_API_KEY is set correctly.")

if __name__ == "__main__":
    main()