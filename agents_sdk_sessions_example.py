#!/usr/bin/env python3
"""
OpenAI Agents SDK - Sessions Example

This demonstrates how to use sessions for conversation memory.
Sessions automatically maintain conversation history across multiple agent runs.
"""

import os
import asyncio
from agents import Agent, Runner, SQLiteSession, function_tool

@function_tool
def save_user_preference(preference_type: str, value: str) -> str:
    """Save a user preference for later reference."""
    # In a real app, you'd save this to a database
    return f"✅ Saved your {preference_type} preference: {value}"

@function_tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def main():
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create agent with tools
    agent = Agent(
        name="Memory Assistant",
        instructions="""You are a helpful assistant with memory. You can:
        - Remember previous conversations within the same session
        - Save user preferences for later reference
        - Provide the current time when asked
        
        Always be personal and reference previous parts of the conversation 
        when relevant. Be concise but friendly.""",
        tools=[save_user_preference, get_current_time],
    )

    async def run_session_demo():
        print("🧠 Testing Sessions Example")
        print("=" * 50)
        
        # Create a session instance for user "alice_123"
        session = SQLiteSession("alice_123", "conversations.db")
        
        print("\n=== FIRST CONVERSATION ===")
        
        # First turn - introduction
        print("\n1. First interaction:")
        result = await Runner.run(
            agent,
            "Hi! My name is Alice and I love pizza. What city is the Golden Gate Bridge in?",
            session=session
        )
        print(f"Assistant: {result.final_output}")
        
        # Second turn - agent should remember context
        print("\n2. Follow-up question (should remember context):")
        result = await Runner.run(
            agent,
            "What state is it in?",
            session=session
        )
        print(f"Assistant: {result.final_output}")
        
        # Third turn - save preference
        print("\n3. Saving preference:")
        result = await Runner.run(
            agent,
            "Please save my favorite food preference",
            session=session
        )
        print(f"Assistant: {result.final_output}")
        
        print("\n=== SIMULATING TIME PASSING ===")
        print("(In a real app, this could be hours or days later)")
        
        # Fourth turn - should still remember everything
        print("\n4. Later conversation (should remember Alice and pizza):")
        result = await Runner.run(
            agent,
            "What time is it? Also, do you remember my name and what I like to eat?",
            session=session
        )
        print(f"Assistant: {result.final_output}")
        
        print("\n=== DIFFERENT USER SESSION ===")
        
        # Different session for different user
        bob_session = SQLiteSession("bob_456", "conversations.db")
        
        print("\n5. Different user (separate session):")
        result = await Runner.run(
            agent,
            "Hi! Do you know anything about Alice or pizza?",
            session=bob_session
        )
        print(f"Assistant: {result.final_output}")
        
        print("\n=== SYNCHRONOUS EXAMPLE ===")
        
        # Show synchronous usage with sessions
        print("\n6. Synchronous session usage:")
        sync_session = SQLiteSession("sync_user", "conversations.db")
        
        result = Runner.run_sync(
            agent,
            "I prefer coffee over tea. Remember this!",
            session=sync_session
        )
        print(f"Assistant: {result.final_output}")
        
        result = Runner.run_sync(
            agent,
            "What do I prefer to drink?",
            session=sync_session
        )
        print(f"Assistant: {result.final_output}")
        
        print("\n=== SESSION STATISTICS ===")
        print(f"Alice's session has stored conversation history")
        print(f"Bob's session is separate and has no knowledge of Alice")
        print(f"Sync user's session maintains its own conversation thread")
        print(f"All conversations are stored in 'conversations.db'")

    try:
        asyncio.run(run_session_demo())
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your OPENAI_API_KEY is set correctly.")

if __name__ == "__main__":
    main()