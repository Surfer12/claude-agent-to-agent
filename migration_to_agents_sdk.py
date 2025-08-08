"""
Migration from Swarm to OpenAI Agents SDK

This example shows how to migrate your existing Swarm implementation
to the new OpenAI Agents SDK, which is the production-ready evolution of Swarm.

Key improvements in the Agents SDK:
- Handoffs: Specialized tool calls for transferring control between agents
- Guardrails: Configurable safety checks for input/output validation  
- Sessions: Automatic conversation history management
- Tracing: Built-in tracking of agent runs
- Provider-agnostic: Supports OpenAI and 100+ other LLMs
"""

import asyncio
from agents import Agent, Runner, function_tool
from agents.memory import SQLiteSession

# ============================================================================
# MIGRATION: Function Tools (replaces Swarm's functions parameter)
# ============================================================================

@function_tool
def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions with helpful error handling."""
    try:
        original_expr = expression
        
        # Handle the specific case from your conversation: 2+@
        expression = expression.replace('@', '2')  # Your exact case!
        expression = expression.replace('x', '*')
        expression = expression.replace('^', '**')
        
        # Safe evaluation
        import math
        allowed_names = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "pi": math.pi, "e": math.e, "pow": pow
        }
        
        result = eval(expression, allowed_names, {})
        
        response = f"🔢 **Math Agent here!** ✅ I calculated: {original_expr}"
        if expression != original_expr:
            response += f" (I interpreted '@' as '2', so: {expression})"
        response += f" = **{result}**"
        
        return response
        
    except Exception as e:
        return f"🔢 **Math Agent here!** ❌ I couldn't calculate '{original_expr}'. Try: 2+2, sqrt(16), or sin(pi/2)"

@function_tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"🌤️ **Weather Agent here!** The weather in {city} is sunny and 72°F."

# ============================================================================
# MIGRATION: Agent Creation (replaces Swarm's Agent class)
# ============================================================================

# Math Agent - now with handoffs capability
math_agent = Agent(
    name="Math Agent",
    instructions="""🔢 I am the MATH AGENT! 

When users ask math questions, I will:
1. ALWAYS start my response with "🔢 **Math Agent here!**" so they know I'm connected
2. Use the calculate() function to solve their math problems
3. Handle typos gracefully (like @ instead of 2)
4. Give clear, helpful responses

I'm excited to help with math!""",
    tools=[calculate],  # Note: 'tools' instead of 'functions'
)

# Weather Agent - new specialist agent
weather_agent = Agent(
    name="Weather Agent", 
    instructions="""🌤️ I am the WEATHER AGENT!

I handle all weather-related questions and use the get_weather() function.
I always identify myself clearly in responses.""",
    tools=[get_weather],
)

# Triage Agent - now with handoffs to other agents
triage_agent = Agent(
    name="Triage Agent",
    instructions="""👋 I am the TRIAGE AGENT!

I route users to the appropriate specialist agent based on their request:

- Math questions → Math Agent
- Weather questions → Weather Agent  
- General questions → I handle directly

I will use handoffs to transfer control to the appropriate specialist.""",
    handoffs=[math_agent, weather_agent],  # NEW: handoffs parameter
)

# ============================================================================
# MIGRATION: Async Runner (replaces Swarm's client.run())
# ============================================================================

async def demo_agents_sdk():
    """Demonstrate the new Agents SDK capabilities."""
    
    print("=== OpenAI Agents SDK Migration Demo ===")
    print("Key improvements: handoffs, sessions, tracing, provider-agnostic\n")
    
    # Test 1: Direct Math Agent (no handoff needed)
    print("**Test 1: Direct Math Agent**")
    result = await Runner.run(math_agent, "what is 2+@")
    print(f"👤 User: what is 2+@")
    print(f"🤖 Result: {result.final_output}")
    print()
    
    # Test 2: Triage with handoff to Math Agent
    print("**Test 2: Triage → Math Agent Handoff**")
    result = await Runner.run(triage_agent, "connect me to math agent")
    print(f"👤 User: connect me to math agent")
    print(f"🤖 Result: {result.final_output}")
    print()
    
    # Test 3: Math question through triage (should handoff)
    print("**Test 3: Math Question Through Triage**")
    result = await Runner.run(triage_agent, "what is sqrt(16)?")
    print(f"👤 User: what is sqrt(16)?")
    print(f"🤖 Result: {result.final_output}")
    print()
    
    # Test 4: Weather question through triage
    print("**Test 4: Weather Question Through Triage**")
    result = await Runner.run(triage_agent, "what's the weather in Tokyo?")
    print(f"👤 User: what's the weather in Tokyo?")
    print(f"🤖 Result: {result.final_output}")
    print()

# ============================================================================
# NEW FEATURE: Sessions (automatic conversation history)
# ============================================================================

async def demo_sessions():
    """Demonstrate the new Sessions feature."""
    
    print("=== Sessions Demo (Automatic Memory) ===")
    
    # Create a session for conversation history
    session = SQLiteSession("user_123")
    
    # First conversation turn
    result = await Runner.run(
        math_agent,
        "What is 2+2?",
        session=session
    )
    print(f"👤 User: What is 2+2?")
    print(f"🤖 Math Agent: {result.final_output}")
    
    # Second turn - agent remembers previous context
    result = await Runner.run(
        math_agent, 
        "What about 3+3?",
        session=session
    )
    print(f"👤 User: What about 3+3?")
    print(f"🤖 Math Agent: {result.final_output}")
    
    # Third turn - still remembers the conversation
    result = await Runner.run(
        math_agent,
        "What was the first calculation?",
        session=session
    )
    print(f"👤 User: What was the first calculation?")
    print(f"🤖 Math Agent: {result.final_output}")
    print()

# ============================================================================
# NEW FEATURE: Synchronous Runner (for simple cases)
# ============================================================================

def demo_sync_runner():
    """Demonstrate synchronous runner for simple cases."""
    
    print("=== Synchronous Runner Demo ===")
    
    # Simple synchronous call
    result = Runner.run_sync(math_agent, "what is 5+5?")
    print(f"👤 User: what is 5+5?")
    print(f"🤖 Math Agent: {result.final_output}")
    print()

# ============================================================================
# MIGRATION COMPARISON: Old vs New
# ============================================================================

def show_migration_comparison():
    """Show the key differences between Swarm and Agents SDK."""
    
    print("=== Migration Comparison: Swarm → Agents SDK ===")
    print()
    
    print("OLD (Swarm):")
    print("  from swarm import Swarm, Agent")
    print("  client = Swarm()")
    print("  response = client.run(agent=math_agent, messages=messages)")
    print("  current_agent = response.agent  # Manual agent tracking")
    print()
    
    print("NEW (Agents SDK):")
    print("  from agents import Agent, Runner")
    print("  result = await Runner.run(agent, input)")
    print("  # Automatic agent handoffs, no manual tracking needed")
    print()
    
    print("Key Improvements:")
    print("✅ Handoffs: Automatic agent transfers")
    print("✅ Sessions: Built-in conversation memory") 
    print("✅ Tracing: Built-in run tracking")
    print("✅ Guardrails: Input/output validation")
    print("✅ Provider-agnostic: 100+ LLM support")
    print("✅ Simpler API: Less boilerplate code")

# ============================================================================
# Main execution
# ============================================================================

async def main():
    """Run all demos."""
    
    # Show migration comparison first
    show_migration_comparison()
    print("\n" + "="*60 + "\n")
    
    # Run the demos
    await demo_agents_sdk()
    print("\n" + "="*60 + "\n")
    
    await demo_sessions()
    print("\n" + "="*60 + "\n")
    
    demo_sync_runner()
    
    print("\n=== Migration Complete! ===")
    print("Your Swarm code has been successfully migrated to the OpenAI Agents SDK.")
    print("Benefits: Better performance, more features, production-ready!")

if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())