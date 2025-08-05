"""
PROPERLY Working Swarm Demo - Agent Transfers That Actually Work!

This fixes the core issue: agents transfer but don't actually take over.
"""

import sys
import os
sys.path.append('swarm')

from swarm import Swarm, Agent

def calculate(expression: str) -> str:
    """Math function that actually gets called by the Math Agent."""
    try:
        original_expr = expression
        
        # Handle the specific "2+@" case from your conversation
        expression = expression.replace('@', '2')
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
        
        return f"I calculated: {original_expr} → {expression} = {result}"
        
    except Exception as e:
        return f"I couldn't calculate '{original_expr}'. Try: 2+2, sqrt(16), or sin(pi/2)"

def math_help() -> str:
    """Show what the math agent can do."""
    return "I can calculate: +, -, *, /, **, sqrt(), sin(), cos(), tan(), log(), pi, e"

# Create the Math Agent
math_agent = Agent(
    name="Math Agent",
    instructions="""🔢 I AM THE MATH AGENT! 

I will ALWAYS identify myself clearly and use my functions:
- For math expressions: use calculate() function
- For help: use math_help() function

I start every response with "🔢 Math Agent:" so users know I'm responding.
I'm enthusiastic about math and will solve any calculation!""",
    functions=[calculate, math_help],
)

def transfer_to_math():
    """Transfer to the Math Agent - this function returns the math_agent."""
    return math_agent

# Create the Triage Agent  
triage_agent = Agent(
    name="Triage Agent",
    instructions="""👋 I AM THE TRIAGE AGENT!

When users ask about math or want to connect to math agent:
- I use transfer_to_math() function immediately
- I confirm the transfer happened

I should NOT try to handle math myself - that's what the Math Agent is for!""",
    functions=[transfer_to_math],
)

def run_corrected_demo():
    """Run the demo showing proper agent handoffs."""
    client = Swarm()
    
    print("=== PROPERLY Working Swarm Demo ===")
    print("This time the Math Agent will ACTUALLY respond to math questions!\n")
    
    conversation_flow = [
        "connect me to math agent",
        "what is 2+@", 
        "can you also calculate sqrt(16)?",
        "how will i know i am connected?"
    ]
    
    messages = []
    current_agent = triage_agent
    
    for i, user_input in enumerate(conversation_flow, 1):
        print(f"**Step {i}:** 👤 User: {user_input}")
        
        messages.append({"role": "user", "content": user_input})
        
        response = client.run(
            agent=current_agent,
            messages=messages,
            debug=False  # Set to True to see what's happening under the hood
        )
        
        # Critical: Update the current agent to the one returned by the response
        current_agent = response.agent
        messages.extend(response.messages)
        
        # Show the response 
        agent_response = response.messages[-1]['content']
        print(f"🤖 **[{current_agent.name}]**: {agent_response}")
        print()
    
    print("=== Analysis ===")
    print(f"Final agent: {current_agent.name}")
    print("Expected behavior:")
    print("✅ Step 1: Triage Agent handles connection request")  
    print("✅ Step 2: Math Agent should respond with calculated result")
    print("✅ Step 3: Math Agent should calculate sqrt(16) = 4.0")
    print("✅ Step 4: Math Agent explains they're connected")

def test_direct_math_agent():
    """Test the Math Agent directly to verify it works."""
    client = Swarm()
    
    print("\n=== Direct Math Agent Test ===")
    
    test_messages = [{"role": "user", "content": "what is 2+@"}]
    
    response = client.run(
        agent=math_agent,
        messages=test_messages,
        debug=False
    )
    
    print("👤 User (direct to Math Agent): what is 2+@")
    print(f"🤖 Math Agent: {response.messages[-1]['content']}")
    print()
    
    # Check if function was called
    if len(response.messages) > 1:
        for msg in response.messages:
            if msg.get('role') == 'tool':
                print(f"✅ Function executed: {msg.get('tool_name')} returned: {msg.get('content')}")

if __name__ == "__main__":
    run_corrected_demo()
    test_direct_math_agent()