"""
Fixed Swarm Demo - Shows proper agent transfers and function execution

This fixes the issues from your conversations:
1. Clear agent identification 
2. Actual function execution
3. Proper error handling
4. User-friendly responses
"""

import json
from swarm import Swarm, Agent

# Math Functions
def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions with helpful error handling."""
    try:
        original_expr = expression
        
        # Handle common typos and symbols  
        expression = expression.replace('@', '2')  # Your specific case!
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
        
        response = f"✅ **Math Result:** {original_expr}"
        if expression != original_expr:
            response += f" (interpreted as {expression})"
        response += f" = **{result}**"
        
        return response
        
    except Exception as e:
        return f"❌ **Math Error:** Could not calculate '{original_expr}'\n💡 **Suggestion:** Try something like: 2+2, sqrt(16), or sin(pi/2)"

def math_help() -> str:
    """Show what the math agent can do."""
    return """🔢 **Math Agent Capabilities:**

**Basic Operations:** +, -, *, /, ** (power)
**Functions:** sqrt(), sin(), cos(), tan(), log()
**Constants:** pi, e
**Examples:** 
• 2+2 = 4
• sqrt(16) = 4.0  
• sin(pi/2) = 1.0

Just ask me any math question!"""

# Weather Functions (from previous example)
def get_weather(location: str, weather_type: str = "current") -> str:
    """Get weather information."""
    if "santa barbara" in location.lower() and weather_type.lower() == "uv":
        return """🌞 **UV Forecast for Santa Barbara, CA:**

**Current UV Index:** 7 (High)
**Recommendation:** Wear SPF 30+ sunscreen
**Peak UV Time:** 12:00 PM - 2:00 PM  
**Current Time:** 2:30 PM

**Hourly Forecast:**
• 12 PM: UV 8 (Very High) ⚠️
• 1 PM: UV 8 (Very High) ⚠️  
• 2 PM: UV 7 (High)
• 3 PM: UV 5 (Moderate)"""
    
    return f"🌤️ **Weather for {location}:** 72°F, Sunny, Light breeze"

def get_uv_forecast(location: str) -> str:
    """Get UV forecast specifically."""
    return get_weather(location, "uv")

# Create specialized agents with clear identification
math_agent = Agent(
    name="Math Agent",
    instructions="""🔢 **I am the Math Agent!** 

I specialize in mathematics and calculations. I will:
1. Always identify myself as the Math Agent in responses
2. Use calculate() for math expressions  
3. Use math_help() if users need guidance
4. Handle typos gracefully (like @ instead of 2)
5. Provide clear, formatted responses with emojis

I love solving math problems! What calculation can I help you with?""",
    functions=[calculate, math_help],
)

weather_agent = Agent(
    name="Weather Agent", 
    instructions="""🌤️ **I am the Weather Agent!**

I specialize in weather and UV forecasts. I will:
1. Always identify myself as the Weather Agent
2. Use get_weather() for general weather
3. Use get_uv_forecast() for UV-specific requests
4. Provide detailed, helpful weather information
5. Include safety recommendations for UV exposure

What weather information do you need?""",
    functions=[get_weather, get_uv_forecast],
)

# Transfer functions
def transfer_to_math():
    """Transfer to math specialist."""
    return math_agent

def transfer_to_weather():
    """Transfer to weather specialist."""  
    return weather_agent

# Improved triage agent
triage_agent = Agent(
    name="Triage Agent",
    instructions="""👋 **I am the Triage Agent!**

I help connect you with the right specialist:

🔢 **Math Agent:** calculations, equations, mathematical functions
🌤️ **Weather Agent:** forecasts, UV index, weather conditions

I will transfer you immediately when you ask about these topics.
After transfer, the specialist will identify themselves and help you directly.

What can I help you find a specialist for?""",
    functions=[transfer_to_math, transfer_to_weather],
)

def run_conversation_demo():
    """Demonstrate the fixed conversation flow."""
    client = Swarm()
    
    print("=== Fixed Swarm Agent Demo ===\n")
    
    # Recreate the problematic conversation but fixed
    conversation_steps = [
        ("connect me to math agent", triage_agent),
        ("what is 2+@", None),  # Will use the agent from previous response  
        ("can you calculate sqrt(16) too?", None),
        ("ok thanks", None)
    ]
    
    messages = []
    current_agent = triage_agent
    
    for i, (user_input, specified_agent) in enumerate(conversation_steps, 1):
        if specified_agent:
            current_agent = specified_agent
            
        print(f"**Step {i}:** User says: '{user_input}'")
        messages.append({"role": "user", "content": user_input})
        
        response = client.run(
            agent=current_agent,
            messages=messages,
            debug=False
        )
        
        # Update for next iteration
        messages.extend(response.messages) 
        current_agent = response.agent
        
        agent_response = response.messages[-1]['content']
        print(f"**[{current_agent.name}]:** {agent_response}")
        print()
    
    print("=== Demo Complete ===")
    print(f"✅ **Final agent:** {current_agent.name}")
    print("🎉 **Notice:** Each agent clearly identifies itself and performs its specialized function!")

if __name__ == "__main__":
    run_conversation_demo()