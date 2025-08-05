"""
Improved Swarm Weather Agent Demo

This demonstrates how to fix the issues in your original conversation:
1. Better agent instructions
2. Proper UV forecast handling
3. Clear transfer logic
4. Avoid repetitive loops
"""

import json
from swarm import Swarm, Agent

# Weather functions with UV support
def get_weather(location: str, weather_type: str = "current") -> str:
    """Get weather information for a location."""
    location_lower = location.lower()
    
    if "santa barbara" in location_lower:
        if weather_type.lower() == "uv":
            return json.dumps({
                "location": location,
                "uv_index": 7,
                "uv_level": "High",
                "recommendation": "Wear SPF 30+ sunscreen, seek shade 12-2 PM",
                "peak_time": "12:00 PM - 2:00 PM",
                "current_time": "2:30 PM",
                "hourly_forecast": {
                    "11 AM": "UV 6 (High)",
                    "12 PM": "UV 8 (Very High)", 
                    "1 PM": "UV 8 (Very High)",
                    "2 PM": "UV 7 (High)",
                    "3 PM": "UV 5 (Moderate)"
                }
            })
        else:
            return json.dumps({
                "location": location,
                "temperature": "72°F (22°C)",
                "conditions": "Sunny and clear",
                "humidity": "65%",
                "wind": "8 mph SW",
                "uv_index": 7
            })
    
    return json.dumps({
        "location": location,
        "temperature": "68°F",
        "conditions": "Partly cloudy",
        "note": f"Mock data for {location}"
    })

def get_uv_forecast(location: str) -> str:
    """Specialized UV forecast function."""
    return get_weather(location, "uv")

# Create agents
weather_agent = Agent(
    name="Weather Agent",
    instructions="""You are a weather specialist. You excel at providing:

• Current weather conditions
• UV index and sun safety recommendations  
• Weather forecasts and advice

When users ask about UV forecasts or sun safety, use get_uv_forecast().
For general weather, use get_weather().

Always provide actionable advice and explain UV levels:
- UV 0-2: Low (minimal protection needed)
- UV 3-5: Moderate (wear sunscreen)
- UV 6-7: High (sunscreen + seek shade midday)
- UV 8-10: Very High (extra protection needed)
- UV 11+: Extreme (avoid sun exposure)

Be thorough and helpful with your weather information.""",
    functions=[get_weather, get_uv_forecast],
)

def transfer_to_weather():
    """Transfer weather-related requests to weather specialist."""
    return weather_agent

triage_agent = Agent(
    name="Triage Agent", 
    instructions="""You are a helpful triage agent that routes users to specialists.

Available specialists:
• Weather Agent: weather, forecasts, UV index, temperature, conditions

When users ask about weather topics, immediately transfer using transfer_to_weather().

For greetings, briefly explain available services and ask how you can help.
Keep responses concise and route efficiently.""",
    functions=[transfer_to_weather],
)

def run_demo():
    """Run the improved swarm demo."""
    client = Swarm()
    
    print("=== Improved Swarm Weather Agent Demo ===\n")
    
    # Simulate the problematic conversation flow
    test_cases = [
        "Hello",
        "what specialists are there", 
        "weather",
        "Hello, whats the uv forecast for today",
        "California Santa Barbara"
    ]
    
    messages = []
    current_agent = triage_agent
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"Step {i}: User says: '{user_input}'")
        
        messages.append({"role": "user", "content": user_input})
        
        response = client.run(
            agent=current_agent,
            messages=messages,
            debug=False
        )
        
        # Update for next iteration
        messages.extend(response.messages)
        current_agent = response.agent
        
        print(f"[{current_agent.name}]: {response.messages[-1]['content']}")
        print()
    
    print("=== Conversation Complete ===")
    print(f"Final agent: {current_agent.name}")

if __name__ == "__main__":
    run_demo()