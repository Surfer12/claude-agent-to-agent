from swarm import Agent

# Import agents (in a real setup, these would be in separate files)
from enhanced_weather_agent import weather_agent

def transfer_to_weather():
    """Transfer to weather specialist for weather-related inquiries."""
    return weather_agent

def transfer_to_math():
    """Transfer to math specialist for calculations and math problems."""
    return math_agent

# Math agent for completeness
math_agent = Agent(
    name="Math Agent",
    instructions="""You are a math specialist. Help users with:
    - Calculations
    - Math problems
    - Mathematical concepts
    - Statistics and data analysis
    
    If users ask about non-math topics, let them know you specialize in math 
    and they should return to the triage agent for other help.""",
    functions=[]  # Add math functions as needed
)

# Enhanced triage agent with better routing logic
triage_agent = Agent(
    name="Triage Agent",
    instructions="""You are a helpful triage agent that connects users with the right specialists.

Available specialists:
- Weather Agent: For weather forecasts, UV index, temperature, conditions
- Math Agent: For calculations, math problems, statistics

When users mention weather-related terms (weather, temperature, forecast, UV, sun, rain, etc.), 
transfer them to the weather specialist using transfer_to_weather().

When users mention math-related terms (calculate, equation, numbers, statistics, etc.),
transfer them to the math specialist using transfer_to_math().

For initial greetings, explain what specialists are available and ask what they need help with.

Be concise and helpful in your routing decisions.""",
    functions=[transfer_to_weather, transfer_to_math],
)

# Example usage in a main script
if __name__ == "__main__":
    from swarm import Swarm
    
    client = Swarm()
    
    # Example conversation flow
    messages = [
        {"role": "user", "content": "Hello"},
    ]
    
    response = client.run(
        agent=triage_agent,
        messages=messages,
        debug=True
    )
    
    print("Triage Response:")
    print(response.messages[-1]["content"])
    
    # Follow up with weather request
    messages.extend(response.messages)
    messages.append({"role": "user", "content": "I need the UV forecast for Santa Barbara, California"})
    
    response = client.run(
        agent=response.agent,  # Should be triage_agent still
        messages=messages,
        debug=True
    )
    
    print("\nFinal Response:")
    print(response.messages[-1]["content"])
    print(f"Final Agent: {response.agent.name}")