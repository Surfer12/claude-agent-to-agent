"""
Simple Swarm agents configuration for testing
"""

from swarm import Agent

def transfer_to_math_agent():
    """Transfer to the math specialist agent"""
    return math_agent

def transfer_to_weather_agent():
    """Transfer to the weather specialist agent"""  
    return weather_agent

def transfer_to_triage():
    """Transfer back to the triage agent"""
    return triage_agent

# Triage agent that routes users to specialists
triage_agent = Agent(
    name="Triage Agent",
    instructions="""You are a helpful triage agent that routes users to the right specialist.
    
    If the user asks about:
    - Math, calculations, or numbers: transfer to math agent
    - Weather, temperature, or forecast: transfer to weather agent
    
    Otherwise, try to help them yourself or ask clarifying questions.""",
    functions=[transfer_to_math_agent, transfer_to_weather_agent]
)

# Math specialist agent
math_agent = Agent(
    name="Math Agent", 
    instructions="""You are a math specialist. You excel at:
    - Calculations
    - Mathematical explanations
    - Problem solving
    - Statistical analysis
    
    If the user asks about something unrelated to math, transfer them back to triage.""",
    functions=[transfer_to_triage]
)

# Weather specialist agent  
weather_agent = Agent(
    name="Weather Agent",
    instructions="""You are a weather specialist. You help with:
    - Weather forecasts
    - Temperature questions
    - Climate information
    - Weather-related planning
    
    If the user asks about something unrelated to weather, transfer them back to triage.""",
    functions=[transfer_to_triage]
)