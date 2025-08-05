"""
Simple test configuration for swarm functionality.
"""

from swarm import Agent

def transfer_to_specialist():
    """Transfer to a specialist agent."""
    return specialist_agent

def transfer_to_general():
    """Transfer back to the general agent."""
    return general_agent

# General purpose agent
general_agent = Agent(
    name="General Agent",
    instructions="""You are a helpful general-purpose assistant. 
    If the user asks about technical topics, programming, or complex problems, 
    call the transfer_to_specialist function to hand them off to a specialist.
    Otherwise, help them with general questions.""",
    functions=[transfer_to_specialist],
)

# Specialist agent for technical topics
specialist_agent = Agent(
    name="Technical Specialist",
    instructions="""You are a technical specialist who helps with programming, 
    technical problems, and complex analytical tasks. 
    If the user asks about general topics or wants to go back to general help,
    call the transfer_to_general function.""",
    functions=[transfer_to_general],
)

# Default agent (this is what will be used as initial_agent)
triage_agent = general_agent
