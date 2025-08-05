#!/usr/bin/env python3
"""
Demo script to test swarm functionality without requiring API keys.
"""

import sys
import os

# Add swarm to path
sys.path.insert(0, 'swarm')

from swarm import Swarm, Agent

def transfer_to_specialist():
    """Transfer to a specialist agent."""
    print("🔄 Transferring to Technical Specialist...")
    return specialist_agent

def transfer_to_general():
    """Transfer back to the general agent."""
    print("🔄 Transferring back to General Agent...")
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

def demo_swarm_structure():
    """Demonstrate the swarm structure without making API calls."""
    print("🐝 Swarm Demo - Agent Structure")
    print("=" * 50)
    
    print(f"📋 General Agent:")
    print(f"   Name: {general_agent.name}")
    print(f"   Functions: {[f.__name__ for f in general_agent.functions]}")
    print()
    
    print(f"🔧 Specialist Agent:")
    print(f"   Name: {specialist_agent.name}")
    print(f"   Functions: {[f.__name__ for f in specialist_agent.functions]}")
    print()
    
    print("🔄 Transfer Functions:")
    print("   - transfer_to_specialist(): Routes technical questions to specialist")
    print("   - transfer_to_general(): Routes general questions back to general agent")
    print()
    
    print("💡 How it works:")
    print("   1. User starts with General Agent")
    print("   2. If technical question → transfers to Specialist")
    print("   3. If general question → transfers back to General")
    print("   4. Each agent has specific expertise and can hand off appropriately")
    print()
    
    print("🚀 To test with real API:")
    print("   1. Set OPENAI_API_KEY environment variable")
    print("   2. Run: python -m unified_agent.cli --swarm-config test_swarm_config.py --initial-agent triage_agent")

if __name__ == "__main__":
    demo_swarm_structure()
