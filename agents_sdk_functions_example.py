#!/usr/bin/env python3
"""
OpenAI Agents SDK - Functions/Tools Example

This demonstrates how to use function tools with agents.
This replaces Swarm's function calling capabilities.
"""

import os
import asyncio
from agents import Agent, Runner, function_tool

# Define function tools using the decorator
@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In a real implementation, you'd call a weather API
    weather_data = {
        "tokyo": "sunny, 22°C",
        "london": "cloudy, 15°C", 
        "new york": "rainy, 18°C",
        "paris": "partly cloudy, 20°C",
        "sydney": "sunny, 25°C"
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"The weather in {city} is {weather_data[city_lower]}."
    else:
        return f"Sorry, I don't have weather data for {city}. Try Tokyo, London, New York, Paris, or Sydney."

@function_tool
def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> str:
    """Calculate tip amount and total bill."""
    tip_amount = bill_amount * (tip_percentage / 100)
    total_amount = bill_amount + tip_amount
    return f"Bill: ${bill_amount:.2f}, Tip ({tip_percentage}%): ${tip_amount:.2f}, Total: ${total_amount:.2f}"

@function_tool
def convert_temperature(temperature: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin."""
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    # Convert to Celsius first
    if from_unit == "fahrenheit" or from_unit == "f":
        celsius = (temperature - 32) * 5/9
    elif from_unit == "kelvin" or from_unit == "k":
        celsius = temperature - 273.15
    elif from_unit == "celsius" or from_unit == "c":
        celsius = temperature
    else:
        return f"Unknown unit: {from_unit}. Use Celsius, Fahrenheit, or Kelvin."
    
    # Convert from Celsius to target unit
    if to_unit == "fahrenheit" or to_unit == "f":
        result = celsius * 9/5 + 32
        unit_symbol = "°F"
    elif to_unit == "kelvin" or to_unit == "k":
        result = celsius + 273.15
        unit_symbol = "K"
    elif to_unit == "celsius" or to_unit == "c":
        result = celsius
        unit_symbol = "°C"
    else:
        return f"Unknown unit: {to_unit}. Use Celsius, Fahrenheit, or Kelvin."
    
    return f"{temperature}° {from_unit} = {result:.2f}{unit_symbol}"

def main():
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create agent with multiple tools
    agent = Agent(
        name="Multi-Tool Assistant",
        instructions="""You are a helpful assistant with access to several tools:
        - Weather information for major cities
        - Tip calculator for restaurant bills
        - Temperature converter between different units
        
        Use these tools when appropriate to help users with their requests.""",
        tools=[get_weather, calculate_tip, convert_temperature],
    )

    async def run_examples():
        print("🤖 Testing Functions/Tools Example")
        print("=" * 50)
        
        # Test weather function
        print("\n1. Testing weather function:")
        result = await Runner.run(agent, input="What's the weather in Tokyo?")
        print(f"Response: {result.final_output}")
        
        # Test tip calculator
        print("\n2. Testing tip calculator:")
        result = await Runner.run(agent, input="Calculate a 20% tip on a $85.50 bill")
        print(f"Response: {result.final_output}")
        
        # Test temperature converter
        print("\n3. Testing temperature converter:")
        result = await Runner.run(agent, input="Convert 100 degrees Fahrenheit to Celsius")
        print(f"Response: {result.final_output}")
        
        # Test multiple tools in one conversation
        print("\n4. Testing multiple tools:")
        result = await Runner.run(agent, input="What's the weather in London, and if I have dinner there costing £45, what would a 15% tip be in pounds?")
        print(f"Response: {result.final_output}")

    try:
        asyncio.run(run_examples())
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your OPENAI_API_KEY is set correctly.")

if __name__ == "__main__":
    main()