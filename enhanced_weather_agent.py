import json
import requests
from typing import Optional
from swarm import Agent

def get_weather(location: str, weather_type: str = "current") -> str:
    """
    Get weather information for a location.
    
    Args:
        location: City and state/country (e.g., "Santa Barbara, CA")
        weather_type: Type of weather info - "current", "uv", "forecast"
    """
    # This would integrate with a real weather API like OpenWeatherMap
    # For demo purposes, returning structured mock data
    
    if "santa barbara" in location.lower():
        if weather_type.lower() == "uv":
            return json.dumps({
                "location": location,
                "uv_index": 7,
                "uv_level": "High",
                "recommendation": "Wear sunscreen, seek shade during midday hours",
                "peak_time": "12:00 PM - 2:00 PM",
                "current_time": "2:30 PM"
            })
        elif weather_type.lower() == "current":
            return json.dumps({
                "location": location,
                "temperature": "72°F",
                "conditions": "Sunny",
                "humidity": "65%",
                "wind": "8 mph SW"
            })
    
    # Default response for other locations
    return json.dumps({
        "location": location,
        "message": f"Weather data for {location} - {weather_type} information",
        "temperature": "68°F",
        "conditions": "Partly cloudy"
    })

def get_uv_forecast(location: str) -> str:
    """Get UV index forecast for a specific location."""
    return get_weather(location, "uv")

def transfer_back_to_triage():
    """Transfer back to triage if user needs help with non-weather topics."""
    from triage_agent import triage_agent  # Import when needed to avoid circular imports
    return triage_agent

# Enhanced Weather Agent with better instructions and UV capability
weather_agent = Agent(
    name="Weather Agent",
    instructions="""You are a specialized weather assistant. You can provide:

1. Current weather conditions
2. UV index and sun safety information
3. Weather forecasts
4. Location-specific weather advice

When users ask about UV forecasts or sun safety:
- Use the get_uv_forecast function for UV-specific requests
- Provide clear recommendations based on UV levels
- Include timing for peak UV exposure

For general weather: use get_weather function.

If users ask about topics outside weather (sports, news, etc.), 
use transfer_back_to_triage to send them to the appropriate specialist.

Always be helpful and provide actionable weather advice.""",
    functions=[get_weather, get_uv_forecast, transfer_back_to_triage],
)

# For real weather API integration, you'd add:
def get_real_weather_api(location: str, api_key: str) -> dict:
    """
    Example integration with OpenWeatherMap API
    """
    base_url = "http://api.openweathermap.org/data/2.5"
    
    # Current weather
    current_url = f"{base_url}/weather?q={location}&appid={api_key}&units=metric"
    
    # UV Index (requires lat/lon)
    # uv_url = f"{base_url}/uvi?lat={lat}&lon={lon}&appid={api_key}"
    
    try:
        response = requests.get(current_url)
        return response.json()
    except Exception as e:
        return {"error": str(e)}