import anthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_basic_message():
    """Create a basic message using the Anthropic API."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    message = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, Claude"}
        ]
    )
    return message

def create_conversation():
    """Create a multi-turn conversation."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    message = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, Claude"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Can you describe LLMs to me?"}
        ]
    )
    return message

def create_message_with_tools():
    """Create a message with tool usage."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    weather_tool = {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
    
    message = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=1024,
        tools=[weather_tool],
        messages=[
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ]
    )
    return message

def main():
    # Example 1: Basic message
    print("Example 1: Basic Message")
    message = create_basic_message()
    print(message)
    print("\n" + "="*50 + "\n")
    
    # Example 2: Conversation
    print("Example 2: Conversation")
    conversation = create_conversation()
    print(conversation)
    print("\n" + "="*50 + "\n")
    
    # Example 3: Message with tools
    print("Example 3: Message with Tools")
    tool_message = create_message_with_tools()
    print(tool_message)

if __name__ == "__main__":
    main() 