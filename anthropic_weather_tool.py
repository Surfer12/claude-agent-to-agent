import anthropic

client = anthropic.Anthropic()

response = client.beta.messages.create(
    max_tokens=1024,
    model="claude-3-7-sonnet-20250219",
    tools=[{
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
        "required": [
          "location"
        ]
      }
    }],
    messages=[{
      "role": "user",
      "content": "Tell me the weather in San Francisco."
    }],
    betas=["token-efficient-tools-2025-02-19"]
)

print(response.usage)
