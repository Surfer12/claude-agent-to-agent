import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "How do I update a web app to TypeScript 5.5?"
        }
    ],
    tools=[{
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5
    }]
)
print(response)
