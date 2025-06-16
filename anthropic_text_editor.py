import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=1024,
    tools=[
        {
            "type": "text_editor_20250429",
            "name": "str_replace_based_edit_tool"
        }
    ],
    messages=[
        {
            "role": "user", 
            "content": "There's a syntax error in my primes.py file. Can you help me fix it?"
        }
    ]
)

print(response)
