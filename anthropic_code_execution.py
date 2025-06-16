import anthropic

client = anthropic.Anthropic()

response = client.beta.messages.create(
    model="claude-opus-4-20250514",
    betas=["code-execution-2025-05-22"],
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": "Calculate the mean and standard deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
    }],
    tools=[{
        "type": "code_execution_20250522",
        "name": "code_execution"
    }]
)

print(response)
