import anthropic

client = anthropic.Anthropic()

response = client.messages.stream(
    max_tokens=65536,
    model="claude-sonnet-4-20250514",
    tools=[{
      "name": "make_file",
      "description": "Write text to a file",
      "input_schema": {
        "type": "object",
        "properties": {
          "filename": {
            "type": "string",
            "description": "The filename to write text to"
          },
          "lines_of_text": {
            "type": "array",
            "description": "An array of lines of text to write to the file"
          }
        },
        "required": ["filename", "lines_of_text"]
      }
    }],
    messages=[{
      "role": "user",
      "content": "Can you write a long poem and make a file called poem.txt?"
    }],
    betas=["fine-grained-tool-streaming-2025-05-14"]
)

print(response.usage)
