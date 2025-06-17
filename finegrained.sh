curl https://api.anthropic.com/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: fine-grained-tool-streaming-2025-05-14" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 65536,
    "tools": [
      {
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
      }
    ],
    "messages": [
      {
        "role": "user",
        "content": "Can you write a long poem and make a file called poem.txt?"
      }
    ],
    "stream": true
  }' | jq '.usage'