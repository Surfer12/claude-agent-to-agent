#!/bin/bash

# Example curl command to test the vLLM server
echo "Testing vLLM GPT-OSS-120B server..."

curl -X POST "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data '{
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        "max_tokens": 100
    }'

echo ""
echo "Test completed!"