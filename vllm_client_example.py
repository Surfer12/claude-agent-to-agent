#!/usr/bin/env python3
"""
vLLM Client Example

This script demonstrates how to interact with a vLLM server using the OpenAI client library.
Make sure you have a vLLM server running before executing this script.

Usage:
    python vllm_client_example.py
"""

import openai
import json
import sys
from typing import List, Dict, Any

# Configuration
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "openai/gpt-oss-120b"  # Change this to your served model name

def create_client() -> openai.OpenAI:
    """Create and return an OpenAI client configured for vLLM."""
    return openai.OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="dummy-key"  # vLLM doesn't require a real API key
    )

def test_chat_completion(client: openai.OpenAI, messages: List[Dict[str, str]]) -> None:
    """Test chat completion endpoint."""
    print("🤖 Testing Chat Completion...")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,
            temperature=0.7,
            stream=False
        )
        
        print(f"✅ Response: {response.choices[0].message.content}")
        print(f"📊 Usage: {response.usage}")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("-" * 50)

def test_streaming_completion(client: openai.OpenAI, messages: List[Dict[str, str]]) -> None:
    """Test streaming chat completion."""
    print("🌊 Testing Streaming Chat Completion...")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,
            temperature=0.7,
            stream=True
        )
        
        print("✅ Streaming response:")
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print(f"\n📝 Full response: {full_response}")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("-" * 50)

def test_text_completion(client: openai.OpenAI, prompt: str) -> None:
    """Test text completion endpoint (if supported)."""
    print("📝 Testing Text Completion...")
    print(f"Prompt: {prompt}")
    
    try:
        response = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
            stop=["\n\n"]
        )
        
        print(f"✅ Response: {response.choices[0].text}")
        print(f"📊 Usage: {response.usage}")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("-" * 50)

def list_models(client: openai.OpenAI) -> None:
    """List available models."""
    print("📋 Listing Available Models...")
    
    try:
        models = client.models.list()
        for model in models.data:
            print(f"✅ Model: {model.id}")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("-" * 50)

def check_server_health() -> None:
    """Check if the vLLM server is running."""
    import requests
    
    print("🔍 Checking Server Health...")
    
    try:
        response = requests.get(f"{VLLM_BASE_URL.replace('/v1', '')}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy!")
        else:
            print(f"⚠️ Server returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure vLLM server is running on http://localhost:8000")
        return False
    
    print("-" * 50)
    return True

def main():
    """Main function to run all tests."""
    print("🚀 vLLM Client Example")
    print("=" * 50)
    
    # Check server health first
    if not check_server_health():
        sys.exit(1)
    
    # Create client
    client = create_client()
    
    # List available models
    list_models(client)
    
    # Test cases
    test_cases = [
        {
            "name": "Simple Question",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ]
        },
        {
            "name": "Conversation",
            "messages": [
                {"role": "user", "content": "Hello! Can you help me with Python programming?"},
                {"role": "assistant", "content": "Of course! I'd be happy to help you with Python programming. What specific topic or question do you have?"},
                {"role": "user", "content": "How do I create a simple web server in Python?"}
            ]
        },
        {
            "name": "Code Generation",
            "messages": [
                {"role": "user", "content": "Write a Python function that calculates the factorial of a number using recursion."}
            ]
        }
    ]
    
    # Run chat completion tests
    for test_case in test_cases:
        print(f"🧪 Test Case: {test_case['name']}")
        test_chat_completion(client, test_case["messages"])
    
    # Test streaming with the first test case
    print(f"🧪 Streaming Test: {test_cases[0]['name']}")
    test_streaming_completion(client, test_cases[0]["messages"])
    
    # Test text completion
    test_text_completion(client, "The benefits of artificial intelligence include")
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main()