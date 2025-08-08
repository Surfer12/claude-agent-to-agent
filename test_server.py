#!/usr/bin/env python3
"""
Test script for vLLM GPT-OSS-120B server
"""

import requests
import json
import time
import sys

def test_server(port=8000, max_retries=5):
    """Test the vLLM server"""
    
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # Test data
    test_cases = [
        {
            "name": "Simple question",
            "data": {
                "model": "openai/gpt-oss-120b",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of France?"
                    }
                ],
                "max_tokens": 50
            }
        },
        {
            "name": "Math problem",
            "data": {
                "model": "openai/gpt-oss-120b",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 2 + 2?"
                    }
                ],
                "max_tokens": 30
            }
        },
        {
            "name": "Creative writing",
            "data": {
                "model": "openai/gpt-oss-120b",
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a short poem about AI."
                    }
                ],
                "max_tokens": 100
            }
        }
    ]
    
    print(f"Testing vLLM server at http://localhost:{port}")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 30)
        
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}...")
                
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=test_case['data'], 
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    print(f"✓ Success!")
                    print(f"Response: {content}")
                    break
                else:
                    print(f"✗ Failed: HTTP {response.status_code}")
                    print(f"Error: {response.text}")
                    if attempt < max_retries - 1:
                        print("Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        print("Max retries reached.")
                        
            except requests.exceptions.ConnectionError:
                print(f"✗ Connection error (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("Server not responding. Make sure the server is running.")
                    return False
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                if attempt < max_retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("Max retries reached.")
    
    print("\n" + "=" * 50)
    print("✓ All tests completed!")
    return True

def check_server_status(port=8000):
    """Check if server is running"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    port = 8000
    
    print("vLLM GPT-OSS-120B Server Test")
    print("=" * 40)
    
    # Check if server is running
    if not check_server_status(port):
        print(f"❌ Server not running on port {port}")
        print("Please start the server first using one of these methods:")
        print("1. Python: python run_vllm_server.py")
        print("2. Docker: ./run_docker.sh")
        print("3. Direct: vllm serve openai/gpt-oss-120b")
        return False
    
    print(f"✅ Server is running on port {port}")
    
    # Run tests
    success = test_server(port)
    
    if success:
        print("\n🎉 All tests passed! The server is working correctly.")
        print("\nYou can now use the server with curl:")
        print("curl -X POST http://localhost:8000/v1/chat/completions \\")
        print("  -H 'Content-Type: application/json' \\")
        print("  --data '{\"model\": \"openai/gpt-oss-120b\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'")
    else:
        print("\n❌ Some tests failed. Please check the server logs.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)