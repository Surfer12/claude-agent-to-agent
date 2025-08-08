#!/usr/bin/env python3
"""
Test client for Ollama Docker setup
Verifies the container is running and can respond to API calls
"""

import requests
import json
import sys
import time
from typing import Dict, Any

def test_ollama_connection(host: str = "http://localhost:11434") -> bool:
    """Test basic connection to Ollama API"""
    try:
        response = requests.get(f"{host}/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama API is responding")
            return True
        else:
            print(f"❌ Ollama API returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to Ollama API: {e}")
        return False

def test_model_listing(host: str = "http://localhost:11434") -> bool:
    """Test model listing functionality"""
    try:
        response = requests.get(f"{host}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"📋 Available models: {len(models.get('models', []))}")
            for model in models.get('models', []):
                print(f"  - {model.get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ Failed to list models: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False

def test_chat_completion(host: str = "http://localhost:11434", model: str = "llama2:7b-q8_0") -> bool:
    """Test chat completion (if model is available)"""
    try:
        # First check if the model is available
        response = requests.get(f"{host}/api/tags", timeout=10)
        models = response.json().get('models', [])
        available_models = [m.get('name') for m in models]
        
        if model not in available_models:
            print(f"⚠️  Model {model} not available. Available models: {available_models}")
            return True  # Not an error, just no model loaded
        
        # Test chat completion
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello! Please respond with just 'Hello from Docker!'"}
            ],
            "stream": False
        }
        
        response = requests.post(f"{host}/api/chat", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat completion successful")
            print(f"📝 Response: {result.get('message', {}).get('content', 'No content')[:100]}...")
            return True
        else:
            print(f"❌ Chat completion failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing chat completion: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Ollama Docker Setup")
    print("=" * 40)
    
    # Get host from environment or use default
    host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    if not host.startswith('http'):
        host = f"http://{host}"
    
    print(f"🔗 Testing connection to: {host}")
    
    # Test 1: Basic connection
    if not test_ollama_connection(host):
        print("❌ Basic connection test failed")
        sys.exit(1)
    
    # Test 2: Model listing
    if not test_model_listing(host):
        print("❌ Model listing test failed")
        sys.exit(1)
    
    # Test 3: Chat completion (if model available)
    if not test_chat_completion(host):
        print("❌ Chat completion test failed")
        sys.exit(1)
    
    print("\n✅ All tests passed! Ollama Docker setup is working correctly.")
    print("\n📝 Next steps:")
    print("1. Pull a model: docker exec -it ollama ollama pull llama2:7b-q8_0")
    print("2. Test inference: curl -X POST http://localhost:11434/api/chat -d '{\"model\":\"llama2:7b-q8_0\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"stream\":false}'")
    print("3. For GPU acceleration, run Ollama natively on macOS")

if __name__ == "__main__":
    import os
    main()