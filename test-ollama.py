#!/usr/bin/env python3
"""
Ollama Setup Test Script
Tests the secure Ollama Docker setup for basic functionality.
"""

import requests
import json
import time
import sys
from typing import Dict, Any, Optional

class OllamaTester:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
    
    def test_connection(self) -> bool:
        """Test basic connection to Ollama service."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                print("✅ Connection successful")
                return True
            else:
                print(f"❌ Connection failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def list_models(self) -> Optional[Dict[str, Any]]:
        """List available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                print(f"📋 Available models: {len(models.get('models', []))}")
                for model in models.get('models', []):
                    print(f"  - {model.get('name', 'Unknown')}")
                return models
            else:
                print(f"❌ Failed to list models: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return None
    
    def test_inference(self, model_name: str = "llama2:7b-q8_0") -> bool:
        """Test basic inference with a simple prompt."""
        payload = {
            "model": model_name,
            "prompt": "Hello! Please respond with just 'Hello from Ollama!'",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 50
            }
        }
        
        try:
            print(f"🧠 Testing inference with {model_name}...")
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                print(f"✅ Inference successful!")
                print(f"📝 Response: {response_text.strip()}")
                return True
            else:
                print(f"❌ Inference failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Inference timed out (60s)")
            return False
        except Exception as e:
            print(f"❌ Inference error: {e}")
            return False
    
    def test_streaming(self, model_name: str = "llama2:7b-q8_0") -> bool:
        """Test streaming inference."""
        payload = {
            "model": model_name,
            "prompt": "Count from 1 to 5:",
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_predict": 20
            }
        }
        
        try:
            print(f"🌊 Testing streaming inference with {model_name}...")
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                print("✅ Streaming successful!")
                print("📝 Streamed response:")
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                print(data['response'], end='', flush=True)
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                print()  # New line after streaming
                return True
            else:
                print(f"❌ Streaming failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            return False
    
    def test_health(self) -> Dict[str, Any]:
        """Test overall health of the Ollama service."""
        health_status = {
            "connection": False,
            "models_available": False,
            "inference_working": False,
            "streaming_working": False
        }
        
        print("🔍 Testing Ollama service health...")
        print("=" * 50)
        
        # Test connection
        health_status["connection"] = self.test_connection()
        
        if health_status["connection"]:
            # List models
            models = self.list_models()
            health_status["models_available"] = models is not None and len(models.get('models', [])) > 0
            
            if health_status["models_available"]:
                # Test inference
                health_status["inference_working"] = self.test_inference()
                
                # Test streaming
                health_status["streaming_working"] = self.test_streaming()
        
        print("=" * 50)
        print("📊 Health Summary:")
        for test, status in health_status.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {test.replace('_', ' ').title()}")
        
        return health_status

def main():
    """Main test function."""
    print("🚀 Ollama Secure Setup Test")
    print("=" * 50)
    
    # Check if Docker container is running
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=ollama", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10
        )
        if "ollama" in result.stdout:
            print("✅ Ollama container is running")
        else:
            print("⚠️  Ollama container not found. Make sure it's running with:")
            print("   ./setup-ollama.sh docker")
            print("   or")
            print("   ./setup-ollama.sh compose")
    except Exception as e:
        print(f"⚠️  Could not check Docker container: {e}")
    
    print()
    
    # Run tests
    tester = OllamaTester()
    health_status = tester.test_health()
    
    # Summary
    print()
    print("🎯 Test Summary:")
    all_passed = all(health_status.values())
    
    if all_passed:
        print("🎉 All tests passed! Your Ollama setup is working correctly.")
        print()
        print("💡 Next steps:")
        print("   - Pull more models: ./setup-ollama.sh pull codellama:7b-instruct")
        print("   - Check status: ./setup-ollama.sh status")
        print("   - View logs: docker logs ollama")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print()
        print("🔧 Troubleshooting:")
        print("   - Check container status: ./setup-ollama.sh status")
        print("   - View container logs: docker logs ollama")
        print("   - Restart container: ./setup-ollama.sh restart")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())