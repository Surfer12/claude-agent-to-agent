#!/usr/bin/env python3
"""
Simple Ollama Client
A Python client for interacting with the secure Ollama Docker setup.
"""

import requests
import json
import time
from typing import Dict, Any, Optional, Generator, List, Union
from dataclasses import dataclass

@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    base_url: str = "http://localhost:11434"
    timeout: int = 60
    default_model: str = "llama2:7b-q8_0"
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_max_tokens: int = 1000

class OllamaClient:
    """Client for interacting with Ollama service."""
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.session = requests.Session()
        self.session.timeout = self.config.timeout
    
    def health_check(self) -> bool:
        """Check if Ollama service is healthy."""
        try:
            response = self.session.get(f"{self.config.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            response = self.session.get(f"{self.config.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
            else:
                raise Exception(f"Failed to list models: {response.status_code}")
        except Exception as e:
            raise Exception(f"Error listing models: {e}")
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text using Ollama.
        
        Args:
            prompt: The input prompt
            model: Model name (defaults to config default)
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated text or generator for streaming
        """
        payload = {
            "model": model or self.config.default_model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature or self.config.default_temperature,
                "top_p": top_p or self.config.default_top_p,
                "num_predict": max_tokens or self.config.default_max_tokens
            }
        }
        
        try:
            response = self.session.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                stream=stream
            )
            
            if response.status_code != 200:
                raise Exception(f"Generation failed: {response.status_code}")
            
            if stream:
                return self._stream_response(response)
            else:
                data = response.json()
                return data.get('response', '')
                
        except Exception as e:
            raise Exception(f"Error generating text: {e}")
    
    def _stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """Handle streaming response."""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        yield data['response']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Chat with the model using message history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Generated response or generator for streaming
        """
        # Convert messages to prompt format
        prompt = ""
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
            elif role == 'system':
                prompt += f"System: {content}\n"
        
        prompt += "Assistant: "
        
        return self.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            stream=stream
        )

# Example usage
def main():
    """Example usage of the Ollama client."""
    print("🤖 Ollama Client Example")
    print("=" * 40)
    
    # Initialize client
    client = OllamaClient()
    
    # Health check
    if not client.health_check():
        print("❌ Ollama service is not available")
        print("Make sure to start it with: ./setup-ollama.sh docker")
        return
    
    print("✅ Ollama service is available")
    
    # List models
    try:
        models = client.list_models()
        print(f"📋 Available models: {len(models)}")
        for model in models:
            print(f"  - {model.get('name', 'Unknown')}")
    except Exception as e:
        print(f"❌ Error listing models: {e}")
    
    print()
    
    # Simple generation
    print("🧠 Testing simple generation...")
    try:
        response = client.generate(
            prompt="Write a haiku about artificial intelligence:",
            temperature=0.8,
            max_tokens=100
        )
        print(f"📝 Response: {response}")
    except Exception as e:
        print(f"❌ Generation error: {e}")
    
    print()
    
    # Chat example
    print("💬 Testing chat functionality...")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What is the population of Paris?"}
        ]
        
        response = client.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        print(f"📝 Chat response: {response}")
    except Exception as e:
        print(f"❌ Chat error: {e}")
    
    print()
    
    # Streaming example
    print("🌊 Testing streaming...")
    try:
        print("📝 Streaming response:")
        for chunk in client.generate(
            prompt="Count from 1 to 10:",
            stream=True,
            temperature=0.1
        ):
            print(chunk, end='', flush=True)
        print()  # New line after streaming
    except Exception as e:
        print(f"❌ Streaming error: {e}")

if __name__ == "__main__":
    main()