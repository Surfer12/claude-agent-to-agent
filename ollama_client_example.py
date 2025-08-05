#!/usr/bin/env python3
"""
Ollama Docker Client Example for macOS M4 Max
============================================

This script demonstrates how to interact with the Ollama Docker container
running on macOS M4 Max with security best practices and error handling.

Requirements:
    pip install requests
    
Usage:
    python ollama_client_example.py
"""

import json
import time
import sys
import requests
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    base_url: str = "http://localhost:11434"
    timeout: int = 300  # 5 minutes for model operations
    max_retries: int = 3
    retry_delay: float = 1.0


class OllamaClient:
    """Secure client for interacting with Ollama Docker container."""
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.session = requests.Session()
        self.session.timeout = self.config.timeout
        
        # Set up session headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'OllamaClient/1.0'
        })
    
    @contextmanager
    def error_handling(self, operation: str):
        """Context manager for consistent error handling."""
        try:
            yield
        except requests.exceptions.ConnectionError:
            print(f"❌ Error: Cannot connect to Ollama at {self.config.base_url}")
            print("   Make sure the Docker container is running:")
            print("   docker ps -f name=ollama-m4max")
            sys.exit(1)
        except requests.exceptions.Timeout:
            print(f"⏰ Error: {operation} timed out after {self.config.timeout} seconds")
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            print(f"🚨 Error during {operation}: {e}")
            sys.exit(1)
    
    def health_check(self) -> bool:
        """Check if Ollama service is healthy."""
        with self.error_handling("health check"):
            response = self.session.get(f"{self.config.base_url}/api/tags")
            return response.status_code == 200
    
    def list_models(self) -> List[Dict]:
        """List all available models."""
        with self.error_handling("listing models"):
            response = self.session.get(f"{self.config.base_url}/api/tags")
            response.raise_for_status()
            return response.json().get('models', [])
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from the registry."""
        print(f"📥 Pulling model: {model_name}")
        
        with self.error_handling("pulling model"):
            data = {"name": model_name}
            response = self.session.post(
                f"{self.config.base_url}/api/pull",
                json=data,
                stream=True
            )
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        status = json.loads(line)
                        if 'status' in status:
                            print(f"   {status['status']}")
                        if 'error' in status:
                            print(f"❌ Error: {status['error']}")
                            return False
                    except json.JSONDecodeError:
                        continue
        
        print(f"✅ Successfully pulled {model_name}")
        return True
    
    def generate(self, model: str, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        with self.error_handling("generating text"):
            response = self.session.post(
                f"{self.config.base_url}/api/generate",
                json=data
            )
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
    
    def chat(self, model: str, messages: List[Dict], **kwargs) -> str:
        """Chat completion with conversation history."""
        data = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        with self.error_handling("chat completion"):
            response = self.session.post(
                f"{self.config.base_url}/api/chat",
                json=data
            )
            response.raise_for_status()
            result = response.json()
            return result.get('message', {}).get('content', '')
    
    def stream_chat(self, model: str, messages: List[Dict], **kwargs) -> Generator[str, None, None]:
        """Streaming chat completion."""
        data = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        with self.error_handling("streaming chat"):
            response = self.session.post(
                f"{self.config.base_url}/api/chat",
                json=data,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if 'message' in chunk and 'content' in chunk['message']:
                            yield chunk['message']['content']
                    except json.JSONDecodeError:
                        continue
    
    def show_model_info(self, model: str) -> Dict:
        """Get detailed information about a model."""
        data = {"name": model}
        
        with self.error_handling("getting model info"):
            response = self.session.post(
                f"{self.config.base_url}/api/show",
                json=data
            )
            response.raise_for_status()
            return response.json()


def print_banner():
    """Print application banner."""
    print("=" * 70)
    print("🚀 Ollama Docker Client for macOS M4 Max")
    print("=" * 70)
    print()


def print_system_info(client: OllamaClient):
    """Print system and container information."""
    print("📊 System Information:")
    print(f"   • Ollama URL: {client.config.base_url}")
    
    # Check health
    if client.health_check():
        print("   • Status: ✅ Healthy")
    else:
        print("   • Status: ❌ Unhealthy")
        return
    
    # List models
    models = client.list_models()
    print(f"   • Available models: {len(models)}")
    
    for model in models:
        name = model.get('name', 'Unknown')
        size = model.get('size', 0)
        size_gb = size / (1024**3) if size else 0
        print(f"     - {name} ({size_gb:.1f}GB)")
    
    print()


def demo_text_generation(client: OllamaClient, model: str):
    """Demonstrate text generation capabilities."""
    print("📝 Text Generation Demo:")
    print(f"   Using model: {model}")
    
    prompts = [
        "Explain the benefits of Apple Silicon M4 Max for AI workloads in 2 sentences:",
        "Write a Python function to calculate prime numbers:",
        "What are the key security considerations when running LLMs in Docker containers?"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n   {i}. Prompt: {prompt}")
        print("      Response: ", end="", flush=True)
        
        start_time = time.time()
        try:
            response = client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 200
                }
            )
            elapsed = time.time() - start_time
            
            # Clean up response
            response = response.strip()
            if len(response) > 200:
                response = response[:200] + "..."
            
            print(f"{response}")
            print(f"      (Generated in {elapsed:.1f}s)")
            
        except Exception as e:
            print(f"❌ Error: {e}")


def demo_chat_conversation(client: OllamaClient, model: str):
    """Demonstrate chat conversation capabilities."""
    print("\n💬 Chat Conversation Demo:")
    print(f"   Using model: {model}")
    
    conversation = [
        {"role": "user", "content": "Hello! I'm running you on a macOS M4 Max in a Docker container."},
        {"role": "assistant", "content": "Hello! That's great - the M4 Max is excellent hardware for AI workloads. How can I help you today?"},
        {"role": "user", "content": "What are the advantages of your current setup?"}
    ]
    
    print("\n   Conversation:")
    for msg in conversation[:-1]:
        role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
        print(f"   {role}: {msg['content']}")
    
    print(f"   👤 User: {conversation[-1]['content']}")
    print("   🤖 Assistant: ", end="", flush=True)
    
    start_time = time.time()
    try:
        response = client.chat(
            model=model,
            messages=conversation,
            options={
                "temperature": 0.7,
                "max_tokens": 300
            }
        )
        elapsed = time.time() - start_time
        
        print(f"{response.strip()}")
        print(f"   (Response time: {elapsed:.1f}s)")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_streaming_chat(client: OllamaClient, model: str):
    """Demonstrate streaming chat capabilities."""
    print("\n🌊 Streaming Chat Demo:")
    print(f"   Using model: {model}")
    
    messages = [
        {"role": "user", "content": "Tell me a short story about a Docker container that dreams of using GPU acceleration."}
    ]
    
    print("\n   👤 User: Tell me a short story about a Docker container that dreams of using GPU acceleration.")
    print("   🤖 Assistant: ", end="", flush=True)
    
    try:
        for chunk in client.stream_chat(model=model, messages=messages):
            print(chunk, end="", flush=True)
        print()  # New line after streaming
        
    except Exception as e:
        print(f"❌ Error: {e}")


def interactive_mode(client: OllamaClient, model: str):
    """Interactive chat mode."""
    print(f"\n💻 Interactive Mode (using {model}):")
    print("   Type 'quit' to exit, 'models' to list available models")
    print("   " + "-" * 50)
    
    conversation = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'models':
                models = client.list_models()
                print("📋 Available models:")
                for model_info in models:
                    print(f"   • {model_info.get('name', 'Unknown')}")
                continue
            
            if not user_input:
                continue
            
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})
            
            print("🤖 Assistant: ", end="", flush=True)
            
            # Get streaming response
            response_content = ""
            for chunk in client.stream_chat(model=model, messages=conversation):
                print(chunk, end="", flush=True)
                response_content += chunk
            
            print()  # New line after response
            
            # Add assistant response to conversation
            conversation.append({"role": "assistant", "content": response_content})
            
            # Keep conversation manageable (last 10 messages)
            if len(conversation) > 10:
                conversation = conversation[-10:]
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def ensure_model_available(client: OllamaClient, model: str) -> bool:
    """Ensure a model is available, pull if necessary."""
    models = client.list_models()
    available_models = [m.get('name', '') for m in models]
    
    if model not in available_models:
        print(f"📥 Model '{model}' not found locally. Available models:")
        for available in available_models:
            print(f"   • {available}")
        
        if not available_models:
            print("\n🔄 No models available. Pulling recommended model...")
            return client.pull_model(model)
        else:
            response = input(f"\nWould you like to pull '{model}'? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                return client.pull_model(model)
            else:
                return False
    
    return True


def main():
    """Main application entry point."""
    print_banner()
    
    # Initialize client
    client = OllamaClient()
    
    # Print system information
    print_system_info(client)
    
    # Recommended model for M4 Max (good balance of performance and quality)
    recommended_model = "llama2:7b-q4_0"
    
    # Ensure model is available
    if not ensure_model_available(client, recommended_model):
        print("❌ Cannot proceed without a model. Exiting.")
        sys.exit(1)
    
    # Run demonstrations
    try:
        demo_text_generation(client, recommended_model)
        demo_chat_conversation(client, recommended_model)
        demo_streaming_chat(client, recommended_model)
        
        # Interactive mode
        response = input("\n🎮 Enter interactive mode? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_mode(client, recommended_model)
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()