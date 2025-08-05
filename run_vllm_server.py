#!/usr/bin/env python3
"""
vLLM Server for GPT-OSS-120B Model

This script runs a vLLM server with the GPT-OSS-120B model.
Make sure you have sufficient GPU memory (at least 240GB for the full model).
"""

import os
import sys
import subprocess
import time
import requests
import json
from pathlib import Path

def check_gpu_memory():
    """Check available GPU memory"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"Found {gpu_count} GPU(s)")
            for i in range(gpu_count):
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {memory:.1f} GB")
            return True
        else:
            print("No CUDA GPUs available")
            return False
    except ImportError:
        print("PyTorch not available")
        return False

def run_vllm_server(model_name="openai/gpt-oss-120b", port=8000):
    """Run the vLLM server"""
    print(f"Starting vLLM server with model: {model_name}")
    print(f"Server will be available at: http://localhost:{port}")
    
    # Command to run vLLM server
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", "1"  # Adjust based on your GPU setup
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the server
        process = subprocess.Popen(cmd)
        print(f"Server started with PID: {process.pid}")
        
        # Wait a bit for the server to start
        time.sleep(10)
        
        # Test the server
        test_server(port)
        
        print("\nServer is running. Press Ctrl+C to stop.")
        process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping server...")
        if 'process' in locals():
            process.terminate()
            process.wait()
        print("Server stopped.")

def test_server(port=8000):
    """Test the server with a simple request"""
    try:
        url = f"http://localhost:{port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "openai/gpt-oss-120b",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ],
            "max_tokens": 50
        }
        
        print(f"Testing server at {url}...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Server is working!")
            print(f"Response: {result['choices'][0]['message']['content']}")
        else:
            print(f"✗ Server test failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"✗ Server test failed: {e}")

def create_curl_example():
    """Create an example curl command"""
    curl_example = '''# Example curl command to test the server:
curl -X POST "http://localhost:8000/v1/chat/completions" \\
    -H "Content-Type: application/json" \\
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
'''
    
    with open("curl_example.sh", "w") as f:
        f.write(curl_example)
    
    print("Created curl_example.sh with test command")

def main():
    print("vLLM GPT-OSS-120B Server Setup")
    print("=" * 40)
    
    # Check GPU
    if not check_gpu_memory():
        print("Warning: No GPU detected. The model may not run efficiently on CPU.")
    
    # Create curl example
    create_curl_example()
    
    # Run server
    run_vllm_server()

if __name__ == "__main__":
    main()