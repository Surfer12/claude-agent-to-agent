#!/usr/bin/env python3

import sys
import subprocess
import time
import requests
import json

def test_vllm_installation():
    """Test if vLLM is properly installed and can be imported"""
    try:
        import vllm
        print("✓ vLLM is installed and can be imported")
        return True
    except ImportError as e:
        print(f"✗ vLLM import failed: {e}")
        return False

def test_model_serving():
    """Test if we can serve a small model"""
    try:
        # Try to import and use vLLM
        from vllm import LLM, SamplingParams
        
        print("Testing vLLM with a small model...")
        
        # Use a smaller model for testing
        model_name = "microsoft/DialoGPT-medium"  # Much smaller than GPT-OSS-120B
        
        print(f"Loading model: {model_name}")
        llm = LLM(model=model_name)
        
        # Test sampling
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        outputs = llm.generate(["Hello, how are you?"], sampling_params)
        
        print("✓ vLLM is working correctly!")
        print(f"Generated response: {outputs[0].outputs[0].text}")
        return True
        
    except Exception as e:
        print(f"✗ vLLM test failed: {e}")
        return False

def main():
    print("Testing vLLM installation...")
    
    if not test_vllm_installation():
        print("vLLM installation test failed")
        return False
    
    if not test_model_serving():
        print("vLLM model serving test failed")
        return False
    
    print("\n✓ All tests passed! vLLM is ready to use.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)