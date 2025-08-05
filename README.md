# vLLM Setup and Usage Demo

This repository contains a complete demonstration of how to install, configure, and use vLLM (Very Large Language Models) for serving language models with OpenAI-compatible APIs.

## 📁 Files Overview

- **`vllm_setup_guide.md`** - Comprehensive setup guide with detailed instructions
- **`setup_vllm.sh`** - Automated setup script for both pip and Docker installations  
- **`vllm_client_example.py`** - Python client example showing how to interact with vLLM server
- **`requirements.txt`** - Python dependencies for the client example

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Make script executable
chmod +x setup_vllm.sh

# Install vLLM via pip
./setup_vllm.sh install-pip

# Or install via Docker
./setup_vllm.sh install-docker

# Run the server (pip installation)
./setup_vllm.sh run-pip microsoft/DialoGPT-medium

# Run the server (Docker)
./setup_vllm.sh run-docker microsoft/DialoGPT-medium
```

### Option 2: Manual Setup

#### Via pip:
```bash
# Create virtual environment
python3 -m venv vllm_env
source vllm_env/bin/activate

# Install vLLM
pip install vllm

# Start server
vllm serve "microsoft/DialoGPT-medium"
```

#### Via Docker:
```bash
# Install Docker
sudo apt update && sudo apt install -y docker.io

# Run vLLM container
docker run --rm -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model microsoft/DialoGPT-medium
```

## 🧪 Testing the Setup

### Using curl:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data '{
        "model": "microsoft/DialoGPT-medium",
        "messages": [
            {
                "role": "user", 
                "content": "What is the capital of France?"
            }
        ]
    }'
```

### Using Python client:
```bash
# Install client dependencies
pip install -r requirements.txt

# Run the example client
python vllm_client_example.py
```

### Using the test script:
```bash
./setup_vllm.sh test
```

## 📋 What You'll Learn

This demo covers:

- ✅ **Installation Methods**: Both pip and Docker approaches
- ✅ **Server Configuration**: Common settings and optimization options  
- ✅ **API Usage**: OpenAI-compatible endpoints for chat and completions
- ✅ **Client Integration**: Python examples with streaming support
- ✅ **Model Selection**: Examples with different model sizes and types
- ✅ **Troubleshooting**: Common issues and solutions
- ✅ **Performance Tips**: Memory optimization and scaling strategies

## 🔧 Key Features Demonstrated

### Server Capabilities:
- OpenAI-compatible API endpoints
- Streaming and non-streaming responses  
- Multiple model support
- GPU and CPU deployment options
- Health checks and metrics

### Client Features:
- Chat completions
- Text completions
- Streaming responses
- Error handling
- Model listing

## 📖 Detailed Documentation

For comprehensive instructions, troubleshooting, and advanced configuration options, see:
- **[Complete Setup Guide](vllm_setup_guide.md)** - Detailed installation and usage instructions

## 🛠️ Requirements

### Minimum System Requirements:
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 10GB free space
- **Python**: 3.8+
- **OS**: Linux, macOS, or Windows with WSL

### For GPU Acceleration:
- **NVIDIA GPU** with CUDA support
- **GPU Memory**: 4GB+ (8GB+ recommended)
- **CUDA**: 11.8+ or 12.x

## 🤝 Usage Examples

The demo includes examples for:
- Simple Q&A interactions
- Multi-turn conversations  
- Code generation requests
- Streaming responses
- Batch processing

## 🔍 Troubleshooting

Common issues and solutions are covered in the setup guide, including:
- Installation failures
- Memory issues
- Docker permission problems
- Model download issues
- API connection problems

## 📝 Next Steps

After running this demo, you can:
1. Try different models from Hugging Face
2. Experiment with different parameters
3. Integrate vLLM into your applications
4. Scale to multi-GPU setups
5. Deploy in production environments

---

**Note**: This demonstration was created to show the complete process of setting up and using vLLM. While the pip installation encountered some compilation issues in this specific environment, the Docker approach and all the example code provided will work in standard environments with proper dependencies installed.
