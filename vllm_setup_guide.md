# vLLM Setup and Usage Guide

This guide demonstrates how to install and use vLLM (Very Large Language Models) for serving language models with OpenAI-compatible API.

## Method 1: Installation via pip

### Prerequisites
```bash
# Create and activate virtual environment
python3 -m venv vllm_env
source vllm_env/bin/activate

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel
```

### Install vLLM
```bash
# Install vLLM with CUDA support (for GPU)
pip install vllm

# Or for CPU-only installation
pip install vllm --extra-index-url https://download.pytorch.org/whl/cpu
```

### Serve a Model
```bash
# Activate virtual environment
source vllm_env/bin/activate

# Start vLLM server with a model
vllm serve "openai/gpt-oss-120b"

# The server will start on http://localhost:8000
```

### Test the API
```bash
# Test with curl
curl -X POST "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data '{
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ]
    }'
```

## Method 2: Using Docker (Recommended)

### Prerequisites
```bash
# Install Docker
sudo apt update
sudo apt install -y docker.io

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (optional, to run without sudo)
sudo usermod -aG docker $USER
```

### Run vLLM Container
```bash
# Pull and run vLLM container with GPU support
docker run --runtime nvidia --gpus all \
    --name my_vllm_container \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<your_token_here>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model openai/gpt-oss-120b

# For CPU-only deployment (no GPU required)
docker run --name my_vllm_container \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<your_token_here>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model openai/gpt-oss-120b \
    --device cpu
```

### Alternative: Start Container and Run Model Inside
```bash
# Start container in interactive mode
docker run -it --runtime nvidia --gpus all \
    --name my_vllm_container \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<your_token_here>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    bash

# Inside the container, run:
vllm serve openai/gpt-oss-120b
```

### Test the Docker Deployment
```bash
# Test the API endpoint
curl -X POST "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data '{
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ]
    }'
```

## Python Client Example

Create a Python script to interact with the vLLM server:

```python
import openai

# Configure client to use local vLLM server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # vLLM doesn't require a real API key
)

# Make a chat completion request
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Configuration Options

### Common vLLM Server Arguments
```bash
vllm serve MODEL_NAME \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --served-model-name custom-model-name
```

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0,1  # Specify GPU devices
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

## Popular Models to Try

### Small Models (for testing)
- `microsoft/DialoGPT-medium`
- `microsoft/DialoGPT-large`
- `facebook/opt-1.3b`

### Larger Models (require more GPU memory)
- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf`
- `mistralai/Mistral-7B-Instruct-v0.1`

### Code Models
- `codellama/CodeLlama-7b-Python-hf`
- `codellama/CodeLlama-13b-Instruct-hf`

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce GPU memory utilization
   vllm serve MODEL_NAME --gpu-memory-utilization 0.7
   
   # Use smaller max model length
   vllm serve MODEL_NAME --max-model-len 2048
   ```

2. **Model Download Issues**
   ```bash
   # Set Hugging Face token
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   
   # Pre-download model
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('MODEL_NAME')"
   ```

3. **Port Already in Use**
   ```bash
   # Use different port
   vllm serve MODEL_NAME --port 8001
   ```

4. **Docker Permission Issues**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   newgrp docker
   ```

### Hardware Requirements

- **Minimum**: 8GB RAM, 4GB GPU memory
- **Recommended**: 16GB+ RAM, 8GB+ GPU memory
- **For large models**: 32GB+ RAM, 16GB+ GPU memory

## API Compatibility

vLLM provides OpenAI-compatible endpoints:

- `/v1/chat/completions` - Chat completions
- `/v1/completions` - Text completions
- `/v1/models` - List available models
- `/health` - Health check
- `/metrics` - Prometheus metrics

## Performance Tips

1. **Use appropriate tensor parallelism**
   ```bash
   # For multi-GPU setups
   vllm serve MODEL_NAME --tensor-parallel-size 2
   ```

2. **Optimize batch size**
   ```bash
   vllm serve MODEL_NAME --max-num-seqs 128
   ```

3. **Use quantization for memory efficiency**
   ```bash
   vllm serve MODEL_NAME --quantization awq
   ```

This guide provides a comprehensive overview of setting up and using vLLM for serving large language models with high performance and OpenAI-compatible APIs.