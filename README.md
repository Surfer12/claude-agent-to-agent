# vLLM GPT-OSS-120B Setup

This repository contains scripts and configurations to run the GPT-OSS-120B model using vLLM.

## Prerequisites

### Hardware Requirements
- **GPU Memory**: At least 240GB of GPU memory for the full model
- **Multiple GPUs**: Recommended for optimal performance
- **RAM**: At least 512GB of system RAM
- **Storage**: At least 500GB of free space for model weights

### Software Requirements
- Python 3.8+
- CUDA 11.8+
- Docker (optional, for containerized deployment)

## Installation

### Method 1: Python Installation (Recommended)

1. **Create a virtual environment**:
   ```bash
   python3 -m venv vllm_env
   source vllm_env/bin/activate
   ```

2. **Install PyTorch**:
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **Install vLLM**:
   ```bash
   pip install vllm --no-deps
   pip install transformers huggingface-hub fastapi uvicorn
   ```

4. **Install additional dependencies**:
   ```bash
   pip install msgspec aiohttp blake3 cachetools cloudpickle compressed-tensors depyf einops gguf importlib_metadata lark llguidance lm-format-enforcer mistral_common opencv-python-headless outlines partial-json-parser prometheus_client prometheus-fastapi-instrumentator protobuf psutil py-cpuinfo python-json-logger pyzmq ray scipy sentencepiece six tiktoken watchfiles xgrammar
   ```

### Method 2: Docker Installation

1. **Install Docker and NVIDIA Docker runtime**:
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Install NVIDIA Docker runtime
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

## Usage

### Method 1: Python Script

1. **Activate the virtual environment**:
   ```bash
   source vllm_env/bin/activate
   ```

2. **Run the server**:
   ```bash
   python run_vllm_server.py
   ```

### Method 2: Docker

1. **Make the script executable**:
   ```bash
   chmod +x run_docker.sh
   ```

2. **Run with Docker**:
   ```bash
   ./run_docker.sh
   ```

### Method 3: Direct vLLM Command

```bash
# Activate virtual environment
source vllm_env/bin/activate

# Run vLLM server directly
vllm serve openai/gpt-oss-120b --host 0.0.0.0 --port 8000
```

## Testing the Server

### Using the Test Script

```bash
# Activate virtual environment (if using Python method)
source vllm_env/bin/activate

# Run the test script
python test_server.py
```

### Using curl

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
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
```

### Using Python Requests

```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "openai/gpt-oss-120b",
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "max_tokens": 100
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(result['choices'][0]['message']['content'])
```

## Configuration Options

### GPU Configuration

For multi-GPU setups, adjust the `--tensor-parallel-size` parameter:

```bash
# For 4 GPUs
vllm serve openai/gpt-oss-120b --tensor-parallel-size 4

# For 8 GPUs
vllm serve openai/gpt-oss-120b --tensor-parallel-size 8
```

### Memory Optimization

For limited GPU memory, you can use quantization:

```bash
# Use 4-bit quantization
vllm serve openai/gpt-oss-120b --quantization awq

# Use 8-bit quantization
vllm serve openai/gpt-oss-120b --quantization gptq
```

### Hugging Face Token

If the model requires authentication, set your Hugging Face token:

```bash
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM) Error**:
   - Reduce `--tensor-parallel-size`
   - Use quantization
   - Increase GPU memory
   - Use CPU offloading

2. **Model Download Issues**:
   - Check internet connection
   - Verify Hugging Face token
   - Clear cache: `rm -rf ~/.cache/huggingface`

3. **CUDA Errors**:
   - Verify CUDA installation
   - Check GPU drivers
   - Ensure NVIDIA Docker runtime is installed

4. **Port Already in Use**:
   - Change port: `--port 8001`
   - Kill existing process: `lsof -ti:8000 | xargs kill -9`

### Performance Optimization

1. **Multi-GPU Setup**:
   ```bash
   vllm serve openai/gpt-oss-120b --tensor-parallel-size 4 --gpu-memory-utilization 0.9
   ```

2. **Memory Optimization**:
   ```bash
   vllm serve openai/gpt-oss-120b --max-model-len 4096 --gpu-memory-utilization 0.8
   ```

3. **Batch Processing**:
   ```bash
   vllm serve openai/gpt-oss-120b --max-num-batched-tokens 4096
   ```

## API Endpoints

The server provides OpenAI-compatible API endpoints:

- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions
- `GET /health` - Health check
- `GET /models` - List available models

## Monitoring

### Logs
- Check server logs for errors and performance metrics
- Monitor GPU memory usage: `nvidia-smi`
- Monitor system resources: `htop`

### Metrics
The server exposes Prometheus metrics at `/metrics` for monitoring.

## Security Considerations

1. **Network Security**:
   - Use firewall rules
   - Restrict access to trusted IPs
   - Use HTTPS in production

2. **Resource Limits**:
   - Set memory limits
   - Monitor usage
   - Implement rate limiting

3. **Authentication**:
   - Implement API key authentication
   - Use reverse proxy with auth

## Production Deployment

For production deployment, consider:

1. **Load Balancing**: Use multiple server instances
2. **Monitoring**: Implement comprehensive monitoring
3. **Backup**: Regular model and configuration backups
4. **Updates**: Plan for model and vLLM updates
5. **Scaling**: Auto-scaling based on demand

## License

This project is licensed under the MIT License. The GPT-OSS-120B model has its own license - please check the model's license on Hugging Face.

## Support

For issues and questions:
1. Check the vLLM documentation
2. Review the troubleshooting section
3. Check GPU memory and system resources
4. Verify all dependencies are installed correctly
