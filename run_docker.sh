#!/bin/bash

# vLLM Docker Setup for GPT-OSS-120B
# This script runs vLLM with the GPT-OSS-120B model using Docker

echo "Starting vLLM with GPT-OSS-120B model using Docker..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q "nvidia"; then
    echo "Warning: NVIDIA Docker runtime not detected. GPU acceleration may not work."
fi

# Set Hugging Face token if provided
HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-""}

# Create cache directory
mkdir -p ~/.cache/huggingface

# Run the Docker container
echo "Pulling vLLM Docker image..."
docker pull vllm/vllm-openai:latest

echo "Starting vLLM server..."
docker run --runtime nvidia --gpus all \
    --name vllm-gpt-oss-120b \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1

echo "Server stopped."