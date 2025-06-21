#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install anthropic httpx

# Set up environment variables
echo "export ANTHROPIC_API_KEY='your-api-key-here'" >> .env

# Create necessary directories
mkdir -p src/main/java/com/anthropic/api
mkdir -p src/test/java/com/anthropic/api

echo "Environment setup complete. Please add your Anthropic API key to the .env file." 