#!/bin/bash

# Setup script for OpenAI Agents SDK migration
# This script helps migrate from Swarm to the new OpenAI Agents SDK

echo "🚀 Setting up OpenAI Agents SDK Migration"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python version: $PYTHON_VERSION"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv agents_sdk_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source agents_sdk_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install OpenAI Agents SDK
echo "📥 Installing OpenAI Agents SDK..."
pip install -r requirements_agents_sdk.txt

# Install optional voice support (uncomment if needed)
# echo "🎤 Installing voice support..."
# pip install 'openai-agents[voice]'

# Test installation
echo "🧪 Testing installation..."
python3 -c "from agents import Agent, Runner; print('✅ OpenAI Agents SDK installed successfully!')"

# Create .env template
echo "📝 Creating environment template..."
cat > .env.template << EOF
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom model configuration
# OPENAI_MODEL=gpt-4-turbo-preview

# Optional: Tracing configuration
# TRACING_ENABLED=true
# TRACING_PROVIDER=logfire  # or agentops, braintrust, scorecard, keywords_ai
EOF

echo ""
echo "🎉 Setup complete!"
echo "=================="
echo ""
echo "Next steps:"
echo "1. Copy .env.template to .env and add your OpenAI API key"
echo "2. Run: source agents_sdk_env/bin/activate"
echo "3. Test the migration: python3 migration_to_agents_sdk.py"
echo ""
echo "Key benefits of the migration:"
echo "✅ Automatic agent handoffs (no manual tracking)"
echo "✅ Built-in conversation memory (sessions)"
echo "✅ Provider-agnostic (100+ LLM support)"
echo "✅ Built-in tracing and debugging"
echo "✅ Production-ready with guardrails"
echo ""
echo "For more info: https://github.com/openai/openai-agents-js"