#!/usr/bin/env python3
"""
Quick Start Script for Unified Agent System

This script helps users get started with the unified agent system by:
1. Checking dependencies
2. Setting up environment variables
3. Running a simple example
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "anthropic",
        "openai",
        "asyncio",
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    return True


def setup_environment():
    """Set up environment variables."""
    print("\n🔑 Setting up environment variables...")
    
    # Check for existing API keys
    claude_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if claude_key:
        print("✅ ANTHROPIC_API_KEY already set")
    else:
        print("❌ ANTHROPIC_API_KEY not set")
        print("   Get your key from: https://console.anthropic.com/")
    
    if openai_key:
        print("✅ OPENAI_API_KEY already set")
    else:
        print("❌ OPENAI_API_KEY not set")
        print("   Get your key from: https://platform.openai.com/api-keys")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("\n📝 Creating .env file...")
        with open(env_file, "w") as f:
            f.write("# Unified Agent System Environment Variables\n")
            f.write("# Add your API keys here\n\n")
            f.write("# Claude API Key\n")
            f.write("# ANTHROPIC_API_KEY=your-claude-api-key-here\n\n")
            f.write("# OpenAI API Key\n")
            f.write("# OPENAI_API_KEY=your-openai-api-key-here\n\n")
            f.write("# Optional: Enable debug mode\n")
            f.write("# DEBUG=true\n")
            f.write("# VERBOSE=true\n")
        print("✅ Created .env file")
        print("   Please edit .env file and add your API keys")
    
    return bool(claude_key or openai_key)


def run_simple_example():
    """Run a simple example to test the system."""
    print("\n🚀 Running simple example...")
    
    try:
        # Import the unified agent
        from unified_agent import UnifiedAgent, AgentConfig, ProviderType
        
        # Check which providers are available
        claude_key = os.environ.get("ANTHROPIC_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        
        if not claude_key and not openai_key:
            print("❌ No API keys available. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY")
            return False
        
        # Choose provider
        if claude_key:
            provider = ProviderType.CLAUDE
            model = "claude-3-5-sonnet-20241022"
            print("🤖 Using Claude provider")
        else:
            provider = ProviderType.OPENAI
            model = "gpt-4o"
            print("🤖 Using OpenAI provider")
        
        # Create configuration
        config = AgentConfig(
            provider=provider,
            model=model,
            system_prompt="You are a helpful AI assistant. Be concise.",
            verbose=True
        )
        
        # Create agent
        agent = UnifiedAgent(config)
        
        # Test with a simple query
        test_query = "What is 2 + 2? Please respond briefly."
        print(f"📝 Testing with: {test_query}")
        
        response = agent.run(test_query)
        
        # Extract text content
        text_content = []
        for block in response.get("content", []):
            if block.get("type") == "text":
                text_content.append(block.get("text", ""))
        
        if text_content:
            print(f"🤖 Response: {' '.join(text_content)}")
            print("✅ Example completed successfully!")
            return True
        else:
            print("❌ No text response received")
            return False
            
    except Exception as e:
        print(f"❌ Example failed: {str(e)}")
        return False


def show_next_steps():
    """Show next steps for users."""
    print("\n🎯 Next Steps:")
    print("1. Set up your API keys in the .env file")
    print("2. Run the basic example: python examples/basic_usage.py")
    print("3. Try the CLI: python -m unified_agent.cli --help")
    print("4. Check the README.md for more examples")
    print("\n📚 Documentation:")
    print("- README.md: Complete documentation")
    print("- examples/: Example scripts")
    print("- unified_agent/: Source code")


async def main():
    """Main function."""
    print("🚀 Unified Agent System - Quick Start")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    has_keys = setup_environment()
    
    # Run example if keys are available
    if has_keys:
        success = run_simple_example()
        if not success:
            print("\n⚠️  Example failed, but you can still use the system")
    else:
        print("\n⚠️  No API keys found. Please set them up to run examples")
    
    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    asyncio.run(main()) 