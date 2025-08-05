#!/bin/bash

# Java Swarm Environment Setup Script

echo "🚀 Setting up Java Swarm environment..."

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY environment variable is not set"
    echo "   Please set it with: export OPENAI_API_KEY='your-api-key-here'"
    echo ""
fi

# Check Java version
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
    if [ "$JAVA_VERSION" -ge 17 ]; then
        echo "✅ Java $JAVA_VERSION detected (Java 17+ required)"
    else
        echo "❌ Java 17+ required, but Java $JAVA_VERSION found"
        echo "   Please install Java 17 or higher"
    fi
else
    echo "❌ Java not found in PATH"
    echo "   Java will be installed via pixi dependencies"
fi

# Check Maven
if command -v mvn &> /dev/null; then
    MVN_VERSION=$(mvn -version 2>&1 | head -n 1 | cut -d' ' -f3)
    echo "✅ Maven $MVN_VERSION detected"
else
    echo "📦 Maven will be installed via pixi dependencies"
fi

# Create necessary directories
mkdir -p target
mkdir -p logs

# Set useful aliases
echo "📝 Setting up helpful aliases..."
echo "   Use 'pixi run help' to see all available commands"
echo "   Use 'pixi run quick-start' to build and run interactively"
echo "   Use 'pixi run interactive-stream' for streaming mode"

echo ""
echo "🎉 Java Swarm environment setup complete!"
echo ""
echo "Quick commands:"
echo "  pixi run build              # Build the project"
echo "  pixi run interactive        # Start interactive mode"
echo "  pixi run interactive-stream # Start with streaming"
echo "  pixi run help              # Show CLI help"
echo ""
