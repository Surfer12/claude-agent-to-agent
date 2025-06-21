#!/bin/bash

# Diagnostic script for Claude Agent Framework (Java)

echo "Claude Agent Framework - Java Diagnostic Script"
echo "=============================================="
echo ""

# Check Java installation
echo "1. Checking Java installation..."
if command -v java &> /dev/null; then
    echo "   ✓ Java is installed"
    echo "   Version: $(java -version 2>&1 | head -n 1)"
else
    echo "   ✗ Java is not installed or not in PATH"
    echo "   Please install Java 11 or higher"
    exit 1
fi
echo ""

# Check for JAR file
echo "2. Checking for JAR file..."
JAR_PATH="claude-agent-framework/java/target/claude-agent-framework-1.0.0.jar"
if [ -f "$JAR_PATH" ]; then
    echo "   ✓ JAR file found at: $JAR_PATH"
else
    echo "   ✗ JAR file not found at: $JAR_PATH"
    echo "   Please build the project first with: cd claude-agent-framework/java && mvn package"
    exit 1
fi
echo ""

# Check API key
echo "3. Checking ANTHROPIC_API_KEY..."
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "   ✓ ANTHROPIC_API_KEY is set"
    echo "   Key starts with: ${ANTHROPIC_API_KEY:0:10}..."
    echo "   Key length: ${#ANTHROPIC_API_KEY} characters"
else
    echo "   ✗ ANTHROPIC_API_KEY is not set"
    echo "   Please set it with: export ANTHROPIC_API_KEY='your-api-key-here'"
    echo ""
    echo "   You can get an API key from: https://console.anthropic.com/"
fi
echo ""

# Check for config files
echo "4. Checking for configuration files..."
CONFIG_FILES=(
    "claude-agent.yaml"
    "claude-agent.yml"
    "$HOME/.claude-agent.yaml"
    "$HOME/.claude-agent.yml"
)

found_config=false
for config in "${CONFIG_FILES[@]}"; do
    if [ -f "$config" ]; then
        echo "   ✓ Found config file: $config"
        found_config=true
    fi
done

if [ "$found_config" = false ]; then
    echo "   ℹ No configuration files found (using defaults)"
fi
echo ""

# Test basic functionality
echo "5. Testing basic functionality..."
echo "   Running: java -jar $JAR_PATH generate-config"
echo ""
java -jar "$JAR_PATH" generate-config
echo ""

# Provide next steps
echo "=============================================="
echo "Diagnostic Summary:"
echo ""

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✓ Environment appears to be set up correctly"
    echo ""
    echo "To test the agent, try:"
    echo "  java -jar $JAR_PATH chat -p \"Hello, Claude!\""
    echo ""
    echo "For interactive mode:"
    echo "  java -jar $JAR_PATH interactive"
    echo ""
    echo "For verbose output (helpful for debugging):"
    echo "  java -jar $JAR_PATH -v interactive"
else
    echo "✗ Missing ANTHROPIC_API_KEY"
    echo ""
    echo "Please set your API key first:"
    echo "  export ANTHROPIC_API_KEY='your-api-key-here'"
    echo ""
    echo "You can get an API key from: https://console.anthropic.com/"
fi

echo ""
echo "For more help, see the documentation or run:"
echo "  java -jar $JAR_PATH --help"
