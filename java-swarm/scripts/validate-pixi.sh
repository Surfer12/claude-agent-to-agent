#!/bin/bash

# Pixi Configuration Validation Script

echo "🔍 Validating Pixi configuration for Java Swarm..."
echo ""

# Check if pixi is installed
if ! command -v pixi &> /dev/null; then
    echo "❌ Pixi is not installed"
    echo "   Install with: curl -fsSL https://pixi.sh/install.sh | bash"
    exit 1
fi

echo "✅ Pixi is installed: $(pixi --version)"

# Check if pixi.toml exists
if [ ! -f "pixi.toml" ]; then
    echo "❌ pixi.toml not found in current directory"
    exit 1
fi

echo "✅ pixi.toml found"

# Validate pixi.toml syntax
if pixi info &> /dev/null; then
    echo "✅ pixi.toml syntax is valid"
else
    echo "❌ pixi.toml has syntax errors"
    pixi info
    exit 1
fi

# Check if environment can be created
echo ""
echo "🔧 Testing pixi environment creation..."
if pixi install --quiet; then
    echo "✅ Pixi environment created successfully"
else
    echo "❌ Failed to create pixi environment"
    exit 1
fi

# Test key pixi tasks
echo ""
echo "🧪 Testing key pixi tasks..."

# Test help command
if pixi run help &> /dev/null; then
    echo "✅ 'pixi run help' works"
else
    echo "❌ 'pixi run help' failed"
fi

# Test build command
echo "🏗️  Testing build process..."
if pixi run build &> /dev/null; then
    echo "✅ 'pixi run build' works"
else
    echo "❌ 'pixi run build' failed"
fi

# Check if JAR was created
if [ -f "target/java-swarm-1.0.0.jar" ]; then
    echo "✅ JAR file created successfully"
else
    echo "❌ JAR file not found after build"
fi

# Test version command
if pixi run version &> /dev/null; then
    echo "✅ 'pixi run version' works"
else
    echo "❌ 'pixi run version' failed"
fi

# List all available tasks
echo ""
echo "📋 Available pixi tasks:"
pixi task list | head -20

echo ""
echo "🎉 Pixi configuration validation complete!"
echo ""
echo "Quick start commands:"
echo "  pixi run interactive        # Start interactive mode"
echo "  pixi run interactive-stream # Start with streaming"
echo "  pixi run quick-start       # Build and run"
echo "  pixi run help              # Show CLI help"
echo ""
echo "For complete command reference, see PIXI_USAGE.md"
