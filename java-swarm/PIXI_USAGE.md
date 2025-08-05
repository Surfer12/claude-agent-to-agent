# Java Swarm with Pixi

This document describes how to use Java Swarm with [Pixi](https://pixi.sh), a fast package manager for conda environments.

## Quick Start with Pixi

1. **Install Pixi** (if not already installed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Initialize the environment**:
   ```bash
   pixi install
   ```

3. **Start using Java Swarm**:
   ```bash
   pixi run quick-start
   ```

## Available Pixi Commands

### 🏗️ Build Commands
```bash
pixi run build              # Build the entire project
pixi run compile            # Compile source code only
pixi run test              # Run unit tests
pixi run clean             # Clean build artifacts
pixi run rebuild           # Clean and rebuild
pixi run full-test         # Complete test cycle
```

### 💬 Interactive Chat Commands
```bash
pixi run interactive              # Basic interactive mode
pixi run interactive-debug        # Interactive with debug logging
pixi run interactive-stream       # Interactive with streaming
pixi run interactive-stream-debug # Interactive with streaming and debug
```

### 🤖 Model-Specific Commands
```bash
pixi run gpt4              # Use GPT-4o model
pixi run gpt4-mini         # Use GPT-4o-mini model
pixi run gpt35             # Use GPT-3.5-turbo model
```

### 🎭 Specialized Agent Commands
```bash
pixi run math-bot          # Mathematics expert agent
pixi run code-bot          # Programming expert agent
pixi run story-bot         # Creative storytelling agent (with streaming)
```

### 💬 Single Message Commands
```bash
pixi run chat "Your message here"           # Send single message
pixi run chat-stream "Your message here"    # Send with streaming
pixi run chat-debug "Your message here"     # Send with debug info
```

### 🧪 Example Commands
```bash
pixi run streaming-demo    # Demonstrate streaming responses
pixi run calculator-demo   # Demonstrate function calling
pixi run https-demo        # Demonstrate HTTPS configuration
```

### 🔧 Development Commands
```bash
pixi run dev               # Development mode with hot reload
pixi run dev-stream        # Development mode with streaming
pixi run debug-verbose     # Verbose debug mode
pixi run profile          # Run with Java profiling
```

### 📊 Testing Commands
```bash
pixi run unit-tests        # Run unit tests only
pixi run integration-tests # Run integration tests
pixi run test-streaming    # Test streaming functionality
pixi run test-connection   # Test OpenAI API connection
```

### ℹ️ Utility Commands
```bash
pixi run help              # Show CLI help
pixi run version           # Show version info
```

## Usage Examples

### Basic Chat Session
```bash
# Start interactive mode
pixi run interactive

# Or with streaming for real-time responses
pixi run interactive-stream
```

### Quick Single Messages
```bash
# Ask a question
pixi run chat "What is machine learning?"

# Ask with streaming response
pixi run chat-stream "Tell me a story about robots"

# Ask with debug information
pixi run chat-debug "Calculate 15 * 23 + 7"
```

### Specialized Agents
```bash
# Math expert
pixi run math-bot
# Then ask: "Solve the quadratic equation x² + 5x + 6 = 0"

# Code expert
pixi run code-bot
# Then ask: "Review this Python function for bugs"

# Story teller with streaming
pixi run story-bot
# Then ask: "Write a sci-fi story about time travel"
```

### Development Workflow
```bash
# Build and test
pixi run full-test

# Start development mode
pixi run dev

# Test streaming functionality
pixi run test-streaming

# Run examples
pixi run streaming-demo
pixi run https-demo
```

## Environment Configuration

### Setting API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
pixi run interactive
```

### Debug Mode
```bash
export DEBUG=true
pixi run interactive-debug
```

### Proxy Configuration
```bash
export https_proxy=http://proxy.company.com:8080
pixi run interactive
```

## Pixi Environments

The project supports multiple pixi environments:

### Default Environment
```bash
pixi run interactive  # Uses default environment
```

### Development Environment
```bash
pixi run -e dev interactive  # Uses development environment with extra tools
```

### Testing Environment
```bash
pixi run -e test unit-tests  # Uses testing environment
```

### Production Environment
```bash
pixi run -e prod interactive  # Uses minimal production environment
```

## Advanced Usage

### Custom Commands
You can create custom pixi tasks by editing `pixi.toml`:

```toml
[tasks]
my-custom-agent = "java -jar target/java-swarm-1.0.0.jar --interactive --agent-name MyAgent --instructions 'Custom instructions here'"
```

### Chaining Commands
```bash
# Build and run in one command
pixi run quick-start

# Clean, build, test, and run
pixi run full-test && pixi run interactive
```

### Environment Variables in Tasks
```bash
# Set environment variables for specific runs
OPENAI_API_KEY=sk-... DEBUG=true pixi run interactive-debug
```

## Troubleshooting

### Common Issues

1. **"JAR file not found"**:
   ```bash
   pixi run build  # Ensure project is built first
   ```

2. **"OPENAI_API_KEY not set"**:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

3. **Java version issues**:
   ```bash
   pixi install  # Reinstall to get correct Java version
   ```

4. **Maven issues**:
   ```bash
   pixi run clean && pixi run build  # Clean rebuild
   ```

### Debug Information
```bash
# Check environment
pixi info

# Check installed packages
pixi list

# Verbose output
pixi run -v interactive
```

## Integration with IDEs

### VS Code
Add to `.vscode/tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Java Swarm Interactive",
            "type": "shell",
            "command": "pixi",
            "args": ["run", "interactive"],
            "group": "build"
        }
    ]
}
```

### IntelliJ IDEA
Configure external tools to run pixi commands directly from the IDE.

## Performance Tips

1. **Use development mode** for faster iteration:
   ```bash
   pixi run dev
   ```

2. **Pre-build for faster startup**:
   ```bash
   pixi run build  # Build once, then use interactive commands
   ```

3. **Use streaming** for better user experience:
   ```bash
   pixi run interactive-stream
   ```

## Contributing

When contributing to the project, use pixi commands for consistency:

```bash
# Development workflow
pixi run full-test      # Run all tests
pixi run dev           # Start development mode
pixi run streaming-demo # Test new features
```

This ensures all contributors use the same environment and commands.
