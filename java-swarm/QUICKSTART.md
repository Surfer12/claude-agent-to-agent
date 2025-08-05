# Java Swarm Quick Start Guide

Get up and running with Java Swarm in 5 minutes!

## Prerequisites

- [Pixi](https://pixi.sh) (recommended) OR Java 17+ and Maven 3.6+
- OpenAI API key

## Quick Setup with Pixi (Recommended)

1. **Install Pixi** (if not already installed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Setup the project**:
   ```bash
   pixi install
   ```

3. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

4. **Start chatting**:
   ```bash
   pixi run quick-start
   ```

## Alternative Setup (Manual)

1. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

2. **Build the project:**
   ```bash
   ./build.sh
   ```

3. **Run in interactive mode:**
   ```bash
   java -jar target/java-swarm-1.0.0.jar --interactive
   ```

## First Conversation

Once you run the interactive mode, you'll see:

```
Java Swarm CLI v1.0.0
Agent: Assistant (Model: gpt-4o)
Debug: false

Interactive mode. Type 'quit' or 'exit' to end the session.
Type 'clear' to clear the conversation history.
Type 'help' for available commands.

You: 
```

Try these examples:

### Basic Chat
```
You: Hello! How are you today?
Assistant: Hello! I'm doing well, thank you for asking. I'm here and ready to help you with any questions or tasks you might have. How can I assist you today?
```

### Using Built-in Functions
```
You: Can you calculate 15 * 23 + 7 for me?
Assistant: I'll calculate that for you.
Tool result: Result: 352
Assistant: The result of 15 * 23 + 7 is 352.
```

```
You: Can you echo "Hello World"?
Assistant: I'll echo that message for you.
Tool result: Echo: Hello World
Assistant: I've echoed your message: "Hello World"
```

### Streaming Responses
```
You: toggle-stream
Streaming enabled
You: Tell me a story about robots
Assistant: Once upon a time, in a world where technology had advanced beyond imagination, there lived a small robot named Zara...
[Text appears in real-time as the AI generates it]
```

## Command Line Options

### Pixi Commands (Recommended)
```bash
pixi run interactive              # Basic interactive mode
pixi run interactive-stream       # Interactive with streaming
pixi run interactive-debug        # Interactive with debug logging
pixi run chat "Your message"      # Single message
pixi run chat-stream "Message"    # Single message with streaming
```

### Manual Commands
```bash
java -jar target/java-swarm-1.0.0.jar --interactive
java -jar target/java-swarm-1.0.0.jar --input "Hello, world!"
java -jar target/java-swarm-1.0.0.jar --interactive --stream --debug
```

## Interactive Commands

While in interactive mode:
- `help` - Show available commands and functions
- `clear` - Clear conversation history
- `toggle-stream` - Toggle streaming mode on/off
- `quit` or `exit` - Exit the program

## Pixi Command Examples

### Quick Tasks
```bash
pixi run help              # Show CLI help
pixi run version           # Show version info
pixi run build             # Build the project
pixi run test              # Run tests
```

### Specialized Agents
```bash
pixi run math-bot          # Mathematics expert
pixi run code-bot          # Programming expert
pixi run story-bot         # Creative storyteller (with streaming)
```

### Model Selection
```bash
pixi run gpt4              # Use GPT-4o model
pixi run gpt4-mini         # Use GPT-4o-mini model
pixi run gpt35             # Use GPT-3.5-turbo model
```

### Development
```bash
pixi run dev               # Development mode with hot reload
pixi run streaming-demo    # Demonstrate streaming responses
pixi run calculator-demo   # Demonstrate function calling
pixi run https-demo        # Demonstrate HTTPS configuration
```

## Available Functions

The CLI comes with these built-in functions:

1. **echo(message)** - Echo back any message
2. **calculate(expression)** - Calculate mathematical expressions

## Troubleshooting

### "OPENAI_API_KEY environment variable is required"
Make sure you've set your API key:
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### "Pixi not found"
Install Pixi:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### "JAR file not found"
Build the project first:
```bash
pixi run build
```

### "Maven is not installed" (Manual setup)
Install Maven:
- **macOS:** `brew install maven`
- **Ubuntu:** `sudo apt install maven`
- **Windows:** Download from [maven.apache.org](https://maven.apache.org)

### "Java 17 or higher is required" (Manual setup)
Install Java 17+:
- **macOS:** `brew install openjdk@17`
- **Ubuntu:** `sudo apt install openjdk-17-jdk`
- **Windows:** Download from [adoptium.net](https://adoptium.net)

### Build Fails
Try cleaning and rebuilding:
```bash
pixi run rebuild
# OR manually:
mvn clean
mvn compile
mvn package
```

### Connection Issues
Test your connection:
```bash
pixi run test-connection
```

## Next Steps

1. **Explore Pixi commands** - Run `pixi task list` to see all available commands
2. **Try specialized agents** - Use `pixi run math-bot` or `pixi run story-bot`
3. **Experiment with streaming** - Use `pixi run interactive-stream` for real-time responses
4. **Check examples** - Run `pixi run streaming-demo` and `pixi run https-demo`
5. **Read detailed docs** - Check [PIXI_USAGE.md](PIXI_USAGE.md) for complete command reference
6. **Explore the code** - Check out the source code in `src/main/java/com/swarm/`
7. **Create custom functions** - See the README for how to add your own functions
8. **Run tests** - Execute `pixi run test` to run the test suite

## Getting Help

- Run `pixi run help` for CLI options
- Run `pixi task list` to see all available pixi commands
- Type `help` in interactive mode for available commands
- Check the full README.md for detailed documentation
- Check [PIXI_USAGE.md](PIXI_USAGE.md) for pixi-specific usage
- Open an issue on GitHub for bugs or questions

Happy coding with Java Swarm! 🚀
