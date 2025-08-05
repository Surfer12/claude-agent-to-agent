# Java Swarm Quick Start Guide

Get up and running with Java Swarm in 5 minutes!

## Prerequisites

- Java 17 or higher
- Maven 3.6 or higher
- OpenAI API key

## Quick Setup

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

## Command Line Options

### Single Message Mode
```bash
java -jar target/java-swarm-1.0.0.jar --input "What's 2+2?"
```

### Debug Mode
```bash
java -jar target/java-swarm-1.0.0.jar --interactive --debug
```

### Custom Model
```bash
java -jar target/java-swarm-1.0.0.jar --interactive --model gpt-4o-mini
```

### Custom Agent
```bash
java -jar target/java-swarm-1.0.0.jar --interactive --agent-name "MathBot" --instructions "You are a mathematics expert."
```

## Interactive Commands

While in interactive mode:
- `help` - Show available commands and functions
- `clear` - Clear conversation history
- `quit` or `exit` - Exit the program

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

### "Maven is not installed"
Install Maven:
- **macOS:** `brew install maven`
- **Ubuntu:** `sudo apt install maven`
- **Windows:** Download from [maven.apache.org](https://maven.apache.org)

### "Java 17 or higher is required"
Install Java 17+:
- **macOS:** `brew install openjdk@17`
- **Ubuntu:** `sudo apt install openjdk-17-jdk`
- **Windows:** Download from [adoptium.net](https://adoptium.net)

### Build Fails
Try cleaning and rebuilding:
```bash
mvn clean
mvn compile
mvn package
```

## Next Steps

1. **Explore the code** - Check out the source code in `src/main/java/com/swarm/`
2. **Create custom functions** - See the README for how to add your own functions
3. **Run tests** - Execute `mvn test` to run the test suite
4. **Check examples** - Look at `examples/basic-usage.java` for programmatic usage

## Getting Help

- Run `java -jar target/java-swarm-1.0.0.jar --help` for CLI options
- Type `help` in interactive mode for available commands
- Check the full README.md for detailed documentation
- Open an issue on GitHub for bugs or questions

Happy coding with Java Swarm! 🚀
