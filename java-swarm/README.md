# Java Swarm Implementation

A Java implementation of the Swarm multi-agent framework, providing a command-line interface for interacting with OpenAI-powered agents.

## Features

- **Multi-Agent Support**: Create and manage multiple AI agents
- **Function Calling**: Agents can call custom functions and tools
- **Context Management**: Maintain conversation context across agent interactions
- **CLI Interface**: Easy-to-use command-line interface
- **Debug Mode**: Detailed logging for troubleshooting
- **Extensible**: Easy to add new functions and capabilities

## Architecture

```
java-swarm/
├── src/main/java/com/swarm/
│   ├── cli/                    # Command-line interface
│   │   ├── SwarmCLI.java      # Main CLI class
│   │   ├── EchoFunction.java  # Example echo function
│   │   └── CalculatorFunction.java # Example calculator function
│   ├── core/                   # Core Swarm functionality
│   │   ├── Swarm.java         # Main Swarm orchestrator
│   │   └── SwarmToolHandler.java # Tool call handling
│   ├── types/                  # Type definitions
│   │   ├── Agent.java         # Agent class
│   │   ├── AgentFunction.java # Function interface
│   │   ├── Response.java      # Response class
│   │   └── Result.java        # Result class
│   └── util/                   # Utility classes
│       └── SwarmUtil.java     # Utility functions
└── pom.xml                     # Maven configuration
```

## Prerequisites

- Java 17 or higher
- Maven 3.6 or higher
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd java-swarm
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

3. Build the project:
```bash
mvn clean package
```

## Usage

### Command Line Interface

#### Interactive Mode
```bash
java -jar target/java-swarm-1.0.0.jar --interactive
```

#### Single Message
```bash
java -jar target/java-swarm-1.0.0.jar --input "Hello, how are you?"
```

#### With Custom Options
```bash
java -jar target/java-swarm-1.0.0.jar --interactive --debug --model gpt-4o-mini --agent-name "MyAgent"
```

### CLI Options

- `--help, -h`: Show help message
- `--version, -v`: Show version information
- `--interactive, -i`: Run in interactive mode
- `--input MESSAGE`: Send a single message
- `--model MODEL`: OpenAI model to use (default: gpt-4o)
- `--debug, -d`: Enable debug mode
- `--max-turns TURNS`: Maximum number of conversation turns
- `--agent-name NAME`: Name for the agent (default: Assistant)
- `--instructions TEXT`: System instructions for the agent

### Interactive Commands

When running in interactive mode, you can use these commands:
- `help`: Show available commands
- `clear`: Clear conversation history
- `quit` or `exit`: Exit the program

### Built-in Functions

The CLI includes example functions that agents can call:
- `echo(message)`: Echo back a message
- `calculate(expression)`: Calculate mathematical expressions

## Programmatic Usage

```java
import com.swarm.core.Swarm;
import com.swarm.types.Agent;
import com.swarm.types.Response;

// Create a Swarm instance
Swarm swarm = new Swarm();

// Create an agent
Agent agent = Agent.builder()
    .name("Assistant")
    .model("gpt-4o")
    .instructions("You are a helpful assistant.")
    .build();

// Create a message
List<Map<String, Object>> messages = Arrays.asList(
    Map.of("role", "user", "content", "Hello!")
);

// Run the conversation
Response response = swarm.run(agent, messages);

// Get the response
System.out.println(response.getMessages().get(0).get("content"));
```

## Creating Custom Functions

To create a custom function, implement the `AgentFunction` interface:

```java
import com.swarm.types.AgentFunction;
import java.util.Map;

public class MyCustomFunction implements AgentFunction {
    
    @Override
    public Object execute(Map<String, Object> args) {
        String input = (String) args.get("input");
        // Your function logic here
        return "Processed: " + input;
    }
    
    @Override
    public String getName() {
        return "my_function";
    }
    
    @Override
    public String getDescription() {
        return "Description of what this function does";
    }
    
    @Override
    public Map<String, Object> getParameterSchema() {
        return Map.of(
            "type", "object",
            "properties", Map.of(
                "input", Map.of(
                    "type", "string",
                    "description", "Input parameter description"
                )
            ),
            "required", new String[]{"input"}
        );
    }
}
```

Then add it to your agent:

```java
agent.getFunctions().add(new MyCustomFunction());
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `DEBUG`: Enable debug mode (optional)

### Supported Models

- `gpt-4o` (default)
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

## Examples

### Basic Chat
```bash
java -jar target/java-swarm-1.0.0.jar --input "What's the weather like?"
```

### Calculator Usage
```bash
java -jar target/java-swarm-1.0.0.jar --interactive
You: Can you calculate 15 * 23 + 7?
Assistant: I'll calculate that for you.
[Function call: calculate(15 * 23 + 7)]
Result: 352
```

### Debug Mode
```bash
java -jar target/java-swarm-1.0.0.jar --interactive --debug
```

## Development

### Building from Source
```bash
mvn clean compile
```

### Running Tests
```bash
mvn test
```

### Creating a Fat JAR
```bash
mvn clean package
```

The executable JAR will be created at `target/java-swarm-1.0.0.jar`.

## Comparison with Python Implementation

This Java implementation provides equivalent functionality to the Python Swarm framework:

| Feature | Python | Java |
|---------|--------|------|
| Agent Creation | ✅ | ✅ |
| Function Calling | ✅ | ✅ |
| Context Variables | ✅ | ✅ |
| Multi-turn Conversations | ✅ | ✅ |
| Debug Mode | ✅ | ✅ |
| CLI Interface | ✅ | ✅ |
| Streaming | ✅ | ⚠️ (Planned) |

## Limitations

- Streaming responses are not yet implemented
- Limited to OpenAI models (no Claude support yet)
- Function parameter introspection is more limited than Python

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review the examples

## Roadmap

- [ ] Streaming response support
- [ ] Claude API integration
- [ ] Web interface
- [ ] More built-in functions
- [ ] Agent handoff capabilities
- [ ] Configuration file support
- [ ] Docker containerization
