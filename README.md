# Claude Agent-to-Agent CLI

A command-line interface for interacting with Claude-powered agents. This CLI integrates with the Claude API and supports a variety of tools for enhanced capabilities.

## Installation

Install directly from GitHub:

```bash
# Install with pip
pip install -e .

# Or use with pixi
pixi install
pixi run python cli.py
```

## Usage

### Basic Usage

```bash
# Interactive session
claude-agent --interactive

# Single prompt
claude-agent --prompt "What is the capital of France?"

# From file
claude-agent --file prompt.txt

# From stdin
cat prompt.txt | claude-agent --file -
```

### Tool Configuration

```bash
# Enable specific tools
claude-agent --interactive --tools think file_read

# Enable all available tools
claude-agent --interactive --tools all
```

### MCP Server Integration

```bash
# Connect to an MCP tool server
claude-agent --interactive --mcp-server http://localhost:8080

# Connect to multiple MCP servers
claude-agent --interactive --mcp-server http://localhost:8080 --mcp-server http://localhost:8081
```

### Model Configuration

```bash
# Configure model parameters
claude-agent --interactive --model claude-sonnet-4-20250514 --max-tokens 2048 --temperature 0.7
```

### API Configuration

```bash
# Use a specific API key
claude-agent --interactive --api-key your_api_key_here
```

## Available Tools

- `think`: A tool for internal reasoning
- `file_read`: A tool for reading files and listing directories
- `file_write`: A tool for writing and editing files
- MCP-based tools: Connect to MCP servers for additional capabilities

## Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key

## Examples

```bash
# Interactive session with all tools
claude-agent --interactive --tools all --verbose

# Generate a response to a specific prompt
claude-agent --prompt "Write a haiku about programming"

# Process a complex request with specific tools
claude-agent --prompt "List all Python files in the current directory" --tools file_read

# Connect to an MCP server for additional capabilities
claude-agent --interactive --mcp-server http://localhost:8080
```

## Requirements

- Python 3.10+
- Anthropic API key
- `anthropic` Python library

# Anthropic API Client

This project provides both Java and Python implementations for interacting with the Anthropic API. It includes examples of basic message creation, multi-turn conversations, and tool usage.

## Features

- Java client with immutable collections for thread safety
- Python examples demonstrating various API features
- Comprehensive test coverage
- Environment setup scripts
- Maven and pip dependency management

## Prerequisites

- Java 17 or higher
- Python 3.8 or higher
- Maven
- pip

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Set up the environment:
```bash
# For Unix-like systems
chmod +x setup_api_env.sh
./setup_api_env.sh

# For Windows
.\setup_api_env.sh
```

3. Add your Anthropic API key to the `.env` file:
```
ANTHROPIC_API_KEY=your-api-key-here
```

## Building and Testing

### Java

```bash
# Build the project
mvn clean install

# Run tests
mvn test
```

### Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run example
python examples/api_example.py
```

## Project Structure

```
.
├── src/
│   ├── main/java/com/anthropic/api/
│   │   └── AnthropicClient.java
│   └── test/java/com/anthropic/api/
│       └── AnthropicClientTest.java
├── examples/
│   └── api_example.py
├── pom.xml
├── requirements.txt
├── setup_api_env.sh
└── README.md
```

## Usage Examples

### Java

```java
AnthropicClient client = new AnthropicClient.Builder()
    .apiKey("your-api-key")
    .model("claude-opus-4-20250514")
    .maxTokens(1024)
    .build();

List<AnthropicClient.Message> messages = List.of(
    new AnthropicClient.Message("msg1", "user", 
        List.of(new AnthropicClient.Content("text", "Hello, Claude")))
);

AnthropicClient.Message response = client.createMessage(messages);
```

### Python

```python
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.