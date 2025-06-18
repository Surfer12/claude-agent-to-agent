# Anthropic API Client Library

A comprehensive, dual-language client library for Anthropic's Claude API, providing support for various tools, streaming, and advanced features in both Python and Java.

## 📁 Package Structure

The library is organized following Java package naming conventions for consistency across languages:

```
src/main/
├── python/
│   └── com/
│       └── anthropic/
│           └── api/
│               ├── __init__.py          # Main package initialization
│               ├── client.py            # Core API client functionality
│               ├── tools.py             # Tool implementations
│               ├── streaming.py         # Streaming utilities
│               └── cli.py               # Command-line interface
└── java/
    └── com/
        └── anthropic/
            └── api/
                ├── AnthropicClientEnhanced.java    # Enhanced Java client
                ├── tools/
                │   └── AnthropicTools.java         # Java tool implementations
                └── cli/
                    └── CognitiveAgentCLI.java      # Java CLI implementation
```

## 🚀 Quick Start

### Python

```python
from com.anthropic.api import AnthropicClient, create_bash_tool

# Create client
client = AnthropicClient()

# Use with tools
response = client.create_message(
    messages=[{"role": "user", "content": "List files in current directory"}],
    tools=["bash"],
    model="claude-sonnet-4-20250514"
)

print(response.content[0].text)
```

### Java

```java
import com.anthropic.api.AnthropicClientEnhanced;
import com.anthropic.api.tools.AnthropicTools;

// Create client
AnthropicClientEnhanced client = AnthropicClientEnhanced.createBasicClient(apiKey);

// Use with tools
List<AnthropicClientEnhanced.Message> messages = Arrays.asList(
    new AnthropicClientEnhanced.Message("", "user", Arrays.asList(
        new AnthropicClientEnhanced.Content("text", "List files in current directory")
    ))
);

AnthropicClientEnhanced.Message response = client.createMessage(
    messages,
    Arrays.asList("bash"),
    null
);

System.out.println(response.getContent().get(0).getText());
```

## 🛠️ Available Tools

Both Python and Java implementations support the following tools:

| Tool | Type | Description |
|------|------|-------------|
| `bash` | `bash_20250124` | Execute bash commands in a secure environment |
| `web_search` | `web_search_20250305` | Search the web for current information |
| `weather` | `weather_tool` | Get current weather information for a location |
| `text_editor` | `text_editor_20250429` | Edit text files with string replacement operations |
| `code_execution` | `code_execution_20250522` | Execute code in a secure environment |
| `computer` | `computer_20250124` | Interact with computer interface |

## 📦 Installation

### Python

```bash
# From source
cd src/main/python
pip install -e .

# Or install dependencies
pip install anthropic frozendict typing-extensions
```

### Java

```xml
<!-- Add to pom.xml -->
<dependency>
    <groupId>com.anthropic</groupId>
    <artifactId>anthropic-api-client</artifactId>
    <version>1.0.0</version>
</dependency>
```

## 🔧 Configuration

### Environment Variables

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Python Configuration

```python
from com.anthropic.api import AnthropicClient

# Basic client
client = AnthropicClient()

# With custom configuration
client = AnthropicClient(
    api_key="your-api-key",
    default_model="claude-opus-4-20250514"
)
```

### Java Configuration

```java
import com.anthropic.api.AnthropicClientEnhanced;

// Basic client
AnthropicClientEnhanced client = AnthropicClientEnhanced.createBasicClient(apiKey);

// With custom configuration
AnthropicClientEnhanced client = new AnthropicClientEnhanced.Builder()
    .apiKey(apiKey)
    .model("claude-opus-4-20250514")
    .maxTokens(2048)
    .build();
```

## 🎯 Usage Examples

### Basic Message Creation

#### Python
```python
from com.anthropic.api import AnthropicClient

client = AnthropicClient()

response = client.create_message(
    messages=[{"role": "user", "content": "Hello, Claude!"}],
    model="claude-sonnet-4-20250514"
)

print(response.content[0].text)
```

#### Java
```java
import com.anthropic.api.AnthropicClientEnhanced;

AnthropicClientEnhanced client = AnthropicClientEnhanced.createBasicClient(apiKey);

List<AnthropicClientEnhanced.Message> messages = Arrays.asList(
    new AnthropicClientEnhanced.Message("", "user", Arrays.asList(
        new AnthropicClientEnhanced.Content("text", "Hello, Claude!")
    ))
);

AnthropicClientEnhanced.Message response = client.createMessage(messages, null, null);
System.out.println(response.getContent().get(0).getText());
```

### Using Tools

#### Python
```python
from com.anthropic.api import AnthropicClient
from com.anthropic.api.tools import create_bash_tool, create_web_search_tool

client = AnthropicClient()

# Use bash tool
response = client.create_message(
    messages=[{"role": "user", "content": "List all Python files"}],
    tools=["bash"]
)

# Use web search tool
response = client.create_message(
    messages=[{"role": "user", "content": "What's the latest news about AI?"}],
    tools=["web_search"]
)
```

#### Java
```java
import com.anthropic.api.AnthropicClientEnhanced;
import com.anthropic.api.tools.AnthropicTools;

AnthropicClientEnhanced client = AnthropicClientEnhanced.createBasicClient(apiKey);

// Use bash tool
List<AnthropicClientEnhanced.Message> messages = Arrays.asList(
    new AnthropicClientEnhanced.Message("", "user", Arrays.asList(
        new AnthropicClientEnhanced.Content("text", "List all Java files")
    ))
);

AnthropicClientEnhanced.Message response = client.createMessage(
    messages,
    Arrays.asList("bash"),
    null
);
```

### Streaming Responses

#### Python
```python
from com.anthropic.api import AnthropicClient
from com.anthropic.api.streaming import print_streaming_response

client = AnthropicClient()

streaming_response = client.create_streaming_message(
    messages=[{"role": "user", "content": "Write a story about a robot"}],
    model="claude-sonnet-4-20250514"
)

print_streaming_response(streaming_response)
```

#### Java
```java
import com.anthropic.api.AnthropicClientEnhanced;

AnthropicClientEnhanced client = AnthropicClientEnhanced.createBasicClient(apiKey);

List<AnthropicClientEnhanced.Message> messages = Arrays.asList(
    new AnthropicClientEnhanced.Message("", "user", Arrays.asList(
        new AnthropicClientEnhanced.Content("text", "Write a story about a robot")
    ))
);

AnthropicClientEnhanced.StreamingResponse response = client.createStreamingMessage(
    messages,
    null,
    null
);

System.out.println("🤖 Streaming Response:");
System.out.println(response.getResponseBody());
```

## 🖥️ Command Line Interface

### Python CLI

```bash
# Interactive mode
python -m com.anthropic.api.cli

# Single query mode
python -m com.anthropic.api.cli --query "What is the weather in San Francisco?"

# With specific tools
python -m com.anthropic.api.cli --tools bash web_search weather

# With custom model
python -m com.anthropic.api.cli --model claude-opus-4-20250514
```

### Java CLI

```bash
# Interactive mode
java -cp target/classes com.anthropic.api.cli.CognitiveAgentCLI

# With custom configuration
java -cp target/classes com.anthropic.api.cli.CognitiveAgentCLI --name MyAgent --verbose
```

## 🔒 Security Features

- **Immutable Collections**: Uses immutable collections to prevent accidental modification
- **Thread Safety**: Designed for concurrent access with proper synchronization
- **API Key Management**: Secure handling of API keys through environment variables
- **Input Validation**: Comprehensive validation of all inputs and parameters

## 🧪 Testing

### Python Tests

```bash
cd src/main/python
pytest tests/
```

### Java Tests

```bash
cd src/main/java
mvn test
```

## 📚 API Reference

### Python Classes

- `AnthropicClient`: Main client for API interactions
- `BaseTool`: Base class for all tools
- `BashTool`, `WebSearchTool`, `WeatherTool`, etc.: Specific tool implementations
- `StreamingResponse`: Wrapper for streaming responses
- `CognitiveAgentCLI`: Command-line interface

### Java Classes

- `AnthropicClientEnhanced`: Enhanced Java client
- `AnthropicTools`: Tool implementations and utilities
- `CognitiveAgentCLI`: Java command-line interface
- `ToolDefinition`, `BaseTool`: Tool framework classes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Python Package Index](https://pypi.org/project/anthropic-api-client/)
- [Maven Central Repository](https://search.maven.org/artifact/com.anthropic/anthropic-api-client)

## 🆘 Support

For support and questions:

- Create an issue on GitHub
- Check the documentation
- Review the examples in the `examples/` directory

---

**Note**: This library is designed to work with Anthropic's Claude API. Make sure you have a valid API key and are following Anthropic's usage guidelines. 