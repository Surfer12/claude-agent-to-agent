# Claude Agent Framework

A comprehensive, multi-language framework for building Claude-powered agents with support for beta tools including computer use and code execution.

## 🌟 Features

- **🐍 Python & ☕ Java**: Full implementations in both languages
- **🛠️ Beta Tools**: Computer use, code execution, and more
- **🔌 MCP Integration**: Model Context Protocol support
- **💻 CLI Interface**: Command-line tools for both languages
- **🧩 Extensible**: Plugin architecture for custom tools
- **🚀 Production Ready**: Comprehensive testing and documentation

## 🚀 Quick Start

### Python

```bash
# Install
cd python/
pip install -e .

# Use CLI
export ANTHROPIC_API_KEY="your-api-key"
claude-agent interactive

# Use programmatically
from claude_agent import Agent, AgentConfig
config = AgentConfig(api_key="your-key")
agent = Agent(config=config)
response = await agent.run_async("Hello!")
```

### Java

```bash
# Build
cd java/
mvn clean package

# Use CLI
export ANTHROPIC_API_KEY='your-api-key'
java -jar target/claude-agent-framework-1.0.0.jar -v interactive

# Use programmatically
AgentConfig config = AgentConfig.builder()
    .apiKey("your-key")
    .build();
Agent agent = new Agent(config);
AgentResponse response = agent.chatSync("Hello!");
```

## 🛠️ Available Tools

| Tool | Description | Python | Java |
|------|-------------|--------|------|
| **think** | Internal reasoning | ✅ | ✅ |
| **file_read** | File operations | ✅ | ✅ |
| **file_write** | File editing | ✅ | ✅ |
| **computer** | Desktop interaction (beta) | ✅ | ✅ |
| **code_execution** | Python sandbox (beta) | ✅ | ✅ |

## 📖 Documentation

- [Getting Started](docs/getting-started.md)
- [Python API Reference](docs/api-reference/python/)
- [Java API Reference](docs/api-reference/java/)
- [Tool Development Guide](docs/guides/tool-development.md)
- [Beta Tools Documentation](docs/guides/beta-tools.md)

## 🏗️ Architecture

```
claude-agent-framework/
├── python/                 # Python implementation
│   └── claude_agent/      # Main package
├── java/                  # Java implementation
│   └── src/main/java/     # Source code
├── docs/                  # Documentation
└── examples/              # Usage examples
```

## 🧪 Examples

### Configuration File
```yaml
agent:
  name: my-agent
  system_prompt: "You are a helpful assistant"
  verbose: true

model:
  model: claude-sonnet-4-20250514
  max_tokens: 4096
  temperature: 0.7

tools:
  enabled:
    - think
    - file_read
    - computer
```

### CLI Usage
```bash
# Python
claude-agent list-tools
claude-agent interactive --tools think,file_read
claude-agent chat --prompt "What tools do you have?"

# Java
java -jar claude-agent.jar list-tools
java -jar claude-agent.jar interactive -t think,file_read
java -jar claude-agent.jar chat -p "What tools do you have?"
```

## 🔧 Development

### Python Development
```bash
cd python/
pip install -e .[dev]
pytest
black .
```

### Java Development
```bash
cd java/
mvn clean test
mvn clean package
```

## 📋 Requirements

### Python
- Python 3.8+
- Dependencies: anthropic, click, pyyaml, aiohttp

### Java
- Java 11+
- Dependencies: OkHttp, Jackson, Picocli

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with ❤️ using:
- [Anthropic Claude API](https://www.anthropic.com/)
- [Click](https://click.palletsprojects.com/) (Python CLI)
- [Picocli](https://picocli.info/) (Java CLI)
- [OkHttp](https://square.github.io/okhttp/) (Java HTTP)
- [Jackson](https://github.com/FasterXML/jackson) (JSON/YAML)
