# Unified Agent System

A comprehensive agent system that combines CLI usability, computer use capabilities, and multi-provider AI integration (Claude and OpenAI) into a single, unified codebase.

## Features

- **Multi-Provider Support**: Seamlessly switch between Claude and OpenAI
- **CLI Interface**: Interactive command-line interface with rich features
- **Computer Use**: Full computer automation capabilities
- **Tool Integration**: Comprehensive tool ecosystem including code execution, file operations, and MCP integration
- **Unified Agent Composition**: Same sophisticated agent behavior across providers
- **Configuration Management**: Environment-based configuration for different use cases

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Use Claude agent with CLI
python -m unified_agent_system.cli --provider claude

# Use OpenAI agent with computer use
python -m unified_agent_system.cli --provider openai --computer local-playwright

# Run with specific model
python -m unified_agent_system.cli --provider claude --model claude-sonnet-4-20250514
```

### Environment Setup

```bash
# For Claude
export ANTHROPIC_API_KEY="your-claude-api-key"

# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"
```

## Architecture

### Core Components

1. **Agent Framework** (`core/agent.py`)
   - Provider-agnostic agent implementation
   - Unified tool integration
   - Message history management

2. **Provider Abstraction** (`providers/`)
   - `claude_provider.py`: Claude API integration
   - `openai_provider.py`: OpenAI API integration
   - Common interface for both providers

3. **Tool System** (`tools/`)
   - Computer use tools
   - Code execution tools
   - File operation tools
   - MCP integration tools

4. **CLI Interface** (`cli.py`)
   - Interactive command-line interface
   - Configuration management
   - Multi-provider support

5. **Computer Use Module** (`computer_use/`)
   - Browser automation
   - System interaction
   - Screenshot capabilities

### Configuration

The system uses environment-based configuration:

```python
# config.py
PROVIDERS = {
    "claude": {
        "default_model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "temperature": 1.0,
    },
    "openai": {
        "default_model": "gpt-4o",
        "max_tokens": 4096,
        "temperature": 1.0,
    }
}

COMPUTER_ENVIRONMENTS = {
    "local-playwright": LocalPlaywrightComputer,
    "browserbase": BrowserBaseComputer,
    "docker": DockerComputer,
}
```

## Examples

### Basic Agent Usage

```python
from unified_agent_system.core.agent import Agent
from unified_agent_system.providers.claude_provider import ClaudeProvider

# Create agent with Claude
provider = ClaudeProvider()
agent = Agent(
    name="assistant",
    system="You are a helpful AI assistant.",
    provider=provider,
    tools=[...]
)

# Run agent
response = agent.run("Hello, how can you help me?")
```

### Computer Use Example

```python
from unified_agent_system.computer_use.local_playwright import LocalPlaywrightComputer

with LocalPlaywrightComputer() as computer:
    agent = Agent(
        name="computer_assistant",
        system="You can control the computer to help users.",
        provider=provider,
        computer=computer
    )
    
    # Agent can now use computer tools
    response = agent.run("Open a browser and search for Python tutorials")
```

### CLI with Custom Tools

```bash
# Run with custom tools
python -m unified_agent_system.cli \
    --provider claude \
    --tools computer,code_execution,file_tools \
    --computer local-playwright
```

## Development

### Project Structure

```
unified_agent_system/
├── core/
│   ├── agent.py          # Main agent implementation
│   ├── types.py          # Common types and data structures
│   └── config.py         # Configuration management
├── providers/
│   ├── base.py           # Provider abstraction
│   ├── claude_provider.py
│   └── openai_provider.py
├── tools/
│   ├── base.py           # Tool base classes
│   ├── computer_use.py
│   ├── code_execution.py
│   └── file_tools.py
├── computer_use/
│   ├── base.py           # Computer environment base
│   ├── local_playwright.py
│   └── browserbase.py
├── cli.py                # Command-line interface
└── utils/
    ├── history.py        # Message history management
    └── tool_utils.py     # Tool execution utilities
```

### Adding New Providers

1. Implement the `Provider` interface in `providers/base.py`
2. Create provider-specific implementation
3. Add configuration in `core/config.py`
4. Update CLI to support new provider

### Adding New Tools

1. Extend the `Tool` base class in `tools/base.py`
2. Implement tool logic
3. Register tool in agent configuration
4. Add any necessary provider-specific handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details. 