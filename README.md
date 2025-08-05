# Unified Agent System with Swarm Integration

A provider-agnostic agent framework that supports both Claude and OpenAI backends, with unified CLI, computer use, and multi-agent swarm capabilities.

## Features

- **Provider Agnostic**: Switch seamlessly between Claude and OpenAI
- **Unified Interface**: Same agent composition works across providers
- **CLI Interface**: Command-line interface for both providers
- **Computer Use**: Browser automation and computer interaction
- **Tool Integration**: Code execution, file operations, and more
- **Modular Design**: Easy to extend with new tools and providers

## Architecture

```
unified_agent/
├── __init__.py          # Main package exports
├── core.py              # Core agent framework
├── providers.py         # Provider implementations (Claude/OpenAI)
├── tools.py             # Tool registry and management
├── cli.py               # Command-line interface
├── computer_use.py      # Computer use interface
└── tools/               # Individual tool implementations
    ├── __init__.py
    ├── base.py          # Base tool class
    ├── computer_use.py  # Computer use tool
    ├── code_execution.py # Code execution tool
    └── file_tools.py    # File manipulation tools
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd unified-agent-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# For Claude
export ANTHROPIC_API_KEY="your-claude-api-key"

# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"
```

## Usage

### Basic CLI Usage

```bash
# Basic interaction with Claude
python -m unified_agent.cli --provider claude --input "Hello, how are you?"

# OpenAI with interactive mode
python -m unified_agent.cli --provider openai --interactive

# Enable code execution
python -m unified_agent.cli --provider claude --enable-code-execution --interactive

# Enable computer use
python -m unified_agent.cli --provider openai --enable-computer-use --computer-type local-playwright

# Swarm integration
python -m unified_agent.cli --swarm-config swarm/examples/airline/configs/agents.py --initial-agent triage_agent
```

### Programmatic Usage

```python
from unified_agent import UnifiedAgent, AgentConfig, ProviderType

# Create configuration
config = AgentConfig(
    provider=ProviderType.CLAUDE,
    model="claude-3-5-sonnet-20241022",
    enable_tools=True,
    enable_computer_use=True,
    verbose=True
)

# Create agent
agent = UnifiedAgent(config)

# Run agent
response = agent.run("Hello, can you help me with a task?")
print(response)
```

### Computer Use Example

```python
from unified_agent import ComputerUseAgent, AgentConfig, ProviderType

# Create computer use agent
config = AgentConfig(
    provider=ProviderType.OPENAI,
    enable_computer_use=True,
    computer_type="local-playwright",
    start_url="https://google.com"
)

agent = ComputerUseAgent(config)

# Run interactive computer use
await agent.run_interactive()
```

## Configuration Options

### Agent Configuration

- `provider`: AI provider (CLAUDE or OPENAI)
- `model`: Model name (provider-specific defaults)
- `api_key`: API key (from environment if not provided)
- `max_tokens`: Maximum response tokens
- `temperature`: Response randomness (0.0-1.0)
- `system_prompt`: System prompt for the agent
- `verbose`: Enable detailed logging

### Tool Configuration

- `enable_tools`: Enable basic tools
- `enable_code_execution`: Enable code execution tools
- `enable_computer_use`: Enable computer use capabilities

### Computer Use Configuration

- `computer_type`: Computer environment type
  - `local-playwright`: Local Playwright browser
  - `browserbase`: Browserbase cloud browser
- `start_url`: Starting URL for browser sessions
- `show_images`: Show screenshots during execution
- `debug`: Enable debug mode

## Supported Models

### Claude Models
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

### OpenAI Models
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

## Tools

### Built-in Tools

1. **Code Execution**: Execute Python code in sandboxed environment
2. **File Operations**: Read, write, list, and delete files
3. **Computer Use**: Browser automation and computer interaction

### Computer Use Actions

- `navigate`: Navigate to a URL
- `click`: Click on page elements
- `type`: Type text into form fields
- `screenshot`: Take page screenshots
- `scroll`: Scroll the page
- `wait`: Wait for specified time

## Development

### Adding New Tools

1. Create a new tool class inheriting from `BaseTool`:
```python
from unified_agent.tools.base import BaseTool

class MyTool(BaseTool):
    def __init__(self):
        super().__init__("my_tool", "Description of my tool")
    
    async def execute(self, input_data):
        # Tool implementation
        return "Tool result"
    
    def get_input_schema(self):
        return {
            "type": "object",
            "properties": {
                "param": {"type": "string"}
            },
            "required": ["param"]
        }
```

2. Register the tool in the registry:
```python
from unified_agent.tools import ToolRegistry

registry = ToolRegistry()
registry.register_tool(MyTool())
```

### Adding New Providers

1. Create a provider class implementing `ProviderInterface`:
```python
from unified_agent.core import ProviderInterface

class MyProvider(ProviderInterface):
    async def create_message(self, messages, tools=None, **kwargs):
        # Provider implementation
        pass
    
    def get_tool_schema(self, tools):
        # Convert tools to provider format
        pass
```

2. Add the provider to the core agent:
```python
# In core.py, add to _create_provider method
elif self.config.provider == ProviderType.MY_PROVIDER:
    return MyProvider(self.config)
```

## Examples

### Web Scraping with Computer Use

```python
from unified_agent import ComputerUseAgent, AgentConfig, ProviderType

config = AgentConfig(
    provider=ProviderType.OPENAI,
    enable_computer_use=True,
    computer_type="local-playwright",
    start_url="https://example.com"
)

agent = ComputerUseAgent(config)

# The agent can now navigate, click, and extract information
response = await agent.run("Go to the homepage and find the main navigation menu")
```

### Code Analysis with Code Execution

```python
from unified_agent import UnifiedAgent, AgentConfig, ProviderType

config = AgentConfig(
    provider=ProviderType.CLAUDE,
    enable_code_execution=True,
    system_prompt="You are a Python code analyzer. Analyze and improve the provided code."
)

agent = UnifiedAgent(config)

response = agent.run("""
Analyze this code:
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""")
```

## Environment Variables

- `ANTHROPIC_API_KEY`: Claude API key
- `OPENAI_API_KEY`: OpenAI API key
- `DEBUG`: Enable debug mode
- `VERBOSE`: Enable verbose logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Open an issue on GitHub
- Check the documentation
- Review the examples

## Roadmap

- [ ] Integration with actual computer use implementations
- [ ] Additional provider support (Google, Azure, etc.)
- [ ] Web UI interface
- [ ] Plugin system for custom tools
- [ ] Multi-agent coordination
- [ ] Advanced computer use capabilities
