# Unified Agent System with Swarm Integration

A provider-agnostic agent framework that supports both Claude and OpenAI backends, with unified CLI, computer use, and multi-agent swarm capabilities.

## 🌐 Multi-Endpoint Deployment

Prototype command line endpoints for Python, Java, and Mojo are available under
`multi_endpoint/`. Each reads a mock API key from the `MOCK_API_KEY`
environment variable and simply echoes input, providing a starting point for
language-specific deployments.

## 🎯 Pixi Integration Complete

### **📋 New Files Created**
java-swarm/
├── pixi.toml                      # Main Pixi configuration
├── PIXI_USAGE.md                  # Complete Pixi usage guide
├── scripts/
│   ├── setup-env.sh              # Environment setup script
│   └── validate-pixi.sh          # Pixi configuration validator
└── examples/
    └── custom-pixi-tasks.toml     # Custom task examples


### **🚀 Available Pixi Commands**

#### **Build & Development**
bash
pixi run build              # Build the project
pixi run compile            # Compile source only
pixi run test              # Run unit tests
pixi run clean             # Clean build artifacts
pixi run rebuild           # Clean and rebuild
pixi run dev               # Development mode


#### **Interactive Chat**
bash
pixi run interactive              # Basic interactive mode
pixi run interactive-debug        # Interactive with debug
pixi run interactive-stream       # Interactive with streaming
pixi run interactive-stream-debug # Interactive with streaming + debug


#### **Single Messages**
bash
pixi run chat "Your message"           # Send single message
pixi run chat-stream "Your message"    # Send with streaming
pixi run chat-debug "Your message"     # Send with debug info


#### **Specialized Agents**
bash
pixi run math-bot          # Mathematics expert
pixi run code-bot          # Programming expert
pixi run story-bot         # Creative storyteller (with streaming)


#### **Model Selection**
bash
pixi run gpt4              # Use GPT-4o
pixi run gpt4-mini         # Use GPT-4o-mini
pixi run gpt35             # Use GPT-3.5-turbo


#### **Examples & Demos**
bash
pixi run streaming-demo    # Demonstrate streaming
pixi run calculator-demo   # Demonstrate function calling
pixi run https-demo        # Demonstrate HTTPS configuration


#### **Quick Start**
bash
pixi run quick-start       # Build and run interactively
pixi run quick-stream      # Build and run with streaming


### **🛠 Key Features**

1. Automatic Dependency Management: Pixi handles Java 17+ and Maven installation
2. Environment Isolation: Each project has its own isolated environment
3. Cross-Platform: Works on macOS, Linux, and Windows
4. Task Dependencies: Tasks automatically ensure prerequisites are met
5. Multiple Environments: Support for dev, test, and production environments
6. Custom Tasks: Easy to add custom agent configurations and workflows

### **📖 Usage Examples**

#### **Quick Start**
bash
# Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Setup project
pixi install

# Set API key
export OPENAI_API_KEY="your-key-here"

# Start chatting
pixi run quick-start


#### **Development Workflow**
bash
# Build and test
pixi run rebuild

# Start development mode
pixi run dev

# Test streaming
pixi run interactive-stream

# Run demos
pixi run streaming-demo


#### **Specialized Use Cases**
bash
# Math tutoring
pixi run math-bot

# Code assistance
pixi run code-bot

# Creative writing with streaming
pixi run story-bot


### **🔧 Advanced Features**

#### **Multiple Environments**
bash
pixi run -e dev interactive     # Development environment
pixi run -e test unit-tests     # Testing environment
pixi run -e prod interactive    # Production environment


#### **Custom Tasks**
Users can easily add custom tasks to pixi.toml:
toml
[tasks]
my-agent = "java -jar target/java-swarm-1.0.0.jar --interactive --agent-name MyBot --instructions 'Custom instructions'"


#### **Task Dependencies**
Tasks automatically handle dependencies:
toml
[tasks]
chat = { cmd = "java -jar target/java-swarm-1.0.0.jar --input", depends_on = ["ensure-built"] }


### **📚 Documentation**

1. PIXI_USAGE.md: Complete reference for all Pixi commands
2. Updated README.md: Includes Pixi as the recommended installation method
3. Updated QUICKSTART.md: Pixi-first approach with fallback to manual
4. Custom task examples: Shows how to extend functionality

### **✅ Benefits of Pixi Integration**

1. Simplified Setup: One command installs everything needed
2. Consistent Environment: Same environment across all developers
3. Easy Commands: Memorable, short commands instead of long Java CLI
4. Cross-Platform: Works identically on all operating systems
5. Dependency Management: Automatic handling of Java and Maven versions
6. Task Organization: Logical grouping of related commands
7. Environment Isolation: No conflicts with system-installed tools

### **🎯 Example Workflows**

#### **New User Experience**
bash
# Complete setup in 3 commands
curl -fsSL https://pixi.sh/install.sh | bash
pixi install
pixi run quick-start


#### **Daily Development**
bash
pixi run dev               # Start development
pixi run test              # Run tests
pixi run streaming-demo    # Test features


#### **Production Usage**
bash
pixi run -e prod build     # Production build
pixi run interactive       # Run application

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

## Testing

To run the tests, use the following command:

```bash
PYTHONPATH=.:swarm pytest
```

This will run all the tests in the `tests` directory.


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
