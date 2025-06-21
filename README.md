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
- `computer_use`: A tool for desktop interaction via screenshots and input control (Beta)
- `code_execution`: A tool for executing Python code in a secure sandbox environment (Beta)
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

# Use computer use tool with custom display settings
claude-agent --interactive --tools computer_use --display-width 1280 --display-height 800

# Computer use with specific tool version
claude-agent --interactive --tools computer_use --computer-tool-version computer_20241022

# Use code execution tool
claude-agent --interactive --tools code_execution

# Code execution with file support
claude-agent --interactive --tools code_execution --enable-file-support
```

### Computer Use Tool Usage

```bash
# Enable computer use tool with default settings
claude-agent --interactive --tools computer_use

# Configure display dimensions
claude-agent --interactive --tools computer_use --display-width 1280 --display-height 800

# Use specific tool version for Claude Sonnet 3.5
claude-agent --interactive --tools computer_use --computer-tool-version computer_20241022

# Use with specific display number (X11 environments)
claude-agent --interactive --tools computer_use --display-number 1

# Example prompts for computer use:
# "Take a screenshot of the current desktop"
# "Click on the button at coordinates 100, 200"
# "Type 'Hello World' into the current window"
# "Press Ctrl+C to copy"
# "Scroll down 3 times in the current window"
```

## Code Execution Tool (Beta)

The code execution tool allows Claude to execute Python code in a secure, sandboxed environment. Claude can analyze data, create visualizations, perform complex calculations, and process uploaded files directly within the API conversation.

### Supported Models

The code execution tool is available on:
- Claude Opus 4 (`claude-opus-4-20250514`)
- Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- Claude Sonnet 3.7 (`claude-3-7-sonnet-20250219`)
- Claude Haiku 3.5 (`claude-3-5-haiku-latest`)

### Features

- **Secure sandbox**: Code runs in an isolated Linux container
- **Pre-installed libraries**: pandas, numpy, matplotlib, scikit-learn, and more
- **File processing**: Support for CSV, Excel, JSON, images, and other formats
- **Data visualization**: Create charts and graphs with matplotlib
- **No internet access**: Completely isolated for security

### Runtime Environment

- **Python version**: 3.11.12
- **Memory**: 1GiB RAM
- **Disk space**: 5GiB workspace storage
- **CPU**: 1 CPU
- **Container lifetime**: 1 hour

### Usage Examples

```bash
# Enable code execution tool
claude-agent --interactive --tools code_execution

# With file upload support
claude-agent --interactive --tools code_execution --enable-file-support

# Example prompts for code execution:
# "Calculate the mean and standard deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
# "Create a matplotlib chart showing sales data over time"
# "Analyze this CSV file and provide summary statistics"
# "Generate a random dataset and perform linear regression"
```

### Pre-installed Libraries

- **Data Science**: pandas, numpy, scipy, scikit-learn, statsmodels
- **Visualization**: matplotlib
- **File Processing**: pyarrow, openpyxl, xlrd, pillow
- **Math & Computing**: sympy, mpmath
- **Utilities**: tqdm, python-dateutil, pytz, joblib

### Pricing

Code execution usage is tracked separately from token usage:
- **Pricing**: $0.05 per session-hour
- **Minimum billing**: 5 minutes per session
- **File preloading**: Billed even if tool isn't used when files are included

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
## Computer Use Tool (Beta)

The computer use tool enables Claude to interact with desktop environments through screenshot capabilities and mouse/keyboard control for autonomous desktop interaction.

### Beta Requirements

Computer use requires a beta header based on your Claude model:
- "computer-use-2025-01-24" (Claude 4 and 3.7 models)
- "computer-use-2024-10-22" (Claude Sonnet 3.5)

### Features

- Screenshot capture: View current display content
- Mouse control: Click, drag, and cursor movement
- Keyboard input: Text entry and keyboard shortcuts
- Desktop automation: Interact with applications and interfaces

### Model Compatibility

| Model | Tool Version | Beta Flag |
|-------|--------------|-----------|
| Claude 4 Opus & Sonnet | computer_20250124 | computer-use-2025-01-24 |
| Claude Sonnet 3.7 | computer_20250124 | computer-use-2025-01-24 |
| Claude Sonnet 3.5 | computer_20241022 | computer-use-2024-10-22 |

### Security Considerations

- Use dedicated virtual machines or containers with minimal privileges
- Avoid exposing sensitive data or credentials
- Limit internet access to allowlisted domains
- Require human confirmation for consequential actions
- Implement safeguards against prompt injection

### Quick Start

```python
import anthropic

client = anthropic.Anthropic()

response = client.beta.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[
        {
          "type": "computer_20250124",
          "name": "computer",
          "display_width_px": 1024,
          "display_height_px": 768,
          "display_number": 1,
        }
    ],
    messages=[{"role": "user", "content": "Take a screenshot of the desktop"}],
    betas=["computer-use-2025-01-24"]
)
```

### Available Actions

Basic actions (all versions):
- screenshot: Capture current display
- left_click: Click at coordinates [x, y]
- type: Enter text string
- key: Press key or key combination
- mouse_move: Move cursor to coordinates

Enhanced actions (computer_20250124):
- scroll: Directional scrolling with amount control
- left_click_drag: Click and drag between coordinates
- right_click, middle_click: Additional mouse buttons
- double_click, triple_click: Multiple clicks
- left_mouse_down, left_mouse_up: Fine-grained click control
- hold_key: Hold keys while performing other actions
- wait: Add pauses between actions

### Tool Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| type | Yes | Tool version (computer_20250124 or computer_20241022) |
| name | Yes | Must be "computer" |
| display_width_px | Yes | Display width in pixels |
| display_height_px | Yes | Display height in pixels |
| display_number | No | Display number for X11 environments |

Note: Keep display resolution at or below 1280x800 (WXGA) for optimal performance.

### Limitations

- Latency in human-AI interactions
- Computer vision accuracy and reliability
- Tool selection accuracy
- Scrolling reliability (improved in newer versions)
- Spreadsheet interaction challenges
- Limited social platform interaction
- Potential vulnerabilities to prompt injection

### Pricing

Computer use follows standard tool use pricing with additional considerations:
- System prompt overhead: 466-499 tokens
- Tool definition tokens:
  - Claude 4 / Sonnet 3.7: 735 tokens
  - Claude Sonnet 3.5: 683 tokens
- Additional costs for screenshots and tool execution results
