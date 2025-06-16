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