# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
A CLI application that creates Claude-powered agents with tool capabilities and MCP server integration. The project includes multiple sub-projects including a computer-use demo and financial data analyst interface.

## Common Commands
- `pixi install` - Install all dependencies
- `pixi run cli` - Run the main CLI (equivalent to `python cli.py`)
- `pixi run example-simple` - Run simple CLI example
- `pixi run example-mcp` - Run MCP tools example
- `pip install -e .` - Install as editable package for development

## Architecture Overview
The core architecture centers around the `Agent` class in `agents/agent.py` which:
- Manages message history with automatic context window truncation
- Handles tool execution through async loops
- Integrates with MCP servers for dynamic tool loading
- Uses the Anthropic client with configurable model parameters

Key architectural components:
- **Agent**: Main orchestrator with message history and tool management
- **Tools**: Base tool interface with async execution (`agents/tools/base.py`)
- **MCP Integration**: Dynamic tool loading from Model Context Protocol servers
- **Message History**: Context-aware history management with token counting
- **CLI**: Main entry point with argument parsing and interactive mode

## Tool System
Tools inherit from the base `Tool` class and implement:
- `name`, `description`, and `input_schema` properties
- Async `execute()` method for tool functionality
- Conversion to Claude API format via `to_dict()`

Available built-in tools: `think`, `file_read`, `file_write`

## MCP Server Integration
The agent can connect to external MCP servers for additional tools:
- Servers are configured as a list of connection details
- Tools are dynamically loaded and made available to the agent
- MCP connections are managed through `AsyncExitStack` for proper cleanup

## Environment Requirements
- Python 3.10+ (configured for 3.13+ in pixi.toml)
- `ANTHROPIC_API_KEY` environment variable
- Pixi package manager for dependency management