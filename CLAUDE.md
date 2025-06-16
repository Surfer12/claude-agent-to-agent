# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This project, `claude-agent-to-agent`, is a Python-based application that uses the Anthropic API to facilitate agent-to-agent communication between Claude AI assistants. It's currently in the initial setup phase, with the environment configured using Pixi package manager.

## Environment Management
- `pixi install` - Install dependencies from pixi.toml
- `pixi run python [script.py]` - Run Python scripts using the managed environment
- `pixi add [package]` - Add a new dependency to the project
- `pixi shell` - Activate the Pixi environment for interactive use

## Development Guidelines
- Use Python 3.13+ features when applicable (as specified in the environment)
- Follow PEP 8 style guide for Python code
- Include type hints for function parameters and return values
- Use the Anthropic Python SDK for all API calls (available via the dependency)

## Project Structure (Recommended)
- `src/` - Main source code directory
- `tests/` - Test files
- `examples/` - Example usage scripts
- `README.md` - Project documentation

## Working with the Anthropic API
- Use version-specific imports from the anthropic package
- Store API keys in environment variables, never hardcode them
- Follow Anthropic's rate limiting guidelines for API calls
- Handle API errors gracefully with appropriate error handling