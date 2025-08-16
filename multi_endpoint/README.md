# Multi-Endpoint Deployment

This directory contains prototype command line endpoints for different languages.
Each endpoint reads a mock API key from the `MOCK_API_KEY` environment variable
and simply echoes a message provided on the command line.

## Layout

- `python_cli/main.py` – Python implementation
- `java_cli/Main.java` – Java implementation
- `mojo_cli/main.mojo` – Mojo implementation

These examples are placeholders to demonstrate how language-specific entry
points can be organized for deployment. Real API keys and request logic can be
added later.
