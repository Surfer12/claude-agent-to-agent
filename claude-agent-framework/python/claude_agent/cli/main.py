#!/usr/bin/env python3
"""Claude Agent Framework CLI"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

import click
from anthropic import Anthropic

from ..core import Agent, AgentConfig, load_config
from ..tools import get_available_tools, get_tool
from ..version import __version__


@click.group()
@click.version_option(version=__version__)
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """Claude Agent Framework CLI
    
    A comprehensive framework for building Claude-powered agents with beta tool support.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Load configuration
    if config:
        ctx.obj['config'] = AgentConfig.from_file(config)
    else:
        ctx.obj['config'] = load_config()
    
    # Override verbose setting if provided
    if verbose:
        ctx.obj['config'].verbose = True


@cli.command()
@click.option('--prompt', '-p', help='Single prompt to send to the agent')
@click.option('--file', '-f', type=click.File('r'), help='Read prompt from file')
@click.option('--tools', '-t', multiple=True, help='Enable specific tools (can be used multiple times)')
@click.option('--model', '-m', help='Claude model to use')
@click.option('--max-tokens', type=int, help='Maximum tokens to generate')
@click.option('--temperature', type=float, help='Sampling temperature')
@click.option('--api-key', help='Anthropic API key')
@click.pass_context
def chat(ctx, prompt, file, tools, model, max_tokens, temperature, api_key):
    """Start a chat session with Claude"""
    config = ctx.obj['config']
    
    # Override config with command line options
    if model:
        config.model_config.model = model
    if max_tokens:
        config.model_config.max_tokens = max_tokens
    if temperature is not None:
        config.model_config.temperature = temperature
    if api_key:
        config.api_key = api_key
    
    # Get input
    if prompt:
        user_input = prompt
    elif file:
        user_input = file.read()
    else:
        # Interactive mode
        return asyncio.run(interactive_session(config, tools))
    
    # Single prompt mode
    return asyncio.run(single_prompt(config, tools, user_input))


@cli.command()
@click.option('--tools', '-t', multiple=True, help='Enable specific tools')
@click.pass_context
def interactive(ctx, tools):
    """Start an interactive chat session"""
    config = ctx.obj['config']
    return asyncio.run(interactive_session(config, tools))


@cli.command()
@click.pass_context
def list_tools(ctx):
    """List all available tools"""
    tools = get_available_tools()
    
    if not tools:
        click.echo("No tools available")
        return
    
    click.echo("Available tools:")
    for tool_name in sorted(tools):
        try:
            tool = get_tool(tool_name)
            click.echo(f"  {tool_name}: {tool.description}")
        except Exception as e:
            click.echo(f"  {tool_name}: Error loading tool - {e}")


@cli.command()
@click.argument('tool_name')
@click.pass_context
def tool_info(ctx, tool_name):
    """Show detailed information about a specific tool"""
    try:
        tool = get_tool(tool_name)
        
        click.echo(f"Tool: {tool.name}")
        click.echo(f"Description: {tool.description}")
        click.echo(f"Input Schema:")
        
        import json
        schema_json = json.dumps(tool.input_schema, indent=2)
        click.echo(schema_json)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def generate_config(ctx, output):
    """Generate a sample configuration file"""
    config = AgentConfig()
    
    if output:
        config.to_file(output)
        click.echo(f"Configuration saved to: {output}")
    else:
        # Print to stdout
        import yaml
        data = {
            'agent': {
                'name': config.name,
                'system_prompt': config.system_prompt,
                'verbose': config.verbose,
            },
            'model': {
                'model': config.model_config.model,
                'max_tokens': config.model_config.max_tokens,
                'temperature': config.model_config.temperature,
            },
            'tools': {
                'enabled': config.enabled_tools,
            }
        }
        click.echo(yaml.dump(data, default_flow_style=False, indent=2))


async def create_agent_with_tools(config: AgentConfig, tool_names: tuple) -> Agent:
    """Create an agent with specified tools."""
    # Determine which tools to enable
    if not tool_names:
        tool_names = config.enabled_tools
    
    if "all" in tool_names:
        available_tools = get_available_tools()
        tool_names = available_tools
    
    # Create tool instances
    tools = []
    for tool_name in tool_names:
        try:
            # Get tool configuration from config
            tool_config = config.tool_config.get(tool_name, {})
            tool = get_tool(tool_name, **tool_config)
            tools.append(tool)
        except Exception as e:
            if config.verbose:
                click.echo(f"Warning: Could not load tool {tool_name}: {e}", err=True)
    
    # Create agent
    agent = Agent(config=config, tools=tools)
    return agent


async def interactive_session(config: AgentConfig, tool_names: tuple):
    """Run an interactive session with the agent."""
    try:
        agent = await create_agent_with_tools(config, tool_names)
    except Exception as e:
        click.echo(f"Error creating agent: {e}", err=True)
        return
    
    click.echo(f"Starting interactive session with {agent.name}...")
    click.echo("Type 'exit' or 'quit' to end the session.")
    click.echo("Type 'clear' to clear conversation history.")
    click.echo("Type 'help' for more commands.")
    click.echo("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ("exit", "quit"):
                click.echo("Ending session.")
                break
                
            if user_input.lower() == "clear":
                agent.history.clear()
                click.echo("Conversation history cleared.")
                continue
                
            if user_input.lower() == "help":
                click.echo("Commands:")
                click.echo("  exit, quit - End the session")
                click.echo("  clear - Clear conversation history")
                click.echo("  help - Show this help")
                continue
                
            if not user_input.strip():
                continue
                
            response = await agent.run_async(user_input)
            
            # Format response
            output_parts = []
            for block in response.content:
                if block.type == "text":
                    output_parts.append(block.text)
            
            if output_parts:
                click.echo(f"\nClaude: {' '.join(output_parts)}")
            else:
                click.echo("\nClaude: [No text response]")
            
        except KeyboardInterrupt:
            click.echo("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            click.echo(f"\nError: {str(e)}", err=True)


async def single_prompt(config: AgentConfig, tool_names: tuple, prompt: str):
    """Run a single prompt through the agent."""
    try:
        agent = await create_agent_with_tools(config, tool_names)
        response = await agent.run_async(prompt)
        
        # Format response
        output_parts = []
        for block in response.content:
            if block.type == "text":
                output_parts.append(block.text)
        
        if output_parts:
            click.echo(' '.join(output_parts))
        else:
            click.echo("[No text response]")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
