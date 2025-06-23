#!/usr/bin/env python3
"""
Claude Agent-to-Agent CLI:
An Interdisciplinary Tool for Cognitive-Computational Interaction

This CLI represents a sophisticated interface for agent-based computational
exploration, bridging neurocognitive insights with advanced computational
methodologies.

Design Principles:
- Cognitive Flexibility: Dynamic agent configuration
- Methodological Rigor: Systematic interaction tracking
- Interdisciplinary Integration: Support for multiple interaction modalities
"""

import os
import sys
import argparse
import asyncio
import logging
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

# Extend system path to ensure local module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import Agent, ModelConfig
from agents.tools.think import ThinkTool
from agents.tools.file_tools import FileReadTool, FileWriteTool
from agents.tools.computer_use import ComputerUseTool
from agents.tools.code_execution import CodeExecutionTool, CodeExecutionWithFilesTool, is_model_supported
from agents.utils.connections import setup_mcp_connections

# Configure advanced logging with cognitive performance tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('claude_agent_interactions.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InteractionMetrics:
    """Immutable container for interaction metrics."""
    total_interactions: int = 0
    successful_interactions: int = 0
    average_response_time: float = 0.0
    last_interaction_time: Optional[datetime] = None


@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for agent settings."""
    name: str
    system_prompt: str
    tools: tuple  # Changed from FrozenSet to tuple for tool objects
    verbose: bool
    model: str


class CognitiveAgentCLI:
    """
    A sophisticated CLI for agent interactions that integrates
    computational tools with cognitive performance insights.
    """

    def __init__(
        self,
        name: str = "CognitiveAgent",
        system_prompt: Optional[str] = None,
        tools: Optional[List] = None,
        verbose: bool = False,
        model: str = 'claude-3-5-sonnet-20240620'
    ):
        """
        Initialize the Cognitive Agent CLI with flexible configuration.

        Args:
            name: Identifier for the agent interaction session
            system_prompt: Custom system-level instructions for agent behavior
            tools: Additional computational tools to enable
            verbose: Enable detailed logging and interaction tracking
            model: Claude model to use for interactions
        """
        # Default system prompt with cognitive performance framing
        default_prompt = (
            "You are a cognitive enhancement agent designed to support "
            "interdisciplinary problem-solving and computational exploration. "
            "Approach each interaction with systematic analytical thinking, "
            "drawing insights from multiple domains of knowledge."
        )

        # Configure default and optional tools
        default_tools = (
            ThinkTool(),
            FileReadTool(),
            FileWriteTool()
        )

        # Create immutable configuration
        self.config = AgentConfig(
            name=name,
            system_prompt=system_prompt or default_prompt,
            tools=tuple(tools) if tools else default_tools,
            verbose=verbose,
            model=model
        )

        # Initialize metrics with immutable container
        self.metrics = InteractionMetrics()

    async def interactive_session(self) -> None:
        """
        Launch an interactive cognitive agent session with
        advanced interaction tracking.
        """
        logger.info(f"Initiating Cognitive Agent Session: {self.config.name}")

        try:
            agent = Agent(
                name=self.config.name,
                system=self.config.system_prompt,
                tools=list(self.config.tools),
                verbose=self.config.verbose
            )

            while True:
                try:
                    user_input = input("\n🧠 Cognitive Agent > ")

                    if user_input.lower() in ['exit', 'quit', 'q']:
                        break

                    # Track interaction performance
                    start_time = asyncio.get_event_loop().time()

                    response = await agent.run_async(user_input)

                    end_time = asyncio.get_event_loop().time()
                    response_time = end_time - start_time

                    # Update metrics with new immutable instance
                    self.metrics = InteractionMetrics(
                        total_interactions=self.metrics.total_interactions + 1,
                        successful_interactions=(
                            self.metrics.successful_interactions + 1
                        ),
                        average_response_time=(
                            0.9 * self.metrics.average_response_time +
                            0.1 * response_time
                        ) if self.metrics.total_interactions > 0 else response_time,
                        last_interaction_time=datetime.now()
                    )

                    # Output response with cognitive performance context
                    print("\n🤖 Response:")
                    print(format_response(response))

                except Exception as interaction_error:
                    logger.error(f"Interaction Error: {interaction_error}")
                    print(f"⚠️ Error in interaction: {interaction_error}")
                    # Update metrics for failed interaction
                    self.metrics = InteractionMetrics(
                        total_interactions=self.metrics.total_interactions + 1,
                        successful_interactions=self.metrics.successful_interactions,
                        average_response_time=self.metrics.average_response_time,
                        last_interaction_time=datetime.now()
                    )

        except KeyboardInterrupt:
            print("\n\nCognitive Agent Session Terminated.")

        finally:
            # Log session summary
            logger.info("Cognitive Agent Session Summary:")
            logger.info(f"Total Interactions: {self.metrics.total_interactions}")
            logger.info(
                f"Successful Interactions: {self.metrics.successful_interactions}"
            )
            logger.info(
                f"Average Response Time: "
                f"{self.metrics.average_response_time:.4f} seconds"
            )
            if self.metrics.last_interaction_time:
                logger.info(
                    f"Last Interaction: {self.metrics.last_interaction_time}"
                )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cognitive Agent CLI: Advanced Computational Interaction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Configuration parameters with cognitive performance framing
    parser.add_argument(
        '--name',
        default='CognitiveAgent',
        help='Custom name for the cognitive agent session'
    )
    parser.add_argument(
        '--system-prompt',
        help='Custom system-level instructions for agent behavior'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable detailed interaction logging and tracking'
    )
    ##TODO column alignment below
    parser.add_argument(
        '--model',
        default='claude-3-5-sonnet-20240620',
        choices=[
           'claude-opus-4-20250514',
            'claude-sonnet-4-20250514'
            'claude-3-7-sonnet-20250219', 
        ],
        help='Select the Claude model for cognitive interactions'
    
    # Tool configuration
    tool_group = parser.add_argument_group("Tool options")
    tool_group.add_argument(
        "--tools", 
        nargs="+", 
        choices=["think", "file_read", "file_write", "computer_use", "code_execution", "all"],
        default=["all"],
        help="Enable specific tools"
    )
    tool_group.add_argument(
        "--mcp-server", 
        action="append", 
        help="MCP server URL (can be specified multiple times)"
    )
    
    # Computer use configuration
    computer_group = parser.add_argument_group("Computer use options")
    computer_group.add_argument(
        "--display-width", 
        type=int, 
        default=1024, 
        help="Display width in pixels for computer use"
    )
    computer_group.add_argument(
        "--display-height", 
        type=int, 
        default=768, 
        help="Display height in pixels for computer use"
    )
    computer_group.add_argument(
        "--display-number", 
        type=int, 
        help="X11 display number for computer use"
    )
    computer_group.add_argument(
        "--computer-tool-version",
        choices=["computer_20241022", "computer_20250124"],
        default="computer_20250124",
        help="Computer use tool version"
    )
    
    # Code execution configuration
    code_group = parser.add_argument_group("Code execution options")
    code_group.add_argument(
        "--enable-file-support",
        action="store_true",
        help="Enable file upload support for code execution"
    )
    
    # API configuration
    api_group = parser.add_argument_group("API options")
    api_group.add_argument(
        "--api-key", 
        help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Validate input mode
    input_modes = sum(
        bool(mode) for mode in [args.prompt, args.interactive, args.file]
    )
    if input_modes != 1:
        parser.error("Exactly one input mode (--prompt, --interactive, or --file) is required")
    
    # Set system prompt if not provided
    if not args.system:
        args.system = "You are Claude, an AI assistant. Be concise and helpful."
    
    return args


def get_enabled_tools(tool_names: List[str], args) -> List:
    """Get enabled tool instances based on names."""
    tools = []
    
    if "all" in tool_names or "think" in tool_names:
        tools.append(ThinkTool())
    
    if "all" in tool_names or "file_read" in tool_names:
        tools.append(FileReadTool())
    
    if "all" in tool_names or "file_write" in tool_names:
        tools.append(FileWriteTool())
        
    if "all" in tool_names or "computer_use" in tool_names:
        tools.append(ComputerUseTool(
            display_width_px=args.display_width,
            display_height_px=args.display_height,
            display_number=args.display_number,
            tool_version=args.computer_tool_version,
        ))
    
    if "all" in tool_names or "code_execution" in tool_names:
        # Check if model supports code execution
        if not is_model_supported(args.model):
            print(f"Warning: Model {args.model} may not support code execution. Supported models:")
            for model in ["claude-opus-4-20250514", "claude-sonnet-4-20250514", 
                         "claude-3-7-sonnet-20250219", "claude-3-5-haiku-latest"]:
                print(f"  - {model}")
        
        if args.enable_file_support:
            tools.append(CodeExecutionWithFilesTool())
        else:
            tools.append(CodeExecutionTool())
    
    return tools


def setup_mcp_servers(server_urls: Optional[List[str]]) -> List[dict]:
    """Configure MCP server connections."""
    if not server_urls:
        return []
    
    return [
        {
            "url": url,
            "connection_type": "sse" if url.startswith("http") else "stdio",
        }
        for url in server_urls
    ]


def format_response(response):
    """Format agent response for display."""
    output = []
    
    for block in response.content:
        if block.type == "text":
            output.append(block.text)
    
    return "\n".join(output)


async def handle_interactive_session(agent: Agent):
    """Run an interactive session with the agent."""
    print(f"Starting interactive session with {agent.name}...")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'clear' to clear conversation history.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ("exit", "quit"):
                print("Ending session.")
                break
                
            if user_input.lower() == "clear":
                agent.history.clear()
                print("Conversation history cleared.")
                continue
                
            if not user_input.strip():
                continue
                
            response = await agent.run_async(user_input)
            print("\nClaude:", format_response(response))
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


async def handle_single_prompt(agent: Agent, prompt: str):
    """Run a single prompt through the agent."""
    try:
        response = await agent.run_async(prompt)
        print(format_response(response))
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


async def handle_file_input(agent: Agent, file_path: str):
    """Run agent with input from a file."""
    try:
        if file_path == "-":
            content = sys.stdin.read()
        else:
            with open(file_path, "r") as f:
                content = f.read()
                
        response = await agent.run_async(content)
        print(format_response(response))
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


async def main_async():
    """Async entry point for the CLI."""
    args = parse_args()
    
    # Configure API client
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: Anthropic API key not provided. Set ANTHROPIC_API_KEY environment "
            "variable or use --api-key",
            file=sys.stderr,
        )
        sys.exit(1)
    
    client = Anthropic(api_key=api_key)
    
    # Configure agent
    config = ModelConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    tools = get_enabled_tools(args.tools, args)
    mcp_servers = setup_mcp_servers(args.mcp_server)
    
    agent = Agent(
        name=args.name,
        system_prompt=args.system_prompt,
        tools=tools,
        verbose=args.verbose,
        model=args.model
    )

    # Run based on input mode
    if args.interactive:
        asyncio.run(cognitive_cli.interactive_session())
    elif args.prompt:
        asyncio.run(handle_single_prompt_async(args, tools))
    elif args.file:
        asyncio.run(handle_file_input_async(args, tools))


if __name__ == "__main__":
    main()
