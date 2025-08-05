#!/usr/bin/env python3
"""
Enhanced Claude Agent-to-Agent CLI:
User-friendly, interactive interface with intuitive commands
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
from agents.tools.code_execution import (
    CodeExecutionTool,
    CodeExecutionWithFilesTool,
    is_model_supported
)

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InteractionMetrics:
    """Immutable container for interaction metrics."""
    total_interactions: int = 0
    successful_interactions: int = 0
    average_response_time: float = 0.0
    last_interaction_time: Optional[datetime] = None


class EnhancedCLI:
    """Enhanced, user-friendly CLI for Claude Agent interactions."""

    def __init__(
        self,
        name: str = "Claude",
        system_prompt: Optional[str] = None,
        tools: Optional[List] = None,
        verbose: bool = False,
        model: str = 'claude-3-5-sonnet-20240620'
    ):
        self.name = name
        self.system_prompt = system_prompt or (
            "You are Claude, a helpful AI assistant. Be conversational, "
            "clear, and helpful. Use your tools when appropriate."
        )
        self.tools = tools or [ThinkTool(), FileReadTool(), FileWriteTool()]
        self.verbose = verbose
        self.model = model
        self.metrics = InteractionMetrics()

    async def interactive_session(self) -> None:
        """Launch enhanced interactive session."""
        self._show_welcome()
        
        try:
            agent = Agent(
                name=self.name,
                system=self.system_prompt,
                tools=self.tools,
                verbose=self.verbose
            )

            while True:
                try:
                    user_input = input("\n💬 ").strip()

                    if not user_input:
                        continue

                    if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
                        print("\n👋 Goodbye!")
                        break
                    elif user_input.lower() in ['help', '?']:
                        self._show_help()
                        continue
                    elif user_input.lower() == 'tools':
                        self._show_tools()
                        continue
                    elif user_input.lower() == 'clear':
                        agent.history.clear()
                        print("🗑️  Chat history cleared!")
                        continue
                    elif user_input.lower() == 'stats':
                        self._show_stats()
                        continue

                    # Process the request
                    print("🤔 ", end="", flush=True)
                    start_time = asyncio.get_event_loop().time()

                    response = await agent.run_async(user_input)

                    end_time = asyncio.get_event_loop().time()
                    response_time = end_time - start_time

                    # Update metrics
                    self.metrics = InteractionMetrics(
                        total_interactions=self.metrics.total_interactions + 1,
                        successful_interactions=self.metrics.successful_interactions + 1,
                        average_response_time=(
                            0.9 * self.metrics.average_response_time + 0.1 * response_time
                        ) if self.metrics.total_interactions > 0 else response_time,
                        last_interaction_time=datetime.now()
                    )

                    # Display response
                    print(f"\r🤖 Claude:")
                    if hasattr(response, 'content'):
                        for content in response.content:
                            if hasattr(content, 'text'):
                                print(content.text)
                    else:
                        print(str(response))
                    
                    if response_time > 2:
                        print(f"\n⏱️  ({response_time:.1f}s)")

                except Exception as e:
                    print(f"\n❌ Oops! {str(e)}")
                    print("💡 Try rephrasing or type 'help'")
                    
                    self.metrics = InteractionMetrics(
                        total_interactions=self.metrics.total_interactions + 1,
                        successful_interactions=self.metrics.successful_interactions,
                        average_response_time=self.metrics.average_response_time,
                        last_interaction_time=datetime.now()
                    )

        except KeyboardInterrupt:
            print("\n\n👋 Session ended!")
        finally:
            self._show_summary()

    def _show_welcome(self):
        """Show welcome message."""
        print("🤖 " + "="*50)
        print("   CLAUDE AGENT CLI")
        print("   Enhanced Interactive Mode")
        print("="*54)
        print()
        print("💡 Quick commands:")
        print("   help    - Show help")
        print("   tools   - List available tools") 
        print("   clear   - Clear chat history")
        print("   stats   - Show session stats")
        print("   exit    - End session")
        print()
        print(f"🔧 Tools: {', '.join([t.__class__.__name__.replace('Tool', '') for t in self.tools])}")
        print(f"🧠 Model: {self.model}")
        print("-" * 54)
        print("Just type your questions naturally!")

    def _show_help(self):
        """Show help information."""
        print("\n🆘 HELP")
        print("="*20)
        print("Commands:")
        print("  help/? - This help")
        print("  tools  - List tools")
        print("  clear  - Clear history")
        print("  stats  - Show stats")
        print("  exit   - Quit")
        print()
        print("💬 Examples:")
        print("  • What files are here?")
        print("  • Help me code a function")
        print("  • What's the weather like?")
        print("  • Calculate 15 * 23")

    def _show_tools(self):
        """Show active tools."""
        print("\n🔧 ACTIVE TOOLS")
        print("="*20)
        for i, tool in enumerate(self.tools, 1):
            name = tool.__class__.__name__.replace('Tool', '')
            print(f"  {i}. {name}")

    def _show_stats(self):
        """Show session statistics."""
        print("\n📊 SESSION STATS")
        print("="*20)
        print(f"Interactions: {self.metrics.total_interactions}")
        print(f"Successful: {self.metrics.successful_interactions}")
        if self.metrics.total_interactions > 0:
            rate = (self.metrics.successful_interactions / self.metrics.total_interactions) * 100
            print(f"Success rate: {rate:.1f}%")
        print(f"Avg response: {self.metrics.average_response_time:.1f}s")

    def _show_summary(self):
        """Show session summary."""
        if self.metrics.total_interactions > 0:
            print("\n📊 SESSION COMPLETE")
            print("="*25)
            print(f"Total interactions: {self.metrics.total_interactions}")
            print(f"Success rate: {(self.metrics.successful_interactions/self.metrics.total_interactions)*100:.1f}%")
            print(f"Average response time: {self.metrics.average_response_time:.1f}s")


def get_tools(tool_names: List[str], args) -> List:
    """Get enabled tool instances."""
    tools = []

    if "all" in tool_names or "think" in tool_names:
        tools.append(ThinkTool())
    if "all" in tool_names or "file_read" in tool_names:
        tools.append(FileReadTool())
    if "all" in tool_names or "file_write" in tool_names:
        tools.append(FileWriteTool())
    if "all" in tool_names or "computer_use" in tool_names:
        tools.append(ComputerUseTool(
            display_width_px=getattr(args, 'display_width', 1024),
            display_height_px=getattr(args, 'display_height', 768),
            display_number=getattr(args, 'display_number', None),
            tool_version=getattr(args, 'computer_tool_version', 'computer_20250124'),
        ))
    if "all" in tool_names or "code_execution" in tool_names:
        if getattr(args, 'enable_file_support', False):
            tools.append(CodeExecutionWithFilesTool())
        else:
            tools.append(CodeExecutionTool())

    return tools


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Claude Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Simple, intuitive options
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        default=True,
        help='Interactive mode (default)'
    )
    parser.add_argument(
        '--prompt', '-p',
        help='Single prompt to process'
    )
    parser.add_argument(
        '--tools', '-t',
        nargs='+',
        choices=['think', 'file_read', 'file_write', 'computer_use', 'code_execution', 'all'],
        default=['think', 'file_read', 'file_write'],
        help='Tools to enable (default: think file_read file_write)'
    )
    parser.add_argument(
        '--model', '-m',
        default='claude-3-5-sonnet-20240620',
        help='Claude model to use'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Check API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        print("💡 Add to your shell profile: export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)

    # Get tools
    tools = get_tools(args.tools, args)

    # Create CLI instance
    cli = EnhancedCLI(
        tools=tools,
        verbose=args.verbose,
        model=args.model
    )

    # Run based on mode
    if args.prompt:
        # Single prompt mode
        async def run_prompt():
            agent = Agent(
                name="Claude",
                system=cli.system_prompt,
                tools=tools,
                verbose=args.verbose
            )
            response = await agent.run_async(args.prompt)
            if hasattr(response, 'content'):
                for content in response.content:
                    if hasattr(content, 'text'):
                        print(content.text)
            else:
                print(str(response))
        
        asyncio.run(run_prompt())
    else:
        # Interactive mode
        asyncio.run(cli.interactive_session())


if __name__ == "__main__":
    main()
