"""CLI for the Unified Agent System."""

import asyncio
import argparse
import logging
from typing import Optional

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'swarm'))
import importlib.util

from .core import UnifiedAgent
from .core.types import AgentConfig, ProviderType
from .tool_registry import ToolRegistry
from swarm import Swarm, Agent

def main():
    """Main entry point for CLI."""
    cli = CLIInterface()
    parser = cli.create_parser()
    args = parser.parse_args()
    
    # Run the CLI
    asyncio.run(cli.run(args))


class CLIInterface:
    """Command-line interface for the unified agent."""
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.agent: Optional[UnifiedAgent] = None
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="Unified Agent System - CLI and Computer Use Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic CLI with Claude
  python -m unified_agent_system.cli --provider claude --input "Hello, how are you?"
  
  # OpenAI with computer use
  python -m unified_agent_system.cli --provider openai --enable-computer-use --computer-type local-playwright
  
  # Interactive mode with code execution
  python -m unified_agent_system.cli --provider claude --enable-code-execution --interactive
  
  # Computer use with specific start URL
  python -m unified_agent_system.cli --provider openai --enable-computer-use --start-url https://google.com

  # Swarm interactive mode
  python -m unified_agent_system.cli --swarm-config swarm/examples/airline/configs/agents.py --initial-agent triage_agent
            """
        )
        
        # Provider settings
        parser.add_argument(
            "--provider",
            choices=["claude", "openai"],
            default="claude",
            help="AI provider to use (default: claude)"
        )
        parser.add_argument(
            "--model",
            type=str,
            help="Model to use (default: provider-specific)"
        )
        parser.add_argument(
            "--api-key",
            type=str,
            help="API key (default: from environment variables)"
        )
        
        # Agent settings
        parser.add_argument(
            "--system-prompt",
            type=str,
            default="You are a helpful AI assistant.",
            help="System prompt for the agent"
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=4096,
            help="Maximum tokens for responses"
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="Temperature for responses"
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output"
        )
        
        # Tool settings
        parser.add_argument(
            "--enable-tools",
            action="store_true",
            help="Enable basic tools"
        )
        parser.add_argument(
            "--enable-code-execution",
            action="store_true",
            help="Enable code execution tools"
        )
        parser.add_argument(
            "--enable-computer-use",
            action="store_true",
            help="Enable computer use capabilities"
        )
        # Swarm settings
        parser.add_argument(
            "--swarm-config",
            type=str,
            help="Path to the Swarm configuration file"
        )
        parser.add_argument(
            "--initial-agent",
            type=str,
            default="agent_a",
            help="Name of the initial agent in the swarm"
        )
        
        # Computer use settings
        parser.add_argument(
            "--computer-type",
            type=str,
            default="local-playwright",
            help="Computer environment type (default: local-playwright)"
        )
        parser.add_argument(
            "--start-url",
            type=str,
            default="https://bing.com",
            help="Starting URL for browser environments"
        )
        parser.add_argument(
            "--show-images",
            action="store_true",
            help="Show images during computer use"
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode"
        )
        
        # Input/output settings
        parser.add_argument(
            "--input",
            type=str,
            help="Single input to process (non-interactive mode)"
        )
        parser.add_argument(
            "--interactive",
            action="store_true",
            help="Run in interactive mode"
        )
        parser.add_argument(
            "--output-file",
            type=str,
            help="Save output to file"
        )
        
        return parser
    
    def create_config(self, args) -> AgentConfig:
        """Create agent configuration from arguments."""
        # Determine provider
        provider = ProviderType.CLAUDE if args.provider == "claude" else ProviderType.OPENAI
        
        # Set default models
        if not args.model:
            if provider == ProviderType.CLAUDE:
                args.model = "claude-3-5-sonnet-20240620"
            else:
                args.model = "gpt-4o"
        
        # Create config
        config = AgentConfig(
            provider=provider,
            model=args.model,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system_prompt=args.system_prompt,
            verbose=args.verbose,
            enable_tools=args.enable_tools or args.enable_code_execution or args.enable_computer_use,
            enable_computer_use=args.enable_computer_use,
            enable_code_execution=args.enable_code_execution,
            computer_type=args.computer_type,
            start_url=args.start_url,
            show_images=args.show_images,
            debug=args.debug
        )
        
        return config
    
    async def run_single_input(self, input_text: str, output_file: Optional[str] = None):
        """Run agent with a single input."""
        if not self.agent:
            raise RuntimeError("Agent not initialized")
        
        print(f"[Agent] Processing: {input_text}")
        
        try:
            response = await self.agent.run_async(input_text)
            
            # Extract text content
            text_content = []
            for block in response.get("content", []):
                if block.get("type") == "text":
                    text_content.append(block.get("text", ""))
            
            output = "\n".join(text_content)
            
            # Display output
            print(f"\n[Agent] Response:\n{output}")
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(output)
                print(f"\n[Agent] Output saved to: {output_file}")
            
            return response
            
        except Exception as e:
            print(f"[Error] {str(e)}")
            return None
    
    async def run_interactive(self):
        """Run agent in interactive mode."""
        if not self.agent:
            raise RuntimeError("Agent not initialized")
        
        print("\n[Agent] Interactive mode started. Type 'exit' to quit, 'reset' to clear history.")
        print(f"[Agent] Provider: {self.agent.config.provider.value}")
        print(f"[Agent] Model: {self.agent.config.model}")
        print(f"[Agent] Tools enabled: {self.agent.tool_registry.list_tools()}")
        print()
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if user_input.lower() == 'exit':
                    print("[Agent] Goodbye!")
                    break
                elif user_input.lower() == 'reset':
                    self.agent.reset()
                    print("[Agent] History cleared.")
                    continue
                elif not user_input:
                    continue
                
                response = await self.agent.run_async(user_input)
                
                # Extract and display text content
                text_content = []
                for block in response.get("content", []):
                    if block.get("type") == "text":
                        text_content.append(block.get("text", ""))
                
                if text_content:
                    print(f"\n[Agent] {' '.join(text_content)}")
                print()
                
            except KeyboardInterrupt:
                print("\n[Agent] Interrupted. Type 'exit' to quit.")
            except EOFError:
                print("\n[Agent] End of input. Goodbye!")
                break
            except Exception as e:
                print(f"[Error] {str(e)}")
    
    async def run_swarm_interactive(self, args):
        """Run swarm in interactive mode."""
        print("\n[Swarm] Interactive mode started. Type 'exit' to quit.")
        
        if not args.swarm_config:
            print("[Error] No swarm configuration file provided. Use --swarm-config to specify the path.")
            return
        
        try:
            # Dynamically load the swarm configuration
            spec = importlib.util.spec_from_file_location("swarm_config", args.swarm_config)
            swarm_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(swarm_config)
            
            # Get the initial agent
            initial_agent = getattr(swarm_config, args.initial_agent)
            
        except (FileNotFoundError, AttributeError) as e:
            print(f"[Error] Could not load swarm configuration: {e}")
            return
            
        client = Swarm()
        
        messages = []
        while True:
            try:
                user_input = input("> ").strip()

                if user_input.lower() == 'exit':
                    print("[Swarm] Goodbye!")
                    break
                elif not user_input:
                    continue
                
                messages.append({"role": "user", "content": user_input})

                response = client.run(
                    agent=initial_agent,
                    messages=messages,
                )
                
                # The last message is the response from the swarm
                last_message = response.messages[-1]
                messages.append(last_message)
                
                print(f"\n[Swarm] {last_message['content']}")

            except KeyboardInterrupt:
                print("\n[Swarm] Interrupted. Type 'exit' to quit.")
            except EOFError:
                print("\n[Swarm] End of input. Goodbye!")
                break
            except Exception as e:
                print(f"[Error] {str(e)}")

    async def run(self, args):
        """Run the CLI interface."""
        try:
            if args.swarm_config:
                await self.run_swarm_interactive(args)
                return

            # Create configuration
            config = self.create_config(args)
            
            # Validate configuration
            if not config.api_key:
                provider_name = config.provider.value.upper()
                print(f"[Error] No API key provided. Set {provider_name}_API_KEY environment variable or use --api-key.")
                sys.exit(1)
            
            # Create agent
            self.agent = UnifiedAgent(config)
            
            # Initialize computer use if enabled
            if config.enable_computer_use:
                print(f"[Agent] Initializing computer use with type: {config.computer_type}")
                # This would integrate with the actual computer use implementations
                # For now, just print a message
                print("[Agent] Computer use initialized (placeholder)")
            
            # Run based on mode
            if args.input:
                # Single input mode
                await self.run_single_input(args.input, args.output_file)
            elif args.interactive:
                # Interactive mode
                await self.run_interactive()
            else:
                # Default to interactive mode
                await self.run_interactive()
                
        except Exception as e:
            print(f"[Error] {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()
