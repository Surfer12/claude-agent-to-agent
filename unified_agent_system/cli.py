"""CLI for the Unified Agent System."""

import asyncio
import argparse
import logging
from .core.agent import Agent
from .core.types import AgentConfig, ProviderType

def main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(description="Unified Agent System")
    parser.add_argument("--provider", type=str, default="mock", 
                       choices=["claude", "openai", "mock"], help="The AI provider to use.")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022", 
                       help="The model to use.")
    parser.add_argument("--instructions", type=str, default="You are a helpful assistant.", 
                       help="The system prompt for the agent.")
    parser.add_argument("--user-input", type=str, required=True, 
                       help="The user input to the agent.")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    parser.add_argument("--log-sensitive", action="store_true", 
                       help="Log sensitive data (opt-in).")

    args = parser.parse_args()
    
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    if not args.log_sensitive:
        logger.warning("Sensitive data logging is disabled.")

    # Convert string provider to ProviderType enum
    if args.provider == "claude":
        provider_type = ProviderType.CLAUDE
    elif args.provider == "openai":
        provider_type = ProviderType.OPENAI
    else:
        provider_type = ProviderType.MOCK
    
    config = AgentConfig(
        provider=provider_type,
        model=args.model,
        system_prompt=args.instructions,
    )
    
    agent = Agent(config)
    
    response = asyncio.run(agent.process_message(args.user_input))
    
    logger.info(f"Agent response: {response}")
    print(f"Agent response: {response}")

if __name__ == "__main__":
    main()
