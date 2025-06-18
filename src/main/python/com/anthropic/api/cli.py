"""
Command Line Interface for Anthropic API

This module provides a sophisticated CLI for interacting with Anthropic's Claude API,
including support for various tools and cognitive performance tracking.
"""

import os
import sys
import argparse
import asyncio
import logging
from typing import List, Optional, Dict, Any, FrozenSet, Tuple
from dataclasses import dataclass, field
from collections.abc import Mapping
from frozendict import frozendict
from datetime import datetime

# Import local modules
from .client import AnthropicClient
from .tools import (
    BashTool,
    WebSearchTool,
    WeatherTool,
    TextEditorTool,
    CodeExecutionTool,
    ComputerTool
)


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
    tools: FrozenSet[str]
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
        tools: Optional[List[str]] = None,
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
        default_tools = frozenset(['bash', 'web_search', 'weather'])
        
        # Create immutable configuration
        self.config = AgentConfig(
            name=name,
            system_prompt=system_prompt or default_prompt,
            tools=frozenset(tools) if tools else default_tools,
            verbose=verbose,
            model=model
        )
        
        # Initialize metrics with immutable container
        self.metrics = InteractionMetrics()
        
        # Initialize the API client
        self.client = AnthropicClient()
    
    async def interactive_session(self) -> None:
        """
        Launch an interactive cognitive agent session with 
        advanced interaction tracking.
        """
        logger.info(f"Initiating Cognitive Agent Session: {self.config.name}")
        
        try:
            while True:
                try:
                    user_input = input("\n🧠 Cognitive Agent > ")
                    
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        break
                    
                    # Track interaction performance
                    start_time = asyncio.get_event_loop().time()
                    
                    # Create message with system prompt and user input
                    messages = [
                        {"role": "system", "content": self.config.system_prompt},
                        {"role": "user", "content": user_input}
                    ]
                    
                    # Get available tools for this session
                    available_tools = list(self.config.tools)
                    
                    response = self.client.create_message(
                        messages=messages,
                        model=self.config.model,
                        max_tokens=1024,
                        tools=available_tools if available_tools else None
                    )
                    
                    end_time = asyncio.get_event_loop().time()
                    response_time = end_time - start_time
                    
                    # Update metrics with new immutable instance
                    self.metrics = InteractionMetrics(
                        total_interactions=self.metrics.total_interactions + 1,
                        successful_interactions=self.metrics.successful_interactions + 1,
                        average_response_time=(
                            0.9 * self.metrics.average_response_time + 
                            0.1 * response_time
                        ) if self.metrics.total_interactions > 0 else response_time,
                        last_interaction_time=datetime.now()
                    )
                    
                    # Output response with cognitive performance context
                    print("\n🤖 Response:")
                    for content in response.content:
                        if hasattr(content, 'text'):
                            print(content.text)
                
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
            logger.info(f"Successful Interactions: {self.metrics.successful_interactions}")
            logger.info(f"Average Response Time: {self.metrics.average_response_time:.4f} seconds")
            if self.metrics.last_interaction_time:
                logger.info(f"Last Interaction: {self.metrics.last_interaction_time}")
    
    def run_single_query(self, query: str) -> str:
        """
        Run a single query without entering interactive mode.
        
        Args:
            query: The user query to process
            
        Returns:
            The response text
        """
        try:
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": query}
            ]
            
            available_tools = list(self.config.tools)
            
            response = self.client.create_message(
                messages=messages,
                model=self.config.model,
                max_tokens=1024,
                tools=available_tools if available_tools else None
            )
            
            # Extract text content
            response_text = ""
            for content in response.content:
                if hasattr(content, 'text'):
                    response_text += content.text
            
            return response_text
            
        except Exception as e:
            logger.error(f"Query Error: {e}")
            return f"Error processing query: {e}"


def main() -> None:
    """
    Entry point for the Cognitive Agent CLI, supporting 
    advanced configuration and interaction modes.
    """
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
    parser.add_argument(
        '--model', 
        default='claude-3-5-sonnet-20240620',
        choices=[
            'claude-3-haiku-20240307', 
            'claude-3-5-sonnet-20240620', 
            'claude-3-opus-20240229',
            'claude-sonnet-4-20250514',
            'claude-opus-4-20250514'
        ],
        help='Select the Claude model for cognitive interactions'
    )
    parser.add_argument(
        '--tools',
        nargs='+',
        choices=['bash', 'web_search', 'weather', 'text_editor', 'code_execution', 'computer'],
        help='Tools to enable for this session'
    )
    parser.add_argument(
        '--query',
        help='Run a single query and exit (non-interactive mode)'
    )
    
    args = parser.parse_args()
    
    # Validate Anthropic API key
    if 'ANTHROPIC_API_KEY' not in os.environ:
        print("⚠️ Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)
    
    # Initialize and launch cognitive agent session
    cognitive_cli = CognitiveAgentCLI(
        name=args.name,
        system_prompt=args.system_prompt,
        tools=args.tools,
        verbose=args.verbose,
        model=args.model
    )
    
    if args.query:
        # Run single query mode
        response = cognitive_cli.run_single_query(args.query)
        print(response)
    else:
        # Run interactive mode
        asyncio.run(cognitive_cli.interactive_session())


if __name__ == "__main__":
    main() 