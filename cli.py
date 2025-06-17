#!/usr/bin/env python3
"""
Claude Agent-to-Agent CLI: 
An Interdisciplinary Tool for Cognitive-Computational Interaction

This CLI represents a sophisticated interface for agent-based computational exploration,
bridging neurocognitive insights with advanced computational methodologies.

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
from typing import List, Optional, Dict, Any, FrozenSet, Tuple
from dataclasses import dataclass, field
from collections.abc import Mapping
from frozendict import frozendict
from datetime import datetime

# Extend system path to ensure local module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import Agent, ModelConfig
from agents.tools.think import ThinkTool
from agents.tools.file_tools import FileReadTool, FileWriteTool
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
    tools: FrozenSet[Any]
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
        tools: Optional[List[Any]] = None,
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
        default_tools = frozenset([
            ThinkTool(),
            FileReadTool(),
            FileWriteTool()
        ])
        
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
            'claude-3-opus-20240229'
        ],
        help='Select the Claude model for cognitive interactions'
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
        verbose=args.verbose,
        model=args.model
    )
    
    asyncio.run(cognitive_cli.interactive_session())

if __name__ == "__main__":
    main()