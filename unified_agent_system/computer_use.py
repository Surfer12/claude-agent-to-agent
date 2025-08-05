"""
Computer use interface for the unified agent system.
"""

import asyncio
from typing import Any, Dict, Optional

from .core import UnifiedAgent, AgentConfig, ProviderType


class ComputerUseInterface:
    """Interface for computer use capabilities."""
    
    def __init__(self, agent: UnifiedAgent):
        """Initialize the computer use interface."""
        self.agent = agent
        self.computer = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the computer environment."""
        if self.initialized:
            return
        
        computer_type = self.agent.config.computer_type
        
        try:
            # Initialize based on computer type
            if computer_type == "local-playwright":
                await self._initialize_local_playwright()
            elif computer_type == "browserbase":
                await self._initialize_browserbase()
            else:
                # Default to local playwright
                await self._initialize_local_playwright()
            
            self.initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize computer environment: {str(e)}")
    
    async def _initialize_local_playwright(self):
        """Initialize local Playwright environment."""
        # This would integrate with the actual Playwright implementation
        # For now, create a placeholder
        self.computer = {
            "type": "local-playwright",
            "initialized": True,
            "browser": None,
            "page": None
        }
        
        if self.agent.config.verbose:
            print("[Computer] Local Playwright environment initialized")
    
    async def _initialize_browserbase(self):
        """Initialize Browserbase environment."""
        # This would integrate with the actual Browserbase implementation
        # For now, create a placeholder
        self.computer = {
            "type": "browserbase",
            "initialized": True,
            "session": None
        }
        
        if self.agent.config.verbose:
            print("[Computer] Browserbase environment initialized")
    
    async def navigate_to(self, url: str):
        """Navigate to a URL."""
        if not self.initialized:
            await self.initialize()
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # This would integrate with the actual browser implementation
        if self.agent.config.verbose:
            print(f"[Computer] Navigating to: {url}")
        
        # Placeholder implementation
        self.computer["current_url"] = url
    
    async def take_screenshot(self) -> str:
        """Take a screenshot of the current page."""
        if not self.initialized:
            await self.initialize()
        
        # This would integrate with the actual screenshot implementation
        if self.agent.config.verbose:
            print("[Computer] Taking screenshot")
        
        # Placeholder implementation
        return "screenshot_placeholder.png"
    
    async def click(self, selector: str):
        """Click on an element."""
        if not self.initialized:
            await self.initialize()
        
        if self.agent.config.verbose:
            print(f"[Computer] Clicking: {selector}")
        
        # Placeholder implementation
        pass
    
    async def type_text(self, selector: str, text: str):
        """Type text into an element."""
        if not self.initialized:
            await self.initialize()
        
        if self.agent.config.verbose:
            print(f"[Computer] Typing '{text}' into: {selector}")
        
        # Placeholder implementation
        pass
    
    async def execute_action(self, action: str, params: Dict[str, Any]) -> str:
        """Execute a computer action."""
        if not self.initialized:
            await self.initialize()
        
        try:
            if action == "navigate":
                url = params.get("url", "")
                await self.navigate_to(url)
                return f"Navigated to {url}"
            
            elif action == "click":
                selector = params.get("selector", "")
                await self.click(selector)
                return f"Clicked on {selector}"
            
            elif action == "type":
                selector = params.get("selector", "")
                text = params.get("text", "")
                await self.type_text(selector, text)
                return f"Typed '{text}' into {selector}"
            
            elif action == "screenshot":
                screenshot_path = await self.take_screenshot()
                return f"Screenshot saved to {screenshot_path}"
            
            elif action == "scroll":
                direction = params.get("direction", "down")
                if self.agent.config.verbose:
                    print(f"[Computer] Scrolling {direction}")
                return f"Scrolled {direction}"
            
            elif action == "wait":
                seconds = params.get("seconds", 1)
                await asyncio.sleep(seconds)
                return f"Waited {seconds} seconds"
            
            else:
                return f"Unknown action: {action}"
                
        except Exception as e:
            return f"Error executing action {action}: {str(e)}"
    
    async def run_agent_with_computer(self, user_input: str) -> Dict[str, Any]:
        """Run the agent with computer use capabilities."""
        # Ensure computer is initialized
        if not self.initialized:
            await self.initialize()
        
        # Navigate to start URL if this is the first interaction
        if not hasattr(self, '_first_interaction_done'):
            if self.agent.config.start_url:
                await self.navigate_to(self.agent.config.start_url)
            self._first_interaction_done = True
        
        # Run the agent
        response = await self.agent.run_async(user_input)
        
        return response
    
    async def cleanup(self):
        """Clean up computer resources."""
        if self.computer:
            # This would integrate with the actual cleanup implementation
            if self.agent.config.verbose:
                print("[Computer] Cleaning up resources")
            
            # Placeholder cleanup
            self.computer = None
            self.initialized = False


class ComputerUseAgent:
    """Agent wrapper with computer use capabilities."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the computer use agent."""
        self.agent = UnifiedAgent(config)
        self.computer_interface = ComputerUseInterface(self.agent)
    
    async def run(self, user_input: str) -> Dict[str, Any]:
        """Run the agent with computer use."""
        return await self.computer_interface.run_agent_with_computer(user_input)
    
    async def run_interactive(self):
        """Run the agent in interactive mode with computer use."""
        print("\n[Computer Agent] Interactive mode started. Type 'exit' to quit, 'reset' to clear history.")
        print(f"[Computer Agent] Provider: {self.agent.config.provider.value}")
        print(f"[Computer Agent] Model: {self.agent.config.model}")
        print(f"[Computer Agent] Computer type: {self.agent.config.computer_type}")
        print(f"[Computer Agent] Start URL: {self.agent.config.start_url}")
        print()
        
        try:
            while True:
                user_input = input("> ").strip()
                
                if user_input.lower() == 'exit':
                    print("[Computer Agent] Goodbye!")
                    break
                elif user_input.lower() == 'reset':
                    self.agent.reset()
                    print("[Computer Agent] History cleared.")
                    continue
                elif not user_input:
                    continue
                
                response = await self.run(user_input)
                
                # Extract and display text content
                text_content = []
                for block in response.get("content", []):
                    if block.get("type") == "text":
                        text_content.append(block.get("text", ""))
                
                if text_content:
                    print(f"\n[Computer Agent] {' '.join(text_content)}")
                print()
                
        except KeyboardInterrupt:
            print("\n[Computer Agent] Interrupted. Goodbye!")
        except EOFError:
            print("\n[Computer Agent] End of input. Goodbye!")
        finally:
            await self.computer_interface.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        await self.computer_interface.cleanup() 