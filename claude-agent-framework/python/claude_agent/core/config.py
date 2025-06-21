"""Configuration management for Claude Agent Framework."""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration settings for Claude model parameters."""
    
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 1.0
    context_window_tokens: int = 180000


@dataclass
class AgentConfig:
    """Main configuration for Claude Agent."""
    
    # API Configuration
    api_key: Optional[str] = None
    
    # Model Configuration
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # Agent Configuration
    name: str = "claude-agent"
    system_prompt: str = "You are Claude, an AI assistant. Be concise and helpful."
    verbose: bool = False
    
    # Tool Configuration
    enabled_tools: List[str] = field(default_factory=lambda: ["all"])
    tool_config: Dict[str, Any] = field(default_factory=dict)
    
    # MCP Configuration
    mcp_servers: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.api_key is None:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AgentConfig':
        """Load configuration from a YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract model config
        model_data = data.get('model', {})
        model_config = ModelConfig(**model_data)
        
        # Extract agent config
        agent_data = data.get('agent', {})
        agent_data['model_config'] = model_config
        
        # Extract tool config
        tool_data = data.get('tools', {})
        agent_data['enabled_tools'] = tool_data.get('enabled', ["all"])
        agent_data['tool_config'] = {k: v for k, v in tool_data.items() if k != 'enabled'}
        
        # Extract MCP config
        mcp_data = data.get('mcp', {})
        agent_data['mcp_servers'] = mcp_data.get('servers', [])
        
        return cls(**agent_data)
    
    def to_file(self, config_path: str):
        """Save configuration to a YAML file."""
        data = {
            'agent': {
                'name': self.name,
                'system_prompt': self.system_prompt,
                'verbose': self.verbose,
            },
            'model': {
                'model': self.model_config.model,
                'max_tokens': self.model_config.max_tokens,
                'temperature': self.model_config.temperature,
                'context_window_tokens': self.model_config.context_window_tokens,
            },
            'tools': {
                'enabled': self.enabled_tools,
                **self.tool_config
            },
            'mcp': {
                'servers': self.mcp_servers
            }
        }
        
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)


def load_config(config_path: Optional[str] = None) -> AgentConfig:
    """Load configuration from file or environment."""
    if config_path:
        return AgentConfig.from_file(config_path)
    
    # Look for default config files
    default_paths = [
        "claude-agent.yaml",
        "claude-agent.yml", 
        "~/.claude-agent.yaml",
        "~/.claude-agent.yml"
    ]
    
    for path in default_paths:
        expanded_path = Path(path).expanduser()
        if expanded_path.exists():
            return AgentConfig.from_file(str(expanded_path))
    
    # Return default config
    return AgentConfig()
