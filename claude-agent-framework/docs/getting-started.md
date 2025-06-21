# Getting Started

## Installation

### Python

```bash
pip install claude-agent-framework
```

### Java

Download the latest JAR from the [releases page](https://github.com/anthropic/claude-agent-framework/releases).

## Configuration

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Basic Usage

### Python

```python
from claude_agent import Agent, AgentConfig

config = AgentConfig(
    model="claude-sonnet-4-20250514",
    api_key="your-api-key"
)

agent = Agent(config=config)
response = await agent.chat("Hello, Claude!")
print(response.content)
```

### Java

```java
import com.anthropic.claude.agent.Agent;
import com.anthropic.claude.agent.AgentConfig;

AgentConfig config = AgentConfig.builder()
    .model("claude-sonnet-4-20250514")
    .apiKey("your-api-key")
    .build();

Agent agent = new Agent(config);
AgentResponse response = agent.chat("Hello, Claude!").get();
System.out.println(response.getContent());
```

## CLI Usage

### Interactive Mode

```bash
# Python
claude-agent --interactive

# Java
java -jar claude-agent-framework.jar --interactive
```

### Single Commands

```bash
# Python
claude-agent --prompt "What is the capital of France?"

# Java
java -jar claude-agent-framework.jar --prompt "What is the capital of France?"
```

## Next Steps

- [Tool Development Guide](guides/tool-development.md)
- [Beta Tools Documentation](guides/beta-tools.md)
- [MCP Integration](guides/mcp-integration.md)
