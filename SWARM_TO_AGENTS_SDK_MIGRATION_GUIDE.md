# Migration Guide: From Swarm to OpenAI Agents SDK

## Overview

The OpenAI Agents SDK is the production-ready evolution of Swarm, featuring key improvements and active maintenance by the OpenAI team. This guide will help you migrate your existing Swarm applications to the new Agents SDK.

## Why Migrate?

- **Production-ready**: Built for production use cases with enterprise-grade features
- **Active maintenance**: Officially supported and maintained by OpenAI
- **Enhanced features**: Sessions, tracing, guardrails, and more
- **Provider-agnostic**: Supports 100+ LLMs beyond just OpenAI
- **Better performance**: Optimized agent loop and conversation management

## Installation

### Remove Swarm (if desired)
```bash
# Remove old Swarm dependency
pip uninstall swarm
```

### Install Agents SDK
```bash
# Basic installation
pip install openai-agents

# With voice support
pip install 'openai-agents[voice]'
```

## Key Concept Mappings

| Swarm Concept | Agents SDK Equivalent | Notes |
|---------------|----------------------|-------|
| `Agent` | `Agent` | Similar concept, enhanced features |
| `Swarm().run()` | `Runner.run()` / `Runner.run_sync()` | New runner pattern |
| Agent functions | `@function_tool` | New decorator approach |
| Agent transfers | `handoffs` | Built-in handoff system |
| Context variables | Sessions | Persistent conversation memory |
| Manual history | Automatic sessions | Built-in memory management |

## Migration Examples

### 1. Basic Agent Creation

**Swarm (Old):**
```python
from swarm import Swarm, Agent

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant"
)

client = Swarm()
response = client.run(agent=agent, messages=[{"role": "user", "content": "Hello"}])
```

**Agents SDK (New):**
```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant", 
    instructions="You are a helpful assistant"
)

result = Runner.run_sync(agent, "Hello")
print(result.final_output)
```

### 2. Function Tools

**Swarm (Old):**
```python
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

agent = Agent(
    name="Weather Agent",
    instructions="You help with weather",
    functions=[get_weather]
)
```

**Agents SDK (New):**
```python
from agents import function_tool

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

agent = Agent(
    name="Weather Agent",
    instructions="You help with weather",
    tools=[get_weather]
)
```

### 3. Agent Transfers/Handoffs

**Swarm (Old):**
```python
def transfer_to_math():
    return math_agent

math_agent = Agent(
    name="Math Agent",
    instructions="You are a math specialist"
)

triage_agent = Agent(
    name="Triage Agent", 
    instructions="Transfer to math agent for calculations",
    functions=[transfer_to_math]
)
```

**Agents SDK (New):**
```python
math_agent = Agent(
    name="Math Agent",
    instructions="You are a math specialist"
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="Handoff to math agent for calculations", 
    handoffs=[math_agent]
)
```

### 4. Conversation Memory

**Swarm (Old):**
```python
# Manual message history management
messages = []
messages.append({"role": "user", "content": "Hello"})

response = client.run(agent=agent, messages=messages)
messages.extend(response.messages)

# Next turn
messages.append({"role": "user", "content": "Remember me?"})
response = client.run(agent=agent, messages=messages)
```

**Agents SDK (New):**
```python
from agents import SQLiteSession

# Automatic session management
session = SQLiteSession("user_123")

result = await Runner.run(agent, "Hello", session=session)
# Session automatically stores history

result = await Runner.run(agent, "Remember me?", session=session)
# Agent automatically has access to conversation history
```

## Complete Migration Example

Here's a complete migration of a Swarm application:

### Original Swarm Code
```python
from swarm import Swarm, Agent
import math

def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"Result: {result}"
    except:
        return "Error in calculation"

def transfer_to_math():
    return math_agent

math_agent = Agent(
    name="Math Agent",
    instructions="You are a math specialist",
    functions=[calculate]
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="Transfer to math agent for calculations",
    functions=[transfer_to_math]
)

client = Swarm()
response = client.run(
    agent=triage_agent,
    messages=[{"role": "user", "content": "Calculate 2+2"}]
)
```

### Migrated Agents SDK Code
```python
from agents import Agent, Runner, function_tool
import math
import asyncio

@function_tool
def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"Result: {result}"
    except:
        return "Error in calculation"

math_agent = Agent(
    name="Math Agent", 
    instructions="You are a math specialist",
    tools=[calculate]
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="Handoff to math agent for calculations",
    handoffs=[math_agent]
)

async def main():
    result = await Runner.run(triage_agent, "Calculate 2+2")
    print(result.final_output)

asyncio.run(main())
```

## New Features Available in Agents SDK

### 1. Sessions for Memory
```python
from agents import SQLiteSession

session = SQLiteSession("user_id", "conversations.db")
result = await Runner.run(agent, "Hello", session=session)
```

### 2. Structured Output Types
```python
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    city: str
    temperature: float
    condition: str

agent = Agent(
    name="Weather Agent",
    instructions="Return weather data",
    output_type=WeatherResponse
)
```

### 3. Built-in Tracing
```python
# Automatic tracing to various platforms
# Supports Logfire, AgentOps, Braintrust, etc.
result = await Runner.run(agent, "Hello")
# Tracing data automatically captured
```

### 4. Guardrails
```python
agent = Agent(
    name="Safe Agent",
    instructions="You are helpful but safe",
    # Guardrails can be configured for safety
)
```

## Migration Checklist

- [ ] Install `openai-agents` package
- [ ] Replace `from swarm import` with `from agents import`
- [ ] Change `Agent(functions=[...])` to `Agent(tools=[...])`
- [ ] Add `@function_tool` decorator to all functions
- [ ] Replace transfer functions with `handoffs` parameter
- [ ] Update `Swarm().run()` to `Runner.run()` or `Runner.run_sync()`
- [ ] Replace manual message history with Sessions
- [ ] Update async/await patterns if needed
- [ ] Test all agent interactions
- [ ] Consider adding new features like structured outputs

## Common Migration Issues

### 1. Function Decorator Required
**Problem**: Functions not being called
**Solution**: Add `@function_tool` decorator

### 2. Async/Await
**Problem**: Blocking calls in async context
**Solution**: Use `await Runner.run()` or `Runner.run_sync()`

### 3. Message Format
**Problem**: Different message handling
**Solution**: Use simple strings instead of message arrays

### 4. Agent Transfer Logic
**Problem**: Transfer functions not working
**Solution**: Replace with `handoffs` parameter

## Testing Your Migration

Create a test script to verify your migration:

```python
import asyncio
from agents import Agent, Runner

async def test_migration():
    # Test basic agent
    agent = Agent(name="Test", instructions="You are helpful")
    result = await Runner.run(agent, "Hello")
    assert result.final_output is not None
    
    # Test with tools
    @function_tool
    def test_tool() -> str:
        return "Tool works!"
    
    tool_agent = Agent(name="Tool Agent", tools=[test_tool])
    result = await Runner.run(tool_agent, "Use the tool")
    assert "Tool works!" in result.final_output
    
    print("✅ Migration tests passed!")

if __name__ == "__main__":
    asyncio.run(test_migration())
```

## Performance Considerations

The Agents SDK offers several performance improvements:

1. **Optimized Agent Loop**: More efficient conversation processing
2. **Built-in Caching**: Session and conversation caching
3. **Async Support**: Better concurrency handling
4. **Memory Management**: Automatic cleanup and optimization

## Getting Help

- **Documentation**: [Agents SDK Docs](https://github.com/openai/openai-agents-js)
- **Examples**: Check the `examples/` directory in the SDK
- **Issues**: Report problems on the GitHub repository

## Conclusion

The migration from Swarm to the OpenAI Agents SDK brings significant improvements in production readiness, performance, and features. While the core concepts remain similar, the new SDK provides a more robust foundation for building multi-agent applications.

The key benefits of migration include:
- Production-ready architecture
- Built-in session management
- Enhanced debugging and tracing
- Better error handling
- Official OpenAI support

Start with simple agents and gradually migrate more complex functionality, testing thoroughly at each step.