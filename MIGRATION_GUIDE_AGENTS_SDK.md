# Migration Guide: Swarm → OpenAI Agents SDK

This guide helps you migrate from Swarm to the new OpenAI Agents SDK, which is the production-ready evolution of Swarm.

## 🚀 Quick Start

### 1. Install the Agents SDK

```bash
# Option A: Using venv (traditional)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Option B: Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Agents SDK
pip install openai-agents

# For voice support
pip install 'openai-agents[voice]'
```

### 2. Set up your environment

```bash
# Copy the setup script and run it
chmod +x setup_agents_sdk.sh
./setup_agents_sdk.sh

# Or manually set your API key
export OPENAI_API_KEY="your_api_key_here"
```

## 🔄 Migration Patterns

### Pattern 1: Basic Agent Migration

**OLD (Swarm):**
```python
from swarm import Swarm, Agent

def calculate(expression: str) -> str:
    return f"Result: {eval(expression)}"

math_agent = Agent(
    name="Math Agent",
    instructions="I solve math problems",
    functions=[calculate]
)

client = Swarm()
response = client.run(agent=math_agent, messages=messages)
```

**NEW (Agents SDK):**
```python
from agents import Agent, Runner, function_tool

@function_tool
def calculate(expression: str) -> str:
    return f"Result: {eval(expression)}"

math_agent = Agent(
    name="Math Agent", 
    instructions="I solve math problems",
    tools=[calculate]  # 'tools' instead of 'functions'
)

result = await Runner.run(math_agent, "what is 2+2?")
print(result.final_output)
```

### Pattern 2: Agent Handoffs

**OLD (Swarm):**
```python
def transfer_to_math():
    return math_agent

triage_agent = Agent(
    name="Triage Agent",
    instructions="Route to math agent",
    functions=[transfer_to_math]
)

# Manual agent tracking required
current_agent = triage_agent
response = client.run(agent=current_agent, messages=messages)
current_agent = response.agent  # Manual tracking
```

**NEW (Agents SDK):**
```python
triage_agent = Agent(
    name="Triage Agent",
    instructions="Route to math agent",
    handoffs=[math_agent]  # Automatic handoffs
)

# No manual tracking needed!
result = await Runner.run(triage_agent, "solve 2+2")
# Handoff happens automatically
```

### Pattern 3: Conversation Memory

**OLD (Swarm):**
```python
# Manual message history management
messages = []
messages.append({"role": "user", "content": "What is 2+2?"})
response = client.run(agent=math_agent, messages=messages)
messages.extend(response.messages)

messages.append({"role": "user", "content": "What about 3+3?"})
response = client.run(agent=math_agent, messages=messages)
```

**NEW (Agents SDK):**
```python
from agents.memory import SQLiteSession

session = SQLiteSession("user_123")

# Automatic conversation history
result1 = await Runner.run(math_agent, "What is 2+2?", session=session)
result2 = await Runner.run(math_agent, "What about 3+3?", session=session)
# Agent automatically remembers previous context
```

## 🆕 New Features

### 1. Handoffs
Automatic agent transfers without manual tracking:

```python
from agents import Agent, Runner

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent", 
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on language",
    handoffs=[spanish_agent, english_agent],
)

# Automatic language detection and handoff
result = await Runner.run(triage_agent, "Hola, ¿cómo estás?")
```

### 2. Sessions
Built-in conversation memory:

```python
from agents.memory import SQLiteSession

session = SQLiteSession("user_123")

# First turn
result = await Runner.run(agent, "What city is the Golden Gate Bridge in?", session=session)

# Second turn - agent remembers previous context
result = await Runner.run(agent, "What state is it in?", session=session)
```

### 3. Tracing
Built-in run tracking and debugging:

```python
# Tracing is automatic - no setup required
result = await Runner.run(agent, "Hello")
# All runs are automatically traced for debugging
```

### 4. Guardrails
Configurable safety checks:

```python
from agents import Agent, Runner, Guardrail

# Add input validation
guardrail = Guardrail(
    input_validators=[...],
    output_validators=[...]
)

agent = Agent(
    name="Safe Agent",
    instructions="...",
    guardrails=[guardrail]
)
```

## 🔧 Advanced Migration

### Complex Agent Networks

**OLD (Swarm):**
```python
# Complex manual agent management
agents = [math_agent, weather_agent, triage_agent]
current_agent = triage_agent
messages = []

for user_input in conversation:
    messages.append({"role": "user", "content": user_input})
    response = client.run(agent=current_agent, messages=messages)
    current_agent = response.agent
    messages.extend(response.messages)
```

**NEW (Agents SDK):**
```python
# Simple automatic handoffs
result = await Runner.run(triage_agent, user_input)
# All handoffs and agent management handled automatically
```

### Custom Session Implementations

```python
from agents.memory import Session
from typing import List

class MyCustomSession:
    """Custom session implementation following the Session protocol."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
    
    async def get_items(self, limit: int | None = None) -> List[dict]:
        # Retrieve conversation history
        pass
    
    async def add_items(self, items: List[dict]) -> None:
        # Store new items
        pass
    
    async def pop_item(self) -> dict | None:
        # Remove and return most recent item
        pass
    
    async def clear_session(self) -> None:
        # Clear all items
        pass

# Use your custom session
result = await Runner.run(agent, "Hello", session=MyCustomSession("my_session"))
```

## 🧪 Testing Your Migration

### 1. Run the migration demo

```bash
python3 migration_to_agents_sdk.py
```

### 2. Test your existing functionality

```python
# Test your migrated agents
async def test_migration():
    result = await Runner.run(math_agent, "what is 2+@")
    assert "Math Agent here!" in result.final_output
    assert "4" in result.final_output
    print("✅ Migration test passed!")

asyncio.run(test_migration())
```

### 3. Compare performance

```python
import time

# Test Swarm performance
start = time.time()
# Your old Swarm code
swarm_time = time.time() - start

# Test Agents SDK performance  
start = time.time()
result = await Runner.run(agent, input)
sdk_time = time.time() - start

print(f"Swarm: {swarm_time:.2f}s")
print(f"Agents SDK: {sdk_time:.2f}s")
```

## 🚨 Common Migration Issues

### Issue 1: Functions → Tools
**Problem:** `functions` parameter doesn't exist
**Solution:** Use `tools` parameter and `@function_tool` decorator

```python
# OLD
@function_tool  # This decorator doesn't exist in Swarm
def my_function():
    pass

agent = Agent(functions=[my_function])  # 'functions' parameter

# NEW  
@function_tool  # This decorator exists in Agents SDK
def my_function():
    pass

agent = Agent(tools=[my_function])  # 'tools' parameter
```

### Issue 2: Manual Agent Tracking
**Problem:** Need to track current agent manually
**Solution:** Use handoffs for automatic transfers

```python
# OLD - Manual tracking
current_agent = triage_agent
response = client.run(agent=current_agent, messages=messages)
current_agent = response.agent

# NEW - Automatic handoffs
result = await Runner.run(triage_agent, input)
# Handoffs happen automatically
```

### Issue 3: Message History Management
**Problem:** Manual message list management
**Solution:** Use sessions for automatic memory

```python
# OLD - Manual messages
messages = []
messages.append({"role": "user", "content": input})
response = client.run(agent=agent, messages=messages)
messages.extend(response.messages)

# NEW - Automatic sessions
session = SQLiteSession("user_123")
result = await Runner.run(agent, input, session=session)
```

## 📚 Additional Resources

- [OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-js)
- [Examples Directory](https://github.com/openai/openai-agents-js/tree/main/examples)
- [Tracing Documentation](https://github.com/openai/openai-agents-js/blob/main/docs/tracing.md)
- [Sessions Documentation](https://github.com/openai/openai-agents-js/blob/main/docs/sessions.md)

## 🎯 Migration Checklist

- [ ] Install OpenAI Agents SDK
- [ ] Set up environment variables
- [ ] Convert `functions` to `tools` with `@function_tool`
- [ ] Replace manual agent tracking with `handoffs`
- [ ] Replace manual message management with `sessions`
- [ ] Test all agent interactions
- [ ] Update error handling for new API
- [ ] Test performance improvements
- [ ] Deploy to production

## 🎉 Benefits of Migration

✅ **Automatic handoffs** - No more manual agent tracking  
✅ **Built-in sessions** - Automatic conversation memory  
✅ **Provider-agnostic** - Support for 100+ LLMs  
✅ **Built-in tracing** - Easy debugging and optimization  
✅ **Production-ready** - Guardrails and safety features  
✅ **Simpler API** - Less boilerplate code  
✅ **Active maintenance** - Supported by OpenAI team  

The OpenAI Agents SDK represents a significant evolution from Swarm, providing a more robust, feature-rich, and production-ready framework for building multi-agent workflows.