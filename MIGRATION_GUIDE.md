# Migration Guide: Old Structure to New Package Organization

This guide helps you migrate from the old scattered file structure to the new organized package structure that follows Java naming conventions.

## 📋 Overview of Changes

### Old Structure
```
claude-agent-to-agent/
├── anthropic_client.py
├── anthropic_bash_tool.py
├── anthropic_code_execution.py
├── anthropic_streaming_tools.py
├── anthropic_text_editor.py
├── anthropic_web_search.py
├── anthropic_weather_tool.py
├── cli.py
└── src/main/java/com/anthropic/api/
    ├── AnthropicClient.java
    └── MessageCreateParams.java
```

### New Structure
```
src/main/
├── python/
│   └── com/
│       └── anthropic/
│           └── api/
│               ├── __init__.py
│               ├── client.py
│               ├── tools.py
│               ├── streaming.py
│               └── cli.py
└── java/
    └── com/
        └── anthropic/
            └── api/
                ├── AnthropicClientEnhanced.java
                ├── tools/
                │   └── AnthropicTools.java
                └── cli/
                    └── CognitiveAgentCLI.java
```

## 🔄 Migration Steps

### 1. Python Migration

#### Old Import Style
```python
# Old way - direct file imports
import anthropic_client
import anthropic_bash_tool
import anthropic_web_search
```

#### New Import Style
```python
# New way - package imports
from com.anthropic.api import AnthropicClient
from com.anthropic.api.tools import create_bash_tool, create_web_search_tool
from com.anthropic.api.streaming import StreamingResponse
from com.anthropic.api.cli import CognitiveAgentCLI
```

#### Old Client Usage
```python
# Old way
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[{
        "type": "bash_20250124",
        "name": "bash"
    }],
    messages=[{"role": "user", "content": "List files"}]
)
```

#### New Client Usage
```python
# New way
from com.anthropic.api import AnthropicClient

client = AnthropicClient()
response = client.create_message(
    messages=[{"role": "user", "content": "List files"}],
    tools=["bash"],
    model="claude-sonnet-4-20250514"
)
```

### 2. Java Migration

#### Old Import Style
```java
// Old way
import com.anthropic.api.AnthropicClient;
import com.anthropic.api.MessageCreateParams;
```

#### New Import Style
```java
// New way
import com.anthropic.api.AnthropicClientEnhanced;
import com.anthropic.api.tools.AnthropicTools;
import com.anthropic.api.cli.CognitiveAgentCLI;
```

#### Old Client Usage
```java
// Old way
AnthropicClient client = new AnthropicClient.Builder()
    .apiKey(apiKey)
    .build();

Message message = client.createMessage(messages);
```

#### New Client Usage
```java
// New way
AnthropicClientEnhanced client = AnthropicClientEnhanced.createBasicClient(apiKey);

AnthropicClientEnhanced.Message response = client.createMessage(
    messages,
    Arrays.asList("bash"),
    null
);
```

## 🛠️ Tool Migration

### Python Tools

#### Old Tool Usage
```python
# Old way - individual tool files
import anthropic_bash_tool
import anthropic_web_search

# Tools were used directly in API calls
tools = [
    {
        "type": "bash_20250124",
        "name": "bash"
    },
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5
    }
]
```

#### New Tool Usage
```python
# New way - unified tools module
from com.anthropic.api.tools import (
    create_bash_tool,
    create_web_search_tool,
    create_weather_tool
)

# Create tools
bash_tool = create_bash_tool()
web_search_tool = create_web_search_tool(max_uses=5)

# Use with client
client = AnthropicClient()
response = client.create_message(
    messages=[{"role": "user", "content": "Search and execute"}],
    tools=["bash", "web_search"]
)
```

### Java Tools

#### Old Tool Usage
```java
// Old way - no dedicated tool classes
Map<String, Object> bashTool = new HashMap<>();
bashTool.put("type", "bash_20250124");
bashTool.put("name", "bash");
```

#### New Tool Usage
```java
// New way - dedicated tool classes
import com.anthropic.api.tools.AnthropicTools;

AnthropicTools.BashTool bashTool = AnthropicTools.createBashTool();
AnthropicTools.WebSearchTool webSearchTool = AnthropicTools.createWebSearchTool(5);

// Convert to maps for API usage
List<Map<String, Object>> tools = AnthropicTools.toolsToMaps(
    Arrays.asList(bashTool, webSearchTool)
);
```

## 🖥️ CLI Migration

### Python CLI

#### Old CLI Usage
```bash
# Old way - direct script execution
python cli.py --name MyAgent --verbose
```

#### New CLI Usage
```bash
# New way - module execution
python -m com.anthropic.api.cli --name MyAgent --verbose

# Or install and use as command
pip install -e .
anthropic-cli --name MyAgent --verbose
```

### Java CLI

#### Old CLI Usage
```bash
# Old way - no dedicated CLI
java -cp target/classes com.anthropic.api.AnthropicClient
```

#### New CLI Usage
```bash
# New way - dedicated CLI class
java -cp target/classes com.anthropic.api.cli.CognitiveAgentCLI --name MyAgent --verbose
```

## 📦 Package Installation

### Python Package

#### Old Installation
```bash
# Old way - no package structure
pip install anthropic
# Copy individual files as needed
```

#### New Installation
```bash
# New way - proper package installation
cd src/main/python
pip install -e .

# Or install from PyPI (when published)
pip install anthropic-api-client
```

### Java Package

#### Old Installation
```xml
<!-- Old way - no dedicated package -->
<dependency>
    <groupId>com.anthropic</groupId>
    <artifactId>anthropic-java</artifactId>
    <version>1.0.0</version>
</dependency>
```

#### New Installation
```xml
<!-- New way - dedicated API client package -->
<dependency>
    <groupId>com.anthropic</groupId>
    <artifactId>anthropic-api-client</artifactId>
    <version>1.0.0</version>
</dependency>
```

## 🔧 Configuration Changes

### Environment Variables

No changes needed - the same environment variables are used:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### API Key Handling

#### Old Way
```python
# Python
import anthropic
client = anthropic.Anthropic(api_key="your-key")
```

```java
// Java
AnthropicClient client = new AnthropicClient.Builder()
    .apiKey("your-key")
    .build();
```

#### New Way
```python
# Python
from com.anthropic.api import AnthropicClient
client = AnthropicClient(api_key="your-key")
```

```java
// Java
AnthropicClientEnhanced client = AnthropicClientEnhanced.createBasicClient("your-key");
```

## 🧪 Testing Migration

### Python Tests

#### Old Test Structure
```python
# Old way - test individual files
import anthropic_client
# Test individual functions
```

#### New Test Structure
```python
# New way - test package modules
from com.anthropic.api import AnthropicClient
from com.anthropic.api.tools import create_bash_tool

def test_client_creation():
    client = AnthropicClient()
    assert client is not None

def test_tool_creation():
    tool = create_bash_tool()
    assert tool.definition.name == "bash"
```

### Java Tests

#### Old Test Structure
```java
// Old way - test individual classes
@Test
public void testAnthropicClient() {
    AnthropicClient client = new AnthropicClient.Builder().build();
    assertNotNull(client);
}
```

#### New Test Structure
```java
// New way - test enhanced client
@Test
public void testAnthropicClientEnhanced() {
    AnthropicClientEnhanced client = AnthropicClientEnhanced.createBasicClient(apiKey);
    assertNotNull(client);
    
    List<String> tools = client.getAvailableTools();
    assertTrue(tools.contains("bash"));
}
```

## 🚀 Benefits of New Structure

1. **Consistent Naming**: Both Python and Java follow the same package naming convention
2. **Better Organization**: Related functionality is grouped together
3. **Easier Maintenance**: Clear separation of concerns
4. **Improved Testing**: Better test organization and coverage
5. **Enhanced Security**: Immutable collections and thread safety
6. **Better Documentation**: Comprehensive examples and guides
7. **Simplified Imports**: Cleaner import statements
8. **Tool Integration**: Unified tool management system

## ⚠️ Breaking Changes

1. **Import Paths**: All import statements need to be updated
2. **Client Class Names**: `AnthropicClient` → `AnthropicClientEnhanced` (Java)
3. **Tool Creation**: Tools now use factory methods instead of direct instantiation
4. **CLI Execution**: CLI now runs as a module instead of a script
5. **Package Installation**: Requires proper package installation instead of file copying

## 🔍 Migration Checklist

- [ ] Update all import statements
- [ ] Replace direct client usage with new client classes
- [ ] Update tool creation and usage
- [ ] Modify CLI execution commands
- [ ] Update test files
- [ ] Install packages properly
- [ ] Update documentation references
- [ ] Test all functionality
- [ ] Update CI/CD pipelines if applicable

## 🆘 Getting Help

If you encounter issues during migration:

1. Check the examples in `src/main/python/examples/` and `src/main/java/examples/`
2. Review the comprehensive README in `src/main/README.md`
3. Look at the API reference documentation
4. Create an issue on GitHub with details about your problem

## 📈 Performance Improvements

The new structure includes several performance improvements:

- **Immutable Collections**: Prevents accidental modifications and improves thread safety
- **Lazy Loading**: Tools are created only when needed
- **Better Memory Management**: Proper resource cleanup
- **Optimized Imports**: Reduced import overhead
- **Caching**: Tool configurations are cached for better performance 