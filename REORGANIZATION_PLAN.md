# Claude Agent Framework - Reorganization Plan

## Current State Analysis

### Strengths
- ✅ Working Python CLI with comprehensive tool support
- ✅ Complete implementation of beta tools (computer use, code execution)
- ✅ Solid agent framework with MCP integration
- ✅ Good documentation and examples
- ✅ Proper async/await patterns

### Issues to Address
- 🔧 Mixed project structure (multiple unrelated components)
- 🔧 Inconsistent naming and organization
- 🔧 Legacy files that should be cleaned up
- 🔧 Missing Java CLI implementation
- 🔧 No unified SDK/library structure
- 🔧 Incomplete tool exports in __init__.py

## Proposed New Structure

```
claude-agent-framework/
├── README.md                           # Main project documentation
├── LICENSE                            # Project license
├── CHANGELOG.md                       # Version history
├── .gitignore                        # Git ignore rules
├── pyproject.toml                    # Python project configuration
├── pom.xml                           # Maven configuration for Java
│
├── core/                             # Core framework (language-agnostic)
│   ├── README.md                     # Core framework documentation
│   ├── schemas/                      # JSON schemas for tools and configs
│   │   ├── tool-schema.json
│   │   ├── agent-config.json
│   │   └── beta-headers.json
│   └── docs/                         # Framework documentation
│       ├── architecture.md
│       ├── tool-development.md
│       └── beta-tools.md
│
├── python/                           # Python implementation
│   ├── README.md                     # Python-specific documentation
│   ├── pyproject.toml               # Python package configuration
│   ├── requirements.txt             # Python dependencies
│   ├── requirements-dev.txt         # Development dependencies
│   │
│   ├── claude_agent/                # Main Python package
│   │   ├── __init__.py
│   │   ├── version.py
│   │   │
│   │   ├── core/                    # Core agent functionality
│   │   │   ├── __init__.py
│   │   │   ├── agent.py            # Main agent class
│   │   │   ├── config.py           # Configuration management
│   │   │   ├── history.py          # Message history handling
│   │   │   └── beta_headers.py     # Beta header management
│   │   │
│   │   ├── tools/                  # Tool implementations
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Base tool class
│   │   │   ├── builtin/           # Built-in tools
│   │   │   │   ├── __init__.py
│   │   │   │   ├── think.py
│   │   │   │   ├── file_tools.py
│   │   │   │   ├── computer_use.py
│   │   │   │   └── code_execution.py
│   │   │   ├── beta/              # Beta tools
│   │   │   │   ├── __init__.py
│   │   │   │   ├── computer_use_v2.py
│   │   │   │   └── code_execution_v2.py
│   │   │   └── mcp/               # MCP integration
│   │   │       ├── __init__.py
│   │   │       ├── client.py
│   │   │       └── tools.py
│   │   │
│   │   ├── cli/                   # CLI implementation
│   │   │   ├── __init__.py
│   │   │   ├── main.py           # Main CLI entry point
│   │   │   ├── commands/         # CLI commands
│   │   │   │   ├── __init__.py
│   │   │   │   ├── interactive.py
│   │   │   │   ├── single.py
│   │   │   │   └── batch.py
│   │   │   └── utils/            # CLI utilities
│   │   │       ├── __init__.py
│   │   │       ├── formatting.py
│   │   │       └── validation.py
│   │   │
│   │   └── utils/                # General utilities
│   │       ├── __init__.py
│   │       ├── logging.py
│   │       ├── exceptions.py
│   │       └── helpers.py
│   │
│   ├── tests/                    # Python tests
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── unit/
│   │   ├── integration/
│   │   └── e2e/
│   │
│   └── examples/                 # Python examples
│       ├── basic_usage.py
│       ├── computer_use_demo.py
│       ├── code_execution_demo.py
│       └── mcp_integration.py
│
├── java/                         # Java implementation
│   ├── README.md                 # Java-specific documentation
│   ├── pom.xml                  # Maven configuration
│   ├── gradle.build             # Gradle configuration (alternative)
│   │
│   ├── src/main/java/com/anthropic/claude/agent/
│   │   ├── core/                # Core agent functionality
│   │   │   ├── Agent.java
│   │   │   ├── AgentConfig.java
│   │   │   ├── MessageHistory.java
│   │   │   └── BetaHeaderManager.java
│   │   │
│   │   ├── tools/               # Tool implementations
│   │   │   ├── Tool.java        # Base tool interface
│   │   │   ├── builtin/         # Built-in tools
│   │   │   │   ├── ThinkTool.java
│   │   │   │   ├── FileTools.java
│   │   │   │   ├── ComputerUseTool.java
│   │   │   │   └── CodeExecutionTool.java
│   │   │   ├── beta/            # Beta tools
│   │   │   └── mcp/             # MCP integration
│   │   │
│   │   ├── cli/                 # CLI implementation
│   │   │   ├── ClaudeAgentCLI.java
│   │   │   ├── commands/
│   │   │   └── utils/
│   │   │
│   │   └── utils/               # General utilities
│   │       ├── Logger.java
│   │       ├── Exceptions.java
│   │       └── Helpers.java
│   │
│   ├── src/test/java/           # Java tests
│   └── examples/                # Java examples
│       ├── BasicUsage.java
│       ├── ComputerUseDemo.java
│       └── CodeExecutionDemo.java
│
├── docker/                      # Docker configurations
│   ├── python/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   ├── java/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── computer-use/            # Computer use environment
│       ├── Dockerfile
│       └── docker-compose.yml
│
├── docs/                        # Comprehensive documentation
│   ├── README.md
│   ├── getting-started.md
│   ├── api-reference/
│   │   ├── python/
│   │   └── java/
│   ├── guides/
│   │   ├── tool-development.md
│   │   ├── beta-tools.md
│   │   ├── mcp-integration.md
│   │   └── deployment.md
│   └── examples/
│       ├── use-cases/
│       └── tutorials/
│
├── scripts/                     # Build and utility scripts
│   ├── build.sh                # Cross-platform build script
│   ├── test.sh                 # Cross-platform test script
│   ├── release.sh              # Release automation
│   └── setup/                  # Setup scripts
│       ├── python-setup.sh
│       └── java-setup.sh
│
└── tools/                      # Development tools
    ├── code-generation/        # Code generators
    ├── schema-validation/      # Schema validators
    └── testing/               # Testing utilities
```

## Migration Plan

### Phase 1: Core Restructuring (Week 1)
1. **Create new directory structure**
2. **Move and reorganize Python code**
   - Migrate `agents/` → `python/claude_agent/core/`
   - Migrate `cli.py` → `python/claude_agent/cli/main.py`
   - Update all imports and references
3. **Clean up legacy files**
   - Remove unused anthropic_*.py files
   - Archive financial-data-analyst (separate project)
   - Clean up computer-use-demo integration
4. **Update package configuration**
   - Create proper pyproject.toml
   - Update setup.py or replace with pyproject.toml
   - Fix dependencies and entry points

### Phase 2: Python Enhancement (Week 2)
1. **Complete tool integration**
   - Update __init__.py exports
   - Add proper tool discovery
   - Implement tool registry
2. **Enhance CLI**
   - Modularize CLI commands
   - Add better error handling
   - Improve user experience
3. **Add comprehensive testing**
   - Unit tests for all components
   - Integration tests for tool interactions
   - E2E tests for CLI workflows
4. **Documentation updates**
   - API documentation
   - Usage guides
   - Examples

### Phase 3: Java Implementation (Week 3-4)
1. **Core Java framework**
   - Implement Agent class
   - HTTP client for Anthropic API
   - Message history management
   - Beta header handling
2. **Tool system**
   - Base Tool interface
   - Built-in tool implementations
   - Beta tool support
3. **CLI implementation**
   - Command-line argument parsing
   - Interactive mode
   - Output formatting
4. **Testing and examples**

### Phase 4: Integration & Polish (Week 5)
1. **Cross-platform testing**
2. **Docker containers**
3. **CI/CD pipeline**
4. **Release preparation**
5. **Documentation finalization**

## Key Improvements

### 1. Unified Architecture
- **Language-agnostic core concepts**
- **Consistent API design across languages**
- **Shared schemas and documentation**

### 2. Better Tool Management
- **Tool registry system**
- **Automatic tool discovery**
- **Plugin architecture for custom tools**
- **Beta tool lifecycle management**

### 3. Enhanced CLI Experience
- **Consistent command structure**
- **Better error messages and help**
- **Configuration file support**
- **Shell completion**

### 4. Robust Testing
- **Comprehensive test coverage**
- **Mock services for testing**
- **Performance benchmarks**
- **Integration test suites**

### 5. Professional Packaging
- **Proper semantic versioning**
- **Release automation**
- **Package distribution**
- **Docker images**

## Implementation Details

### Python Package Structure
```python
# python/claude_agent/__init__.py
from .core import Agent, AgentConfig
from .tools import Tool, get_available_tools
from .cli import main as cli_main

__version__ = "1.0.0"
__all__ = ["Agent", "AgentConfig", "Tool", "get_available_tools", "cli_main"]
```

### Java Package Structure
```java
// java/src/main/java/com/anthropic/claude/agent/Agent.java
package com.anthropic.claude.agent;

public class Agent {
    private final AgentConfig config;
    private final List<Tool> tools;
    private final MessageHistory history;
    
    // Implementation
}
```

### Tool Registry System
```python
# python/claude_agent/tools/__init__.py
from .registry import ToolRegistry
from .base import Tool

# Auto-discover and register tools
registry = ToolRegistry()
registry.discover_tools()

def get_available_tools():
    return registry.list_tools()
```

### Configuration Management
```python
# python/claude_agent/core/config.py
@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 1.0
    api_key: Optional[str] = None
    beta_features: List[str] = field(default_factory=list)
    
    @classmethod
    def from_file(cls, path: str) -> 'AgentConfig':
        # Load from YAML/JSON config file
        pass
```

## Benefits of Reorganization

### 1. **Professional Structure**
- Industry-standard project layout
- Clear separation of concerns
- Easy to navigate and understand

### 2. **Multi-language Support**
- Consistent API across Python and Java
- Shared documentation and examples
- Cross-platform compatibility

### 3. **Maintainability**
- Modular architecture
- Clear dependencies
- Easy to extend and modify

### 4. **Developer Experience**
- Better IDE support
- Comprehensive documentation
- Rich examples and tutorials

### 5. **Production Ready**
- Proper testing infrastructure
- CI/CD pipeline
- Release automation
- Docker support

## Next Steps

1. **Approve reorganization plan**
2. **Begin Phase 1 migration**
3. **Set up new repository structure**
4. **Migrate existing code**
5. **Update documentation**
6. **Begin Java implementation**

This reorganization will transform the current project into a professional, multi-language agent framework suitable for production use and community contribution.
