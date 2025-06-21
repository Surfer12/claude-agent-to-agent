#!/usr/bin/env python3
"""
Migration script to reorganize the Claude Agent project into the new structure.
This script will create the new directory structure and move files appropriately.
"""

import os
import shutil
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Command failed: {cmd}")
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {cmd} - {e}")
        return False

def create_directory_structure():
    """Create the new directory structure."""
    print("Creating new directory structure...")
    
    directories = [
        # Core framework
        "claude-agent-framework/core/schemas",
        "claude-agent-framework/core/docs",
        
        # Python implementation
        "claude-agent-framework/python/claude_agent/core",
        "claude-agent-framework/python/claude_agent/tools/builtin",
        "claude-agent-framework/python/claude_agent/tools/beta",
        "claude-agent-framework/python/claude_agent/tools/mcp",
        "claude-agent-framework/python/claude_agent/cli/commands",
        "claude-agent-framework/python/claude_agent/cli/utils",
        "claude-agent-framework/python/claude_agent/utils",
        "claude-agent-framework/python/tests/unit",
        "claude-agent-framework/python/tests/integration",
        "claude-agent-framework/python/tests/e2e",
        "claude-agent-framework/python/examples",
        
        # Java implementation
        "claude-agent-framework/java/src/main/java/com/anthropic/claude/agent/core",
        "claude-agent-framework/java/src/main/java/com/anthropic/claude/agent/tools/builtin",
        "claude-agent-framework/java/src/main/java/com/anthropic/claude/agent/tools/beta",
        "claude-agent-framework/java/src/main/java/com/anthropic/claude/agent/tools/mcp",
        "claude-agent-framework/java/src/main/java/com/anthropic/claude/agent/cli/commands",
        "claude-agent-framework/java/src/main/java/com/anthropic/claude/agent/cli/utils",
        "claude-agent-framework/java/src/main/java/com/anthropic/claude/agent/utils",
        "claude-agent-framework/java/src/test/java",
        "claude-agent-framework/java/examples",
        
        # Docker configurations
        "claude-agent-framework/docker/python",
        "claude-agent-framework/docker/java",
        "claude-agent-framework/docker/computer-use",
        
        # Documentation
        "claude-agent-framework/docs/api-reference/python",
        "claude-agent-framework/docs/api-reference/java",
        "claude-agent-framework/docs/guides",
        "claude-agent-framework/docs/examples/use-cases",
        "claude-agent-framework/docs/examples/tutorials",
        
        # Scripts and tools
        "claude-agent-framework/scripts/setup",
        "claude-agent-framework/tools/code-generation",
        "claude-agent-framework/tools/schema-validation",
        "claude-agent-framework/tools/testing",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

def migrate_python_files():
    """Migrate existing Python files to the new structure."""
    print("\nMigrating Python files...")
    
    # File migrations
    migrations = [
        # Core agent files
        ("agents/agent.py", "claude-agent-framework/python/claude_agent/core/agent.py"),
        ("agents/utils/history_util.py", "claude-agent-framework/python/claude_agent/core/history.py"),
        ("agents/utils/tool_util.py", "claude-agent-framework/python/claude_agent/utils/tool_util.py"),
        ("agents/utils/connections.py", "claude-agent-framework/python/claude_agent/tools/mcp/connections.py"),
        
        # Tool files
        ("agents/tools/base.py", "claude-agent-framework/python/claude_agent/tools/base.py"),
        ("agents/tools/think.py", "claude-agent-framework/python/claude_agent/tools/builtin/think.py"),
        ("agents/tools/file_tools.py", "claude-agent-framework/python/claude_agent/tools/builtin/file_tools.py"),
        ("agents/tools/computer_use.py", "claude-agent-framework/python/claude_agent/tools/beta/computer_use.py"),
        ("agents/tools/code_execution.py", "claude-agent-framework/python/claude_agent/tools/beta/code_execution.py"),
        ("agents/tools/mcp_tool.py", "claude-agent-framework/python/claude_agent/tools/mcp/mcp_tool.py"),
        ("agents/tools/calculator_mcp.py", "claude-agent-framework/python/claude_agent/tools/mcp/calculator_mcp.py"),
        
        # CLI files
        ("cli.py", "claude-agent-framework/python/claude_agent/cli/main.py"),
        
        # Examples
        ("examples/simple_cli_example.py", "claude-agent-framework/python/examples/basic_usage.py"),
        ("examples/computer_use_example.py", "claude-agent-framework/python/examples/computer_use_demo.py"),
        ("examples/code_execution_example.py", "claude-agent-framework/python/examples/code_execution_demo.py"),
        ("examples/mcp_tools_example.py", "claude-agent-framework/python/examples/mcp_integration.py"),
        
        # Configuration files
        ("requirements.txt", "claude-agent-framework/python/requirements.txt"),
        ("setup.py", "claude-agent-framework/python/setup.py"),
        ("pixi.toml", "claude-agent-framework/python/pixi.toml"),
        
        # Documentation
        ("README.md", "claude-agent-framework/README.md"),
        ("COMPUTER_USE_IMPLEMENTATION.md", "claude-agent-framework/docs/guides/computer-use.md"),
        ("CODE_EXECUTION_IMPLEMENTATION.md", "claude-agent-framework/docs/guides/code-execution.md"),
    ]
    
    for src, dst in migrations:
        if os.path.exists(src):
            try:
                # Create destination directory if it doesn't exist
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                print(f"Migrated: {src} → {dst}")
            except Exception as e:
                print(f"Error migrating {src}: {e}")
        else:
            print(f"Source file not found: {src}")

def create_new_package_files():
    """Create new package files for the reorganized structure."""
    print("\nCreating new package files...")
    
    # Python package __init__.py files
    init_files = {
        "claude-agent-framework/python/claude_agent/__init__.py": '''"""Claude Agent Framework - Python Implementation"""

from .core import Agent, AgentConfig
from .tools import ToolRegistry, get_available_tools
from .version import __version__

__all__ = [
    "Agent", 
    "AgentConfig", 
    "ToolRegistry", 
    "get_available_tools",
    "__version__"
]
''',
        
        "claude-agent-framework/python/claude_agent/version.py": '''"""Version information for Claude Agent Framework"""

__version__ = "1.0.0"
''',
        
        "claude-agent-framework/python/claude_agent/core/__init__.py": '''"""Core agent functionality"""

from .agent import Agent
from .config import AgentConfig

__all__ = ["Agent", "AgentConfig"]
''',
        
        "claude-agent-framework/python/claude_agent/tools/__init__.py": '''"""Tool system for Claude Agent Framework"""

from .base import Tool
from .registry import ToolRegistry

def get_available_tools():
    """Get list of all available tools"""
    registry = ToolRegistry()
    registry.discover_tools()
    return registry.list_tools()

__all__ = ["Tool", "ToolRegistry", "get_available_tools"]
''',
        
        "claude-agent-framework/python/claude_agent/tools/builtin/__init__.py": '''"""Built-in tools"""

from .think import ThinkTool
from .file_tools import FileReadTool, FileWriteTool

__all__ = ["ThinkTool", "FileReadTool", "FileWriteTool"]
''',
        
        "claude-agent-framework/python/claude_agent/tools/beta/__init__.py": '''"""Beta tools"""

from .computer_use import ComputerUseTool
from .code_execution import CodeExecutionTool, CodeExecutionWithFilesTool

__all__ = ["ComputerUseTool", "CodeExecutionTool", "CodeExecutionWithFilesTool"]
''',
        
        "claude-agent-framework/python/claude_agent/cli/__init__.py": '''"""CLI interface"""

from .main import main

__all__ = ["main"]
''',
    }
    
    for file_path, content in init_files.items():
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Created: {file_path}")
        except Exception as e:
            print(f"Error creating {file_path}: {e}")

def create_configuration_files():
    """Create new configuration files."""
    print("\nCreating configuration files...")
    
    # Python pyproject.toml
    pyproject_content = '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "claude-agent-framework"
version = "1.0.0"
description = "A comprehensive framework for building Claude-powered agents"
authors = [{name = "Anthropic", email = "support@anthropic.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "anthropic>=0.54.0",
    "aiohttp>=3.8.0",
    "aiostream>=0.4.5",
    "jsonschema>=4.22.0",
    "click>=8.0.0",
    "pyyaml>=6.0",
    "mcp>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

[project.scripts]
claude-agent = "claude_agent.cli.main:main"

[project.urls]
Homepage = "https://github.com/anthropic/claude-agent-framework"
Documentation = "https://claude-agent-framework.readthedocs.io"
Repository = "https://github.com/anthropic/claude-agent-framework"
Issues = "https://github.com/anthropic/claude-agent-framework/issues"

[tool.setuptools.packages.find]
where = ["."]
include = ["claude_agent*"]

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--cov=claude_agent --cov-report=html --cov-report=term-missing"
'''
    
    # Java pom.xml
    pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.anthropic</groupId>
    <artifactId>claude-agent-framework</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>
    
    <name>Claude Agent Framework</name>
    <description>A comprehensive framework for building Claude-powered agents</description>
    <url>https://github.com/anthropic/claude-agent-framework</url>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <junit.version>5.9.2</junit.version>
        <mockito.version>5.1.1</mockito.version>
        <jackson.version>2.15.2</jackson.version>
        <okhttp.version>4.11.0</okhttp.version>
        <picocli.version>4.7.1</picocli.version>
    </properties>
    
    <dependencies>
        <!-- HTTP Client -->
        <dependency>
            <groupId>com.squareup.okhttp3</groupId>
            <artifactId>okhttp</artifactId>
            <version>${okhttp.version}</version>
        </dependency>
        
        <!-- JSON Processing -->
        <dependency>
            <groupId>com.fasterxml.jackson.core</groupId>
            <artifactId>jackson-databind</artifactId>
            <version>${jackson.version}</version>
        </dependency>
        
        <!-- CLI Framework -->
        <dependency>
            <groupId>info.picocli</groupId>
            <artifactId>picocli</artifactId>
            <version>${picocli.version}</version>
        </dependency>
        
        <!-- Testing -->
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>${junit.version}</version>
            <scope>test</scope>
        </dependency>
        
        <dependency>
            <groupId>org.mockito</groupId>
            <artifactId>mockito-core</artifactId>
            <version>${mockito.version}</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
                <configuration>
                    <source>11</source>
                    <target>11</target>
                </configuration>
            </plugin>
            
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.0.0-M9</version>
            </plugin>
            
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
                <version>3.0.4</version>
                <configuration>
                    <mainClass>com.anthropic.claude.agent.cli.ClaudeAgentCLI</mainClass>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
'''
    
    config_files = {
        "claude-agent-framework/python/pyproject.toml": pyproject_content,
        "claude-agent-framework/java/pom.xml": pom_content,
        "claude-agent-framework/.gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Java
target/
*.class
*.jar
*.war
*.ear
*.logs
hs_err_pid*

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Testing
.coverage
htmlcov/
.pytest_cache/
.tox/

# Documentation
docs/_build/
''',
    }
    
    for file_path, content in config_files.items():
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Created: {file_path}")
        except Exception as e:
            print(f"Error creating {file_path}: {e}")

def create_documentation():
    """Create initial documentation files."""
    print("\nCreating documentation...")
    
    readme_content = '''# Claude Agent Framework

A comprehensive, multi-language framework for building Claude-powered agents with support for beta tools including computer use and code execution.

## Features

- **Multi-language Support**: Python and Java implementations
- **Beta Tools**: Computer use, code execution, and more
- **MCP Integration**: Model Context Protocol support
- **CLI Interface**: Command-line tools for both languages
- **Extensible**: Plugin architecture for custom tools
- **Production Ready**: Comprehensive testing and documentation

## Quick Start

### Python

```bash
pip install claude-agent-framework
claude-agent --interactive
```

### Java

```bash
# Download the latest JAR from releases
java -jar claude-agent-framework.jar --interactive
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference/)
- [User Guides](docs/guides/)
- [Examples](docs/examples/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
'''
    
    getting_started_content = '''# Getting Started

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
'''
    
    docs = {
        "claude-agent-framework/README.md": readme_content,
        "claude-agent-framework/docs/getting-started.md": getting_started_content,
    }
    
    for file_path, content in docs.items():
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Created: {file_path}")
        except Exception as e:
            print(f"Error creating {file_path}: {e}")

def main():
    """Main migration function."""
    print("Claude Agent Framework Migration Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("agents") or not os.path.exists("cli.py"):
        print("Error: This script must be run from the claude-agent-to-agent directory")
        print("Current directory:", os.getcwd())
        return
    
    # Create backup
    print("Creating backup of current state...")
    if not os.path.exists("backup"):
        run_command("cp -r . backup")
        print("Backup created in 'backup' directory")
    
    # Perform migration
    create_directory_structure()
    migrate_python_files()
    create_new_package_files()
    create_configuration_files()
    create_documentation()
    
    print("\n" + "=" * 50)
    print("Migration completed successfully!")
    print("\nNext steps:")
    print("1. Review the migrated files in claude-agent-framework/")
    print("2. Update import statements in migrated files")
    print("3. Test the Python package: cd claude-agent-framework/python && pip install -e .")
    print("4. Begin Java implementation in claude-agent-framework/java/")
    print("5. Update documentation as needed")
    
    print(f"\nNew structure created in: {os.path.abspath('claude-agent-framework')}")

if __name__ == "__main__":
    main()
