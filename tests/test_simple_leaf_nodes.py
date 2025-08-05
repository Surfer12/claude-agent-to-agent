"""Simple tests for leaf nodes without complex dependencies."""

import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSimpleLeafNodes:
    """Test simple leaf node files."""
    
    def test_anthropic_bash_tool_structure(self):
        """Test the structure of anthropic_bash_tool.py."""
        file_path = project_root / "anthropic_bash_tool.py"
        assert file_path.exists(), "anthropic_bash_tool.py should exist"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check basic structure
        assert 'import anthropic' in content
        assert 'client = anthropic.Anthropic()' in content
        assert 'messages.create' in content
        assert 'bash_20250124' in content
        
    def test_anthropic_weather_tool_structure(self):
        """Test the structure of anthropic_weather_tool.py."""
        file_path = project_root / "anthropic_weather_tool.py"
        assert file_path.exists(), "anthropic_weather_tool.py should exist"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check basic structure
        assert 'import anthropic' in content
        assert 'get_weather' in content
        assert 'location' in content
        assert 'San Francisco' in content
        
    def test_anthropic_text_editor_structure(self):
        """Test the structure of anthropic_text_editor.py."""
        file_path = project_root / "anthropic_text_editor.py"
        assert file_path.exists(), "anthropic_text_editor.py should exist"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check basic structure
        assert 'import anthropic' in content
        assert 'text_editor_20250429' in content
        assert 'str_replace_based_edit_tool' in content
        
    def test_anthropic_web_search_structure(self):
        """Test the structure of anthropic_web_search.py."""
        file_path = project_root / "anthropic_web_search.py"
        assert file_path.exists(), "anthropic_web_search.py should exist"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check basic structure
        assert 'import anthropic' in content
        assert 'web_search_20250305' in content
        assert 'TypeScript' in content


class TestUnifiedAgentLeafNodes:
    """Test unified agent leaf node structure."""
    
    def test_unified_agent_init_structure(self):
        """Test unified_agent/__init__.py structure."""
        file_path = project_root / "unified_agent" / "__init__.py"
        assert file_path.exists(), "unified_agent/__init__.py should exist"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check exports
        assert 'UnifiedAgent' in content
        assert 'AgentConfig' in content
        assert 'ProviderType' in content
        assert '__version__' in content
        assert '__all__' in content
        
    def test_unified_agent_base_tool_structure(self):
        """Test unified_agent/tools/base.py structure."""
        file_path = project_root / "unified_agent" / "tools" / "base.py"
        assert file_path.exists(), "unified_agent/tools/base.py should exist"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check base tool structure
        assert 'class BaseTool' in content
        assert 'ABC' in content
        assert 'abstractmethod' in content
        assert 'execute' in content
        assert 'get_input_schema' in content


class TestJavaLeafNodes:
    """Test Java leaf node files."""
    
    def test_message_create_params_structure(self):
        """Test MessageCreateParams.java structure."""
        file_path = project_root / "src" / "main" / "java" / "com" / "anthropic" / "api" / "MessageCreateParams.java"
        assert file_path.exists(), "MessageCreateParams.java should exist"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check builder pattern structure
        assert 'class MessageCreateParams' in content
        assert 'static class Builder' in content
        assert 'public Builder model(' in content
        assert 'public Builder maxTokens(' in content
        assert 'public MessageCreateParams build()' in content
        
        # Check getters
        assert 'public String getModel()' in content
        assert 'public int getMaxTokens()' in content
        
    def test_basic_usage_example_structure(self):
        """Test BasicUsageExample.java structure."""
        file_path = project_root / "src" / "main" / "java" / "examples" / "BasicUsageExample.java"
        assert file_path.exists(), "BasicUsageExample.java should exist"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check example structure
        assert 'class BasicUsageExample' in content
        assert 'public static void main(' in content
        assert 'ANTHROPIC_API_KEY' in content


class TestProjectStructure:
    """Test overall project structure for leaf node identification."""
    
    def test_pixi_toml_exists(self):
        """Test that pixi.toml exists and has pytest."""
        file_path = project_root / "pixi.toml"
        assert file_path.exists(), "pixi.toml should exist"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        assert 'pytest' in content
        assert 'pytest-mock' in content
        assert 'pytest-asyncio' in content
        
    def test_pom_xml_has_mockito(self):
        """Test that pom.xml has Mockito dependencies."""
        file_path = project_root / "pom.xml"
        assert file_path.exists(), "pom.xml should exist"
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        assert 'mockito-core' in content
        assert 'mockito-junit-jupiter' in content
        
    def test_directory_structure(self):
        """Test key directories exist."""
        assert (project_root / "agents").exists()
        assert (project_root / "unified_agent").exists()
        assert (project_root / "src" / "main" / "java").exists()
        assert (project_root / "tests").exists()


class TestLeafNodeCharacteristics:
    """Test characteristics that make nodes 'leaf' nodes."""
    
    def test_simple_tool_files_are_minimal(self):
        """Test that simple tool files have minimal complexity."""
        tool_files = [
            "anthropic_bash_tool.py",
            "anthropic_weather_tool.py", 
            "anthropic_text_editor.py",
            "anthropic_web_search.py",
            "anthropic_client.py"
        ]
        
        for tool_file in tool_files:
            file_path = project_root / tool_file
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                
                lines = content.splitlines()
                # Simple tool files should be relatively short
                assert len(lines) < 50, f"{tool_file} should be a simple leaf node"
                
                # Should have minimal imports
                import_lines = [line for line in lines if line.strip().startswith('import')]
                assert len(import_lines) <= 3, f"{tool_file} should have minimal imports"
    
    def test_base_classes_are_abstract(self):
        """Test that base classes are properly abstract."""
        base_tool_path = project_root / "unified_agent" / "tools" / "base.py"
        if base_tool_path.exists():
            with open(base_tool_path, 'r') as f:
                content = f.read()
            
            # Should use ABC pattern
            assert 'ABC' in content
            assert '@abstractmethod' in content
            
    def test_leaf_directories_exist(self):
        """Test that leaf directories (no subdirectories) exist."""
        leaf_dirs = []
        
        for item in project_root.rglob("*"):
            if item.is_dir() and not any(part.startswith('.') for part in item.parts):
                subdirs = [p for p in item.iterdir() if p.is_dir()]
                files = [p for p in item.iterdir() if p.is_file() and not p.name.startswith('.')]
                
                if len(subdirs) == 0 and len(files) > 0:
                    leaf_dirs.append(item)
        
        # Should have some leaf directories
        assert len(leaf_dirs) > 0, "Project should have leaf directories"
        
        # Check some expected leaf directories
        expected_leaves = ['tests', 'scripts']
        for expected in expected_leaves:
            leaf_names = [d.name for d in leaf_dirs]
            if expected in [d.name for d in project_root.iterdir() if d.is_dir()]:
                assert expected in leaf_names, f"{expected} should be a leaf directory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
