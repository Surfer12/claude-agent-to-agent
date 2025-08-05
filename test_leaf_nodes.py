#!/usr/bin/env python3
"""
Leaf Node Testing and Analysis Script

This script identifies, tests, and analyzes leaf nodes in the claude-agent-to-agent project.
Leaf nodes are defined as:
- Files/modules at the edge of the directory tree
- Classes without further subclasses  
- Functions with no internal calls
- Entities that need revision based on static analysis
"""

import os
import sys
import ast
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import importlib.util


@dataclass
class LeafNode:
    """Represents a leaf node in the project."""
    path: str
    type: str  # 'file', 'class', 'function'
    name: str
    dependencies: List[str]
    complexity: int
    needs_revision: bool
    revision_reasons: List[str]


class LeafNodeAnalyzer:
    """Analyzes the project to identify leaf nodes."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.leaf_nodes: List[LeafNode] = []
        self.python_files: List[Path] = []
        self.java_files: List[Path] = []
        
    def analyze_project(self) -> Dict[str, List[LeafNode]]:
        """Analyze the entire project for leaf nodes."""
        print("🔍 Analyzing project structure for leaf nodes...")
        
        # Find all source files
        self._find_source_files()
        
        # Analyze Python files
        python_leaves = self._analyze_python_files()
        
        # Analyze Java files  
        java_leaves = self._analyze_java_files()
        
        # Analyze directory structure
        directory_leaves = self._analyze_directory_structure()
        
        return {
            'python': python_leaves,
            'java': java_leaves,
            'directories': directory_leaves
        }
    
    def _find_source_files(self):
        """Find all Python and Java source files."""
        for file_path in self.project_root.rglob("*.py"):
            if not any(part.startswith('.') for part in file_path.parts):
                if 'test' not in str(file_path).lower() or 'tests' not in str(file_path).lower():
                    self.python_files.append(file_path)
        
        for file_path in self.project_root.rglob("*.java"):
            if not any(part.startswith('.') for part in file_path.parts):
                self.java_files.append(file_path)
    
    def _analyze_python_files(self) -> List[LeafNode]:
        """Analyze Python files for leaf nodes."""
        leaf_nodes = []
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST
                tree = ast.parse(content)
                
                # Analyze file-level characteristics
                file_leaf = self._analyze_python_file(file_path, tree, content)
                if file_leaf:
                    leaf_nodes.append(file_leaf)
                
                # Analyze classes and functions
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_leaf = self._analyze_python_class(file_path, node, tree)
                        if class_leaf:
                            leaf_nodes.append(class_leaf)
                    
                    elif isinstance(node, ast.FunctionDef):
                        func_leaf = self._analyze_python_function(file_path, node, tree)
                        if func_leaf:
                            leaf_nodes.append(func_leaf)
            
            except Exception as e:
                print(f"⚠️  Error analyzing {file_path}: {e}")
        
        return leaf_nodes
    
    def _analyze_python_file(self, file_path: Path, tree: ast.AST, content: str) -> LeafNode:
        """Analyze a Python file as a potential leaf node."""
        relative_path = str(file_path.relative_to(self.project_root))
        
        # Check if it's a simple script (leaf characteristics)
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        # Simple files with minimal structure are leaf candidates
        is_simple = len(classes) <= 1 and len(functions) <= 3
        is_tool_file = 'tool' in file_path.name.lower()
        is_utility = any(keyword in file_path.name.lower() for keyword in ['util', 'helper', 'config'])
        
        needs_revision = False
        revision_reasons = []
        
        # Check for revision needs
        if len(content.splitlines()) < 50 and not content.strip():
            needs_revision = True
            revision_reasons.append("Empty or very small file")
        
        if 'TODO' in content or 'FIXME' in content:
            needs_revision = True
            revision_reasons.append("Contains TODO/FIXME comments")
        
        if len(imports) > 10:
            needs_revision = True
            revision_reasons.append("Too many imports - high coupling")
        
        if is_simple or is_tool_file or is_utility:
            return LeafNode(
                path=relative_path,
                type='file',
                name=file_path.stem,
                dependencies=[self._extract_import_name(imp) for imp in imports],
                complexity=len(classes) + len(functions),
                needs_revision=needs_revision,
                revision_reasons=revision_reasons
            )
        
        return None
    
    def _analyze_python_class(self, file_path: Path, node: ast.ClassDef, tree: ast.AST) -> LeafNode:
        """Analyze a Python class as a potential leaf node."""
        # Check if class has no subclasses in the project
        class_name = node.name
        has_subclasses = self._has_subclasses(class_name, tree)
        
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        
        needs_revision = False
        revision_reasons = []
        
        # Check for revision indicators
        if len(methods) == 0:
            needs_revision = True
            revision_reasons.append("Empty class with no methods")
        
        if len(methods) > 20:
            needs_revision = True
            revision_reasons.append("Class too large - consider splitting")
        
        # Classes without subclasses are leaf candidates
        if not has_subclasses:
            return LeafNode(
                path=str(file_path.relative_to(self.project_root)),
                type='class',
                name=class_name,
                dependencies=self._get_class_dependencies(node),
                complexity=len(methods),
                needs_revision=needs_revision,
                revision_reasons=revision_reasons
            )
        
        return None
    
    def _analyze_python_function(self, file_path: Path, node: ast.FunctionDef, tree: ast.AST) -> LeafNode:
        """Analyze a Python function as a potential leaf node."""
        func_name = node.name
        
        # Check if function has no internal calls to other functions in the same module
        has_internal_calls = self._has_internal_calls(node, tree)
        
        needs_revision = False
        revision_reasons = []
        
        # Check complexity
        complexity = self._calculate_cyclomatic_complexity(node)
        if complexity > 10:
            needs_revision = True
            revision_reasons.append(f"High cyclomatic complexity: {complexity}")
        
        # Check function length
        func_lines = len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
        if func_lines > 50:
            needs_revision = True
            revision_reasons.append(f"Function too long: {func_lines} statements")
        
        # Functions without internal calls are leaf candidates
        if not has_internal_calls and not func_name.startswith('_'):
            return LeafNode(
                path=str(file_path.relative_to(self.project_root)),
                type='function',
                name=func_name,
                dependencies=self._get_function_dependencies(node),
                complexity=complexity,
                needs_revision=needs_revision,
                revision_reasons=revision_reasons
            )
        
        return None
    
    def _analyze_java_files(self) -> List[LeafNode]:
        """Analyze Java files for leaf nodes."""
        leaf_nodes = []
        
        for file_path in self.java_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple heuristic analysis for Java files
                class_count = content.count('class ')
                method_count = content.count('public ') + content.count('private ') + content.count('protected ')
                
                needs_revision = False
                revision_reasons = []
                
                if 'TODO' in content or 'FIXME' in content:
                    needs_revision = True
                    revision_reasons.append("Contains TODO/FIXME comments")
                
                if len(content.splitlines()) > 500:
                    needs_revision = True
                    revision_reasons.append("File too large")
                
                # Simple files are leaf candidates
                if class_count <= 2:
                    leaf_nodes.append(LeafNode(
                        path=str(file_path.relative_to(self.project_root)),
                        type='java_file',
                        name=file_path.stem,
                        dependencies=[],  # Would need proper Java parsing
                        complexity=method_count,
                        needs_revision=needs_revision,
                        revision_reasons=revision_reasons
                    ))
            
            except Exception as e:
                print(f"⚠️  Error analyzing Java file {file_path}: {e}")
        
        return leaf_nodes
    
    def _analyze_directory_structure(self) -> List[LeafNode]:
        """Analyze directory structure for leaf directories."""
        leaf_dirs = []
        
        for dir_path in self.project_root.rglob("*"):
            if dir_path.is_dir() and not any(part.startswith('.') for part in dir_path.parts):
                # Check if directory has no subdirectories (leaf directory)
                subdirs = [p for p in dir_path.iterdir() if p.is_dir()]
                files = [p for p in dir_path.iterdir() if p.is_file()]
                
                if len(subdirs) == 0 and len(files) > 0:
                    needs_revision = len(files) > 20  # Too many files in one directory
                    
                    leaf_dirs.append(LeafNode(
                        path=str(dir_path.relative_to(self.project_root)),
                        type='directory',
                        name=dir_path.name,
                        dependencies=[],
                        complexity=len(files),
                        needs_revision=needs_revision,
                        revision_reasons=["Too many files in directory"] if needs_revision else []
                    ))
        
        return leaf_dirs
    
    # Helper methods
    def _extract_import_name(self, node) -> str:
        """Extract import name from AST node."""
        if isinstance(node, ast.Import):
            return node.names[0].name if node.names else ""
        elif isinstance(node, ast.ImportFrom):
            return node.module or ""
        return ""
    
    def _has_subclasses(self, class_name: str, tree: ast.AST) -> bool:
        """Check if a class has subclasses in the current file."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.bases:
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == class_name:
                        return True
        return False
    
    def _has_internal_calls(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function makes calls to other functions in the same module."""
        # Get all function names in the module
        func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        
        # Check if this function calls any of them
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in func_names and node.func.id != func_node.name:
                    return True
        return False
    
    def _get_class_dependencies(self, node: ast.ClassDef) -> List[str]:
        """Get class dependencies."""
        deps = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                deps.append(base.id)
        return deps
    
    def _get_function_dependencies(self, node: ast.FunctionDef) -> List[str]:
        """Get function dependencies."""
        deps = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                deps.append(n.func.id)
        return list(set(deps))
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for n in ast.walk(node):
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(n, ast.ExceptHandler):
                complexity += 1
            elif isinstance(n, ast.BoolOp):
                complexity += len(n.values) - 1
        
        return complexity


def run_tests():
    """Run all tests for leaf nodes."""
    print("🧪 Running leaf node tests...")
    
    # Run Python tests
    python_result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/test_think_tool.py",
        "tests/test_tool_util.py", 
        "tests/test_anthropic_tools.py",
        "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    print("Python Test Results:")
    print(python_result.stdout)
    if python_result.stderr:
        print("Errors:", python_result.stderr)
    
    # Run Java tests
    java_result = subprocess.run([
        "mvn", "test", "-Dtest=MessageCreateParamsTest"
    ], capture_output=True, text=True)
    
    print("\nJava Test Results:")
    print(java_result.stdout)
    if java_result.stderr:
        print("Errors:", java_result.stderr)
    
    return python_result.returncode == 0 and java_result.returncode == 0


def run_static_analysis():
    """Run static analysis tools."""
    print("🔍 Running static analysis...")
    
    # Run flake8
    flake8_result = subprocess.run([
        "flake8", ".", "--max-line-length=88", "--extend-ignore=E203,W503,E402",
        "--exclude=.git,__pycache__,venv,.venv,.pixi"
    ], capture_output=True, text=True)
    
    print("Flake8 Results:")
    print(flake8_result.stdout)
    
    # Run mypy on key files
    mypy_result = subprocess.run([
        "mypy", "agents/", "--ignore-missing-imports"
    ], capture_output=True, text=True)
    
    print("\nMypy Results:")
    print(mypy_result.stdout)
    
    return flake8_result.returncode == 0 and mypy_result.returncode == 0


def generate_todo_list(leaf_nodes: Dict[str, List[LeafNode]]) -> List[str]:
    """Generate TODO list based on analysis."""
    todos = []
    
    for category, nodes in leaf_nodes.items():
        for node in nodes:
            if node.needs_revision:
                todos.append(f"📝 {category.upper()}: {node.path} - {', '.join(node.revision_reasons)}")
    
    # Add general TODOs
    todos.extend([
        "🧪 Add more comprehensive integration tests",
        "📚 Update documentation for leaf node APIs",
        "🔧 Refactor high-complexity functions",
        "🛡️  Add input validation to public APIs",
        "⚡ Optimize performance of frequently called functions",
        "🔍 Add logging to critical leaf nodes",
        "🧹 Remove unused imports and dependencies"
    ])
    
    return todos


def main():
    """Main execution function."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    print("🌳 Claude Agent-to-Agent Leaf Node Analysis")
    print("=" * 50)
    
    # Analyze project structure
    analyzer = LeafNodeAnalyzer(project_root)
    leaf_nodes = analyzer.analyze_project()
    
    # Print analysis results
    print(f"\n📊 Analysis Results:")
    for category, nodes in leaf_nodes.items():
        print(f"\n{category.upper()} Leaf Nodes: {len(nodes)}")
        for node in nodes[:5]:  # Show first 5
            status = "⚠️ " if node.needs_revision else "✅"
            print(f"  {status} {node.name} ({node.type}) - Complexity: {node.complexity}")
        
        if len(nodes) > 5:
            print(f"  ... and {len(nodes) - 5} more")
    
    # Run tests
    print(f"\n🧪 Testing Leaf Nodes:")
    test_success = run_tests()
    print(f"Tests {'✅ PASSED' if test_success else '❌ FAILED'}")
    
    # Run static analysis
    print(f"\n🔍 Static Analysis:")
    analysis_success = run_static_analysis()
    print(f"Analysis {'✅ CLEAN' if analysis_success else '⚠️  ISSUES FOUND'}")
    
    # Generate TODO list
    print(f"\n📝 TODO List:")
    todos = generate_todo_list(leaf_nodes)
    for i, todo in enumerate(todos[:10], 1):
        print(f"{i:2d}. {todo}")
    
    if len(todos) > 10:
        print(f"    ... and {len(todos) - 10} more items")
    
    # Summary
    total_nodes = sum(len(nodes) for nodes in leaf_nodes.values())
    revision_needed = sum(1 for nodes in leaf_nodes.values() for node in nodes if node.needs_revision)
    
    print(f"\n📈 Summary:")
    print(f"   Total leaf nodes identified: {total_nodes}")
    print(f"   Nodes needing revision: {revision_needed}")
    print(f"   Test coverage: {'Good' if test_success else 'Needs improvement'}")
    print(f"   Code quality: {'Good' if analysis_success else 'Needs attention'}")
    
    return 0 if test_success and analysis_success else 1


if __name__ == "__main__":
    sys.exit(main())
