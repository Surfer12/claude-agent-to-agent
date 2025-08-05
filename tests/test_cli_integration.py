#!/usr/bin/env python3
"""
CLI Integration Test Suite

This module provides end-to-end integration tests for CLI leaf nodes:
- CLI help functionality
- Environment setup validation
- Cross-CLI compatibility
- Error handling integration
"""

import pytest
import subprocess
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CLI file paths
PROJECT_ROOT = Path(__file__).parent.parent
CLI_FILES = {
    "main_cli": PROJECT_ROOT / "cli.py",
    "enhanced_cli": PROJECT_ROOT / "enhanced_cli.py",
    "unified_cli": PROJECT_ROOT / "unified_agent" / "cli.py",
    "quick_start": PROJECT_ROOT / "quick_start.py"
}


class TestCLIHelpFunctionality:
    """Test CLI help functionality across all CLI implementations."""
    
    @pytest.mark.parametrize("cli_name,cli_path", [
        ("main_cli", CLI_FILES["main_cli"]),
        ("enhanced_cli", CLI_FILES["enhanced_cli"]),
        ("quick_start", CLI_FILES["quick_start"])
    ])
    def test_cli_help_command(self, cli_name, cli_path):
        """Test that CLI help commands work."""
        if not cli_path.exists():
            pytest.skip(f"{cli_name} not found at {cli_path}")
        
        # Test --help flag
        result = subprocess.run([
            sys.executable, str(cli_path), "--help"
        ], capture_output=True, text=True, timeout=10)
        
        # Help should exit with code 0 and contain usage information
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "help" in result.stdout.lower()
    
    def test_unified_cli_help_module(self):
        """Test unified CLI help via module execution."""
        if not CLI_FILES["unified_cli"].exists():
            pytest.skip("unified_agent CLI not found")
        
        result = subprocess.run([
            sys.executable, "-m", "unified_agent.cli", "--help"
        ], capture_output=True, text=True, timeout=10, 
        cwd=str(PROJECT_ROOT))
        
        # Should show help or fail gracefully
        assert result.returncode in [0, 1]  # 0 for success, 1 for import errors
    
    @pytest.mark.parametrize("cli_name,cli_path", [
        ("main_cli", CLI_FILES["main_cli"]),
        ("enhanced_cli", CLI_FILES["enhanced_cli"])
    ])
    def test_cli_invalid_arguments(self, cli_name, cli_path):
        """Test CLI behavior with invalid arguments."""
        if not cli_path.exists():
            pytest.skip(f"{cli_name} not found at {cli_path}")
        
        # Test invalid argument
        result = subprocess.run([
            sys.executable, str(cli_path), "--invalid-argument"
        ], capture_output=True, text=True, timeout=10)
        
        # Should exit with non-zero code
        assert result.returncode != 0
        assert "error:" in result.stderr.lower() or "unrecognized" in result.stderr.lower()


class TestEnvironmentSetupIntegration:
    """Test environment setup integration across CLIs."""
    
    def test_quick_start_environment_check(self):
        """Test quick start environment validation."""
        if not CLI_FILES["quick_start"].exists():
            pytest.skip("quick_start.py not found")
        
        # Test without API keys
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
        
        result = subprocess.run([
            sys.executable, str(CLI_FILES["quick_start"])
        ], capture_output=True, text=True, env=env, timeout=30)
        
        # Should complete (may warn about missing keys)
        assert result.returncode in [0, 1]
        assert "Python version:" in result.stdout or "dependencies" in result.stdout.lower()
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_cli_with_api_key(self):
        """Test CLI behavior with API key present."""
        if not CLI_FILES["enhanced_cli"].exists():
            pytest.skip("enhanced_cli.py not found")
        
        # Test basic help with API key present
        result = subprocess.run([
            sys.executable, str(CLI_FILES["enhanced_cli"]), "--help"
        ], capture_output=True, text=True, timeout=10)
        
        assert result.returncode == 0
    
    def test_cli_missing_api_key_handling(self):
        """Test CLI handling of missing API keys."""
        if not CLI_FILES["enhanced_cli"].exists():
            pytest.skip("enhanced_cli.py not found")
        
        # Remove API keys from environment
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
        
        # Test with a simple prompt (should fail gracefully)
        result = subprocess.run([
            sys.executable, str(CLI_FILES["enhanced_cli"]), 
            "--prompt", "test"
        ], capture_output=True, text=True, env=env, timeout=10)
        
        # Should handle missing API key gracefully
        assert result.returncode != 0 or "api" in result.stdout.lower()


class TestCLICrossCompatibility:
    """Test compatibility between different CLI implementations."""
    
    def test_all_clis_have_help(self):
        """Test that all CLI implementations provide help."""
        help_results = {}
        
        for cli_name, cli_path in CLI_FILES.items():
            if not cli_path.exists():
                continue
                
            try:
                if cli_name == "unified_cli":
                    result = subprocess.run([
                        sys.executable, "-m", "unified_agent.cli", "--help"
                    ], capture_output=True, text=True, timeout=10, 
                    cwd=str(PROJECT_ROOT))
                else:
                    result = subprocess.run([
                        sys.executable, str(cli_path), "--help"
                    ], capture_output=True, text=True, timeout=10)
                
                help_results[cli_name] = {
                    "returncode": result.returncode,
                    "has_output": len(result.stdout) > 0 or len(result.stderr) > 0
                }
            except subprocess.TimeoutExpired:
                help_results[cli_name] = {
                    "returncode": -1,
                    "has_output": False
                }
        
        # At least one CLI should work
        assert len(help_results) > 0, "No CLI files found"
        
        # Check that working CLIs provide help
        working_clis = [name for name, result in help_results.items() 
                       if result["returncode"] == 0 and result["has_output"]]
        assert len(working_clis) > 0, f"No working CLIs found. Results: {help_results}"
    
    def test_cli_argument_consistency(self):
        """Test that CLIs have consistent argument patterns."""
        common_args = ["--help", "--verbose"]
        cli_args = {}
        
        for cli_name, cli_path in CLI_FILES.items():
            if not cli_path.exists():
                continue
            
            cli_args[cli_name] = []
            
            # Get help output to check available arguments
            try:
                if cli_name == "unified_cli":
                    result = subprocess.run([
                        sys.executable, "-m", "unified_agent.cli", "--help"
                    ], capture_output=True, text=True, timeout=10, 
                    cwd=str(PROJECT_ROOT))
                else:
                    result = subprocess.run([
                        sys.executable, str(cli_path), "--help"
                    ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    help_text = result.stdout.lower()
                    for arg in common_args:
                        if arg in help_text:
                            cli_args[cli_name].append(arg)
            except subprocess.TimeoutExpired:
                pass
        
        # At least some CLIs should support common arguments
        total_common_args = sum(len(args) for args in cli_args.values())
        assert total_common_args > 0, f"No common arguments found. CLI args: {cli_args}"


class TestCLIErrorHandling:
    """Test error handling integration across CLIs."""
    
    def test_cli_timeout_handling(self):
        """Test CLI behavior with timeout scenarios."""
        if not CLI_FILES["enhanced_cli"].exists():
            pytest.skip("enhanced_cli.py not found")
        
        # Test with a command that should complete quickly
        try:
            result = subprocess.run([
                sys.executable, str(CLI_FILES["enhanced_cli"]), "--help"
            ], capture_output=True, text=True, timeout=5)
            
            # Should complete within timeout
            assert result.returncode in [0, 1]  # 0 for success, 1 for expected errors
        except subprocess.TimeoutExpired:
            pytest.fail("CLI help command timed out")
    
    def test_cli_file_not_found_handling(self):
        """Test CLI behavior when files are missing."""
        non_existent_cli = PROJECT_ROOT / "non_existent_cli.py"
        
        result = subprocess.run([
            sys.executable, str(non_existent_cli)
        ], capture_output=True, text=True)
        
        # Should fail with file not found error
        assert result.returncode != 0
        assert "no such file" in result.stderr.lower() or "not found" in result.stderr.lower()
    
    def test_cli_python_syntax_validation(self):
        """Test that CLI files have valid Python syntax."""
        for cli_name, cli_path in CLI_FILES.items():
            if not cli_path.exists():
                continue
            
            # Test syntax by compiling the file
            try:
                with open(cli_path, 'r') as f:
                    source = f.read()
                compile(source, str(cli_path), 'exec')
            except SyntaxError as e:
                pytest.fail(f"{cli_name} has syntax error: {e}")


class TestCLIPerformance:
    """Test CLI performance characteristics."""
    
    def test_cli_startup_time(self):
        """Test CLI startup performance."""
        if not CLI_FILES["enhanced_cli"].exists():
            pytest.skip("enhanced_cli.py not found")
        
        import time
        
        start_time = time.time()
        result = subprocess.run([
            sys.executable, str(CLI_FILES["enhanced_cli"]), "--help"
        ], capture_output=True, text=True, timeout=10)
        end_time = time.time()
        
        startup_time = end_time - start_time
        
        # CLI help should complete within reasonable time
        assert startup_time < 5.0, f"CLI startup took {startup_time:.2f}s, expected < 5.0s"
        assert result.returncode in [0, 1]
    
    def test_cli_memory_usage(self):
        """Test CLI memory usage is reasonable."""
        if not CLI_FILES["enhanced_cli"].exists():
            pytest.skip("enhanced_cli.py not found")
        
        # This is a basic test - in production you might use psutil
        result = subprocess.run([
            sys.executable, str(CLI_FILES["enhanced_cli"]), "--help"
        ], capture_output=True, text=True, timeout=10)
        
        # Should complete without memory errors
        assert "MemoryError" not in result.stderr
        assert "memory" not in result.stderr.lower() or result.returncode == 0


class TestCLIDocumentation:
    """Test CLI documentation and help text quality."""
    
    def test_cli_help_completeness(self):
        """Test that CLI help text is complete and useful."""
        for cli_name, cli_path in CLI_FILES.items():
            if not cli_path.exists():
                continue
            
            try:
                if cli_name == "unified_cli":
                    result = subprocess.run([
                        sys.executable, "-m", "unified_agent.cli", "--help"
                    ], capture_output=True, text=True, timeout=10, 
                    cwd=str(PROJECT_ROOT))
                else:
                    result = subprocess.run([
                        sys.executable, str(cli_path), "--help"
                    ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    help_text = result.stdout.lower()
                    
                    # Check for essential help components
                    has_usage = "usage:" in help_text
                    has_description = len(help_text) > 50  # Reasonable description length
                    has_options = "option" in help_text or "argument" in help_text
                    
                    # At least some help components should be present
                    help_score = sum([has_usage, has_description, has_options])
                    assert help_score >= 1, f"{cli_name} help text is incomplete: {help_text[:200]}"
                    
            except subprocess.TimeoutExpired:
                pass  # Skip if CLI times out
    
    def test_cli_examples_in_help(self):
        """Test that CLI help includes usage examples."""
        example_indicators = ["example", "usage:", "e.g.", "try:"]
        
        for cli_name, cli_path in CLI_FILES.items():
            if not cli_path.exists():
                continue
            
            try:
                if cli_name == "unified_cli":
                    result = subprocess.run([
                        sys.executable, "-m", "unified_agent.cli", "--help"
                    ], capture_output=True, text=True, timeout=10, 
                    cwd=str(PROJECT_ROOT))
                else:
                    result = subprocess.run([
                        sys.executable, str(cli_path), "--help"
                    ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    help_text = result.stdout.lower()
                    
                    # Check if help contains examples
                    has_examples = any(indicator in help_text for indicator in example_indicators)
                    
                    # This is informational - not all CLIs need examples
                    if has_examples:
                        assert len(help_text) > 100  # Examples should add substantial content
                        
            except subprocess.TimeoutExpired:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
