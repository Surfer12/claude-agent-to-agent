#!/usr/bin/env python
"""Simple example using the Claude Agent CLI."""

import subprocess
import tempfile
import os
import sys

# Configure environment
if "ANTHROPIC_API_KEY" not in os.environ:
    print("Error: Please set ANTHROPIC_API_KEY environment variable.")
    sys.exit(1)

# Example 1: Simple prompt
print("Example 1: Simple prompt")
print("-" * 50)
result = subprocess.run(
    ["python", "../cli.py", "--prompt", "What is the capital of France?"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    capture_output=True,
    text=True,
)
print(result.stdout)
print("\n")

# Example 2: Using file_read tool
print("Example 2: Using file_read tool")
print("-" * 50)
with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
    temp_file.write("This is a test file created by the example script.")
    temp_file_path = temp_file.name

try:
    result = subprocess.run(
        [
            "python", 
            "../cli.py", 
            "--prompt", 
            f"Read the contents of this file: {temp_file_path}",
            "--tools", 
            "file_read",
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
finally:
    os.unlink(temp_file_path)

print("\n")

# Example 3: Using think tool
print("Example 3: Using think tool")
print("-" * 50)
result = subprocess.run(
    [
        "python", 
        "../cli.py", 
        "--prompt", 
        "I need to solve the quadratic equation x² + 5x + 6 = 0. Use the thinking tool to work through the solution step by step.",
        "--tools", 
        "think",
    ],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    capture_output=True,
    text=True,
)
print(result.stdout)

print("\nAll examples completed.")