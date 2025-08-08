#!/usr/bin/env python3
"""
Test script for OpenAI Agents SDK examples

This script validates that all the migration examples are properly structured
and can be imported without errors. For actual execution, you'll need to set
your OPENAI_API_KEY environment variable.
"""

import os
import importlib.util
import sys

def test_import(file_path, description):
    """Test if a Python file can be imported successfully."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        module = importlib.util.module_from_spec(spec)
        
        # Test import without executing main()
        spec.loader.exec_module(module)
        print(f"✅ {description}: Import successful")
        return True
    except Exception as e:
        print(f"❌ {description}: Import failed - {e}")
        return False

def main():
    print("🧪 Testing OpenAI Agents SDK Examples")
    print("=" * 50)
    
    # Check if Agents SDK is installed
    try:
        import agents
        print("✅ OpenAI Agents SDK is installed")
    except ImportError:
        print("❌ OpenAI Agents SDK not found. Run: pip install openai-agents")
        return
    
    # Test each example file
    examples = [
        ("agents_sdk_hello_world.py", "Hello World Example"),
        ("agents_sdk_handoffs_example.py", "Handoffs Example"),
        ("agents_sdk_functions_example.py", "Functions/Tools Example"),
        ("agents_sdk_math_agent_migration.py", "Math Agent Migration"),
        ("agents_sdk_sessions_example.py", "Sessions Example"),
    ]
    
    passed = 0
    total = len(examples)
    
    for filename, description in examples:
        file_path = os.path.join("/workspace", filename)
        if os.path.exists(file_path):
            if test_import(file_path, description):
                passed += 1
        else:
            print(f"❌ {description}: File not found - {filename}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} examples passed import tests")
    
    if passed == total:
        print("🎉 All examples are properly structured!")
        print("\n📝 To run the examples:")
        print("1. Set your API key: export OPENAI_API_KEY='your-key-here'")
        print("2. Run any example: python3 agents_sdk_hello_world.py")
    else:
        print("⚠️  Some examples have issues. Check the error messages above.")
    
    # Show key differences summary
    print("\n🔄 Key Migration Changes:")
    print("- Swarm → Agents SDK")
    print("- functions=[...] → tools=[...] with @function_tool")
    print("- transfer functions → handoffs=[...]")
    print("- Swarm().run() → Runner.run() / Runner.run_sync()")
    print("- Manual message history → Sessions")
    
    print("\n📚 Files created:")
    for filename, description in examples:
        print(f"- {filename}: {description}")
    print("- SWARM_TO_AGENTS_SDK_MIGRATION_GUIDE.md: Complete migration guide")

if __name__ == "__main__":
    main()