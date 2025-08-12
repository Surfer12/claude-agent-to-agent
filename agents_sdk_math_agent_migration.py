#!/usr/bin/env python3
"""
OpenAI Agents SDK - Math Agent Migration

This migrates the existing Swarm math agent to the new Agents SDK.
It maintains the same functionality but uses the production-ready framework.
"""

import os
import asyncio
import math
from agents import Agent, Runner, function_tool

@function_tool
def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions with helpful error handling."""
    try:
        original_expr = expression
        
        # Handle the specific case from the original: 2+@
        expression = expression.replace('@', '2')  # Your exact case!
        expression = expression.replace('x', '*')
        expression = expression.replace('^', '**')
        
        # Safe evaluation with allowed functions
        allowed_names = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "pi": math.pi, "e": math.e, "pow": pow, "log": math.log,
            "log10": math.log10, "exp": math.exp, "floor": math.floor, "ceil": math.ceil
        }
        
        result = eval(expression, allowed_names, {})
        
        response = f"🔢 **Math Agent here!** ✅ I calculated: {original_expr}"
        if expression != original_expr:
            response += f" (I interpreted '@' as '2' and 'x' as '*', so: {expression})"
        response += f" = **{result}**"
        
        return response
        
    except Exception as e:
        return f"🔢 **Math Agent here!** ❌ I couldn't calculate '{original_expr}'. Try: 2+2, sqrt(16), sin(pi/2), or 3^2"

@function_tool
def get_math_constants() -> str:
    """Get common mathematical constants."""
    constants = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau if hasattr(math, 'tau') else 2 * math.pi,
        "golden_ratio": (1 + math.sqrt(5)) / 2
    }
    
    result = "🔢 **Mathematical Constants:**\n"
    for name, value in constants.items():
        result += f"- {name}: {value:.10f}\n"
    
    return result

@function_tool  
def solve_quadratic(a: float, b: float, c: float) -> str:
    """Solve quadratic equation ax² + bx + c = 0."""
    try:
        discriminant = b**2 - 4*a*c
        
        if discriminant > 0:
            x1 = (-b + math.sqrt(discriminant)) / (2*a)
            x2 = (-b - math.sqrt(discriminant)) / (2*a)
            return f"🔢 **Quadratic Solutions:** x₁ = {x1:.6f}, x₂ = {x2:.6f}"
        elif discriminant == 0:
            x = -b / (2*a)
            return f"🔢 **Quadratic Solution:** x = {x:.6f} (double root)"
        else:
            real_part = -b / (2*a)
            imag_part = math.sqrt(-discriminant) / (2*a)
            return f"🔢 **Complex Solutions:** x₁ = {real_part:.6f} + {imag_part:.6f}i, x₂ = {real_part:.6f} - {imag_part:.6f}i"
            
    except Exception as e:
        return f"🔢 **Math Agent Error:** Could not solve quadratic equation. Make sure 'a' is not zero."

def main():
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create the math specialist agent
    math_agent = Agent(
        name="Math Specialist",
        instructions="""You are a mathematical computation specialist. You can:
        
        1. **Calculate expressions**: Use the calculate function for any mathematical expression
           - Supports basic operations: +, -, *, /, ^(power)
           - Supports functions: sqrt, sin, cos, tan, log, exp, etc.
           - Handles special cases like '@' symbol (converts to '2')
           
        2. **Solve quadratic equations**: Use solve_quadratic for ax² + bx + c = 0
        
        3. **Provide constants**: Use get_math_constants for mathematical constants
        
        Always be enthusiastic about math and provide clear explanations!
        If a user asks for a calculation, always use the calculate function.
        If they ask about quadratic equations, use the solve_quadratic function.
        """,
        tools=[calculate, get_math_constants, solve_quadratic],
    )
    
    # Create a general agent that can handoff to math specialist
    general_agent = Agent(
        name="General Assistant", 
        instructions="""You are a helpful general assistant. When users ask mathematical questions 
        or want calculations performed, handoff to the Math Specialist who has specialized tools 
        for mathematical computations.""",
        handoffs=[math_agent]
    )

    async def run_examples():
        print("🔢 Testing Math Agent Migration")
        print("=" * 50)
        
        # Test the original problematic case: 2+@
        print("\n1. Testing the original '2+@' case:")
        result = await Runner.run(math_agent, input="Calculate 2+@")
        print(f"Response: {result.final_output}")
        
        # Test complex expression
        print("\n2. Testing complex expression:")
        result = await Runner.run(math_agent, input="Calculate sqrt(16) + sin(pi/2) * 5^2")
        print(f"Response: {result.final_output}")
        
        # Test quadratic equation
        print("\n3. Testing quadratic equation:")
        result = await Runner.run(math_agent, input="Solve the quadratic equation x² - 5x + 6 = 0")
        print(f"Response: {result.final_output}")
        
        # Test mathematical constants
        print("\n4. Testing mathematical constants:")
        result = await Runner.run(math_agent, input="Show me some important mathematical constants")
        print(f"Response: {result.final_output}")
        
        # Test handoff from general agent
        print("\n5. Testing handoff from general agent:")
        result = await Runner.run(general_agent, input="I need help calculating the area of a circle with radius 5")
        print(f"Response: {result.final_output}")

    try:
        asyncio.run(run_examples())
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your OPENAI_API_KEY is set correctly.")

if __name__ == "__main__":
    main()