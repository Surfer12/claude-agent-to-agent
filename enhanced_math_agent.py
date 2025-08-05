import json
import math
import re
from typing import Union
from swarm import Agent

def calculate(expression: str) -> str:
    """
    Safely evaluate mathematical expressions.
    Handles basic arithmetic, functions like sqrt, sin, cos, etc.
    """
    try:
        # Clean the expression - remove invalid characters but be helpful about it
        original_expr = expression
        
        # Replace common typos/symbols
        expression = expression.replace('@', '2')  # Common typo
        expression = expression.replace('x', '*')  # Multiplication
        expression = expression.replace('^', '**') # Exponentiation
        
        # Check for invalid characters (basic safety)
        if re.search(r'[a-zA-Z]', expression.replace('sqrt', '').replace('sin', '').replace('cos', '').replace('tan', '').replace('log', '').replace('pi', '').replace('e', '')):
            return json.dumps({
                "original": original_expr,
                "error": "Expression contains invalid characters",
                "suggestion": f"Did you mean: {expression}?",
                "help": "I can handle: +, -, *, /, **, sqrt(), sin(), cos(), tan(), log(), pi, e"
            })
        
        # Safe evaluation with math functions available
        allowed_names = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "pi": math.pi, "e": math.e, "pow": pow
        }
        
        result = eval(expression, allowed_names, {})
        
        return json.dumps({
            "original": original_expr,
            "expression": expression if expression != original_expr else None,
            "result": result,
            "type": type(result).__name__
        })
        
    except ZeroDivisionError:
        return json.dumps({
            "original": original_expr,
            "error": "Division by zero",
            "result": "undefined"
        })
    except Exception as e:
        return json.dumps({
            "original": original_expr,
            "error": str(e),
            "help": "Try simple expressions like: 2+2, sqrt(16), sin(pi/2)"
        })

def solve_equation(equation: str) -> str:
    """
    Help solve basic equations (educational purposes).
    """
    return json.dumps({
        "equation": equation,
        "note": "For complex equation solving, I recommend using specialized tools",
        "simple_help": "I can evaluate expressions like 2*x when x=5: calculate('2*5')"
    })

def math_help() -> str:
    """Provide help with available math functions."""
    return json.dumps({
        "available_operations": [
            "Basic arithmetic: +, -, *, /, **",
            "Functions: sqrt(), sin(), cos(), tan(), log(), log10(), exp()",
            "Constants: pi, e",
            "Examples: sqrt(16), sin(pi/2), 2**3, log(10)"
        ],
        "tips": [
            "Use * for multiplication (not x)",
            "Use ** for exponents (not ^)",
            "Parentheses work: (2+3)*4"
        ]
    })

def transfer_back_to_triage():
    """Transfer back to triage for non-math questions."""
    # Import here to avoid circular imports
    from improved_triage_system import triage_agent
    return triage_agent

# Enhanced Math Agent
math_agent = Agent(
    name="Math Agent",
    instructions="""You are a specialized math assistant! 🔢

I can help with:
• Basic calculations and arithmetic
• Mathematical functions (sqrt, sin, cos, tan, log)
• Expression evaluation
• Math explanations and guidance

When users give me math expressions:
1. Use calculate() function to process them
2. If there are typos (like @ instead of 2), I'll suggest corrections
3. Provide clear, helpful responses
4. Show the result and explain if needed

For equations or complex problems, use solve_equation() or provide guidance.

If users ask about help, use math_help() to show what I can do.

Always identify myself as the Math Agent so users know they're connected!
For non-math questions, use transfer_back_to_triage().""",
    functions=[calculate, solve_equation, math_help, transfer_back_to_triage],
)

# Example usage
if __name__ == "__main__":
    from swarm import Swarm
    
    client = Swarm()
    
    # Test the problematic expression from your conversation
    test_cases = [
        "Hi, I'm the user asking about 2+@",
        "what is 2+@",
        "can you help me with math?",
        "what can you do?"
    ]
    
    messages = []
    
    for user_input in test_cases:
        print(f"\nUser: {user_input}")
        messages.append({"role": "user", "content": user_input})
        
        response = client.run(
            agent=math_agent,
            messages=messages,
            debug=False
        )
        
        messages.extend(response.messages)
        print(f"Math Agent: {response.messages[-1]['content']}")