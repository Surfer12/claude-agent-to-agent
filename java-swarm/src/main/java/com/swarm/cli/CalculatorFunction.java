package com.swarm.cli;

import com.swarm.types.AgentFunction;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import java.util.Map;

/**
 * Simple calculator function for demonstration purposes.
 */
public class CalculatorFunction implements AgentFunction {
    
    private final ScriptEngine engine;
    
    public CalculatorFunction() {
        ScriptEngineManager manager = new ScriptEngineManager();
        this.engine = manager.getEngineByName("JavaScript");
    }
    
    @Override
    public Object execute(Map<String, Object> args) {
        String expression = (String) args.get("expression");
        
        if (expression == null || expression.trim().isEmpty()) {
            return "Error: No expression provided";
        }
        
        try {
            // Basic security check - only allow numbers, operators, and parentheses
            if (!expression.matches("[0-9+\\-*/().\\s]+")) {
                return "Error: Invalid characters in expression. Only numbers, +, -, *, /, (, ) are allowed.";
            }
            
            Object result = engine.eval(expression);
            return "Result: " + result;
            
        } catch (Exception e) {
            return "Error calculating expression: " + e.getMessage();
        }
    }
    
    @Override
    public String getName() {
        return "calculate";
    }
    
    @Override
    public String getDescription() {
        return "Calculate a mathematical expression";
    }
    
    @Override
    public Map<String, Object> getParameterSchema() {
        return Map.of(
            "type", "object",
            "properties", Map.of(
                "expression", Map.of(
                    "type", "string",
                    "description", "Mathematical expression to calculate (e.g., '2 + 3 * 4')"
                )
            ),
            "required", new String[]{"expression"}
        );
    }
}
