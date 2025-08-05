package com.swarm.cli;

import com.swarm.types.AgentFunction;

import java.util.Map;

/**
 * Simple echo function for demonstration purposes.
 */
public class EchoFunction implements AgentFunction {
    
    @Override
    public Object execute(Map<String, Object> args) {
        String message = (String) args.get("message");
        return "Echo: " + (message != null ? message : "");
    }
    
    @Override
    public String getName() {
        return "echo";
    }
    
    @Override
    public String getDescription() {
        return "Echo back the provided message";
    }
    
    @Override
    public Map<String, Object> getParameterSchema() {
        return Map.of(
            "type", "object",
            "properties", Map.of(
                "message", Map.of(
                    "type", "string",
                    "description", "The message to echo back"
                )
            ),
            "required", new String[]{"message"}
        );
    }
}
