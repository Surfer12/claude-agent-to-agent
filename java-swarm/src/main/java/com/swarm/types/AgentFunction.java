package com.swarm.types;

import java.util.Map;

/**
 * Functional interface representing an agent function.
 * Equivalent to the Python AgentFunction type.
 */
@FunctionalInterface
public interface AgentFunction {
    /**
     * Execute the function with the given arguments.
     * 
     * @param args The function arguments
     * @return The result which can be a String, Agent, or Result object
     */
    Object execute(Map<String, Object> args);

    /**
     * Get the name of this function.
     * Default implementation uses the class name.
     */
    default String getName() {
        return this.getClass().getSimpleName();
    }

    /**
     * Get the description of this function.
     * Should be overridden to provide meaningful descriptions.
     */
    default String getDescription() {
        return "";
    }

    /**
     * Get the parameter schema for this function.
     * Should be overridden to define the expected parameters.
     */
    default Map<String, Object> getParameterSchema() {
        return Map.of(
            "type", "object",
            "properties", Map.of(),
            "required", new String[0]
        );
    }
}
