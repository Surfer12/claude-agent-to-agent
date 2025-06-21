package com.anthropic.claude.agent.tools;

import com.anthropic.claude.agent.core.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.Map;

/**
 * Base interface for all tools.
 */
public interface Tool {
    
    /**
     * Get the tool name.
     */
    String getName();
    
    /**
     * Get the tool description.
     */
    String getDescription();
    
    /**
     * Get the input schema for the tool.
     */
    Map<String, Object> getInputSchema();
    
    /**
     * Execute the tool with the given input.
     */
    ToolResult execute(JsonNode input) throws Exception;
    
    /**
     * Convert tool to map format for API.
     */
    Map<String, Object> toMap();
    
    /**
     * Get tool version (for beta tools).
     */
    default String getToolVersion() {
        return null;
    }
    
    /**
     * Check if tool supports files.
     */
    default boolean supportsFiles() {
        return false;
    }
    
    /**
     * Get tool type (for beta tools).
     */
    default String getToolType() {
        return "function";
    }
}
