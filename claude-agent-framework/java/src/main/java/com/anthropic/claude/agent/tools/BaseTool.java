package com.anthropic.claude.agent.tools;

import java.util.HashMap;
import java.util.Map;

/**
 * Abstract base class for tools.
 */
public abstract class BaseTool implements Tool {
    
    protected final String name;
    protected final String description;
    protected final Map<String, Object> inputSchema;
    
    public BaseTool(String name, String description, Map<String, Object> inputSchema) {
        this.name = name;
        this.description = description;
        this.inputSchema = inputSchema != null ? inputSchema : new HashMap<>();
    }
    
    @Override
    public String getName() {
        return name;
    }
    
    @Override
    public String getDescription() {
        return description;
    }
    
    @Override
    public Map<String, Object> getInputSchema() {
        return new HashMap<>(inputSchema);
    }
    
    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> toolMap = new HashMap<>();
        toolMap.put("type", getToolType());
        toolMap.put("name", getName());
        toolMap.put("description", getDescription());
        toolMap.put("input_schema", getInputSchema());
        return toolMap;
    }
    
    @Override
    public String toString() {
        return getClass().getSimpleName() + "{name='" + name + "'}";
    }
}
