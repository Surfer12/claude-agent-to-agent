package com.anthropic.claude.agent.tools;

import com.anthropic.claude.agent.tools.beta.CodeExecutionTool;
import com.anthropic.claude.agent.tools.beta.ComputerUseTool;
import com.anthropic.claude.agent.tools.builtin.FileReadTool;
import com.anthropic.claude.agent.tools.builtin.FileWriteTool;
import com.anthropic.claude.agent.tools.builtin.ThinkTool;
import java.util.*;

/**
 * Registry for managing and discovering tools.
 */
public class ToolRegistry {

    private final Map<String, Class<? extends Tool>> toolClasses;
    private final Map<String, Tool> toolInstances;
    private boolean discovered = false;

    public ToolRegistry() {
        this.toolClasses = new HashMap<>();
        this.toolInstances = new HashMap<>();
    }

    /**
     * Register a tool class.
     */
    public void registerTool(String name, Class<? extends Tool> toolClass) {
        toolClasses.put(name, toolClass);
    }

    /**
     * Discover and register built-in tools.
     */
    public void discoverTools() {
        if (discovered) {
            return;
        }

        // Register built-in tools
        registerTool("think", ThinkTool.class);
        registerTool("file_read", FileReadTool.class);
        registerTool("file_write", FileWriteTool.class);

        // Register beta tools
        registerTool("computer", ComputerUseTool.class);
        registerTool("code_execution", CodeExecutionTool.class);

        discovered = true;
    }

    /**
     * Get a tool instance by name.
     */
    public Tool getTool(String name) throws Exception {
        return getTool(name, new HashMap<>());
    }

    /**
     * Get a tool instance by name with configuration.
     */
    public Tool getTool(String name, Map<String, Object> config)
        throws Exception {
        if (!discovered) {
            discoverTools();
        }

        if (!toolClasses.containsKey(name)) {
            throw new IllegalArgumentException(
                "Tool not found: " +
                name +
                ". Available tools: " +
                getAvailableTools()
            );
        }

        Class<? extends Tool> toolClass = toolClasses.get(name);

        // Try to create instance with configuration
        try {
            // First try constructor with config parameter
            try {
                return toolClass.getConstructor(Map.class).newInstance(config);
            } catch (NoSuchMethodException e) {
                // Fall back to default constructor
                return toolClass.getConstructor().newInstance();
            }
        } catch (Exception e) {
            throw new Exception(
                "Failed to create tool instance for " + name,
                e
            );
        }
    }

    /**
     * Get a cached tool instance.
     */
    public Tool getCachedTool(String name) throws Exception {
        return getCachedTool(name, new HashMap<>());
    }

    /**
     * Get a cached tool instance with configuration.
     */
    public Tool getCachedTool(String name, Map<String, Object> config)
        throws Exception {
        String cacheKey = name + "_" + config.hashCode();

        if (!toolInstances.containsKey(cacheKey)) {
            toolInstances.put(cacheKey, getTool(name, config));
        }

        return toolInstances.get(cacheKey);
    }

    /**
     * Get list of available tool names.
     */
    public List<String> getAvailableTools() {
        return new ArrayList<>(toolClasses.keySet());
    }

    /**
     * Get information about a tool.
     */
    public Map<String, Object> getToolInfo(String name) throws Exception {
        if (!toolClasses.containsKey(name)) {
            throw new IllegalArgumentException("Tool not found: " + name);
        }

        Tool tool = getTool(name);

        Map<String, Object> info = new HashMap<>();
        info.put("name", tool.getName());
        info.put("description", tool.getDescription());
        info.put("input_schema", tool.getInputSchema());
        info.put("class", tool.getClass().getSimpleName());
        info.put("supports_files", tool.supportsFiles());

        String toolVersion = tool.getToolVersion();
        if (toolVersion != null) {
            info.put("tool_version", toolVersion);
        }

        return info;
    }

    /**
     * Clear cached tool instances.
     */
    public void clearCache() {
        toolInstances.clear();
    }

    /**
     * Reset the registry.
     */
    public void reset() {
        toolClasses.clear();
        toolInstances.clear();
        discovered = false;
    }

    /**
     * Check if a tool is available.
     */
    public boolean isToolAvailable(String name) {
        return toolClasses.containsKey(name);
    }
}
