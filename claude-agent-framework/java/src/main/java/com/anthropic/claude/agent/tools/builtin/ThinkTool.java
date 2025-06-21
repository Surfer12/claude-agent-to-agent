package com.anthropic.claude.agent.tools.builtin;

import com.anthropic.claude.agent.core.ToolResult;
import com.anthropic.claude.agent.tools.BaseTool;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.HashMap;
import java.util.Map;

/**
 * Think tool for internal reasoning.
 */
public class ThinkTool extends BaseTool {
    
    public ThinkTool() {
        super(
            "think",
            "Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed.",
            createInputSchema()
        );
    }
    
    private static Map<String, Object> createInputSchema() {
        Map<String, Object> schema = new HashMap<>();
        schema.put("type", "object");
        
        Map<String, Object> properties = new HashMap<>();
        Map<String, Object> thoughtProperty = new HashMap<>();
        thoughtProperty.put("type", "string");
        thoughtProperty.put("description", "The thought or reasoning to record");
        properties.put("thought", thoughtProperty);
        
        schema.put("properties", properties);
        schema.put("required", new String[]{"thought"});
        
        return schema;
    }
    
    @Override
    public ToolResult execute(JsonNode input) throws Exception {
        if (!input.has("thought")) {
            return ToolResult.error("Missing required parameter: thought");
        }
        
        String thought = input.get("thought").asText();
        
        // Log the thought (in a real implementation, you might want to store this)
        System.out.println("[THINK] " + thought);
        
        return ToolResult.success("Thought recorded: " + thought);
    }
}
