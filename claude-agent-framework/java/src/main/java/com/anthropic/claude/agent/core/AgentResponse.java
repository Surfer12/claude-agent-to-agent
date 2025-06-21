package com.anthropic.claude.agent.core;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.ArrayList;
import java.util.List;

/**
 * Response object from the Claude API.
 */
public class AgentResponse {
    
    private String id;
    private String model;
    private String stopReason;
    private List<ContentBlock> content;
    private int inputTokens;
    private int outputTokens;
    
    // Default constructor
    public AgentResponse() {
        this.content = new ArrayList<>();
    }
    
    // Builder pattern
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private AgentResponse response = new AgentResponse();
        
        public Builder id(String id) {
            response.id = id;
            return this;
        }
        
        public Builder model(String model) {
            response.model = model;
            return this;
        }
        
        public Builder stopReason(String stopReason) {
            response.stopReason = stopReason;
            return this;
        }
        
        public Builder addTextContent(String text) {
            response.content.add(new TextContentBlock(text));
            return this;
        }
        
        public Builder addToolUse(String toolName, String toolId, JsonNode input) {
            response.content.add(new ToolUseContentBlock(toolName, toolId, input));
            return this;
        }
        
        public Builder inputTokens(int inputTokens) {
            response.inputTokens = inputTokens;
            return this;
        }
        
        public Builder outputTokens(int outputTokens) {
            response.outputTokens = outputTokens;
            return this;
        }
        
        public AgentResponse build() {
            return response;
        }
    }
    
    /**
     * Get all text content from the response.
     */
    public String getTextContent() {
        StringBuilder sb = new StringBuilder();
        for (ContentBlock block : content) {
            if (block instanceof TextContentBlock) {
                if (sb.length() > 0) {
                    sb.append(" ");
                }
                sb.append(((TextContentBlock) block).getText());
            }
        }
        return sb.toString();
    }
    
    /**
     * Get all tool use blocks from the response.
     */
    public List<ToolUseContentBlock> getToolUses() {
        List<ToolUseContentBlock> toolUses = new ArrayList<>();
        for (ContentBlock block : content) {
            if (block instanceof ToolUseContentBlock) {
                toolUses.add((ToolUseContentBlock) block);
            }
        }
        return toolUses;
    }
    
    /**
     * Check if the response contains tool uses.
     */
    public boolean hasToolUses() {
        return !getToolUses().isEmpty();
    }
    
    // Getters and Setters
    public String getId() {
        return id;
    }
    
    public void setId(String id) {
        this.id = id;
    }
    
    public String getModel() {
        return model;
    }
    
    public void setModel(String model) {
        this.model = model;
    }
    
    public String getStopReason() {
        return stopReason;
    }
    
    public void setStopReason(String stopReason) {
        this.stopReason = stopReason;
    }
    
    public List<ContentBlock> getContent() {
        return content;
    }
    
    public void setContent(List<ContentBlock> content) {
        this.content = content;
    }
    
    public int getInputTokens() {
        return inputTokens;
    }
    
    public void setInputTokens(int inputTokens) {
        this.inputTokens = inputTokens;
    }
    
    public int getOutputTokens() {
        return outputTokens;
    }
    
    public void setOutputTokens(int outputTokens) {
        this.outputTokens = outputTokens;
    }
    
    @Override
    public String toString() {
        return "AgentResponse{" +
                "id='" + id + '\'' +
                ", model='" + model + '\'' +
                ", stopReason='" + stopReason + '\'' +
                ", content=" + content +
                ", inputTokens=" + inputTokens +
                ", outputTokens=" + outputTokens +
                '}';
    }
    
    // Content block classes
    public abstract static class ContentBlock {
        protected String type;
        
        public ContentBlock(String type) {
            this.type = type;
        }
        
        public String getType() {
            return type;
        }
    }
    
    public static class TextContentBlock extends ContentBlock {
        private String text;
        
        public TextContentBlock(String text) {
            super("text");
            this.text = text;
        }
        
        public String getText() {
            return text;
        }
        
        @Override
        public String toString() {
            return "TextContentBlock{text='" + text + "'}";
        }
    }
    
    public static class ToolUseContentBlock extends ContentBlock {
        private String name;
        private String id;
        private JsonNode input;
        
        public ToolUseContentBlock(String name, String id, JsonNode input) {
            super("tool_use");
            this.name = name;
            this.id = id;
            this.input = input;
        }
        
        public String getName() {
            return name;
        }
        
        public String getId() {
            return id;
        }
        
        public JsonNode getInput() {
            return input;
        }
        
        @Override
        public String toString() {
            return "ToolUseContentBlock{name='" + name + "', id='" + id + "', input=" + input + "}";
        }
    }
}
