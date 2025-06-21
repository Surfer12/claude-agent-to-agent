package com.anthropic.claude.agent.core;

/**
 * Result of a tool execution.
 */
public class ToolResult {
    
    private String toolUseId;
    private String content;
    private boolean isError;
    private String errorMessage;
    
    // Default constructor
    public ToolResult() {}
    
    // Constructor with content
    public ToolResult(String content) {
        this.content = content;
        this.isError = false;
    }
    
    // Constructor for error result
    public ToolResult(String content, boolean isError) {
        this.content = content;
        this.isError = isError;
    }
    
    // Static factory methods
    public static ToolResult success(String content) {
        return new ToolResult(content, false);
    }
    
    public static ToolResult error(String errorMessage) {
        ToolResult result = new ToolResult(errorMessage, true);
        result.errorMessage = errorMessage;
        return result;
    }
    
    // Getters and Setters
    public String getToolUseId() {
        return toolUseId;
    }
    
    public void setToolUseId(String toolUseId) {
        this.toolUseId = toolUseId;
    }
    
    public String getContent() {
        return content;
    }
    
    public void setContent(String content) {
        this.content = content;
    }
    
    public boolean isError() {
        return isError;
    }
    
    public void setIsError(boolean isError) {
        this.isError = isError;
    }
    
    public String getErrorMessage() {
        return errorMessage;
    }
    
    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }
    
    @Override
    public String toString() {
        return "ToolResult{" +
                "toolUseId='" + toolUseId + '\'' +
                ", content='" + content + '\'' +
                ", isError=" + isError +
                ", errorMessage='" + errorMessage + '\'' +
                '}';
    }
}
