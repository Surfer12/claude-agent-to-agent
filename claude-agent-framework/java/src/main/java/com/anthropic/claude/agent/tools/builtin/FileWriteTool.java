package com.anthropic.claude.agent.tools.builtin;

import com.anthropic.claude.agent.core.ToolResult;
import com.anthropic.claude.agent.tools.BaseTool;
import com.fasterxml.jackson.databind.JsonNode;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;

/**
 * File write tool for writing and editing files.
 */
public class FileWriteTool extends BaseTool {
    
    public FileWriteTool() {
        super(
            "file_write",
            "Write or edit files.\n\nOperations:\n- write: Create or completely replace a file\n- edit: Make targeted changes to parts of a file",
            createInputSchema()
        );
    }
    
    private static Map<String, Object> createInputSchema() {
        Map<String, Object> schema = new HashMap<>();
        schema.put("type", "object");
        
        Map<String, Object> properties = new HashMap<>();
        
        // Operation property
        Map<String, Object> operationProperty = new HashMap<>();
        operationProperty.put("type", "string");
        operationProperty.put("enum", new String[]{"write", "edit"});
        operationProperty.put("description", "Operation to perform: 'write' to create/replace a file, 'edit' to modify existing content");
        properties.put("operation", operationProperty);
        
        // Path property
        Map<String, Object> pathProperty = new HashMap<>();
        pathProperty.put("type", "string");
        pathProperty.put("description", "Path to the file");
        properties.put("path", pathProperty);
        
        // Content property
        Map<String, Object> contentProperty = new HashMap<>();
        contentProperty.put("type", "string");
        contentProperty.put("description", "Content to write to the file");
        properties.put("content", contentProperty);
        
        // Old content property (for edit operation)
        Map<String, Object> oldContentProperty = new HashMap<>();
        oldContentProperty.put("type", "string");
        oldContentProperty.put("description", "Old content to replace (for edit operation)");
        properties.put("old_content", oldContentProperty);
        
        schema.put("properties", properties);
        schema.put("required", new String[]{"operation", "path", "content"});
        
        return schema;
    }
    
    @Override
    public ToolResult execute(JsonNode input) throws Exception {
        if (!input.has("operation") || !input.has("path") || !input.has("content")) {
            return ToolResult.error("Missing required parameters: operation, path, and content");
        }
        
        String operation = input.get("operation").asText();
        String pathStr = input.get("path").asText();
        String content = input.get("content").asText();
        
        Path path = Paths.get(pathStr);
        
        try {
            switch (operation) {
                case "write":
                    return writeFile(path, content);
                case "edit":
                    String oldContent = input.has("old_content") ? input.get("old_content").asText() : null;
                    return editFile(path, content, oldContent);
                default:
                    return ToolResult.error("Unknown operation: " + operation);
            }
        } catch (IOException e) {
            return ToolResult.error("IO error: " + e.getMessage());
        } catch (SecurityException e) {
            return ToolResult.error("Access denied: " + e.getMessage());
        }
    }
    
    private ToolResult writeFile(Path path, String content) throws IOException {
        // Create parent directories if they don't exist
        Path parent = path.getParent();
        if (parent != null && !Files.exists(parent)) {
            Files.createDirectories(parent);
        }
        
        // Write content to file
        Files.writeString(path, content, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        
        return ToolResult.success("File written successfully: " + path + " (" + content.length() + " characters)");
    }
    
    private ToolResult editFile(Path path, String newContent, String oldContent) throws IOException {
        if (!Files.exists(path)) {
            return ToolResult.error("File does not exist: " + path);
        }
        
        if (!Files.isRegularFile(path)) {
            return ToolResult.error("Path is not a regular file: " + path);
        }
        
        // Read current content
        String currentContent = Files.readString(path);
        
        String updatedContent;
        if (oldContent != null && !oldContent.isEmpty()) {
            // Replace specific old content with new content
            if (!currentContent.contains(oldContent)) {
                return ToolResult.error("Old content not found in file");
            }
            updatedContent = currentContent.replace(oldContent, newContent);
        } else {
            // Append new content
            updatedContent = currentContent + "\n" + newContent;
        }
        
        // Write updated content
        Files.writeString(path, updatedContent, StandardOpenOption.TRUNCATE_EXISTING);
        
        return ToolResult.success("File edited successfully: " + path + " (" + updatedContent.length() + " characters)");
    }
}
