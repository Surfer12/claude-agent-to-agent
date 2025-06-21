package com.anthropic.claude.agent.tools.builtin;

import com.anthropic.claude.agent.core.ToolResult;
import com.anthropic.claude.agent.tools.BaseTool;
import com.fasterxml.jackson.databind.JsonNode;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * File read tool for reading files and listing directories.
 */
public class FileReadTool extends BaseTool {
    
    public FileReadTool() {
        super(
            "file_read",
            "Read files or list directory contents.\n\nOperations:\n- read: Read the contents of a file\n- list: List files in a directory",
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
        operationProperty.put("enum", new String[]{"read", "list"});
        operationProperty.put("description", "Operation to perform: 'read' to read a file, 'list' to list directory contents");
        properties.put("operation", operationProperty);
        
        // Path property
        Map<String, Object> pathProperty = new HashMap<>();
        pathProperty.put("type", "string");
        pathProperty.put("description", "Path to the file or directory");
        properties.put("path", pathProperty);
        
        schema.put("properties", properties);
        schema.put("required", new String[]{"operation", "path"});
        
        return schema;
    }
    
    @Override
    public ToolResult execute(JsonNode input) throws Exception {
        if (!input.has("operation") || !input.has("path")) {
            return ToolResult.error("Missing required parameters: operation and path");
        }
        
        String operation = input.get("operation").asText();
        String pathStr = input.get("path").asText();
        
        Path path = Paths.get(pathStr);
        
        try {
            switch (operation) {
                case "read":
                    return readFile(path);
                case "list":
                    return listDirectory(path);
                default:
                    return ToolResult.error("Unknown operation: " + operation);
            }
        } catch (IOException e) {
            return ToolResult.error("IO error: " + e.getMessage());
        } catch (SecurityException e) {
            return ToolResult.error("Access denied: " + e.getMessage());
        }
    }
    
    private ToolResult readFile(Path path) throws IOException {
        if (!Files.exists(path)) {
            return ToolResult.error("File does not exist: " + path);
        }
        
        if (!Files.isRegularFile(path)) {
            return ToolResult.error("Path is not a regular file: " + path);
        }
        
        if (!Files.isReadable(path)) {
            return ToolResult.error("File is not readable: " + path);
        }
        
        // Check file size (limit to 1MB for safety)
        long size = Files.size(path);
        if (size > 1024 * 1024) {
            return ToolResult.error("File too large (>1MB): " + path);
        }
        
        String content = Files.readString(path);
        return ToolResult.success("File content:\n" + content);
    }
    
    private ToolResult listDirectory(Path path) throws IOException {
        if (!Files.exists(path)) {
            return ToolResult.error("Directory does not exist: " + path);
        }
        
        if (!Files.isDirectory(path)) {
            return ToolResult.error("Path is not a directory: " + path);
        }
        
        if (!Files.isReadable(path)) {
            return ToolResult.error("Directory is not readable: " + path);
        }
        
        String listing = Files.list(path)
                .map(p -> {
                    try {
                        String type = Files.isDirectory(p) ? "DIR" : "FILE";
                        long size = Files.isRegularFile(p) ? Files.size(p) : 0;
                        return String.format("%s  %8d  %s", type, size, p.getFileName());
                    } catch (IOException e) {
                        return "ERROR         " + p.getFileName();
                    }
                })
                .collect(Collectors.joining("\n"));
        
        return ToolResult.success("Directory listing:\n" + listing);
    }
}
