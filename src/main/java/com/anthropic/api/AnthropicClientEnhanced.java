package com.anthropic.api;

import java.util.*;
import java.util.stream.Collectors;
import kong.unirest.HttpResponse;
import kong.unirest.Unirest;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;

/**
 * Enhanced Anthropic API Client for Java
 * 
 * This client provides comprehensive functionality for interacting with Anthropic's Claude API,
 * including support for various tools, streaming, and advanced features.
 */
public final class AnthropicClientEnhanced {
    private final String apiKey;
    private final String model;
    private final int maxTokens;
    private final Map<String, ToolConfig> toolConfigs;
    private final ObjectMapper objectMapper;

    private AnthropicClientEnhanced(Builder builder) {
        this.apiKey = Objects.requireNonNull(builder.apiKey, "API key cannot be null");
        this.model = Objects.requireNonNull(builder.model, "Model cannot be null");
        this.maxTokens = builder.maxTokens;
        this.toolConfigs = Collections.unmodifiableMap(builder.toolConfigs);
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Builder for creating AnthropicClientEnhanced instances
     */
    public static class Builder {
        private String apiKey;
        private String model = "claude-opus-4-20250514";
        private int maxTokens = 1024;
        private Map<String, ToolConfig> toolConfigs = new HashMap<>();

        public Builder() {
            // Initialize default tool configurations
            toolConfigs.put("bash", new ToolConfig("bash", "bash_20250124", "Execute bash commands in a secure environment"));
            toolConfigs.put("web_search", new ToolConfig("web_search", "web_search_20250305", "Search the web for current information", 5));
            toolConfigs.put("weather", new ToolConfig("get_weather", "weather_tool", "Get current weather information for a location"));
            toolConfigs.put("text_editor", new ToolConfig("str_replace_based_edit_tool", "text_editor_20250429", "Edit text files with string replacement operations"));
            toolConfigs.put("code_execution", new ToolConfig("code_execution", "code_execution_20250522", "Execute code in a secure environment"));
            toolConfigs.put("computer", new ToolConfig("computer", "computer_20250124", "Interact with computer interface"));
        }

        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder maxTokens(int maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        public Builder addToolConfig(String name, ToolConfig config) {
            this.toolConfigs.put(name, config);
            return this;
        }

        public AnthropicClientEnhanced build() {
            return new AnthropicClientEnhanced(this);
        }
    }

    /**
     * Create a message using the Anthropic API
     */
    public Message createMessage(List<Message> messages, List<String> tools, List<String> betas) {
        Objects.requireNonNull(messages, "Messages cannot be null");
        
        // Convert tool names to tool configurations
        List<Map<String, Object>> toolConfigs = new ArrayList<>();
        if (tools != null) {
            for (String toolName : tools) {
                ToolConfig config = this.toolConfigs.get(toolName);
                if (config != null) {
                    Map<String, Object> toolConfig = new HashMap<>();
                    toolConfig.put("type", config.getToolType());
                    toolConfig.put("name", config.getName());
                    toolConfig.put("description", config.getDescription());
                    
                    if (config.getInputSchema() != null) {
                        toolConfig.put("input_schema", config.getInputSchema());
                    }
                    if (config.getMaxUses() != null) {
                        toolConfig.put("max_uses", config.getMaxUses());
                    }
                    if (config.getDisplayConfig() != null) {
                        toolConfig.putAll(config.getDisplayConfig());
                    }
                    
                    toolConfigs.add(toolConfig);
                }
            }
        }

        // Prepare request parameters
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", model);
        requestBody.put("max_tokens", maxTokens);
        requestBody.put("messages", messages);
        
        if (!toolConfigs.isEmpty()) {
            requestBody.put("tools", toolConfigs);
        }
        if (betas != null && !betas.isEmpty()) {
            requestBody.put("betas", betas);
        }

        try {
            HttpResponse<String> response = Unirest.post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", apiKey)
                .header("Content-Type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(objectMapper.writeValueAsString(requestBody))
                .asString();

            if (response.isSuccess()) {
                JsonNode responseJson = objectMapper.readTree(response.getBody());
                return parseMessageResponse(responseJson);
            } else {
                throw new RuntimeException("API request failed: " + response.getStatus() + " - " + response.getBody());
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to create message", e);
        }
    }

    /**
     * Create a streaming message using the Anthropic API
     */
    public StreamingResponse createStreamingMessage(List<Message> messages, List<String> tools, List<String> betas) {
        Objects.requireNonNull(messages, "Messages cannot be null");
        
        // Convert tool names to tool configurations
        List<Map<String, Object>> toolConfigs = new ArrayList<>();
        if (tools != null) {
            for (String toolName : tools) {
                ToolConfig config = this.toolConfigs.get(toolName);
                if (config != null) {
                    Map<String, Object> toolConfig = new HashMap<>();
                    toolConfig.put("type", config.getToolType());
                    toolConfig.put("name", config.getName());
                    toolConfig.put("description", config.getDescription());
                    
                    if (config.getInputSchema() != null) {
                        toolConfig.put("input_schema", config.getInputSchema());
                    }
                    if (config.getMaxUses() != null) {
                        toolConfig.put("max_uses", config.getMaxUses());
                    }
                    if (config.getDisplayConfig() != null) {
                        toolConfig.putAll(config.getDisplayConfig());
                    }
                    
                    toolConfigs.add(toolConfig);
                }
            }
        }

        // Prepare request parameters
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", model);
        requestBody.put("max_tokens", maxTokens);
        requestBody.put("messages", messages);
        requestBody.put("stream", true);
        
        if (!toolConfigs.isEmpty()) {
            requestBody.put("tools", toolConfigs);
        }
        if (betas != null && !betas.isEmpty()) {
            requestBody.put("betas", betas);
        }

        try {
            HttpResponse<String> response = Unirest.post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", apiKey)
                .header("Content-Type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(objectMapper.writeValueAsString(requestBody))
                .asString();

            if (response.isSuccess()) {
                return new StreamingResponse(response.getBody());
            } else {
                throw new RuntimeException("API request failed: " + response.getStatus() + " - " + response.getBody());
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to create streaming message", e);
        }
    }

    /**
     * Get list of available tool names
     */
    public List<String> getAvailableTools() {
        return new ArrayList<>(toolConfigs.keySet());
    }

    /**
     * Get configuration for a specific tool
     */
    public Optional<ToolConfig> getToolConfig(String toolName) {
        return Optional.ofNullable(toolConfigs.get(toolName));
    }

    private Message parseMessageResponse(JsonNode responseJson) {
        String id = responseJson.get("id").asText();
        String role = responseJson.get("role").asText();
        
        List<Content> content = new ArrayList<>();
        JsonNode contentArray = responseJson.get("content");
        if (contentArray != null && contentArray.isArray()) {
            for (JsonNode contentNode : contentArray) {
                String type = contentNode.get("type").asText();
                String text = contentNode.has("text") ? contentNode.get("text").asText() : "";
                content.add(new Content(type, text));
            }
        }
        
        return new Message(id, role, content);
    }

    /**
     * Tool configuration class
     */
    public static final class ToolConfig {
        private final String name;
        private final String toolType;
        private final String description;
        private final Map<String, Object> inputSchema;
        private final Integer maxUses;
        private final Map<String, Object> displayConfig;

        public ToolConfig(String name, String toolType, String description) {
            this(name, toolType, description, null, null, null);
        }

        public ToolConfig(String name, String toolType, String description, Integer maxUses) {
            this(name, toolType, description, null, maxUses, null);
        }

        public ToolConfig(String name, String toolType, String description, 
                         Map<String, Object> inputSchema, Integer maxUses, 
                         Map<String, Object> displayConfig) {
            this.name = name;
            this.toolType = toolType;
            this.description = description;
            this.inputSchema = inputSchema;
            this.maxUses = maxUses;
            this.displayConfig = displayConfig;
        }

        public String getName() { return name; }
        public String getToolType() { return toolType; }
        public String getDescription() { return description; }
        public Map<String, Object> getInputSchema() { return inputSchema; }
        public Integer getMaxUses() { return maxUses; }
        public Map<String, Object> getDisplayConfig() { return displayConfig; }
    }

    /**
     * Message class
     */
    public static final class Message {
        private final String id;
        private final String role;
        private final List<Content> content;

        public Message(String id, String role, List<Content> content) {
            this.id = id;
            this.role = role;
            this.content = List.copyOf(content);
        }

        public String getId() { return id; }
        public String getRole() { return role; }
        public List<Content> getContent() { return content; }
    }

    /**
     * Content class
     */
    public static final class Content {
        private final String type;
        private final String text;

        public Content(String type, String text) {
            this.type = type;
            this.text = text;
        }

        public String getType() { return type; }
        public String getText() { return text; }
    }

    /**
     * Streaming response wrapper
     */
    public static final class StreamingResponse {
        private final String responseBody;
        private final StringBuilder accumulatedText = new StringBuilder();

        public StreamingResponse(String responseBody) {
            this.responseBody = responseBody;
        }

        public String getResponseBody() { return responseBody; }
        public String getAccumulatedText() { return accumulatedText.toString(); }
        
        public void appendText(String text) {
            accumulatedText.append(text);
        }
    }

    // Convenience methods
    public static AnthropicClientEnhanced createBasicClient(String apiKey) {
        return new Builder().apiKey(apiKey).build();
    }

    public static AnthropicClientEnhanced createToolEnabledClient(String apiKey, List<String> tools) {
        Builder builder = new Builder().apiKey(apiKey);
        // Validate tools
        Map<String, ToolConfig> availableTools = new HashMap<>();
        availableTools.put("bash", new ToolConfig("bash", "bash_20250124", "Execute bash commands in a secure environment"));
        availableTools.put("web_search", new ToolConfig("web_search", "web_search_20250305", "Search the web for current information", 5));
        availableTools.put("weather", new ToolConfig("get_weather", "weather_tool", "Get current weather information for a location"));
        availableTools.put("text_editor", new ToolConfig("str_replace_based_edit_tool", "text_editor_20250429", "Edit text files with string replacement operations"));
        availableTools.put("code_execution", new ToolConfig("code_execution", "code_execution_20250522", "Execute code in a secure environment"));
        availableTools.put("computer", new ToolConfig("computer", "computer_20250124", "Interact with computer interface"));
        
        for (String tool : tools) {
            if (!availableTools.containsKey(tool)) {
                throw new IllegalArgumentException("Unknown tool: " + tool);
            }
        }
        
        return builder.build();
    }
} 