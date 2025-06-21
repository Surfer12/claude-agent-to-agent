package com.anthropic.api;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import kong.unirest.HttpResponse;
import kong.unirest.Unirest;

public final class AnthropicClient {

    private final String apiKey;
    private final String model;
    private final int maxTokens;
    private final Set<String> tools;
    private final Map<String, Object> toolChoice;
    private final ObjectMapper objectMapper = new ObjectMapper();

    private AnthropicClient(Builder builder) {
        this.apiKey = Objects.requireNonNull(
            builder.apiKey,
            "API key cannot be null"
        );
        this.model = Objects.requireNonNull(
            builder.model,
            "Model cannot be null"
        );
        this.maxTokens = builder.maxTokens;
        this.tools = Collections.unmodifiableSet(builder.tools);
        this.toolChoice = Collections.unmodifiableMap(builder.toolChoice);
    }

    public static class Builder {

        private String apiKey;
        private String model = "claude-opus-4-20250514";
        private int maxTokens = 1024;
        private Set<String> tools = Set.of();
        private Map<String, Object> toolChoice = Map.of();

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

        public Builder tools(Set<String> tools) {
            this.tools = Set.copyOf(tools);
            return this;
        }

        public Builder toolChoice(Map<String, Object> toolChoice) {
            this.toolChoice = Map.copyOf(toolChoice);
            return this;
        }

        public AnthropicClient build() {
            return new AnthropicClient(this);
        }
    }

    public Message createMessage(List<Message> messages) {
        Objects.requireNonNull(messages, "Messages cannot be null");

        Map<String, Object> requestBody = new java.util.HashMap<>();
        requestBody.put("model", model);
        requestBody.put("max_tokens", maxTokens);
        requestBody.put(
            "messages",
            messages
                .stream()
                .map(m ->
                    Map.of(
                        "role",
                        m.getRole(),
                        "content",
                        m
                            .getContent()
                            .stream()
                            .map(c ->
                                Map.of("type", c.getType(), "text", c.getText())
                            )
                            .collect(Collectors.toList())
                    )
                )
                .collect(Collectors.toList())
        );

        // Add tools if configured
        if (!tools.isEmpty()) {
            List<Map<String, Object>> toolsList = tools
                .stream()
                .map(toolName ->
                    Map.<String, Object>of(
                        "name",
                        toolName,
                        "description",
                        "Tool: " + toolName,
                        "input_schema",
                        Map.of(
                            "type",
                            "object",
                            "properties",
                            Map.of(),
                            "required",
                            List.of()
                        )
                    )
                )
                .collect(Collectors.toList());
            requestBody.put("tools", toolsList);
        }

        // Add tool choice if configured
        if (!toolChoice.isEmpty()) {
            requestBody.put("tool_choice", toolChoice);
        }

        HttpResponse<String> response = Unirest.post(
            "https://api.anthropic.com/v1/messages"
        )
            .header("x-api-key", apiKey)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .body(requestBody)
            .asString();

        if (!response.isSuccess()) {
            throw new RuntimeException(
                "API call failed: " +
                response.getStatus() +
                " - " +
                response.getBody()
            );
        }

        // Parse JSON response
        try {
            JsonNode jsonResponse = objectMapper.readTree(response.getBody());
            String id = jsonResponse.get("id").asText();
            String role = jsonResponse.get("role").asText();

            List<Content> responseContent = new java.util.ArrayList<>();
            JsonNode contentArray = jsonResponse.get("content");
            if (contentArray != null && contentArray.isArray()) {
                for (JsonNode contentNode : contentArray) {
                    responseContent.add(
                        new Content(
                            contentNode.get("type").asText(),
                            contentNode.get("text").asText()
                        )
                    );
                }
            }

            return new Message(id, role, responseContent);
        } catch (Exception e) {
            throw new RuntimeException(
                "Failed to parse response: " + e.getMessage(),
                e
            );
        }
    }

    public Set<String> getTools() {
        return tools;
    }

    public static final class Message {

        @JsonIgnore
        private final String id;

        private final String role;
        private final List<Content> content;

        public Message(String id, String role, List<Content> content) {
            this.id = id;
            this.role = role;
            this.content = List.copyOf(content);
        }

        public String getId() {
            return id;
        }

        public String getRole() {
            return role;
        }

        public List<Content> getContent() {
            return content;
        }
    }

    public static final class Content {

        private final String type;
        private final String text;

        public Content(String type, String text) {
            this.type = type;
            this.text = text;
        }

        public String getType() {
            return type;
        }

        public String getText() {
            return text;
        }
    }

    public record GeneratePromptRequest(
        @JsonProperty("target_model") String targetModel,
        @JsonProperty("task") String task
    ) {}

    public record ImprovePromptRequest(
        @JsonProperty("feedback") String feedback,
        @JsonProperty("messages") List<Message> messages,
        @JsonProperty("system") String system,
        @JsonProperty("target_model") String targetModel
    ) {}

    public record TemplatizePromptRequest(
        @JsonProperty("messages") List<Message> messages,
        @JsonProperty("system") String system
    ) {}

    public HttpResponse<String> generatePrompt(GeneratePromptRequest request) {
        Objects.requireNonNull(request, "Request cannot be null");
        try {
            return Unirest.post(
                "https://api.anthropic.com/v1/experimental/generate_prompt"
            )
                .header("x-api-key", apiKey)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .body(objectMapper.writeValueAsString(request))
                .asString();
        } catch (Exception e) {
            throw new RuntimeException(
                "Failed to serialize request: " + e.getMessage(),
                e
            );
        }
    }

    public HttpResponse<String> improvePrompt(ImprovePromptRequest request) {
        Objects.requireNonNull(request, "Request cannot be null");
        try {
            return Unirest.post(
                "https://api.anthropic.com/v1/experimental/improve_prompt"
            )
                .header("x-api-key", apiKey)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .body(objectMapper.writeValueAsString(request))
                .asString();
        } catch (Exception e) {
            throw new RuntimeException(
                "Failed to serialize request: " + e.getMessage(),
                e
            );
        }
    }

    public HttpResponse<String> templatizePrompt(
        TemplatizePromptRequest request
    ) {
        Objects.requireNonNull(request, "Request cannot be null");
        try {
            return Unirest.post(
                "https://api.anthropic.com/v1/experimental/templatize_prompt"
            )
                .header("x-api-key", apiKey)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .body(objectMapper.writeValueAsString(request))
                .asString();
        } catch (Exception e) {
            throw new RuntimeException(
                "Failed to serialize request: " + e.getMessage(),
                e
            );
        }
    }
}
