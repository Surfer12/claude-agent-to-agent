package com.anthropic.claude.agent.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import okhttp3.*;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * HTTP client for interacting with the Anthropic API.
 */
public class AnthropicClient {
    
    private static final String BASE_URL = "https://api.anthropic.com/v1";
    private static final String ANTHROPIC_VERSION = "2023-06-01";
    
    private final OkHttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String apiKey;
    
    public AnthropicClient(String apiKey) {
        this.apiKey = apiKey;
        this.objectMapper = new ObjectMapper();
        this.httpClient = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .writeTimeout(60, TimeUnit.SECONDS)
                .build();
    }
    
    /**
     * Create a message using the Anthropic API.
     */
    public CompletableFuture<AgentResponse> createMessage(MessageRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return createMessageSync(request);
            } catch (IOException e) {
                throw new RuntimeException("Failed to create message", e);
            }
        });
    }
    
    /**
     * Create a message synchronously.
     */
    public AgentResponse createMessageSync(MessageRequest request) throws IOException {
        // Build request body
        ObjectNode requestBody = objectMapper.createObjectNode();
        requestBody.put("model", request.getModel());
        requestBody.put("max_tokens", request.getMaxTokens());
        requestBody.put("temperature", request.getTemperature());
        requestBody.put("system", request.getSystem());
        
        // Add messages
        ArrayNode messagesArray = objectMapper.createArrayNode();
        for (Message message : request.getMessages()) {
            ObjectNode messageNode = objectMapper.createObjectNode();
            messageNode.put("role", message.getRole());
            messageNode.put("content", message.getContent());
            messagesArray.add(messageNode);
        }
        requestBody.set("messages", messagesArray);
        
        // Add tools if present
        if (request.getTools() != null && !request.getTools().isEmpty()) {
            ArrayNode toolsArray = objectMapper.createArrayNode();
            for (Map<String, Object> tool : request.getTools()) {
                JsonNode toolNode = objectMapper.valueToTree(tool);
                toolsArray.add(toolNode);
            }
            requestBody.set("tools", toolsArray);
        }
        
        // Determine if we need beta client
        boolean useBeta = request.getBetas() != null && !request.getBetas().isEmpty();
        String endpoint = useBeta ? "/messages" : "/messages";
        
        // Build HTTP request
        Request.Builder httpRequestBuilder = new Request.Builder()
                .url(BASE_URL + endpoint)
                .addHeader("Content-Type", "application/json")
                .addHeader("x-api-key", apiKey)
                .addHeader("anthropic-version", ANTHROPIC_VERSION);
        
        // Add beta headers if needed
        if (useBeta) {
            String betaHeader = String.join(",", request.getBetas());
            httpRequestBuilder.addHeader("anthropic-beta", betaHeader);
        }
        
        RequestBody body = RequestBody.create(
                requestBody.toString(),
                MediaType.get("application/json")
        );
        httpRequestBuilder.post(body);
        
        Request httpRequest = httpRequestBuilder.build();
        
        // Execute request
        try (Response response = httpClient.newCall(httpRequest).execute()) {
            if (!response.isSuccessful()) {
                String errorBody = response.body() != null ? response.body().string() : "Unknown error";
                throw new IOException("API request failed: " + response.code() + " - " + errorBody);
            }
            
            String responseBody = response.body().string();
            JsonNode responseJson = objectMapper.readTree(responseBody);
            
            return parseResponse(responseJson);
        }
    }
    
    /**
     * Create a beta message (for tools that require beta headers).
     */
    public CompletableFuture<AgentResponse> createBetaMessage(MessageRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return createBetaMessageSync(request);
            } catch (IOException e) {
                throw new RuntimeException("Failed to create beta message", e);
            }
        });
    }
    
    /**
     * Create a beta message synchronously.
     */
    public AgentResponse createBetaMessageSync(MessageRequest request) throws IOException {
        // Beta messages use the same endpoint but with beta headers
        return createMessageSync(request);
    }
    
    /**
     * Parse the API response into an AgentResponse object.
     */
    private AgentResponse parseResponse(JsonNode responseJson) {
        AgentResponse.Builder builder = AgentResponse.builder();
        
        // Basic response fields
        if (responseJson.has("id")) {
            builder.id(responseJson.get("id").asText());
        }
        
        if (responseJson.has("model")) {
            builder.model(responseJson.get("model").asText());
        }
        
        if (responseJson.has("stop_reason")) {
            builder.stopReason(responseJson.get("stop_reason").asText());
        }
        
        // Parse content blocks
        if (responseJson.has("content")) {
            JsonNode contentArray = responseJson.get("content");
            for (JsonNode contentBlock : contentArray) {
                if (contentBlock.has("type")) {
                    String type = contentBlock.get("type").asText();
                    
                    if ("text".equals(type) && contentBlock.has("text")) {
                        builder.addTextContent(contentBlock.get("text").asText());
                    } else if ("tool_use".equals(type)) {
                        String toolName = contentBlock.get("name").asText();
                        String toolId = contentBlock.get("id").asText();
                        JsonNode toolInput = contentBlock.get("input");
                        builder.addToolUse(toolName, toolId, toolInput);
                    }
                }
            }
        }
        
        // Parse usage information
        if (responseJson.has("usage")) {
            JsonNode usage = responseJson.get("usage");
            if (usage.has("input_tokens")) {
                builder.inputTokens(usage.get("input_tokens").asInt());
            }
            if (usage.has("output_tokens")) {
                builder.outputTokens(usage.get("output_tokens").asInt());
            }
        }
        
        return builder.build();
    }
    
    /**
     * Close the HTTP client.
     */
    public void close() {
        httpClient.dispatcher().executorService().shutdown();
        httpClient.connectionPool().evictAll();
    }
}
