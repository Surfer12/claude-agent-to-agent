package com.anthropic.claude.agent.core;

import com.anthropic.claude.agent.tools.Tool;
import com.anthropic.claude.agent.tools.ToolRegistry;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.*;
import java.util.concurrent.CompletableFuture;

/**
 * Main Agent class for interacting with Claude API.
 */
public class Agent {

    private final AgentConfig config;
    private final AnthropicClient client;
    private final ToolRegistry toolRegistry;
    private final MessageHistory history;
    private final List<Tool> tools;

    /**
     * Constructor with configuration.
     */
    public Agent(AgentConfig config) {
        this(config, null, null);
    }

    /**
     * Constructor with configuration and tools.
     */
    public Agent(AgentConfig config, List<Tool> tools) {
        this(config, tools, null);
    }

    /**
     * Full constructor.
     */
    public Agent(AgentConfig config, List<Tool> tools, AnthropicClient client) {
        this.config = config;
        this.client = client != null
            ? client
            : new AnthropicClient(config.getApiKey());
        this.toolRegistry = new ToolRegistry();
        this.history = new MessageHistory(config.getModelConfig());
        this.tools = tools != null ? new ArrayList<>(tools) : new ArrayList<>();

        // Auto-discover tools if none provided
        if (this.tools.isEmpty()) {
            loadToolsFromConfig();
        }

        if (config.isVerbose()) {
            System.out.println("[" + config.getName() + "] Agent initialized");
        }
    }

    /**
     * Load tools based on configuration.
     */
    private void loadToolsFromConfig() {
        toolRegistry.discoverTools();

        List<String> enabledTools = config.getEnabledTools();
        if (enabledTools.contains("all")) {
            // Load all available tools
            for (String toolName : toolRegistry.getAvailableTools()) {
                try {
                    Map<String, Object> toolConfig = getToolConfig(toolName);
                    Tool tool = toolRegistry.getTool(toolName, toolConfig);
                    tools.add(tool);
                } catch (Exception e) {
                    if (config.isVerbose()) {
                        System.err.println(
                            "Warning: Could not load tool " +
                            toolName +
                            ": " +
                            e.getMessage()
                        );
                    }
                }
            }
        } else {
            // Load specific tools
            for (String toolName : enabledTools) {
                try {
                    Map<String, Object> toolConfig = getToolConfig(toolName);
                    Tool tool = toolRegistry.getTool(toolName, toolConfig);
                    tools.add(tool);
                } catch (Exception e) {
                    if (config.isVerbose()) {
                        System.err.println(
                            "Warning: Could not load tool " +
                            toolName +
                            ": " +
                            e.getMessage()
                        );
                    }
                }
            }
        }
    }

    /**
     * Get tool-specific configuration.
     */
    private Map<String, Object> getToolConfig(String toolName) {
        Map<String, Object> allToolConfig = config.getToolConfig();
        if (allToolConfig.containsKey(toolName)) {
            Object toolConfig = allToolConfig.get(toolName);
            if (toolConfig instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> configMap = (Map<
                        String,
                        Object
                    >) toolConfig;
                return configMap;
            }
        }
        return new HashMap<>();
    }

    /**
     * Send a message to Claude and get a response.
     */
    public CompletableFuture<AgentResponse> chat(String message) {
        return chatAsync(message);
    }

    /**
     * Send a message to Claude asynchronously.
     */
    public CompletableFuture<AgentResponse> chatAsync(String message) {
        if (config.isVerbose()) {
            System.out.println(
                "[" + config.getName() + "] Received: " + message
            );
        }

        // Add user message to history
        history.addMessage("user", message);

        return agentLoop();
    }

    /**
     * Send a message to Claude synchronously.
     */
    public AgentResponse chatSync(String message) {
        try {
            return chatAsync(message).get();
        } catch (Exception e) {
            throw new RuntimeException("Failed to get response", e);
        }
    }

    /**
     * Main agent loop that handles tool calls.
     */
    private CompletableFuture<AgentResponse> agentLoop() {
        return CompletableFuture.supplyAsync(() -> {
            try {
                while (true) {
                    // Prepare message request
                    MessageRequest request = prepareMessageRequest();

                    // Send request to API
                    AgentResponse response;
                    if (needsBetaClient()) {
                        response = client.createBetaMessageSync(request);
                    } else {
                        response = client.createMessageSync(request);
                    }

                    if (config.isVerbose()) {
                        System.out.println(
                            "[" +
                            config.getName() +
                            "] Response: " +
                            response.getTextContent()
                        );
                        if (response.hasToolUses()) {
                            System.out.println(
                                "[" +
                                config.getName() +
                                "] Tool calls: " +
                                response.getToolUses().size()
                            );
                        }
                    }

                    // Add assistant response to history
                    history.addAssistantResponse(response);

                    // Handle tool calls if present
                    if (response.hasToolUses()) {
                        List<ToolResult> toolResults = executeTools(
                            response.getToolUses()
                        );
                        history.addToolResults(toolResults);

                        if (config.isVerbose()) {
                            for (ToolResult result : toolResults) {
                                System.out.println(
                                    "[" +
                                    config.getName() +
                                    "] Tool result: " +
                                    result.getContent()
                                );
                            }
                        }
                    } else {
                        // No tool calls, return final response
                        return response;
                    }
                }
            } catch (Exception e) {
                throw new RuntimeException("Agent loop failed", e);
            }
        });
    }

    /**
     * Prepare message request for API call.
     */
    private MessageRequest prepareMessageRequest() {
        MessageRequest.Builder builder = MessageRequest.builder()
            .model(config.getModelConfig().getModel())
            .maxTokens(config.getModelConfig().getMaxTokens())
            .temperature(config.getModelConfig().getTemperature())
            .system(config.getSystemPrompt())
            .messages(history.getMessages());

        // Add tools
        for (Tool tool : tools) {
            builder.addTool(tool.toMap());
        }

        // Add beta headers if needed
        Set<String> betaHeaders = getBetaHeaders();
        for (String beta : betaHeaders) {
            builder.addBeta(beta);
        }

        return builder.build();
    }

    /**
     * Determine if beta client is needed.
     */
    private boolean needsBetaClient() {
        return !getBetaHeaders().isEmpty();
    }

    /**
     * Get required beta headers for enabled tools.
     */
    private Set<String> getBetaHeaders() {
        Set<String> betaHeaders = new HashSet<>();

        for (Tool tool : tools) {
            // Check for computer use tools
            if ("computer".equals(tool.getName())) {
                String model = config.getModelConfig().getModel().toLowerCase();
                String toolVersion = tool.getToolVersion();

                if (
                    model.contains("claude-4") ||
                    model.contains("claude-sonnet-3.7") ||
                    model.contains("claude-sonnet-4")
                ) {
                    if ("computer_20250124".equals(toolVersion)) {
                        betaHeaders.add("computer-use-2025-01-24");
                    } else {
                        betaHeaders.add("computer-use-2024-10-22");
                    }
                } else if (model.contains("claude-sonnet-3.5")) {
                    betaHeaders.add("computer-use-2024-10-22");
                } else {
                    betaHeaders.add("computer-use-2025-01-24");
                }
            }

            // Check for code execution tools
            if ("code_execution".equals(tool.getName())) {
                betaHeaders.add("code-execution-2025-05-22");

                // Check if tool supports files
                if (tool.supportsFiles()) {
                    betaHeaders.add("files-api-2025-04-14");
                }
            }
        }

        return betaHeaders;
    }

    /**
     * Execute tool calls and return results.
     */
    private List<ToolResult> executeTools(
        List<AgentResponse.ToolUseContentBlock> toolUses
    ) {
        List<ToolResult> results = new ArrayList<>();

        // Create tool map for quick lookup
        Map<String, Tool> toolMap = new HashMap<>();
        for (Tool tool : tools) {
            toolMap.put(tool.getName(), tool);
        }

        for (AgentResponse.ToolUseContentBlock toolUse : toolUses) {
            String toolName = toolUse.getName();
            String toolId = toolUse.getId();
            JsonNode input = toolUse.getInput();

            if (toolMap.containsKey(toolName)) {
                Tool tool = toolMap.get(toolName);
                try {
                    ToolResult result = tool.execute(input);
                    result.setToolUseId(toolId);
                    results.add(result);
                } catch (Exception e) {
                    ToolResult errorResult = new ToolResult();
                    errorResult.setToolUseId(toolId);
                    errorResult.setContent(
                        "Error executing tool: " + e.getMessage()
                    );
                    errorResult.setIsError(true);
                    results.add(errorResult);
                }
            } else {
                ToolResult errorResult = new ToolResult();
                errorResult.setToolUseId(toolId);
                errorResult.setContent("Tool not found: " + toolName);
                errorResult.setIsError(true);
                results.add(errorResult);
            }
        }

        return results;
    }

    /**
     * Clear conversation history.
     */
    public void clearHistory() {
        history.clear();
    }

    /**
     * Get conversation history.
     */
    public List<Message> getHistory() {
        return history.getMessages();
    }

    /**
     * Close the agent and cleanup resources.
     */
    public void close() {
        client.close();
    }

    // Getters
    public AgentConfig getConfig() {
        return config;
    }

    public List<Tool> getTools() {
        return new ArrayList<>(tools);
    }

    public List<String> getToolNames() {
        List<String> names = new ArrayList<>();
        for (Tool tool : tools) {
            names.add(tool.getName());
        }
        return names;
    }
}
