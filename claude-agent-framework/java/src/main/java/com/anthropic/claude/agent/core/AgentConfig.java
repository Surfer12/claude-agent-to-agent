package com.anthropic.claude.agent.core;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Configuration class for Claude Agent.
 */
public class AgentConfig {

    // API Configuration
    @JsonProperty("api_key")
    private String apiKey;

    // Model Configuration
    @JsonProperty("model_config")
    private ModelConfig modelConfig = new ModelConfig();

    // Agent Configuration
    @JsonProperty("name")
    private String name = "claude-agent";

    @JsonProperty("system_prompt")
    private String systemPrompt =
        "You are Claude, an AI assistant. Be concise and helpful.";

    @JsonProperty("verbose")
    private boolean verbose = false;

    // Tool Configuration
    @JsonProperty("enabled_tools")
    private List<String> enabledTools = new ArrayList<>();

    @JsonProperty("tool_config")
    private Map<String, Object> toolConfig = new HashMap<>();

    // MCP Configuration
    @JsonProperty("mcp_servers")
    private List<String> mcpServers = new ArrayList<>();

    // Default constructor
    public AgentConfig() {
        this.enabledTools.add("all");

        // Set API key from environment if not provided
        if (this.apiKey == null) {
            this.apiKey = System.getenv("ANTHROPIC_API_KEY");
        }
    }

    // Builder pattern
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

        private AgentConfig config = new AgentConfig();

        public Builder apiKey(String apiKey) {
            config.apiKey = apiKey;
            return this;
        }

        public Builder name(String name) {
            config.name = name;
            return this;
        }

        public Builder systemPrompt(String systemPrompt) {
            config.systemPrompt = systemPrompt;
            return this;
        }

        public Builder verbose(boolean verbose) {
            config.verbose = verbose;
            return this;
        }

        public Builder model(String model) {
            config.modelConfig.setModel(model);
            return this;
        }

        public Builder maxTokens(int maxTokens) {
            config.modelConfig.setMaxTokens(maxTokens);
            return this;
        }

        public Builder temperature(double temperature) {
            config.modelConfig.setTemperature(temperature);
            return this;
        }

        public Builder enabledTools(List<String> tools) {
            config.enabledTools = new ArrayList<>(tools);
            return this;
        }

        public Builder addTool(String tool) {
            config.enabledTools.add(tool);
            return this;
        }

        public AgentConfig build() {
            return config;
        }
    }

    /**
     * Load configuration from YAML file.
     */
    public static AgentConfig fromFile(String configPath) throws IOException {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());

        // Read the YAML structure
        @SuppressWarnings("unchecked")
        Map<String, Object> data = mapper.readValue(
            new File(configPath),
            Map.class
        );

        AgentConfig config = new AgentConfig();

        // Extract agent config
        if (data.containsKey("agent")) {
            Object agentObj = data.get("agent");
            if (agentObj instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> agentData = (Map<String, Object>) agentObj;
                if (agentData.containsKey("name")) {
                    config.name = (String) agentData.get("name");
                }
                if (agentData.containsKey("system_prompt")) {
                    config.systemPrompt = (String) agentData.get(
                        "system_prompt"
                    );
                }
                if (agentData.containsKey("verbose")) {
                    config.verbose = (Boolean) agentData.get("verbose");
                }
            }
        }

        // Extract model config
        if (data.containsKey("model")) {
            Object modelObj = data.get("model");
            if (modelObj instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> modelData = (Map<String, Object>) modelObj;
                if (modelData.containsKey("model")) {
                    config.modelConfig.setModel(
                        (String) modelData.get("model")
                    );
                }
                if (modelData.containsKey("max_tokens")) {
                    config.modelConfig.setMaxTokens(
                        (Integer) modelData.get("max_tokens")
                    );
                }
                if (modelData.containsKey("temperature")) {
                    Object temp = modelData.get("temperature");
                    if (temp instanceof Double) {
                        config.modelConfig.setTemperature((Double) temp);
                    } else if (temp instanceof Integer) {
                        config.modelConfig.setTemperature(
                            ((Integer) temp).doubleValue()
                        );
                    }
                }
            }
        }

        // Extract tool config
        if (data.containsKey("tools")) {
            Object toolsObj = data.get("tools");
            if (toolsObj instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> toolData = (Map<String, Object>) toolsObj;
                if (toolData.containsKey("enabled")) {
                    Object enabledObj = toolData.get("enabled");
                    if (enabledObj instanceof List) {
                        @SuppressWarnings("unchecked")
                        List<String> enabledList = (List<String>) enabledObj;
                        config.enabledTools = enabledList;
                    }
                }
                // Extract tool-specific configurations
                for (Map.Entry<String, Object> entry : toolData.entrySet()) {
                    if (!"enabled".equals(entry.getKey())) {
                        config.toolConfig.put(entry.getKey(), entry.getValue());
                    }
                }
            }

            // Extract MCP config
            if (data.containsKey("mcp")) {
                Object mcpValue = data.get("mcp");
                if (mcpValue instanceof Map) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> mcpData = (Map<
                            String,
                            Object
                        >) mcpValue;
                    if (mcpData.containsKey("servers")) {
                        Object serversValue = mcpData.get("servers");
                        if (serversValue instanceof List) {
                            @SuppressWarnings("unchecked")
                            List<String> servers = (List<String>) serversValue;
                            config.mcpServers = servers;
                        }
                    }
                }
            }
        }

        return config;
    }

    /**
     * Save configuration to YAML file.
     */
    public void toFile(String configPath) throws IOException {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());

        Map<String, Object> data = new HashMap<>();

        // Agent section
        Map<String, Object> agentData = new HashMap<>();
        agentData.put("name", name);
        agentData.put("system_prompt", systemPrompt);
        agentData.put("verbose", verbose);
        data.put("agent", agentData);

        // Model section
        Map<String, Object> modelData = new HashMap<>();
        modelData.put("model", modelConfig.getModel());
        modelData.put("max_tokens", modelConfig.getMaxTokens());
        modelData.put("temperature", modelConfig.getTemperature());
        data.put("model", modelData);

        // Tools section
        Map<String, Object> toolData = new HashMap<>(toolConfig);
        toolData.put("enabled", enabledTools);
        data.put("tools", toolData);

        // MCP section
        Map<String, Object> mcpData = new HashMap<>();
        mcpData.put("servers", mcpServers);
        data.put("mcp", mcpData);

        mapper.writeValue(new File(configPath), data);
    }

    // Getters and Setters
    public String getApiKey() {
        return apiKey;
    }

    public void setApiKey(String apiKey) {
        this.apiKey = apiKey;
    }

    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    public void setModelConfig(ModelConfig modelConfig) {
        this.modelConfig = modelConfig;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getSystemPrompt() {
        return systemPrompt;
    }

    public void setSystemPrompt(String systemPrompt) {
        this.systemPrompt = systemPrompt;
    }

    public boolean isVerbose() {
        return verbose;
    }

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    public List<String> getEnabledTools() {
        return enabledTools;
    }

    public void setEnabledTools(List<String> enabledTools) {
        this.enabledTools = enabledTools;
    }

    public Map<String, Object> getToolConfig() {
        return toolConfig;
    }

    public void setToolConfig(Map<String, Object> toolConfig) {
        this.toolConfig = toolConfig;
    }

    public List<String> getMcpServers() {
        return mcpServers;
    }

    public void setMcpServers(List<String> mcpServers) {
        this.mcpServers = mcpServers;
    }
}
