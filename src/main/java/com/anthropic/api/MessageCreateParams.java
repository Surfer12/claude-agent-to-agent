package com.anthropic.api;

import java.util.List;
import java.util.Map;

public class MessageCreateParams {
    private String model;
    private int maxTokens;
    private double temperature;
    private String system;
    private List<Map<String, Object>> messages;
    private List<Map<String, Object>> tools;
    private List<String> betas;

    private MessageCreateParams(Builder builder) {
        this.model = builder.model;
        this.maxTokens = builder.maxTokens;
        this.temperature = builder.temperature;
        this.system = builder.system;
        this.messages = builder.messages;
        this.tools = builder.tools;
        this.betas = builder.betas;
    }

    public static class Builder {
        private String model;
        private int maxTokens = 4096;
        private double temperature = 1.0;
        private String system;
        private List<Map<String, Object>> messages;
        private List<Map<String, Object>> tools;
        private List<String> betas;

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder maxTokens(int maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        public Builder temperature(double temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder system(String system) {
            this.system = system;
            return this;
        }

        public Builder messages(List<Map<String, Object>> messages) {
            this.messages = messages;
            return this;
        }

        public Builder tools(List<Map<String, Object>> tools) {
            this.tools = tools;
            return this;
        }

        public Builder betas(List<String> betas) {
            this.betas = betas;
            return this;
        }

        public MessageCreateParams build() {
            return new MessageCreateParams(this);
        }
    }

    public String getModel() { return model; }
    public int getMaxTokens() { return maxTokens; }
    public double getTemperature() { return temperature; }
    public String getSystem() { return system; }
    public List<Map<String, Object>> getMessages() { return messages; }
    public List<Map<String, Object>> getTools() { return tools; }
    public List<String> getBetas() { return betas; }
}
