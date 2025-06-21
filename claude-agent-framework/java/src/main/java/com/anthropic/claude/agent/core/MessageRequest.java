package com.anthropic.claude.agent.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Request object for creating messages with the Anthropic API.
 */
public class MessageRequest {
    
    private String model;
    private int maxTokens;
    private double temperature;
    private String system;
    private List<Message> messages;
    private List<Map<String, Object>> tools;
    private List<String> betas;
    
    // Default constructor
    public MessageRequest() {
        this.messages = new ArrayList<>();
        this.tools = new ArrayList<>();
        this.betas = new ArrayList<>();
    }
    
    // Builder pattern
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private MessageRequest request = new MessageRequest();
        
        public Builder model(String model) {
            request.model = model;
            return this;
        }
        
        public Builder maxTokens(int maxTokens) {
            request.maxTokens = maxTokens;
            return this;
        }
        
        public Builder temperature(double temperature) {
            request.temperature = temperature;
            return this;
        }
        
        public Builder system(String system) {
            request.system = system;
            return this;
        }
        
        public Builder messages(List<Message> messages) {
            request.messages = new ArrayList<>(messages);
            return this;
        }
        
        public Builder addMessage(Message message) {
            request.messages.add(message);
            return this;
        }
        
        public Builder addMessage(String role, String content) {
            request.messages.add(new Message(role, content));
            return this;
        }
        
        public Builder tools(List<Map<String, Object>> tools) {
            request.tools = new ArrayList<>(tools);
            return this;
        }
        
        public Builder addTool(Map<String, Object> tool) {
            request.tools.add(tool);
            return this;
        }
        
        public Builder betas(List<String> betas) {
            request.betas = new ArrayList<>(betas);
            return this;
        }
        
        public Builder addBeta(String beta) {
            request.betas.add(beta);
            return this;
        }
        
        public MessageRequest build() {
            return request;
        }
    }
    
    // Getters and Setters
    public String getModel() {
        return model;
    }
    
    public void setModel(String model) {
        this.model = model;
    }
    
    public int getMaxTokens() {
        return maxTokens;
    }
    
    public void setMaxTokens(int maxTokens) {
        this.maxTokens = maxTokens;
    }
    
    public double getTemperature() {
        return temperature;
    }
    
    public void setTemperature(double temperature) {
        this.temperature = temperature;
    }
    
    public String getSystem() {
        return system;
    }
    
    public void setSystem(String system) {
        this.system = system;
    }
    
    public List<Message> getMessages() {
        return messages;
    }
    
    public void setMessages(List<Message> messages) {
        this.messages = messages;
    }
    
    public List<Map<String, Object>> getTools() {
        return tools;
    }
    
    public void setTools(List<Map<String, Object>> tools) {
        this.tools = tools;
    }
    
    public List<String> getBetas() {
        return betas;
    }
    
    public void setBetas(List<String> betas) {
        this.betas = betas;
    }
}
