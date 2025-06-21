package com.anthropic.claude.agent.core;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Configuration settings for Claude model parameters.
 */
public class ModelConfig {
    
    @JsonProperty("model")
    private String model = "claude-sonnet-4-20250514";
    
    @JsonProperty("max_tokens")
    private int maxTokens = 4096;
    
    @JsonProperty("temperature")
    private double temperature = 1.0;
    
    @JsonProperty("context_window_tokens")
    private int contextWindowTokens = 180000;
    
    // Default constructor
    public ModelConfig() {}
    
    // Constructor with parameters
    public ModelConfig(String model, int maxTokens, double temperature) {
        this.model = model;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
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
    
    public int getContextWindowTokens() {
        return contextWindowTokens;
    }
    
    public void setContextWindowTokens(int contextWindowTokens) {
        this.contextWindowTokens = contextWindowTokens;
    }
    
    @Override
    public String toString() {
        return "ModelConfig{" +
                "model='" + model + '\'' +
                ", maxTokens=" + maxTokens +
                ", temperature=" + temperature +
                ", contextWindowTokens=" + contextWindowTokens +
                '}';
    }
}
