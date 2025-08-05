package com.swarm.types;

import java.util.HashMap;
import java.util.Map;

/**
 * Encapsulates the possible return values for an agent function.
 * Equivalent to the Python Result class.
 */
public class Result {
    private String value;
    private Agent agent;
    private Map<String, Object> contextVariables;

    public Result() {
        this.value = "";
        this.agent = null;
        this.contextVariables = new HashMap<>();
    }

    public Result(String value) {
        this.value = value;
        this.agent = null;
        this.contextVariables = new HashMap<>();
    }

    public Result(String value, Agent agent, Map<String, Object> contextVariables) {
        this.value = value != null ? value : "";
        this.agent = agent;
        this.contextVariables = contextVariables != null ? contextVariables : new HashMap<>();
    }

    // Builder pattern
    public static class Builder {
        private String value = "";
        private Agent agent = null;
        private Map<String, Object> contextVariables = new HashMap<>();

        public Builder value(String value) {
            this.value = value;
            return this;
        }

        public Builder agent(Agent agent) {
            this.agent = agent;
            return this;
        }

        public Builder contextVariables(Map<String, Object> contextVariables) {
            this.contextVariables = contextVariables;
            return this;
        }

        public Builder addContextVariable(String key, Object value) {
            this.contextVariables.put(key, value);
            return this;
        }

        public Result build() {
            return new Result(value, agent, contextVariables);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    // Getters and setters
    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public Agent getAgent() {
        return agent;
    }

    public void setAgent(Agent agent) {
        this.agent = agent;
    }

    public Map<String, Object> getContextVariables() {
        return contextVariables;
    }

    public void setContextVariables(Map<String, Object> contextVariables) {
        this.contextVariables = contextVariables;
    }

    @Override
    public String toString() {
        return "Result{" +
                "value='" + value + '\'' +
                ", agent=" + (agent != null ? agent.getName() : "null") +
                ", contextVariables=" + contextVariables.size() +
                '}';
    }
}
