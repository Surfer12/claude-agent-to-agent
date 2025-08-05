package com.swarm.types;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents a response from the Swarm system.
 * Equivalent to the Python Response class.
 */
public class Response {
    private List<Map<String, Object>> messages;
    private Agent agent;
    private Map<String, Object> contextVariables;

    public Response() {
        this.messages = new ArrayList<>();
        this.agent = null;
        this.contextVariables = new HashMap<>();
    }

    public Response(List<Map<String, Object>> messages, Agent agent, Map<String, Object> contextVariables) {
        this.messages = messages != null ? messages : new ArrayList<>();
        this.agent = agent;
        this.contextVariables = contextVariables != null ? contextVariables : new HashMap<>();
    }

    // Builder pattern
    public static class Builder {
        private List<Map<String, Object>> messages = new ArrayList<>();
        private Agent agent = null;
        private Map<String, Object> contextVariables = new HashMap<>();

        public Builder messages(List<Map<String, Object>> messages) {
            this.messages = messages;
            return this;
        }

        public Builder addMessage(Map<String, Object> message) {
            this.messages.add(message);
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

        public Response build() {
            return new Response(messages, agent, contextVariables);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    // Getters and setters
    public List<Map<String, Object>> getMessages() {
        return messages;
    }

    public void setMessages(List<Map<String, Object>> messages) {
        this.messages = messages;
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
        return "Response{" +
                "messages=" + messages.size() +
                ", agent=" + (agent != null ? agent.getName() : "null") +
                ", contextVariables=" + contextVariables.size() +
                '}';
    }
}
