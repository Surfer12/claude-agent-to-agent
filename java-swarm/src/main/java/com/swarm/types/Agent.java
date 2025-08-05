package com.swarm.types;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Represents an agent in the Swarm system.
 * Equivalent to the Python Agent class.
 */
public class Agent {
    private String name;
    private String model;
    private Object instructions; // Can be String or Function<Map<String, Object>, String>
    private List<AgentFunction> functions;
    private String toolChoice;
    private boolean parallelToolCalls;

    public Agent() {
        this.name = "Agent";
        this.model = "gpt-4o";
        this.instructions = "You are a helpful agent.";
        this.functions = new ArrayList<>();
        this.toolChoice = null;
        this.parallelToolCalls = true;
    }

    public Agent(String name, String model, Object instructions, 
                 List<AgentFunction> functions, String toolChoice, 
                 boolean parallelToolCalls) {
        this.name = name;
        this.model = model;
        this.instructions = instructions;
        this.functions = functions != null ? functions : new ArrayList<>();
        this.toolChoice = toolChoice;
        this.parallelToolCalls = parallelToolCalls;
    }

    // Builder pattern for easier construction
    public static class Builder {
        private String name = "Agent";
        private String model = "gpt-4o";
        private Object instructions = "You are a helpful agent.";
        private List<AgentFunction> functions = new ArrayList<>();
        private String toolChoice = null;
        private boolean parallelToolCalls = true;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder instructions(String instructions) {
            this.instructions = instructions;
            return this;
        }

        public Builder instructions(Function<Map<String, Object>, String> instructions) {
            this.instructions = instructions;
            return this;
        }

        public Builder functions(List<AgentFunction> functions) {
            this.functions = functions;
            return this;
        }

        public Builder addFunction(AgentFunction function) {
            this.functions.add(function);
            return this;
        }

        public Builder toolChoice(String toolChoice) {
            this.toolChoice = toolChoice;
            return this;
        }

        public Builder parallelToolCalls(boolean parallelToolCalls) {
            this.parallelToolCalls = parallelToolCalls;
            return this;
        }

        public Agent build() {
            return new Agent(name, model, instructions, functions, toolChoice, parallelToolCalls);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    // Getters and setters
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public Object getInstructions() {
        return instructions;
    }

    public void setInstructions(Object instructions) {
        this.instructions = instructions;
    }

    public List<AgentFunction> getFunctions() {
        return functions;
    }

    public void setFunctions(List<AgentFunction> functions) {
        this.functions = functions;
    }

    public String getToolChoice() {
        return toolChoice;
    }

    public void setToolChoice(String toolChoice) {
        this.toolChoice = toolChoice;
    }

    public boolean isParallelToolCalls() {
        return parallelToolCalls;
    }

    public void setParallelToolCalls(boolean parallelToolCalls) {
        this.parallelToolCalls = parallelToolCalls;
    }

    /**
     * Get instructions as string, handling both string and function cases
     */
    public String getInstructionsAsString(Map<String, Object> contextVariables) {
        if (instructions instanceof String) {
            return (String) instructions;
        } else if (instructions instanceof Function) {
            @SuppressWarnings("unchecked")
            Function<Map<String, Object>, String> func = (Function<Map<String, Object>, String>) instructions;
            return func.apply(contextVariables);
        }
        return "You are a helpful agent.";
    }

    @Override
    public String toString() {
        return "Agent{" +
                "name='" + name + '\'' +
                ", model='" + model + '\'' +
                ", functions=" + functions.size() +
                ", toolChoice='" + toolChoice + '\'' +
                ", parallelToolCalls=" + parallelToolCalls +
                '}';
    }
}
