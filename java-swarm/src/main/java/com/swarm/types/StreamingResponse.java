package com.swarm.types;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * Represents a streaming response chunk from OpenAI API.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class StreamingResponse {
    
    @JsonProperty("id")
    private String id;
    
    @JsonProperty("object")
    private String object;
    
    @JsonProperty("created")
    private Long created;
    
    @JsonProperty("model")
    private String model;
    
    @JsonProperty("choices")
    private List<StreamingChoice> choices;
    
    // Constructors
    public StreamingResponse() {}
    
    public StreamingResponse(String id, String object, Long created, String model, List<StreamingChoice> choices) {
        this.id = id;
        this.object = object;
        this.created = created;
        this.model = model;
        this.choices = choices;
    }
    
    // Getters and setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    
    public String getObject() { return object; }
    public void setObject(String object) { this.object = object; }
    
    public Long getCreated() { return created; }
    public void setCreated(Long created) { this.created = created; }
    
    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }
    
    public List<StreamingChoice> getChoices() { return choices; }
    public void setChoices(List<StreamingChoice> choices) { this.choices = choices; }
    
    @Override
    public String toString() {
        return "StreamingResponse{" +
                "id='" + id + '\'' +
                ", object='" + object + '\'' +
                ", created=" + created +
                ", model='" + model + '\'' +
                ", choices=" + choices +
                '}';
    }
    
    /**
     * Represents a choice in a streaming response.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class StreamingChoice {
        
        @JsonProperty("index")
        private Integer index;
        
        @JsonProperty("delta")
        private StreamingDelta delta;
        
        @JsonProperty("finish_reason")
        private String finishReason;
        
        // Constructors
        public StreamingChoice() {}
        
        public StreamingChoice(Integer index, StreamingDelta delta, String finishReason) {
            this.index = index;
            this.delta = delta;
            this.finishReason = finishReason;
        }
        
        // Getters and setters
        public Integer getIndex() { return index; }
        public void setIndex(Integer index) { this.index = index; }
        
        public StreamingDelta getDelta() { return delta; }
        public void setDelta(StreamingDelta delta) { this.delta = delta; }
        
        public String getFinishReason() { return finishReason; }
        public void setFinishReason(String finishReason) { this.finishReason = finishReason; }
        
        @Override
        public String toString() {
            return "StreamingChoice{" +
                    "index=" + index +
                    ", delta=" + delta +
                    ", finishReason='" + finishReason + '\'' +
                    '}';
        }
    }
    
    /**
     * Represents the delta (incremental change) in a streaming response.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class StreamingDelta {
        
        @JsonProperty("role")
        private String role;
        
        @JsonProperty("content")
        private String content;
        
        @JsonProperty("tool_calls")
        private List<StreamingToolCall> toolCalls;
        
        @JsonProperty("function_call")
        private StreamingFunctionCall functionCall;
        
        // Constructors
        public StreamingDelta() {}
        
        public StreamingDelta(String role, String content, List<StreamingToolCall> toolCalls, StreamingFunctionCall functionCall) {
            this.role = role;
            this.content = content;
            this.toolCalls = toolCalls;
            this.functionCall = functionCall;
        }
        
        // Getters and setters
        public String getRole() { return role; }
        public void setRole(String role) { this.role = role; }
        
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        
        public List<StreamingToolCall> getToolCalls() { return toolCalls; }
        public void setToolCalls(List<StreamingToolCall> toolCalls) { this.toolCalls = toolCalls; }
        
        public StreamingFunctionCall getFunctionCall() { return functionCall; }
        public void setFunctionCall(StreamingFunctionCall functionCall) { this.functionCall = functionCall; }
        
        @Override
        public String toString() {
            return "StreamingDelta{" +
                    "role='" + role + '\'' +
                    ", content='" + content + '\'' +
                    ", toolCalls=" + toolCalls +
                    ", functionCall=" + functionCall +
                    '}';
        }
    }
    
    /**
     * Represents a streaming tool call.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class StreamingToolCall {
        
        @JsonProperty("index")
        private Integer index;
        
        @JsonProperty("id")
        private String id;
        
        @JsonProperty("type")
        private String type;
        
        @JsonProperty("function")
        private StreamingFunctionCall function;
        
        // Constructors
        public StreamingToolCall() {}
        
        // Getters and setters
        public Integer getIndex() { return index; }
        public void setIndex(Integer index) { this.index = index; }
        
        public String getId() { return id; }
        public void setId(String id) { this.id = id; }
        
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        
        public StreamingFunctionCall getFunction() { return function; }
        public void setFunction(StreamingFunctionCall function) { this.function = function; }
        
        @Override
        public String toString() {
            return "StreamingToolCall{" +
                    "index=" + index +
                    ", id='" + id + '\'' +
                    ", type='" + type + '\'' +
                    ", function=" + function +
                    '}';
        }
    }
    
    /**
     * Represents a streaming function call.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class StreamingFunctionCall {
        
        @JsonProperty("name")
        private String name;
        
        @JsonProperty("arguments")
        private String arguments;
        
        // Constructors
        public StreamingFunctionCall() {}
        
        public StreamingFunctionCall(String name, String arguments) {
            this.name = name;
            this.arguments = arguments;
        }
        
        // Getters and setters
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        
        public String getArguments() { return arguments; }
        public void setArguments(String arguments) { this.arguments = arguments; }
        
        @Override
        public String toString() {
            return "StreamingFunctionCall{" +
                    "name='" + name + '\'' +
                    ", arguments='" + arguments + '\'' +
                    '}';
        }
    }
}
