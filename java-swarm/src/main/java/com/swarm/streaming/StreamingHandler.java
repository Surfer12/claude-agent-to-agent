package com.swarm.streaming;

import com.swarm.types.StreamingResponse;
import com.swarm.util.SwarmUtil;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Handles streaming responses and merges chunks into complete messages.
 * Equivalent to the Python streaming functionality.
 */
public class StreamingHandler {
    
    /**
     * Merge a streaming chunk into the final response message.
     * Equivalent to the Python merge_chunk function.
     */
    public static void mergeStreamingChunk(Map<String, Object> finalResponse, StreamingResponse.StreamingDelta delta) {
        if (delta == null) {
            return;
        }
        
        // Handle role
        if (delta.getRole() != null) {
            finalResponse.put("role", delta.getRole());
        }
        
        // Handle content
        if (delta.getContent() != null) {
            String existingContent = (String) finalResponse.getOrDefault("content", "");
            finalResponse.put("content", existingContent + delta.getContent());
        }
        
        // Handle function calls
        if (delta.getFunctionCall() != null) {
            @SuppressWarnings("unchecked")
            Map<String, Object> existingFunctionCall = (Map<String, Object>) finalResponse.get("function_call");
            if (existingFunctionCall == null) {
                existingFunctionCall = new HashMap<>();
                finalResponse.put("function_call", existingFunctionCall);
            }
            
            if (delta.getFunctionCall().getName() != null) {
                String existingName = (String) existingFunctionCall.getOrDefault("name", "");
                existingFunctionCall.put("name", existingName + delta.getFunctionCall().getName());
            }
            
            if (delta.getFunctionCall().getArguments() != null) {
                String existingArgs = (String) existingFunctionCall.getOrDefault("arguments", "");
                existingFunctionCall.put("arguments", existingArgs + delta.getFunctionCall().getArguments());
            }
        }
        
        // Handle tool calls
        if (delta.getToolCalls() != null && !delta.getToolCalls().isEmpty()) {
            @SuppressWarnings("unchecked")
            Map<Integer, Map<String, Object>> toolCallsMap = (Map<Integer, Map<String, Object>>) 
                finalResponse.computeIfAbsent("tool_calls", k -> new ConcurrentHashMap<>());
            
            for (StreamingResponse.StreamingToolCall toolCall : delta.getToolCalls()) {
                Integer index = toolCall.getIndex();
                if (index == null) continue;
                
                Map<String, Object> existingToolCall = toolCallsMap.computeIfAbsent(index, k -> new HashMap<>());
                
                if (toolCall.getId() != null) {
                    existingToolCall.put("id", toolCall.getId());
                }
                
                if (toolCall.getType() != null) {
                    existingToolCall.put("type", toolCall.getType());
                }
                
                if (toolCall.getFunction() != null) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> existingFunction = (Map<String, Object>) 
                        existingToolCall.computeIfAbsent("function", k -> new HashMap<>());
                    
                    if (toolCall.getFunction().getName() != null) {
                        String existingName = (String) existingFunction.getOrDefault("name", "");
                        existingFunction.put("name", existingName + toolCall.getFunction().getName());
                    }
                    
                    if (toolCall.getFunction().getArguments() != null) {
                        String existingArgs = (String) existingFunction.getOrDefault("arguments", "");
                        existingFunction.put("arguments", existingArgs + toolCall.getFunction().getArguments());
                    }
                }
            }
        }
    }
    
    /**
     * Convert the tool_calls map to a list format for consistency with OpenAI API.
     */
    public static void finalizeToolCalls(Map<String, Object> message) {
        @SuppressWarnings("unchecked")
        Map<Integer, Map<String, Object>> toolCallsMap = (Map<Integer, Map<String, Object>>) message.get("tool_calls");
        
        if (toolCallsMap != null && !toolCallsMap.isEmpty()) {
            // Convert map to list, preserving order by index
            java.util.List<Map<String, Object>> toolCallsList = new java.util.ArrayList<>();
            toolCallsMap.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .forEach(entry -> toolCallsList.add(entry.getValue()));
            
            message.put("tool_calls", toolCallsList);
        } else {
            message.remove("tool_calls");
        }
    }
    
    /**
     * Create a streaming event for the client.
     */
    public static Map<String, Object> createStreamingEvent(String type, Object data) {
        Map<String, Object> event = new HashMap<>();
        event.put("type", type);
        event.put("data", data);
        event.put("timestamp", System.currentTimeMillis());
        return event;
    }
    
    /**
     * Create a delimiter event (start/end markers).
     */
    public static Map<String, Object> createDelimiterEvent(String delimiter) {
        return createStreamingEvent("delimiter", Map.of("delim", delimiter));
    }
    
    /**
     * Create a delta event with streaming response data.
     */
    public static Map<String, Object> createDeltaEvent(StreamingResponse.StreamingDelta delta, String sender) {
        Map<String, Object> deltaData = new HashMap<>();
        
        if (delta.getRole() != null) {
            deltaData.put("role", delta.getRole());
        }
        
        if (delta.getContent() != null) {
            deltaData.put("content", delta.getContent());
        }
        
        if (delta.getFunctionCall() != null) {
            Map<String, Object> functionCall = new HashMap<>();
            if (delta.getFunctionCall().getName() != null) {
                functionCall.put("name", delta.getFunctionCall().getName());
            }
            if (delta.getFunctionCall().getArguments() != null) {
                functionCall.put("arguments", delta.getFunctionCall().getArguments());
            }
            deltaData.put("function_call", functionCall);
        }
        
        if (delta.getToolCalls() != null && !delta.getToolCalls().isEmpty()) {
            java.util.List<Map<String, Object>> toolCalls = new java.util.ArrayList<>();
            for (StreamingResponse.StreamingToolCall toolCall : delta.getToolCalls()) {
                Map<String, Object> toolCallData = new HashMap<>();
                if (toolCall.getIndex() != null) {
                    toolCallData.put("index", toolCall.getIndex());
                }
                if (toolCall.getId() != null) {
                    toolCallData.put("id", toolCall.getId());
                }
                if (toolCall.getType() != null) {
                    toolCallData.put("type", toolCall.getType());
                }
                if (toolCall.getFunction() != null) {
                    Map<String, Object> function = new HashMap<>();
                    if (toolCall.getFunction().getName() != null) {
                        function.put("name", toolCall.getFunction().getName());
                    }
                    if (toolCall.getFunction().getArguments() != null) {
                        function.put("arguments", toolCall.getFunction().getArguments());
                    }
                    toolCallData.put("function", function);
                }
                toolCalls.add(toolCallData);
            }
            deltaData.put("tool_calls", toolCalls);
        }
        
        if (sender != null) {
            deltaData.put("sender", sender);
        }
        
        return createStreamingEvent("delta", deltaData);
    }
    
    /**
     * Create a response event with the final response.
     */
    public static Map<String, Object> createResponseEvent(com.swarm.types.Response response) {
        return createStreamingEvent("response", Map.of(
            "messages", response.getMessages(),
            "agent", response.getAgent() != null ? response.getAgent().getName() : null,
            "context_variables", response.getContextVariables()
        ));
    }
    
    /**
     * Check if a streaming response indicates completion.
     */
    public static boolean isStreamComplete(StreamingResponse response) {
        if (response.getChoices() == null || response.getChoices().isEmpty()) {
            return false;
        }
        
        StreamingResponse.StreamingChoice choice = response.getChoices().get(0);
        return choice.getFinishReason() != null;
    }
    
    /**
     * Extract the finish reason from a streaming response.
     */
    public static String getFinishReason(StreamingResponse response) {
        if (response.getChoices() == null || response.getChoices().isEmpty()) {
            return null;
        }
        
        return response.getChoices().get(0).getFinishReason();
    }
}
