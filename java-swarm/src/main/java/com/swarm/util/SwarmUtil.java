package com.swarm.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.swarm.types.AgentFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * Utility functions for the Swarm system.
 * Equivalent to the Python util.py module.
 */
public class SwarmUtil {
    private static final Logger logger = LoggerFactory.getLogger(SwarmUtil.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static final DateTimeFormatter TIMESTAMP_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    /**
     * Debug print function that prints timestamped messages when debug is enabled.
     * Equivalent to the Python debug_print function.
     */
    public static void debugPrint(boolean debug, Object... args) {
        if (!debug) {
            return;
        }
        
        String timestamp = LocalDateTime.now().format(TIMESTAMP_FORMAT);
        StringBuilder message = new StringBuilder();
        for (Object arg : args) {
            if (message.length() > 0) {
                message.append(" ");
            }
            message.append(arg != null ? arg.toString() : "null");
        }
        
        // ANSI color codes for formatting (similar to Python version)
        System.out.println("\033[97m[\033[90m" + timestamp + "\033[97m]\033[90m " + message + "\033[0m");
    }

    /**
     * Merge fields from source map into target map.
     * Handles nested maps recursively.
     */
    public static void mergeFields(Map<String, Object> target, Map<String, Object> source) {
        for (Map.Entry<String, Object> entry : source.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            
            if (value instanceof String && target.containsKey(key) && target.get(key) instanceof String) {
                target.put(key, target.get(key) + value);
            } else if (value != null && value instanceof Map && target.containsKey(key) && target.get(key) instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> targetMap = (Map<String, Object>) target.get(key);
                @SuppressWarnings("unchecked")
                Map<String, Object> sourceMap = (Map<String, Object>) value;
                mergeFields(targetMap, sourceMap);
            } else if (value != null) {
                target.put(key, value);
            }
        }
    }

    /**
     * Merge a chunk (delta) into the final response.
     * Equivalent to the Python merge_chunk function.
     */
    public static void mergeChunk(Map<String, Object> finalResponse, Map<String, Object> delta) {
        // Remove role from delta as it's handled separately
        delta.remove("role");
        mergeFields(finalResponse, delta);

        // Handle tool_calls specially
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> toolCalls = (List<Map<String, Object>>) delta.get("tool_calls");
        if (toolCalls != null && !toolCalls.isEmpty()) {
            Map<String, Object> toolCall = toolCalls.get(0);
            Integer index = (Integer) toolCall.remove("index");
            
            if (index != null) {
                @SuppressWarnings("unchecked")
                Map<Integer, Map<String, Object>> finalToolCalls = 
                    (Map<Integer, Map<String, Object>>) finalResponse.get("tool_calls");
                if (finalToolCalls == null) {
                    finalToolCalls = new HashMap<>();
                    finalResponse.put("tool_calls", finalToolCalls);
                }
                
                Map<String, Object> existingToolCall = finalToolCalls.get(index);
                if (existingToolCall == null) {
                    existingToolCall = new HashMap<>();
                    finalToolCalls.put(index, existingToolCall);
                }
                
                mergeFields(existingToolCall, toolCall);
            }
        }
    }

    /**
     * Convert an AgentFunction to JSON format for OpenAI API.
     * Equivalent to the Python function_to_json function.
     */
    public static Map<String, Object> functionToJson(AgentFunction func) {
        Map<String, Object> parameters = new HashMap<>();
        List<String> required = new ArrayList<>();

        // Get parameter schema from the function
        Map<String, Object> schema = func.getParameterSchema();
        
        Map<String, Object> function = new HashMap<>();
        function.put("name", func.getName());
        function.put("description", func.getDescription());
        function.put("parameters", schema);

        Map<String, Object> result = new HashMap<>();
        result.put("type", "function");
        result.put("function", function);

        return result;
    }

    /**
     * Convert a Java method to JSON format for OpenAI API.
     * This is a helper method for reflection-based function registration.
     */
    public static Map<String, Object> methodToJson(Method method) {
        Map<String, Object> parameters = new HashMap<>();
        Map<String, Object> properties = new HashMap<>();
        List<String> required = new ArrayList<>();

        // Type mapping from Java to JSON Schema
        Map<Class<?>, String> typeMap = Map.of(
            String.class, "string",
            Integer.class, "integer",
            int.class, "integer",
            Double.class, "number",
            double.class, "number",
            Float.class, "number",
            float.class, "number",
            Boolean.class, "boolean",
            boolean.class, "boolean",
            List.class, "array",
            Map.class, "object"
        );

        for (Parameter param : method.getParameters()) {
            String paramType = typeMap.getOrDefault(param.getType(), "string");
            properties.put(param.getName(), Map.of("type", paramType));
            
            // Assume all parameters are required for simplicity
            // In a real implementation, you might use annotations to mark optional parameters
            required.add(param.getName());
        }

        parameters.put("type", "object");
        parameters.put("properties", properties);
        parameters.put("required", required);

        Map<String, Object> function = new HashMap<>();
        function.put("name", method.getName());
        function.put("description", ""); // Could be extracted from JavaDoc
        function.put("parameters", parameters);

        Map<String, Object> result = new HashMap<>();
        result.put("type", "function");
        result.put("function", function);

        return result;
    }

    /**
     * Create a default context variables map with empty string defaults.
     * Equivalent to Python's defaultdict(str).
     */
    public static Map<String, Object> createDefaultContextVariables(Map<String, Object> contextVariables) {
        return new HashMap<String, Object>() {
            @Override
            public Object get(Object key) {
                Object value = super.get(key);
                return value != null ? value : "";
            }
        };
    }

    /**
     * Deep copy a map structure.
     */
    @SuppressWarnings("unchecked")
    public static <T> T deepCopy(T original) {
        try {
            String json = objectMapper.writeValueAsString(original);
            return (T) objectMapper.readValue(json, original.getClass());
        } catch (Exception e) {
            logger.warn("Failed to deep copy object, returning original", e);
            return original;
        }
    }

    /**
     * Convert an object to JSON string.
     */
    public static String toJson(Object obj) {
        try {
            return objectMapper.writeValueAsString(obj);
        } catch (Exception e) {
            logger.warn("Failed to convert object to JSON", e);
            return obj.toString();
        }
    }

    /**
     * Parse JSON string to Map.
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> fromJson(String json) {
        try {
            return objectMapper.readValue(json, Map.class);
        } catch (Exception e) {
            logger.warn("Failed to parse JSON", e);
            return new HashMap<>();
        }
    }
}
