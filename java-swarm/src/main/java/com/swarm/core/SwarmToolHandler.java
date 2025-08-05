package com.swarm.core;

import com.swarm.types.*;
import com.swarm.util.SwarmUtil;
import com.theokanning.openai.completion.chat.ChatCompletionChoice;
import com.theokanning.openai.completion.chat.ChatFunctionCall;
import com.theokanning.openai.completion.chat.ChatMessage;
import com.theokanning.openai.completion.chat.ChatMessageRole;

import java.util.*;

/**
 * Handles tool calls and function execution for the Swarm system.
 */
public class SwarmToolHandler {
    private static final String CTX_VARS_NAME = "context_variables";

    /**
     * Handle tool calls from the chat completion.
     * Equivalent to the Python handle_tool_calls method.
     */
    public static Response handleToolCalls(List<ChatFunctionCall> toolCalls, 
                                         List<AgentFunction> functions,
                                         Map<String, Object> contextVariables, 
                                         boolean debug) {
        
        Map<String, AgentFunction> functionMap = new HashMap<>();
        for (AgentFunction func : functions) {
            functionMap.put(func.getName(), func);
        }

        Response.Builder responseBuilder = Response.builder();

        for (ChatFunctionCall toolCall : toolCalls) {
            String name = toolCall.getName();
            
            // Handle missing tool case, skip to next tool
            if (!functionMap.containsKey(name)) {
                SwarmUtil.debugPrint(debug, "Tool " + name + " not found in function map.");
                
                Map<String, Object> errorMessage = new HashMap<>();
                errorMessage.put("role", "tool");
                errorMessage.put("tool_call_id", toolCall.getName()); // Using name as ID for simplicity
                errorMessage.put("tool_name", name);
                errorMessage.put("content", "Error: Tool " + name + " not found.");
                
                responseBuilder.addMessage(errorMessage);
                continue;
            }

            Map<String, Object> args = SwarmUtil.fromJson(toolCall.getArguments());
            SwarmUtil.debugPrint(debug, "Processing tool call: " + name + " with arguments " + args);

            AgentFunction func = functionMap.get(name);
            
            // Pass context_variables to agent functions if they expect it
            // Note: In Java, we can't easily inspect parameter names like Python's co_varnames
            // So we'll always pass context_variables and let the function ignore it if not needed
            args.put(CTX_VARS_NAME, contextVariables);
            
            Object rawResult = func.execute(args);
            Result result = handleFunctionResult(rawResult, debug);
            
            Map<String, Object> toolMessage = new HashMap<>();
            toolMessage.put("role", "tool");
            toolMessage.put("tool_call_id", toolCall.getName());
            toolMessage.put("tool_name", name);
            toolMessage.put("content", result.getValue());
            
            responseBuilder.addMessage(toolMessage);
            
            // Update context variables
            for (Map.Entry<String, Object> entry : result.getContextVariables().entrySet()) {
                responseBuilder.addContextVariable(entry.getKey(), entry.getValue());
            }
            
            if (result.getAgent() != null) {
                responseBuilder.agent(result.getAgent());
            }
        }

        return responseBuilder.build();
    }

    /**
     * Handle the result of a function call.
     * Equivalent to the Python handle_function_result method.
     */
    private static Result handleFunctionResult(Object result, boolean debug) {
        if (result instanceof Result) {
            return (Result) result;
        } else if (result instanceof Agent) {
            Agent agent = (Agent) result;
            return Result.builder()
                    .value(SwarmUtil.toJson(Map.of("assistant", agent.getName())))
                    .agent(agent)
                    .build();
        } else {
            try {
                return Result.builder()
                        .value(result != null ? result.toString() : "")
                        .build();
            } catch (Exception e) {
                String errorMessage = String.format(
                    "Failed to cast response to string: %s. Make sure agent functions return a string or Result object. Error: %s",
                    result, e.getMessage()
                );
                SwarmUtil.debugPrint(debug, errorMessage);
                throw new RuntimeException(errorMessage, e);
            }
        }
    }

    /**
     * Convert ChatMessage to Map format for consistency with Python implementation.
     */
    public static Map<String, Object> chatMessageToMap(ChatMessage message) {
        Map<String, Object> map = new HashMap<>();
        map.put("role", message.getRole());
        map.put("content", message.getContent());
        
        if (message.getFunctionCall() != null) {
            Map<String, Object> functionCall = new HashMap<>();
            functionCall.put("name", message.getFunctionCall().getName());
            functionCall.put("arguments", message.getFunctionCall().getArguments());
            map.put("function_call", functionCall);
        }
        
        return map;
    }

    /**
     * Convert Map to ChatMessage for OpenAI API calls.
     */
    public static ChatMessage mapToChatMessage(Map<String, Object> map) {
        String role = (String) map.get("role");
        String content = (String) map.get("content");
        
        ChatMessage message = new ChatMessage();
        message.setRole(role);
        message.setContent(content);
        
        @SuppressWarnings("unchecked")
        Map<String, Object> functionCall = (Map<String, Object>) map.get("function_call");
        if (functionCall != null) {
            ChatFunctionCall chatFunctionCall = new ChatFunctionCall();
            chatFunctionCall.setName((String) functionCall.get("name"));
            chatFunctionCall.setArguments((String) functionCall.get("arguments"));
            message.setFunctionCall(chatFunctionCall);
        }
        
        return message;
    }
}
