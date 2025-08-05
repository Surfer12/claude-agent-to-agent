package com.swarm.core;

import com.swarm.types.*;
import com.swarm.util.SwarmUtil;
import com.theokanning.openai.OpenAiService;
import com.theokanning.openai.completion.chat.*;
import com.theokanning.openai.service.FunctionExecutor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Main Swarm class that orchestrates multi-agent conversations.
 * Equivalent to the Python Swarm class.
 */
public class Swarm {
    private static final Logger logger = LoggerFactory.getLogger(Swarm.class);
    private static final String CTX_VARS_NAME = "context_variables";
    
    private final OpenAiService client;

    public Swarm() {
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.trim().isEmpty()) {
            throw new IllegalStateException("OPENAI_API_KEY environment variable is required");
        }
        this.client = new OpenAiService(apiKey);
    }

    public Swarm(OpenAiService client) {
        this.client = Objects.requireNonNull(client, "OpenAI client cannot be null");
    }

    public Swarm(String apiKey) {
        this.client = new OpenAiService(apiKey);
    }

    /**
     * Get chat completion from OpenAI API.
     * Equivalent to the Python get_chat_completion method.
     */
    public ChatCompletionResult getChatCompletion(Agent agent, List<ChatMessage> history, 
                                                  Map<String, Object> contextVariables,
                                                  String modelOverride, boolean stream, boolean debug) {
        
        Map<String, Object> defaultContextVars = SwarmUtil.createDefaultContextVariables(contextVariables);
        String instructions = agent.getInstructionsAsString(defaultContextVars);
        
        List<ChatMessage> messages = new ArrayList<>();
        messages.add(new ChatMessage(ChatMessageRole.SYSTEM.value(), instructions));
        messages.addAll(history);
        
        SwarmUtil.debugPrint(debug, "Getting chat completion for...:", messages);

        // Convert agent functions to OpenAI tools format
        List<ChatTool> tools = agent.getFunctions().stream()
                .map(func -> {
                    Map<String, Object> toolDef = SwarmUtil.functionToJson(func);
                    // Hide context_variables from model
                    @SuppressWarnings("unchecked")
                    Map<String, Object> function = (Map<String, Object>) toolDef.get("function");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> parameters = (Map<String, Object>) function.get("parameters");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> properties = (Map<String, Object>) parameters.get("properties");
                    @SuppressWarnings("unchecked")
                    List<String> required = (List<String>) parameters.get("required");
                    
                    if (properties != null) {
                        properties.remove(CTX_VARS_NAME);
                    }
                    if (required != null) {
                        required.remove(CTX_VARS_NAME);
                    }
                    
                    return ChatTool.builder()
                            .type(ChatTool.Type.FUNCTION)
                            .function(ChatFunction.builder()
                                    .name((String) function.get("name"))
                                    .description((String) function.get("description"))
                                    .parameters(parameters)
                                    .build())
                            .build();
                })
                .collect(Collectors.toList());

        ChatCompletionRequest.Builder requestBuilder = ChatCompletionRequest.builder()
                .model(modelOverride != null ? modelOverride : agent.getModel())
                .messages(messages)
                .stream(stream);

        if (!tools.isEmpty()) {
            requestBuilder.tools(tools);
            if (agent.getToolChoice() != null) {
                requestBuilder.toolChoice(agent.getToolChoice());
            }
            requestBuilder.parallelToolCalls(agent.isParallelToolCalls());
        }

        return client.createChatCompletion(requestBuilder.build());
    }

    /**
     * Handle the result of a function call.
     * Equivalent to the Python handle_function_result method.
     */
    public Result handleFunctionResult(Object result, boolean debug) {
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
     * Main run method for the Swarm system.
     * Equivalent to the Python run method.
     */
    public Response run(Agent agent, List<Map<String, Object>> messages) {
        return run(agent, messages, new HashMap<>(), null, false, false, Integer.MAX_VALUE, true);
    }

    public Response run(Agent agent, List<Map<String, Object>> messages, 
                       Map<String, Object> contextVariables, String modelOverride,
                       boolean stream, boolean debug, int maxTurns, boolean executeTools) {
        
        Agent activeAgent = agent;
        Map<String, Object> currentContextVariables = SwarmUtil.deepCopy(contextVariables);
        List<Map<String, Object>> history = SwarmUtil.deepCopy(messages);
        int initLen = messages.size();

        while (history.size() - initLen < maxTurns && activeAgent != null) {
            
            // Convert history to ChatMessage format
            List<ChatMessage> chatMessages = history.stream()
                    .map(SwarmToolHandler::mapToChatMessage)
                    .collect(Collectors.toList());

            // Get completion with current history and agent
            ChatCompletionResult completion = getChatCompletion(
                    activeAgent, chatMessages, currentContextVariables, 
                    modelOverride, stream, debug);

            ChatMessage message = completion.getChoices().get(0).getMessage();
            SwarmUtil.debugPrint(debug, "Received completion:", message);
            
            // Add sender information
            Map<String, Object> messageMap = SwarmToolHandler.chatMessageToMap(message);
            messageMap.put("sender", activeAgent.getName());
            history.add(messageMap);

            if (message.getFunctionCall() == null || !executeTools) {
                SwarmUtil.debugPrint(debug, "Ending turn.");
                break;
            }

            // Handle function calls, updating context_variables, and switching agents
            List<ChatFunctionCall> toolCalls = Arrays.asList(message.getFunctionCall());
            Response partialResponse = SwarmToolHandler.handleToolCalls(
                    toolCalls, activeAgent.getFunctions(), currentContextVariables, debug);
            
            history.addAll(partialResponse.getMessages());
            currentContextVariables.putAll(partialResponse.getContextVariables());
            
            if (partialResponse.getAgent() != null) {
                activeAgent = partialResponse.getAgent();
            }
        }

        return Response.builder()
                .messages(history.subList(initLen, history.size()))
                .agent(activeAgent)
                .contextVariables(currentContextVariables)
                .build();
    }
}
}
