package com.swarm.core;

import com.swarm.client.StreamingOpenAIClient;
import com.swarm.streaming.StreamingHandler;
import com.swarm.types.*;
import com.swarm.util.SwarmUtil;
import com.theokanning.openai.OpenAiService;
import com.theokanning.openai.completion.chat.*;
import io.reactivex.rxjava3.core.Observable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Main Swarm class that orchestrates multi-agent conversations.
 * Equivalent to the Python Swarm class with streaming support.
 */
public class Swarm {
    private static final Logger logger = LoggerFactory.getLogger(Swarm.class);
    private static final String CTX_VARS_NAME = "context_variables";
    
    private final OpenAiService client;
    private final StreamingOpenAIClient streamingClient;

    public Swarm() {
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.trim().isEmpty()) {
            throw new IllegalStateException("OPENAI_API_KEY environment variable is required");
        }
        this.client = new OpenAiService(apiKey);
        this.streamingClient = new StreamingOpenAIClient(apiKey);
    }

    public Swarm(OpenAiService client) {
        this.client = Objects.requireNonNull(client, "OpenAI client cannot be null");
        // Extract API key from client if possible, otherwise use environment
        String apiKey = System.getenv("OPENAI_API_KEY");
        this.streamingClient = new StreamingOpenAIClient(apiKey);
    }

    public Swarm(String apiKey) {
        this.client = new OpenAiService(apiKey);
        this.streamingClient = new StreamingOpenAIClient(apiKey);
    }

    public Swarm(String apiKey, StreamingOpenAIClient streamingClient) {
        this.client = new OpenAiService(apiKey);
        this.streamingClient = streamingClient;
    }

    /**
     * Get chat completion from OpenAI API (non-streaming).
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
     * Get streaming chat completion from OpenAI API.
     */
    public Observable<StreamingResponse> getStreamingChatCompletion(Agent agent, List<ChatMessage> history, 
                                                                   Map<String, Object> contextVariables,
                                                                   String modelOverride, boolean debug) {
        
        Map<String, Object> defaultContextVars = SwarmUtil.createDefaultContextVariables(contextVariables);
        String instructions = agent.getInstructionsAsString(defaultContextVars);
        
        // Convert messages to the format expected by the streaming client
        List<Map<String, Object>> messageList = new ArrayList<>();
        messageList.add(Map.of("role", "system", "content", instructions));
        
        for (ChatMessage msg : history) {
            Map<String, Object> messageMap = new HashMap<>();
            messageMap.put("role", msg.getRole());
            messageMap.put("content", msg.getContent());
            messageList.add(messageMap);
        }
        
        SwarmUtil.debugPrint(debug, "Getting streaming chat completion for...:", messageList);

        // Prepare request body
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", modelOverride != null ? modelOverride : agent.getModel());
        requestBody.put("messages", messageList);
        requestBody.put("stream", true);
        
        // Add tools if available
        if (!agent.getFunctions().isEmpty()) {
            List<Map<String, Object>> tools = agent.getFunctions().stream()
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
                        
                        return toolDef;
                    })
                    .collect(Collectors.toList());
            
            requestBody.put("tools", tools);
            
            if (agent.getToolChoice() != null) {
                requestBody.put("tool_choice", agent.getToolChoice());
            }
            
            requestBody.put("parallel_tool_calls", agent.isParallelToolCalls());
        }
        
        return streamingClient.createStreamingChatCompletion(requestBody);
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
     * Run and stream method - equivalent to Python run_and_stream.
     */
    public Observable<Map<String, Object>> runAndStream(Agent agent, List<Map<String, Object>> messages,
                                                       Map<String, Object> contextVariables, String modelOverride,
                                                       boolean debug, int maxTurns, boolean executeTools) {
        
        return Observable.create(emitter -> {
            Agent activeAgent = agent;
            Map<String, Object> currentContextVariables = SwarmUtil.deepCopy(contextVariables);
            List<Map<String, Object>> history = SwarmUtil.deepCopy(messages);
            int initLen = messages.size();
            
            try {
                while (history.size() - initLen < maxTurns && activeAgent != null) {
                    
                    // Convert history to ChatMessage format
                    List<ChatMessage> chatMessages = history.stream()
                            .map(SwarmToolHandler::mapToChatMessage)
                            .collect(Collectors.toList());
                    
                    // Initialize message accumulator
                    Map<String, Object> message = new HashMap<>();
                    message.put("content", "");
                    message.put("sender", activeAgent.getName());
                    message.put("role", "assistant");
                    message.put("function_call", null);
                    message.put("tool_calls", new ConcurrentHashMap<Integer, Map<String, Object>>());
                    
                    // Emit start delimiter
                    emitter.onNext(StreamingHandler.createDelimiterEvent("start"));
                    
                    // Get streaming completion
                    Observable<StreamingResponse> streamingObservable = getStreamingChatCompletion(
                            activeAgent, chatMessages, currentContextVariables, modelOverride, debug);
                    
                    // Process streaming responses
                    streamingObservable.blockingSubscribe(
                        streamingResponse -> {
                            if (streamingResponse.getChoices() != null && !streamingResponse.getChoices().isEmpty()) {
                                StreamingResponse.StreamingChoice choice = streamingResponse.getChoices().get(0);
                                StreamingResponse.StreamingDelta delta = choice.getDelta();
                                
                                if (delta != null) {
                                    // Emit delta event
                                    emitter.onNext(StreamingHandler.createDeltaEvent(delta, activeAgent.getName()));
                                    
                                    // Merge chunk into message
                                    StreamingHandler.mergeStreamingChunk(message, delta);
                                }
                            }
                        },
                        error -> {
                            logger.error("Streaming error", error);
                            emitter.onError(error);
                        }
                    );
                    
                    // Emit end delimiter
                    emitter.onNext(StreamingHandler.createDelimiterEvent("end"));
                    
                    // Finalize tool calls
                    StreamingHandler.finalizeToolCalls(message);
                    
                    SwarmUtil.debugPrint(debug, "Received completion:", message);
                    history.add(message);
                    
                    // Check if we should continue
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> toolCalls = (List<Map<String, Object>>) message.get("tool_calls");
                    
                    if (toolCalls == null || toolCalls.isEmpty() || !executeTools) {
                        SwarmUtil.debugPrint(debug, "Ending turn.");
                        break;
                    }
                    
                    // Convert tool calls and handle them
                    List<ChatFunctionCall> functionCalls = toolCalls.stream()
                            .map(toolCall -> {
                                @SuppressWarnings("unchecked")
                                Map<String, Object> function = (Map<String, Object>) toolCall.get("function");
                                ChatFunctionCall functionCall = new ChatFunctionCall();
                                functionCall.setName((String) function.get("name"));
                                functionCall.setArguments((String) function.get("arguments"));
                                return functionCall;
                            })
                            .collect(Collectors.toList());
                    
                    // Handle function calls
                    Response partialResponse = SwarmToolHandler.handleToolCalls(
                            functionCalls, activeAgent.getFunctions(), currentContextVariables, debug);
                    
                    history.addAll(partialResponse.getMessages());
                    currentContextVariables.putAll(partialResponse.getContextVariables());
                    
                    if (partialResponse.getAgent() != null) {
                        activeAgent = partialResponse.getAgent();
                    }
                }
                
                // Emit final response
                Response finalResponse = Response.builder()
                        .messages(history.subList(initLen, history.size()))
                        .agent(activeAgent)
                        .contextVariables(currentContextVariables)
                        .build();
                
                emitter.onNext(StreamingHandler.createResponseEvent(finalResponse));
                emitter.onComplete();
                
            } catch (Exception e) {
                logger.error("Error in streaming run", e);
                emitter.onError(e);
            }
        });
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
        
        if (stream) {
            // For streaming, we need to collect all events and return the final response
            final Response[] finalResponse = {null};
            
            runAndStream(agent, messages, contextVariables, modelOverride, debug, maxTurns, executeTools)
                .blockingSubscribe(
                    event -> {
                        String eventType = (String) event.get("type");
                        if ("response".equals(eventType)) {
                            @SuppressWarnings("unchecked")
                            Map<String, Object> responseData = (Map<String, Object>) event.get("data");
                            
                            @SuppressWarnings("unchecked")
                            List<Map<String, Object>> responseMessages = (List<Map<String, Object>>) responseData.get("messages");
                            String agentName = (String) responseData.get("agent");
                            @SuppressWarnings("unchecked")
                            Map<String, Object> responseContextVars = (Map<String, Object>) responseData.get("context_variables");
                            
                            // Find the agent by name (simplified)
                            Agent responseAgent = agentName != null ? 
                                Agent.builder().name(agentName).build() : null;
                            
                            finalResponse[0] = Response.builder()
                                .messages(responseMessages)
                                .agent(responseAgent)
                                .contextVariables(responseContextVars)
                                .build();
                        }
                    },
                    error -> {
                        throw new RuntimeException("Streaming error", error);
                    }
                );
            
            return finalResponse[0];
        }
        
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

    /**
     * Test connection to OpenAI API.
     */
    public boolean testConnection() {
        return streamingClient.testConnection();
    }

    /**
     * Close resources.
     */
    public void close() {
        if (streamingClient != null) {
            streamingClient.close();
        }
    }
}
}
