package com.anthropic.claude.agent.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Manages conversation history and context window.
 */
public class MessageHistory {

    private final List<Message> messages;

    public MessageHistory(ModelConfig modelConfig) {
        this.messages = new ArrayList<>();
    }

    /**
     * Add a user message to history.
     */
    public void addMessage(String role, String content) {
        messages.add(new Message(role, content));
        truncateIfNeeded();
    }

    /**
     * Add an assistant response to history.
     */
    public void addAssistantResponse(AgentResponse response) {
        StringBuilder content = new StringBuilder();

        // Add text content
        String textContent = response.getTextContent();
        if (!textContent.isEmpty()) {
            content.append(textContent);
        }

        // Add tool use information
        if (response.hasToolUses()) {
            if (content.length() > 0) {
                content.append("\n");
            }
            content
                .append("[Tool calls: ")
                .append(response.getToolUses().size())
                .append("]");
        }

        messages.add(new Message("assistant", content.toString()));
        truncateIfNeeded();
    }

    /**
     * Add tool results to history.
     */
    public void addToolResults(List<ToolResult> toolResults) {
        StringBuilder content = new StringBuilder();
        content.append("[Tool results: ");

        for (int i = 0; i < toolResults.size(); i++) {
            if (i > 0) {
                content.append(", ");
            }
            ToolResult result = toolResults.get(i);
            content.append(result.getContent());
        }

        content.append("]");

        messages.add(new Message("user", content.toString()));
        truncateIfNeeded();
    }

    /**
     * Truncate history if it exceeds context window.
     */
    private void truncateIfNeeded() {
        // Simple truncation - remove oldest messages if we have too many
        // In a real implementation, you'd want to estimate token count
        int maxMessages = 50; // Rough estimate

        while (messages.size() > maxMessages) {
            messages.remove(0);
        }
    }

    /**
     * Clear all messages.
     */
    public void clear() {
        messages.clear();
    }

    /**
     * Get all messages.
     */
    public List<Message> getMessages() {
        return new ArrayList<>(messages);
    }

    /**
     * Get the number of messages.
     */
    public int size() {
        return messages.size();
    }

    /**
     * Check if history is empty.
     */
    public boolean isEmpty() {
        return messages.isEmpty();
    }
}
