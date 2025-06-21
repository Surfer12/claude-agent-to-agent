package com.anthropic.claude.agent.core;

/**
 * Represents a message in the conversation history.
 */
public class Message {
    
    private String role;
    private String content;
    
    // Default constructor
    public Message() {}
    
    // Constructor with parameters
    public Message(String role, String content) {
        this.role = role;
        this.content = content;
    }
    
    // Getters and Setters
    public String getRole() {
        return role;
    }
    
    public void setRole(String role) {
        this.role = role;
    }
    
    public String getContent() {
        return content;
    }
    
    public void setContent(String content) {
        this.content = content;
    }
    
    @Override
    public String toString() {
        return "Message{" +
                "role='" + role + '\'' +
                ", content='" + content + '\'' +
                '}';
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        
        Message message = (Message) o;
        
        if (role != null ? !role.equals(message.role) : message.role != null) return false;
        return content != null ? content.equals(message.content) : message.content == null;
    }
    
    @Override
    public int hashCode() {
        int result = role != null ? role.hashCode() : 0;
        result = 31 * result + (content != null ? content.hashCode() : 0);
        return result;
    }
}
