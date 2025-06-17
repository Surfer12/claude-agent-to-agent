package com.anthropic.api;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class AnthropicClientTest {
    private AnthropicClient client;

    @BeforeEach
    void setUp() {
        client = new AnthropicClient.Builder()
            .apiKey("test-api-key")
            .model("claude-opus-4-20250514")
            .maxTokens(1024)
            .build();
    }

    @Test
    void testCreateMessage() {
        List<AnthropicClient.Message> messages = List.of(
            new AnthropicClient.Message("msg1", "user", 
                List.of(new AnthropicClient.Content("text", "Hello, Claude")))
        );

        AnthropicClient.Message response = client.createMessage(messages);
        
        assertNotNull(response);
        assertNotNull(response.getId());
        assertEquals("assistant", response.getRole());
        assertFalse(response.getContent().isEmpty());
    }

    @Test
    void testBuilderWithTools() {
        Set<String> tools = Set.of("get_weather", "get_time");
        Map<String, Object> toolChoice = Map.of("type", "auto");

        AnthropicClient clientWithTools = new AnthropicClient.Builder()
            .apiKey("test-api-key")
            .tools(tools)
            .toolChoice(toolChoice)
            .build();

        assertNotNull(clientWithTools);
        assertEquals(2, clientWithTools.getTools().size());
        assertTrue(clientWithTools.getTools().contains("get_weather"));
        assertTrue(clientWithTools.getTools().contains("get_time"));
    }

    @Test
    void testNullApiKey() {
        assertThrows(NullPointerException.class, () -> 
            new AnthropicClient.Builder()
                .model("claude-opus-4-20250514")
                .build()
        );
    }

    @Test
    void testNullModel() {
        assertThrows(NullPointerException.class, () -> 
            new AnthropicClient.Builder()
                .apiKey("test-api-key")
                .model(null)
                .build()
        );
    }

    @Test
    void testNullMessages() {
        assertThrows(NullPointerException.class, () -> 
            client.createMessage(null)
        );
    }
} 