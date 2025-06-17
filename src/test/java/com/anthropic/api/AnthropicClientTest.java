package com.anthropic.api;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Map;
import java.util.Set;
import kong.unirest.HttpResponse;

public class AnthropicClientTest {
    private AnthropicClient client;
    private static final String TEST_API_KEY = System.getenv("ANTHROPIC_API_KEY");
    private static final String TEST_MODEL = "claude-3-7-sonnet-20250219";

    @BeforeEach
    void setUp() {
        assertNotNull(TEST_API_KEY, "ANTHROPIC_API_KEY environment variable must be set");
        client = new AnthropicClient.Builder()
            .apiKey(TEST_API_KEY)
            .model(TEST_MODEL)
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

    @Test
    void testMultiStepPromptGeneration() {
        // Step 1: Generate initial prompt
        var generateRequest = new AnthropicClient.GeneratePromptRequest(
            TEST_MODEL,
            "a chef for a meal prep planning service"
        );
        HttpResponse<String> generateResponse = client.generatePrompt(generateRequest);
        assertNotNull(generateResponse);
        assertTrue(generateResponse.isSuccess());
        String generatedPrompt = generateResponse.getBody();
        assertNotNull(generatedPrompt);

        // Step 2: Improve the generated prompt
        var improveRequest = new AnthropicClient.ImprovePromptRequest(
            "Make it more detailed and include cooking times",
            List.of(new AnthropicClient.Message("", "user", 
                List.of(new AnthropicClient.Content("text", generatedPrompt)))),
            "You are a professional chef",
            TEST_MODEL
        );
        HttpResponse<String> improveResponse = client.improvePrompt(improveRequest);
        assertNotNull(improveResponse);
        assertTrue(improveResponse.isSuccess());
        String improvedPrompt = improveResponse.getBody();
        assertNotNull(improvedPrompt);

        // Step 3: Create a template from the improved prompt
        var templatizeRequest = new AnthropicClient.TemplatizePromptRequest(
            List.of(new AnthropicClient.Message("", "user", 
                List.of(new AnthropicClient.Content("text", improvedPrompt)))),
            "You are a professional chef"
        );
        HttpResponse<String> templatizeResponse = client.templatizePrompt(templatizeRequest);
        assertNotNull(templatizeResponse);
        assertTrue(templatizeResponse.isSuccess());
        String template = templatizeResponse.getBody();
        assertNotNull(template);

        // Verify the final template contains expected elements
        assertTrue(template.contains("cooking times"));
        assertTrue(template.contains("meal prep"));
    }
} 