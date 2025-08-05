package com.anthropic.api;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.mockito.junit.jupiter.MockitoExtension;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Test suite for MessageCreateParams - a leaf node data class.
 * Tests the builder pattern, getters, and data integrity.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("MessageCreateParams Tests")
class MessageCreateParamsTest {

    private MessageCreateParams.Builder builder;
    
    @BeforeEach
    void setUp() {
        builder = new MessageCreateParams.Builder();
    }

    @Nested
    @DisplayName("Builder Pattern Tests")
    class BuilderPatternTests {

        @Test
        @DisplayName("Should create params with default values")
        void shouldCreateParamsWithDefaults() {
            MessageCreateParams params = builder
                .model("claude-3-sonnet")
                .build();

            assertEquals("claude-3-sonnet", params.getModel());
            assertEquals(4096, params.getMaxTokens());
            assertEquals(1.0, params.getTemperature(), 0.001);
            assertNull(params.getSystem());
            assertNull(params.getMessages());
            assertNull(params.getTools());
            assertNull(params.getBetas());
        }

        @Test
        @DisplayName("Should create params with all values set")
        void shouldCreateParamsWithAllValues() {
            List<Map<String, Object>> messages = Arrays.asList(
                Map.of("role", "user", "content", "Hello")
            );
            List<Map<String, Object>> tools = Arrays.asList(
                Map.of("name", "test_tool", "description", "A test tool")
            );
            List<String> betas = Arrays.asList("beta-feature-1");

            MessageCreateParams params = builder
                .model("claude-3-opus")
                .maxTokens(2048)
                .temperature(0.7)
                .system("You are a helpful assistant")
                .messages(messages)
                .tools(tools)
                .betas(betas)
                .build();

            assertEquals("claude-3-opus", params.getModel());
            assertEquals(2048, params.getMaxTokens());
            assertEquals(0.7, params.getTemperature(), 0.001);
            assertEquals("You are a helpful assistant", params.getSystem());
            assertEquals(messages, params.getMessages());
            assertEquals(tools, params.getTools());
            assertEquals(betas, params.getBetas());
        }

        @Test
        @DisplayName("Should support method chaining")
        void shouldSupportMethodChaining() {
            MessageCreateParams params = builder
                .model("claude-3-haiku")
                .maxTokens(1024)
                .temperature(0.5)
                .system("Test system")
                .build();

            assertNotNull(params);
            assertEquals("claude-3-haiku", params.getModel());
            assertEquals(1024, params.getMaxTokens());
            assertEquals(0.5, params.getTemperature(), 0.001);
            assertEquals("Test system", params.getSystem());
        }

        @Test
        @DisplayName("Should handle null values gracefully")
        void shouldHandleNullValues() {
            MessageCreateParams params = builder
                .model(null)
                .system(null)
                .messages(null)
                .tools(null)
                .betas(null)
                .build();

            assertNull(params.getModel());
            assertNull(params.getSystem());
            assertNull(params.getMessages());
            assertNull(params.getTools());
            assertNull(params.getBetas());
        }
    }

    @Nested
    @DisplayName("Data Integrity Tests")
    class DataIntegrityTests {

        @Test
        @DisplayName("Should preserve message data structure")
        void shouldPreserveMessageDataStructure() {
            Map<String, Object> message1 = new HashMap<>();
            message1.put("role", "user");
            message1.put("content", "Hello world");
            
            Map<String, Object> message2 = new HashMap<>();
            message2.put("role", "assistant");
            message2.put("content", "Hi there!");

            List<Map<String, Object>> messages = Arrays.asList(message1, message2);

            MessageCreateParams params = builder
                .model("claude-3-sonnet")
                .messages(messages)
                .build();

            List<Map<String, Object>> retrievedMessages = params.getMessages();
            assertEquals(2, retrievedMessages.size());
            assertEquals("user", retrievedMessages.get(0).get("role"));
            assertEquals("Hello world", retrievedMessages.get(0).get("content"));
            assertEquals("assistant", retrievedMessages.get(1).get("role"));
            assertEquals("Hi there!", retrievedMessages.get(1).get("content"));
        }

        @Test
        @DisplayName("Should preserve tool data structure")
        void shouldPreserveToolDataStructure() {
            Map<String, Object> tool = new HashMap<>();
            tool.put("name", "calculator");
            tool.put("description", "Performs calculations");
            tool.put("input_schema", Map.of("type", "object"));

            List<Map<String, Object>> tools = Arrays.asList(tool);

            MessageCreateParams params = builder
                .model("claude-3-sonnet")
                .tools(tools)
                .build();

            List<Map<String, Object>> retrievedTools = params.getTools();
            assertEquals(1, retrievedTools.size());
            assertEquals("calculator", retrievedTools.get(0).get("name"));
            assertEquals("Performs calculations", retrievedTools.get(0).get("description"));
        }

        @Test
        @DisplayName("Should handle empty collections")
        void shouldHandleEmptyCollections() {
            MessageCreateParams params = builder
                .model("claude-3-sonnet")
                .messages(Arrays.asList())
                .tools(Arrays.asList())
                .betas(Arrays.asList())
                .build();

            assertTrue(params.getMessages().isEmpty());
            assertTrue(params.getTools().isEmpty());
            assertTrue(params.getBetas().isEmpty());
        }
    }

    @Nested
    @DisplayName("Edge Cases and Validation")
    class EdgeCasesTests {

        @Test
        @DisplayName("Should handle extreme temperature values")
        void shouldHandleExtremeTemperatureValues() {
            MessageCreateParams params1 = builder
                .model("claude-3-sonnet")
                .temperature(0.0)
                .build();
            assertEquals(0.0, params1.getTemperature(), 0.001);

            MessageCreateParams params2 = new MessageCreateParams.Builder()
                .model("claude-3-sonnet")
                .temperature(2.0)
                .build();
            assertEquals(2.0, params2.getTemperature(), 0.001);
        }

        @Test
        @DisplayName("Should handle extreme token values")
        void shouldHandleExtremeTokenValues() {
            MessageCreateParams params1 = builder
                .model("claude-3-sonnet")
                .maxTokens(1)
                .build();
            assertEquals(1, params1.getMaxTokens());

            MessageCreateParams params2 = new MessageCreateParams.Builder()
                .model("claude-3-sonnet")
                .maxTokens(200000)
                .build();
            assertEquals(200000, params2.getMaxTokens());
        }

        @Test
        @DisplayName("Should handle very long system prompts")
        void shouldHandleVeryLongSystemPrompts() {
            String longSystem = "A".repeat(10000);
            MessageCreateParams params = builder
                .model("claude-3-sonnet")
                .system(longSystem)
                .build();

            assertEquals(longSystem, params.getSystem());
            assertEquals(10000, params.getSystem().length());
        }

        @Test
        @DisplayName("Should handle unicode in system prompts")
        void shouldHandleUnicodeInSystemPrompts() {
            String unicodeSystem = "Hello 世界 🌍 café naïve résumé";
            MessageCreateParams params = builder
                .model("claude-3-sonnet")
                .system(unicodeSystem)
                .build();

            assertEquals(unicodeSystem, params.getSystem());
        }

        @Test
        @DisplayName("Should handle complex nested message structures")
        void shouldHandleComplexNestedMessageStructures() {
            Map<String, Object> complexMessage = new HashMap<>();
            complexMessage.put("role", "user");
            complexMessage.put("content", Arrays.asList(
                Map.of("type", "text", "text", "Hello"),
                Map.of("type", "image", "source", Map.of("type", "base64", "data", "iVBORw0KGgo="))
            ));

            List<Map<String, Object>> messages = Arrays.asList(complexMessage);

            MessageCreateParams params = builder
                .model("claude-3-sonnet")
                .messages(messages)
                .build();

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> content = (List<Map<String, Object>>) 
                params.getMessages().get(0).get("content");
            
            assertEquals(2, content.size());
            assertEquals("text", content.get(0).get("type"));
            assertEquals("image", content.get(1).get("type"));
        }
    }

    @Nested
    @DisplayName("Builder Reusability Tests")
    class BuilderReusabilityTests {

        @Test
        @DisplayName("Should create independent instances from same builder")
        void shouldCreateIndependentInstancesFromSameBuilder() {
            MessageCreateParams.Builder sharedBuilder = new MessageCreateParams.Builder()
                .model("claude-3-sonnet")
                .maxTokens(1000);

            MessageCreateParams params1 = sharedBuilder
                .temperature(0.5)
                .build();

            MessageCreateParams params2 = sharedBuilder
                .temperature(0.8)
                .build();

            assertEquals(0.5, params1.getTemperature(), 0.001);
            assertEquals(0.8, params2.getTemperature(), 0.001);
            assertEquals("claude-3-sonnet", params1.getModel());
            assertEquals("claude-3-sonnet", params2.getModel());
        }

        @Test
        @DisplayName("Should allow builder modification after build")
        void shouldAllowBuilderModificationAfterBuild() {
            MessageCreateParams params1 = builder
                .model("claude-3-sonnet")
                .maxTokens(1000)
                .build();

            MessageCreateParams params2 = builder
                .model("claude-3-opus")
                .maxTokens(2000)
                .build();

            assertEquals("claude-3-sonnet", params1.getModel());
            assertEquals(1000, params1.getMaxTokens());
            assertEquals("claude-3-opus", params2.getModel());
            assertEquals(2000, params2.getMaxTokens());
        }
    }
}
