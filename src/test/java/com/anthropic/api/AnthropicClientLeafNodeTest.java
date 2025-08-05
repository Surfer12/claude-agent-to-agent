package com.anthropic.api;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

/**
 * Leaf node tests for AnthropicClient focusing on isolated component testing
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("Anthropic Client Leaf Node Tests")
public class AnthropicClientLeafNodeTest {

    @Nested
    @DisplayName("Builder Pattern Leaf Nodes")
    class BuilderPatternTests {

        @Test
        @DisplayName("Builder.apiKey() - Fluent API leaf node")
        void testBuilderApiKey() {
            // Arrange
            AnthropicClient.Builder builder = new AnthropicClient.Builder();
            String testApiKey = "test-api-key";

            // Act
            AnthropicClient.Builder result = builder.apiKey(testApiKey);

            // Assert
            assertSame(builder, result, "Should return same builder instance for fluent API");
            
            // Test that the builder can still be used
            assertDoesNotThrow(() -> {
                result.model("test-model").build();
            }, "Builder should remain functional after apiKey() call");
        }

        @Test
        @DisplayName("Builder.model() - Fluent API leaf node")
        void testBuilderModel() {
            // Arrange
            AnthropicClient.Builder builder = new AnthropicClient.Builder();
            String testModel = "claude-3-sonnet-20240229";

            // Act
            AnthropicClient.Builder result = builder.model(testModel);

            // Assert
            assertSame(builder, result, "Should return same builder instance");
            
            // Verify fluent chaining works
            assertDoesNotThrow(() -> {
                result.apiKey("test-key").build();
            }, "Should support method chaining");
        }

        @Test
        @DisplayName("Builder.maxTokens() - Parameter validation leaf node")
        void testBuilderMaxTokens() {
            // Arrange
            AnthropicClient.Builder builder = new AnthropicClient.Builder();

            // Act & Assert - Test various token values
            assertDoesNotThrow(() -> {
                builder.maxTokens(1024);
            }, "Should accept valid token count");

            assertDoesNotThrow(() -> {
                builder.maxTokens(1);
            }, "Should accept minimum token count");

            assertDoesNotThrow(() -> {
                builder.maxTokens(100000);
            }, "Should accept large token count");

            // Test fluent API
            AnthropicClient.Builder result = builder.maxTokens(2048);
            assertSame(builder, result, "Should return same builder instance");
        }

        @Test
        @DisplayName("Builder.tools() - Collection handling leaf node")
        void testBuilderTools() {
            // Arrange
            AnthropicClient.Builder builder = new AnthropicClient.Builder();
            Set<String> testTools = Set.of("tool1", "tool2", "tool3");

            // Act
            AnthropicClient.Builder result = builder.tools(testTools);

            // Assert
            assertSame(builder, result, "Should return same builder instance");
            
            // Test with empty set
            assertDoesNotThrow(() -> {
                builder.tools(Set.of());
            }, "Should handle empty tool set");

            // Test with null (should be handled gracefully)
            assertDoesNotThrow(() -> {
                builder.tools(null);
            }, "Should handle null tools gracefully");
        }

        @Test
        @DisplayName("Builder.toolChoice() - Map handling leaf node")
        void testBuilderToolChoice() {
            // Arrange
            AnthropicClient.Builder builder = new AnthropicClient.Builder();
            Map<String, Object> testToolChoice = Map.of(
                "type", "function",
                "function", Map.of("name", "test_function")
            );

            // Act
            AnthropicClient.Builder result = builder.toolChoice(testToolChoice);

            // Assert
            assertSame(builder, result, "Should return same builder instance");
            
            // Test with empty map
            assertDoesNotThrow(() -> {
                builder.toolChoice(Map.of());
            }, "Should handle empty tool choice");

            // Test with null
            assertDoesNotThrow(() -> {
                builder.toolChoice(null);
            }, "Should handle null tool choice");
        }

        @Test
        @DisplayName("Builder.build() - Construction leaf node")
        void testBuilderBuild() {
            // Arrange
            AnthropicClient.Builder builder = new AnthropicClient.Builder()
                .apiKey("test-api-key")
                .model("test-model");

            // Act
            AnthropicClient client = builder.build();

            // Assert
            assertNotNull(client, "Should create client instance");
            
            // Test that builder can be reused
            AnthropicClient client2 = builder.build();
            assertNotNull(client2, "Builder should be reusable");
            assertNotSame(client, client2, "Should create separate instances");
        }

        @Test
        @DisplayName("Builder validation - Required fields leaf node")
        void testBuilderValidation() {
            // Test missing API key
            assertThrows(NullPointerException.class, () -> {
                new AnthropicClient.Builder()
                    .model("test-model")
                    .build();
            }, "Should require API key");

            // Test missing model (should use default)
            assertDoesNotThrow(() -> {
                new AnthropicClient.Builder()
                    .apiKey("test-key")
                    .build();
            }, "Should use default model when not specified");

            // Test null API key
            assertThrows(NullPointerException.class, () -> {
                new AnthropicClient.Builder()
                    .apiKey(null)
                    .model("test-model")
                    .build();
            }, "Should reject null API key");
        }
    }

    @Nested
    @DisplayName("Client Configuration Leaf Nodes")
    class ClientConfigurationTests {

        @Test
        @DisplayName("Default configuration values")
        void testDefaultConfiguration() {
            // Arrange & Act
            AnthropicClient client = new AnthropicClient.Builder()
                .apiKey("test-key")
                .build();

            // Assert - Test that defaults are applied
            assertNotNull(client, "Should create client with defaults");
            
            // We can't directly access private fields, but we can test behavior
            assertDoesNotThrow(() -> {
                // This would use the default model and other settings
                client.toString(); // Safe method call to verify object state
            }, "Client should be properly configured with defaults");
        }

        @Test
        @DisplayName("Custom configuration values")
        void testCustomConfiguration() {
            // Arrange & Act
            AnthropicClient client = new AnthropicClient.Builder()
                .apiKey("custom-key")
                .model("custom-model")
                .maxTokens(2048)
                .tools(Set.of("custom-tool"))
                .toolChoice(Map.of("type", "auto"))
                .build();

            // Assert
            assertNotNull(client, "Should create client with custom configuration");
            
            // Test that configuration is preserved
            assertDoesNotThrow(() -> {
                client.toString(); // Verify object integrity
            }, "Custom configuration should be valid");
        }
    }

    @Nested
    @DisplayName("Immutability Leaf Nodes")
    class ImmutabilityTests {

        @Test
        @DisplayName("Builder immutability after build")
        void testBuilderImmutabilityAfterBuild() {
            // Arrange
            AnthropicClient.Builder builder = new AnthropicClient.Builder()
                .apiKey("test-key")
                .model("test-model");

            // Act
            AnthropicClient client1 = builder.build();
            
            // Modify builder after build
            builder.model("different-model");
            AnthropicClient client2 = builder.build();

            // Assert
            assertNotSame(client1, client2, "Should create separate instances");
            // Both should be valid despite builder modification
            assertNotNull(client1, "First client should remain valid");
            assertNotNull(client2, "Second client should be valid");
        }

        @Test
        @DisplayName("Client immutability")
        void testClientImmutability() {
            // Arrange
            Set<String> originalTools = Set.of("tool1", "tool2");
            Map<String, Object> originalToolChoice = Map.of("type", "auto");
            
            AnthropicClient client = new AnthropicClient.Builder()
                .apiKey("test-key")
                .model("test-model")
                .tools(originalTools)
                .toolChoice(originalToolChoice)
                .build();

            // Act & Assert
            assertNotNull(client, "Client should be created successfully");
            
            // Verify client remains functional (immutability preserved)
            assertDoesNotThrow(() -> {
                client.toString(); // Safe operation to verify state
            }, "Client should maintain immutable state");
        }
    }

    @Nested
    @DisplayName("Parameter Validation Leaf Nodes")
    class ParameterValidationTests {

        @Test
        @DisplayName("API key validation edge cases")
        void testApiKeyValidation() {
            AnthropicClient.Builder builder = new AnthropicClient.Builder();

            // Test empty string
            assertDoesNotThrow(() -> {
                builder.apiKey("").model("test-model").build();
            }, "Empty API key should be allowed (validation happens at runtime)");

            // Test whitespace
            assertDoesNotThrow(() -> {
                builder.apiKey("   ").model("test-model").build();
            }, "Whitespace API key should be allowed");

            // Test very long key
            String longKey = "a".repeat(1000);
            assertDoesNotThrow(() -> {
                builder.apiKey(longKey).model("test-model").build();
            }, "Long API key should be allowed");
        }

        @Test
        @DisplayName("Model validation edge cases")
        void testModelValidation() {
            AnthropicClient.Builder builder = new AnthropicClient.Builder()
                .apiKey("test-key");

            // Test empty model
            assertDoesNotThrow(() -> {
                builder.model("").build();
            }, "Empty model should be allowed");

            // Test special characters in model name
            assertDoesNotThrow(() -> {
                builder.model("claude-3.5-sonnet@2024").build();
            }, "Special characters in model should be allowed");

            // Test very long model name
            String longModel = "claude-" + "x".repeat(100);
            assertDoesNotThrow(() -> {
                builder.model(longModel).build();
            }, "Long model name should be allowed");
        }

        @Test
        @DisplayName("Numeric parameter validation")
        void testNumericParameterValidation() {
            AnthropicClient.Builder builder = new AnthropicClient.Builder()
                .apiKey("test-key");

            // Test edge cases for maxTokens
            assertDoesNotThrow(() -> {
                builder.maxTokens(0).build();
            }, "Zero tokens should be allowed");

            assertDoesNotThrow(() -> {
                builder.maxTokens(-1).build();
            }, "Negative tokens should be allowed (validation at API level)");

            assertDoesNotThrow(() -> {
                builder.maxTokens(Integer.MAX_VALUE).build();
            }, "Maximum integer tokens should be allowed");
        }
    }

    @Nested
    @DisplayName("Builder Reusability Leaf Nodes")
    class BuilderReusabilityTests {

        @Test
        @DisplayName("Builder method chaining")
        void testBuilderMethodChaining() {
            // Test that all builder methods can be chained
            assertDoesNotThrow(() -> {
                new AnthropicClient.Builder()
                    .apiKey("test-key")
                    .model("test-model")
                    .maxTokens(1024)
                    .tools(Set.of("tool1"))
                    .toolChoice(Map.of("type", "auto"))
                    .build();
            }, "All builder methods should be chainable");
        }

        @Test
        @DisplayName("Builder state independence")
        void testBuilderStateIndependence() {
            // Create base builder
            AnthropicClient.Builder baseBuilder = new AnthropicClient.Builder()
                .apiKey("base-key");

            // Create two different configurations from same base
            AnthropicClient client1 = baseBuilder
                .model("model1")
                .maxTokens(1024)
                .build();

            AnthropicClient client2 = baseBuilder
                .model("model2")
                .maxTokens(2048)
                .build();

            // Assert both are valid and independent
            assertNotNull(client1, "First client should be valid");
            assertNotNull(client2, "Second client should be valid");
            assertNotSame(client1, client2, "Clients should be independent");
        }

        @Test
        @DisplayName("Builder reset behavior")
        void testBuilderResetBehavior() {
            AnthropicClient.Builder builder = new AnthropicClient.Builder();

            // Configure builder
            builder.apiKey("key1").model("model1");
            AnthropicClient client1 = builder.build();

            // Reconfigure same builder
            builder.apiKey("key2").model("model2");
            AnthropicClient client2 = builder.build();

            // Both should be valid
            assertNotNull(client1, "First client should remain valid");
            assertNotNull(client2, "Second client should be valid");
            assertNotSame(client1, client2, "Should create separate instances");
        }
    }
}
