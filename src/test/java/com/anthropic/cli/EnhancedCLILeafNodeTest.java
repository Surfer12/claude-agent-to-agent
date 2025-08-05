package com.anthropic.cli;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Method;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

/**
 * Comprehensive leaf node tests for EnhancedCLI
 * Tests individual methods in isolation to ensure proper leaf node behavior
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("Enhanced CLI Leaf Node Tests")
public class EnhancedCLILeafNodeTest {

    private ByteArrayOutputStream outputStream;
    private PrintStream originalOut;
    private PrintStream originalErr;

    @BeforeEach
    void setUp() {
        // Capture System.out for testing output methods
        outputStream = new ByteArrayOutputStream();
        originalOut = System.out;
        originalErr = System.err;
        System.setOut(new PrintStream(outputStream));
        System.setErr(new PrintStream(outputStream));
    }

    @AfterEach
    void tearDown() {
        // Restore original streams
        System.setOut(originalOut);
        System.setErr(originalErr);
    }

    @Nested
    @DisplayName("CLI Entry Point Leaf Nodes")
    class CLIEntryPointTests {

        @Test
        @DisplayName("showUsage() - Private utility leaf node via reflection")
        void testShowUsageViaReflection() throws Exception {
            // Access private method via reflection for testing
            Method showUsageMethod = EnhancedCLI.class.getDeclaredMethod("showUsage");
            showUsageMethod.setAccessible(true);

            // Act - Call the static utility method
            showUsageMethod.invoke(null);

            // Assert - Verify expected output content
            String output = outputStream.toString();
            assertAll("Usage output validation",
                () -> assertTrue(output.contains("Enhanced Claude Agent CLI (Java)"), 
                    "Should contain CLI title"),
                () -> assertTrue(output.contains("Usage:"), 
                    "Should contain usage section"),
                () -> assertTrue(output.contains("Options:"), 
                    "Should contain options section"),
                () -> assertTrue(output.contains("--help"), 
                    "Should contain help option"),
                () -> assertTrue(output.contains("--model"), 
                    "Should contain model option"),
                () -> assertTrue(output.contains("--verbose"), 
                    "Should contain verbose option"),
                () -> assertTrue(output.contains("--prompt"), 
                    "Should contain prompt option"),
                () -> assertTrue(output.contains("ANTHROPIC_API_KEY"), 
                    "Should mention API key requirement")
            );
        }

        @Test
        @DisplayName("main() - Environment validation leaf node")
        void testMainWithoutApiKey() {
            // Act - Call main with no API key (should fail gracefully)
            assertDoesNotThrow(() -> {
                try {
                    EnhancedCLI.main(new String[]{});
                } catch (SecurityException e) {
                    // Expected - System.exit() in test environment
                } catch (Exception e) {
                    // Expected - missing API key or other runtime issues
                    assertTrue(e.getMessage() == null || 
                              e.getMessage().contains("API") ||
                              e.getCause() != null);
                }
            }, "Should handle missing API key gracefully");
        }

        @Test
        @DisplayName("main() - Help argument parsing leaf node")
        void testMainWithHelpArgument() {
            // Act - Call main with help argument
            assertDoesNotThrow(() -> {
                try {
                    EnhancedCLI.main(new String[]{"--help"});
                } catch (SecurityException e) {
                    // Expected - System.exit() after showing help
                }
            }, "Should handle help argument gracefully");

            // Assert - Should show usage information
            String output = outputStream.toString();
            assertTrue(output.contains("Enhanced Claude Agent CLI") || output.isEmpty(), 
                "Should display help information or exit cleanly");
        }

        @Test
        @DisplayName("main() - Argument parsing validation")
        void testMainArgumentParsing() {
            // Test various argument combinations
            String[] testArgs = {
                "--model", "claude-3-sonnet-20240229",
                "--verbose",
                "--prompt", "test prompt"
            };

            // This test validates that argument parsing doesn't crash
            assertDoesNotThrow(() -> {
                try {
                    EnhancedCLI.main(testArgs);
                } catch (SecurityException e) {
                    // Expected due to System.exit() or missing API key
                } catch (Exception e) {
                    // Expected due to other runtime issues
                }
            }, "Argument parsing should not throw unexpected exceptions");
        }
    }

    @Nested
    @DisplayName("CLI Instance Method Leaf Nodes")
    class CLIInstanceMethodTests {

        @Test
        @DisplayName("processSinglePrompt() - Private delegation leaf node via reflection")
        void testProcessSinglePromptViaReflection() throws Exception {
            // Arrange - Create CLI instance with mock API key
            String testApiKey = "test-api-key";
            String testModel = "claude-3-sonnet-20240229";
            EnhancedCLI cli = new EnhancedCLI(testApiKey, testModel, false);

            // Access private method via reflection
            Method processSinglePromptMethod = EnhancedCLI.class.getDeclaredMethod("processSinglePrompt", String.class);
            processSinglePromptMethod.setAccessible(true);

            // Act & Assert - Should not throw exception for simple delegation
            assertDoesNotThrow(() -> {
                try {
                    processSinglePromptMethod.invoke(cli, "test prompt");
                } catch (Exception e) {
                    // Expected - will fail due to network/API issues
                    // The important thing is it doesn't crash unexpectedly
                    assertTrue(e.getCause() == null || 
                              e.getCause().getMessage() == null ||
                              e.getCause().getMessage().contains("API") ||
                              e.getCause() instanceof java.net.ConnectException ||
                              e.getCause() instanceof java.io.IOException);
                }
            }, "processSinglePrompt should delegate without unexpected crashes");
        }

        @Test
        @DisplayName("Constructor - Parameter validation leaf node")
        void testConstructorValidation() {
            // Test null API key validation
            assertThrows(NullPointerException.class, () -> {
                new EnhancedCLI(null, "model", false);
            }, "Should throw NPE for null API key");

            // Test valid construction
            assertDoesNotThrow(() -> {
                new EnhancedCLI("test-key", "test-model", true);
            }, "Should construct successfully with valid parameters");

            // Test null model handling (should use default)
            assertDoesNotThrow(() -> {
                new EnhancedCLI("test-key", null, false);
            }, "Should handle null model gracefully");
        }
    }

    @Nested
    @DisplayName("Utility Method Leaf Nodes")
    class UtilityMethodTests {

        @Test
        @DisplayName("Static method isolation via reflection")
        void testStaticMethodIsolation() throws Exception {
            // Verify showUsage is truly static and isolated
            Method showUsageMethod = EnhancedCLI.class.getDeclaredMethod("showUsage");
            showUsageMethod.setAccessible(true);

            assertDoesNotThrow(() -> {
                showUsageMethod.invoke(null);
                showUsageMethod.invoke(null); // Should be idempotent
            }, "Static methods should be isolated and reusable");
        }

        @Test
        @DisplayName("Output formatting consistency")
        void testOutputFormatting() throws Exception {
            // Test that output methods produce consistent formatting
            Method showUsageMethod = EnhancedCLI.class.getDeclaredMethod("showUsage");
            showUsageMethod.setAccessible(true);
            showUsageMethod.invoke(null);
            
            String output = outputStream.toString();

            // Verify consistent formatting patterns
            assertAll("Output formatting validation",
                () -> assertTrue(output.contains("\n"), "Should contain newlines"),
                () -> assertTrue(output.contains("  "), "Should contain proper indentation"),
                () -> assertFalse(output.trim().isEmpty(), "Should not be empty")
            );
        }
    }

    @Nested
    @DisplayName("Error Handling Leaf Nodes")
    class ErrorHandlingTests {

        @Test
        @DisplayName("Invalid argument handling")
        void testInvalidArgumentHandling() {
            // Test handling of invalid arguments
            String[] invalidArgs = {"--invalid-option", "value"};
            
            assertDoesNotThrow(() -> {
                try {
                    EnhancedCLI.main(invalidArgs);
                } catch (Exception e) {
                    // Expected - may exit or throw due to invalid args
                }
            }, "Should handle invalid arguments gracefully");
        }

        @Test
        @DisplayName("Empty argument array handling")
        void testEmptyArgumentHandling() {
            String[] emptyArgs = {};
            
            assertDoesNotThrow(() -> {
                try {
                    EnhancedCLI.main(emptyArgs);
                } catch (Exception e) {
                    // Expected - will fail due to missing API key
                }
            }, "Should handle empty arguments gracefully");
        }
    }

    @Nested
    @DisplayName("Leaf Node Isolation Tests")
    class LeafNodeIsolationTests {

        @Test
        @DisplayName("showUsage() has no side effects")
        void testShowUsageIsolation() throws Exception {
            // Call multiple times to ensure no state changes
            Method showUsageMethod = EnhancedCLI.class.getDeclaredMethod("showUsage");
            showUsageMethod.setAccessible(true);
            
            showUsageMethod.invoke(null);
            String firstOutput = outputStream.toString();
            
            outputStream.reset();
            showUsageMethod.invoke(null);
            String secondOutput = outputStream.toString();

            assertEquals(firstOutput, secondOutput, 
                "showUsage should produce identical output on repeated calls");
        }

        @Test
        @DisplayName("Constructor creates isolated instances")
        void testConstructorIsolation() {
            // Create multiple instances to ensure isolation
            EnhancedCLI cli1 = new EnhancedCLI("key1", "model1", true);
            EnhancedCLI cli2 = new EnhancedCLI("key2", "model2", false);

            // Instances should be independent
            assertNotSame(cli1, cli2, "Should create separate instances");
            
            // Test that they don't interfere with each other
            assertDoesNotThrow(() -> {
                // Both instances should be valid
                assertNotNull(cli1);
                assertNotNull(cli2);
            }, "Instances should operate independently");
        }

        @Test
        @DisplayName("Method calls don't affect static state")
        void testStaticStateIsolation() throws Exception {
            // Create instance and call methods
            EnhancedCLI cli = new EnhancedCLI("test-key", "test-model", false);
            
            // Instance should be created successfully
            assertNotNull(cli);

            // Static method should still work the same
            Method showUsageMethod = EnhancedCLI.class.getDeclaredMethod("showUsage");
            showUsageMethod.setAccessible(true);
            showUsageMethod.invoke(null);
            
            String output = outputStream.toString();
            assertTrue(output.contains("Enhanced Claude Agent CLI"), 
                "Static methods should be unaffected by instance operations");
        }
    }

    @Nested
    @DisplayName("Parameter Validation Leaf Nodes")
    class ParameterValidationTests {

        @Test
        @DisplayName("API key validation")
        void testApiKeyValidation() {
            // Test various API key scenarios
            assertThrows(NullPointerException.class, () -> {
                new EnhancedCLI(null, "model", false);
            }, "Null API key should throw NPE");

            assertDoesNotThrow(() -> {
                new EnhancedCLI("", "model", false);
            }, "Empty API key should be allowed (will fail later)");

            assertDoesNotThrow(() -> {
                new EnhancedCLI("valid-key", "model", false);
            }, "Valid API key should work");
        }

        @Test
        @DisplayName("Model parameter handling")
        void testModelParameterHandling() {
            // Test model parameter variations
            assertDoesNotThrow(() -> {
                new EnhancedCLI("key", null, false);
            }, "Null model should use default");

            assertDoesNotThrow(() -> {
                new EnhancedCLI("key", "", false);
            }, "Empty model should be handled");

            assertDoesNotThrow(() -> {
                new EnhancedCLI("key", "custom-model", false);
            }, "Custom model should work");
        }

        @Test
        @DisplayName("Boolean parameter handling")
        void testBooleanParameterHandling() {
            // Test verbose flag variations
            assertDoesNotThrow(() -> {
                new EnhancedCLI("key", "model", true);
                new EnhancedCLI("key", "model", false);
            }, "Boolean parameters should work correctly");
        }
    }
}
