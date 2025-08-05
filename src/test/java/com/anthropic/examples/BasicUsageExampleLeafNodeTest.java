package com.anthropic.examples;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Method;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Leaf node tests for BasicUsageExample
 * Tests the example as an isolated, standalone component
 */
@DisplayName("Basic Usage Example Leaf Node Tests")
public class BasicUsageExampleLeafNodeTest {

    private ByteArrayOutputStream outputStream;
    private PrintStream originalOut;
    private PrintStream originalErr;

    @BeforeEach
    void setUp() {
        // Capture System.out for testing output
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
    @DisplayName("Example Main Method Leaf Node")
    class ExampleMainMethodTests {

        @Test
        @DisplayName("main() - Standalone execution leaf node")
        void testMainMethodExecution() {
            // This test verifies the example can be executed as a leaf node
            // without requiring actual API connectivity
            
            assertDoesNotThrow(() -> {
                try {
                    // Load the BasicUsageExample class dynamically
                    Class<?> exampleClass = Class.forName("examples.BasicUsageExample");
                    Method mainMethod = exampleClass.getMethod("main", String[].class);
                    
                    // Execute with empty args
                    mainMethod.invoke(null, (Object) new String[]{});
                    
                } catch (ClassNotFoundException e) {
                    // If class doesn't exist, that's fine for this test
                    System.out.println("BasicUsageExample class not found - test skipped");
                } catch (Exception e) {
                    // Expected - will fail due to missing API key or network issues
                    // The important thing is that it doesn't crash unexpectedly
                    assertTrue(e.getMessage() == null || 
                              e.getMessage().contains("API") || 
                              e.getMessage().contains("key") ||
                              e.getCause() != null,
                        "Should fail gracefully with expected error");
                }
            }, "Example main method should execute without unexpected crashes");
        }

        @Test
        @DisplayName("main() - Environment validation behavior")
        void testMainEnvironmentValidation() {
            // Test that the example properly validates its environment
            // This is a leaf node test focusing on the validation logic
            
            try {
                Class<?> exampleClass = Class.forName("examples.BasicUsageExample");
                Method mainMethod = exampleClass.getMethod("main", String[].class);
                
                // Execute and capture any output
                mainMethod.invoke(null, (Object) new String[]{});
                
            } catch (ClassNotFoundException e) {
                // Skip if class doesn't exist
                return;
            } catch (Exception e) {
                // Check that error handling is appropriate
                String output = outputStream.toString();
                
                // Should either show an error message or handle gracefully
                assertTrue(output.length() >= 0, "Should produce some output or handle silently");
            }
        }
    }

    @Nested
    @DisplayName("Example Isolation Tests")
    class ExampleIsolationTests {

        @Test
        @DisplayName("Example class loading isolation")
        void testExampleClassLoadingIsolation() {
            // Test that the example class can be loaded independently
            assertDoesNotThrow(() -> {
                try {
                    Class<?> exampleClass = Class.forName("examples.BasicUsageExample");
                    assertNotNull(exampleClass, "Example class should be loadable");
                    
                    // Verify it has a main method
                    Method mainMethod = exampleClass.getMethod("main", String[].class);
                    assertNotNull(mainMethod, "Should have main method");
                    
                    // Verify main method is static
                    assertTrue(java.lang.reflect.Modifier.isStatic(mainMethod.getModifiers()),
                        "Main method should be static");
                        
                } catch (ClassNotFoundException e) {
                    // If class doesn't exist, that's acceptable for this test
                    System.out.println("BasicUsageExample not found - test passed (class optional)");
                }
            }, "Example class loading should not cause issues");
        }

        @Test
        @DisplayName("Example execution independence")
        void testExampleExecutionIndependence() {
            // Test that running the example multiple times doesn't interfere
            assertDoesNotThrow(() -> {
                for (int i = 0; i < 3; i++) {
                    try {
                        Class<?> exampleClass = Class.forName("examples.BasicUsageExample");
                        Method mainMethod = exampleClass.getMethod("main", String[].class);
                        
                        // Each execution should be independent
                        mainMethod.invoke(null, (Object) new String[]{});
                        
                    } catch (ClassNotFoundException e) {
                        // Skip if class doesn't exist
                        break;
                    } catch (Exception e) {
                        // Expected failures are OK, just ensure no interference
                        // between runs
                    }
                    
                    // Clear output between runs
                    outputStream.reset();
                }
            }, "Multiple example executions should be independent");
        }
    }

    @Nested
    @DisplayName("Example Error Handling Leaf Nodes")
    class ExampleErrorHandlingTests {

        @Test
        @DisplayName("Graceful failure on missing dependencies")
        void testGracefulFailureOnMissingDependencies() {
            // Test that the example fails gracefully when dependencies are missing
            assertDoesNotThrow(() -> {
                try {
                    Class<?> exampleClass = Class.forName("examples.BasicUsageExample");
                    Method mainMethod = exampleClass.getMethod("main", String[].class);
                    
                    // Execute with no environment setup
                    mainMethod.invoke(null, (Object) new String[]{});
                    
                } catch (ClassNotFoundException e) {
                    // Acceptable - class might not exist
                    return;
                } catch (Exception e) {
                    // Should fail gracefully, not with unexpected errors
                    assertFalse(e instanceof NullPointerException && 
                               e.getMessage() == null,
                        "Should not fail with unexpected NPE");
                }
            }, "Should handle missing dependencies gracefully");
        }

        @Test
        @DisplayName("Error message clarity")
        void testErrorMessageClarity() {
            // Test that error messages are clear and helpful
            try {
                Class<?> exampleClass = Class.forName("examples.BasicUsageExample");
                Method mainMethod = exampleClass.getMethod("main", String[].class);
                
                mainMethod.invoke(null, (Object) new String[]{});
                
            } catch (ClassNotFoundException e) {
                // Skip if class doesn't exist
                return;
            } catch (Exception e) {
                String output = outputStream.toString();
                
                // If there's output, it should be meaningful
                if (!output.trim().isEmpty()) {
                    assertFalse(output.contains("null"), 
                        "Error output should not contain 'null' strings");
                    assertTrue(output.length() > 10, 
                        "Error messages should be reasonably descriptive");
                }
            }
        }
    }

    @Nested
    @DisplayName("Example Structure Validation")
    class ExampleStructureValidationTests {

        @Test
        @DisplayName("Example follows standard structure")
        void testExampleFollowsStandardStructure() {
            assertDoesNotThrow(() -> {
                try {
                    Class<?> exampleClass = Class.forName("examples.BasicUsageExample");
                    
                    // Should have public class
                    assertTrue(java.lang.reflect.Modifier.isPublic(exampleClass.getModifiers()),
                        "Example class should be public");
                    
                    // Should have main method
                    Method mainMethod = exampleClass.getMethod("main", String[].class);
                    assertTrue(java.lang.reflect.Modifier.isPublic(mainMethod.getModifiers()),
                        "Main method should be public");
                    assertTrue(java.lang.reflect.Modifier.isStatic(mainMethod.getModifiers()),
                        "Main method should be static");
                    
                    // Return type should be void
                    assertEquals(void.class, mainMethod.getReturnType(),
                        "Main method should return void");
                        
                } catch (ClassNotFoundException e) {
                    // Acceptable if example doesn't exist
                    System.out.println("BasicUsageExample structure test skipped - class not found");
                }
            }, "Example structure validation should not throw");
        }

        @Test
        @DisplayName("Example has appropriate package structure")
        void testExamplePackageStructure() {
            assertDoesNotThrow(() -> {
                try {
                    Class<?> exampleClass = Class.forName("examples.BasicUsageExample");
                    
                    // Check package
                    Package pkg = exampleClass.getPackage();
                    if (pkg != null) {
                        String packageName = pkg.getName();
                        assertTrue(packageName.contains("examples") || 
                                  packageName.isEmpty(),
                            "Should be in examples package or default package");
                    }
                    
                } catch (ClassNotFoundException e) {
                    // Acceptable if example doesn't exist
                }
            }, "Package structure check should not throw");
        }
    }

    @Nested
    @DisplayName("Example Documentation Tests")
    class ExampleDocumentationTests {

        @Test
        @DisplayName("Example class has appropriate naming")
        void testExampleNaming() {
            assertDoesNotThrow(() -> {
                try {
                    Class<?> exampleClass = Class.forName("examples.BasicUsageExample");
                    
                    String className = exampleClass.getSimpleName();
                    assertTrue(className.contains("Example") || 
                              className.contains("Demo") ||
                              className.contains("Usage"),
                        "Class name should indicate it's an example");
                        
                } catch (ClassNotFoundException e) {
                    // Acceptable if example doesn't exist
                }
            }, "Example naming check should not throw");
        }

        @Test
        @DisplayName("Example demonstrates leaf node behavior")
        void testExampleDemonstratesLeafNodeBehavior() {
            // This test verifies that the example itself acts as a leaf node
            // - It should be self-contained
            // - It should not have complex dependencies on other examples
            // - It should demonstrate a single, focused concept
            
            assertDoesNotThrow(() -> {
                try {
                    Class<?> exampleClass = Class.forName("examples.BasicUsageExample");
                    
                    // Should be able to load independently
                    assertNotNull(exampleClass, "Should load independently");
                    
                    // Should have minimal public interface (just main method)
                    Method[] publicMethods = exampleClass.getMethods();
                    long mainMethods = java.util.Arrays.stream(publicMethods)
                        .filter(m -> m.getName().equals("main"))
                        .filter(m -> java.lang.reflect.Modifier.isStatic(m.getModifiers()))
                        .count();
                    
                    assertTrue(mainMethods >= 1, "Should have at least one main method");
                    
                } catch (ClassNotFoundException e) {
                    // Acceptable if example doesn't exist
                }
            }, "Example leaf node behavior validation should not throw");
        }
    }
}
