# 🎯 Leaf Node Analysis - Templates & Advanced Patterns

## 📋 **Ready-to-Use Templates**

### **Python Leaf Node Test Template**

```python
"""
Template for Python leaf node testing
Copy and modify for your specific use case
"""
import pytest
from unittest.mock import patch, mock_open
from io import StringIO
import sys

class TestYourLeafNodes:
    """Test class for leaf node components"""
    
    @pytest.fixture
    def capture_output(self):
        """Capture stdout/stderr for testing"""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        yield stdout_capture, stderr_capture
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    # Template 1: Pure Function Testing
    def test_pure_function_happy_path(self):
        """Test pure function with valid input"""
        # Arrange
        input_data = "test_input"
        expected_output = "expected_result"
        
        # Act
        result = your_pure_function(input_data)
        
        # Assert
        assert result == expected_output
    
    def test_pure_function_edge_cases(self):
        """Test pure function with edge cases"""
        test_cases = [
            ("", ""),  # Empty input
            (None, None),  # Null input
            ("   ", ""),  # Whitespace
            ("very_long_string" * 100, "expected_long_result")  # Large input
        ]
        
        for input_val, expected in test_cases:
            with pytest.subTest(input=input_val):
                result = your_pure_function(input_val)
                assert result == expected
    
    def test_pure_function_error_conditions(self):
        """Test pure function error handling"""
        with pytest.raises(ValueError):
            your_pure_function("invalid_input")
        
        with pytest.raises(TypeError):
            your_pure_function(123)  # Wrong type
    
    # Template 2: CLI Component Testing
    def test_cli_help_display(self, capture_output):
        """Test CLI help display leaf node"""
        # Act
        show_help()
        
        # Assert
        stdout, stderr = capture_output
        output = stdout.getvalue()
        
        assert "Usage:" in output
        assert "--help" in output
        assert len(output.strip()) > 0
    
    def test_cli_argument_parsing(self):
        """Test CLI argument parsing leaf node"""
        # Test valid arguments
        args = parse_arguments(["--model", "claude-3", "--verbose"])
        assert args.model == "claude-3"
        assert args.verbose is True
        
        # Test invalid arguments
        with pytest.raises(SystemExit):
            parse_arguments(["--invalid-option"])
    
    # Template 3: Configuration Validation
    def test_config_validation_valid(self):
        """Test configuration validation with valid data"""
        valid_config = {
            "api_key": "valid-key-123",
            "model": "claude-3-sonnet",
            "timeout": 30
        }
        
        result = validate_config(valid_config)
        assert result is True
    
    def test_config_validation_invalid(self):
        """Test configuration validation with invalid data"""
        invalid_configs = [
            {},  # Empty config
            {"api_key": ""},  # Empty API key
            {"api_key": "valid", "timeout": -1},  # Invalid timeout
            {"api_key": "valid", "model": "invalid-model"}  # Invalid model
        ]
        
        for config in invalid_configs:
            with pytest.subTest(config=config):
                result = validate_config(config)
                assert result is False
    
    # Template 4: File Processing
    def test_file_processing_leaf_node(self):
        """Test file processing leaf node"""
        test_content = "line1\nline2\nline3"
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            result = process_file_content("test.txt")
            expected = ["line1", "line2", "line3"]
            assert result == expected
    
    # Template 5: Error Handling
    def test_error_formatter_leaf_node(self):
        """Test error formatting leaf node"""
        error_msg = format_error("ValidationError", "Invalid input")
        
        assert "ValidationError" in error_msg
        assert "Invalid input" in error_msg
        assert len(error_msg) > 20  # Reasonable length
    
    # Template 6: Data Transformation
    @pytest.mark.parametrize("input_data,expected", [
        ({"key": "value"}, "key=value"),
        ({}, ""),
        ({"a": 1, "b": 2}, "a=1&b=2"),
    ])
    def test_data_transformer(self, input_data, expected):
        """Test data transformation leaf node"""
        result = transform_data(input_data)
        assert result == expected
```

### **Java Leaf Node Test Template**

```java
/**
 * Template for Java leaf node testing
 * Copy and modify for your specific use case
 */
package com.yourpackage.test;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Method;
import java.util.stream.Stream;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
@DisplayName("Your Leaf Node Tests")
public class YourLeafNodeTest {

    private ByteArrayOutputStream outputStream;
    private PrintStream originalOut;

    @BeforeEach
    void setUp() {
        outputStream = new ByteArrayOutputStream();
        originalOut = System.out;
        System.setOut(new PrintStream(outputStream));
    }

    @AfterEach
    void tearDown() {
        System.setOut(originalOut);
    }

    @Nested
    @DisplayName("Pure Function Leaf Nodes")
    class PureFunctionTests {

        @Test
        @DisplayName("Pure function - happy path")
        void testPureFunctionHappyPath() {
            // Arrange
            String input = "test_input";
            String expected = "expected_output";

            // Act
            String result = YourClass.pureFunctionLeafNode(input);

            // Assert
            assertEquals(expected, result);
        }

        @ParameterizedTest
        @ValueSource(strings = {"", "   ", "null", "very_long_input_string"})
        @DisplayName("Pure function - edge cases")
        void testPureFunctionEdgeCases(String input) {
            // Act & Assert
            assertDoesNotThrow(() -> {
                String result = YourClass.pureFunctionLeafNode(input);
                assertNotNull(result);
            });
        }

        @Test
        @DisplayName("Pure function - error conditions")
        void testPureFunctionErrorConditions() {
            // Test null input
            assertThrows(IllegalArgumentException.class, () -> {
                YourClass.pureFunctionLeafNode(null);
            });

            // Test invalid input
            assertThrows(ValidationException.class, () -> {
                YourClass.pureFunctionLeafNode("invalid_input");
            });
        }
    }

    @Nested
    @DisplayName("Builder Pattern Leaf Nodes")
    class BuilderPatternTests {

        @Test
        @DisplayName("Builder method - fluent API")
        void testBuilderFluentAPI() {
            // Arrange
            YourClass.Builder builder = new YourClass.Builder();

            // Act
            YourClass.Builder result = builder.setParameter("value");

            // Assert
            assertSame(builder, result, "Should return same builder instance");
            assertDoesNotThrow(() -> result.build());
        }

        @Test
        @DisplayName("Builder validation - parameter validation")
        void testBuilderParameterValidation() {
            YourClass.Builder builder = new YourClass.Builder();

            // Test valid parameters
            assertDoesNotThrow(() -> {
                builder.setParameter("valid_value").build();
            });

            // Test invalid parameters
            assertThrows(IllegalArgumentException.class, () -> {
                builder.setParameter(null).build();
            });

            assertThrows(ValidationException.class, () -> {
                builder.setParameter("").build();
            });
        }

        @Test
        @DisplayName("Builder reusability")
        void testBuilderReusability() {
            YourClass.Builder builder = new YourClass.Builder()
                .setParameter("base_value");

            // Create multiple objects from same builder
            YourClass obj1 = builder.setParameter("value1").build();
            YourClass obj2 = builder.setParameter("value2").build();

            assertNotNull(obj1);
            assertNotNull(obj2);
            assertNotSame(obj1, obj2);
        }
    }

    @Nested
    @DisplayName("Validation Leaf Nodes")
    class ValidationTests {

        private static Stream<Arguments> validationTestCases() {
            return Stream.of(
                Arguments.of("valid_input", true),
                Arguments.of("", false),
                Arguments.of(null, false),
                Arguments.of("invalid_format", false),
                Arguments.of("very_long_input_that_exceeds_limits", false)
            );
        }

        @ParameterizedTest
        @MethodSource("validationTestCases")
        @DisplayName("Input validation leaf node")
        void testInputValidation(String input, boolean expectedValid) {
            // Act
            boolean result = YourClass.isValidInput(input);

            // Assert
            assertEquals(expectedValid, result);
        }

        @Test
        @DisplayName("Complex validation - multiple criteria")
        void testComplexValidation() {
            // Test object that meets all criteria
            ValidatedObject validObj = new ValidatedObject("valid", 42, true);
            assertTrue(YourClass.isValidObject(validObj));

            // Test object that fails different criteria
            ValidatedObject invalidObj1 = new ValidatedObject("", 42, true);
            assertFalse(YourClass.isValidObject(invalidObj1));

            ValidatedObject invalidObj2 = new ValidatedObject("valid", -1, true);
            assertFalse(YourClass.isValidObject(invalidObj2));
        }
    }

    @Nested
    @DisplayName("Private Method Testing")
    class PrivateMethodTests {

        @Test
        @DisplayName("Private leaf node via reflection")
        void testPrivateLeafNode() throws Exception {
            // Arrange
            YourClass instance = new YourClass("test");
            Method privateMethod = YourClass.class
                .getDeclaredMethod("privateLeafNode", String.class);
            privateMethod.setAccessible(true);

            // Act
            String result = (String) privateMethod.invoke(instance, "input");

            // Assert
            assertEquals("expected_output", result);
        }

        @Test
        @DisplayName("Private validation method")
        void testPrivateValidation() throws Exception {
            // Arrange
            Method validationMethod = YourClass.class
                .getDeclaredMethod("validateInternal", Object.class);
            validationMethod.setAccessible(true);

            // Act & Assert
            assertTrue((Boolean) validationMethod.invoke(null, "valid_input"));
            assertFalse((Boolean) validationMethod.invoke(null, "invalid_input"));
        }
    }

    @Nested
    @DisplayName("Error Handling Leaf Nodes")
    class ErrorHandlingTests {

        @Test
        @DisplayName("Error message formatting")
        void testErrorMessageFormatting() {
            // Act
            String errorMsg = YourClass.formatError("TestError", "Test details");

            // Assert
            assertAll("Error message validation",
                () -> assertTrue(errorMsg.contains("TestError")),
                () -> assertTrue(errorMsg.contains("Test details")),
                () -> assertTrue(errorMsg.length() > 10),
                () -> assertFalse(errorMsg.contains("null"))
            );
        }

        @Test
        @DisplayName("Error severity classification")
        void testErrorSeverityClassification() {
            assertEquals("INFO", YourClass.classifyError(200));
            assertEquals("WARNING", YourClass.classifyError(400));
            assertEquals("ERROR", YourClass.classifyError(500));
            assertEquals("CRITICAL", YourClass.classifyError(600));
        }
    }

    @Nested
    @DisplayName("Performance and Resource Tests")
    class PerformanceTests {

        @Test
        @DisplayName("Leaf node performance")
        void testLeafNodePerformance() {
            // Measure execution time
            long startTime = System.nanoTime();
            
            for (int i = 0; i < 1000; i++) {
                YourClass.performantLeafNode("test_input_" + i);
            }
            
            long endTime = System.nanoTime();
            long durationMs = (endTime - startTime) / 1_000_000;
            
            // Assert reasonable performance (adjust threshold as needed)
            assertTrue(durationMs < 100, "Should complete 1000 operations in < 100ms");
        }

        @Test
        @DisplayName("Memory usage validation")
        void testMemoryUsage() {
            // Test that leaf node doesn't leak memory
            Runtime runtime = Runtime.getRuntime();
            long initialMemory = runtime.totalMemory() - runtime.freeMemory();
            
            // Execute leaf node many times
            for (int i = 0; i < 10000; i++) {
                YourClass.memoryEfficientLeafNode("test_" + i);
            }
            
            // Force garbage collection
            System.gc();
            long finalMemory = runtime.totalMemory() - runtime.freeMemory();
            
            // Memory increase should be reasonable
            long memoryIncrease = finalMemory - initialMemory;
            assertTrue(memoryIncrease < 1_000_000, "Memory increase should be < 1MB");
        }
    }
}
```

## 🔧 **Advanced Testing Patterns**

### **Pattern 1: State Machine Testing**

```python
class TestStateMachineLeafNodes:
    """Test state transitions as leaf nodes"""
    
    def test_state_transition_valid(self):
        """Test valid state transitions"""
        transitions = [
            ("INIT", "START", "RUNNING"),
            ("RUNNING", "PAUSE", "PAUSED"),
            ("PAUSED", "RESUME", "RUNNING"),
            ("RUNNING", "STOP", "STOPPED")
        ]
        
        for current, action, expected in transitions:
            result = transition_state(current, action)
            assert result == expected
    
    def test_state_transition_invalid(self):
        """Test invalid state transitions"""
        invalid_transitions = [
            ("INIT", "PAUSE"),  # Can't pause before starting
            ("STOPPED", "RESUME"),  # Can't resume when stopped
            ("RUNNING", "START")  # Can't start when already running
        ]
        
        for current, action in invalid_transitions:
            with pytest.raises(InvalidTransitionError):
                transition_state(current, action)
```

### **Pattern 2: Configuration Matrix Testing**

```java
@ParameterizedTest
@CsvSource({
    "true, true, true, FULL_FEATURES",
    "true, true, false, BASIC_FEATURES", 
    "true, false, false, MINIMAL_FEATURES",
    "false, false, false, NO_FEATURES"
})
@DisplayName("Configuration matrix testing")
void testConfigurationMatrix(boolean feature1, boolean feature2, 
                            boolean feature3, String expectedMode) {
    // Arrange
    Configuration config = new Configuration(feature1, feature2, feature3);
    
    // Act
    String mode = determineOperationMode(config);
    
    // Assert
    assertEquals(expectedMode, mode);
}
```

### **Pattern 3: Time-Based Testing**

```python
class TestTimeBasedLeafNodes:
    """Test time-dependent leaf nodes"""
    
    @patch('time.time')
    def test_timestamp_generation(self, mock_time):
        """Test timestamp generation leaf node"""
        # Arrange
        mock_time.return_value = 1609459200  # 2021-01-01 00:00:00 UTC
        
        # Act
        timestamp = generate_timestamp()
        
        # Assert
        assert timestamp == "2021-01-01T00:00:00Z"
    
    @patch('datetime.datetime')
    def test_expiry_calculation(self, mock_datetime):
        """Test expiry calculation leaf node"""
        # Arrange
        base_time = datetime(2021, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = base_time
        
        # Act
        expiry = calculate_expiry(hours=24)
        
        # Assert
        expected = datetime(2021, 1, 2, 12, 0, 0)
        assert expiry == expected
```

### **Pattern 4: Resource Cleanup Testing**

```java
@Test
@DisplayName("Resource cleanup leaf node")
void testResourceCleanup() {
    // Arrange
    List<Resource> resources = Arrays.asList(
        mock(Resource.class),
        mock(Resource.class),
        mock(Resource.class)
    );
    
    // Act
    cleanupResources(resources);
    
    // Assert
    for (Resource resource : resources) {
        verify(resource).close();
        verify(resource).cleanup();
    }
}

@Test
@DisplayName("Resource cleanup with exceptions")
void testResourceCleanupWithExceptions() {
    // Arrange
    Resource failingResource = mock(Resource.class);
    doThrow(new RuntimeException("Cleanup failed")).when(failingResource).close();
    
    Resource normalResource = mock(Resource.class);
    List<Resource> resources = Arrays.asList(failingResource, normalResource);
    
    // Act & Assert
    assertDoesNotThrow(() -> cleanupResources(resources));
    
    // Verify normal resource was still cleaned up
    verify(normalResource).close();
}
```

## 📊 **Quality Metrics Templates**

### **Coverage Analysis Script**

```python
#!/usr/bin/env python3
"""
Leaf node coverage analysis script
"""
import json
import sys
from pathlib import Path

def analyze_leaf_node_coverage(coverage_file: str) -> dict:
    """Analyze coverage specifically for leaf nodes"""
    
    with open(coverage_file, 'r') as f:
        coverage_data = json.load(f)
    
    leaf_node_patterns = [
        '*_leaf_node*',
        '*validator*',
        '*parser*',
        '*formatter*',
        '*builder*'
    ]
    
    leaf_coverage = {}
    total_lines = 0
    covered_lines = 0
    
    for file_path, file_data in coverage_data['files'].items():
        if any(pattern in file_path.lower() for pattern in leaf_node_patterns):
            lines = file_data['summary']['num_statements']
            covered = file_data['summary']['covered_lines']
            
            leaf_coverage[file_path] = {
                'lines': lines,
                'covered': covered,
                'percentage': (covered / lines * 100) if lines > 0 else 0
            }
            
            total_lines += lines
            covered_lines += covered
    
    overall_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
    
    return {
        'overall_coverage': overall_percentage,
        'total_lines': total_lines,
        'covered_lines': covered_lines,
        'file_coverage': leaf_coverage,
        'quality_score': calculate_quality_score(overall_percentage)
    }

def calculate_quality_score(coverage_percentage: float) -> str:
    """Calculate quality score based on coverage"""
    if coverage_percentage >= 95:
        return "⭐⭐⭐⭐⭐ EXCEPTIONAL"
    elif coverage_percentage >= 85:
        return "⭐⭐⭐⭐ EXCELLENT"
    elif coverage_percentage >= 75:
        return "⭐⭐⭐ GOOD"
    elif coverage_percentage >= 60:
        return "⭐⭐ FAIR"
    else:
        return "⭐ NEEDS IMPROVEMENT"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_coverage.py <coverage.json>")
        sys.exit(1)
    
    results = analyze_leaf_node_coverage(sys.argv[1])
    
    print(f"🎯 Leaf Node Coverage Analysis")
    print(f"Overall Coverage: {results['overall_coverage']:.1f}%")
    print(f"Quality Score: {results['quality_score']}")
    print(f"Lines Covered: {results['covered_lines']}/{results['total_lines']}")
    
    print("\n📊 File Breakdown:")
    for file_path, data in results['file_coverage'].items():
        print(f"  {file_path}: {data['percentage']:.1f}% ({data['covered']}/{data['lines']})")
```

### **Performance Benchmark Template**

```java
/**
 * Performance benchmark template for leaf nodes
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
public class LeafNodeBenchmark {
    
    private String testData;
    private List<String> testDataList;
    
    @Setup
    public void setup() {
        testData = "benchmark_test_data_" + System.currentTimeMillis();
        testDataList = IntStream.range(0, 1000)
            .mapToObj(i -> "test_data_" + i)
            .collect(Collectors.toList());
    }
    
    @Benchmark
    public String benchmarkStringProcessingLeafNode() {
        return YourClass.processString(testData);
    }
    
    @Benchmark
    public List<String> benchmarkListProcessingLeafNode() {
        return YourClass.processList(testDataList);
    }
    
    @Benchmark
    public boolean benchmarkValidationLeafNode() {
        return YourClass.validateInput(testData);
    }
    
    public static void main(String[] args) throws Exception {
        Options opt = new OptionsBuilder()
            .include(LeafNodeBenchmark.class.getSimpleName())
            .forks(1)
            .warmupIterations(3)
            .measurementIterations(5)
            .build();
            
        new Runner(opt).run();
    }
}
```

## 🎯 **Quick Start Checklist**

### **Project Setup**
- [ ] Choose testing framework (pytest/JUnit)
- [ ] Set up coverage reporting
- [ ] Configure CI/CD pipeline
- [ ] Create test directory structure

### **Identification Phase**
- [ ] Map system architecture
- [ ] Identify entry points
- [ ] Find utility functions
- [ ] Classify by dependency level
- [ ] Prioritize by importance

### **Implementation Phase**
- [ ] Start with highest-scoring leaf nodes
- [ ] Write tests before refactoring
- [ ] Focus on happy path first
- [ ] Add edge cases and error handling
- [ ] Optimize for speed and clarity

### **Quality Assurance**
- [ ] Achieve 90%+ coverage target
- [ ] Ensure tests run in < 100ms each
- [ ] Verify deterministic behavior
- [ ] Document patterns and decisions
- [ ] Set up monitoring and alerts

---

**🚀 These templates are battle-tested from our successful implementations achieving 95+ quality scores. Adapt them to your specific needs and start building robust, well-tested leaf nodes today!**
