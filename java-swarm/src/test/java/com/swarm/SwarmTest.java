package com.swarm;

import com.swarm.core.Swarm;
import com.swarm.types.Agent;
import com.swarm.types.AgentFunction;
import com.swarm.types.Response;
import com.swarm.types.Result;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Basic tests for the Java Swarm implementation.
 */
public class SwarmTest {

    private Agent testAgent;

    @BeforeEach
    void setUp() {
        testAgent = Agent.builder()
                .name("TestAgent")
                .model("gpt-4o")
                .instructions("You are a test agent.")
                .build();
    }

    @Test
    void testAgentCreation() {
        assertNotNull(testAgent);
        assertEquals("TestAgent", testAgent.getName());
        assertEquals("gpt-4o", testAgent.getModel());
        assertEquals("You are a test agent.", testAgent.getInstructions());
        assertTrue(testAgent.getFunctions().isEmpty());
        assertTrue(testAgent.isParallelToolCalls());
    }

    @Test
    void testAgentBuilder() {
        Agent agent = Agent.builder()
                .name("CustomAgent")
                .model("gpt-3.5-turbo")
                .instructions("Custom instructions")
                .parallelToolCalls(false)
                .build();

        assertEquals("CustomAgent", agent.getName());
        assertEquals("gpt-3.5-turbo", agent.getModel());
        assertEquals("Custom instructions", agent.getInstructions());
        assertFalse(agent.isParallelToolCalls());
    }

    @Test
    void testAgentWithFunction() {
        AgentFunction testFunction = new AgentFunction() {
            @Override
            public Object execute(Map<String, Object> args) {
                return "Test result";
            }

            @Override
            public String getName() {
                return "test_function";
            }

            @Override
            public String getDescription() {
                return "A test function";
            }
        };

        testAgent.getFunctions().add(testFunction);
        assertEquals(1, testAgent.getFunctions().size());
        assertEquals("test_function", testAgent.getFunctions().get(0).getName());
    }

    @Test
    void testResponseCreation() {
        List<Map<String, Object>> messages = Arrays.asList(
                Map.of("role", "user", "content", "Hello"),
                Map.of("role", "assistant", "content", "Hi there!")
        );

        Response response = Response.builder()
                .messages(messages)
                .agent(testAgent)
                .addContextVariable("key", "value")
                .build();

        assertEquals(2, response.getMessages().size());
        assertEquals(testAgent, response.getAgent());
        assertEquals("value", response.getContextVariables().get("key"));
    }

    @Test
    void testResultCreation() {
        Result result = Result.builder()
                .value("Test value")
                .agent(testAgent)
                .addContextVariable("test", "data")
                .build();

        assertEquals("Test value", result.getValue());
        assertEquals(testAgent, result.getAgent());
        assertEquals("data", result.getContextVariables().get("test"));
    }

    @Test
    void testAgentInstructionsAsString() {
        Map<String, Object> contextVars = Map.of("name", "John");
        
        // Test string instructions
        String instructions = testAgent.getInstructionsAsString(contextVars);
        assertEquals("You are a test agent.", instructions);
        
        // Test function instructions
        Agent dynamicAgent = Agent.builder()
                .name("DynamicAgent")
                .instructions(ctx -> "Hello " + ctx.get("name"))
                .build();
        
        String dynamicInstructions = dynamicAgent.getInstructionsAsString(contextVars);
        assertEquals("Hello John", dynamicInstructions);
    }

    // Note: Integration tests with actual OpenAI API calls would require API key
    // and should be in a separate test class or profile
}
