// Example: Basic usage of Java Swarm programmatically
// This is a standalone example showing how to use the Swarm API

import com.swarm.core.Swarm;
import com.swarm.types.*;
import java.util.*;

public class BasicUsageExample {
    
    public static void main(String[] args) {
        // Check for API key
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.trim().isEmpty()) {
            System.err.println("Please set OPENAI_API_KEY environment variable");
            System.exit(1);
        }
        
        // Create Swarm instance
        Swarm swarm = new Swarm(apiKey);
        
        // Create a simple agent
        Agent agent = Agent.builder()
                .name("Assistant")
                .model("gpt-4o")
                .instructions("You are a helpful assistant that can perform calculations.")
                .addFunction(new SimpleCalculator())
                .build();
        
        // Create initial message
        List<Map<String, Object>> messages = Arrays.asList(
            Map.of("role", "user", "content", "Can you calculate 25 * 4 + 10?")
        );
        
        try {
            // Run the conversation
            Response response = swarm.run(agent, messages);
            
            // Print the conversation
            System.out.println("Conversation:");
            for (Map<String, Object> message : response.getMessages()) {
                String role = (String) message.get("role");
                String content = (String) message.get("content");
                String sender = (String) message.get("sender");
                
                if ("user".equals(role)) {
                    System.out.println("User: " + content);
                } else if ("assistant".equals(role) && content != null) {
                    System.out.println((sender != null ? sender : "Assistant") + ": " + content);
                } else if ("tool".equals(role)) {
                    System.out.println("Tool result: " + content);
                }
            }
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // Simple calculator function
    static class SimpleCalculator implements AgentFunction {
        
        @Override
        public Object execute(Map<String, Object> args) {
            String expression = (String) args.get("expression");
            
            try {
                // Simple evaluation (in real code, use a proper expression evaluator)
                // This is just for demonstration
                if (expression.matches("[0-9+\\-*/().\\s]+")) {
                    // Use JavaScript engine for evaluation
                    javax.script.ScriptEngineManager manager = new javax.script.ScriptEngineManager();
                    javax.script.ScriptEngine engine = manager.getEngineByName("JavaScript");
                    Object result = engine.eval(expression);
                    return "The result is: " + result;
                } else {
                    return "Invalid expression. Only numbers and basic operators allowed.";
                }
            } catch (Exception e) {
                return "Error calculating: " + e.getMessage();
            }
        }
        
        @Override
        public String getName() {
            return "calculate";
        }
        
        @Override
        public String getDescription() {
            return "Calculate a mathematical expression";
        }
        
        @Override
        public Map<String, Object> getParameterSchema() {
            return Map.of(
                "type", "object",
                "properties", Map.of(
                    "expression", Map.of(
                        "type", "string",
                        "description", "Mathematical expression to evaluate"
                    )
                ),
                "required", new String[]{"expression"}
            );
        }
    }
}
