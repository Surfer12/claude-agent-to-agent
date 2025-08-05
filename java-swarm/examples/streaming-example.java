// Example: Streaming responses with Java Swarm
// This example demonstrates how to use streaming responses programmatically

import com.swarm.core.Swarm;
import com.swarm.types.*;
import java.util.*;

public class StreamingExample {
    
    public static void main(String[] args) {
        // Check for API key
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.trim().isEmpty()) {
            System.err.println("Please set OPENAI_API_KEY environment variable");
            System.exit(1);
        }
        
        // Create Swarm instance
        Swarm swarm = new Swarm(apiKey);
        
        // Test connection
        if (!swarm.testConnection()) {
            System.err.println("Could not connect to OpenAI API. Please check your API key.");
            System.exit(1);
        }
        
        // Create a simple agent
        Agent agent = Agent.builder()
                .name("StreamingBot")
                .model("gpt-4o")
                .instructions("You are a helpful assistant that provides detailed explanations. Be verbose and informative.")
                .build();
        
        // Create initial message
        List<Map<String, Object>> messages = Arrays.asList(
            Map.of("role", "user", "content", "Explain how machine learning works in simple terms")
        );
        
        System.out.println("=== Streaming Response Example ===");
        System.out.println("Question: Explain how machine learning works in simple terms");
        System.out.println();
        
        try {
            // Run with streaming
            StringBuilder fullResponse = new StringBuilder();
            String currentSender = null;
            
            swarm.runAndStream(agent, messages, new HashMap<>(), null, false, 10, true)
                .blockingSubscribe(
                    event -> {
                        String eventType = (String) event.get("type");
                        
                        switch (eventType) {
                            case "delimiter":
                                @SuppressWarnings("unchecked")
                                Map<String, Object> delimData = (Map<String, Object>) event.get("data");
                                String delim = (String) delimData.get("delim");
                                
                                if ("start".equals(delim)) {
                                    System.out.println("🔄 Starting response...");
                                } else if ("end".equals(delim)) {
                                    System.out.println("\n✅ Response complete!");
                                }
                                break;
                                
                            case "delta":
                                @SuppressWarnings("unchecked")
                                Map<String, Object> deltaData = (Map<String, Object>) event.get("data");
                                String content = (String) deltaData.get("content");
                                String sender = (String) deltaData.get("sender");
                                
                                if (sender != null && !sender.equals(currentSender)) {
                                    if (fullResponse.length() > 0) {
                                        System.out.println();
                                    }
                                    System.out.print(sender + ": ");
                                    currentSender = sender;
                                }
                                
                                if (content != null) {
                                    System.out.print(content);
                                    System.out.flush();
                                    fullResponse.append(content);
                                }
                                break;
                                
                            case "response":
                                @SuppressWarnings("unchecked")
                                Map<String, Object> responseData = (Map<String, Object>) event.get("data");
                                
                                System.out.println("\n");
                                System.out.println("=== Final Response Summary ===");
                                System.out.println("Total characters: " + fullResponse.length());
                                System.out.println("Agent: " + responseData.get("agent"));
                                
                                @SuppressWarnings("unchecked")
                                Map<String, Object> contextVars = (Map<String, Object>) responseData.get("context_variables");
                                if (!contextVars.isEmpty()) {
                                    System.out.println("Context variables: " + contextVars);
                                }
                                break;
                        }
                    },
                    error -> {
                        System.err.println("Streaming error: " + error.getMessage());
                        error.printStackTrace();
                    },
                    () -> {
                        System.out.println("🎉 Streaming completed successfully!");
                    }
                );
                
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Clean up resources
            swarm.close();
        }
    }
}

/*
Expected output:
=== Streaming Response Example ===
Question: Explain how machine learning works in simple terms

🔄 Starting response...
StreamingBot: Machine learning is like teaching a computer to recognize patterns and make predictions, similar to how humans learn from experience...
[content streams in real-time]
✅ Response complete!

=== Final Response Summary ===
Total characters: 1247
Agent: StreamingBot
🎉 Streaming completed successfully!
*/
