import com.anthropic.claude.agent.core.Agent;
import com.anthropic.claude.agent.core.AgentConfig;
import com.anthropic.claude.agent.core.AgentResponse;
import com.anthropic.claude.agent.tools.Tool;
import com.anthropic.claude.agent.tools.ToolRegistry;

import java.util.ArrayList;
import java.util.List;

/**
 * Quick start example for Claude Agent Framework (Java).
 */
public class QuickStartExample {
    
    public static void main(String[] args) {
        System.out.println("Claude Agent Framework - Java Quick Start");
        System.out.println("=" + "=".repeat(40));
        
        // Check for API key
        String apiKey = System.getenv("ANTHROPIC_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            System.out.println("Please set ANTHROPIC_API_KEY environment variable");
            return;
        }
        
        try {
            // Create configuration
            AgentConfig config = AgentConfig.builder()
                    .name("quick-start-agent")
                    .systemPrompt("You are a helpful AI assistant. Be concise and friendly.")
                    .verbose(true)
                    .apiKey(apiKey)
                    .build();
            
            // Get some tools
            ToolRegistry registry = new ToolRegistry();
            registry.discoverTools();
            
            List<Tool> tools = new ArrayList<>();
            tools.add(registry.getTool("think"));
            tools.add(registry.getTool("file_read"));
            
            // Create agent with specific tools
            Agent agent = new Agent(config, tools);
            
            System.out.println("Created agent: " + agent.getConfig().getName());
            System.out.print("Available tools: ");
            System.out.println(agent.getToolNames());
            
            // Test a simple interaction
            System.out.println("\nTesting agent interaction...");
            
            AgentResponse response = agent.chatSync("Hello! Can you tell me what tools you have available?");
            
            System.out.println("Agent: " + response.getTextContent());
            
            // Clean up
            agent.close();
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            System.err.println("Note: This requires a valid ANTHROPIC_API_KEY");
            e.printStackTrace();
        }
    }
}
