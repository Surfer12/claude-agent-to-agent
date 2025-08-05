package examples;

import com.anthropic.api.AnthropicClientEnhanced;
import com.anthropic.api.tools.AnthropicTools;
import com.anthropic.api.cli.CognitiveAgentCLI;

import java.util.Arrays;
import java.util.List;

/**
 * Basic usage example for the Anthropic API Client Library
 * 
 * This example demonstrates how to use the basic features of the library
 * including message creation, tool usage, and CLI interaction.
 */
public class BasicUsageExample {
    
    public static void main(String[] args) {
        System.out.println("🚀 Anthropic API Client Library - Basic Usage Examples");
        System.out.println("=" .repeat(60));
        System.out.println();
        
        // Check if API key is set
        String apiKey = System.getenv("ANTHROPIC_API_KEY");
        if (apiKey == null || apiKey.trim().isEmpty()) {
            System.out.println("⚠️ Warning: ANTHROPIC_API_KEY environment variable not set.");
            System.out.println("Some examples may not work without a valid API key.");
            System.out.println();
        }
        
        try {
            basicMessageExample(apiKey);
            toolUsageExample(apiKey);
            multipleToolsExample(apiKey);
            clientConfigurationExample(apiKey);
            cliExample(apiKey);
            
            System.out.println("✅ All examples completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Error running examples: " + e.getMessage());
            System.err.println("Make sure you have a valid API key and internet connection.");
            e.printStackTrace();
        }
    }
    
    /**
     * Example of creating a basic message
     */
    private static void basicMessageExample(String apiKey) {
        System.out.println("🤖 Basic Message Example");
        System.out.println("=" .repeat(50));
        
        AnthropicClientEnhanced client = AnthropicClientEnhanced.createBasicClient(apiKey);
        
        List<AnthropicClientEnhanced.Message> messages = Arrays.asList(
            new AnthropicClientEnhanced.Message("", "user", Arrays.asList(
                new AnthropicClientEnhanced.Content("text", "Hello, Claude! Can you tell me a short joke?")
            ))
        );
        
        AnthropicClientEnhanced.Message response = client.createMessage(messages, null, null);
        
        System.out.println("Response:");
        for (AnthropicClientEnhanced.Content content : response.getContent()) {
            if ("text".equals(content.getType())) {
                System.out.println(content.getText());
            }
        }
        System.out.println();
    }
    
    /**
     * Example of using tools
     */
    private static void toolUsageExample(String apiKey) {
        System.out.println("🛠️ Tool Usage Example");
        System.out.println("=" .repeat(50));
        
        AnthropicClientEnhanced client = AnthropicClientEnhanced.createBasicClient(apiKey);
        
        // Example with bash tool
        System.out.println("Using bash tool to list files:");
        List<AnthropicClientEnhanced.Message> messages = Arrays.asList(
            new AnthropicClientEnhanced.Message("", "user", Arrays.asList(
                new AnthropicClientEnhanced.Content("text", "List all Java files in the current directory")
            ))
        );
        
        AnthropicClientEnhanced.Message response = client.createMessage(
            messages,
            Arrays.asList("bash"),
            null
        );
        
        System.out.println("Response:");
        for (AnthropicClientEnhanced.Content content : response.getContent()) {
            if ("text".equals(content.getType())) {
                System.out.println(content.getText());
            }
        }
        System.out.println();
    }
    
    /**
     * Example of using multiple tools
     */
    private static void multipleToolsExample(String apiKey) {
        System.out.println("🔧 Multiple Tools Example");
        System.out.println("=" .repeat(50));
        
        AnthropicClientEnhanced client = AnthropicClientEnhanced.createBasicClient(apiKey);
        
        // Example with multiple tools
        List<AnthropicClientEnhanced.Message> messages = Arrays.asList(
            new AnthropicClientEnhanced.Message("", "user", Arrays.asList(
                new AnthropicClientEnhanced.Content("text", "What's the weather like and can you search for the latest AI news?")
            ))
        );
        
        AnthropicClientEnhanced.Message response = client.createMessage(
            messages,
            Arrays.asList("weather", "web_search"),
            null
        );
        
        System.out.println("Response:");
        for (AnthropicClientEnhanced.Content content : response.getContent()) {
            if ("text".equals(content.getType())) {
                System.out.println(content.getText());
            }
        }
        System.out.println();
    }
    
    /**
     * Example of client configuration
     */
    private static void clientConfigurationExample(String apiKey) {
        System.out.println("⚙️ Client Configuration Example");
        System.out.println("=" .repeat(50));
        
        // Create client with custom configuration
        AnthropicClientEnhanced client = new AnthropicClientEnhanced.Builder()
            .apiKey(apiKey)
            .model("claude-sonnet-4-20250514")
            .maxTokens(2048)
            .build();
        
        // Get available tools
        List<String> availableTools = client.getAvailableTools();
        System.out.println("Available tools: " + availableTools);
        
        // Get specific tool configuration
        client.getToolConfig("bash").ifPresent(config -> {
            System.out.println("Bash tool config: " + config.getName() + " - " + config.getDescription());
        });
        System.out.println();
    }
    
    /**
     * Example of CLI usage
     */
    private static void cliExample(String apiKey) {
        System.out.println("🖥️ CLI Example");
        System.out.println("=" .repeat(50));
        
        // Create CLI with specific configuration
        CognitiveAgentCLI cli = new CognitiveAgentCLI.Builder()
            .name("ExampleAgent")
            .systemPrompt("You are a helpful assistant that provides concise answers.")
            .tools(java.util.Set.of("bash", "web_search"))
            .verbose(true)
            .client(AnthropicClientEnhanced.createBasicClient(apiKey))
            .build();
        
        // Run a single query
        String response = cli.runSingleQuery("What is the capital of France?");
        System.out.println("CLI Response:");
        System.out.println(response);
        System.out.println();
        
        // Show metrics
        CognitiveAgentCLI.Metrics metrics = cli.getMetrics();
        System.out.println("Metrics:");
        System.out.println("- Total interactions: " + metrics.getTotalInteractions());
        System.out.println("- Successful interactions: " + metrics.getSuccessfulInteractions());
        System.out.println("- Average response time: " + String.format("%.2f", metrics.getAverageResponseTime()) + " ms");
        System.out.println();
    }
    
    /**
     * Example of tool creation and usage
     */
    private static void toolCreationExample() {
        System.out.println("🔨 Tool Creation Example");
        System.out.println("=" .repeat(50));
        
        // Create individual tools
        AnthropicTools.BashTool bashTool = AnthropicTools.createBashTool();
        AnthropicTools.WebSearchTool webSearchTool = AnthropicTools.createWebSearchTool(3);
        AnthropicTools.WeatherTool weatherTool = AnthropicTools.createWeatherTool();
        
        System.out.println("Created tools:");
        System.out.println("- Bash tool: " + bashTool.getDefinition().getName());
        System.out.println("- Web search tool: " + webSearchTool.getDefinition().getName() + 
                          " (max uses: " + webSearchTool.getDefinition().getMaxUses() + ")");
        System.out.println("- Weather tool: " + weatherTool.getDefinition().getName());
        
        // Convert tools to maps for API usage
        List<AnthropicTools.BaseTool> tools = Arrays.asList(bashTool, webSearchTool, weatherTool);
        List<java.util.Map<String, Object>> toolMaps = AnthropicTools.toolsToMaps(tools);
        
        System.out.println("Tool maps created: " + toolMaps.size());
        System.out.println();
    }
} 