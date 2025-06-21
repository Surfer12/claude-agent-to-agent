package com.anthropic.claude.agent.cli;

import com.anthropic.claude.agent.core.Agent;
import com.anthropic.claude.agent.core.AgentConfig;
import com.anthropic.claude.agent.core.AgentResponse;
import com.anthropic.claude.agent.tools.Tool;
import com.anthropic.claude.agent.tools.ToolRegistry;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.Callable;

/**
 * Main CLI class for Claude Agent Framework.
 */
@Command(
    name = "claude-agent",
    description = "Claude Agent Framework CLI - A comprehensive framework for building Claude-powered agents",
    version = "1.0.0",
    mixinStandardHelpOptions = true,
    subcommands = {
        ClaudeAgentCLI.ChatCommand.class,
        ClaudeAgentCLI.InteractiveCommand.class,
        ClaudeAgentCLI.ListToolsCommand.class,
        ClaudeAgentCLI.ToolInfoCommand.class,
        ClaudeAgentCLI.GenerateConfigCommand.class
    }
)
public class ClaudeAgentCLI implements Callable<Integer> {
    
    @Option(names = {"-c", "--config"}, description = "Configuration file path")
    private File configFile;
    
    @Option(names = {"-v", "--verbose"}, description = "Enable verbose output")
    private boolean verbose;
    
    private AgentConfig globalConfig;
    
    @Override
    public Integer call() throws Exception {
        // Load global configuration
        if (configFile != null) {
            globalConfig = AgentConfig.fromFile(configFile.getAbsolutePath());
        } else {
            globalConfig = loadDefaultConfig();
        }
        
        if (verbose) {
            globalConfig.setVerbose(true);
        }
        
        // Show help if no subcommand
        CommandLine.usage(this, System.out);
        return 0;
    }
    
    private AgentConfig loadDefaultConfig() {
        // Try to load from default locations
        String[] defaultPaths = {
            "claude-agent.yaml",
            "claude-agent.yml",
            System.getProperty("user.home") + "/.claude-agent.yaml",
            System.getProperty("user.home") + "/.claude-agent.yml"
        };
        
        for (String path : defaultPaths) {
            File file = new File(path);
            if (file.exists()) {
                try {
                    return AgentConfig.fromFile(path);
                } catch (IOException e) {
                    System.err.println("Warning: Could not load config from " + path + ": " + e.getMessage());
                }
            }
        }
        
        return new AgentConfig();
    }
    
    /**
     * Chat command for single prompts or interactive sessions.
     */
    @Command(name = "chat", description = "Start a chat session with Claude")
    static class ChatCommand implements Callable<Integer> {
        
        @Option(names = {"-p", "--prompt"}, description = "Single prompt to send to the agent")
        private String prompt;
        
        @Option(names = {"-f", "--file"}, description = "Read prompt from file")
        private File promptFile;
        
        @Option(names = {"-t", "--tools"}, split = ",", description = "Enable specific tools (comma-separated)")
        private List<String> tools = new ArrayList<>();
        
        @Option(names = {"-m", "--model"}, description = "Claude model to use")
        private String model;
        
        @Option(names = {"--max-tokens"}, description = "Maximum tokens to generate")
        private Integer maxTokens;
        
        @Option(names = {"--temperature"}, description = "Sampling temperature")
        private Double temperature;
        
        @Option(names = {"--api-key"}, description = "Anthropic API key")
        private String apiKey;
        
        @CommandLine.ParentCommand
        private ClaudeAgentCLI parent;
        
        @Override
        public Integer call() throws Exception {
            AgentConfig config = parent.globalConfig;
            
            // Override config with command line options
            if (model != null) {
                config.getModelConfig().setModel(model);
            }
            if (maxTokens != null) {
                config.getModelConfig().setMaxTokens(maxTokens);
            }
            if (temperature != null) {
                config.getModelConfig().setTemperature(temperature);
            }
            if (apiKey != null) {
                config.setApiKey(apiKey);
            }
            
            // Get input
            String userInput;
            if (prompt != null) {
                userInput = prompt;
            } else if (promptFile != null) {
                userInput = Files.readString(promptFile.toPath());
            } else {
                // Interactive mode
                return runInteractiveSession(config, tools);
            }
            
            // Single prompt mode
            return runSinglePrompt(config, tools, userInput);
        }
        
        private Integer runSinglePrompt(AgentConfig config, List<String> toolNames, String prompt) {
            try {
                Agent agent = createAgentWithTools(config, toolNames);
                AgentResponse response = agent.chatSync(prompt);
                
                System.out.println(response.getTextContent());
                
                agent.close();
                return 0;
                
            } catch (Exception e) {
                System.err.println("Error: " + e.getMessage());
                return 1;
            }
        }
        
        private Integer runInteractiveSession(AgentConfig config, List<String> toolNames) {
            try {
                Agent agent = createAgentWithTools(config, toolNames);
                
                System.out.println("Starting interactive session with " + agent.getConfig().getName() + "...");
                System.out.println("Type 'exit' or 'quit' to end the session.");
                System.out.println("Type 'clear' to clear conversation history.");
                System.out.println("Type 'help' for more commands.");
                System.out.println("-".repeat(50));
                
                Scanner scanner = new Scanner(System.in);
                
                while (true) {
                    System.out.print("\nYou: ");
                    String input = scanner.nextLine();
                    
                    if (input.toLowerCase().matches("exit|quit")) {
                        System.out.println("Ending session.");
                        break;
                    }
                    
                    if (input.toLowerCase().equals("clear")) {
                        agent.clearHistory();
                        System.out.println("Conversation history cleared.");
                        continue;
                    }
                    
                    if (input.toLowerCase().equals("help")) {
                        System.out.println("Commands:");
                        System.out.println("  exit, quit - End the session");
                        System.out.println("  clear - Clear conversation history");
                        System.out.println("  help - Show this help");
                        continue;
                    }
                    
                    if (input.trim().isEmpty()) {
                        continue;
                    }
                    
                    try {
                        AgentResponse response = agent.chatSync(input);
                        System.out.println("\nClaude: " + response.getTextContent());
                    } catch (Exception e) {
                        System.err.println("\nError: " + e.getMessage());
                    }
                }
                
                agent.close();
                return 0;
                
            } catch (Exception e) {
                System.err.println("Error: " + e.getMessage());
                return 1;
            }
        }
        
        private Agent createAgentWithTools(AgentConfig config, List<String> toolNames) throws Exception {
            ToolRegistry registry = new ToolRegistry();
            registry.discoverTools();
            
            List<Tool> tools = new ArrayList<>();
            
            // Determine which tools to enable
            if (toolNames.isEmpty()) {
                toolNames = config.getEnabledTools();
            }
            
            if (toolNames.contains("all")) {
                toolNames = registry.getAvailableTools();
            }
            
            // Create tool instances
            for (String toolName : toolNames) {
                try {
                    Map<String, Object> toolConfig = getToolConfig(config, toolName);
                    Tool tool = registry.getTool(toolName, toolConfig);
                    tools.add(tool);
                } catch (Exception e) {
                    if (config.isVerbose()) {
                        System.err.println("Warning: Could not load tool " + toolName + ": " + e.getMessage());
                    }
                }
            }
            
            return new Agent(config, tools);
        }
        
        private Map<String, Object> getToolConfig(AgentConfig config, String toolName) {
            Map<String, Object> allToolConfig = config.getToolConfig();
            if (allToolConfig.containsKey(toolName)) {
                Object toolConfig = allToolConfig.get(toolName);
                if (toolConfig instanceof Map) {
                    return (Map<String, Object>) toolConfig;
                }
            }
            return new HashMap<>();
        }
    }
    
    /**
     * Interactive command.
     */
    @Command(name = "interactive", description = "Start an interactive chat session")
    static class InteractiveCommand implements Callable<Integer> {
        
        @Option(names = {"-t", "--tools"}, split = ",", description = "Enable specific tools")
        private List<String> tools = new ArrayList<>();
        
        @CommandLine.ParentCommand
        private ClaudeAgentCLI parent;
        
        @Override
        public Integer call() throws Exception {
            ChatCommand chatCommand = new ChatCommand();
            chatCommand.parent = parent;
            chatCommand.tools = tools;
            return chatCommand.runInteractiveSession(parent.globalConfig, tools);
        }
    }
    
    /**
     * List tools command.
     */
    @Command(name = "list-tools", description = "List all available tools")
    static class ListToolsCommand implements Callable<Integer> {
        
        @Override
        public Integer call() throws Exception {
            ToolRegistry registry = new ToolRegistry();
            registry.discoverTools();
            
            List<String> tools = registry.getAvailableTools();
            
            if (tools.isEmpty()) {
                System.out.println("No tools available");
                return 0;
            }
            
            System.out.println("Available tools:");
            for (String toolName : tools) {
                try {
                    Map<String, Object> info = registry.getToolInfo(toolName);
                    System.out.println("  " + toolName + ": " + info.get("description"));
                } catch (Exception e) {
                    System.out.println("  " + toolName + ": Error loading tool - " + e.getMessage());
                }
            }
            
            return 0;
        }
    }
    
    /**
     * Tool info command.
     */
    @Command(name = "tool-info", description = "Show detailed information about a specific tool")
    static class ToolInfoCommand implements Callable<Integer> {
        
        @Parameters(index = "0", description = "Tool name")
        private String toolName;
        
        @Override
        public Integer call() throws Exception {
            ToolRegistry registry = new ToolRegistry();
            registry.discoverTools();
            
            try {
                Map<String, Object> info = registry.getToolInfo(toolName);
                
                System.out.println("Tool: " + info.get("name"));
                System.out.println("Description: " + info.get("description"));
                System.out.println("Class: " + info.get("class"));
                System.out.println("Supports files: " + info.get("supports_files"));
                
                if (info.containsKey("tool_version")) {
                    System.out.println("Tool version: " + info.get("tool_version"));
                }
                
                System.out.println("Input Schema:");
                System.out.println(info.get("input_schema"));
                
                return 0;
                
            } catch (Exception e) {
                System.err.println("Error: " + e.getMessage());
                return 1;
            }
        }
    }
    
    /**
     * Generate config command.
     */
    @Command(name = "generate-config", description = "Generate a sample configuration file")
    static class GenerateConfigCommand implements Callable<Integer> {
        
        @Option(names = {"-o", "--output"}, description = "Output file path")
        private File outputFile;
        
        @Override
        public Integer call() throws Exception {
            AgentConfig config = new AgentConfig();
            
            if (outputFile != null) {
                config.toFile(outputFile.getAbsolutePath());
                System.out.println("Configuration saved to: " + outputFile.getAbsolutePath());
            } else {
                // Print to stdout (simplified YAML)
                System.out.println("agent:");
                System.out.println("  name: " + config.getName());
                System.out.println("  system_prompt: \"" + config.getSystemPrompt() + "\"");
                System.out.println("  verbose: " + config.isVerbose());
                System.out.println("model:");
                System.out.println("  model: " + config.getModelConfig().getModel());
                System.out.println("  max_tokens: " + config.getModelConfig().getMaxTokens());
                System.out.println("  temperature: " + config.getModelConfig().getTemperature());
                System.out.println("tools:");
                System.out.println("  enabled:");
                for (String tool : config.getEnabledTools()) {
                    System.out.println("    - " + tool);
                }
            }
            
            return 0;
        }
    }
    
    /**
     * Main method.
     */
    public static void main(String[] args) {
        int exitCode = new CommandLine(new ClaudeAgentCLI()).execute(args);
        System.exit(exitCode);
    }
}
