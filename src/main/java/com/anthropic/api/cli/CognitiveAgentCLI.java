package com.anthropic.api.cli;

import com.anthropic.api.AnthropicClientEnhanced;
import com.anthropic.api.tools.AnthropicTools;
import com.anthropic.api.tools.AnthropicTools.BaseTool;
import com.anthropic.api.processors.UPOFProcessor;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;
import java.util.logging.Level;
import java.util.logging.FileHandler;
import java.util.logging.SimpleFormatter;

/**
 * Command Line Interface for Anthropic API
 * 
 * This class provides a sophisticated CLI for interacting with Anthropic's Claude API,
 * including support for various tools and cognitive performance tracking.
 */
public final class CognitiveAgentCLI {
    private static final Logger LOGGER = Logger.getLogger(CognitiveAgentCLI.class.getName());
    
    private final String name;
    private final String systemPrompt;
    private final Set<String> tools;
    private final boolean verbose;
    private final String model;
    private final AnthropicClientEnhanced client;
    
    // Metrics tracking
    private final AtomicInteger totalInteractions = new AtomicInteger(0);
    private final AtomicInteger successfulInteractions = new AtomicInteger(0);
    private final AtomicLong totalResponseTime = new AtomicLong(0);
    private volatile LocalDateTime lastInteractionTime;

    private CognitiveAgentCLI(Builder builder) {
        this.name = builder.name;
        this.systemPrompt = builder.systemPrompt;
        this.tools = Collections.unmodifiableSet(builder.tools);
        this.verbose = builder.verbose;
        this.model = builder.model;
        this.client = builder.client;
        
        // Setup logging
        setupLogging();
    }

    /**
     * Builder for creating CognitiveAgentCLI instances
     */
    public static class Builder {
        private String name = "CognitiveAgent";
        private String systemPrompt = getDefaultSystemPrompt();
        private Set<String> tools = new HashSet<>(Arrays.asList("bash", "web_search", "weather"));
        private boolean verbose = false;
        private String model = "claude-3-5-sonnet-20240620";
        private AnthropicClientEnhanced client;

        public Builder() {
            // Initialize with default tools
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder systemPrompt(String systemPrompt) {
            this.systemPrompt = systemPrompt;
            return this;
        }

        public Builder tools(Set<String> tools) {
            this.tools = new HashSet<>(tools);
            return this;
        }

        public Builder addTool(String tool) {
            this.tools.add(tool);
            return this;
        }

        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder client(AnthropicClientEnhanced client) {
            this.client = client;
            return this;
        }

        public CognitiveAgentCLI build() {
            if (client == null) {
                throw new IllegalStateException("Client must be provided");
            }
            return new CognitiveAgentCLI(this);
        }

        private static String getDefaultSystemPrompt() {
            return "You are a cognitive enhancement agent designed to support " +
                   "interdisciplinary problem-solving and computational exploration. " +
                   "Approach each interaction with systematic analytical thinking, " +
                   "drawing insights from multiple domains of knowledge.";
        }
    }

    /**
     * Launch an interactive cognitive agent session
     */
    public void interactiveSession() {
        LOGGER.info("Initiating Cognitive Agent Session: " + name);
        
        Scanner scanner = new Scanner(System.in);
        
        try {
            while (true) {
                try {
                    System.out.print("\n🧠 Cognitive Agent > ");
                    String userInput = scanner.nextLine();
                    
                    if (userInput.toLowerCase().matches("exit|quit|q")) {
                        break;
                    }
                    
                    if (userInput.toLowerCase().startsWith("@publicationv1")) {
                        runUPOFMode(userInput.substring(14).trim());
                        continue;
                    }

                    if (userInput.toLowerCase().startsWith("@ninestep")) {
                        UPOFProcessor processor = new UPOFProcessor();
                        String ninestepResult = processor.applyNinestep(userInput.substring(10).trim());
                        System.out.println("Ninestep Process Result:\n" + ninestepResult);
                        LOGGER.info("Applied Ninestep with consciousness protection enabled");
                        continue;
                    }
                    
                    // Track interaction performance
                    long startTime = System.currentTimeMillis();
                    
                    String response = runSingleQuery(userInput);
                    
                    long endTime = System.currentTimeMillis();
                    long responseTime = endTime - startTime;
                    
                    // Update metrics
                    totalInteractions.incrementAndGet();
                    successfulInteractions.incrementAndGet();
                    totalResponseTime.addAndGet(responseTime);
                    lastInteractionTime = LocalDateTime.now();
                    
                    // Output response
                    System.out.println("\n🤖 Response:");
                    System.out.println(response);
                    
                } catch (Exception e) {
                    LOGGER.log(Level.SEVERE, "Interaction Error", e);
                    System.out.println("⚠️ Error in interaction: " + e.getMessage());
                    
                    // Update metrics for failed interaction
                    totalInteractions.incrementAndGet();
                    lastInteractionTime = LocalDateTime.now();
                }
            }
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Session Error", e);
        } finally {
            // Log session summary
            logSessionSummary();
        }
    }

    /**
     * Run a single query without entering interactive mode
     */
    public String runSingleQuery(String query) {
        try {
            List<AnthropicClientEnhanced.Message> messages = Arrays.asList(
                new AnthropicClientEnhanced.Message("", "system", Arrays.asList(
                    new AnthropicClientEnhanced.Content("text", systemPrompt)
                )),
                new AnthropicClientEnhanced.Message("", "user", Arrays.asList(
                    new AnthropicClientEnhanced.Content("text", query)
                ))
            );
            
            List<String> availableTools = new ArrayList<>(tools);
            
            AnthropicClientEnhanced.Message response = client.createMessage(
                messages,
                availableTools.isEmpty() ? null : availableTools,
                null
            );
            
            // Extract text content
            StringBuilder responseText = new StringBuilder();
            for (AnthropicClientEnhanced.Content content : response.getContent()) {
                if ("text".equals(content.getType())) {
                    responseText.append(content.getText());
                }
            }
            
            return responseText.toString();
            
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Query Error", e);
            return "Error processing query: " + e.getMessage();
        }
    }

    /**
     * Run a single query asynchronously
     */
    public CompletableFuture<String> runSingleQueryAsync(String query) {
        return CompletableFuture.supplyAsync(() -> runSingleQuery(query));
    }

    /**
     * Get current metrics
     */
    public Metrics getMetrics() {
        int total = totalInteractions.get();
        int successful = successfulInteractions.get();
        long totalTime = totalResponseTime.get();
        
        double averageResponseTime = total > 0 ? (double) totalTime / total : 0.0;
        
        return new Metrics(total, successful, averageResponseTime, lastInteractionTime);
    }

    /**
     * Get available tools
     */
    public Set<String> getAvailableTools() {
        return new HashSet<>(tools);
    }

    /**
     * Add a tool to the session
     */
    public void addTool(String tool) {
        if (AnthropicTools.getAvailableToolTypes().contains(tool)) {
            tools.add(tool);
        } else {
            throw new IllegalArgumentException("Unknown tool: " + tool);
        }
    }

    /**
     * Remove a tool from the session
     */
    public void removeTool(String tool) {
        tools.remove(tool);
    }

    private void setupLogging() {
        try {
            FileHandler fileHandler = new FileHandler("claude_agent_interactions.log", true);
            fileHandler.setFormatter(new SimpleFormatter());
            LOGGER.addHandler(fileHandler);
            LOGGER.setLevel(verbose ? Level.ALL : Level.INFO);
        } catch (Exception e) {
            System.err.println("Failed to setup logging: " + e.getMessage());
        }
    }

    private void logSessionSummary() {
        Metrics metrics = getMetrics();
        LOGGER.info("Cognitive Agent Session Summary:");
        LOGGER.info("Total Interactions: " + metrics.getTotalInteractions());
        LOGGER.info("Successful Interactions: " + metrics.getSuccessfulInteractions());
        LOGGER.info("Average Response Time: " + String.format("%.4f", metrics.getAverageResponseTime()) + " ms");
        if (metrics.getLastInteractionTime() != null) {
            LOGGER.info("Last Interaction: " + metrics.getLastInteractionTime());
        }
    }

    // Add runUPOFMode method
    private void runUPOFMode(String query) {
        LOGGER.info("Running UPOF publicationv1 mode for query: " + query);
        
        UPOFProcessor processor = new UPOFProcessor();
        
        // Simulate inputs (from publicationv1 examples)
        double S_x = 0.8; // Symbolic output
        double N_x = 0.3; // Neural output
        double R_cognitive = 0.5;
        double R_efficiency = 0.4;
        double P_H_E = 0.7;
        
        double psi = processor.computePsi(S_x, N_x, R_cognitive, R_efficiency, P_H_E);
        String ninestepResult = processor.applyNinestep(query);
        double proofResult = processor.simulateSwiftSwarmProof(3.0, 2.0); // Example values
        
        System.out.println("UPOF Psi Computation: " + psi);
        System.out.println(ninestepResult);
        System.out.println("Swift Swarm Proof Simulation: " + proofResult);
    }

    /**
     * Metrics container
     */
    public static final class Metrics {
        private final int totalInteractions;
        private final int successfulInteractions;
        private final double averageResponseTime;
        private final LocalDateTime lastInteractionTime;

        public Metrics(int totalInteractions, int successfulInteractions, 
                      double averageResponseTime, LocalDateTime lastInteractionTime) {
            this.totalInteractions = totalInteractions;
            this.successfulInteractions = successfulInteractions;
            this.averageResponseTime = averageResponseTime;
            this.lastInteractionTime = lastInteractionTime;
        }

        public int getTotalInteractions() { return totalInteractions; }
        public int getSuccessfulInteractions() { return successfulInteractions; }
        public double getAverageResponseTime() { return averageResponseTime; }
        public LocalDateTime getLastInteractionTime() { return lastInteractionTime; }
    }

    // Convenience methods
    public static CognitiveAgentCLI createBasicCLI(String apiKey) {
        AnthropicClientEnhanced client = AnthropicClientEnhanced.createBasicClient(apiKey);
        return new Builder().client(client).build();
    }

    public static CognitiveAgentCLI createToolEnabledCLI(String apiKey, Set<String> tools) {
        AnthropicClientEnhanced client = AnthropicClientEnhanced.createToolEnabledClient(apiKey, new ArrayList<>(tools));
        return new Builder().client(client).tools(tools).build();
    }

    public static void main(String[] args) {
        // Simple command line argument parsing
        String apiKey = System.getenv("ANTHROPIC_API_KEY");
        if (apiKey == null || apiKey.trim().isEmpty()) {
            System.err.println("⚠️ Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.");
            System.exit(1);
        }

        CognitiveAgentCLI cli = createBasicCLI(apiKey);
        cli.interactiveSession();
    }
} 