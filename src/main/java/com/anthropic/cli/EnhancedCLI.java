package com.anthropic.cli;

import com.anthropic.api.AnthropicClient;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.TimeUnit;
import okhttp3.*;

/**
 * Enhanced CLI for Claude Agent interactions in Java
 * Provides an intuitive, interactive interface similar to the Python version
 */
public class EnhancedCLI {

    private static final String API_BASE = "https://api.anthropic.com/v1";
    private static final String DEFAULT_MODEL = "claude-3-5-sonnet-20240620";

    private final String apiKey;
    private final String model;
    private final boolean verbose;
    private final OkHttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final SessionMetrics metrics;
    private final List<Map<String, Object>> conversationHistory;

    public EnhancedCLI(String apiKey, String model, boolean verbose) {
        this.apiKey = Objects.requireNonNull(apiKey, "API key cannot be null");
        this.model = model != null ? model : DEFAULT_MODEL;
        this.verbose = verbose;
        this.httpClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .build();
        this.objectMapper = new ObjectMapper();
        this.metrics = new SessionMetrics();
        this.conversationHistory = new ArrayList<>();
    }

    public void startInteractiveSession() {
        showWelcome();
        Scanner scanner = new Scanner(System.in);

        try {
            while (true) {
                System.out.print("\n💬 ");
                String input = scanner.nextLine().trim();

                if (input.isEmpty()) continue;

                String lowerInput = input.toLowerCase();

                // Handle commands
                if (
                    Arrays.asList("exit", "quit", "q", "bye").contains(
                        lowerInput
                    )
                ) {
                    System.out.println("\n👋 Goodbye!");
                    break;
                } else if (Arrays.asList("help", "?").contains(lowerInput)) {
                    showHelp();
                    continue;
                } else if ("clear".equals(lowerInput)) {
                    conversationHistory.clear();
                    System.out.println("🗑️  Chat history cleared!");
                    continue;
                } else if ("stats".equals(lowerInput)) {
                    showStats();
                    continue;
                }

                // Process the request
                processUserInput(input);
            }
        } catch (Exception e) {
            System.err.println("❌ Session error: " + e.getMessage());
        } finally {
            scanner.close();
            showSummary();
        }
    }

    private void processUserInput(String input) {
        System.out.print("🤔 ");
        long startTime = System.currentTimeMillis();

        try {
            // Add user message to history
            Map<String, Object> userMessage = Map.of(
                "role",
                "user",
                "content",
                input
            );
            conversationHistory.add(userMessage);

            // Create request
            Map<String, Object> request = Map.of(
                "model",
                model,
                "max_tokens",
                4096,
                "messages",
                conversationHistory,
                "system",
                "You are Claude, a helpful AI assistant. Be conversational, clear, and helpful."
            );

            // Make API call
            String response = makeApiCall(request);

            long endTime = System.currentTimeMillis();
            double responseTime = (endTime - startTime) / 1000.0;

            // Parse and display response
            displayResponse(response, responseTime);

            // Update metrics
            metrics.recordSuccess(responseTime);
        } catch (Exception e) {
            System.out.println(f("\n❌ Oops! %s", e.getMessage()));
            System.out.println("💡 Try rephrasing or type 'help'");
            metrics.recordFailure();
        }
    }

    private String makeApiCall(Map<String, Object> request) throws IOException {
        String jsonBody = objectMapper.writeValueAsString(request);

        RequestBody body = RequestBody.create(
            jsonBody,
            MediaType.get("application/json; charset=utf-8")
        );

        Request httpRequest = new Request.Builder()
            .url(API_BASE + "/messages")
            .header("x-api-key", apiKey)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .post(body)
            .build();

        try (Response response = httpClient.newCall(httpRequest).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException(
                    "API call failed: " +
                    response.code() +
                    " " +
                    response.message()
                );
            }
            return response.body().string();
        }
    }

    @SuppressWarnings("unchecked")
    private void displayResponse(String jsonResponse, double responseTime)
        throws IOException {
        Map<String, Object> response = objectMapper.readValue(
            jsonResponse,
            Map.class
        );

        System.out.print("\r🤖 Claude:\n");

        List<Map<String, Object>> content = (List<
                Map<String, Object>
            >) response.get("content");
        if (content != null) {
            for (Map<String, Object> block : content) {
                if ("text".equals(block.get("type"))) {
                    System.out.println(block.get("text"));
                }
            }

            // Add assistant response to history
            Map<String, Object> assistantMessage = Map.of(
                "role",
                "assistant",
                "content",
                content
            );
            conversationHistory.add(assistantMessage);
        }

        if (responseTime > 2.0) {
            System.out.printf("\n⏱️  (%.1fs)\n", responseTime);
        }
    }

    private void showWelcome() {
        System.out.println("🤖 " + "=".repeat(50));
        System.out.println("   CLAUDE AGENT CLI (Java)");
        System.out.println("   Enhanced Interactive Mode");
        System.out.println("=".repeat(54));
        System.out.println();
        System.out.println("💡 Quick commands:");
        System.out.println("   help    - Show help");
        System.out.println("   clear   - Clear chat history");
        System.out.println("   stats   - Show session stats");
        System.out.println("   exit    - End session");
        System.out.println();
        System.out.println("🧠 Model: " + model);
        System.out.println("-".repeat(54));
        System.out.println("Just type your questions naturally!");
    }

    private void showHelp() {
        System.out.println("\n🆘 HELP");
        System.out.println("=".repeat(20));
        System.out.println("Commands:");
        System.out.println("  help/? - This help");
        System.out.println("  clear  - Clear history");
        System.out.println("  stats  - Show stats");
        System.out.println("  exit   - Quit");
        System.out.println();
        System.out.println("💬 Examples:");
        System.out.println("  • What's the weather like?");
        System.out.println("  • Help me code a Java function");
        System.out.println("  • Explain design patterns");
        System.out.println("  • Calculate 15 * 23");
    }

    private void showStats() {
        System.out.println("\n📊 SESSION STATS");
        System.out.println("=".repeat(20));
        System.out.println("Interactions: " + metrics.getTotalInteractions());
        System.out.println(
            "Successful: " + metrics.getSuccessfulInteractions()
        );
        if (metrics.getTotalInteractions() > 0) {
            double rate =
                ((double) metrics.getSuccessfulInteractions() /
                    metrics.getTotalInteractions()) *
                100;
            System.out.printf("Success rate: %.1f%%\n", rate);
        }
        System.out.printf(
            "Avg response: %.1fs\n",
            metrics.getAverageResponseTime()
        );
    }

    private void showSummary() {
        if (metrics.getTotalInteractions() > 0) {
            System.out.println("\n📊 SESSION COMPLETE");
            System.out.println("=".repeat(25));
            System.out.println(
                "Total interactions: " + metrics.getTotalInteractions()
            );
            double rate =
                ((double) metrics.getSuccessfulInteractions() /
                    metrics.getTotalInteractions()) *
                100;
            System.out.printf("Success rate: %.1f%%\n", rate);
            System.out.printf(
                "Average response time: %.1fs\n",
                metrics.getAverageResponseTime()
            );
        }
    }

    // Helper method for string formatting (Java doesn't have f-strings)
    private static String f(String format, Object... args) {
        return String.format(format, args);
    }

    /**
     * Session metrics tracking
     */
    private static class SessionMetrics {

        private int totalInteractions = 0;
        private int successfulInteractions = 0;
        private double totalResponseTime = 0.0;

        public void recordSuccess(double responseTime) {
            totalInteractions++;
            successfulInteractions++;
            totalResponseTime += responseTime;
        }

        public void recordFailure() {
            totalInteractions++;
        }

        public int getTotalInteractions() {
            return totalInteractions;
        }

        public int getSuccessfulInteractions() {
            return successfulInteractions;
        }

        public double getAverageResponseTime() {
            return totalInteractions > 0
                ? totalResponseTime / successfulInteractions
                : 0.0;
        }
    }

    /**
     * Main entry point
     */
    public static void main(String[] args) {
        String apiKey = System.getenv("ANTHROPIC_API_KEY");
        if (apiKey == null || apiKey.trim().isEmpty()) {
            System.err.println(
                "❌ Please set ANTHROPIC_API_KEY environment variable"
            );
            System.err.println(
                "💡 Add to your shell profile: export ANTHROPIC_API_KEY='your-key'"
            );
            System.exit(1);
        }

        // Parse command line arguments
        String model = DEFAULT_MODEL;
        boolean verbose = false;
        String prompt = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--model", "-m":
                    if (i + 1 < args.length) model = args[++i];
                    break;
                case "--verbose", "-v":
                    verbose = true;
                    break;
                case "--prompt", "-p":
                    if (i + 1 < args.length) prompt = args[++i];
                    break;
                case "--help", "-h":
                    showUsage();
                    return;
            }
        }

        EnhancedCLI cli = new EnhancedCLI(apiKey, model, verbose);

        if (prompt != null) {
            // Single prompt mode
            cli.processSinglePrompt(prompt);
        } else {
            // Interactive mode
            cli.startInteractiveSession();
        }
    }

    private void processSinglePrompt(String prompt) {
        processUserInput(prompt);
    }

    private static void showUsage() {
        System.out.println("Enhanced Claude Agent CLI (Java)");
        System.out.println();
        System.out.println("Usage:");
        System.out.println(
            "  java -cp target/classes com.anthropic.cli.EnhancedCLI [options]"
        );
        System.out.println();
        System.out.println("Options:");
        System.out.println("  -h, --help           Show this help");
        System.out.println("  -m, --model MODEL    Claude model to use");
        System.out.println("  -v, --verbose        Verbose output");
        System.out.println("  -p, --prompt TEXT    Single prompt mode");
        System.out.println();
        System.out.println("Environment:");
        System.out.println(
            "  ANTHROPIC_API_KEY    Your Anthropic API key (required)"
        );
    }
}
