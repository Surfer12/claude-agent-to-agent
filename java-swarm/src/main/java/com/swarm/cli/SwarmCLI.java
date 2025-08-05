package com.swarm.cli;

import com.swarm.core.Swarm;
import com.swarm.types.*;
import com.swarm.util.SwarmUtil;
import org.apache.commons.cli.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

/**
 * Command Line Interface for the Java Swarm system.
 */
public class SwarmCLI {
    private static final String VERSION = "1.0.0";
    
    public static void main(String[] args) {
        Options options = createOptions();
        CommandLineParser parser = new DefaultParser();
        
        try {
            CommandLine cmd = parser.parse(options, args);
            
            if (cmd.hasOption("help")) {
                printHelp(options);
                return;
            }
            
            if (cmd.hasOption("version")) {
                System.out.println("Java Swarm CLI v" + VERSION);
                return;
            }
            
            SwarmCLI cli = new SwarmCLI();
            cli.run(cmd);
            
        } catch (ParseException e) {
            System.err.println("Error parsing command line: " + e.getMessage());
            printHelp(options);
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            if (System.getenv("DEBUG") != null) {
                e.printStackTrace();
            }
            System.exit(1);
        }
    }
    
    private static Options createOptions() {
        Options options = new Options();
        
        options.addOption(Option.builder("h")
                .longOpt("help")
                .desc("Show this help message")
                .build());
                
        options.addOption(Option.builder("v")
                .longOpt("version")
                .desc("Show version information")
                .build());
                
        options.addOption(Option.builder("i")
                .longOpt("interactive")
                .desc("Run in interactive mode")
                .build());
                
        options.addOption(Option.builder("m")
                .longOpt("model")
                .hasArg()
                .argName("MODEL")
                .desc("OpenAI model to use (default: gpt-4o)")
                .build());
                
        options.addOption(Option.builder("d")
                .longOpt("debug")
                .desc("Enable debug mode")
                .build());
                
        options.addOption(Option.builder("t")
                .longOpt("max-turns")
                .hasArg()
                .argName("TURNS")
                .desc("Maximum number of turns (default: unlimited)")
                .build());
                
        options.addOption(Option.builder()
                .longOpt("input")
                .hasArg()
                .argName("MESSAGE")
                .desc("Input message to send to the agent")
                .build());
                
        options.addOption(Option.builder()
                .longOpt("agent-name")
                .hasArg()
                .argName("NAME")
                .desc("Name for the agent (default: Assistant)")
                .build());
                
        options.addOption(Option.builder()
                .longOpt("instructions")
                .hasArg()
                .argName("INSTRUCTIONS")
                .desc("System instructions for the agent")
                .build());
                
        options.addOption(Option.builder("s")
                .longOpt("stream")
                .desc("Enable streaming responses")
                .build());
                
        options.addOption(Option.builder()
                .longOpt("no-stream")
                .desc("Disable streaming responses (default)")
                .build());
        
        return options;
    }
    
    private static void printHelp(Options options) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.printHelp("java-swarm", "Java implementation of the Swarm multi-agent framework", 
                          options, "\nExamples:\n" +
                          "  java -jar java-swarm.jar --interactive\n" +
                          "  java -jar java-swarm.jar --input \"Hello, how are you?\"\n" +
                          "  java -jar java-swarm.jar --interactive --debug --model gpt-4o-mini\n");
    }
    
    public void run(CommandLine cmd) throws IOException {
        // Initialize Swarm
        Swarm swarm = new Swarm();
        
        // Parse options
        String model = cmd.getOptionValue("model", "gpt-4o");
        boolean debug = cmd.hasOption("debug");
        int maxTurns = Integer.parseInt(cmd.getOptionValue("max-turns", String.valueOf(Integer.MAX_VALUE)));
        String agentName = cmd.getOptionValue("agent-name", "Assistant");
        String instructions = cmd.getOptionValue("instructions", "You are a helpful assistant.");
        
        // Create agent
        Agent agent = Agent.builder()
                .name(agentName)
                .model(model)
                .instructions(instructions)
                .build();
        
        // Add some basic functions
        agent.getFunctions().add(new EchoFunction());
        agent.getFunctions().add(new CalculatorFunction());
        
        System.out.println("Java Swarm CLI v" + VERSION);
        System.out.println("Agent: " + agent.getName() + " (Model: " + agent.getModel() + ")");
        System.out.println("Debug: " + debug);
        System.out.println();
        
        if (cmd.hasOption("interactive")) {
            runInteractive(swarm, agent, debug, maxTurns);
        } else if (cmd.hasOption("input")) {
            String input = cmd.getOptionValue("input");
            runSingle(swarm, agent, input, debug, maxTurns);
        } else {
            System.err.println("Either --interactive or --input must be specified");
            System.exit(1);
        }
    }
    
    private void runInteractive(Swarm swarm, Agent agent, boolean debug, int maxTurns) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        List<Map<String, Object>> history = new ArrayList<>();
        Map<String, Object> contextVariables = new HashMap<>();
        
        System.out.println("Interactive mode. Type 'quit' or 'exit' to end the session.");
        System.out.println("Type 'clear' to clear the conversation history.");
        System.out.println("Type 'help' for available commands.");
        System.out.println();
        
        while (true) {
            System.out.print("You: ");
            String input = reader.readLine();
            
            if (input == null || input.trim().equalsIgnoreCase("quit") || input.trim().equalsIgnoreCase("exit")) {
                System.out.println("Goodbye!");
                break;
            }
            
            if (input.trim().equalsIgnoreCase("clear")) {
                history.clear();
                contextVariables.clear();
                System.out.println("Conversation history cleared.");
                continue;
            }
            
            if (input.trim().equalsIgnoreCase("help")) {
                printInteractiveHelp();
                continue;
            }
            
            if (input.trim().isEmpty()) {
                continue;
            }
            
            // Add user message to history
            Map<String, Object> userMessage = new HashMap<>();
            userMessage.put("role", "user");
            userMessage.put("content", input.trim());
            history.add(userMessage);
            
            try {
                // Run the swarm
                Response response = swarm.run(agent, history, contextVariables, null, false, debug, maxTurns, true);
                
                // Update history and context
                history.addAll(response.getMessages());
                contextVariables.putAll(response.getContextVariables());
                
                // Print the response
                if (!response.getMessages().isEmpty()) {
                    Map<String, Object> lastMessage = response.getMessages().get(response.getMessages().size() - 1);
                    String content = (String) lastMessage.get("content");
                    String sender = (String) lastMessage.get("sender");
                    
                    if (content != null && !content.trim().isEmpty()) {
                        System.out.println(sender + ": " + content);
                    }
                }
                
                if (debug && !contextVariables.isEmpty()) {
                    System.out.println("Context variables: " + contextVariables);
                }
                
            } catch (Exception e) {
                System.err.println("Error: " + e.getMessage());
                if (debug) {
                    e.printStackTrace();
                }
            }
            
            System.out.println();
        }
    }
    
    private void runSingle(Swarm swarm, Agent agent, String input, boolean debug, int maxTurns) {
        List<Map<String, Object>> messages = new ArrayList<>();
        Map<String, Object> userMessage = new HashMap<>();
        userMessage.put("role", "user");
        userMessage.put("content", input);
        messages.add(userMessage);
        
        try {
            Response response = swarm.run(agent, messages, new HashMap<>(), null, false, debug, maxTurns, true);
            
            if (!response.getMessages().isEmpty()) {
                Map<String, Object> lastMessage = response.getMessages().get(response.getMessages().size() - 1);
                String content = (String) lastMessage.get("content");
                
                if (content != null && !content.trim().isEmpty()) {
                    System.out.println(content);
                }
            }
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            if (debug) {
                e.printStackTrace();
            }
            System.exit(1);
        }
    }
    
    private void printInteractiveHelp() {
        System.out.println("Available commands:");
        System.out.println("  help  - Show this help message");
        System.out.println("  clear - Clear conversation history");
        System.out.println("  quit  - Exit the program");
        System.out.println("  exit  - Exit the program");
        System.out.println();
        System.out.println("Available functions:");
        System.out.println("  echo(message) - Echo back a message");
        System.out.println("  calculate(expression) - Calculate a mathematical expression");
    }
}
