package com.anthropic.api.tools;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Tool implementations for Anthropic API
 * 
 * This class provides specialized tool classes for various Anthropic API features
 * including bash execution, web search, weather, text editing, and code execution.
 */
public final class AnthropicTools {
    
    /**
     * Base tool definition
     */
    public static final class ToolDefinition {
        private final String name;
        private final String toolType;
        private final String description;
        private final Map<String, Object> inputSchema;
        private final Integer maxUses;
        private final Map<String, Object> displayConfig;

        public ToolDefinition(String name, String toolType, String description) {
            this(name, toolType, description, null, null, null);
        }

        public ToolDefinition(String name, String toolType, String description, Integer maxUses) {
            this(name, toolType, description, null, maxUses, null);
        }

        public ToolDefinition(String name, String toolType, String description, 
                             Map<String, Object> inputSchema, Integer maxUses, 
                             Map<String, Object> displayConfig) {
            this.name = name;
            this.toolType = toolType;
            this.description = description;
            this.inputSchema = inputSchema;
            this.maxUses = maxUses;
            this.displayConfig = displayConfig;
        }

        public String getName() { return name; }
        public String getToolType() { return toolType; }
        public String getDescription() { return description; }
        public Map<String, Object> getInputSchema() { return inputSchema; }
        public Integer getMaxUses() { return maxUses; }
        public Map<String, Object> getDisplayConfig() { return displayConfig; }
    }

    /**
     * Base class for all Anthropic API tools
     */
    public static abstract class BaseTool {
        protected final ToolDefinition definition;

        protected BaseTool(ToolDefinition definition) {
            this.definition = definition;
        }

        public Map<String, Object> toMap() {
            Map<String, Object> result = new HashMap<>();
            result.put("type", definition.getToolType());
            result.put("name", definition.getName());
            result.put("description", definition.getDescription());

            if (definition.getInputSchema() != null) {
                result.put("input_schema", definition.getInputSchema());
            }
            if (definition.getMaxUses() != null) {
                result.put("max_uses", definition.getMaxUses());
            }
            if (definition.getDisplayConfig() != null) {
                result.putAll(definition.getDisplayConfig());
            }

            return result;
        }

        public ToolDefinition getDefinition() {
            return definition;
        }
    }

    /**
     * Tool for executing bash commands
     */
    public static final class BashTool extends BaseTool {
        public BashTool() {
            super(new ToolDefinition(
                "bash",
                "bash_20250124",
                "Execute bash commands in a secure environment"
            ));
        }
    }

    /**
     * Tool for web search functionality
     */
    public static final class WebSearchTool extends BaseTool {
        public WebSearchTool() {
            this(5);
        }

        public WebSearchTool(int maxUses) {
            super(new ToolDefinition(
                "web_search",
                "web_search_20250305",
                "Search the web for current information",
                maxUses
            ));
        }
    }

    /**
     * Tool for weather information
     */
    public static final class WeatherTool extends BaseTool {
        public WeatherTool() {
            super(new ToolDefinition(
                "get_weather",
                "weather_tool",
                "Get the current weather in a given location",
                createWeatherInputSchema(),
                null,
                null
            ));
        }

        private static Map<String, Object> createWeatherInputSchema() {
            Map<String, Object> schema = new HashMap<>();
            schema.put("type", "object");
            
            Map<String, Object> properties = new HashMap<>();
            Map<String, Object> locationProperty = new HashMap<>();
            locationProperty.put("type", "string");
            locationProperty.put("description", "The city and state, e.g. San Francisco, CA");
            properties.put("location", locationProperty);
            
            schema.put("properties", properties);
            schema.put("required", Arrays.asList("location"));
            
            return schema;
        }
    }

    /**
     * Tool for text editing operations
     */
    public static final class TextEditorTool extends BaseTool {
        public TextEditorTool() {
            super(new ToolDefinition(
                "str_replace_based_edit_tool",
                "text_editor_20250429",
                "Edit text files with string replacement operations"
            ));
        }
    }

    /**
     * Tool for code execution
     */
    public static final class CodeExecutionTool extends BaseTool {
        public CodeExecutionTool() {
            super(new ToolDefinition(
                "code_execution",
                "code_execution_20250522",
                "Execute code in a secure environment"
            ));
        }
    }

    /**
     * Tool for computer interface interaction
     */
    public static final class ComputerTool extends BaseTool {
        public ComputerTool() {
            this(1024, 768, 1);
        }

        public ComputerTool(int displayWidth, int displayHeight, int displayNumber) {
            super(new ToolDefinition(
                "computer",
                "computer_20250124",
                "Interact with computer interface",
                null,
                null,
                createDisplayConfig(displayWidth, displayHeight, displayNumber)
            ));
        }

        private static Map<String, Object> createDisplayConfig(int width, int height, int number) {
            Map<String, Object> config = new HashMap<>();
            config.put("display_width_px", width);
            config.put("display_height_px", height);
            config.put("display_number", number);
            return config;
        }
    }

    /**
     * Collection of tools for streaming operations
     */
    public static final class StreamingTools {
        public static Map<String, Object> makeFileTool() {
            Map<String, Object> tool = new HashMap<>();
            tool.put("name", "make_file");
            tool.put("description", "Write text to a file");
            
            Map<String, Object> inputSchema = new HashMap<>();
            inputSchema.put("type", "object");
            
            Map<String, Object> properties = new HashMap<>();
            
            Map<String, Object> filenameProperty = new HashMap<>();
            filenameProperty.put("type", "string");
            filenameProperty.put("description", "The filename to write text to");
            properties.put("filename", filenameProperty);
            
            Map<String, Object> linesProperty = new HashMap<>();
            linesProperty.put("type", "array");
            linesProperty.put("description", "An array of lines of text to write to the file");
            properties.put("lines_of_text", linesProperty);
            
            inputSchema.put("properties", properties);
            inputSchema.put("required", Arrays.asList("filename", "lines_of_text"));
            
            tool.put("input_schema", inputSchema);
            return tool;
        }
    }

    // Tool factory methods
    public static BashTool createBashTool() {
        return new BashTool();
    }

    public static WebSearchTool createWebSearchTool() {
        return new WebSearchTool();
    }

    public static WebSearchTool createWebSearchTool(int maxUses) {
        return new WebSearchTool(maxUses);
    }

    public static WeatherTool createWeatherTool() {
        return new WeatherTool();
    }

    public static TextEditorTool createTextEditorTool() {
        return new TextEditorTool();
    }

    public static CodeExecutionTool createCodeExecutionTool() {
        return new CodeExecutionTool();
    }

    public static ComputerTool createComputerTool() {
        return new ComputerTool();
    }

    public static ComputerTool createComputerTool(int displayWidth, int displayHeight) {
        return new ComputerTool(displayWidth, displayHeight, 1);
    }

    // New tool for UPOF processing
    public static final class UPOFTool extends BaseTool {
        public UPOFTool() {
            super(new ToolDefinition(
                "upof_processor",
                "upof_20250801",
                "Process queries using Unified Onto-Phenomenological Consciousness Framework",
                createUPOFInputSchema(),
                null,
                null
            ));
        }

        private static Map<String, Object> createUPOFInputSchema() {
            Map<String, Object> schema = new HashMap<>();
            schema.put("type", "object");
            
            Map<String, Object> properties = new HashMap<>();
            
            Map<String, Object> queryProperty = new HashMap<>();
            queryProperty.put("type", "string");
            queryProperty.put("description", "The query to process through UPOF");
            properties.put("query", queryProperty);
            
            schema.put("properties", properties);
            schema.put("required", Arrays.asList("query"));
            
            return schema;
        }
    }

    // New tool for Ninestep integration
    public static final class NinestepTool extends BaseTool {
        public NinestepTool() {
            super(new ToolDefinition(
                "ninestep",
                "ninestep_20250801",
                "Apply 9-step AI integration framework with consciousness protection",
                createNinestepInputSchema(),
                null,
                null
            ));
        }

        private static Map<String, Object> createNinestepInputSchema() {
            Map<String, Object> schema = new HashMap<>();
            schema.put("type", "object");
            
            Map<String, Object> properties = new HashMap<>();
            
            Map<String, Object> queryProperty = new HashMap<>();
            queryProperty.put("type", "string");
            queryProperty.put("description", "The query to process through Ninestep");
            properties.put("query", queryProperty);
            
            schema.put("properties", properties);
            schema.put("required", Arrays.asList("query"));
            
            return schema;
        }
    }

    // New tool for Swift Swarm Proof simulation
    public static final class SwiftSwarmTool extends BaseTool {
        public SwiftSwarmTool() {
            super(new ToolDefinition(
                "swift_swarm_proof",
                "swift_swarm_20250801",
                "Simulate mathematical proof elements from swift_swarm_witten",
                createSwiftSwarmInputSchema(),
                null,
                null
            ));
        }

        private static Map<String, Object> createSwiftSwarmInputSchema() {
            Map<String, Object> schema = new HashMap<>();
            schema.put("type", "object");
            
            Map<String, Object> properties = new HashMap<>();
            
            Map<String, Object> degreeProperty = new HashMap<>();
            degreeProperty.put("type", "number");
            degreeProperty.put("description", "Degree for the simulation");
            properties.put("degree", degreeProperty);
            
            Map<String, Object> contactOrderProperty = new HashMap<>();
            contactOrderProperty.put("type", "number");
            contactOrderProperty.put("description", "Contact order for the simulation");
            properties.put("contact_order", contactOrderProperty);
            
            schema.put("properties", properties);
            schema.put("required", Arrays.asList("degree", "contact_order"));
            
            return schema;
        }
    }

    /**
     * Get all available tool types
     */
    public static List<String> getAvailableToolTypes() {
        return Arrays.asList("bash", "web_search", "weather", "text_editor", "code_execution", "computer",
                             "upof_processor", "upof_framework", "ninestep", "ninestep_framework", "swift_swarm_proof");
    }

    /**
     * Create a tool by type name
     */
    public static Optional<BaseTool> createToolByType(String toolType) {
        switch (toolType.toLowerCase()) {
            case "bash":
                return Optional.of(createBashTool());
            case "web_search":
                return Optional.of(createWebSearchTool());
            case "weather":
                return Optional.of(createWeatherTool());
            case "text_editor":
                return Optional.of(createTextEditorTool());
            case "code_execution":
                return Optional.of(createCodeExecutionTool());
            case "computer":
                return Optional.of(createComputerTool());
            case "upof_processor":
            case "upof_framework":
                return Optional.of(new com.anthropic.api.tools.UPOFTool());
            case "ninestep":
            case "ninestep_framework":
                return Optional.of(new com.anthropic.api.tools.NinestepTool());
            case "swift_swarm_proof":
                return Optional.of(new com.anthropic.api.tools.SwiftSwarmTool());
            default:
                return Optional.empty();
        }
    }

    /**
     * Convert a list of tool names to tool instances
     */
    public static List<BaseTool> createToolsFromNames(List<String> toolNames) {
        return toolNames.stream()
            .map(AnthropicTools::createToolByType)
            .filter(Optional::isPresent)
            .map(Optional::get)
            .collect(Collectors.toList());
    }

    /**
     * Convert a list of tools to their map representations
     */
    public static List<Map<String, Object>> toolsToMaps(List<BaseTool> tools) {
        return tools.stream()
            .map(BaseTool::toMap)
            .collect(Collectors.toList());
    }
} 