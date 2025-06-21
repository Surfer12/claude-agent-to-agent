import com.anthropic.claude.agent.tools.ToolRegistry;

public class TestResetFix {
    public static void main(String[] args) {
        try {
            ToolRegistry registry = new ToolRegistry();

            // Discover tools
            registry.discoverTools();

            // Verify tools are available
            System.out.println("Before reset - Available tools: " + registry.getAvailableTools());
            System.out.println("Before reset - Tool count: " + registry.getAvailableTools().size());
            System.out.println("Before reset - Has 'think' tool: " + registry.isToolAvailable("think"));

            // Reset the registry
            registry.reset();

            // Verify tools are cleared
            System.out.println("\nAfter reset - Available tools: " + registry.getAvailableTools());
            System.out.println("After reset - Tool count: " + registry.getAvailableTools().size());
            System.out.println("After reset - Has 'think' tool: " + registry.isToolAvailable("think"));

            // Test passed if we get here and tools are empty
            if (registry.getAvailableTools().isEmpty() && !registry.isToolAvailable("think")) {
                System.out.println("\n✓ TEST PASSED: Reset correctly cleared all tools");
            } else {
                System.out.println("\n✗ TEST FAILED: Reset did not clear tools properly");
                System.exit(1);
            }

        } catch (Exception e) {
            System.err.println("Test failed with exception: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }
}
