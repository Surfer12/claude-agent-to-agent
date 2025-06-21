package com.anthropic.claude.agent.tools;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for ToolRegistry class.
 */
class ToolRegistryTest {
    
    private ToolRegistry registry;
    
    @BeforeEach
    void setUp() {
        registry = new ToolRegistry();
    }
    
    @Test
    void testDiscoverTools() {
        registry.discoverTools();
        
        List<String> tools = registry.getAvailableTools();
        assertFalse(tools.isEmpty());
        
        // Check that basic tools are discovered
        assertTrue(tools.contains("think"));
        assertTrue(tools.contains("file_read"));
        assertTrue(tools.contains("file_write"));
        assertTrue(tools.contains("computer"));
        assertTrue(tools.contains("code_execution"));
    }
    
    @Test
    void testGetTool() throws Exception {
        registry.discoverTools();
        
        // Test getting think tool
        Tool thinkTool = registry.getTool("think");
        assertNotNull(thinkTool);
        assertEquals("think", thinkTool.getName());
        assertTrue(thinkTool.getDescription().toLowerCase().contains("think"));
        
        // Test getting file_read tool
        Tool fileReadTool = registry.getTool("file_read");
        assertNotNull(fileReadTool);
        assertEquals("file_read", fileReadTool.getName());
        assertTrue(fileReadTool.getDescription().toLowerCase().contains("read"));
    }
    
    @Test
    void testGetToolWithConfig() throws Exception {
        registry.discoverTools();
        
        Map<String, Object> config = new HashMap<>();
        config.put("display_width", 1280);
        config.put("display_height", 800);
        
        Tool computerTool = registry.getTool("computer", config);
        assertNotNull(computerTool);
        assertEquals("computer", computerTool.getName());
    }
    
    @Test
    void testGetToolInfo() throws Exception {
        registry.discoverTools();
        
        Map<String, Object> info = registry.getToolInfo("think");
        
        assertEquals("think", info.get("name"));
        assertNotNull(info.get("description"));
        assertNotNull(info.get("input_schema"));
        assertEquals("ThinkTool", info.get("class"));
        assertEquals(false, info.get("supports_files"));
    }
    
    @Test
    void testUnknownTool() {
        registry.discoverTools();
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            registry.getTool("unknown_tool");
        });
        
        assertTrue(exception.getMessage().contains("Tool not found: unknown_tool"));
    }
    
    @Test
    void testIsToolAvailable() {
        registry.discoverTools();
        
        assertTrue(registry.isToolAvailable("think"));
        assertTrue(registry.isToolAvailable("file_read"));
        assertFalse(registry.isToolAvailable("unknown_tool"));
    }
    
    @Test
    void testCachedTool() throws Exception {
        registry.discoverTools();
        
        Tool tool1 = registry.getCachedTool("think");
        Tool tool2 = registry.getCachedTool("think");
        
        // Should be the same instance when cached
        assertSame(tool1, tool2);
        
        // Clear cache and get again
        registry.clearCache();
        Tool tool3 = registry.getCachedTool("think");
        
        // Should be different instance after cache clear
        assertNotSame(tool1, tool3);
    }
    
    @Test
    void testReset() throws Exception {
        registry.discoverTools();
        
        assertFalse(registry.getAvailableTools().isEmpty());
        
        registry.reset();
        
        assertTrue(registry.getAvailableTools().isEmpty());
        assertFalse(registry.isToolAvailable("think"));
    }
}
