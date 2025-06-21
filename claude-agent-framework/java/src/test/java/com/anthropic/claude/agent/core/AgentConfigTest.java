package com.anthropic.claude.agent.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for AgentConfig class.
 */
class AgentConfigTest {
    
    @Test
    void testDefaultConfig() {
        AgentConfig config = new AgentConfig();
        
        assertEquals("claude-agent", config.getName());
        assertEquals("claude-sonnet-4-20250514", config.getModelConfig().getModel());
        assertEquals(4096, config.getModelConfig().getMaxTokens());
        assertEquals(1.0, config.getModelConfig().getTemperature());
        assertFalse(config.isVerbose());
        assertEquals(Arrays.asList("all"), config.getEnabledTools());
    }
    
    @Test
    void testBuilder() {
        AgentConfig config = AgentConfig.builder()
                .name("test-agent")
                .systemPrompt("Test prompt")
                .verbose(true)
                .model("claude-haiku-3-20240307")
                .maxTokens(2048)
                .temperature(0.5)
                .addTool("think")
                .addTool("file_read")
                .build();
        
        assertEquals("test-agent", config.getName());
        assertEquals("Test prompt", config.getSystemPrompt());
        assertTrue(config.isVerbose());
        assertEquals("claude-haiku-3-20240307", config.getModelConfig().getModel());
        assertEquals(2048, config.getModelConfig().getMaxTokens());
        assertEquals(0.5, config.getModelConfig().getTemperature());
        assertTrue(config.getEnabledTools().contains("think"));
        assertTrue(config.getEnabledTools().contains("file_read"));
    }
    
    @Test
    void testConfigFileOperations(@TempDir Path tempDir) throws IOException {
        // Create original config
        AgentConfig originalConfig = AgentConfig.builder()
                .name("file-test-agent")
                .systemPrompt("File test prompt")
                .verbose(true)
                .model("claude-sonnet-3-20240229")
                .maxTokens(1024)
                .temperature(0.7)
                .addTool("think")
                .build();
        
        // Save to file
        File configFile = tempDir.resolve("test-config.yaml").toFile();
        originalConfig.toFile(configFile.getAbsolutePath());
        
        assertTrue(configFile.exists());
        
        // Load from file
        AgentConfig loadedConfig = AgentConfig.fromFile(configFile.getAbsolutePath());
        
        assertEquals(originalConfig.getName(), loadedConfig.getName());
        assertEquals(originalConfig.getSystemPrompt(), loadedConfig.getSystemPrompt());
        assertEquals(originalConfig.isVerbose(), loadedConfig.isVerbose());
        assertEquals(originalConfig.getModelConfig().getModel(), loadedConfig.getModelConfig().getModel());
        assertEquals(originalConfig.getModelConfig().getMaxTokens(), loadedConfig.getModelConfig().getMaxTokens());
        assertEquals(originalConfig.getModelConfig().getTemperature(), loadedConfig.getModelConfig().getTemperature());
        assertEquals(originalConfig.getEnabledTools(), loadedConfig.getEnabledTools());
    }
    
    @Test
    void testModelConfig() {
        ModelConfig modelConfig = new ModelConfig();
        
        assertEquals("claude-sonnet-4-20250514", modelConfig.getModel());
        assertEquals(4096, modelConfig.getMaxTokens());
        assertEquals(1.0, modelConfig.getTemperature());
        assertEquals(180000, modelConfig.getContextWindowTokens());
        
        // Test with parameters
        ModelConfig customConfig = new ModelConfig("claude-haiku-3-20240307", 2048, 0.5);
        assertEquals("claude-haiku-3-20240307", customConfig.getModel());
        assertEquals(2048, customConfig.getMaxTokens());
        assertEquals(0.5, customConfig.getTemperature());
    }
}
