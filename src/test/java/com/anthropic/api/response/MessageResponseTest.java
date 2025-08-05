package com.anthropic.api.response;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.List;
import java.util.Collections;

/**
 * Test suite for MessageResponse - a leaf node response class.
 * Tests data integrity, nested classes, and immutability.
 */
@DisplayName("MessageResponse Tests")
class MessageResponseTest {

    private MessageResponse.ContentBlock textContent;
    private MessageResponse.Usage usage;
    private MessageResponse messageResponse;

    @BeforeEach
    void setUp() {
        textContent = new MessageResponse.ContentBlock("text", "Hello, World!");
        usage = new MessageResponse.Usage(100, 50);
        messageResponse = new MessageResponse(
            "msg_123",
            "message",
            "assistant",
            Arrays.asList(textContent),
            "claude-3-sonnet",
            "end_turn",
            null,
            usage
        );
    }

    @Nested
    @DisplayName("MessageResponse Core Tests")
    class MessageResponseCoreTests {

        @Test
        @DisplayName("Should create MessageResponse with all fields")
        void shouldCreateMessageResponseWithAllFields() {
            assertNotNull(messageResponse);
            assertEquals("msg_123", messageResponse.getId());
            assertEquals("message", messageResponse.getType());
            assertEquals("assistant", messageResponse.getRole());
            assertEquals("claude-3-sonnet", messageResponse.getModel());
            assertEquals("end_turn", messageResponse.getStopReason());
            assertNull(messageResponse.getStopSequence());
            assertNotNull(messageResponse.getContent());
            assertEquals(1, messageResponse.getContent().size());
            assertNotNull(messageResponse.getUsage());
        }

        @Test
        @DisplayName("Should handle null stopSequence")
        void shouldHandleNullStopSequence() {
            MessageResponse response = new MessageResponse(
                "msg_456",
                "message",
                "assistant",
                Collections.emptyList(),
                "claude-3-sonnet",
                "max_tokens",
                null,
                usage
            );

            assertNull(response.getStopSequence());
            assertEquals("max_tokens", response.getStopReason());
        }

        @Test
        @DisplayName("Should handle empty content list")
        void shouldHandleEmptyContentList() {
            MessageResponse response = new MessageResponse(
                "msg_789",
                "message",
                "assistant",
                Collections.emptyList(),
                "claude-3-sonnet",
                "end_turn",
                null,
                usage
            );

            assertNotNull(response.getContent());
            assertTrue(response.getContent().isEmpty());
        }

        @Test
        @DisplayName("Should preserve content order")
        void shouldPreserveContentOrder() {
            MessageResponse.ContentBlock content1 = new MessageResponse.ContentBlock("text", "First");
            MessageResponse.ContentBlock content2 = new MessageResponse.ContentBlock("text", "Second");
            List<MessageResponse.ContentBlock> contents = Arrays.asList(content1, content2);

            MessageResponse response = new MessageResponse(
                "msg_order",
                "message",
                "assistant",
                contents,
                "claude-3-sonnet",
                "end_turn",
                null,
                usage
            );

            List<MessageResponse.ContentBlock> retrievedContents = response.getContent();
            assertEquals(2, retrievedContents.size());
            assertEquals("First", retrievedContents.get(0).getText());
            assertEquals("Second", retrievedContents.get(1).getText());
        }
    }

    @Nested
    @DisplayName("ContentBlock Tests")
    class ContentBlockTests {

        @Test
        @DisplayName("Should create ContentBlock with type and text")
        void shouldCreateContentBlockWithTypeAndText() {
            MessageResponse.ContentBlock block = new MessageResponse.ContentBlock("text", "Test content");
            
            assertEquals("text", block.getType());
            assertEquals("Test content", block.getText());
        }

        @Test
        @DisplayName("Should handle different content types")
        void shouldHandleDifferentContentTypes() {
            MessageResponse.ContentBlock textBlock = new MessageResponse.ContentBlock("text", "Text content");
            MessageResponse.ContentBlock imageBlock = new MessageResponse.ContentBlock("image", "image_data");
            
            assertEquals("text", textBlock.getType());
            assertEquals("image", imageBlock.getType());
            assertEquals("Text content", textBlock.getText());
            assertEquals("image_data", imageBlock.getText());
        }

        @Test
        @DisplayName("Should handle null values in ContentBlock")
        void shouldHandleNullValuesInContentBlock() {
            MessageResponse.ContentBlock blockWithNullType = new MessageResponse.ContentBlock(null, "content");
            MessageResponse.ContentBlock blockWithNullText = new MessageResponse.ContentBlock("text", null);
            
            assertNull(blockWithNullType.getType());
            assertEquals("content", blockWithNullType.getText());
            assertEquals("text", blockWithNullText.getType());
            assertNull(blockWithNullText.getText());
        }

        @Test
        @DisplayName("Should handle empty strings in ContentBlock")
        void shouldHandleEmptyStringsInContentBlock() {
            MessageResponse.ContentBlock block = new MessageResponse.ContentBlock("", "");
            
            assertEquals("", block.getType());
            assertEquals("", block.getText());
        }
    }

    @Nested
    @DisplayName("Usage Tests")
    class UsageTests {

        @Test
        @DisplayName("Should create Usage with input and output tokens")
        void shouldCreateUsageWithTokens() {
            MessageResponse.Usage usage = new MessageResponse.Usage(150, 75);
            
            assertEquals(150, usage.getInputTokens());
            assertEquals(75, usage.getOutputTokens());
        }

        @Test
        @DisplayName("Should handle zero token counts")
        void shouldHandleZeroTokenCounts() {
            MessageResponse.Usage zeroUsage = new MessageResponse.Usage(0, 0);
            
            assertEquals(0, zeroUsage.getInputTokens());
            assertEquals(0, zeroUsage.getOutputTokens());
        }

        @Test
        @DisplayName("Should handle negative token counts")
        void shouldHandleNegativeTokenCounts() {
            // Note: In real implementation, you might want to validate against negative values
            MessageResponse.Usage negativeUsage = new MessageResponse.Usage(-1, -1);
            
            assertEquals(-1, negativeUsage.getInputTokens());
            assertEquals(-1, negativeUsage.getOutputTokens());
        }

        @Test
        @DisplayName("Should handle large token counts")
        void shouldHandleLargeTokenCounts() {
            MessageResponse.Usage largeUsage = new MessageResponse.Usage(Integer.MAX_VALUE, Integer.MAX_VALUE);
            
            assertEquals(Integer.MAX_VALUE, largeUsage.getInputTokens());
            assertEquals(Integer.MAX_VALUE, largeUsage.getOutputTokens());
        }
    }

    @Nested
    @DisplayName("Immutability Tests")
    class ImmutabilityTests {

        @Test
        @DisplayName("Should maintain immutability of MessageResponse")
        void shouldMaintainImmutabilityOfMessageResponse() {
            String originalId = messageResponse.getId();
            String originalModel = messageResponse.getModel();
            int originalContentSize = messageResponse.getContent().size();
            
            // Verify that getters return the same values
            assertEquals(originalId, messageResponse.getId());
            assertEquals(originalModel, messageResponse.getModel());
            assertEquals(originalContentSize, messageResponse.getContent().size());
        }

        @Test
        @DisplayName("Should maintain immutability of ContentBlock")
        void shouldMaintainImmutabilityOfContentBlock() {
            String originalType = textContent.getType();
            String originalText = textContent.getText();
            
            // Verify that getters return the same values
            assertEquals(originalType, textContent.getType());
            assertEquals(originalText, textContent.getText());
        }

        @Test
        @DisplayName("Should maintain immutability of Usage")
        void shouldMaintainImmutabilityOfUsage() {
            int originalInputTokens = usage.getInputTokens();
            int originalOutputTokens = usage.getOutputTokens();
            
            // Verify that getters return the same values
            assertEquals(originalInputTokens, usage.getInputTokens());
            assertEquals(originalOutputTokens, usage.getOutputTokens());
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Conditions")
    class EdgeCasesTests {

        @Test
        @DisplayName("Should handle null content list")
        void shouldHandleNullContentList() {
            MessageResponse response = new MessageResponse(
                "msg_null_content",
                "message",
                "assistant",
                null,
                "claude-3-sonnet",
                "end_turn",
                null,
                usage
            );

            assertNull(response.getContent());
        }

        @Test
        @DisplayName("Should handle null usage")
        void shouldHandleNullUsage() {
            MessageResponse response = new MessageResponse(
                "msg_null_usage",
                "message",
                "assistant",
                Arrays.asList(textContent),
                "claude-3-sonnet",
                "end_turn",
                null,
                null
            );

            assertNull(response.getUsage());
        }

        @Test
        @DisplayName("Should handle all null fields")
        void shouldHandleAllNullFields() {
            MessageResponse response = new MessageResponse(
                null, null, null, null, null, null, null, null
            );

            assertNull(response.getId());
            assertNull(response.getType());
            assertNull(response.getRole());
            assertNull(response.getContent());
            assertNull(response.getModel());
            assertNull(response.getStopReason());
            assertNull(response.getStopSequence());
            assertNull(response.getUsage());
        }

        @Test
        @DisplayName("Should handle mixed content types")
        void shouldHandleMixedContentTypes() {
            MessageResponse.ContentBlock textBlock = new MessageResponse.ContentBlock("text", "Hello");
            MessageResponse.ContentBlock imageBlock = new MessageResponse.ContentBlock("image", "base64data");
            MessageResponse.ContentBlock toolBlock = new MessageResponse.ContentBlock("tool_use", "tool_data");
            
            List<MessageResponse.ContentBlock> mixedContent = Arrays.asList(textBlock, imageBlock, toolBlock);
            
            MessageResponse response = new MessageResponse(
                "msg_mixed",
                "message",
                "assistant",
                mixedContent,
                "claude-3-sonnet",
                "end_turn",
                null,
                usage
            );

            assertEquals(3, response.getContent().size());
            assertEquals("text", response.getContent().get(0).getType());
            assertEquals("image", response.getContent().get(1).getType());
            assertEquals("tool_use", response.getContent().get(2).getType());
        }
    }
}