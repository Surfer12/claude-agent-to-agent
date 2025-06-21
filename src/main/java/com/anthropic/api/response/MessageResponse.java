package com.anthropic.api.response;

import java.util.List;

public final class MessageResponse {
    private final String id;
    private final String type;
    private final String role;
    private final List<ContentBlock> content;
    private final String model;
    private final String stopReason;
    private final String stopSequence;
    private final Usage usage;

    public MessageResponse(
        String id,
        String type,
        String role,
        List<ContentBlock> content,
        String model,
        String stopReason,
        String stopSequence,
        Usage usage
    ) {
        this.id = id;
        this.type = type;
        this.role = role;
        this.content = content;
        this.model = model;
        this.stopReason = stopReason;
        this.stopSequence = stopSequence;
        this.usage = usage;
    }

    public String getId() {
        return id;
    }

    public String getType() {
        return type;
    }

    public String getRole() {
        return role;
    }

    public List<ContentBlock> getContent() {
        return content;
    }

    public String getModel() {
        return model;
    }

    public String getStopReason() {
        return stopReason;
    }

    public String getStopSequence() {
        return stopSequence;
    }

    public Usage getUsage() {
        return usage;
    }

    public static final class ContentBlock {
        private final String type;
        private final String text;

        public ContentBlock(String type, String text) {
            this.type = type;
            this.text = text;
        }

        public String getType() {
            return type;
        }

        public String getText() {
            return text;
        }
    }

    public static final class Usage {
        private final int inputTokens;
        private final int outputTokens;

        public Usage(int inputTokens, int outputTokens) {
            this.inputTokens = inputTokens;
            this.outputTokens = outputTokens;
        }

        public int getInputTokens() {
            return inputTokens;
        }

        public int getOutputTokens() {
            return outputTokens;
        }
    }
}
