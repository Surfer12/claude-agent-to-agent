# Multi-stage build for better security and smaller image size
FROM python:3.13-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -r -s /bin/bash claude-agent && \
    mkdir -p /app && \
    chown -R claude-agent:claude-agent /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/claude-agent/.local

# Copy application code
COPY --chown=claude-agent:claude-agent . .

# Set proper permissions
RUN chown -R claude-agent:claude-agent /app

# Switch to non-root user
USER claude-agent

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/claude-agent/.local/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "cli.py", "--interactive"]
