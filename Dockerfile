FROM python:3.9.6 

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -r -s /bin/bash claude-agent
RUN chown -R claude-agent:claude-agent /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set proper permissions
RUN chown -R claude-agent:claude-agent /app

# Switch to non-root user
USER claude-agent

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/claude-agent/.local/bin:${PATH}"

# Default command
CMD ["python", "cli.py", "--interactive"]
