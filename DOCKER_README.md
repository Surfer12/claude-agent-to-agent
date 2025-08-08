# Docker Setup for Claude Agent-to-Agent

This document provides comprehensive instructions for running the Claude Agent-to-Agent system using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (usually included with Docker Desktop)
- Anthropic API key

## Quick Start

### 1. Set up Environment Variables

Copy the environment template and add your API key:

```bash
cp env.example .env
```

Edit `.env` and add your Anthropic API key:
```bash
ANTHROPIC_API_KEY=your_actual_api_key_here
```

### 2. Build and Run

#### Option A: Using Docker Compose (Recommended)

```bash
# Build and start the main service
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# For development with hot reload
docker-compose up claude-agent-dev

# Run tests
docker-compose up claude-agent-test
```

#### Option B: Using Docker directly

```bash
# Build the image
docker build -t claude-agent-to-agent .

# Run the container
docker run -it --env-file .env claude-agent-to-agent

# Run with volume mounting for development
docker run -it --env-file .env -v $(pwd):/app claude-agent-to-agent
```

## Available Services

### Main Service (`claude-agent`)
- Production-ready service
- Runs on port 8000
- Includes health checks
- Optimized for performance

### Development Service (`claude-agent-dev`)
- Includes verbose logging
- Hot reload capabilities
- Runs on port 8001
- Better for debugging

### Test Service (`claude-agent-test`)
- Runs the test suite
- Useful for CI/CD pipelines
- Exits after test completion

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Your Anthropic API key |
| `CLAUDE_MODEL` | No | `claude-3-5-sonnet-20240620` | Claude model to use |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `DEBUG` | No | `false` | Enable debug mode |
| `VERBOSE` | No | `false` | Enable verbose output |

## Volume Mounts

The Docker setup includes several volume mounts for persistence:

- **Application Code**: `.:/app` - Mounts current directory for development
- **Logs**: `./logs:/app/logs` - Persists log files
- **Data**: `./data:/app/data` - Persists application data

## Security Features

- **Non-root User**: Container runs as `claude-agent` user
- **Multi-stage Build**: Reduces attack surface
- **Health Checks**: Monitors container health
- **Resource Limits**: Prevents resource exhaustion

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Fix volume permissions
   sudo chown -R $USER:$USER .
   ```

2. **API Key Not Found**
   ```bash
   # Check environment file
   cat .env
   # Ensure ANTHROPIC_API_KEY is set
   ```

3. **Port Already in Use**
   ```bash
   # Change ports in docker-compose.yml
   ports:
     - "8002:8000"  # Use different host port
   ```

4. **Build Fails**
   ```bash
   # Clean and rebuild
   docker-compose down
   docker system prune -f
   docker-compose up --build
   ```

### Debugging

```bash
# View logs
docker-compose logs claude-agent

# Access container shell
docker-compose exec claude-agent bash

# Check container health
docker-compose ps
```

## Development Workflow

### 1. Local Development
```bash
# Start development service
docker-compose up claude-agent-dev

# Make changes to code (hot reload enabled)
# View logs in real-time
docker-compose logs -f claude-agent-dev
```

### 2. Testing
```bash
# Run tests
docker-compose up claude-agent-test

# Run specific test file
docker-compose run claude-agent-test python -m pytest tests/test_cli.py -v
```

### 3. Production Deployment
```bash
# Build production image
docker build -t claude-agent-to-agent:latest .

# Run with production settings
docker run -d --env-file .env -p 8000:8000 claude-agent-to-agent:latest
```

## Advanced Configuration

### Custom Dockerfile
You can create a custom Dockerfile for specific needs:

```dockerfile
FROM claude-agent-to-agent:latest

# Add custom dependencies
RUN pip install additional-package

# Custom configuration
ENV CUSTOM_SETTING=value
```

### Docker Compose Overrides
Create `docker-compose.override.yml` for local customizations:

```yaml
version: '3.8'
services:
  claude-agent:
    environment:
      - DEBUG=true
      - VERBOSE=true
    volumes:
      - ./custom-config:/app/config
```

## Performance Optimization

### Resource Limits
Add to `docker-compose.yml`:
```yaml
services:
  claude-agent:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
```

### Caching
```bash
# Use build cache
docker-compose build --no-cache

# Use volume for pip cache
volumes:
  - pip-cache:/root/.cache/pip
```

## Monitoring

### Health Checks
The container includes health checks that monitor:
- Python process availability
- Basic functionality tests
- Resource usage

### Logging
```bash
# View real-time logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f claude-agent

# Export logs
docker-compose logs > logs.txt
```

## Cleanup

```bash
# Stop all services
docker-compose down

# Remove containers and networks
docker-compose down --volumes --remove-orphans

# Clean up images
docker system prune -f

# Remove all project data
docker-compose down -v
rm -rf logs/ data/
```

## Support

For issues related to Docker setup:
1. Check the troubleshooting section above
2. Review Docker and Docker Compose logs
3. Ensure all prerequisites are met
4. Verify environment variables are correctly set

## Contributing

When contributing to the Docker setup:
1. Test changes locally first
2. Update this README if needed
3. Ensure backward compatibility
4. Add appropriate tests for new features
