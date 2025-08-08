# Quick Start Guide - Docker

Get up and running with Claude Agent-to-Agent using Docker in 5 minutes!

## Prerequisites

- Docker installed and running
- Anthropic API key

## Step 1: Set up Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env and add your API key
# Replace 'your_anthropic_api_key_here' with your actual API key
```

## Step 2: Build and Run

### Option A: Using the build script (Recommended)

```bash
# Make script executable
chmod +x docker-build.sh

# Check environment
./docker-build.sh check

# Build and run
./docker-build.sh build
./docker-build.sh run interactive
```

### Option B: Using Make

```bash
# Quick setup and run
make quick

# Or step by step
make setup
make run
```

### Option C: Using Docker Compose

```bash
# Start all services
docker-compose up --build

# Or run specific service
docker-compose up claude-agent-dev
```

### Option D: Direct Docker commands

```bash
# Build image
docker build -t claude-agent-to-agent .

# Run container
docker run -it --env-file .env claude-agent-to-agent
```

## Step 3: Verify Installation

Once the container is running, you should see the Claude Agent CLI prompt. Try:

```
Hello! What tools do you have available?
```

## Common Commands

### Development
```bash
# Development mode with hot reload
make dev

# Or with docker-compose
docker-compose up claude-agent-dev
```

### Testing
```bash
# Run tests
make test

# Or
./docker-build.sh test
```

### Management
```bash
# Stop containers
make stop

# View logs
make logs

# Clean up
make clean
```

## Troubleshooting

### Issue: "Docker is not running"
```bash
# Start Docker Desktop or Docker daemon
# On macOS: Open Docker Desktop
# On Linux: sudo systemctl start docker
```

### Issue: "Permission denied"
```bash
# Fix permissions
sudo chown -R $USER:$USER .
```

### Issue: "API key not found"
```bash
# Check .env file
cat .env
# Ensure ANTHROPIC_API_KEY is set correctly
```

### Issue: "Port already in use"
```bash
# Change port in docker-compose.yml
# Or stop existing containers
docker-compose down
```

## Next Steps

1. **Explore the CLI**: Try different commands and tools
2. **Read Documentation**: Check `DOCKER_README.md` for detailed information
3. **Customize**: Modify the Dockerfile or docker-compose.yml for your needs
4. **Deploy**: Use the production setup for deployment

## Support

- Check `DOCKER_README.md` for comprehensive documentation
- Review logs: `make logs` or `docker-compose logs`
- Run diagnostics: `./docker-build.sh check`

Happy coding! 🚀
