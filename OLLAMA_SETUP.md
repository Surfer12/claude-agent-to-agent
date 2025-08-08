# Secure Ollama Docker Setup

This repository provides a secure, production-ready setup for running Ollama in Docker containers with comprehensive security hardening.

## 🚀 Quick Start

### Option 1: One-liner (Original)
```bash
docker run -d \
  --name ollama \
  --restart unless-stopped \
  --read-only \
  --tmpfs /tmp \
  -p 127.0.0.1:11434:11434 \
  --mount type=bind,src=/Volumes/SSD1/ollama_models,dst=/ollama \
  -e OLLAMA_NO_NETWORK=0 \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --memory "32g" \
  --cpus "8" \
  local/ollama:0.2.8
```

### Option 2: Using the Setup Script
```bash
# Start with Docker Compose (recommended)
./setup-ollama.sh compose

# Or start with Docker run command
./setup-ollama.sh docker

# Pull a model
./setup-ollama.sh pull llama2:7b-q8_0

# Check status
./setup-ollama.sh status
```

### Option 3: Docker Compose
```bash
docker-compose -f ollama-docker-compose.yml up -d
```

## 🔒 Security Features

### Container Hardening
- **Read-only filesystem**: Prevents malicious writes
- **Dropped capabilities**: Removes unnecessary privileges
- **No new privileges**: Prevents privilege escalation
- **Non-root user**: Runs as user 1000:1000
- **Resource limits**: Memory and CPU constraints
- **Isolated networking**: Only localhost access

### Resource Management
- **Memory limit**: 32GB maximum
- **CPU limit**: 8 cores maximum
- **Reservations**: 16GB memory, 4 cores minimum
- **Log rotation**: 10MB max, 3 files

### Network Security
- **Localhost binding**: Only accessible from localhost
- **Optional offline mode**: `OLLAMA_NO_NETWORK=1` for air-gapped environments
- **Bridge networking**: Isolated container network

## 📁 File Structure

```
.
├── setup-ollama.sh              # Main setup script
├── ollama-docker-compose.yml    # Docker Compose configuration
├── OLLAMA_SETUP.md             # This documentation
└── /Volumes/SSD1/ollama_models # Model storage (external)
```

## 🛠️ Usage

### Setup Script Commands

```bash
# Show help
./setup-ollama.sh help

# Start with Docker Compose
./setup-ollama.sh compose

# Start with Docker run
./setup-ollama.sh docker

# Pull a specific model
./setup-ollama.sh pull llama2:7b-q8_0

# Check container status
./setup-ollama.sh status

# Stop container
./setup-ollama.sh stop

# Restart container
./setup-ollama.sh restart
```

### Model Management

```bash
# Pull models
docker exec -it ollama ollama pull llama2:7b-q8_0
docker exec -it ollama ollama pull codellama:7b-instruct
docker exec -it ollama ollama pull mistral:7b-instruct

# List models
docker exec -it ollama ollama list

# Run inference
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2:7b-q8_0",
    "prompt": "Hello, how are you?",
    "stream": false
  }'
```

## 🔧 Configuration

### Environment Variables
- `OLLAMA_NO_NETWORK=0`: Allow network access (default)
- `OLLAMA_NO_NETWORK=1`: Disable network access (air-gapped)

### Resource Limits
- **Memory**: 32GB maximum, 16GB reserved
- **CPU**: 8 cores maximum, 4 cores reserved
- **Storage**: External volume for model persistence

### Port Configuration
- **Default**: `127.0.0.1:11434:11434`
- **External access**: Change to `0.0.0.0:11434:11434` (not recommended)

## 🚨 Security Considerations

### Best Practices
1. **Never run as root**: Container runs as non-root user
2. **Use read-only filesystem**: Prevents malicious writes
3. **Drop capabilities**: Removes unnecessary privileges
4. **Limit resources**: Prevents resource exhaustion
5. **Bind to localhost**: Prevents external access
6. **Use tmpfs for /tmp**: Prevents disk-based attacks

### Production Recommendations
- Use secrets management for API keys
- Implement proper logging and monitoring
- Regular security updates
- Network segmentation
- Backup strategies for models

## 🐛 Troubleshooting

### Common Issues

**Container won't start**
```bash
# Check Docker logs
docker logs ollama

# Check resource availability
docker stats ollama

# Verify image exists
docker images | grep ollama
```

**Model download fails**
```bash
# Check network connectivity
docker exec -it ollama ping google.com

# Check disk space
df -h /Volumes/SSD1/ollama_models

# Check permissions
ls -la /Volumes/SSD1/ollama_models
```

**Performance issues**
```bash
# Monitor resource usage
docker stats ollama

# Check memory usage
docker exec -it ollama free -h

# Adjust resource limits in docker-compose.yml
```

### Health Checks
```bash
# Check if service is responding
curl -f http://localhost:11434/api/tags

# Check container health
docker ps --filter "name=ollama"
```

## 🔄 Updates and Maintenance

### Updating Ollama
```bash
# Stop container
./setup-ollama.sh stop

# Pull new image
docker pull local/ollama:latest

# Restart with new image
./setup-ollama.sh docker
```

### Backup Models
```bash
# Backup models directory
tar -czf ollama_models_backup.tar.gz /Volumes/SSD1/ollama_models

# Restore models
tar -xzf ollama_models_backup.tar.gz -C /
```

## 📊 Performance Tuning

### For M4 Max Mac (48GB RAM)
- **Memory limit**: 32GB (leaves 16GB for system)
- **CPU limit**: 8 cores (utilizes M4 Max efficiently)
- **Model storage**: SSD for fast I/O

### For Different Hardware
- **16GB RAM**: Use 12GB memory limit
- **32GB RAM**: Use 24GB memory limit
- **8 cores**: Use 6 CPU limit
- **4 cores**: Use 3 CPU limit

## 🎯 Metal GPU Acceleration

For Metal GPU acceleration on macOS:

1. **Install native Ollama**:
   ```bash
   brew install ollama
   ```

2. **Run native Ollama**:
   ```bash
   ollama serve
   ```

3. **Point containers to native service**:
   ```bash
   # In your application
   OLLAMA_HOST=http://host.docker.internal:11434
   ```

## 📚 Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Model Library](https://ollama.ai/library)
- [API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This setup is provided as-is for educational and production use.