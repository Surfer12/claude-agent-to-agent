# 🚀 Secure Ollama Docker Setup - Complete Guide

Your original one-liner was excellent! I've enhanced it with additional security features, management tools, and comprehensive documentation. Here's what we've created:

## 📁 Complete File Structure

```
.
├── setup-ollama.sh              # 🎯 Main setup script (enhanced)
├── ollama-docker-compose.yml    # 🐳 Docker Compose configuration
├── test-ollama.py              # 🧪 Comprehensive test suite
├── ollama_client.py            # 🤖 Python client library
├── OLLAMA_SETUP.md             # 📚 Complete documentation
└── OLLAMA_SECURE_SETUP_SUMMARY.md  # 📋 This summary
```

## 🔒 Enhanced Security Features

### Your Original Security (✅ Maintained)
- `--read-only` - Read-only filesystem
- `--tmpfs /tmp` - Temporary filesystem for /tmp
- `--cap-drop ALL` - Drop all capabilities
- `--security-opt no-new-privileges:true` - Prevent privilege escalation
- `-p 127.0.0.1:11434:11434` - Localhost-only binding
- Resource limits (`--memory "32g"`, `--cpus "8"`)

### Additional Security Enhancements (🆕 Added)
- **Non-root user**: `user: "1000:1000"`
- **Health checks**: Automatic service monitoring
- **Log rotation**: 10MB max, 3 files
- **Resource reservations**: Minimum resource guarantees
- **Enhanced error handling**: Graceful failure recovery

## 🛠️ Three Ways to Run

### 1. Your Original One-liner (Still Works!)
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

### 2. Enhanced Setup Script (Recommended)
```bash
# Make executable
chmod +x setup-ollama.sh

# Start with Docker Compose (recommended)
./setup-ollama.sh compose

# Or start with Docker run
./setup-ollama.sh docker

# Pull a model
./setup-ollama.sh pull llama2:7b-q8_0

# Check status
./setup-ollama.sh status
```

### 3. Docker Compose (Production Ready)
```bash
docker-compose -f ollama-docker-compose.yml up -d
```

## 🧪 Testing & Validation

### Comprehensive Test Suite
```bash
# Run the test suite
python test-ollama.py
```

**Tests include:**
- ✅ Connection health check
- ✅ Model listing
- ✅ Basic inference
- ✅ Streaming inference
- ✅ Container status verification

### Python Client Library
```python
from ollama_client import OllamaClient

# Initialize client
client = OllamaClient()

# Generate text
response = client.generate("Hello, world!")

# Chat with history
messages = [
    {"role": "user", "content": "What is AI?"}
]
response = client.chat(messages)
```

## 📊 Performance Optimizations

### For M4 Max Mac (48GB RAM)
- **Memory**: 32GB limit (leaves 16GB for system)
- **CPU**: 8 cores (utilizes M4 Max efficiently)
- **Storage**: SSD for fast model I/O
- **Reservations**: 16GB memory, 4 cores minimum

### Resource Management
```yaml
# From docker-compose.yml
deploy:
  resources:
    limits:
      memory: 32G
      cpus: '8.0'
    reservations:
      memory: 16G
      cpus: '4.0'
```

## 🔧 Management Commands

### Setup Script Features
```bash
./setup-ollama.sh help          # Show all options
./setup-ollama.sh compose       # Start with Docker Compose
./setup-ollama.sh docker        # Start with Docker run
./setup-ollama.sh pull MODEL    # Pull specific model
./setup-ollama.sh status        # Check container status
./setup-ollama.sh stop          # Stop container
./setup-ollama.sh restart       # Restart container
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
  -d '{"model": "llama2:7b-q8_0", "prompt": "Hello!"}'
```

## 🎯 Metal GPU Acceleration

For optimal performance on M4 Max:

1. **Install native Ollama**:
   ```bash
   brew install ollama
   ```

2. **Run native service**:
   ```bash
   ollama serve
   ```

3. **Point containers to native service**:
   ```bash
   # In your application
   OLLAMA_HOST=http://host.docker.internal:11434
   ```

## 🚨 Security Best Practices

### Container Hardening
1. **Read-only filesystem** - Prevents malicious writes
2. **Dropped capabilities** - Removes unnecessary privileges
3. **No new privileges** - Prevents privilege escalation
4. **Non-root user** - Runs as user 1000:1000
5. **Resource limits** - Prevents resource exhaustion
6. **Localhost binding** - Prevents external access

### Production Recommendations
- Use secrets management for API keys
- Implement proper logging and monitoring
- Regular security updates
- Network segmentation
- Backup strategies for models

## 🔄 Maintenance & Updates

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

## 📈 Monitoring & Troubleshooting

### Health Checks
```bash
# Check service health
curl -f http://localhost:11434/api/tags

# Check container status
docker ps --filter "name=ollama"

# View logs
docker logs ollama
```

### Common Issues
- **Container won't start**: Check Docker logs and resource availability
- **Model download fails**: Check network connectivity and disk space
- **Performance issues**: Monitor resource usage and adjust limits

## 🎉 Quick Start Summary

1. **Start the service**:
   ```bash
   ./setup-ollama.sh compose
   ```

2. **Pull a model**:
   ```bash
   ./setup-ollama.sh pull llama2:7b-q8_0
   ```

3. **Test the setup**:
   ```bash
   python test-ollama.py
   ```

4. **Use the client**:
   ```bash
   python ollama_client.py
   ```

## 🏆 Key Benefits

### Security
- ✅ Comprehensive container hardening
- ✅ Resource isolation and limits
- ✅ Network security
- ✅ Non-root execution

### Usability
- ✅ Simple one-command setup
- ✅ Comprehensive testing
- ✅ Python client library
- ✅ Production-ready configuration

### Performance
- ✅ Optimized for M4 Max
- ✅ Resource reservations
- ✅ Health monitoring
- ✅ Metal GPU support option

Your original setup was already excellent! These enhancements add production-ready features while maintaining the same security posture. The setup script makes it even easier to manage, and the test suite ensures everything works correctly.

**Bottom line**: You now have a secure, reproducible, and production-ready local LLM service that's perfect for your 48GB M4-Max Mac! 🚀