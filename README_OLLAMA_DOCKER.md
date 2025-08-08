# Ollama Docker Setup for macOS M4 Max

A secure, reproducible Docker deployment of Ollama optimized for Apple Silicon Macs with 48GB unified memory.

## 🎯 Overview

This setup provides a containerized Ollama environment with:
- **Security-first design** with non-root user, read-only filesystem, and dropped capabilities
- **CPU-only inference** (Metal GPU not available in Docker on macOS)
- **Persistent model storage** on external SSD
- **Resource limits** optimized for M4 Max (32GB RAM, 8 CPU cores)
- **Offline mode support** for air-gapped environments

## ⚠️ Important Limitations

### Metal GPU Support
Docker containers on macOS **cannot access the Metal GPU**. This setup provides CPU-only inference, which is slower but still functional for many use cases.

**For Metal GPU acceleration:**
1. Install Ollama natively: `curl -fsSL https://ollama.com/install.sh | sh`
2. Run with Metal: `ollama serve`
3. Connect containers to native Ollama: `OLLAMA_HOST=http://host.docker.internal:11434`

## 🚀 Quick Start

### Prerequisites
- Docker Desktop for macOS
- External SSD for model storage (recommended)
- 48GB+ unified memory (M4 Max)

### Automated Setup
```bash
# Make the setup script executable
chmod +x setup_ollama_docker.sh

# Run the automated setup
./setup_ollama_docker.sh
```

### Manual Setup
```bash
# 1. Create model storage directory
sudo mkdir -p /Volumes/SSD1/ollama_models
sudo chown -R 1000:1000 /Volumes/SSD1/ollama_models

# 2. Build the Docker image
docker build -f Dockerfile.ollama -t local/ollama:0.2.8 .

# 3. Run the container
docker run -d \
  --name ollama \
  --restart unless-stopped \
  --read-only \
  --tmpfs /tmp \
  -p 127.0.0.1:11434:11434 \
  --mount type=bind,src=/Volumes/SSD1/ollama_models,dst=/ollama \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --memory "32g" \
  --cpus "8" \
  local/ollama:0.2.8

# 4. Pull a model
docker exec -it ollama ollama pull llama2:7b-q8_0
```

## 📁 File Structure

```
.
├── Dockerfile.ollama          # Ollama container definition
├── docker-compose.ollama.yml  # Docker Compose configuration
├── ollama-config.yaml         # Ollama configuration file
├── test_ollama_client.py      # Python test client
├── setup_ollama_docker.sh     # Automated setup script
└── README_OLLAMA_DOCKER.md    # This file
```

## 🔒 Security Features

### Container Security
- **Non-root user**: Runs as `ollama` user (UID 1000)
- **Read-only filesystem**: Only `/ollama` and `/tmp` are writable
- **Dropped capabilities**: All Linux capabilities removed
- **No privilege escalation**: `no-new-privileges:true`
- **Resource limits**: 32GB RAM, 8 CPU cores max
- **Network isolation**: Binds to localhost only

### Verification
- **SHA256 checksum verification** of Ollama binary
- **Official source**: Downloads from GitHub releases
- **Version pinning**: Explicit version control

### Model Security
- **Persistent storage**: Models stored on encrypted external SSD
- **Offline mode**: Can disable network after model download
- **Access control**: Localhost-only API access

## 🛠️ Usage

### Basic Operations

```bash
# Check container status
docker ps --filter name=ollama

# View logs
docker logs ollama

# Pull a model
docker exec -it ollama ollama pull llama2:7b-q8_0

# List models
docker exec -it ollama ollama list

# Test inference
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2:7b-q8_0",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### Python Client Example

```python
import requests

url = "http://localhost:11434/api/chat"
payload = {
    "model": "llama2:7b-q8_0",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "stream": False
}

response = requests.post(url, json=payload)
print(response.json()["message"]["content"])
```

### Docker Compose Usage

```bash
# Start services
docker-compose -f docker-compose.ollama.yml up -d

# Test with included client
docker-compose -f docker-compose.ollama.yml --profile test up ollama-client

# Stop services
docker-compose -f docker-compose.ollama.yml down
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOME` | `/ollama` | Model storage directory |
| `OLLAMA_HOST` | `0.0.0.0:11434` | API host and port |
| `OLLAMA_NO_NETWORK` | `false` | Disable network access |

### Resource Limits

| Resource | Limit | Reason |
|----------|-------|--------|
| Memory | 32GB | Leaves 16GB for system |
| CPU | 8 cores | Optimized for M4 Max |
| Storage | External SSD | Fast I/O for models |

## 📊 Performance

### CPU-Only Performance (M4 Max)
- **7B model**: ~2-4 tokens/second
- **13B model**: ~1-2 tokens/second
- **Memory usage**: 8-16GB per model

### Comparison with Metal GPU
- **CPU-only**: 2-4 tokens/second
- **Metal GPU**: 20-50 tokens/second
- **Recommendation**: Use native Ollama for production workloads

## 🚨 Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check Docker logs
docker logs ollama

# Verify storage permissions
sudo chown -R 1000:1000 /Volumes/SSD1/ollama_models
```

**Model download fails:**
```bash
# Check network connectivity
docker exec -it ollama curl -I https://ollama.com

# Verify storage space
df -h /Volumes/SSD1/ollama_models
```

**API not responding:**
```bash
# Check if container is running
docker ps --filter name=ollama

# Test API directly
curl -f http://localhost:11434/api/tags
```

### Performance Issues

**Slow inference:**
- This is expected with CPU-only mode
- Consider running Ollama natively for GPU acceleration
- Reduce model size (use q4_0 instead of q8_0)

**High memory usage:**
- Check resource limits: `docker stats ollama`
- Reduce memory limit if needed
- Use smaller quantized models

## 🔄 Maintenance

### Updates

```bash
# Update Ollama version
# 1. Update OLLAMA_VER in Dockerfile.ollama
# 2. Update OLLAMA_SHA256 with new checksum
# 3. Rebuild image
docker build -f Dockerfile.ollama -t local/ollama:NEW_VERSION .

# 4. Restart container
docker stop ollama
docker rm ollama
# Run new container with updated image
```

### Backup

```bash
# Backup models
tar -czf ollama_models_backup.tar.gz /Volumes/SSD1/ollama_models

# Restore models
tar -xzf ollama_models_backup.tar.gz -C /
```

### Cleanup

```bash
# Remove unused models
docker exec -it ollama ollama rm MODEL_NAME

# Clean up Docker resources
docker system prune -f
```

## 🎯 Best Practices

### Security
1. **Always verify checksums** before building
2. **Use external SSD** for model storage
3. **Enable offline mode** after model download
4. **Monitor resource usage** regularly
5. **Keep base image updated**

### Performance
1. **Use quantized models** (q4_0, q8_0)
2. **Monitor memory usage** with `docker stats`
3. **Consider native Ollama** for GPU workloads
4. **Use appropriate model sizes** for your use case

### Operations
1. **Test thoroughly** before production
2. **Backup models** regularly
3. **Monitor logs** for issues
4. **Use health checks** in production
5. **Document custom configurations**

## 📚 Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Apple Silicon Docker Guide](https://docs.docker.com/desktop/mac/apple-silicon/)
- [Model Performance Benchmarks](https://ollama.ai/library)

## 🤝 Contributing

To improve this setup:

1. **Update checksums** when new Ollama versions are released
2. **Test on different Mac configurations**
3. **Add new security features**
4. **Improve performance optimizations**
5. **Document additional use cases**

## 📄 License

This setup is provided as-is for educational and development purposes. Please review all security settings before production use.

---

**Note**: This Docker setup provides excellent isolation and reproducibility but trades GPU acceleration for security. For production workloads requiring Metal GPU acceleration, consider running Ollama natively on macOS and connecting your applications to it via the API.