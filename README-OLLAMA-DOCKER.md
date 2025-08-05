# 🚀 Secure Ollama Docker Configuration for macOS M4 Max (48GB)

A production-ready, security-hardened Docker deployment for running Ollama LLM server on Apple Silicon M4 Max systems with 48GB unified memory.

## 🎯 Overview

This configuration provides a containerized Ollama deployment optimized for macOS M4 Max with:

- **🔒 Security-first approach**: Non-root user, read-only filesystem, capability dropping
- **🚀 Performance optimization**: CPU-only inference tuned for Apple Silicon
- **📦 Easy deployment**: One-script setup with comprehensive configuration
- **🛡️ Isolation**: Complete container isolation with resource limits
- **📊 Monitoring**: Health checks and structured logging

> **⚠️ Important**: Docker containers on macOS cannot access Metal GPU acceleration. For Metal performance, run Ollama natively on macOS. This Docker setup is optimized for CPU-only inference with maximum security.

## 📋 Prerequisites

### System Requirements
- **macOS**: Big Sur (11.0) or later
- **Hardware**: Apple Silicon (M1, M2, M3, M4) - optimized for M4 Max
- **Memory**: 32GB+ recommended (48GB+ optimal)
- **Storage**: 50GB+ free space for models
- **Docker**: Docker Desktop for Mac 4.0+

### Software Dependencies
```bash
# Check if you have the required tools
docker --version          # Docker 20.10+
docker-compose --version  # Docker Compose 2.0+
curl --version            # For API testing
jq --version              # For JSON parsing (optional)
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Navigate to your project directory
cd /path/to/your/project

# Run the automated setup script
./setup-ollama-docker.sh
```

The setup script will:
- ✅ Verify system requirements
- 📁 Create necessary directories
- 🔨 Build the Docker image
- 🔒 Apply security configurations
- 🚀 Start the container
- 📥 Optionally download models

### 2. Manual Setup (Alternative)

If you prefer manual control:

```bash
# 1. Create model storage directory
mkdir -p /Volumes/SSD1/ollama_models
# Or for local storage: mkdir -p ./ollama_models

# 2. Build the Docker image
docker build -f Dockerfile.ollama -t local/ollama:0.3.12 .

# 3. Start with Docker Compose
docker-compose -f docker-compose.ollama.yml up -d

# 4. Verify it's running
curl http://localhost:11434/api/tags
```

## 📁 File Structure

```
├── Dockerfile.ollama              # Secure Ollama container definition
├── docker-compose.ollama.yml      # Production-ready compose configuration
├── ollama-config.yaml            # Optimized Ollama settings
├── setup-ollama-docker.sh        # Automated setup script
└── README-OLLAMA-DOCKER.md       # This documentation
```

## 🔧 Configuration Details

### Docker Security Features

| Security Feature | Implementation | Benefit |
|-----------------|----------------|---------|
| **Non-root user** | `USER ollama` (UID 1000) | Prevents privilege escalation |
| **Read-only filesystem** | `--read-only` flag | Immutable container state |
| **Capability dropping** | `--cap-drop ALL` | Minimal Linux capabilities |
| **No new privileges** | `--security-opt no-new-privileges` | Prevents setuid exploitation |
| **Resource limits** | Memory: 32GB, CPU: 8 cores | Prevents resource exhaustion |
| **Network isolation** | Localhost binding only | No external exposure |
| **Tmpfs mounts** | `/tmp`, `/var/tmp` | Secure temporary storage |

### Performance Optimizations

| Setting | Value | Reasoning |
|---------|-------|-----------|
| **Memory limit** | 32GB | Leaves 16GB for macOS + other apps |
| **CPU cores** | 8 | Uses 8 of 14 M4 Max cores |
| **Thread count** | 8 | Matches allocated CPU cores |
| **Model memory fraction** | 70% | Optimal balance for large models |
| **Context length** | 4096-8192 | Balanced performance/memory usage |

## 🎮 Usage Examples

### Basic API Usage

```bash
# List available models
curl http://localhost:11434/api/tags

# Chat completion
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2:7b-q4_0",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms"}
    ]
  }'

# Generate completion
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codellama:7b-q4_0",
    "prompt": "Write a Python function to calculate fibonacci numbers:",
    "stream": false
  }'
```

### Model Management

```bash
# Download models
docker exec ollama-m4max ollama pull llama2:7b-q4_0
docker exec ollama-m4max ollama pull codellama:7b-q4_0
docker exec ollama-m4max ollama pull phi3:mini

# List downloaded models
docker exec ollama-m4max ollama list

# Remove a model
docker exec ollama-m4max ollama rm llama2:7b-q4_0

# Show model information
docker exec ollama-m4max ollama show llama2:7b-q4_0
```

### Container Management

```bash
# View container status
docker ps -f name=ollama-m4max

# View logs
docker logs ollama-m4max

# Monitor resource usage
docker stats ollama-m4max

# Stop/Start container
docker stop ollama-m4max
docker start ollama-m4max

# Restart with new configuration
docker-compose -f docker-compose.ollama.yml down
docker-compose -f docker-compose.ollama.yml up -d
```

## 🛡️ Security Best Practices

### 1. Network Security
```bash
# Container only binds to localhost
netstat -an | grep 11434
# Should show: 127.0.0.1:11434

# Firewall configuration (optional)
sudo pfctl -f /etc/pf.conf  # Reload firewall rules
```

### 2. Storage Security
```bash
# Verify model directory permissions
ls -la /Volumes/SSD1/ollama_models
# Should show: drwxr-xr-x ... user group ...

# Enable FileVault encryption for external SSD
diskutil apfs enableFileVault /Volumes/SSD1
```

### 3. Container Verification
```bash
# Verify security settings
docker inspect ollama-m4max | jq '.[0].HostConfig | {
  ReadonlyRootfs: .ReadonlyRootfs,
  SecurityOpt: .SecurityOpt,
  CapDrop: .CapDrop,
  Memory: .Memory,
  CpuQuota: .CpuQuota
}'
```

### 4. Offline Mode
```bash
# Enable offline mode after downloading models
docker-compose -f docker-compose.ollama.yml down
# Edit docker-compose.ollama.yml: uncomment OLLAMA_NO_NETWORK=1
docker-compose -f docker-compose.ollama.yml up -d
```

## 🔧 Advanced Configuration

### Custom Model Configuration

Edit `ollama-config.yaml` to customize model behavior:

```yaml
model_configs:
  "llama2:7b*":
    context_length: 4096
    temperature: 0.7
    top_p: 0.9
    
  "codellama:*":
    context_length: 8192
    temperature: 0.1
    top_p: 0.95
```

### Resource Tuning

For different system configurations:

```yaml
# docker-compose.ollama.yml
deploy:
  resources:
    limits:
      memory: 24g      # For 32GB systems
      cpus: "6.0"      # For M3 Pro
    reservations:
      memory: 12g
      cpus: "3.0"
```

### External SSD Configuration

```bash
# Format external SSD with APFS (encrypted)
diskutil eraseDisk APFS "OllamaModels" /dev/disk2

# Mount and configure
mkdir -p /Volumes/OllamaModels/ollama_models
chown $(id -u):$(id -g) /Volumes/OllamaModels/ollama_models

# Update docker-compose.yml volume path
volumes:
  - /Volumes/OllamaModels/ollama_models:/ollama:rw
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check Docker daemon
docker info

# Check logs
docker logs ollama-m4max

# Verify image
docker images | grep ollama

# Rebuild if necessary
docker build -f Dockerfile.ollama -t local/ollama:0.3.12 .
```

#### 2. Memory Issues
```bash
# Check system memory
vm_stat | head -5

# Reduce container memory limit
docker update --memory 24g ollama-m4max

# Check container memory usage
docker stats ollama-m4max --no-stream
```

#### 3. Model Download Failures
```bash
# Check network connectivity
curl -I https://ollama.ai

# Check disk space
df -h /Volumes/SSD1

# Manual download with retry
docker exec -it ollama-m4max ollama pull llama2:7b-q4_0
```

#### 4. API Connection Issues
```bash
# Test local connectivity
curl -v http://localhost:11434/api/tags

# Check port binding
lsof -i :11434

# Verify container networking
docker exec ollama-m4max netstat -ln | grep 11434
```

### Performance Optimization

#### Slow Inference
```bash
# Check CPU usage
top -pid $(docker inspect ollama-m4max --format '{{.State.Pid}}')

# Increase thread count (edit ollama-config.yaml)
cpu:
  num_threads: 10  # Use more cores

# Use smaller quantized models
docker exec ollama-m4max ollama pull llama2:7b-q2_k  # Smaller, faster
```

#### Memory Optimization
```bash
# Monitor memory usage
docker exec ollama-m4max cat /proc/meminfo

# Reduce model memory fraction (edit ollama-config.yaml)
memory:
  model_memory_fraction: 0.6  # Use less memory per model

# Limit concurrent models
models:
  max_loaded: 1  # Only one model at a time
```

## 🔄 Alternative: Native Ollama with Metal GPU

For maximum performance with Metal GPU acceleration:

```bash
# Install Ollama natively
curl -fsSL https://ollama.ai/install.sh | sh

# Verify Metal support
ollama serve &
curl http://localhost:11434/api/tags

# The native version will automatically use Metal GPU
# Check Activity Monitor -> GPU tab to verify usage
```

## 📊 Monitoring and Logging

### Container Health Monitoring
```bash
# Check health status
docker inspect ollama-m4max --format='{{.State.Health.Status}}'

# View health check logs
docker inspect ollama-m4max | jq '.[0].State.Health'

# Custom health check
curl -f http://localhost:11434/api/tags || echo "Service unhealthy"
```

### Log Analysis
```bash
# View structured logs
docker logs ollama-m4max | jq '.'

# Follow logs in real-time
docker logs -f ollama-m4max

# Export logs for analysis
docker logs ollama-m4max > ollama-$(date +%Y%m%d).log
```

### Performance Metrics
```bash
# Resource usage over time
docker stats ollama-m4max --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Model performance benchmarking
time curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2:7b-q4_0","prompt":"Count to 100","stream":false}'
```

## 🤝 Contributing

To improve this configuration:

1. **Test changes** in a development environment
2. **Update documentation** for any configuration changes
3. **Verify security** settings remain intact
4. **Benchmark performance** on different hardware
5. **Submit issues** for bugs or enhancement requests

## 📄 License

This configuration is provided under the MIT License. See individual component licenses:
- **Ollama**: Apache 2.0 License
- **Docker**: Docker License
- **Configuration files**: MIT License

## 🆘 Support

For issues specific to this Docker configuration:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review Docker and Ollama logs
3. Verify system requirements
4. Test with minimal configuration

For Ollama-specific issues:
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Ollama Documentation](https://ollama.ai/docs)

---

**🎯 Remember**: This Docker setup prioritizes security and isolation over raw performance. For maximum inference speed with Metal GPU acceleration, consider running Ollama natively on macOS while using this Docker configuration for development, testing, or when security isolation is paramount.