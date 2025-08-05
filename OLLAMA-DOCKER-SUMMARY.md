# 🚀 Ollama Docker Configuration for macOS M4 Max - Complete Setup

## 📋 Overview

This repository provides a **production-ready, security-hardened Docker deployment** for running Ollama LLM server on Apple Silicon M4 Max systems with 48GB unified memory. The configuration prioritizes security, performance optimization, and ease of deployment while acknowledging the CPU-only limitation of Docker containers on macOS.

## 🎯 Key Features

- ✅ **Security-first architecture**: Non-root user, read-only filesystem, capability dropping
- ✅ **M4 Max optimization**: CPU-only inference tuned for Apple Silicon performance
- ✅ **One-script deployment**: Automated setup with comprehensive error handling
- ✅ **Production-ready**: Resource limits, health checks, logging, monitoring
- ✅ **Comprehensive documentation**: Setup guides, troubleshooting, security verification

## 📁 Files Created

### Core Configuration Files
```
├── Dockerfile.ollama              # Security-hardened Ollama container
├── docker-compose.ollama.yml      # Production compose configuration
├── ollama-config.yaml            # Performance-optimized Ollama settings
└── setup-ollama-docker.sh        # Automated setup script
```

### Documentation & Tools
```
├── README-OLLAMA-DOCKER.md       # Comprehensive documentation
├── ollama_client_example.py      # Python client with examples
├── security-check.sh             # Security verification script
└── OLLAMA-DOCKER-SUMMARY.md      # This summary document
```

## 🚀 Quick Start

### 1. Prerequisites Check
```bash
# Verify system requirements
system_profiler SPHardwareDataType | grep "Memory:"    # Should show 48GB+
uname -m                                               # Should show arm64
docker --version                                       # Docker 20.10+
```

### 2. One-Command Setup
```bash
# Run the automated setup script
./setup-ollama-docker.sh
```

### 3. Verify Security
```bash
# Run security verification
./security-check.sh
```

### 4. Test Installation
```bash
# Test with Python client
python ollama_client_example.py

# Or test with curl
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2:7b-q4_0","messages":[{"role":"user","content":"Hello!"}]}'
```

## 🔒 Security Features

### Container Hardening
| Feature | Implementation | Security Benefit |
|---------|----------------|------------------|
| **Non-root execution** | `USER ollama` (UID 1000) | Prevents privilege escalation |
| **Read-only filesystem** | `--read-only` flag | Immutable container state |
| **Capability dropping** | `--cap-drop ALL` | Minimal Linux capabilities |
| **No new privileges** | `--security-opt no-new-privileges` | Prevents setuid exploitation |
| **Resource limits** | Memory: 32GB, CPU: 8 cores | DoS protection |
| **Network isolation** | Localhost binding only | No external exposure |

### Verification Tools
- **Automated security checks**: `security-check.sh` validates all security settings
- **Health monitoring**: Built-in Docker health checks with API validation
- **Log analysis**: Structured logging with rotation and compression
- **Vulnerability scanning**: Integration with Docker Scout/scan tools

## ⚡ Performance Configuration

### M4 Max Optimizations
```yaml
# Resource allocation (optimized for 48GB M4 Max)
Memory: 32GB (leaves 16GB for macOS)
CPU Cores: 8 (uses 8 of 14 available cores)
Thread Count: 8 (matches allocated cores)
Model Memory: 70% of container memory
Context Length: 4096-8192 tokens
```

### Model Recommendations
```bash
# Balanced performance models for M4 Max CPU inference
llama2:7b-q4_0      # 4.1GB - Good balance of quality/speed
codellama:7b-q4_0   # 4.1GB - Code generation optimized
phi3:mini           # 2.3GB - Fast responses, good for testing
llama2:13b-q2_k     # 5.4GB - Larger model, more capability
```

## 🛠️ Usage Examples

### Basic Model Management
```bash
# List available models
docker exec ollama-m4max ollama list

# Pull new models
docker exec ollama-m4max ollama pull mistral:7b-q4_0

# Remove unused models
docker exec ollama-m4max ollama rm llama2:7b-q4_0
```

### API Integration
```python
# Python client example
from ollama_client_example import OllamaClient

client = OllamaClient()
response = client.chat(
    model="llama2:7b-q4_0",
    messages=[{"role": "user", "content": "Explain Docker security"}]
)
print(response)
```

### Container Management
```bash
# Monitor resource usage
docker stats ollama-m4max

# View logs
docker logs -f ollama-m4max

# Restart with new configuration
docker-compose -f docker-compose.ollama.yml restart
```

## 🔧 Configuration Options

### Environment Variables
```bash
# Core Ollama settings
OLLAMA_HOME=/ollama                    # Model storage location
OLLAMA_HOST=0.0.0.0:11434             # Server binding
OLLAMA_ORIGINS=*                      # CORS origins
OLLAMA_NUM_PARALLEL=4                 # Parallel requests
OLLAMA_MAX_LOADED_MODELS=2            # Memory management

# Security settings
OLLAMA_NO_NETWORK=1                   # Offline mode (optional)
OLLAMA_DEBUG=0                        # Debug logging
```

### Volume Mounts
```yaml
volumes:
  # External SSD for models (recommended)
  - /Volumes/SSD1/ollama_models:/ollama:rw
  
  # Configuration file
  - ./ollama-config.yaml:/ollama/config.yaml:ro
  
# Secure temporary storage
tmpfs:
  - /tmp:size=2g,noexec,nosuid,nodev
  - /var/tmp:size=1g,noexec,nosuid,nodev
```

## 🚨 Important Limitations

### Metal GPU Support
```
⚠️  CRITICAL: Docker containers on macOS cannot access Metal GPU acceleration
    
    • CPU-only inference: ~10-50 tokens/second (depending on model)
    • Metal GPU inference: ~100-300 tokens/second (native Ollama)
    
    For maximum performance, consider running Ollama natively:
    curl -fsSL https://ollama.ai/install.sh | sh
```

### Resource Considerations
```
• Memory: 32GB container limit (leaves 16GB for macOS)
• CPU: 8 cores allocated (leaves 6 cores for system)
• Storage: Models require 3-15GB each
• Network: Localhost only (security isolation)
```

## 🔍 Troubleshooting

### Common Issues & Solutions

#### Container Won't Start
```bash
# Check Docker daemon
docker info

# Verify system resources
vm_stat | head -5

# Check logs
docker logs ollama-m4max

# Rebuild image
docker build -f Dockerfile.ollama -t local/ollama:0.3.12 .
```

#### Slow Performance
```bash
# Check resource usage
docker stats ollama-m4max

# Use smaller quantized models
docker exec ollama-m4max ollama pull llama2:7b-q2_k

# Increase thread count (edit ollama-config.yaml)
cpu:
  num_threads: 10
```

#### Memory Issues
```bash
# Check available memory
docker exec ollama-m4max cat /proc/meminfo | grep Available

# Reduce model memory fraction
memory:
  model_memory_fraction: 0.6

# Limit concurrent models
models:
  max_loaded: 1
```

## 📊 Performance Benchmarks

### Expected Performance (M4 Max, CPU-only)
| Model | Size | Tokens/Second | Memory Usage | Use Case |
|-------|------|---------------|--------------|----------|
| `phi3:mini` | 2.3GB | 25-40 | 4GB | Quick responses |
| `llama2:7b-q4_0` | 4.1GB | 15-25 | 8GB | Balanced quality |
| `codellama:7b-q4_0` | 4.1GB | 12-20 | 8GB | Code generation |
| `llama2:13b-q2_k` | 5.4GB | 8-15 | 12GB | High quality |

### Comparison: Docker vs Native
```
Docker Container (CPU):     15-25 tokens/second
Native macOS (Metal GPU):   100-300 tokens/second
Performance Ratio:          ~10x faster with Metal GPU
```

## 🛡️ Security Verification

### Automated Security Checks
```bash
# Run comprehensive security verification
./security-check.sh

# Expected output:
# ✅ Passed: 15+ security checks
# ⚠️  Warnings: 2-3 minor issues
# ❌ Failed: 0 critical issues
```

### Manual Security Verification
```bash
# Verify non-root user
docker exec ollama-m4max whoami  # Should return: ollama

# Check read-only filesystem
docker inspect ollama-m4max --format='{{.HostConfig.ReadonlyRootfs}}'  # Should return: true

# Verify resource limits
docker inspect ollama-m4max --format='{{.HostConfig.Memory}}'  # Should return: 34359738368 (32GB)

# Check network binding
netstat -an | grep 11434  # Should show: 127.0.0.1:11434
```

## 🔄 Alternative Deployment Options

### 1. Native Ollama (Maximum Performance)
```bash
# For Metal GPU acceleration
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &

# Performance: ~10x faster than Docker
# Security: Less isolation
# Use case: Maximum performance needed
```

### 2. Hybrid Approach
```bash
# Run Ollama natively, containerize client applications
ollama serve &  # Native server with Metal GPU

# Docker containers connect to native server
docker run --network host my-app  # App connects to host:11434
```

### 3. Development vs Production
```bash
# Development: Docker for isolation and reproducibility
docker-compose -f docker-compose.ollama.yml up -d

# Production: Native for performance, Docker for apps
ollama serve &  # Native Ollama server
docker-compose up my-apps  # Containerized applications
```

## 📈 Monitoring & Maintenance

### Health Monitoring
```bash
# Container health status
docker inspect ollama-m4max --format='{{.State.Health.Status}}'

# API health check
curl -f http://localhost:11434/api/tags

# Resource monitoring
docker stats ollama-m4max --no-stream
```

### Log Management
```bash
# View structured logs
docker logs ollama-m4max | jq '.'

# Export logs for analysis
docker logs ollama-m4max > ollama-$(date +%Y%m%d).log

# Log rotation (automatic via configuration)
# - Max size: 100MB per file
# - Max files: 3 rotated files
# - Compression: enabled
```

### Updates & Maintenance
```bash
# Update Ollama version
# 1. Edit OLLAMA_VER in setup script
# 2. Update SHA256 checksum
# 3. Rebuild and redeploy
./setup-ollama-docker.sh

# Update models
docker exec ollama-m4max ollama pull llama2:latest

# Clean up unused resources
docker system prune -f
```

## 🎯 Conclusion

This Ollama Docker configuration provides a **secure, production-ready foundation** for running LLM inference on macOS M4 Max systems. While CPU-only inference is slower than native Metal GPU acceleration, the configuration excels in:

- **🔒 Security**: Comprehensive hardening with verification tools
- **🏗️ Reproducibility**: Consistent deployment across environments  
- **🛠️ Maintainability**: Clear documentation and automation
- **📊 Monitoring**: Built-in health checks and logging
- **🔧 Flexibility**: Easy configuration and model management

### When to Use This Setup

✅ **Recommended for:**
- Development and testing environments
- Security-sensitive deployments
- Multi-tenant or shared systems
- Reproducible ML pipelines
- Learning and experimentation

⚠️ **Consider alternatives for:**
- Maximum inference performance requirements
- Real-time applications needing <100ms response times
- Production workloads requiring Metal GPU acceleration

### Next Steps

1. **Deploy**: Run `./setup-ollama-docker.sh` to get started
2. **Secure**: Verify with `./security-check.sh`
3. **Test**: Use `ollama_client_example.py` for integration
4. **Monitor**: Set up log analysis and resource monitoring
5. **Optimize**: Tune configuration based on your workload

---

**📚 Documentation**: See `README-OLLAMA-DOCKER.md` for detailed instructions  
**🔧 Support**: Check troubleshooting section for common issues  
**🚀 Performance**: Consider native deployment for GPU acceleration  

**Happy prompting with secure, containerized LLMs! 🤖**