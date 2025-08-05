# Secure Ollama Docker Setup

🔒 **TL;DR** – One-liner to spin up a secure, model-ready Ollama container with best-practice hardening.

## Quick Start

```bash
# Make the script executable
chmod +x run-ollama.sh

# Start Ollama with all security hardening
./run-ollama.sh start

# Pull a model
./run-ollama.sh pull llama2:7b-q8_0

# Test the service
curl http://127.0.0.1:11434/api/tags
```

## 🎯 What You Get

- **Strong Isolation**: Read-only root filesystem, dropped capabilities, no new privileges
- **Resource Control**: Configurable memory (32GB) and CPU (8 cores) limits  
- **Security Hardening**: Non-root user, minimal attack surface
- **Reproducibility**: Consistent environment across deployments
- **Model Persistence**: Bind-mounted model storage
- **Health Monitoring**: Built-in health checks and service testing

## 🛡️ Security Features

| Feature | Description |
|---------|-------------|
| **Non-root User** | Container runs as user `ollama` (UID 1000) |
| **Read-only Root FS** | Prevents runtime modifications to container filesystem |
| **Dropped Capabilities** | `--cap-drop ALL` removes all Linux capabilities |
| **No New Privileges** | Prevents privilege escalation attacks |
| **Resource Limits** | Memory and CPU constraints prevent resource exhaustion |
| **Network Isolation** | Configurable network access via `OLLAMA_NO_NETWORK` |
| **Minimal Base Image** | Alpine Linux for reduced attack surface |

## 📁 Files Overview

```
├── Dockerfile.ollama          # Secure multi-stage Ollama image
├── docker-compose.ollama.yml  # Docker Compose configuration
├── run-ollama.sh             # Main management script
└── README-ollama.md          # This documentation
```

## 🚀 Usage

### Basic Commands

```bash
# Start the service
./run-ollama.sh start

# Stop and remove container
./run-ollama.sh stop

# Restart the service
./run-ollama.sh restart

# View logs
./run-ollama.sh logs

# Open shell in container
./run-ollama.sh shell
```

### Model Management

```bash
# Pull default model (llama2:7b-q8_0)
./run-ollama.sh pull

# Pull specific model
./run-ollama.sh pull llama2:13b
./run-ollama.sh pull codellama:7b
./run-ollama.sh pull mistral:7b

# List installed models
./run-ollama.sh list
```

### Configuration

Environment variables for customization:

```bash
# Custom model directory
export OLLAMA_MODELS_DIR="/path/to/your/models"

# Resource limits
export OLLAMA_MEMORY="16g"
export OLLAMA_CPUS="4"

# Then start
./run-ollama.sh start
```

## 🐳 Docker Compose Alternative

For those preferring Docker Compose:

```bash
# Edit docker-compose.ollama.yml to set your models path
# Then start with:
docker-compose -f docker-compose.ollama.yml up -d

# Pull models
docker exec -it ollama ollama pull llama2:7b-q8_0
```

## 🔧 Manual Docker Command

The core secure Docker command (as referenced in your original request):

```bash
docker run -d \
  --name ollama \
  --restart unless-stopped \
  --read-only \
  --tmpfs /tmp \
  -p 127.0.0.1:11434:11434 \
  --mount type=bind,src=/path/to/ollama_models,dst=/ollama \
  -e OLLAMA_NO_NETWORK=0 \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --memory "32g" \
  --cpus "8" \
  local/ollama:0.2.8
```

## 🍎 macOS Metal GPU Acceleration

As noted in your original message, Docker containers on macOS cannot access Metal GPU acceleration. For GPU-accelerated inference:

1. **Install native Ollama** on macOS for Metal acceleration
2. **Use containers as clients** pointing to `http://host.docker.internal:11434`
3. **Keep containers for isolation** while heavy inference runs natively

### Hybrid Setup Example

```bash
# Terminal 1: Run native Ollama with Metal acceleration
ollama serve

# Terminal 2: Use containerized clients
docker run --rm -it \
  --add-host host.docker.internal:host-gateway \
  your-app:latest \
  --ollama-url http://host.docker.internal:11434
```

## 🔍 API Usage

Once running, Ollama provides a REST API:

```bash
# List models
curl http://127.0.0.1:11434/api/tags

# Generate text
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "llama2:7b-q8_0",
  "prompt": "Why is the sky blue?",
  "stream": false
}'

# Chat completion
curl http://127.0.0.1:11434/api/chat -d '{
  "model": "llama2:7b-q8_0",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```

## 🐛 Troubleshooting

### Container Won't Start
```bash
# Check logs
./run-ollama.sh logs

# Verify Docker is running
docker --version

# Check available resources
docker system df
```

### Model Download Issues
```bash
# Check network connectivity
docker exec -it ollama ping -c 3 ollama.com

# Verify disk space
df -h ./ollama_models

# Manual model pull with verbose output
docker exec -it ollama ollama pull llama2:7b-q8_0 --verbose
```

### Permission Issues
```bash
# Fix model directory permissions
sudo chown -R $(id -u):$(id -g) ./ollama_models

# Check container user
docker exec -it ollama id
```

## 📊 Resource Requirements

| Model Size | RAM Usage | Disk Space | Recommended CPU |
|------------|-----------|------------|-----------------|
| 7B Q4      | ~4GB      | ~4GB       | 4+ cores        |
| 7B Q8      | ~8GB      | ~7GB       | 4+ cores        |
| 13B Q4     | ~8GB      | ~8GB       | 8+ cores        |
| 13B Q8     | ~16GB     | ~14GB      | 8+ cores        |

## 🔐 Security Considerations

1. **Network Binding**: Default binds to `127.0.0.1` only (localhost)
2. **Model Storage**: Ensure model directory has appropriate permissions
3. **Resource Limits**: Prevent DoS via memory/CPU exhaustion
4. **Container Updates**: Regularly rebuild image for security patches
5. **Firewall**: Consider additional network restrictions if needed

## 📝 Popular Models to Try

```bash
# Coding assistance
./run-ollama.sh pull codellama:7b
./run-ollama.sh pull codellama:13b

# General purpose
./run-ollama.sh pull llama2:7b
./run-ollama.sh pull llama2:13b
./run-ollama.sh pull mistral:7b

# Specialized models
./run-ollama.sh pull vicuna:7b
./run-ollama.sh pull orca-mini:3b
```

## 🎉 Happy Prompting!

You now have a secure, reproducible local LLM service running in Docker with enterprise-grade security hardening. The setup follows best practices for:

- ✅ Container security
- ✅ Resource management  
- ✅ Data persistence
- ✅ Network isolation
- ✅ Operational monitoring

Perfect for experimenting with open-source GPT-style models while keeping your 48GB M4-Max system safe and secure! 🚀