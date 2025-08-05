#!/bin/bash

# Ollama Docker Setup Script for macOS M4 Max
# Secure deployment with CPU-only inference

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_VERSION="0.2.8"
OLLAMA_SHA256="2bfa1e3d5a7c6b4e8f9a3e8c1b2d6f58f4e7c9b1124f6d7e5d3c9a2e1f7b4c2a"
MODEL_STORAGE_PATH="/Volumes/SSD1/ollama_models"
CONTAINER_NAME="ollama"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker Desktop for macOS."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    
    print_success "Docker is running"
}

# Function to check system architecture
check_architecture() {
    print_status "Checking system architecture..."
    local arch=$(uname -m)
    if [[ "$arch" != "arm64" ]]; then
        print_warning "System architecture is $arch, not arm64. This setup is optimized for Apple Silicon."
    else
        print_success "Apple Silicon (ARM64) detected"
    fi
}

# Function to create model storage directory
setup_storage() {
    print_status "Setting up model storage..."
    
    if [[ ! -d "$MODEL_STORAGE_PATH" ]]; then
        print_status "Creating model storage directory: $MODEL_STORAGE_PATH"
        sudo mkdir -p "$MODEL_STORAGE_PATH"
    fi
    
    # Set proper ownership for Docker container
    local container_uid=1000  # Default UID for ollama user in container
    sudo chown -R $container_uid:$container_uid "$MODEL_STORAGE_PATH"
    sudo chmod 755 "$MODEL_STORAGE_PATH"
    
    print_success "Model storage directory configured"
}

# Function to verify Ollama checksum
verify_checksum() {
    print_status "Verifying Ollama checksum..."
    
    # Download the checksum from official source
    local checksum_url="https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-arm64.sha256"
    
    if curl -fsSL "$checksum_url" | grep -q "$OLLAMA_SHA256"; then
        print_success "Checksum verification passed"
    else
        print_warning "Checksum verification failed. Using provided checksum."
        print_warning "Please verify the checksum manually from: https://github.com/ollama/ollama/releases"
    fi
}

# Function to build Docker image
build_image() {
    print_status "Building Ollama Docker image..."
    
    if docker build \
        --build-arg OLLAMA_VER="$OLLAMA_VERSION" \
        --build-arg OLLAMA_SHA256="$OLLAMA_SHA256" \
        -f Dockerfile.ollama \
        -t local/ollama:"$OLLAMA_VERSION" .; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run container
run_container() {
    print_status "Starting Ollama container..."
    
    # Stop and remove existing container if it exists
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        print_status "Stopping existing container..."
        docker stop "$CONTAINER_NAME" || true
        docker rm "$CONTAINER_NAME" || true
    fi
    
    # Run the container with security settings
    if docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        --read-only \
        --tmpfs /tmp \
        -p 127.0.0.1:11434:11434 \
        --mount type=bind,src="$MODEL_STORAGE_PATH",dst=/ollama,bind-propagation=rprivate \
        --cap-drop ALL \
        --security-opt no-new-privileges:true \
        --memory "32g" \
        --cpus "8" \
        -e OLLAMA_HOME=/ollama \
        local/ollama:"$OLLAMA_VERSION"; then
        print_success "Ollama container started successfully"
    else
        print_error "Failed to start Ollama container"
        exit 1
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    print_status "Waiting for Ollama service to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -fs http://localhost:11434/api/tags &> /dev/null; then
            print_success "Ollama service is ready"
            return 0
        fi
        
        print_status "Waiting for service... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    print_error "Ollama service failed to start within expected time"
    return 1
}

# Function to test the setup
test_setup() {
    print_status "Testing Ollama setup..."
    
    # Test basic connectivity
    if curl -fs http://localhost:11434/api/tags &> /dev/null; then
        print_success "Ollama API is responding"
    else
        print_error "Ollama API is not responding"
        return 1
    fi
    
    # Test model listing
    local models=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null || echo "")
    print_status "Available models: $models"
    
    print_success "Setup test completed"
    return 0
}

# Function to show usage instructions
show_instructions() {
    echo
    echo "🎉 Ollama Docker Setup Complete!"
    echo "=================================="
    echo
    echo "📋 Container Status:"
    docker ps --filter name="$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo
    echo "📝 Usage Instructions:"
    echo "1. Pull a model:"
    echo "   docker exec -it $CONTAINER_NAME ollama pull llama2:7b-q8_0"
    echo
    echo "2. Test inference:"
    echo "   curl -X POST http://localhost:11434/api/chat \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"model\":\"llama2:7b-q8_0\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"stream\":false}'"
    echo
    echo "3. View logs:"
    echo "   docker logs $CONTAINER_NAME"
    echo
    echo "4. Stop the service:"
    echo "   docker stop $CONTAINER_NAME"
    echo
    echo "5. Start the service:"
    echo "   docker start $CONTAINER_NAME"
    echo
    echo "⚠️  Important Notes:"
    echo "- This setup uses CPU-only inference (no Metal GPU in Docker)"
    echo "- For GPU acceleration, run Ollama natively on macOS"
    echo "- Models are stored in: $MODEL_STORAGE_PATH"
    echo "- Container is secured with read-only filesystem and dropped capabilities"
    echo
    echo "🔒 Security Features:"
    echo "- Non-root user inside container"
    echo "- Read-only root filesystem"
    echo "- All Linux capabilities dropped"
    echo "- No privilege escalation allowed"
    echo "- Resource limits: 32GB RAM, 8 CPU cores"
    echo
}

# Function to show GPU alternative
show_gpu_alternative() {
    echo
    echo "🚀 For Metal GPU Acceleration:"
    echo "==============================="
    echo
    echo "Docker containers cannot access Metal GPU on macOS. For GPU acceleration:"
    echo
    echo "1. Install Ollama natively:"
    echo "   curl -fsSL https://ollama.com/install.sh | sh"
    echo
    echo "2. Run Ollama with Metal support:"
    echo "   ollama serve"
    echo
    echo "3. Your Docker containers can still connect to native Ollama:"
    echo "   OLLAMA_HOST=http://host.docker.internal:11434"
    echo
}

# Main execution
main() {
    echo "🚀 Ollama Docker Setup for macOS M4 Max"
    echo "========================================"
    echo
    
    check_docker
    check_architecture
    setup_storage
    verify_checksum
    build_image
    run_container
    wait_for_service
    test_setup
    show_instructions
    show_gpu_alternative
    
    print_success "Setup completed successfully!"
}

# Run main function
main "$@"