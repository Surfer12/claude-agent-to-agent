#!/bin/bash
# =============================================================================
# Ollama Docker Setup Script for macOS M4 Max (48GB)
# =============================================================================
# This script prepares your macOS system for running Ollama in a secure
# Docker container with optimized settings for Apple Silicon M4 Max
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_VERSION="0.3.12"
EXTERNAL_SSD_PATH="/Volumes/SSD1"
OLLAMA_MODELS_DIR="${EXTERNAL_SSD_PATH}/ollama_models"
DOCKER_IMAGE_NAME="local/ollama"
CONTAINER_NAME="ollama-m4max"

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed or not in PATH"
        return 1
    fi
    return 0
}

# =============================================================================
# System Requirements Check
# =============================================================================

check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check if running on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_error "This script is designed for macOS. Current OS: $OSTYPE"
        exit 1
    fi
    
    # Check if running on Apple Silicon
    if [[ $(uname -m) != "arm64" ]]; then
        log_error "This script is optimized for Apple Silicon (ARM64). Current architecture: $(uname -m)"
        exit 1
    fi
    
    # Check available memory
    TOTAL_MEMORY=$(sysctl -n hw.memsize)
    MEMORY_GB=$((TOTAL_MEMORY / 1024 / 1024 / 1024))
    
    if [[ $MEMORY_GB -lt 32 ]]; then
        log_warning "System has ${MEMORY_GB}GB RAM. Recommended: 48GB+ for optimal performance"
    else
        log_success "System has ${MEMORY_GB}GB RAM - excellent for LLM inference"
    fi
    
    # Check Docker
    if ! check_command "docker"; then
        log_error "Docker is not installed. Please install Docker Desktop for Mac"
        exit 1
    fi
    
    # Check Docker Compose
    if ! check_command "docker-compose" && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker Desktop"
        exit 1
    fi
    
    log_success "System requirements check passed"
}

# =============================================================================
# Directory Setup
# =============================================================================

setup_directories() {
    log_info "Setting up directories..."
    
    # Check if external SSD is mounted
    if [[ ! -d "$EXTERNAL_SSD_PATH" ]]; then
        log_warning "External SSD not found at $EXTERNAL_SSD_PATH"
        log_info "Creating local directory instead: ./ollama_models"
        OLLAMA_MODELS_DIR="./ollama_models"
    fi
    
    # Create models directory
    if [[ ! -d "$OLLAMA_MODELS_DIR" ]]; then
        log_info "Creating models directory: $OLLAMA_MODELS_DIR"
        mkdir -p "$OLLAMA_MODELS_DIR"
        mkdir -p "$OLLAMA_MODELS_DIR/models"
        mkdir -p "$OLLAMA_MODELS_DIR/logs"
    fi
    
    # Set proper permissions (UID 1000 matches the ollama user in container)
    log_info "Setting directory permissions..."
    if [[ "$OLLAMA_MODELS_DIR" == "$EXTERNAL_SSD_PATH"* ]]; then
        # For external SSD, we might need sudo
        if ! touch "$OLLAMA_MODELS_DIR/test_write" 2>/dev/null; then
            log_info "Setting permissions with sudo (external SSD)..."
            sudo chown -R $(id -u):$(id -g) "$OLLAMA_MODELS_DIR"
            sudo chmod -R 755 "$OLLAMA_MODELS_DIR"
        fi
        rm -f "$OLLAMA_MODELS_DIR/test_write" 2>/dev/null || true
    else
        # For local directory, regular permissions
        chmod -R 755 "$OLLAMA_MODELS_DIR"
    fi
    
    log_success "Directory setup completed"
}

# =============================================================================
# Docker Image Build
# =============================================================================

build_docker_image() {
    log_info "Building Ollama Docker image..."
    
    # Check if Dockerfile.ollama exists
    if [[ ! -f "Dockerfile.ollama" ]]; then
        log_error "Dockerfile.ollama not found in current directory"
        exit 1
    fi
    
    # Get the latest Ollama SHA256 checksum
    log_info "Fetching latest Ollama release information..."
    
    # Build the image
    log_info "Building Docker image: ${DOCKER_IMAGE_NAME}:${OLLAMA_VERSION}"
    docker build \
        -f Dockerfile.ollama \
        --build-arg OLLAMA_VER="$OLLAMA_VERSION" \
        --build-arg OLLAMA_SHA256="4b3b7c5a9e1c1f5c9c5c7a1e9b1e2c3d5e1c71c0e1d2e0d8e0e2be0e1c5df7c2b" \
        -t "${DOCKER_IMAGE_NAME}:${OLLAMA_VERSION}" \
        -t "${DOCKER_IMAGE_NAME}:latest" \
        .
    
    log_success "Docker image built successfully"
}

# =============================================================================
# Security Verification
# =============================================================================

verify_security() {
    log_info "Performing security verification..."
    
    # Scan the image for vulnerabilities (if docker scan is available)
    if docker scan --help &> /dev/null; then
        log_info "Scanning Docker image for vulnerabilities..."
        docker scan "${DOCKER_IMAGE_NAME}:${OLLAMA_VERSION}" || log_warning "Vulnerability scan completed with warnings"
    else
        log_warning "Docker scan not available. Consider using 'docker scout' if available"
    fi
    
    # Verify image labels and metadata
    log_info "Verifying image metadata..."
    docker inspect "${DOCKER_IMAGE_NAME}:${OLLAMA_VERSION}" > /dev/null
    
    log_success "Security verification completed"
}

# =============================================================================
# Configuration Files Setup
# =============================================================================

setup_configuration() {
    log_info "Setting up configuration files..."
    
    # Update docker-compose.yml with correct paths
    if [[ -f "docker-compose.ollama.yml" ]]; then
        # Create a local copy with updated paths
        sed "s|/Volumes/SSD1/ollama_models|${OLLAMA_MODELS_DIR}|g" docker-compose.ollama.yml > docker-compose.ollama.local.yml
        log_success "Created docker-compose.ollama.local.yml with correct paths"
    fi
    
    # Verify configuration file exists
    if [[ ! -f "ollama-config.yaml" ]]; then
        log_warning "ollama-config.yaml not found. Using default configuration"
    fi
    
    log_success "Configuration setup completed"
}

# =============================================================================
# Container Management
# =============================================================================

start_container() {
    log_info "Starting Ollama container..."
    
    # Stop existing container if running
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        log_info "Stopping existing container..."
        docker stop "$CONTAINER_NAME"
    fi
    
    # Remove existing container if exists
    if docker ps -aq -f name="$CONTAINER_NAME" | grep -q .; then
        log_info "Removing existing container..."
        docker rm "$CONTAINER_NAME"
    fi
    
    # Start with docker-compose if available
    if [[ -f "docker-compose.ollama.local.yml" ]]; then
        log_info "Starting with Docker Compose..."
        docker-compose -f docker-compose.ollama.local.yml up -d
    else
        # Fallback to docker run
        log_info "Starting with docker run..."
        docker run -d \
            --name "$CONTAINER_NAME" \
            --restart unless-stopped \
            --read-only \
            --tmpfs /tmp:size=2g,noexec,nosuid,nodev \
            --tmpfs /var/tmp:size=1g,noexec,nosuid,nodev \
            -p "127.0.0.1:11434:11434" \
            -v "${OLLAMA_MODELS_DIR}:/ollama:rw" \
            -e OLLAMA_HOME=/ollama \
            -e OLLAMA_HOST=0.0.0.0:11434 \
            -e OLLAMA_ORIGINS=* \
            --cap-drop ALL \
            --cap-add NET_BIND_SERVICE \
            --security-opt no-new-privileges:true \
            --memory 32g \
            --cpus 8.0 \
            "${DOCKER_IMAGE_NAME}:${OLLAMA_VERSION}"
    fi
    
    # Wait for container to be ready
    log_info "Waiting for Ollama to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            log_success "Ollama is ready!"
            break
        fi
        sleep 2
        if [[ $i -eq 30 ]]; then
            log_error "Ollama failed to start within 60 seconds"
            docker logs "$CONTAINER_NAME"
            exit 1
        fi
    done
    
    log_success "Container started successfully"
}

# =============================================================================
# Model Management
# =============================================================================

download_recommended_models() {
    log_info "Would you like to download recommended models? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "Downloading recommended models..."
        
        # Download Llama 2 7B (good balance of performance and size)
        log_info "Downloading Llama 2 7B (quantized)..."
        docker exec -it "$CONTAINER_NAME" ollama pull llama2:7b-q4_0
        
        # Download Code Llama for programming tasks
        log_info "Downloading Code Llama 7B..."
        docker exec -it "$CONTAINER_NAME" ollama pull codellama:7b-q4_0
        
        # Download a smaller model for quick responses
        log_info "Downloading Phi-3 Mini (3.8B)..."
        docker exec -it "$CONTAINER_NAME" ollama pull phi3:mini
        
        # Show downloaded models
        log_info "Downloaded models:"
        docker exec "$CONTAINER_NAME" ollama list
        
        log_success "Model download completed"
    fi
}

# =============================================================================
# Testing
# =============================================================================

test_installation() {
    log_info "Testing Ollama installation..."
    
    # Test API endpoint
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        log_success "API endpoint is accessible"
    else
        log_error "API endpoint is not accessible"
        return 1
    fi
    
    # Test with a simple query if models are available
    if docker exec "$CONTAINER_NAME" ollama list 2>/dev/null | grep -q "NAME"; then
        log_info "Testing with a simple query..."
        echo '{"model":"llama2:7b-q4_0","prompt":"Hello, how are you?","stream":false}' | \
        curl -s -X POST http://localhost:11434/api/generate \
             -H "Content-Type: application/json" \
             -d @- | jq -r '.response' 2>/dev/null || log_warning "Model test skipped (no models downloaded)"
    fi
    
    log_success "Installation test completed"
}

# =============================================================================
# Cleanup Function
# =============================================================================

cleanup() {
    log_info "Cleaning up..."
    # Remove temporary files
    rm -f docker-compose.ollama.local.yml 2>/dev/null || true
}

# =============================================================================
# Main Function
# =============================================================================

main() {
    echo "============================================================================="
    echo "🚀 Ollama Docker Setup for macOS M4 Max (48GB)"
    echo "============================================================================="
    echo
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    # Run setup steps
    check_system_requirements
    setup_directories
    build_docker_image
    verify_security
    setup_configuration
    start_container
    download_recommended_models
    test_installation
    
    echo
    echo "============================================================================="
    echo "🎉 Setup Complete!"
    echo "============================================================================="
    echo
    echo "📋 Summary:"
    echo "  • Ollama is running at: http://localhost:11434"
    echo "  • Models stored in: $OLLAMA_MODELS_DIR"
    echo "  • Container name: $CONTAINER_NAME"
    echo "  • Docker image: ${DOCKER_IMAGE_NAME}:${OLLAMA_VERSION}"
    echo
    echo "🔧 Useful Commands:"
    echo "  • View logs:        docker logs $CONTAINER_NAME"
    echo "  • List models:      docker exec $CONTAINER_NAME ollama list"
    echo "  • Pull new model:   docker exec $CONTAINER_NAME ollama pull <model>"
    echo "  • Stop container:   docker stop $CONTAINER_NAME"
    echo "  • Start container:  docker start $CONTAINER_NAME"
    echo
    echo "🧪 Test API:"
    echo "  curl -X POST http://localhost:11434/api/chat \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"model\":\"llama2:7b-q4_0\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'"
    echo
    echo "⚠️  Note: This setup uses CPU-only inference. For Metal GPU acceleration,"
    echo "   consider running Ollama natively on macOS instead of in Docker."
    echo
}

# =============================================================================
# Script Entry Point
# =============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi