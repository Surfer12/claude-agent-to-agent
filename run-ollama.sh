#!/bin/bash

# Secure Ollama Docker Setup Script
# This script provides a one-liner to spin up a secure, model-ready Ollama container

set -euo pipefail

# Configuration
CONTAINER_NAME="ollama"
IMAGE_NAME="local/ollama:0.2.8"
HOST_PORT="127.0.0.1:11434"
CONTAINER_PORT="11434"
MODELS_DIR="${OLLAMA_MODELS_DIR:-./ollama_models}"
MEMORY_LIMIT="${OLLAMA_MEMORY:-32g}"
CPU_LIMIT="${OLLAMA_CPUS:-8}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to check if container exists
container_exists() {
    docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"
}

# Function to check if container is running
container_running() {
    docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"
}

# Function to build the Ollama image
build_image() {
    log "Building secure Ollama Docker image..."
    if docker build -f Dockerfile.ollama -t "${IMAGE_NAME}" .; then
        success "Ollama image built successfully"
    else
        error "Failed to build Ollama image"
        exit 1
    fi
}

# Function to create models directory
setup_models_dir() {
    if [[ ! -d "${MODELS_DIR}" ]]; then
        log "Creating models directory: ${MODELS_DIR}"
        mkdir -p "${MODELS_DIR}"
        success "Models directory created"
    else
        success "Models directory already exists: ${MODELS_DIR}"
    fi
}

# Function to run the secure Ollama container
run_container() {
    log "Starting secure Ollama container..."
    
    # The one-liner command with all security hardening
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --restart unless-stopped \
        --read-only \
        --tmpfs /tmp \
        -p "${HOST_PORT}:${CONTAINER_PORT}" \
        --mount type=bind,src="$(realpath "${MODELS_DIR}")",dst=/ollama \
        -e OLLAMA_NO_NETWORK=0 \
        --cap-drop ALL \
        --security-opt no-new-privileges:true \
        --memory "${MEMORY_LIMIT}" \
        --cpus "${CPU_LIMIT}" \
        "${IMAGE_NAME}"
    
    success "Ollama container started successfully"
}

# Function to pull a model
pull_model() {
    local model="${1:-llama2:7b-q8_0}"
    log "Pulling model: ${model}"
    
    if docker exec -it "${CONTAINER_NAME}" ollama pull "${model}"; then
        success "Model ${model} pulled successfully"
    else
        error "Failed to pull model ${model}"
        return 1
    fi
}

# Function to list available models
list_models() {
    log "Available models:"
    docker exec -it "${CONTAINER_NAME}" ollama list
}

# Function to test the Ollama service
test_service() {
    log "Testing Ollama service..."
    
    # Wait for service to be ready
    local max_attempts=30
    local attempt=1
    
    while [[ ${attempt} -le ${max_attempts} ]]; do
        if curl -s "http://127.0.0.1:11434/api/tags" > /dev/null 2>&1; then
            success "Ollama service is ready and responding"
            return 0
        fi
        
        log "Waiting for Ollama service... (attempt ${attempt}/${max_attempts})"
        sleep 2
        ((attempt++))
    done
    
    error "Ollama service failed to start within expected time"
    return 1
}

# Function to stop and remove container
stop_container() {
    if container_running; then
        log "Stopping Ollama container..."
        docker stop "${CONTAINER_NAME}"
        success "Container stopped"
    fi
    
    if container_exists; then
        log "Removing Ollama container..."
        docker rm "${CONTAINER_NAME}"
        success "Container removed"
    fi
}

# Function to show container logs
show_logs() {
    if container_exists; then
        docker logs -f "${CONTAINER_NAME}"
    else
        error "Container ${CONTAINER_NAME} does not exist"
    fi
}

# Function to show usage
usage() {
    cat << EOF
Secure Ollama Docker Setup

Usage: $0 [COMMAND]

Commands:
    start       Build image and start secure Ollama container
    stop        Stop and remove Ollama container
    restart     Restart Ollama container
    pull MODEL  Pull a specific model (default: llama2:7b-q8_0)
    list        List available models
    test        Test if Ollama service is responding
    logs        Show container logs
    shell       Open shell in container
    help        Show this help message

Environment Variables:
    OLLAMA_MODELS_DIR   Directory for model storage (default: ./ollama_models)
    OLLAMA_MEMORY       Memory limit (default: 32g)
    OLLAMA_CPUS         CPU limit (default: 8)

Examples:
    $0 start                    # Start Ollama with default settings
    $0 pull llama2:13b         # Pull a specific model
    $0 pull codellama:7b       # Pull CodeLlama model
    
Security Features:
    ✓ Non-root user inside container
    ✓ Read-only root filesystem
    ✓ All capabilities dropped
    ✓ No new privileges
    ✓ Resource limits (memory/CPU)
    ✓ Network isolation option
    ✓ Bind mount for model persistence

EOF
}

# Main script logic
main() {
    case "${1:-help}" in
        "start")
            setup_models_dir
            
            if container_running; then
                warning "Container is already running"
                exit 0
            fi
            
            if container_exists; then
                log "Removing existing container..."
                docker rm "${CONTAINER_NAME}"
            fi
            
            # Check if image exists, build if not
            if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
                build_image
            fi
            
            run_container
            test_service
            
            success "Ollama is ready! Access it at http://127.0.0.1:11434"
            log "To pull a model, run: $0 pull llama2:7b-q8_0"
            ;;
        "stop")
            stop_container
            ;;
        "restart")
            stop_container
            sleep 2
            main start
            ;;
        "pull")
            if ! container_running; then
                error "Container is not running. Start it first with: $0 start"
                exit 1
            fi
            pull_model "${2:-llama2:7b-q8_0}"
            ;;
        "list")
            if ! container_running; then
                error "Container is not running. Start it first with: $0 start"
                exit 1
            fi
            list_models
            ;;
        "test")
            test_service
            ;;
        "logs")
            show_logs
            ;;
        "shell")
            if ! container_running; then
                error "Container is not running. Start it first with: $0 start"
                exit 1
            fi
            docker exec -it "${CONTAINER_NAME}" /bin/sh
            ;;
        "build")
            build_image
            ;;
        "help"|*)
            usage
            ;;
    esac
}

# Run main function with all arguments
main "$@"