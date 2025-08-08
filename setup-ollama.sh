#!/bin/bash

# Ollama Secure Setup Script
# This script provides multiple options for running Ollama in a secure Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_IMAGE="local/ollama:0.2.8"
CONTAINER_NAME="ollama"
PORT="11434"
MODELS_DIR="/Volumes/SSD1/ollama_models"
MEMORY_LIMIT="32g"
CPU_LIMIT="8"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if models directory exists
check_models_dir() {
    if [ ! -d "$MODELS_DIR" ]; then
        print_warning "Models directory $MODELS_DIR does not exist. Creating it..."
        mkdir -p "$MODELS_DIR"
    fi
}

# Function to stop and remove existing container
cleanup_existing() {
    if docker ps -a --format "table {{.Names}}" | grep -q "^$CONTAINER_NAME$"; then
        print_status "Stopping and removing existing container..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
}

# Function to run with Docker Compose
run_with_compose() {
    print_status "Starting Ollama with Docker Compose..."
    
    if [ -f "ollama-docker-compose.yml" ]; then
        docker-compose -f ollama-docker-compose.yml up -d
        print_status "Ollama started with Docker Compose!"
    else
        print_error "ollama-docker-compose.yml not found. Please ensure it exists in the current directory."
        exit 1
    fi
}

# Function to run with Docker run command
run_with_docker() {
    print_status "Starting Ollama with Docker run command..."
    
    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        --read-only \
        --tmpfs /tmp \
        -p "127.0.0.1:$PORT:$PORT" \
        --mount type=bind,src="$MODELS_DIR",dst=/ollama \
        -e OLLAMA_NO_NETWORK=0 \
        --cap-drop ALL \
        --security-opt no-new-privileges:true \
        --memory "$MEMORY_LIMIT" \
        --cpus "$CPU_LIMIT" \
        "$OLLAMA_IMAGE"
    
    print_status "Ollama started with Docker run!"
}

# Function to pull a model
pull_model() {
    local model_name="$1"
    if [ -z "$model_name" ]; then
        model_name="llama2:7b-q8_0"
    fi
    
    print_status "Pulling model: $model_name"
    docker exec -it "$CONTAINER_NAME" ollama pull "$model_name"
}

# Function to show container status
show_status() {
    print_status "Container status:"
    docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    print_status "Container logs (last 10 lines):"
    docker logs --tail 10 "$CONTAINER_NAME" 2>/dev/null || print_warning "Container not running"
}

# Function to show usage
show_usage() {
    echo -e "${BLUE}Ollama Secure Setup Script${NC}"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  compose     Start Ollama using Docker Compose"
    echo "  docker      Start Ollama using Docker run command"
    echo "  pull MODEL  Pull a specific model (default: llama2:7b-q8_0)"
    echo "  status      Show container status and logs"
    echo "  stop        Stop the Ollama container"
    echo "  restart     Restart the Ollama container"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 compose                    # Start with Docker Compose"
    echo "  $0 docker                     # Start with Docker run"
    echo "  $0 pull llama2:7b-q8_0       # Pull specific model"
    echo "  $0 status                     # Check container status"
    echo ""
    echo "Security Features:"
    echo "  - Read-only filesystem"
    echo "  - Dropped capabilities"
    echo "  - No new privileges"
    echo "  - Resource limits"
    echo "  - Non-root user"
    echo "  - Isolated networking"
}

# Main script logic
main() {
    check_docker
    check_models_dir
    
    case "${1:-help}" in
        "compose")
            cleanup_existing
            run_with_compose
            ;;
        "docker")
            cleanup_existing
            run_with_docker
            ;;
        "pull")
            pull_model "$2"
            ;;
        "status")
            show_status
            ;;
        "stop")
            print_status "Stopping Ollama container..."
            docker stop "$CONTAINER_NAME" 2>/dev/null || print_warning "Container not running"
            ;;
        "restart")
            print_status "Restarting Ollama container..."
            docker restart "$CONTAINER_NAME" 2>/dev/null || print_error "Container not found"
            ;;
        "help"|*)
            show_usage
            ;;
    esac
}

# Run main function with all arguments
main "$@"