#!/bin/bash

# vLLM Setup Script
# This script demonstrates how to install and set up vLLM

set -e  # Exit on error

echo "🚀 vLLM Setup Script"
echo "===================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install vLLM via pip
install_vllm_pip() {
    print_status "Installing vLLM via pip..."
    
    # Check if Python is available
    if ! command_exists python3; then
        print_error "Python3 is not installed. Please install Python3 first."
        exit 1
    fi
    
    # Create virtual environment
    print_status "Creating virtual environment..."
    python3 -m venv vllm_env
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source vllm_env/bin/activate
    
    # Upgrade pip and install build tools
    print_status "Upgrading pip and installing build tools..."
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch first (helps with dependencies)
    print_status "Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install vLLM
    print_status "Installing vLLM..."
    pip install vllm || {
        print_warning "Standard installation failed. Trying with additional flags..."
        pip install vllm --no-build-isolation || {
            print_error "vLLM installation failed. Please check the troubleshooting section."
            return 1
        }
    }
    
    print_success "vLLM installed successfully via pip!"
    return 0
}

# Function to install Docker and set up vLLM container
install_vllm_docker() {
    print_status "Setting up vLLM with Docker..."
    
    # Check if Docker is installed
    if ! command_exists docker; then
        print_status "Installing Docker..."
        sudo apt update
        sudo apt install -y docker.io
        
        # Add user to docker group
        sudo usermod -aG docker $USER
        print_warning "Please log out and log back in for Docker group changes to take effect."
    fi
    
    # Check if Docker is running
    if ! sudo docker info >/dev/null 2>&1; then
        print_status "Starting Docker service..."
        sudo systemctl start docker || {
            print_warning "systemctl not available, trying alternative methods..."
            sudo service docker start || {
                print_error "Could not start Docker service. Please start it manually."
                return 1
            }
        }
    fi
    
    # Pull vLLM Docker image
    print_status "Pulling vLLM Docker image..."
    sudo docker pull vllm/vllm-openai:latest
    
    print_success "Docker setup completed!"
    return 0
}

# Function to run vLLM server with pip installation
run_vllm_pip() {
    print_status "Starting vLLM server (pip installation)..."
    
    if [ ! -d "vllm_env" ]; then
        print_error "Virtual environment not found. Please run installation first."
        return 1
    fi
    
    source vllm_env/bin/activate
    
    # Use a smaller model for demonstration
    MODEL_NAME="${1:-microsoft/DialoGPT-medium}"
    
    print_status "Starting server with model: $MODEL_NAME"
    print_status "Server will be available at http://localhost:8000"
    print_warning "Press Ctrl+C to stop the server"
    
    vllm serve "$MODEL_NAME" --host 0.0.0.0 --port 8000
}

# Function to run vLLM server with Docker
run_vllm_docker() {
    print_status "Starting vLLM server (Docker)..."
    
    MODEL_NAME="${1:-microsoft/DialoGPT-medium}"
    
    print_status "Starting Docker container with model: $MODEL_NAME"
    print_status "Server will be available at http://localhost:8000"
    print_warning "Press Ctrl+C to stop the server"
    
    sudo docker run --rm \
        --name vllm_server \
        -p 8000:8000 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        vllm/vllm-openai:latest \
        --model "$MODEL_NAME" \
        --host 0.0.0.0 \
        --port 8000
}

# Function to test the vLLM server
test_vllm_server() {
    print_status "Testing vLLM server..."
    
    # Wait for server to start
    sleep 5
    
    # Test with curl
    response=$(curl -s -X POST "http://localhost:8000/v1/chat/completions" \
        -H "Content-Type: application/json" \
        --data '{
            "model": "microsoft/DialoGPT-medium",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?"
                }
            ],
            "max_tokens": 50
        }' || echo "ERROR")
    
    if [[ "$response" == "ERROR" ]]; then
        print_error "Failed to connect to vLLM server"
        return 1
    else
        print_success "Server is responding!"
        echo "Response: $response"
        return 0
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  install-pip     Install vLLM using pip"
    echo "  install-docker  Install vLLM using Docker"
    echo "  run-pip         Run vLLM server (pip installation)"
    echo "  run-docker      Run vLLM server (Docker)"
    echo "  test            Test the running vLLM server"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install-pip"
    echo "  $0 run-pip microsoft/DialoGPT-medium"
    echo "  $0 run-docker meta-llama/Llama-2-7b-chat-hf"
    echo ""
}

# Main script logic
case "${1:-help}" in
    "install-pip")
        install_vllm_pip
        ;;
    "install-docker")
        install_vllm_docker
        ;;
    "run-pip")
        run_vllm_pip "$2"
        ;;
    "run-docker")
        run_vllm_docker "$2"
        ;;
    "test")
        test_vllm_server
        ;;
    "help"|*)
        show_usage
        ;;
esac

print_status "Script completed!"