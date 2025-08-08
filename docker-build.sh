#!/bin/bash

# Docker Build Script for Claude Agent-to-Agent
# This script provides easy commands for building and running the Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if .env file exists
check_env() {
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from template..."
        if [ -f env.example ]; then
            cp env.example .env
            print_success "Created .env file from template"
            print_warning "Please edit .env and add your ANTHROPIC_API_KEY"
        else
            print_error "env.example not found. Please create .env file manually."
            exit 1
        fi
    else
        print_success ".env file found"
    fi
}

# Function to build the Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -t claude-agent-to-agent .
    print_success "Docker image built successfully"
}

# Function to run the container
run_container() {
    local mode=${1:-interactive}
    
    case $mode in
        "interactive")
            print_status "Running container in interactive mode..."
            docker run -it --env-file .env claude-agent-to-agent
            ;;
        "detached")
            print_status "Running container in detached mode..."
            docker run -d --env-file .env --name claude-agent claude-agent-to-agent
            print_success "Container started in detached mode"
            ;;
        "dev")
            print_status "Running container in development mode..."
            docker run -it --env-file .env -v $(pwd):/app claude-agent-to-agent python cli.py --interactive --verbose
            ;;
        *)
            print_error "Unknown mode: $mode"
            exit 1
            ;;
    esac
}

# Function to run with docker-compose
run_compose() {
    local service=${1:-claude-agent}
    
    print_status "Starting services with docker-compose..."
    docker-compose up --build $service
}

# Function to stop containers
stop_containers() {
    print_status "Stopping containers..."
    docker-compose down
    print_success "Containers stopped"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to show logs
show_logs() {
    local service=${1:-claude-agent}
    print_status "Showing logs for $service..."
    docker-compose logs -f $service
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    docker-compose up claude-agent-test
}

# Function to show help
show_help() {
    echo "Docker Build Script for Claude Agent-to-Agent"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build              Build the Docker image"
    echo "  run [MODE]         Run the container (interactive|detached|dev)"
    echo "  compose [SERVICE]  Run with docker-compose (claude-agent|claude-agent-dev|claude-agent-test)"
    echo "  stop               Stop all containers"
    echo "  cleanup            Clean up Docker resources"
    echo "  logs [SERVICE]     Show logs for a service"
    echo "  test               Run tests"
    echo "  check              Check Docker and environment setup"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run interactive"
    echo "  $0 compose claude-agent-dev"
    echo "  $0 test"
}

# Main script logic
case "${1:-help}" in
    "build")
        check_docker
        build_image
        ;;
    "run")
        check_docker
        check_env
        run_container "$2"
        ;;
    "compose")
        check_docker
        check_env
        run_compose "$2"
        ;;
    "stop")
        stop_containers
        ;;
    "cleanup")
        cleanup
        ;;
    "logs")
        show_logs "$2"
        ;;
    "test")
        check_docker
        check_env
        run_tests
        ;;
    "check")
        check_docker
        check_env
        print_success "Environment check passed"
        ;;
    "help"|*)
        show_help
        ;;
esac
