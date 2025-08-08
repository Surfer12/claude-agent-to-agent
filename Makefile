# Makefile for Claude Agent-to-Agent Docker Operations

.PHONY: help build run stop clean test logs check setup dev

# Default target
help:
	@echo "Claude Agent-to-Agent Docker Operations"
	@echo ""
	@echo "Available commands:"
	@echo "  make build     - Build Docker image"
	@echo "  make run       - Run container interactively"
	@echo "  make dev       - Run in development mode"
	@echo "  make test      - Run tests"
	@echo "  make stop      - Stop all containers"
	@echo "  make clean     - Clean up Docker resources"
	@echo "  make logs      - Show container logs"
	@echo "  make check     - Check environment setup"
	@echo "  make setup     - Initial setup"
	@echo "  make help      - Show this help"

# Build the Docker image
build:
	@echo "Building Docker image..."
	docker build -t claude-agent-to-agent .
	@echo "Build complete!"

# Run container interactively
run:
	@echo "Running container interactively..."
	docker run -it --env-file .env claude-agent-to-agent

# Run in development mode
dev:
	@echo "Running in development mode..."
	docker run -it --env-file .env -v $(PWD):/app claude-agent-to-agent python cli.py --interactive --verbose

# Run with docker-compose
compose:
	@echo "Starting with docker-compose..."
	docker-compose up --build

# Run tests
test:
	@echo "Running tests..."
	docker-compose up claude-agent-test

# Stop all containers
stop:
	@echo "Stopping containers..."
	docker-compose down

# Clean up Docker resources
clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down --volumes --remove-orphans
	docker system prune -f
	@echo "Cleanup complete!"

# Show logs
logs:
	@echo "Showing logs..."
	docker-compose logs -f claude-agent

# Check environment setup
check:
	@echo "Checking environment..."
	@if [ ! -f .env ]; then \
		echo "Creating .env from template..."; \
		cp env.example .env; \
		echo "Please edit .env and add your ANTHROPIC_API_KEY"; \
	else \
		echo ".env file found"; \
	fi
	@if ! docker info > /dev/null 2>&1; then \
		echo "Docker is not running. Please start Docker."; \
		exit 1; \
	else \
		echo "Docker is running"; \
	fi
	@echo "Environment check complete!"

# Initial setup
setup: check build
	@echo "Setup complete! Run 'make run' to start the container."

# Quick start (setup + run)
quick: setup run

# Development workflow
dev-setup: check
	@echo "Setting up development environment..."
	docker-compose up --build claude-agent-dev

# Production deployment
prod:
	@echo "Building production image..."
	docker build -t claude-agent-to-agent:latest .
	@echo "Production image ready!"

# Show container status
status:
	@echo "Container status:"
	docker-compose ps

# Access container shell
shell:
	@echo "Accessing container shell..."
	docker-compose exec claude-agent bash

# View container resources
resources:
	@echo "Container resource usage:"
	docker stats --no-stream

# Backup data
backup:
	@echo "Creating backup..."
	mkdir -p backups
	tar -czf backups/$(shell date +%Y%m%d_%H%M%S)_backup.tar.gz data/ logs/ .env
	@echo "Backup created!"

# Restore from backup
restore:
	@echo "Available backups:"
	@ls -la backups/ 2>/dev/null || echo "No backups found"
	@echo "To restore: tar -xzf backups/FILENAME.tar.gz"

# Update dependencies
update:
	@echo "Updating dependencies..."
	docker-compose build --no-cache
	@echo "Dependencies updated!"

# Security scan
security:
	@echo "Running security scan..."
	docker run --rm -v $(PWD):/app claude-agent-to-agent python -m bandit -r /app -f json -o security-report.json || echo "Security scan completed"

# Performance test
perf:
	@echo "Running performance test..."
	docker run --rm --env-file .env claude-agent-to-agent python -c "import time; start=time.time(); print('Performance test completed in', time.time()-start, 'seconds')"
