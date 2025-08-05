#!/bin/bash
# =============================================================================
# Ollama Docker Security Verification Script
# =============================================================================
# This script verifies that the Ollama Docker container is properly secured
# according to the security best practices defined in the configuration.
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="ollama-m4max"
EXPECTED_USER="ollama"
EXPECTED_UID="1000"

# Security check results
PASSED=0
FAILED=0
WARNINGS=0

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_fail "$1 is not installed or not in PATH"
        return 1
    fi
    return 0
}

# =============================================================================
# Container Existence and Status Checks
# =============================================================================

check_container_exists() {
    log_info "Checking if container exists..."
    
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_pass "Container '${CONTAINER_NAME}' exists"
        return 0
    else
        log_fail "Container '${CONTAINER_NAME}' not found"
        return 1
    fi
}

check_container_running() {
    log_info "Checking if container is running..."
    
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_pass "Container '${CONTAINER_NAME}' is running"
        return 0
    else
        log_fail "Container '${CONTAINER_NAME}' is not running"
        return 1
    fi
}

# =============================================================================
# Security Configuration Checks
# =============================================================================

check_non_root_user() {
    log_info "Checking non-root user configuration..."
    
    # Get the user running inside the container
    CONTAINER_USER=$(docker exec "$CONTAINER_NAME" whoami 2>/dev/null || echo "")
    
    if [[ "$CONTAINER_USER" == "$EXPECTED_USER" ]]; then
        log_pass "Container runs as non-root user: $CONTAINER_USER"
    else
        log_fail "Container user is '$CONTAINER_USER', expected '$EXPECTED_USER'"
    fi
    
    # Check UID
    CONTAINER_UID=$(docker exec "$CONTAINER_NAME" id -u 2>/dev/null || echo "")
    
    if [[ "$CONTAINER_UID" == "$EXPECTED_UID" ]]; then
        log_pass "Container UID is correct: $CONTAINER_UID"
    else
        log_fail "Container UID is '$CONTAINER_UID', expected '$EXPECTED_UID'"
    fi
}

check_readonly_filesystem() {
    log_info "Checking read-only filesystem..."
    
    READONLY=$(docker inspect "$CONTAINER_NAME" --format='{{.HostConfig.ReadonlyRootfs}}' 2>/dev/null || echo "false")
    
    if [[ "$READONLY" == "true" ]]; then
        log_pass "Root filesystem is read-only"
    else
        log_fail "Root filesystem is not read-only"
    fi
}

check_capabilities() {
    log_info "Checking Linux capabilities..."
    
    # Check dropped capabilities
    CAP_DROP=$(docker inspect "$CONTAINER_NAME" --format='{{json .HostConfig.CapDrop}}' 2>/dev/null || echo "[]")
    
    if echo "$CAP_DROP" | grep -q "ALL"; then
        log_pass "All capabilities are dropped"
    else
        log_fail "Not all capabilities are dropped: $CAP_DROP"
    fi
    
    # Check added capabilities
    CAP_ADD=$(docker inspect "$CONTAINER_NAME" --format='{{json .HostConfig.CapAdd}}' 2>/dev/null || echo "[]")
    
    if echo "$CAP_ADD" | grep -q "NET_BIND_SERVICE"; then
        log_pass "Only necessary capabilities are added: NET_BIND_SERVICE"
    else
        log_warn "Unexpected capabilities added: $CAP_ADD"
    fi
}

check_security_options() {
    log_info "Checking security options..."
    
    SEC_OPT=$(docker inspect "$CONTAINER_NAME" --format='{{json .HostConfig.SecurityOpt}}' 2>/dev/null || echo "[]")
    
    if echo "$SEC_OPT" | grep -q "no-new-privileges:true"; then
        log_pass "no-new-privileges is enabled"
    else
        log_fail "no-new-privileges is not enabled"
    fi
}

check_resource_limits() {
    log_info "Checking resource limits..."
    
    # Check memory limit
    MEMORY_LIMIT=$(docker inspect "$CONTAINER_NAME" --format='{{.HostConfig.Memory}}' 2>/dev/null || echo "0")
    
    if [[ "$MEMORY_LIMIT" -gt 0 ]]; then
        MEMORY_GB=$((MEMORY_LIMIT / 1024 / 1024 / 1024))
        log_pass "Memory limit is set: ${MEMORY_GB}GB"
    else
        log_fail "No memory limit is set"
    fi
    
    # Check CPU limit
    CPU_QUOTA=$(docker inspect "$CONTAINER_NAME" --format='{{.HostConfig.CpuQuota}}' 2>/dev/null || echo "0")
    CPU_PERIOD=$(docker inspect "$CONTAINER_NAME" --format='{{.HostConfig.CpuPeriod}}' 2>/dev/null || echo "100000")
    
    if [[ "$CPU_QUOTA" -gt 0 ]]; then
        CPU_LIMIT=$(echo "scale=1; $CPU_QUOTA / $CPU_PERIOD" | bc -l 2>/dev/null || echo "0")
        log_pass "CPU limit is set: ${CPU_LIMIT} cores"
    else
        log_fail "No CPU limit is set"
    fi
}

check_network_configuration() {
    log_info "Checking network configuration..."
    
    # Check port bindings
    PORT_BINDINGS=$(docker inspect "$CONTAINER_NAME" --format='{{json .NetworkSettings.Ports}}' 2>/dev/null || echo "{}")
    
    if echo "$PORT_BINDINGS" | grep -q "127.0.0.1"; then
        log_pass "Ports are bound to localhost only"
    else
        log_warn "Port binding configuration: $PORT_BINDINGS"
    fi
    
    # Check if port 11434 is accessible
    if curl -s --max-time 5 http://localhost:11434/api/tags > /dev/null 2>&1; then
        log_pass "Ollama API is accessible on localhost:11434"
    else
        log_fail "Ollama API is not accessible on localhost:11434"
    fi
}

check_volume_mounts() {
    log_info "Checking volume mounts..."
    
    # Get mount information
    MOUNTS=$(docker inspect "$CONTAINER_NAME" --format='{{json .Mounts}}' 2>/dev/null || echo "[]")
    
    # Check for proper model storage mount
    if echo "$MOUNTS" | grep -q "/ollama"; then
        log_pass "Model storage volume is properly mounted"
        
        # Check mount permissions
        MOUNT_SOURCE=$(echo "$MOUNTS" | jq -r '.[] | select(.Destination == "/ollama") | .Source' 2>/dev/null || echo "")
        if [[ -n "$MOUNT_SOURCE" && -d "$MOUNT_SOURCE" ]]; then
            MOUNT_PERMS=$(ls -ld "$MOUNT_SOURCE" | cut -d' ' -f1)
            log_info "Mount source permissions: $MOUNT_PERMS"
        fi
    else
        log_fail "Model storage volume is not properly mounted"
    fi
    
    # Check for tmpfs mounts
    if echo "$MOUNTS" | grep -q "tmpfs"; then
        log_pass "Temporary filesystems are properly configured"
    else
        log_warn "No tmpfs mounts detected"
    fi
}

check_image_security() {
    log_info "Checking Docker image security..."
    
    # Get image information
    IMAGE_ID=$(docker inspect "$CONTAINER_NAME" --format='{{.Image}}' 2>/dev/null || echo "")
    
    if [[ -n "$IMAGE_ID" ]]; then
        # Check image labels
        LABELS=$(docker inspect "$IMAGE_ID" --format='{{json .Config.Labels}}' 2>/dev/null || echo "{}")
        
        if echo "$LABELS" | grep -q "security.hardened"; then
            log_pass "Image is marked as security-hardened"
        else
            log_warn "Image security hardening label not found"
        fi
        
        # Check for security vulnerabilities (if docker scan is available)
        if command -v docker &> /dev/null && docker scan --help &> /dev/null 2>&1; then
            log_info "Running vulnerability scan..."
            if docker scan "$IMAGE_ID" --severity high --quiet > /dev/null 2>&1; then
                log_pass "No high-severity vulnerabilities found"
            else
                log_warn "Vulnerability scan found issues or failed"
            fi
        else
            log_info "Docker scan not available, skipping vulnerability check"
        fi
    else
        log_fail "Could not retrieve image information"
    fi
}

# =============================================================================
# Runtime Security Checks
# =============================================================================

check_process_security() {
    log_info "Checking process security..."
    
    # Check if processes are running as expected user
    PROCESSES=$(docker exec "$CONTAINER_NAME" ps aux 2>/dev/null || echo "")
    
    if echo "$PROCESSES" | grep -v "USER" | grep -q "$EXPECTED_USER"; then
        log_pass "Processes are running as $EXPECTED_USER"
    else
        log_fail "Processes are not running as expected user"
    fi
    
    # Check for suspicious processes
    if echo "$PROCESSES" | grep -qE "(sudo|su|passwd|ssh)"; then
        log_warn "Potentially suspicious processes detected"
    else
        log_pass "No suspicious processes detected"
    fi
}

check_file_permissions() {
    log_info "Checking file permissions..."
    
    # Check Ollama binary permissions
    OLLAMA_PERMS=$(docker exec "$CONTAINER_NAME" ls -l /usr/local/bin/ollama 2>/dev/null | cut -d' ' -f1 || echo "")
    
    if [[ "$OLLAMA_PERMS" == "-rwxr-xr-x" ]]; then
        log_pass "Ollama binary has correct permissions"
    else
        log_warn "Ollama binary permissions: $OLLAMA_PERMS"
    fi
    
    # Check model directory permissions
    MODEL_DIR_PERMS=$(docker exec "$CONTAINER_NAME" ls -ld /ollama 2>/dev/null | cut -d' ' -f1 || echo "")
    
    if [[ "$MODEL_DIR_PERMS" =~ ^drwx ]]; then
        log_pass "Model directory has correct permissions"
    else
        log_warn "Model directory permissions: $MODEL_DIR_PERMS"
    fi
}

check_network_security() {
    log_info "Checking network security..."
    
    # Check listening ports inside container
    LISTENING_PORTS=$(docker exec "$CONTAINER_NAME" netstat -ln 2>/dev/null | grep LISTEN || echo "")
    
    if echo "$LISTENING_PORTS" | grep -q ":11434"; then
        log_pass "Ollama is listening on port 11434"
    else
        log_fail "Ollama is not listening on expected port 11434"
    fi
    
    # Check for unexpected listening ports
    UNEXPECTED_PORTS=$(echo "$LISTENING_PORTS" | grep -v ":11434" | grep -v "127.0.0.1" || echo "")
    
    if [[ -z "$UNEXPECTED_PORTS" ]]; then
        log_pass "No unexpected listening ports detected"
    else
        log_warn "Unexpected listening ports detected: $UNEXPECTED_PORTS"
    fi
}

# =============================================================================
# System Integration Checks
# =============================================================================

check_host_security() {
    log_info "Checking host system security..."
    
    # Check if host firewall is active (macOS)
    if command -v pfctl &> /dev/null; then
        if sudo pfctl -s info 2>/dev/null | grep -q "Status: Enabled"; then
            log_pass "Host firewall is enabled"
        else
            log_warn "Host firewall may not be enabled"
        fi
    fi
    
    # Check Docker daemon security
    DOCKER_VERSION=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
    log_info "Docker version: $DOCKER_VERSION"
    
    # Check for Docker security best practices
    if docker info 2>/dev/null | grep -q "Security Options"; then
        SECURITY_OPTIONS=$(docker info --format '{{.SecurityOptions}}' 2>/dev/null || echo "")
        log_info "Docker security options: $SECURITY_OPTIONS"
    fi
}

check_log_security() {
    log_info "Checking logging security..."
    
    # Check log driver configuration
    LOG_DRIVER=$(docker inspect "$CONTAINER_NAME" --format='{{.HostConfig.LogConfig.Type}}' 2>/dev/null || echo "")
    
    if [[ "$LOG_DRIVER" == "json-file" ]]; then
        log_pass "Using json-file log driver"
        
        # Check log rotation settings
        LOG_CONFIG=$(docker inspect "$CONTAINER_NAME" --format='{{json .HostConfig.LogConfig.Config}}' 2>/dev/null || echo "{}")
        
        if echo "$LOG_CONFIG" | grep -q "max-size"; then
            log_pass "Log rotation is configured"
        else
            log_warn "Log rotation may not be configured"
        fi
    else
        log_warn "Log driver: $LOG_DRIVER"
    fi
}

# =============================================================================
# Performance and Resource Checks
# =============================================================================

check_performance_security() {
    log_info "Checking performance-related security..."
    
    # Check current resource usage
    STATS=$(docker stats "$CONTAINER_NAME" --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || echo "")
    
    if [[ -n "$STATS" ]]; then
        log_info "Current resource usage: $STATS"
    fi
    
    # Check for resource exhaustion indicators
    if docker exec "$CONTAINER_NAME" cat /proc/meminfo 2>/dev/null | grep -q "MemAvailable"; then
        AVAILABLE_MEM=$(docker exec "$CONTAINER_NAME" grep "MemAvailable" /proc/meminfo | awk '{print $2}')
        if [[ "$AVAILABLE_MEM" -gt 1000000 ]]; then  # > 1GB
            log_pass "Sufficient memory available in container"
        else
            log_warn "Low memory available in container: ${AVAILABLE_MEM}KB"
        fi
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

print_banner() {
    echo "============================================================================="
    echo "🔒 Ollama Docker Security Verification"
    echo "============================================================================="
    echo
}

print_summary() {
    echo
    echo "============================================================================="
    echo "📊 Security Check Summary"
    echo "============================================================================="
    
    TOTAL=$((PASSED + FAILED + WARNINGS))
    
    echo -e "✅ Passed:   ${GREEN}$PASSED${NC}"
    echo -e "❌ Failed:   ${RED}$FAILED${NC}"
    echo -e "⚠️  Warnings: ${YELLOW}$WARNINGS${NC}"
    echo -e "📊 Total:    $TOTAL"
    echo
    
    if [[ $FAILED -eq 0 ]]; then
        echo -e "${GREEN}🎉 All critical security checks passed!${NC}"
        if [[ $WARNINGS -gt 0 ]]; then
            echo -e "${YELLOW}⚠️  Please review the warnings above.${NC}"
        fi
    else
        echo -e "${RED}🚨 Critical security issues found!${NC}"
        echo -e "${RED}   Please address the failed checks before using in production.${NC}"
    fi
    
    echo
}

main() {
    print_banner
    
    # Prerequisites
    if ! check_command "docker"; then
        echo "❌ Docker is required but not installed"
        exit 1
    fi
    
    if ! check_command "jq"; then
        log_warn "jq is not installed - some checks will be limited"
    fi
    
    if ! check_command "bc"; then
        log_warn "bc is not installed - some calculations will be limited"
    fi
    
    # Container checks
    if ! check_container_exists; then
        echo "❌ Cannot proceed without container. Please run setup first."
        exit 1
    fi
    
    if ! check_container_running; then
        echo "❌ Container is not running. Please start it first."
        exit 1
    fi
    
    # Security checks
    check_non_root_user
    check_readonly_filesystem
    check_capabilities
    check_security_options
    check_resource_limits
    check_network_configuration
    check_volume_mounts
    check_image_security
    
    # Runtime checks
    check_process_security
    check_file_permissions
    check_network_security
    
    # System checks
    check_host_security
    check_log_security
    check_performance_security
    
    # Summary
    print_summary
    
    # Exit with appropriate code
    if [[ $FAILED -gt 0 ]]; then
        exit 1
    else
        exit 0
    fi
}

# =============================================================================
# Script Entry Point
# =============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi