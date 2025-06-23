# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 5.1.x   | :white_check_mark: |
| 5.0.x   | :x:                |
| 4.0.x   | :white_check_mark: |
| < 4.0   | :x:                |

## Threat Model

### Attack Vectors
1. **Code Injection**: Malicious prompts leading to arbitrary code execution
2. **Command Injection**: Shell command manipulation through user input
3. **Path Traversal**: Unauthorized file system access
4. **Credential Theft**: API key and authentication token exposure
5. **Privilege Escalation**: Exploiting system tools for elevated access
6. **Network Attacks**: MitM, credential interception, data exfiltration
7. **Resource Exhaustion**: DoS through resource consumption

### Security Boundaries
- **Code Execution**: Server-side sandboxed environment
- **Computer Use**: Isolated VM/container environment
- **File Operations**: Restricted to designated directories
- **Network Access**: Controlled external connections
- **API Access**: Authenticated and rate-limited endpoints

## Critical Security Requirements

### 🚨 MANDATORY: Isolated Execution Environment
**Computer Use and Code Execution tools MUST run in isolated environments:**

```bash
# Docker isolation (recommended)
docker run --rm -it \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --read-only \
  --tmpfs /tmp \
  --memory 1g \
  --cpus 1.0 \
  --network none \
  your-claude-agent-image

# VM isolation (alternative)
# Run in dedicated VM with network restrictions
# Snapshot and restore VM state between sessions
```

### 🔐 API Key Security

#### Environment File Protection
```bash
# Create .env file with restricted permissions
touch .env
chmod 600 .env
chown $(whoami):$(whoami) .env

# Verify permissions
ls -la .env
# Should show: -rw------- 1 user user ... .env
```

#### Secure Storage Implementation
```python
# Use OS keyring instead of environment variables
import keyring
import getpass

def store_api_key():
    api_key = getpass.getpass("Enter API key: ")
    keyring.set_password("claude-agent", "api_key", api_key)

def get_api_key():
    return keyring.get_password("claude-agent", "api_key")
```

### 🌐 Network Security

#### MCP Server Security
```python
# Secure MCP connection configuration
MCP_CONFIG = {
    "servers": {
        "secure-server": {
            "url": "https://api.example.com/mcp",  # HTTPS only
            "headers": {
                "Authorization": "Bearer {token}",
                "X-API-Key": "{api_key}"
            },
            "verify_ssl": True,
            "timeout": 30
        }
    }
}

# TLS configuration
TLS_CONFIG = {
    "min_version": "TLSv1.2",
    "ciphers": "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS",
    "verify_mode": "CERT_REQUIRED"
}
```

### 🛡️ Input Validation & Sanitization

#### Path Traversal Prevention
```python
import os
from pathlib import Path

def validate_file_path(path: str, allowed_dirs: list[str]) -> bool:
    """Validate file path against directory traversal attacks."""
    try:
        # Resolve path and check if it's within allowed directories
        resolved_path = Path(path).resolve()
        
        for allowed_dir in allowed_dirs:
            allowed_path = Path(allowed_dir).resolve()
            if resolved_path.is_relative_to(allowed_path):
                return True
        
        return False
    except (OSError, ValueError):
        return False

# Usage
ALLOWED_DIRS = ["/app/data", "/tmp/uploads"]
if not validate_file_path(user_path, ALLOWED_DIRS):
    raise SecurityError("Path access denied")
```

#### Command Injection Prevention
```python
import shlex
import subprocess

def safe_command_execution(command: list[str]) -> str:
    """Execute command safely without shell injection."""
    # Validate command whitelist
    allowed_commands = {"ls", "cat", "grep", "find"}
    if command[0] not in allowed_commands:
        raise SecurityError(f"Command not allowed: {command[0]}")
    
    # Execute without shell
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=False
    )
    
    return result.stdout
```

## Security Guidelines

### API Key Management
1. **Storage**:
   - ✅ Use OS keyring or encrypted storage
   - ✅ Set file permissions to 600 for .env files
   - ✅ Regular key rotation policy
   - ❌ Never commit keys to version control
   - ❌ Avoid environment variables in production

2. **Access Control**:
   - Implement role-based access control (RBAC)
   - Audit logging for API key usage
   - Rate limiting on API endpoints
   - Session timeout enforcement

### Computer Use Tool Security
1. **Isolation Requirements**:
   - ✅ MANDATORY: Run in isolated VM or container
   - ✅ Disable network access for computer use sessions
   - ✅ Use read-only file systems where possible
   - ✅ Implement session recording for audit trails

2. **Access Controls**:
   - Whitelist allowed applications
   - Restrict system administration functions
   - Monitor for suspicious activities
   - Implement emergency stop mechanisms

### Code Execution Security
1. **Sandbox Configuration**:
   ```python
   SANDBOX_CONFIG = {
       "memory_limit": "1GB",
       "cpu_limit": "1.0",
       "timeout": 300,
       "network_access": False,
       "file_system": "read-only",
       "allowed_modules": ["math", "json", "datetime"]
   }
   ```

2. **Resource Limits**:
   - Memory: 1GB maximum
   - CPU: 1 core maximum
   - Execution time: 5 minutes maximum
   - File operations: Restricted to /tmp

### File Operations Security
1. **Path Validation**:
   - Prevent directory traversal (../)
   - Validate file extensions
   - Check file permissions
   - Enforce file size limits

2. **Access Controls**:
   - Restrict to designated directories
   - Implement file operation logging
   - Regular permission audits
   - Secure temporary file handling

### Network Security
1. **TLS Requirements**:
   - TLS 1.2+ for all connections
   - Certificate validation enabled
   - Strong cipher suites only
   - HSTS headers for web interfaces

2. **API Security**:
   - Request signing for sensitive operations
   - Rate limiting (100 requests/minute)
   - Input validation on all endpoints
   - CORS policies for web interfaces

### Authentication & Authorization
1. **Authentication**:
   - Multi-factor authentication support
   - Strong password requirements
   - Session management with secure cookies
   - JWT token validation

2. **Authorization**:
   - Granular permission system
   - Tool-specific access controls
   - Resource-based permissions
   - Regular access reviews

## Security Monitoring

### Audit Logging
```python
import logging
import json
from datetime import datetime

# Security event logging
security_logger = logging.getLogger('security')
security_logger.setLevel(logging.INFO)

def log_security_event(event_type: str, details: dict):
    """Log security-relevant events."""
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "details": details,
        "severity": "INFO"
    }
    security_logger.info(json.dumps(event))

# Usage examples
log_security_event("api_key_access", {"user": "admin", "action": "retrieve"})
log_security_event("file_access", {"path": "/secure/data", "operation": "read"})
log_security_event("command_execution", {"command": "ls", "result": "success"})
```

### Security Metrics
- Failed authentication attempts
- Unusual API usage patterns
- File access violations
- Command execution anomalies
- Network connection attempts

## Deployment Security

### Production Checklist
- [ ] API keys stored securely (keyring/vault)
- [ ] File permissions set correctly (600 for secrets)
- [ ] TLS certificates valid and up-to-date
- [ ] Firewall rules configured
- [ ] Security monitoring enabled
- [ ] Backup and recovery procedures tested
- [ ] Incident response plan documented
- [ ] Regular security assessments scheduled

### Container Security
```dockerfile
# Secure Dockerfile example
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r claude && useradd -r -g claude claude

# Install security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    security-updates && \
    rm -rf /var/lib/apt/lists/*

# Set secure permissions
COPY --chown=claude:claude . /app
WORKDIR /app

# Drop privileges
USER claude

# Security options
LABEL security.no-new-privileges=true
LABEL security.read-only-root-fs=true
```

### Kubernetes Security
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: claude-agent
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  containers:
  - name: claude-agent
    image: claude-agent:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    resources:
      limits:
        memory: "1Gi"
        cpu: "1"
      requests:
        memory: "512Mi"
        cpu: "0.5"
```

## Incident Response

### Security Incident Types
1. **Credential Compromise**: API key theft or unauthorized access
2. **Code Injection**: Malicious code execution through prompts
3. **Data Breach**: Unauthorized access to sensitive files
4. **System Compromise**: Privilege escalation or system takeover

### Response Procedures
1. **Immediate Actions**:
   - Isolate affected systems
   - Revoke compromised credentials
   - Preserve evidence
   - Notify stakeholders

2. **Investigation**:
   - Analyze security logs
   - Determine attack vector
   - Assess impact scope
   - Document findings

3. **Recovery**:
   - Patch vulnerabilities
   - Restore from clean backups
   - Update security controls
   - Monitor for reoccurrence

## Reporting a Vulnerability

1. **Process**:
   - Create a security advisory on GitHub
   - Email security@yourdomain.com
   - Include detailed reproduction steps
   - Provide impact assessment

2. **Response Timeline**:
   - Initial response: 24 hours
   - Assessment: 72 hours
   - Fix timeline: Based on severity
   
3. **Disclosure**:
   - Coordinated disclosure after patch
   - Credit to reporters
   - Public security advisories

## Security Checklist

### For Developers
- [ ] Use secure credential storage
- [ ] Implement comprehensive input validation
- [ ] Enable security audit logging
- [ ] Set up resource limits
- [ ] Configure proper access controls
- [ ] Test for security vulnerabilities
- [ ] Review dependencies regularly
- [ ] Follow secure coding practices

### For Administrators
- [ ] Configure authentication systems
- [ ] Set up security monitoring
- [ ] Enable automated backups
- [ ] Plan incident response procedures
- [ ] Document security procedures
- [ ] Train users on security practices
- [ ] Conduct regular security assessments
- [ ] Maintain security compliance

## Contact

For security concerns:
- Create a GitHub security advisory
- Email: security@yourdomain.com
- Emergency: security-urgent@yourdomain.com

---

**Remember: Security is a shared responsibility. Every user and developer must follow these guidelines to maintain a secure environment.**
