#!/usr/bin/env python3
"""
Secure setup script for Claude Agent Framework.
This script implements security best practices and fixes command injection vulnerabilities.
"""

import os
import sys
import subprocess
import shlex
import stat
import getpass
from pathlib import Path
from typing import List, Optional
import keyring
import yaml


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class SecureSetup:
    """Secure setup manager for Claude Agent Framework."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = Path.home() / ".claude-agent"
        self.log_dir = Path("/var/log/claude-agent")
        
    def setup_environment(self):
        """Set up the secure environment."""
        print("🔒 Setting up Claude Agent Framework with security hardening...")
        
        # Create necessary directories
        self._create_secure_directories()
        
        # Set up API key storage
        self._setup_api_key_storage()
        
        # Create secure configuration
        self._create_secure_config()
        
        # Set file permissions
        self._set_secure_permissions()
        
        # Install dependencies
        self._install_secure_dependencies()
        
        # Set up logging
        self._setup_security_logging()
        
        print("✅ Secure setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Review the generated configuration in ~/.claude-agent/config.yaml")
        print("2. Test the setup with: python cli.py --prompt 'Hello, Claude!'")
        print("3. Enable additional tools only after reviewing security implications")
        
    def _create_secure_directories(self):
        """Create directories with secure permissions."""
        directories = [
            self.config_dir,
            self.config_dir / "logs",
            Path("/tmp/claude-agent"),
        ]
        
        for directory in directories:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            print(f"📁 Created secure directory: {directory}")
    
    def _setup_api_key_storage(self):
        """Set up secure API key storage."""
        print("\n🔑 Setting up API key storage...")
        
        # Check if API key is already stored
        try:
            existing_key = keyring.get_password("claude-agent", "api_key")
            if existing_key:
                update = input("API key already exists. Update it? (y/N): ").lower()
                if update != 'y':
                    print("Using existing API key.")
                    return
        except Exception:
            pass
        
        # Get API key securely
        while True:
            api_key = getpass.getpass("Enter your Anthropic API key: ")
            if api_key.startswith("sk-ant-"):
                break
            print("❌ Invalid API key format. Please enter a valid Anthropic API key.")
        
        # Store in keyring
        try:
            keyring.set_password("claude-agent", "api_key", api_key)
            print("✅ API key stored securely in system keyring.")
        except Exception as e:
            print(f"❌ Failed to store API key in keyring: {e}")
            
            # Fallback to encrypted file
            self._store_api_key_encrypted(api_key)
    
    def _store_api_key_encrypted(self, api_key: str):
        """Store API key in encrypted file as fallback."""
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        import base64
        import platform
        
        # Generate encryption key from machine-specific data
        salt = platform.node().encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(b"claude-agent-framework"))
        fernet = Fernet(key)
        
        # Encrypt and store
        encrypted_key = fernet.encrypt(api_key.encode())
        key_file = self.config_dir / "api_key.enc"
        key_file.write_bytes(encrypted_key)
        key_file.chmod(0o600)
        
        print("✅ API key stored in encrypted file as fallback.")
    
    def _create_secure_config(self):
        """Create secure configuration file."""
        config_file = self.config_dir / "config.yaml"
        
        if config_file.exists():
            backup = input("Configuration exists. Create backup? (Y/n): ").lower()
            if backup != 'n':
                backup_file = config_file.with_suffix('.yaml.bak')
                config_file.rename(backup_file)
                print(f"📋 Backup created: {backup_file}")
        
        # Load secure template
        template_file = self.project_root / "secure_config_template.yaml"
        if template_file.exists():
            config = yaml.safe_load(template_file.read_text())
        else:
            config = self._get_default_secure_config()
        
        # Customize for environment
        config['security']['logging']['log_file'] = str(self.config_dir / "logs" / "security.log")
        
        # Write configuration
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        config_file.chmod(0o600)
        print(f"✅ Secure configuration created: {config_file}")
    
    def _get_default_secure_config(self) -> dict:
        """Get default secure configuration."""
        return {
            'agent': {
                'name': 'claude-agent-secure',
                'system_prompt': 'You are Claude, an AI assistant. Be helpful while following security protocols.',
                'verbose': False
            },
            'model': {
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 4096,
                'temperature': 1.0
            },
            'security': {
                'api_key': {
                    'storage_method': 'keyring'
                },
                'file_operations': {
                    'allowed_directories': ['/tmp/claude-agent'],
                    'max_file_size_mb': 1,
                    'enable_path_validation': True
                },
                'logging': {
                    'enable_security_logging': True,
                    'log_level': 'INFO'
                }
            },
            'tools': {
                'enabled': ['think', 'file_read', 'file_write']
            }
        }
    
    def _set_secure_permissions(self):
        """Set secure file permissions."""
        # Set permissions on configuration directory
        self.config_dir.chmod(0o700)
        
        # Set permissions on all files in config directory
        for file_path in self.config_dir.rglob('*'):
            if file_path.is_file():
                file_path.chmod(0o600)
        
        print("✅ Secure file permissions set.")
    
    def _install_secure_dependencies(self):
        """Install dependencies with security considerations."""
        print("\n📦 Installing secure dependencies...")
        
        # Security-focused requirements
        security_packages = [
            "cryptography>=41.0.0",  # For encryption
            "keyring>=24.0.0",       # For secure credential storage
            "pyyaml>=6.0.0",         # For configuration
            "requests>=2.31.0",      # For HTTP with security patches
        ]
        
        for package in security_packages:
            self._safe_pip_install(package)
    
    def _safe_pip_install(self, package: str):
        """Safely install a pip package without shell injection."""
        try:
            # Use subprocess with explicit arguments (no shell=True)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=300,
                check=False
            )
            
            if result.returncode == 0:
                print(f"✅ Installed: {package}")
            else:
                print(f"⚠️  Failed to install {package}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout installing {package}")
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")
    
    def _setup_security_logging(self):
        """Set up security logging."""
        log_dir = self.config_dir / "logs"
        log_dir.mkdir(mode=0o700, exist_ok=True)
        
        # Create log configuration
        log_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'security': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'security_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': str(log_dir / 'security.log'),
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'formatter': 'security'
                }
            },
            'loggers': {
                'security': {
                    'handlers': ['security_file'],
                    'level': 'INFO',
                    'propagate': False
                }
            }
        }
        
        log_config_file = self.config_dir / "logging.yaml"
        with open(log_config_file, 'w') as f:
            yaml.dump(log_config, f)
        
        log_config_file.chmod(0o600)
        print("✅ Security logging configured.")
    
    def validate_environment(self):
        """Validate the security of the environment."""
        print("\n🔍 Validating security configuration...")
        
        issues = []
        
        # Check API key storage
        try:
            api_key = keyring.get_password("claude-agent", "api_key")
            if not api_key:
                issues.append("API key not found in keyring")
        except Exception:
            issues.append("Keyring not accessible")
        
        # Check file permissions
        config_file = self.config_dir / "config.yaml"
        if config_file.exists():
            permissions = oct(config_file.stat().st_mode)[-3:]
            if permissions != '600':
                issues.append(f"Config file permissions too open: {permissions}")
        
        # Check for .env files with weak permissions
        for env_file in Path('.').glob('**/.env'):
            permissions = oct(env_file.stat().st_mode)[-3:]
            if permissions not in ['600', '400']:
                issues.append(f"Environment file has weak permissions: {env_file} ({permissions})")
        
        if issues:
            print("❌ Security issues found:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        else:
            print("✅ Security validation passed.")
            return True
    
    def fix_command_injection_vulnerabilities(self):
        """Fix command injection vulnerabilities in the codebase."""
        print("\n🔧 Fixing command injection vulnerabilities...")
        
        # Files that need to be fixed
        vulnerable_files = [
            "computer-use-demo/computer_use_demo/streamlit.py",
            "migrate_to_new_structure.py"
        ]
        
        for file_path in vulnerable_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self._fix_file_command_injection(full_path)
    
    def _fix_file_command_injection(self, file_path: Path):
        """Fix command injection in a specific file."""
        print(f"🔧 Fixing: {file_path}")
        
        # Read the file
        content = file_path.read_text()
        
        # Common dangerous patterns and their fixes
        fixes = [
            # subprocess.run with shell=True
            (
                r'subprocess\.run\("([^"]+)",\s*shell=True\)',
                lambda m: f'subprocess.run({self._split_command_safely(m.group(1))})'
            ),
            # subprocess.run with shell=True and other args
            (
                r'subprocess\.run\("([^"]+)",\s*shell=True,([^)]+)\)',
                lambda m: f'subprocess.run({self._split_command_safely(m.group(1))},{m.group(2)})'
            ),
        ]
        
        import re
        modified = False
        
        for pattern, replacement in fixes:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
        
        if modified:
            # Create backup
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            file_path.rename(backup_path)
            
            # Write fixed content
            file_path.write_text(content)
            print(f"✅ Fixed command injection in {file_path}")
            print(f"📋 Backup created: {backup_path}")
        else:
            print(f"ℹ️  No command injection patterns found in {file_path}")
    
    def _split_command_safely(self, command: str) -> str:
        """Safely split a shell command into a list."""
        try:
            parts = shlex.split(command)
            return str(parts)
        except ValueError:
            # If shlex fails, split on spaces as fallback
            parts = command.split()
            return str(parts)


def main():
    """Main setup function."""
    if os.geteuid() == 0:
        print("❌ Do not run this script as root for security reasons.")
        sys.exit(1)
    
    setup = SecureSetup()
    
    try:
        # Perform setup
        setup.setup_environment()
        
        # Validate environment
        if not setup.validate_environment():
            print("\n⚠️  Some security issues were found. Please review and fix them.")
        
        # Fix command injection vulnerabilities
        setup.fix_command_injection_vulnerabilities()
        
        print("\n🎉 Claude Agent Framework is now securely configured!")
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 