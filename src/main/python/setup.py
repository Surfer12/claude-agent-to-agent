"""
Setup script for Anthropic API Client Library

This package provides a comprehensive interface to Anthropic's Claude API,
including support for various tools, streaming, and advanced features.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), '..', '..', 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Anthropic API Client Library for Python"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), '..', '..', 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'anthropic>=0.25.0',
        'frozendict>=2.4.0',
        'typing-extensions>=4.0.0'
    ]

setup(
    name="anthropic-api-client",
    version="1.0.0",
    author="Anthropic API Team",
    author_email="api@anthropic.com",
    description="A comprehensive Python client for Anthropic's Claude API",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/anthropics/anthropic-api-client",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "flake8>=5.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "anthropic-cli=com.anthropic.api.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="anthropic claude ai api client tools streaming",
    project_urls={
        "Bug Reports": "https://github.com/anthropics/anthropic-api-client/issues",
        "Source": "https://github.com/anthropics/anthropic-api-client",
        "Documentation": "https://anthropic-api-client.readthedocs.io/",
    },
) 