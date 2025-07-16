#!/usr/bin/env python3
"""
Setup script for the Unified Agent System.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="unified-agent-system",
    version="1.0.0",
    author="Unified Agent System Team",
    author_email="contact@unified-agent-system.com",
    description="A provider-agnostic agent framework supporting Claude and OpenAI with unified CLI and computer use capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/unified-agent-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "computer-use": [
            "playwright>=1.40.0",
            "selenium>=4.15.0",
        ],
        "web": [
            "streamlit>=1.28.0",
            "gradio>=4.0.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "unified-agent=unified_agent.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "unified_agent": ["py.typed"],
    },
    keywords="ai agent claude openai computer-use automation cli",
    project_urls={
        "Bug Reports": "https://github.com/your-username/unified-agent-system/issues",
        "Source": "https://github.com/your-username/unified-agent-system",
        "Documentation": "https://github.com/your-username/unified-agent-system#readme",
    },
)