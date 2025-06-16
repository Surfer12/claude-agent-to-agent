from setuptools import setup, find_packages

setup(
    name="claude-agent-cli",
    version="0.1.0",
    description="Claude Agent-to-Agent CLI",
    author="Anthropic",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "anthropic",
    ],
    entry_points={
        "console_scripts": [
            "claude-agent=cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)