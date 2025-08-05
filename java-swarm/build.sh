#!/bin/bash

# Java Swarm Build Script

set -e

echo "Building Java Swarm..."

# Check if Maven is installed
if ! command -v mvn &> /dev/null; then
    echo "Error: Maven is not installed. Please install Maven first."
    exit 1
fi

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo "Error: Java is not installed. Please install Java 17 or higher."
    exit 1
fi

# Check Java version
JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
if [ "$JAVA_VERSION" -lt 17 ]; then
    echo "Error: Java 17 or higher is required. Current version: $JAVA_VERSION"
    exit 1
fi

# Clean and compile
echo "Cleaning previous builds..."
mvn clean

echo "Compiling source code..."
mvn compile

echo "Running tests..."
mvn test

echo "Creating executable JAR..."
mvn package

# Check if JAR was created successfully
JAR_FILE="target/java-swarm-1.0.0.jar"
if [ -f "$JAR_FILE" ]; then
    echo "Build successful! Executable JAR created at: $JAR_FILE"
    echo ""
    echo "Usage examples:"
    echo "  java -jar $JAR_FILE --help"
    echo "  java -jar $JAR_FILE --interactive"
    echo "  java -jar $JAR_FILE --input \"Hello, world!\""
    echo ""
    echo "Make sure to set your OpenAI API key:"
    echo "  export OPENAI_API_KEY=\"your-api-key-here\""
else
    echo "Error: Build failed. JAR file not found."
    exit 1
fi
