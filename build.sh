#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Ensure we are in the project root (where this script should live)
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "Initializing build for sequence alignment project..."

# 1. Create the bin directory if it doesn't exist
if [ ! -d "bin" ]; then
    echo "Creating build directory: ./bin"
    mkdir bin
fi

# Configure the project
# -S .  : Look for CMakeLists.txt in the current directory
# -B bin: Put all build files/binaries into the bin directory
echo "Configuring CMake..."
cmake -S . -B bin

# Build the project
# Using -j with nproc to use all available CPU cores for a faster compile
echo "Building binary..."
cmake --build bin --config Release -j$(nproc 2>/dev/null || echo 4)

echo "------------------------------------------------"
if [ -f "bin/benchmark" ]; then
    echo "Success! Binary generated at: ./bin/benchmark"
else
    echo "Build finished, but 'benchmark' binary was not found in ./bin/"
fi