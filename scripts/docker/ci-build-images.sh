#!/bin/bash


# Get current script directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build a separate image for each Python version
for v in {11..14}; do
  echo "Building Docker image for CP3${v}..."
  docker build -t "registry.adbi.barkhauseninstitut.org/barkhauseninstitut/wicon/hermespy/cp3${v}:latest" --build-arg "PYTHON_VERSION=3${v}" -f "${SCRIPT_DIR}/ci-env" "${SCRIPT_DIR}/../../"
done
