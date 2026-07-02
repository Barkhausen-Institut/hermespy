#!/bin/bash


# Get current script directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo $SCRIPT_DIR

# Build and run tests for each Python version
for v in {11..14}; do
  echo "Running unit tests for CP3${v}..."
  docker run --rm --name "cp3${v}_tests" -it --mount type=bind,src="${SCRIPT_DIR}/../../",dst="/hermes" "registry.adbi.barkhauseninstitut.org/barkhauseninstitut/wicon/hermespy/cp3${v}:latest" /bin/bash -c "pip install -ve /hermes[test] && python /hermes/tests/test_install.py unit_tests"
done