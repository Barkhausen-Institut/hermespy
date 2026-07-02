#!/bin/bash


docker login registry.adbi.barkhauseninstitut.org

# Push the docker image for each python version
for v in {11..14}; do
  echo "Pushing Docker image for CP3${v}..."
  docker push "registry.adbi.barkhauseninstitut.org/barkhauseninstitut/wicon/hermespy/cp3${v}:latest"
done