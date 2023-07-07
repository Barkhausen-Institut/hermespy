:: This script builds and packages the Linux release of HermesPy.
:: It requires Docker to be installed and running, with the hermes-manylinux-build-env image built.

@echo off

:: Run the manylinux build environment
docker run^
 --name hermes-manylinux-build^
 --mount type=bind,source=%~dp0\..\..\,target=/hermespy/^
 -it^
 --rm^
 hermes-manylinux-build-env:latest
