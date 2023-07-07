:: This script will build the docker image for the manylinux build environment of hermespy
:: Docker must be installed and running

@echo off

pushd  %~dp0
docker build -f manylinux-build-env  --tag hermes-manylinux-build-env .
popd
