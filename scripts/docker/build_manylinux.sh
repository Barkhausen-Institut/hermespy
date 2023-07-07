#!/bin/bash
set -e -u -x

# Configure environment to use all available cores while building
export MAKEFLAGS="-j $(grep -c ^processor /proc/cpuinfo)"

# Declare Pythion versions to build for
declare -a versions=("cp310-cp310" "cp39-cp39")

# Remove manifest file
rm -f /hermespy/MANIFEST.in

# Build for every available python version
for PYBIN in ${versions[@]}; do
    "/opt/python/${PYBIN}/bin/pip" install --upgrade build
    "/opt/python/${PYBIN}/bin/python" -m build --wheel hermespy/ --outdir /dist/
done

# Repair all wheels, i.e. make the names compliant with manylinux standard
mkdir -p /wheelhouse/
for whl in dist/*.whl; do
    auditwheel repair "$whl" -w /wheelhouse/
done

# Copy wheels into the hermespy project dist folder
mkdir -p /hermespy/dist/
cp /wheelhouse/* /hermespy/dist/
