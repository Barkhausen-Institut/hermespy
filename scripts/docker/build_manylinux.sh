#!/bin/bash
set -e -u -x

# Configure the python version
version=1.5.0

# Configure environment to use all available cores while building
export MAKEFLAGS="-j $(grep -c ^processor /proc/cpuinfo)"

# Declare Python versions to build for
declare -a versions=( "cp312-cp312" "cp311-cp311" "cp310-cp310")

# Build for every available python version
for PYBIN in ${versions[@]}; do
    "/opt/python/${PYBIN}/bin/pip" install --upgrade build
    "/opt/python/${PYBIN}/bin/python" -m build --wheel hermespy/ --outdir /dist/
done

# Repair all wheels, i.e. make the names compliant with manylinux standard
mkdir -p /wheelhouse/
for PYBIN in ${versions[@]}; do
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hermespy/build/${PYBIN}-linux_x86_64/submodules/affect/lib/ auditwheel repair /dist/hermespy-${version}-${PYBIN}-linux_x86_64.whl -w /hermespy/dist
done
