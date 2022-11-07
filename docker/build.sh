#!/bin/bash

registry="registry.gitlab.com/barkhauseninstitut/wicon/hermespy"

# Figure out the script's base path
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# Force copy only required setup data
rm -f $SCRIPTPATH/setup.py && rm -f $SCRIPTPATH/README.md
cp $SCRIPTPATH/../setup.py $SCRIPTPATH && cp $SCRIPTPATH/../README.md $SCRIPTPATH

# Build the docker container
docker build -t $registry:python-39-linux -f $SCRIPTPATH/linux-build-env $SCRIPTPATH

# Remove required setup data
rm -f $SCRIPTPATH/setup.py && rm -f $SCRIPTPATH/README.md
