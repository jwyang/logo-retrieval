#!/bin/bash -e

# change to script directory
cd $(dirname $(readlink -f $0))

cd ..

cd build/src/
make -j8 "$@"
