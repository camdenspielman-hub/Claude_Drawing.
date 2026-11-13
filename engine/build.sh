#!/bin/sh
# Build the paint engine and the sculpting helper as shared libraries next to this script.
set -e
cd "$(dirname "$0")"
g++ -O3 -march=native -fPIC -shared -std=c++17 -o libpaint.so paint.cpp
g++ -O3 -march=native -fopenmp -fPIC -shared -std=c++17 -o libsolid.so solid.cpp
echo "built $(pwd)/libpaint.so and libsolid.so"
