#!/bin/sh
# Build the paint engine as a shared library next to this script.
set -e
cd "$(dirname "$0")"
g++ -O3 -march=native -fPIC -shared -std=c++17 -o libpaint.so paint.cpp
echo "built $(pwd)/libpaint.so"
