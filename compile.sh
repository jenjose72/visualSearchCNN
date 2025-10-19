#!/bin/bash

# Visual Search CNN - Linux/WSL Compilation Script
# Run this from the project root directory

echo "========================================"
echo "Visual Search CNN - Compilation Script"
echo "========================================"
echo ""

# Check if g++ is available
if command -v g++ &> /dev/null; then
    echo "✓ Found g++: $(g++ --version | head -n1)"
else
    echo "✗ g++ not found! Please install g++"
    echo "  Ubuntu/Debian: sudo apt-get install g++ libomp-dev"
    exit 1
fi

echo ""
echo "Compiling Sequential Version..."
cd Sequential
if g++ -O3 -std=c++11 Main.cpp -o cnn_sequential; then
    echo "✓ Sequential version compiled successfully!"
else
    echo "✗ Sequential compilation failed!"
fi
cd ..

echo ""
echo "Compiling OpenMP Version..."
cd Openmp
if g++ -O3 -std=c++11 -fopenmp Main.cpp -o cnn_openmp; then
    echo "✓ OpenMP version compiled successfully!"
else
    echo "✗ OpenMP compilation failed!"
fi
cd ..

echo ""
echo "========================================"
echo "Compilation Complete!"
echo "========================================"
echo ""
echo "To run the programs:"
echo "  Sequential: ./Sequential/cnn_sequential"
echo "  OpenMP:     ./Openmp/cnn_openmp -t 4"
echo ""
