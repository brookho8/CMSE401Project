#!/bin/bash

module load python
module load CUDA
python3 -m venv testEnvironment
source testEnvironment/bin/activate
pip3 install open3D
pip3 install pygubu
pip3 install paramiko
pip3 install numpy
pip3 install matplotlib
pip3 install scipy
pip3 install scikit-image
pip3 install cupy-cuda100

nvcc cudaBenchmark.cu -o cudaBenchmark
./cudaBenchmark
python3 PythonBenchmark.py