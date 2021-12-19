# 2D-Ising-Model-CUDA-to-OpenCL

This repository contains an implementation of the 2D Ising Model on GPUs. The original code, written in CUDA, was written by the authors of the paper [GPU accelerated Monte Carlo simulation of the 2D and 3D Ising model](https://www.sciencedirect.com/science/article/pii/S0021999109001387). This original code can be found in the "Original CUDA Code Plus Annotations" folder. Also in that folder is a copy of that same code, but with annotations - in particular of the code running on GPUs - to make it clear exactly how the code is functioning.

In the "OpenCL Code" folder is a conversion of the original CUDA code to a set of files that run the same 2D Ising Model on GPUs or CPUs in OpenCL. 
