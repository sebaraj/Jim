## Multi-GPU Jacobi Solvers for 2D Laplace equations

This repository contains the following implementations of the Jacobi method for solving 2D Laplace
equations using C++, CUDA, NCCL, OpenMP, and MPI.

- memcpy_single: a single-threaded implementation using cudaMemcpy for GPU communication
- memcpy_multi: a multi-threaded implementation with OpenMP using cudaMemcpy
- memcpy_multi_overlap: a multi-threaded implementation with OpenMP using cudaMemcpy with overlapping communication
- mpi: a multi-process implementation using CUDA-aware MPI
- mpi_overlap: a multi-process implementation using CUDA-aware MPI with overlapping communication
- nccl:
- nccl_overlap:

### Requirements

- CUDA
- OpenMP
- MPI
- NCCL
- NVCC-compatible hardware. I used an NVIDIA RTX 4090 for development and a cluster of 4 NVIDIA
  H200s for testing on vast.ai.

### Build

### Run
