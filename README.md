## Multi-GPU Jacobi Solvers for 2D Laplace equations

This repository contains the following implementations of the Jacobi method for solving 2D Laplace
equations using C++, CUDA, NCCL, NVSHMEM, OpenMP, and MPI.

- memcpy_single: a single-threaded implementation using cudaMemcpy for GPU communication
- memcpy_multi: a multi-threaded implementation with OpenMP using cudaMemcpy
- memcpy_multi_overlap: a multi-threaded implementation with OpenMP using cudaMemcpy with overlapping communication
- mpi:
- mpi_overlap:
- nccl:
- nccl_overlap:
- nvshmem:

### Requirements

- CUDA
- OpenMP
- MPI
- NCCL
- NVSHMEM
- NVCC-compatible hardware. I used an NVIDIA RTX 4090 for development and a cluster of 4 NVIDIA
  H200s for testing on vast.ai.

### Build

### Run
