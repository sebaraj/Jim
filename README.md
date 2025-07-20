## Multi-GPU Jacobi Solvers for 2D Laplace equations

This repository contains the following implementations of the Jacobi method for solving 2D Laplace
equations using C++, CUDA, NCCL, OpenMP, and MPI.

- memcpy_single: a single-threaded implementation using cudaMemcpy for GPU communication
- memcpy_multi: a multi-threaded implementation with OpenMP using cudaMemcpy
- memcpy_multi_overlap: a multi-threaded implementation with OpenMP using cudaMemcpy with overlapping communication
- mpi: a multi-process implementation using CUDA-aware MPI
- mpi_overlap: a multi-process implementation using CUDA-aware MPI with overlapping communication
- nccl: a multi-process implementation with MPI + NCCL using NCCL for GPU communication
- nccl_overlap: a multi-process implementation with MPI + NCCL using NCCL with overlapping
  communication

### Requirements

- CUDA
- OpenMP/OpenMP capable compiler (e.g., GCC, Clang)
- MPI (CUDA-aware implementation, e.g. OpenMPI)
- NCCL
- NVCC-compatible hardware. I used an NVIDIA RTX 4090 for development and a cluster of 4 NVIDIA
  H200s for testing on vast.ai.

### Build

Each implementation has its own `Makefile`. Navigate to the desired implementation directory and run:

```
jim$ cd nccl_overlap
nccl_overlap$ make
nvcc -DHAVE_CUB -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -ldl -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -std=c++14 jacobi.cu -o jacobi
nccl_overlap$ ls jacobi
./jacobi
```

### Run

All implementations can be run with the following options:

- -niter: How many iterations to carry out (default 1000)
- -nccheck: How often to check for convergence (default 1)
- -nx: Size of the domain in x direction (default 16384)
- -ny: Size of the domain in y direction (default 16384)
- -csv: Print performance results as -csv
- -use_hp_streams: use high priority streams for `mpi_overlap` to hide kernel launch latencies of boundary kernels

- e.g. `./jacobi -niter 1000 -nccheck 1 -nx 16384 -ny 16384 -csv`
