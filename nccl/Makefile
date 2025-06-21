NP ?= 1
NVCC=nvcc
MPICXX=mpicxx
MPIRUN ?= mpirun
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
# Ampere
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 

# RTX 4090
GENCODE_SM89    := -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89

# Hopper (H200)
GENCODE_SM90    := -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
GENCODE_FLAGS	:= $(GENCODE_SM80) $(GENCODE_SM89) $(GENCODE_SM90)

ifdef DISABLE_CUB
        NVCC_FLAGS = -Xptxas --optimize-float-atomics
else
        NVCC_FLAGS = -DHAVE_CUB
endif
ifdef SKIP_CUDA_AWARENESS_CHECK
        MPICXX_FLAGS = -DSKIP_CUDA_AWARENESS_CHECK
endif

NVCC_FLAGS += -lineinfo $(GENCODE_FLAGS) -std=c++17
MPICXX_FLAGS += -DUSE_NVTX -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -std=c++17
LD_FLAGS = -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib -lcudart -ldl -lnccl

jacobi: Makefile jacobi.cpp jacobi_kernels.o
	$(MPICXX) $(MPICXX_FLAGS) jacobi.cpp jacobi_kernels.o $(LD_FLAGS) -o jacobi

jacobi_kernels.o: Makefile jacobi_kernels.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_kernels.cu -c

.PHONY.: clean
clean:
	rm -f jacobi jacobi_kernels.o *.nsys-rep jacobi.*.compute-sanitizer.log

sanitize: jacobi
	$(MPIRUN) -np $(NP) compute-sanitizer --log-file jacobi.%q{OMPI_COMM_WORLD_RANK}.compute-sanitizer.log ./jacobi -niter 10

run: jacobi
	$(MPIRUN) -np $(NP) ./jacobi

profile: jacobi
	$(MPIRUN) -np $(NP) nsys profile --trace=mpi,cuda,nvtx -o jacobi.%q{OMPI_COMM_WORLD_RANK} ./jacobi -niter 10

