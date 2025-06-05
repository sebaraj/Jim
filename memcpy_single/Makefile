NVCC=nvcc
# Ampere
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 

# Hopper
GENCODE_SM90    := -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
GENCODE_FLAGS	:= $(GENCODE_SM80) $(GENCODE_SM90)

ifdef DISABLE_CUB
        NVCC_FLAGS = -Xptxas --optimize-float-atomics
else
        NVCC_FLAGS = -DHAVE_CUB
endif

NVCC_FLAGS += -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -ldl $(GENCODE_FLAGS) -std=c++17

jacobi: Makefile jacobi.cu
	$(NVCC) $(NVCC_FLAGS) jacobi.cu -o jacobi

.PHONY.: clean
clean:
	rm -f jacobi jacobi.nsys-rep

run: jacobi
	./jacobi

profile: jacobi
	nsys profile --trace=cuda,nvtx -o jacobi ./jacobi -niter 10

sanitize: jacobi
	compute-sanitizer ./jacobi -niter 5

