NVCC=nvcc


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

