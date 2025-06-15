// jacobi_kernals.cu
#include <cstdio>
#include <cstdlib>

#ifdef HAVE_CUB
#include <cub/block/block_reduce.cuh>
#endif  // HAVE_CUB

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit(cudaStatus);                                                               \
        }                                                                                   \
    }

#ifdef USE_DOUBLE
#define MPI_float_TYPE MPI_DOUBLE
#else
typedef float float;
#define MPI_float_TYPE MPI_FLOAT
#endif

__global__ void initialize_boundaries(float* __restrict__ const a_new, float* __restrict__ const a,
                                      const float pi, const int offset, const int nx,
                                      const int my_ny, const int ny) {
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny; iy += blockDim.x * gridDim.x) {
        const float y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
        a[iy * nx + 0] = y0;
        a[iy * nx + (nx - 1)] = y0;
        a_new[iy * nx + 0] = y0;
        a_new[iy * nx + (nx - 1)] = y0;
    }
}

void launch_initialize_boundaries(float* __restrict__ const a_new, float* __restrict__ const a,
                                  const float pi, const int offset, const int nx, const int my_ny,
                                  const int ny) {
    initialize_boundaries<<<my_ny / 128 + 1, 128>>>(a_new, a, pi, offset, nx, my_ny, ny);
    CUDA_RT_CALL(cudaGetLastError());
}

