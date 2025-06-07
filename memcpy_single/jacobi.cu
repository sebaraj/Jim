// #include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

// Generic argument parsing functions
template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T val = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> val;
    }
    return val;
}

bool get_argbool(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    return (itr != end);
}

// CUDA function wrapper to help with debugging. From NVIDIA repo
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

constexpr float PI = M_PI;
constexpr float tol = 1.0e-8;
constexpr int MAX_NUM_DEV = 8;  // max for aws/gcp instance options?

void initialize_bounds(float* const a_new, float* const a, const float pi, const int offset,
                       const int nx, const int my_ny, const int ny) {
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny; iy += blockDim.x * gridDim.x) {
        const float y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
        a[iy * nx] = y0;
        a[iy * nx + (nx - 1)] = y0;
        a_new[iy * nx] = y0;
        a_new[iy * nx + (nx - 1)] = y0;
    }
}

int main(int argc, char* argv[]) {
    // Parse command line arguments. add anything else?
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const bool csv = get_argbool(argv, argv + argc, "-csv");
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 100);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 100);

    float *a[MAX_NUM_DEV], *a_new[MAX_NUM_DEV];
    float *a_h, *a_ref_h;
    double serial_runtime = 0.0;

    cudaStream_t compute_stream[MAX_NUM_DEVICES];
    cudaStream_t push_top_stream[MAX_NUM_DEVICES];
    cudaStream_t push_bottom_stream[MAX_NUM_DEVICES];
    cudaEvent_t compute_done[MAX_NUM_DEVICES];
    cudaEvent_t push_top_done[2][MAX_NUM_DEVICES];
    cudaEvent_t push_bottom_done[2][MAX_NUM_DEVICES];

    int iy_start[MAX_NUM_DEVICES];
    int iy_end[MAX_NUM_DEVICES];

    int chunk_size[MAX_NUM_DEVICES];
}
