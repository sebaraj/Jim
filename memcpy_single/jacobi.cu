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

double single_gpu(const int nx, const int ny, const int iter_max, float* const a_ref_h,
                  const int nccheck, const bool print) {
    // TODO: Finish this impl
    return 0.0;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments. add anything else?
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const bool csv = get_argbool(argv, argv + argc, "-csv");
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 100);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 100);
    const bool nop2p = get_argbool(argv, argv + argc, "-nop2p");
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);

    float *a[MAX_NUM_DEV], *a_new[MAX_NUM_DEV];
    float *a_h, *a_ref_h;
    double serial_runtime = 0.0;

    cudaStream_t compute_stream[MAX_NUM_DEV];
    cudaStream_t push_top_stream[MAX_NUM_DEV];
    cudaStream_t push_bottom_stream[MAX_NUM_DEV];
    cudaEvent_t compute_done[MAX_NUM_DEV];
    cudaEvent_t push_top_done[2][MAX_NUM_DEV];
    cudaEvent_t push_bottom_done[2][MAX_NUM_DEV];

    int iy_start[MAX_NUM_DEV];
    int iy_end[MAX_NUM_DEV];
    int chunk_size[MAX_NUM_DEV];
    float* l2_norm_d[MAX_NUM_DEV];
    float* l2_norm_h[MAX_NUM_DEV];

    int num_dev = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_dev));
    for (int dev_id = 0; dev_id < num_dev; ++dev_id) {
        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(0));  // reset device memory
        if (dev_id == 0) {
            printf("Using %d devices\n", num_dev);
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(float)));
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(float)));
            serial_runtime = single_gpu(nx, ny, iter_max, a_ref_h, nccheck, !csv);
        }

        // optimize load balancing through row dist.
        int chunk_size_low = (ny - 2) / num_devices;
        int num_ranks_low = num_devices * chunk_size_low + num_devices - (ny - 2);
        int chunk_size_high = chunk_size_low + 1;
        chunk_size[dev_id] = (dev_id < num_ranks_low) ? chunk_size_low : chunk_size_high;
        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * (chunk_size[dev_id] + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * (chunk_size[dev_id] + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * (chunk_size[dev_id] + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size[dev_id] + 2) * sizeof(real)));

        // Calculate local domain boundaries
        int iy_start_global;
        if (dev_id < num_ranks_low) {
            iy_start_global = dev_id * chunk_size_low + 1;
        } else {
            iy_start_global
                = num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }

        iy_start[dev_id] = 1;
        iy_end[dev_id] = iy_start[dev_id] + chunk_size[dev_id];

        // TODO: set boundaries on left/right
    }
}
