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
                  const int nccheck, const bool debug) {
    // DONE: Finish this impl
    float* a;
    float* a_new;

    cudaStream_t compute_stream;
    cudaStream_t push_top_stream;
    cudaStream_t push_bottom_stream;
    cudaEvent_t compute_done;
    cudaEvent_t push_top_done;
    cudaEvent_t push_bottom_done;

    float* l2_norm_d;
    float* l2_norm_h;

    int iy_start = 1;
    int iy_end = (ny - 1);

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * sizeof(float)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * sizeof(float)));
    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(float)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(float)));

    // diriclet boundaries
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, 0, nx, ny, ny);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());
    CUDA_RT_CALL(cudaStreamCreate(&compute_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_top_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_bottom_stream));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_top_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_bottom_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(float)));
    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(float)));
    CUDA_RT_CALL(cudaDeviceSynchronize());

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, (ny + dim_block_y - 1) / dim_block_y, 1);

    bool calculate_norm = true;
    int iter = 0;
    float l2_norm = 1.0;
    double start = omp_get_wtime();
    PUSH_RANGE("Jacobi solve", 0)
    while (l2_norm > tol && iter < iter_max) {
        CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d, 0, sizeof(float), compute_stream));
        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_top_done, 0));
        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_bottom_done, 0));
        calculate_norm = (iter % nccheck) == 0 || (print && ((iter % 100) == 0));
        jacobi_kernel<dim_block_x, dim_block_y>
            <<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, compute_stream>>>(
                a_new, a, l2_norm_d, iy_start, iy_end, nx, calculate_norm);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

        if (calculate_norm)
            CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(float),
                                         cudaMemcpyDeviceToHost, compute_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new, a_new + (iy_end - 1) * nx, nx * sizeof(float),
                                     cudaMemcpyDeviceToDevice, push_top_stream));
        CUDA_RT_CALL(cudaEventRecord(push_top_done, push_top_stream));
        CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new + iy_end * nx, a_new + iy_start * nx, nx * sizeof(float),
                                     cudaMemcpyDeviceToDevice, compute_stream));
        CUDA_RT_CALL(cudaEventRecord(push_bottom_done, push_bottom_stream));

        if (calculate_norm) {
            CUDA_RT_CALL(cudaStreamSynchronize(compute_stream));
            l2_norm = *l2_norm_h;
            l2_norm = std::sqrt(l2_norm);
            if (debug && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        }

        std::swap(a_new, a);
        iter++;
    }
    POP_RANGE
    double stop = omp_get_wtime();

    CUDA_RT_CALL(cudaMemcpy(a_ref_h, a, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaEventDestroy(push_bottom_done));
    CUDA_RT_CALL(cudaEventDestroy(push_top_done));
    CUDA_RT_CALL(cudaEventDestroy(compute_done));
    CUDA_RT_CALL(cudaStreamDestroy(push_bottom_stream));
    CUDA_RT_CALL(cudaStreamDestroy(push_top_stream));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));
    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    CUDA_RT_CALL(cudaFree(l2_norm_d));
    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));
    return (stop - start);
    // return 0.0
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
        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * (chunk_size[dev_id] + 2) * sizeof(float)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * (chunk_size[dev_id] + 2) * sizeof(float)));
        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * (chunk_size[dev_id] + 2) * sizeof(float)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size[dev_id] + 2) * sizeof(float)));

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
        initialize_bounds<<<(ny / num_devices) / 128 + 1, 128>>>(
            a[dev_id], a_new[dev_id], PI, iy_start_global - 1, nx, (chunk_size[dev_id] + 2), ny);

        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());
        CUDA_RT_CALL(cudaStreamCreate(compute_stream + dev_id));
        CUDA_RT_CALL(cudaStreamCreate(push_top_stream + dev_id));
        CUDA_RT_CALL(cudaStreamCreate(push_bottom_stream + dev_id));

        CUDA_RT_CALL(cudaEventCreateWithFlags(compute_done + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(cudaEventCreateWithFlags(push_top_done[0] + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(
            cudaEventCreateWithFlags(push_bottom_done[0] + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(cudaEventCreateWithFlags(push_top_done[1] + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(
            cudaEventCreateWithFlags(push_bottom_done[1] + dev_id, cudaEventDisableTiming));

        CUDA_RT_CALL(cudaMalloc(l2_norm_d + dev_id, sizeof(float)));
        CUDA_RT_CALL(cudaMallocHost(l2_norm_h + dev_id, sizeof(float)));

        if (!nop2p) {
            const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
            int canAccessPeer = 0;

            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top));

            if (canAccessPeer) CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
            const int bottom = (dev_id + 1) % num_devices;
            if (top != bottom) {
                canAccessPeer = 0;
                CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, bottom));
                if (canAccessPeer) CUDA_RT_CALL(cudaDeviceEnablePeerAccess(bottom, 0));
            }
        }
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }
}
