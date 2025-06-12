#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#ifdef HAVE_CUB
#include <cub/block/block_reduce.cuh>
#endif  // HAVE_CUB

// From NVIDIA repo
#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>

const uint32_t colors[]
    = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                              \
    {                                                      \
        int color = cid;                                   \
        color = color_id % num_colors;                     \
        nvtxEventAttributes_t eventAttrib = {0};           \
        eventAttrib.version = NVTX_VERSION;                \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
        eventAttrib.colorType = NVTX_COLOR_ARGB;           \
        eventAttrib.color = colors[color_id];              \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name;                  \
        nvtxRangePushEx(&eventAttrib);                     \
    }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

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
constexpr int MAX_NUM_DEV = 4;  // cluster size @ vast.ai

__global__ void initialize_bounds(float* __restrict__ const a_new, float* __restrict__ const a,
                                  const float pi, const int offset, const int nx, const int my_ny,
                                  const int ny) {
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny; iy += blockDim.x * gridDim.x) {
        const float y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
        a[iy * nx] = y0;
        a[iy * nx + (nx - 1)] = y0;
        a_new[iy * nx] = y0;
        a_new[iy * nx + (nx - 1)] = y0;
    }
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(float* __restrict__ const a_new, const float* __restrict__ const a,
                              float* __restrict__ const l2_norm, const int iy_start,
                              const int iy_end, const int nx, const bool calculate_norm) {
#ifdef HAVE_CUB
    typedef cub::BlockReduce<float, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y>
        BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
#endif  // HAVE_CUB
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    float local_l2_norm = 0.0;

    if (iy < iy_end && ix < (nx - 1)) {
        const float val_new = 0.25
                              * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] + a[(iy + 1) * nx + ix]
                                 + a[(iy - 1) * nx + ix]);
        a_new[nx * iy + ix] = val_new;

        if (calculate_norm) {
            float residue = val_new - a[iy * nx + ix];
            local_l2_norm += residue * residue;
        }
    }
    if (calculate_norm) {
#ifdef HAVE_CUB
        float block_l2_norm = BlockReduce(temp_storage).Sum(local_l2_norm);
        if (0 == threadIdx.y && 0 == threadIdx.x) atomicAdd(l2_norm, block_l2_norm);
#else
        atomicAdd(l2_norm, local_l2_norm);
#endif  // HAVE_CUB
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

    // float *a[MAX_NUM_DEV],
    float* a_new[MAX_NUM_DEV];
    float *a_h, *a_ref_h;
    double serial_runtime = 0.0;

    cudaEvent_t push_top_done[2][MAX_NUM_DEV];
    cudaEvent_t push_bottom_done[2][MAX_NUM_DEV];

    // int iy_start[MAX_NUM_DEV];
    int iy_end[MAX_NUM_DEV];
    // int chunk_size[MAX_NUM_DEV];
    // float* l2_norm_d[MAX_NUM_DEV];
    // float* l2_norm_h[MAX_NUM_DEV];
    float l2_norm = 1.0;
    bool result_correct = true;
    int num_dev = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_dev));
#pragma opm parallel num_threads(num_dev) shared(l2_norm)
    {
        float* a;
        cudaStream_t compute_stream;
        cudaStream_t push_top_stream;
        cudaStream_t push_bottom_stream;
        cudaEvent_t compute_done;
        float* l2_norm_d;
        float* l2_norm_h;
        int dev_id = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(0));  // reset device memory
        if (dev_id == 0) {
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(float)));
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(float)));
            serial_runtime = single_gpu(nx, ny, iter_max, a_ref_h, nccheck, !csv);
        }
#pragma omp barrier

        // optimize load balancing through row dist.
        int chunk_size;
        int chunk_size_low = (ny - 2) / num_devices;
        int num_ranks_low = num_devices * chunk_size_low + num_devices - (ny - 2);
        int chunk_size_high = chunk_size_low + 1;
        chunk_size[dev_id] = (dev_id < num_ranks_low) ? chunk_size_low : chunk_size_high;
        CUDA_RT_CALL(cudaMalloc(&a, nx * (chunk_size + 2) * sizeof(float)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * (chunk_size + 2) * sizeof(float)));
        CUDA_RT_CALL(cudaMemset(a, 0, nx * (chunk_size + 2) * sizeof(float)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size + 2) * sizeof(float)));

        // Calculate local domain boundaries
        int iy_start_global;
        if (dev_id < num_ranks_low) {
            iy_start_global = dev_id * chunk_size_low + 1;
        } else {
            iy_start_global
                = num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }

        iy_start = 1;
        iy_end[dev_id] = iy_start + chunk_size;

        // TODO: set boundaries on left/right
        initialize_bounds<<<(ny / num_devices) / 128 + 1, 128>>>(
            a, a_new[dev_id], PI, iy_start_global - 1, nx, (chunk_size + 2), ny);

        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());

        int leastPriority = 0;
        int greatestPriority = leastPriority;
        CUDA_RT_CALL(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

        CUDA_RT_CALL(
            cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, leastPriority));
        CUDA_RT_CALL(
            cudaStreamCreateWithPriority(&push_top_stream, cudaStreamDefault, greatestPriority));
        CUDA_RT_CALL(
            cudaStreamCreateWithPriority(&push_bottom_stream, cudaStreamDefault, greatestPriority));
        //
        // CUDA_RT_CALL(cudaStreamCreate(&compute_stream));
        // CUDA_RT_CALL(cudaStreamCreate(&push_top_stream));
        // CUDA_RT_CALL(cudaStreamCreate(&push_bottom_stream));
        //
        // CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
        CUDA_RT_CALL(cudaEventCreateWithFlags(push_top_done[0] + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(
            cudaEventCreateWithFlags(push_bottom_done[0] + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(cudaEventCreateWithFlags(push_top_done[1] + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(
            cudaEventCreateWithFlags(push_bottom_done[1] + dev_id, cudaEventDisableTiming));

        CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(float)));
        CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(float)));

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
        // CUDA_RT_CALL(cudaDeviceSynchronize());
        // }

        for (int i = 0; i < 4; ++i) {
            // for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
            // CUDA_RT_CALL(cudaSetDevice(dev_id));
            // const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
            // const int bottom = (dev_id + 1) % num_devices;
            CUDA_RT_CALL(cudaMemcpyAsync(a_new[top] + (iy_end[top] * nx),
                                         a_new[dev_id] + iy_start * nx, nx * sizeof(real),
                                         cudaMemcpyDeviceToDevice, push_top_stream));
            CUDA_RT_CALL(cudaMemcpyAsync(a_new[bottom], a_new[dev_id] + (iy_end[dev_id] - 1) * nx,
                                         nx * sizeof(real), cudaMemcpyDeviceToDevice,
                                         push_bottom_stream));
            // }
            // for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
            // CUDA_RT_CALL(cudaSetDevice(dev_id));
            CUDA_RT_CALL(cudaStreamSynchronize(push_top_stream));
            CUDA_RT_CALL(cudaStreamSynchronize(push_bottom_stream));
            // }
            // for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
            std::swap(a_new[dev_id], a[dev_id]);
            // }
        }
        CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma opm master
        {
            if (!csv) printf("norm\n");
        }

        constexpr int dim_block_x = 32;
        constexpr int dim_block_y = 32;
        dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x,
                      (ny + (num_devices * dim_block_y) - 1) / (num_devices * dim_block_y), 1);
        int iter = 0;
        bool calculate_norm = true;
#pragma omp master
        {
            real l2_norm = 1.0;
        }
        CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp barrier

        // for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        //     CUDA_RT_CALL(cudaSetDevice(dev_id));
        //     CUDA_RT_CALL(cudaDeviceSynchronize());
        // }
        double start = omp_get_wtime();
        PUSH_RANGE("Jacobi solve", 0)
        while (l2_norm > tol && iter < iter_max) {
            // for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
            CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d, 0, sizeof(real), compute_stream));

            const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
            const int bottom = (dev_id + 1) % num_devices;
            // CUDA_RT_CALL(cudaSetDevice(dev_id));

            CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_top_done[(iter % 2)][bottom], 0));
            CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_bottom_done[(iter % 2)][top], 0));

            calculate_norm = (iter % nccheck) == 0 || (!csv && (iter % 100) == 0);
            // dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x,
            // (chunk_size[dev_id] + dim_block_y - 1) / dim_block_y, 1);

            jacobi_kernel<dim_block_x, dim_block_y>
                <<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, compute_stream[dev_id]>>>(
                    a_new[dev_id], a, l2_norm_d, iy_start, iy_end[dev_id], nx, calculate_norm);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

            if (calculate_norm)
                CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_h[dev_id], l2_norm_d[dev_id], sizeof(real),
#pragma omp barrier
            // periodic boundary conds
            CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, compute_done, 0));
            CUDA_RT_CALL(cudaMemcpyAsync(a_new[top] + (iy_end[top] * nx),
                                         a_new + iy_start * nx, nx * sizeof(real),
                                         cudaMemcpyDeviceToDevice, push_top_stream));
            CUDA_RT_CALL(
                cudaEventRecord(push_top_done[((iter + 1) % 2)][dev_id], push_top_stream));

            CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, compute_done, 0));
            CUDA_RT_CALL(cudaMemcpyAsync(a_new[bottom], a_new[dev_id] + (iy_end[dev_id] - 1) * nx,
                                         nx * sizeof(real), cudaMemcpyDeviceToDevice,
                                         push_bottom_stream));
            CUDA_RT_CALL(cudaEventRecord(push_bottom_done[((iter + 1) % 2)][dev_id],
                                         push_bottom_stream));
        }
#pragma omp barrier
        if (calculate_norm) {
#pragma omp single
            {
                l2_norm = 0.0;
            }
#pragma omp barrier
            // for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
            CUDA_RT_CALL(cudaStreamSynchronize(compute_stream));
#pragma omp atomic
            l2_norm += *(l2_norm_h);
            // }
#pragma omp barrier
#pragma opm single
            {
                l2_norm = std::sqrt(l2_norm);
            }
#pragma omp barrier
            if (!csv && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        }

#pragma omp master
        // for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        std::swap(a_new[dev_id], a);
        // }
        iter++;
    }
    // for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
    // CUDA_RT_CALL(cudaSetDevice(dev_id));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    // }
#pragma omp barrier
    double stop = omp_get_wtime();
    POP_RANGE

    // int offset = nx;
    // for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
    CUDA_RT_CALL(cudaMemcpy(a_h + iy_start_global * nx, a + nx,
                            std::min((ny - iy_start_global) * nx, chunk_size * nx) * sizeof(float),
                            cudaMemcpyDeviceToHost));
    // CUDA_RT_CALL(
    // cudaMemcpy(a_h + offset, a + nx,
    // std::min((nx * ny) - offset, nx * chunk_size[dev_id]) * sizeof(real),
    // cudaMemcpyDeviceToHost));
    // offset += std::min(chunk_size[dev_id] * nx, (nx * ny) - offset);
#pragma omp barrier

#pragma omp master
    {
        bool result_correct = true;
        for (int iy = 1; result_correct && (iy < (ny - 1)); ++iy) {
            for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
                if (std::fabs(a_ref_h[iy * nx + ix] - a_h[iy * nx + ix]) > tol) {
                    fprintf(stderr,
                            "ERROR: a[%d * %d + %d] = %f does not match %f "
                            "(reference)\n",
                            iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                    result_correct = false;
                }
            }
        }
        if (result_correct) {
            if (csv) {
                printf("single_threaded_copy, %d, %d, %d, %d, %d, %d, %f, %f\n", nx, ny, iter_max,
                       nccheck, num_devices, nop2p ? 0 : 1, (stop - start), runtime_serial);
            } else {
                printf("Num GPUs: %d.\n", num_devices);
                printf(
                    "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, "
                    "efficiency: %8.2f \n",
                    ny, nx, runtime_serial, num_devices, (stop - start),
                    runtime_serial / (stop - start),
                    runtime_serial / (num_devices * (stop - start)) * 100);
            }
        }
    }  // omp master

    // for (int dev_id = (num_devices - 1); dev_id >= 0; --dev_id) {
    // CUDA_RT_CALL(cudaSetDevice(dev_id));
    CUDA_RT_CALL(cudaEventDestroy(push_bottom_done[1][dev_id]));
    CUDA_RT_CALL(cudaEventDestroy(push_top_done[1][dev_id]));
    CUDA_RT_CALL(cudaEventDestroy(push_bottom_done[0][dev_id]));
    CUDA_RT_CALL(cudaEventDestroy(push_top_done[0][dev_id]));
    CUDA_RT_CALL(cudaEventDestroy(compute_done));
    CUDA_RT_CALL(cudaStreamDestroy(push_bottom_stream));
    CUDA_RT_CALL(cudaStreamDestroy(push_top_stream));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    CUDA_RT_CALL(cudaFree(l2_norm_d));

    CUDA_RT_CALL(cudaFree(a_new[dev_id]));
    CUDA_RT_CALL(cudaFree(a));
    if (dev_id == 0) {
        CUDA_RT_CALL(cudaFreeHost(a_h));
        CUDA_RT_CALL(cudaFreeHost(a_ref_h));
    }
}

return !result_correct;
}
