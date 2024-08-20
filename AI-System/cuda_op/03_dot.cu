#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <algorithm>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define WARP_SIZE 32

void cpuDot(float *a, float *b, float *y, const int N) {
    for (int i = 0; i < N; i++) {
        *y += a[i] * b[i];
    }
}

float testError(
    void (*gpuDot)(float *, float *, float *, const int), dim3 gridDim, dim3 blockDim, const int N) {
    size_t size_a = N * sizeof(float);
    size_t size_b = N * sizeof(float);
    size_t size_c = sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < N; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    *h_c = 0.0f;
    cudaMemset(d_c, 0, size_c);

    cpuDot(h_a, h_b, h_c, N);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuDot<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = *h_d_c - *h_c;

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testPerformance(
    void (*gpuDot)(float *, float *, float *, const int),
    dim3 gridDim, dim3 blockDim, const int N, const int repeat) {
    size_t size_a = N * sizeof(float);
    size_t size_b = N * sizeof(float);
    size_t size_c = sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuDot<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}

// Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
// grid 1D block 1D, grid(N/128), block(128)
template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_sum(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];
  
    val = warp_reduce_sum<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    val = warp_reduce_sum<NUM_WARPS>(val);
    return val;
}

// Dot Product
// grid(N/128), block(128)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template <const int NUM_THREADS = 128>
__global__ void dot(float *a, float *b, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
    prod = block_reduce_sum<NUM_THREADS>(prod);
    if (tid == 0) atomicAdd(y, prod);
}

// Dot Product + Vec4
// grid(N/128), block(128/4)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template <const int NUM_THREADS = 128 / 4>
__global__ void dot_vec4(float *a, float *b, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float prod = (idx < N) ? (reg_a.x * reg_b.x + reg_a.y * reg_b.y
                              + reg_a.z * reg_b.z + reg_a.w * reg_b.w) :
                             0.0f;
    prod = block_reduce_sum<NUM_THREADS>(prod);
    if (tid == 0) atomicAdd(y, prod);
}

int main(void) {
    {
        printf("\nKernal = dot\n");
        const int N = 128;
        dim3 blockDim(128);
        dim3 gridDim(N / 128);
        float max_error = testError(dot, gridDim, blockDim, N);
        printf("Max Error = %f\n", max_error);
    }

    {
        printf("\nKernal = dot_vec4\n");
        const int N = 128;
        dim3 blockDim(128 / 4);
        dim3 gridDim(N / 128);
        float max_error = testError(dot_vec4, gridDim, blockDim, N);
        printf("Max Error = %f\n", max_error);
    }

    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int TESTNUM = 15;
    const int outer_repeat = 10, inner_repeat = 1;

    for (int i = 0; i < TESTNUM; i++) {
        const int N = N_list[i];

        dim3 blockDim(128 / 4);
        dim3 gridDim(N / 128);

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(dot_vec4, gridDim, blockDim, N, inner_repeat);
            max_sec = std::max(max_sec, this_sec);
            min_sec = std::min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)N * 2) / 1024 / 1024 / 1024 / avg_sec;

        printf("N = %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", N, min_sec, avg_sec, max_sec, avg_Gflops);
    }

    return 0;
}