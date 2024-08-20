#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <algorithm>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define WARP_SIZE 32

void cpuSoftMax(float *x, float *y, float *total, int N) {
    for (int i = 0; i < N; i++) {
        y[i] = expf(x[i]);
        *total += y[i];
    }

    for (int i = 0; i < N; i++) {
        y[i] /= *total;
    }
}

float testError(
    void (*gpuSoftMax)(float *, float *, float *, int),
    dim3 gridDim, dim3 blockDim, const int N) {
    size_t size_x = N * sizeof(float);
    size_t size_y = N * sizeof(float);
    size_t size_total = sizeof(float);

    float *h_x, *h_y, *h_total, *d_x, *d_y, *d_total, *h_d_y, h_d_total;
    h_x = (float *)malloc(size_x);
    h_y = (float *)malloc(size_y);
    h_total = (float *)malloc(size_total);
    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_y, size_y);
    cudaMalloc(&d_total, size_total);
    h_d_y = (float *)malloc(size_y);

    srand(time(0));
    for (int i = 0; i < N; i++)
        h_x[i] = rand() / float(RAND_MAX);
    *h_total = 0.0;
    cudaMemset(d_y, 0, size_y);
    cudaMemset(d_total, 0, size_total);

    cpuSoftMax(h_x, h_y, h_total, N);

    cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);
    gpuSoftMax<<<gridDim, blockDim>>>(d_x, d_y, d_total, N);
    cudaMemcpy(h_d_y, d_y, size_y, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_d_total, d_total, size_total, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < N; i++) {
        float this_error = abs(h_d_y[i] - h_y[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = std::max(max_error, this_error);
    }

    free(h_x);
    free(h_y);
    free(h_total);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_total);
    free(h_d_y);

    return max_error;
}

float testPerformance(
    void (*gpuSoftMax)(float *, float *, float *, int),
    dim3 gridDim, dim3 blockDim, const int N, const int repeat) {
    size_t size_x = N * sizeof(float);
    size_t size_y = N * sizeof(float);
    size_t size_total = sizeof(float);

    float *d_x, *d_y, *d_total;
    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_y, size_y);
    cudaMalloc(&d_total, size_total);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        gpuSoftMax<<<gridDim, blockDim>>>(d_x, d_y, d_total, N);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_total);

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
template <const int NUM_THREADS = 128>
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

// Softmax x: N, y: N
// grid(N/128), block(K=128)
template <const int NUM_THREADS = 128>
__global__ void softmax(float *x, float *y, float *total, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
    float sum = block_reduce_sum<NUM_THREADS>(exp_val);
    // get the total sum of all blocks.
    if (tid == 0) atomicAdd(total, sum);
    __threadfence(); // grid level memory fence  注意这里需要网格级别的内存同步
    // e^x_i/sum(e^x_0,...,e^x_n-1)
    if (idx < N) y[idx] = exp_val / (*total);
}

// Softmax Vec4 x: N, y: N
// grid(N/128), block(128/4)
template <const int NUM_THREADS = 128 / 4>
__global__ void softmax_vec4(float *x, float *y, float *total, int N) {
    const int tid = threadIdx.x;
    const int idx = (blockIdx.x * blockDim.x + tid) * 4;

    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_exp;
    reg_exp.x = (idx < N) ? expf(reg_x.x) : 0.0f;
    reg_exp.y = (idx < N) ? expf(reg_x.y) : 0.0f;
    reg_exp.z = (idx < N) ? expf(reg_x.z) : 0.0f;
    reg_exp.w = (idx < N) ? expf(reg_x.w) : 0.0f;
    float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
    float sum = block_reduce_sum<NUM_THREADS>(exp_val);
    // get the total sum of all blocks.
    if (tid == 0) atomicAdd(total, sum);
    __threadfence(); // grid level memory fence  注意这里需要网格级别的内存同步
    // e^x_i/sum(e^x_0,...,e^x_n-1)
    if (idx < N) {
        float4 reg_y;
        reg_y.x = reg_exp.x / (*total);
        reg_y.y = reg_exp.y / (*total);
        reg_y.z = reg_exp.z / (*total);
        reg_y.w = reg_exp.w / (*total);
        FLOAT4(y[idx]) = reg_y;
    }
}

int main(void) {
    {
        printf("\nKernal = softmax\n");
        const int N = 128;
        dim3 blockDim(128);
        dim3 gridDim(N / 128);
        float max_error = testError(softmax, gridDim, blockDim, N);
        printf("Max Error = %f\n", max_error);
    }

    {
        printf("\nKernal = softmax_vec4\n");
        const int N = 128;
        dim3 blockDim(128 / 4);
        dim3 gridDim(N / 128);
        float max_error = testError(softmax_vec4, gridDim, blockDim, N);
        printf("Max Error = %f\n", max_error);
    }

    return 0;
}
