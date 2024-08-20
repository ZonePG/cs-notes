#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <algorithm>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define WARP_SIZE 32

void cpuSgemv(
    float *a, float *b, float *c, const int M, const int K) {
    for (int m = 0; m < M; m++) {
        float psum = 0.0;
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[k];
            c[m] = psum;
        }
    }
}

float testError(
    void (*gpuSgemv)(float *, float *, float *, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int K) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * sizeof(float);
    size_t size_c = M * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemv(h_a, h_b, h_c, M, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemv<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = std::max(max_error, this_error);
    }

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
    void (*gpuSgemv)(float *, float *, float *, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int K, const int repeat) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * sizeof(float);
    size_t size_c = M * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemv<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K);
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

// SGEMV: Warp SGEMV K32
// 假设K为32的倍数，每个warp负责一行
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k32(float *a, float *x, float *y, int M, int K) {
    int tx = threadIdx.x;         // 0~31
    int ty = threadIdx.y;         // 0~4
    int bx = blockIdx.x;          // 0~M/4
    int lane = tx % WARP_SIZE;    // 0~31
    int m = bx * blockDim.y + ty; // (0~M/4) * 4 + (0~3)
    if (m < M) {
        float sum = 0.0f;
        int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            // 若NUM_WARPS>=2，先将当前行的数据累加到第一个warp中
            int k = w * WARP_SIZE + lane;
            sum += a[OFFSET(m, k, K)] * x[k];
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == 0) y[m] = sum;
    }
}

// SGEMV: Warp SGEMV K128 + Vec4
// 假设K为128的倍数 float4
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k128(float *a, float *x, float *y, int M, int K) {
    // 每个线程负责4个元素，一个warp覆盖128个元素
    int tx = threadIdx.x;         // 0~31
    int ty = threadIdx.y;         // 0~3
    int bx = blockIdx.x;          // 0~M/4
    int lane = tx % WARP_SIZE;    // 0~31
    int m = blockDim.y * bx + ty; // (0~M/4) * 4 + (0~3)

    if (m < M) {
        float sum = 0.0f;
        // process 4*WARP_SIZE elements per warp.
        int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k = (w * WARP_SIZE + lane) * 4;
            float4 reg_x = FLOAT4(x[k]);
            float4 reg_a = FLOAT4(a[OFFSET(m, k, K)]);
            sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y
                    + reg_a.z * reg_x.z + reg_a.w * reg_x.w);
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == 0) y[m] = sum;
    }
}

// SGEMV: Warp SGEMV K16
// 假设K为16 < 32,每个warp负责2行，每行有16个元素
// NUM_THREADS=128, NUM_WARPS=NUM_THREADS/WARP_SIZE;
// NUM_ROWS=NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32,NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template <const int ROW_PER_WARP = 2>
__global__ void sgemv_k16(float *A, float *x, float *y, int M, int K) {
    constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
    int tx = threadIdx.x;       // 0~31
    int ty = threadIdx.y;       // 0~NUM_WARPS
    int bx = blockIdx.x;        // 0~M/NUM_ROWS (NUM_ROWS=NUM_WARPS * ROW_PER_WARP)
    int lane = tx % WARP_SIZE;  // 0~31
    int k = lane % K_WARP_SIZE; // 0~15
    // gloabl row of a: MxK and y:Mx1, blockDim.y=NUM_WARPS
    int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
    if (m < M) {
        float sum = A[OFFSET(m, k, K)] * x[k];
        sum = warp_reduce_sum<K_WARP_SIZE>(sum);
        // 注意是k == 0，而不是lane == 0
        if (k == 0) y[m] = sum;
    }
}

int main(void) {
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int TESTNUM = 15;
    const int outer_repeat = 10, inner_repeat = 1;

    {
        printf("\nKernal = sgemv_k32\n");
        const int M = 512, K = 512;
        dim3 blockDim(32, 4);
        dim3 gridDim((M + blockDim.y - 1) / blockDim.y);
        float max_error = testError(sgemv_k32, gridDim, blockDim, M, K);
        printf("Max Error = %f\n", max_error);

        for (int i = 0; i < TESTNUM; i++) {
            const int M = M_list[i], K = K_list[i];

            dim3 blockDim(32, 4);
            dim3 gridDim((M + blockDim.y - 1) / blockDim.y);

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int j = 0; j < outer_repeat; j++) {
                double this_sec = testPerformance(sgemv_k128, gridDim, blockDim, M, K, inner_repeat);
                max_sec = std::max(max_sec, this_sec);
                min_sec = std::min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * K * 2 / 1024 / 1024 / 1024 / avg_sec;

            printf("M K = %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, K, min_sec, avg_sec, max_sec, avg_Gflops);
        }
    }

    {
        printf("\nKernal = sgemv_k128\n");
        const int M = 512, K = 512;
        dim3 blockDim(32, 4);
        dim3 gridDim((M + blockDim.y - 1) / blockDim.y);
        float max_error = testError(sgemv_k128, gridDim, blockDim, M, K);
        printf("Max Error = %f\n", max_error);
    }

    {
        printf("\nKernal = sgemv_k16\n");
        const int M = 512, K = 16;
        dim3 blockDim(32, 4);
        dim3 gridDim((M + blockDim.y - 1) / blockDim.y);
        float max_error = testError(sgemv_k16<2>, gridDim, blockDim, M, K);
        printf("Max Error = %f\n", max_error);
    }

    return 0;
}