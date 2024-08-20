# CUDA 算子手撕

参考：[DefTruth/CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes)

## sgemm

参考：[CUDA（三）：通用矩阵乘法：从入门到熟练](https://zhuanlan.zhihu.com/p/657632577)

```c++
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <algorithm>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

float testError(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

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
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
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
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
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

// SGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major
__global__ void sgemm(float *a, float *b, float *c, int M, int N, int K) {
    // [1] Block Tile: 32x32的block处理c上一块32x32的元素计算
    // [2]     K Tile: 使用共享内存，并将K分块为BK大小的块
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    __shared__ float s_a[BM][BK], s_b[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx; // tid within the block
    // load values to shared memory, 32x32 threads working together
    // to fetch data along the row direction of a and b both for s_a
    // and s_b 32x32x4x2=8KB, we use 32x32 threads within block to
    // load 32x32 elements from global memory to shared memory, namely,
    // each thread will load 1 element.
    int load_smem_a_m = tid / 32;                // 0~31, tid / 32, tid / BM, threadIdx.y
    int load_smem_a_k = tid % 32;                // 0~31, tid % 32, tid % BK, threadIdx.x
    int load_smem_b_k = tid / 32;                // 0~31, tid / 32, tid / BK, threadIdx.y
    int load_smem_b_n = tid % 32;                // 0~31, tid % 32, tid % BN, threadIdx.x
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    // if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    float sum = 0.f;
    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = OFFSET(load_gmem_a_m, load_gmem_a_k, K);
        s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = OFFSET(load_gmem_b_k, load_gmem_b_n, N);
        s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            int comp_smem_a_m = load_smem_a_m;
            int comp_smem_b_n = load_smem_b_n;
            sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
        __syncthreads();
    }
    int store_gmem_c_m = load_gmem_a_m;
    int store_gmem_c_n = load_gmem_b_n;
    int store_gmem_c_addr = OFFSET(store_gmem_c_m, store_gmem_c_n, N);
    c[store_gmem_c_addr] = sum;
}

// SGEMM: Block Tile + Thread Tile + K Tile + Vec4, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
__global__ void sgemm_thread_tile_vec4(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    const int M, const int N, const int K) {
    // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
    // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
    // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
    //                  每次计算TM*TN个元素各自的部分乘累加
    // [4]   Vectorize: 减少load和store指令，使用float4
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx; // tid within the block

    // 2*128*8*4=8KB
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    // 8*8
    float r_c[TM][TN] = {0.0};

    // 0. 先计算shared memory中的索引
    // tid 和需要加载的 smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A 行主序
    // 对于 s_a 每行 8 个数据，每个线程读取 4 个，需要 2 个线程；总共128行，需要 128x2 刚好 256 线程
    int load_a_smem_m = tid >> 1;       // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2; // (tid % 2 == 0) ? 0 : 4, col of s_a
    // tid 和需要加载的 smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B 行主序
    // 对于 s_b 每行 128 个数据，每个线程读 4 个数据，需要 32 个线程；总共 8 行，需要 32x8=256 个线程
    int load_b_smem_k = tid >> 5;        // tid/32, row of s_b
    int load_b_smem_n = (tid & 31) << 2; // (tid % 32) * 4, col of s_b
    // 1. 再计算全局内存中的索引
    // 要加载到 s_a 中的元素对应到 A 全局内存中的行数，每个 block 负责出 C 中大小为 BM*BN 的块
    int load_a_gmem_m = by * BM + load_a_smem_m; // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n; // global col of b

    // 2. 先对 K 进行分块，每块 BK 大小
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        // 加载数据到共享内存 smem s_a BM*BK 128*8 vectorize float4
        int load_a_gmem_k = bk * BK + load_a_smem_k; // global col of a
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        // 加载数据到共享内存 smem s_b BK*BN 8*128 vectorize float4
        int load_b_gmem_k = bk * BK + load_b_smem_k; // global row of b
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; k++) {
// 3. 每个线程负责计算 BM*BN(128x128) 中的 TM*TN(8x8) 个元素
#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    // k from 0~7，0 ~ BK, ty and tx range from 0 to 15, 16x8=128
                    int comp_a_smem_m = ty * TM + m; // 128*8 128/TM(8)=16 M方向 16线程
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; m++) {
        int store_c_gmem_m = by * BM + ty * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + n;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}

int main(void) {
    printf("\nKernal = sgemm\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int) = sgemm_thread_tile_vec4;

    {
        const int M = 512, N = 512, K = 512;
        dim3 blockDim(32, 32);
        dim3 gridDim((N + 32 - 1) / 32, (M + 32 - 1) / 32);
        float max_error = testError(sgemm, gridDim, blockDim, M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    {
        const int M = 512, N = 512, K = 512;
        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        float max_error = testError(gpuSgemm, gridDim, blockDim, M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

    const int TESTNUM = 15;
    for (int i = 0; i < TESTNUM; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = std::max(max_sec, this_sec);
            min_sec = std::min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }
    return 0;
}
```

## warp/block reduce sum/max

参考：[CUDA编程入门之Warp-Level Primitives](https://zhuanlan.zhihu.com/p/572820783)

```c++
// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
      val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Warp Reduce Max
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
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

template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_max(float val) {
    // always <= 32 warps per block (limited by 1024 threads per block)
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];
    
    val = warp_reduce_max<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    val = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
    val = warp_reduce_max<NUM_WARPS>(val);
    return val;
}
```

## block all reduce + vec4

```c++
// Block All Reduce Sum
// grid(N/128), block(128)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 128>
__global__ void block_all_reduce_sum(float* a, float* y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    // keep the data in register is enougth for warp operaion.
    float sum = (idx < N) ? a[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp sync reduce.
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    // warp leaders store the data to shared memory.
    if (lane == 0) reduce_smem[warp] = sum;
    __syncthreads(); // make sure the data is in shared memory.
    // the first warp compute the final sum.
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(y, sum);
}

// Block All Reduce Sum + float4
// grid(N/128), block(128/4)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 128/4>
__global__ void block_all_reduce_sum_vec4(float* a, float* y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float4 reg_a = FLOAT4(a[idx]);
    // keep the data in register is enougth for warp operaion.
    float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp sync reduce.
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    // warp leaders store the data to shared memory.
    if (lane == 0) reduce_smem[warp] = sum;
    __syncthreads(); // make sure the data is in shared memory.
    // the first warp compute the final sum.
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(y, sum);
}
```

## sgemv k32/k128/k16 kernel

参考：[深入浅出GPU优化系列：gemv优化](https://zhuanlan.zhihu.com/p/494144694)

```c++
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
```

## dot product, dot product + vec4

```c++
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
```

## elementwise, elementwise + vec4

```c++
// ElementWise Add
// grid(N/128), block(128)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

// ElementWise Add + Vec4
// grid(N/128), block(128/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_vec4(float *a, float *b, float *c, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(c[idx]) = reg_c;
    }
}
```

## histogram, histogram + vec4

```c++
// Histogram
// grid(N/128), block(128)
// a: Nx1, y: count histogram
__global__ void histogram(int *a, int *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(&(y[a[idx]]), 1);
}

// Histogram + Vec4
// grid(N/128), block(128/4)
// a: Nx1, y: count histogram
__global__ void histogram_vec4(int *a, int *y, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        int4 reg_a = INT4(a[idx]);
        atomicAdd(&(y[reg_a.x]), 1);
        atomicAdd(&(y[reg_a.y]), 1);
        atomicAdd(&(y[reg_a.z]), 1);
        atomicAdd(&(y[reg_a.w]), 1);
    }
}
```

## softmax, softmax + vec4

通用实现可以参考：[ops(2)：SoftMax算子的 CUDA 实现](https://zhuanlan.zhihu.com/p/695307283)

softmax稍微要注意的就是内存同步的问题，这里，你需要做一个网格级别的同步，而不能仅仅是block级别，否则拿不到全局的exp sum作为分母项。因此使用 __threadfence 这个网格及内存同步操作。不过效率我还没测过，实在要高效的话，可能得整成FA2那样的 1-pass + online softmax的实现。不过，如果是面试的话，就不要太为难自己了...，但是FA1/FA2的论文很经典，强烈建议多读几遍。

```c++
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
```

## sigmoid, sigmoid + vec4

```c++
// Sigmoid x: N, y: N y=1/(1+exp(-x))
// grid(N/128), block(K=128)
__global__ void sigmoid(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

// Sigmoid x: N, y: N y=1/(1+exp(-x)) Vec4
// grid(N/128), block(128/4)
__global__ void sigmoid_vec4(float *x, float *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
        reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
        reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
        reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
        FLOAT4(y[idx]) = reg_y;
    }
}
```

## relu, relu + vec4

```c++
// Relu x: N, y: N y=max(0,x)
// grid(N/128), block(K=128)
__global__ void relu(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = fmaxf(0.0f, x[idx]);
}

// Relu x: N, y: N y=max(0,x) Vec4
// grid(N/128/4), block(128/4)
__global__ void relu_vec4(float *x, float *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = fmaxf(0.0f, reg_x.x);
        reg_y.y = fmaxf(0.0f, reg_x.y);
        reg_y.z = fmaxf(0.0f, reg_x.z);
        reg_y.w = fmaxf(0.0f, reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}
```

## layer_norm, layer_norm + vec4

```c++
// Layer Norm: x: NxK(K=128<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template <const int NUM_THREADS = 128>
__global__ void layer_norm(float *x, float *y, float g, float b, int N, int K) {
    int tid = threadIdx.x; // 0..K-1
    int bid = blockIdx.x;  // 0..N-1
    int idx = bid * blockDim.x + threadIdx.x;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;                     // shared within block
    __shared__ float s_variance;                 // shared within block
    float value = (idx < N * K) ? x[idx] : 0.0f; // load once only
    float sum = block_reduce_sum<NUM_THREADS>(value);
    if (tid == 0) s_mean = sum / (float)K;
    // wait for s_mean in shared memory to be ready for all threads
    __syncthreads();
    float variance = (value - s_mean) * (value - s_mean);
    variance = block_reduce_sum<NUM_THREADS>(variance);
    if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();
    if (idx < N * K) y[idx] = ((value - s_mean) * s_variance) * g + b;
}

// Layer Norm Vec4: x: NxK(K=128<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template <const int NUM_THREADS = 128 / 4>
__global__ void layer_norm_vec4(float *x, float *y, float g, float b, int N, int K) {
    int tid = threadIdx.x; // 0..K-1
    int bid = blockIdx.x;  // 0..N-1
    int idx = (bid * blockDim.x + threadIdx.x) * 4;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;     // shared within block
    __shared__ float s_variance; // shared within block
    float4 reg_x = FLOAT4(x[idx]) float value = (idx < N * K) ? (reg_x.x + reg_x.y
                                                                 + reg_x.z + reg_x.w) :
                                                                0.0f;
    float sum = block_reduce_sum<NUM_THREADS>(value);
    if (tid == 0) s_mean = sum / (float)K;
    // wait for s_mean in shared memory to be ready for all threads
    __syncthreads();
    float4 reg_x_hat;
    reg_x_hat.x = reg_x.x - s_mean;
    reg_x_hat.y = reg_x.y - s_mean;
    reg_x_hat.z = reg_x.z - s_mean;
    reg_x_hat.w = reg_x.w - s_mean;
    float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y
                     + reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
    variance = block_reduce_sum<NUM_THREADS>(variance);
    if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();
    float4 reg_y;
    reg_y.x = reg_x_hat.x * s_variance * g + b;
    reg_y.y = reg_x_hat.y * s_variance * g + b;
    reg_y.z = reg_x_hat.z * s_variance * g + b;
    reg_y.w = reg_x_hat.w * s_variance * g + b;
    if (idx < N * K) FLOAT4(y[idx]) = reg_y;
}
```

## rms_norm, rms_norm + vec4

```c++
// RMS Norm: x: NxK(K=128<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template <const int NUM_THREADS = 128>
__global__ void rms_norm(float *x, float *y, float g, int N, int K) {
    int tid = threadIdx.x; // 0..K-1
    int bid = blockIdx.x;  // 0..N-1
    int idx = bid * blockDim.x + threadIdx.x;
    const float epsilon = 1e-5f;

    __shared__ float s_variance;                 // shared within block
    float value = (idx < N * K) ? x[idx] : 0.0f; // load once only
    float variance = value * value;
    variance = block_reduce_sum<NUM_THREADS>(variance);
    if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();
    if (idx < N * K) y[idx] = (value * s_variance) * g;
}

// RMS Norm Vec4: x: NxK(K=128<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template <const int NUM_THREADS = 128 / 4>
__global__ void rms_norm_vec4(float *x, float *y, float g, int N, int K) {
    int tid = threadIdx.x; // 0..K-1
    int bid = blockIdx.x;  // 0..N-1
    int idx = (bid * blockDim.x + threadIdx.x) * 4;
    const float epsilon = 1e-5f;

    __shared__ float s_variance; // shared within block
    float4 reg_x = FLOAT4(x[idx]);
    float variance = (idx < N * K) ? (reg_x.x * reg_x.x + reg_x.y * reg_x.y
                                      + reg_x.z * reg_x.z + reg_x.w * reg_x.w) :
                                     0.0f;
    variance = block_reduce_sum<NUM_THREADS>(variance);
    if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
    // wait for s_variance in shared memory to be ready for all threads
    __syncthreads();
    float4 reg_y;
    reg_y.x = reg_x.x * s_variance * g;
    reg_y.y = reg_x.y * s_variance * g;
    reg_y.z = reg_x.z * s_variance * g;
    reg_y.w = reg_x.w * s_variance * g;
    if (idx < N * K) FLOAT4(y[idx]) = reg_y;
}
```
