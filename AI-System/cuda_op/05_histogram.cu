#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <algorithm>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define WARP_SIZE 32

void cpuHistogram(int *a, int *y, const int N) {
    for (int i = 0; i < N; i++) {
        y[a[i]]++;
    }
}

float testError(
    void (*gpuHistogram)(int *, int *, const int),
    dim3 gridDim, dim3 blockDim, const int N) {
    size_t size_a = N * sizeof(int);
    size_t size_y = N * sizeof(int);

    int *h_a, *h_y, *d_a, *d_y, *h_d_y;
    h_a = (int *)malloc(size_a);
    h_y = (int *)malloc(size_y);
    memset(h_y, 0, size_y);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_y, size_y);
    h_d_y = (int *)malloc(size_y);

    srand(time(0));
    for (int i = 0; i < N; i++)
        h_a[i] = rand() % N;
    cudaMemset(d_y, 0, size_y);

    cpuHistogram(h_a, h_y, N);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    gpuHistogram<<<gridDim, blockDim>>>(d_a, d_y, N);
    cudaMemcpy(h_d_y, d_y, size_y, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < N; i++) {
        float this_error = abs(h_d_y[i] - h_y[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = std::max(max_error, this_error);
    }

    free(h_a);
    free(h_y);
    cudaFree(d_a);
    cudaFree(d_y);
    free(h_d_y);

    return max_error;
}

float testPerformance(
    void (*gpuHistogram)(int *, int *, const int),
    dim3 gridDim, dim3 blockDim, const int N, const int repeat) {
    size_t size_a = N * sizeof(int);
    size_t size_y = N * sizeof(int);

    int *d_a, *d_y;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_y, size_y);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuHistogram<<<gridDim, blockDim>>>(d_a, d_y, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_y);

    return sec;
}

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

int main(void) {
    {
        printf("\nKernal = histogram\n");
        const int N = 128;
        dim3 blockDim(128);
        dim3 gridDim(N / 128);
        float max_error = testError(histogram, gridDim, blockDim, N);
        printf("Max Error = %f\n", max_error);
    }

    {
        printf("\nKernal = histogram_vec4\n");
        const int N = 128;
        dim3 blockDim(128 / 4);
        dim3 gridDim(N / 128);
        float max_error = testError(histogram_vec4, gridDim, blockDim, N);
        printf("Max Error = %f\n", max_error);
    }

    return 0;
}