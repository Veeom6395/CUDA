#include <cuda_runtime.h>

__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers
extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    constexpr int THREADS = 256;

    int blocks = (N + THREADS - 1) / THREADS;

    vector_add<<<blocks, THREADS>>>(A, B, C, N);

    cudaDeviceSynchronize();
}
