#include <cuda_runtime.h>

#define TILE 32

__global__ void matrix_multiplication_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {

        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        // Load A tile
        if (row < M && aCol < N) {
            As[threadIdx.y][threadIdx.x] =
                A[row * N + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile
        if (bRow < N && col < K) {
            Bs[threadIdx.y][threadIdx.x] =
                B[bRow * K + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; i++) {
            sum += As[threadIdx.y][i] *
                   Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    dim3 threadsPerBlock(TILE, TILE);

    dim3 blocksPerGrid(
        (K + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    matrix_multiplication_kernel<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(A, B, C, M, N, K);

    cudaDeviceSynchronize();
}
