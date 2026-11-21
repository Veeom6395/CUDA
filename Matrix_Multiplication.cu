//For Colab
%%writefile mul.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

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

        if (row < M && aCol < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < N && col < K)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * K + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; i++) {
            sum += As[threadIdx.y][i] *
                   Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K)
        C[row * K + col] = sum;
}

int main() {

    int M = 4;
    int N = 4;
    int K = 4;

    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    // Host memory
    float *h_A = new float[M * N];
    float *h_B = new float[N * K];
    float *h_C = new float[M * K];

    // Initialize matrices
    for (int i = 0; i < M * N; i++)
        h_A[i] = i + 1;

    for (int i = 0; i < N * K; i++)
        h_B[i] = i + 1;

    // Device memory
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy input matrices to GPU
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Configure CUDA execution
    dim3 threadsPerBlock(TILE, TILE);

    dim3 blocksPerGrid(
        (K + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    // Launch kernel
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, M, N, K
    );

    cudaDeviceSynchronize();

    // Copy result from GPU to CPU
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print Matrix A
    std::cout << "Matrix A:\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            std::cout << h_A[i * N + j] << " ";
        std::cout << "\n";
    }

    // Print Matrix B
    std::cout << "\nMatrix B:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++)
            std::cout << h_B[i * K + j] << " ";
        std::cout << "\n";
    }

    // Print Matrix C
    std::cout << "\nResult Matrix C (A x B):\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++)
            std::cout << h_C[i * K + j] << " ";
        std::cout << "\n";
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free CPU memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
