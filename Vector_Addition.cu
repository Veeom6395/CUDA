//For colab
%%writefile vector_addition.cu

#include <iostream>
#include <cuda_runtime.h>

__global__ void vector_add(
    const float *A,
    const float *B,
    float *C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 5;
    const int SIZE = N * sizeof(float);

    float h_A[N] = {1, 2, 3, 4, 5};
    float h_B[N] = {10, 20, 30, 40, 50};
    float h_C[N];

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, SIZE);
    cudaMalloc(&d_B, SIZE);
    cudaMalloc(&d_C, SIZE);

    cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice);

    int THREADS = 256;
    int blocks = (N + THREADS - 1) / THREADS;

    vector_add<<<blocks, THREADS>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost);

    std::cout << "Vector Addition: ";

    for (int i = 0; i < N; i++) {
        std::cout << h_C[i] << " ";
    }

    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
