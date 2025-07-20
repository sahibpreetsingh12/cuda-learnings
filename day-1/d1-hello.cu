#include <stdio.h>

// CUDA kernel: runs on GPU
__global__ void hello_from_gpu() {
    // threadIdx.x is always initialized by CUDA
   printf("Hello from GPU thread %u\n", threadIdx.x);
}

int main() {
    // Launch 1 block with 8 threads
    hello_from_gpu<<<1, 8>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}

