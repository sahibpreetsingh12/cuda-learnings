#include <stdio.h>

// 🧠 Kernel definition
__global__ void fill_the_id(int *arr, int N) {


    // Compute global thread ID
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < N) {
        // Fill the array with the  global thread ID
        arr[global_id] = global_id;
    }
}

int main() {
    int N = 16;         // total elements
    int h_arr[N];        // host array
    int *d_arr;          // device pointer

    // Allocate memory on GPU
    cudaMalloc(&d_arr, N * sizeof(int));

    // Launch kernel with blocks × threads
    fill_the_id<<<4,4>>>(d_arr, N);

    // Copy result back to host
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();    

    // Print output
    for (int i = 0; i < N; i++) {
        printf("arr[%d] = %d\n", i, h_arr[i]);
    }

    // Free memory
    cudaFree(d_arr);

    return 0;
}
