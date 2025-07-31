#include <stdio.h>


__global__ void parallel_reduction(int *input, int *output, int N)
    {
        __shared__ int temp[8];
        int tid = threadIdx.x;

        temp[tid] = input[tid];
        __syncthreads();

            // Step 2: Perform reduction in shared memory
        for (int stride = 1; stride < N; stride *= 2) {
            if (tid % (2 * stride) == 0 && (tid + stride) < N) {
                temp[tid] += temp[tid + stride];  // Change to max/min/product if needed
            }
            __syncthreads();  // Sync after each stride to avoid race conditions ( Simply that every thread is on same plcae)
        }

        // Step 3: Thread 0 writes the final result
        if (tid == 0) {
            output[0] = temp[0];
        }
        
    }


int main() {
    const int N = 8;
    int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8};  // Input array
    int h_output = 0;

    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block of N threads
    parallel_reduction<<<1, N>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sum = %d\n", h_output);  // Expected: 36

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
