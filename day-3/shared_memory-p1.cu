// this program can definately be done by single thread but we did it to see in future how memory effiecnit parallel computing can be done

#include <stdio.h>

__global__ void shared_memory_sum(int *input, int *output) {

	// Step 1: Declare shared memory
        __shared__ int temp[8];

	
	int tid = threadIdx.x;

	// Step 2: Copy each value from global to shared memory
    	temp[tid] = input[tid];


	// Step 3: Synchronize threads before using shared memory- i.e stopping all threads to wait till every thread copies to shared memory
   	 __syncthreads();

	// Step 4: Only thread 0 will compute the sum of all shared elements
    	if (tid == 0) {
        
	int sum = 0;
        for (int i = 0; i < 8; i++) {
            sum += temp[i];
        }

	// thread 0 is storing the final sum
        output[0] = sum;
    }
}

int main() {
    const int N = 8;
    int h_input[N] = {11, 2, 3, 4, 5, 6, 7, 18};
    int h_output;

    int *d_input, *d_output;
	
	// Allocate memory on device
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));

	// Copy input array to device- we are first coyping array from cpu array to pointer variable  for GPU
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch 1 block of 8 threads
    shared_memory_sum<<<1, N>>>(d_input, d_output);
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Output the result
    printf("Sum = %d\\n", h_output);  // Should print: Sum = 36

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);

return 0;
}


