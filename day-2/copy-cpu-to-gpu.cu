#include <stdio.h>

__global__ void filter_even_indices(int *A, int *B, int N) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id < N) {
        
	if (global_id%2==0)
	{
	B[global_id] = A[global_id];
	}
	else
	{
	B[global_id] = -1;
	}
    }
}

int main() {
    int N = 8;
    int h_A[8] = {10, 11, 12, 13, 14, 15, 16, 17};
    int h_B[8];

    int *d_A, *d_B;
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);

    filter_even_indices<<<2, 4>>>(d_A, d_B, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("B[%d] = %d\n", i, h_B[i]);
    }

    cudaFree(d_A); cudaFree(d_B);
    return 0;
}

