#include <stdio.h>

__global__ void square_if_even(int *arr, int N) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id < N) {

	if (global_id%2==0)
	{
         arr[global_id] = global_id * global_id;
	}
	else
	{
	arr[global_id]=-1;
	}
        
    }
}

int main() {
    int N = 16;
    int h_arr[N];
    int *d_arr;

    cudaMalloc(&d_arr, N * sizeof(int));

    square_if_even<<<4, 4>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("arr[%d] = %d\n", i, h_arr[i]);
    }

    cudaFree(d_arr);
    return 0;
}

