#include <stdio.h>

__global__ void mark_multiples_of_three(int *arr, int N) 
{
	 int tid = threadIdx.x;
	 if (tid<N)
		{
			if (tid%3==0)
			{ 
				arr[tid]=0;
			 }

			else
			 {
       				 arr[tid] = 1;  // odd
    			}
		}
}

int main() {
    int N = 16;
    int h_arr[16];
    int *d_arr;

    cudaMalloc(&d_arr, N * sizeof(int));

    mark_multiples_of_three<<<1, N>>>(d_arr, N);

    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("arr[%d] = %d\n", i, h_arr[i]);
    }

    cudaFree(d_arr);
    return 0;
}
				
 
