#include <stdio.h>

__global__ void fill_sevens(int *arr, int N) {
    int tid = threadIdx.x;
    if (tid < N)
        arr[tid] = 7;
}

int main() {
    int N = 8;
    int h_arr[8];           // CPU array
    int *d_arr;             // GPU array

    cudaMalloc(&d_arr, N * sizeof(int));

    fill_sevens<<<1, N>>>(d_arr, N);
    cudaMemcpy(h_arr, d_arr, (N-1) * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        printf("arr[%d] = %d\n", i, h_arr[i]);

    cudaFree(d_arr);
    return 0;
}

