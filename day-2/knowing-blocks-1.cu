#include <stdio.h>

__global__ void print_global_ids() {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Global thread ID = %d\n", global_id);
}

int main() {
    // 2 blocks, 4 threads per block → 8 threads total
    print_global_ids<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}

