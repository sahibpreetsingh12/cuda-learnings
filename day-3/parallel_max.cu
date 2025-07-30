#include <stdio.h>


__global__ void reduce_max(int *input, int *output, int N)
    {
        __shared__ int shared_data[1024];
        int tid = threadIdx.x;

        if (tid < N)
            shared_data[tid] = input[tid];
        __syncthreads();

        for (int stride =1; stride < N; stride *= 2)
        {
            if (tid % (2 * stride) == 0 && (tid + stride) < N)
            {
                shared_data[tid] = std::max(shared_data[tid], shared_data[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0)
            output[0] = shared_data[0];
    }

int main() {
    const int N = 8;
    int *d_input, *d_output;
    int h_input[N] = {1,2,3,4,5,6,89,900}, h_output;
    cudaMalloc((void**)&d_input, N * sizeof(int));  
    cudaMalloc((void**)&d_output, sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);  
    reduce_max<<<1, N>>>(d_input, d_output, N);
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Max value: %d\n", h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}