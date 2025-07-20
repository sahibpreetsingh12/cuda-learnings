# CUDA Learning Journal - Day 1: Basics

**Date**: Jul-19-2025  
**Status**: ✅ Completed basic CUDA concepts

---

## Q&A: Day 1 CUDA Fundamentals

### What is a CUDA kernel?
**A**: A function that runs on the GPU, marked with `__global__`. It's launched from CPU using `<<<blocks, threads>>>` syntax.

```cpp
__global__ void my_kernel() {
    // runs on GPU
}

// Launch from CPU:
my_kernel<<<2, 4>>>(); // 2 blocks, 4 threads each
```

### What is a thread in CUDA?
**A**: The basic execution unit on GPU. Each thread runs the same kernel code but can work on different data. Threads are identified by `threadIdx.x` which always starts from 0 in each block.

### What is a block?
**A**: A group of threads that can cooperate. Blocks are identified by `blockIdx.x`. Launch syntax is `<<<num_blocks, threads_per_block>>>`.

Example: `kernel<<<3, 8>>>()` creates 3 blocks with 8 threads each = 24 total threads.

### What is a grid?
**A**: A collection of blocks. When you launch a kernel with `<<<blocks, threads>>>`, you're creating a grid with that many blocks. The grid is the top-level organization of your parallel computation.

### How many grids can a GPU have?
**A**: Only **1 grid per kernel launch**. Each time you call `kernel<<<blocks, threads>>>()`, you create one grid that contains all those blocks.

### How many blocks can a grid have?
**A**: Depends on GPU, but typically:
- **Maximum**: ~65,535 blocks per dimension
- **Practical**: Limited by GPU resources (memory, SM capacity)
- **Common range**: Hundreds to thousands of blocks

### How many threads can a block have?
**A**: 
- **Maximum**: 1024 threads per block (on most modern GPUs)
- **Common sizes**: 32, 64, 128, 256, 512, 1024
- **Rule**: Must be multiple of 32 (warp size) for efficiency

### How do I write if-else in CUDA?
**A**: Same as regular C++:
```cpp
if (threadIdx.x % 2 == 0) {
    arr[threadIdx.x] = 0;
} else {
    arr[threadIdx.x] = 1;
}

// Or ternary:
arr[tid] = (tid % 3 == 0) ? 0 : 1;
```

### What is `int *arr` vs `int *d_arr`?
**A**: 
- `int *arr`: Pointer to CPU memory (host)
- `int *d_arr`: Pointer to GPU memory (device)
- The `d_` prefix is a naming convention for device pointers

### What does `cudaMalloc` do and what's its syntax?
**A**: Allocates memory on the GPU.

**Syntax**: `cudaMalloc(void** devPtr, size_t size)`
- `devPtr`: **Address** of your device pointer (why we use `&d_array`)
- `size`: Number of bytes to allocate

```cpp
int *d_array;  // Declare device pointer
cudaMalloc(&d_array, N * sizeof(int));
//          ^^^^^^^   ^^^^^^^^^^^^^^
//          address   bytes to allocate
//          of ptr    (N integers × 4 bytes each)
```

**Why `&d_array`?** Because `cudaMalloc` needs to **modify** your pointer variable, so it needs the address where your pointer is stored.

### What does `cudaMemcpy` do and what's its syntax?
**A**: Copies memory between CPU and GPU.

**Syntax**: `cudaMemcpy(void* dst, void* src, size_t count, cudaMemcpyKind kind)`
- `dst`: Destination pointer (where to copy TO)
- `src`: Source pointer (where to copy FROM) 
- `count`: Number of bytes to copy
- `kind`: Direction of copy

```cpp
// Copy CPU → GPU (before kernel)
cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);
//         ^^^^^^^  ^^^^^^^  ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
//         TO       FROM     bytes           direction
//         (GPU)    (CPU)    to copy         

// Copy GPU → CPU (after kernel)  
cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);
//         ^^^^^^^  ^^^^^^^  ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^
//         TO       FROM     bytes           direction
//         (CPU)    (GPU)    to copy
```

**Memory aid**: Think "copy FROM source TO destination" - first parameter is always destination.

### What's the basic CUDA program structure?
**A**:
```cpp
// 1. Allocate host memory
int *h_array = (int*)malloc(N * sizeof(int));

// 2. Allocate device memory
int *d_array;
cudaMalloc(&d_array, N * sizeof(int));

// 3. Copy data to GPU
cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

// 4. Launch kernel
my_kernel<<<blocks, threads>>>(d_array);

// 5. Copy results back
cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);

// 6. Cleanup
free(h_array);
cudaFree(d_array);
```

### What programs did I write on Day 1?
**A**:
1. **Hello kernel**: Print thread IDs
2. **Fill with 7s**: Each thread sets `arr[tid] = 7`
3. **Store thread ID**: Each thread sets `arr[tid] = tid`
4. **Mark multiples of 3**: Use if-else to mark elements
5. **Memory bug exercise**: Learned about garbage values in uninitialized memory

### What was the operator precedence bug I found?
**A**:
```cpp
// WRONG - copies (N - 4) bytes due to precedence
cudaMemcpy(h_arr, d_arr, N - 1 * sizeof(int), direction);

// CORRECT  
cudaMemcpy(h_arr, d_arr, (N - 1) * sizeof(int), direction);
```

### What printf format should I use for thread IDs?
**A**: Use `%u` because `threadIdx.x` is unsigned int:
```cpp
printf("Thread %u\n", threadIdx.x);
```

### What happens if I access memory I didn't copy?
**A**: You get garbage values. Always match your copy size with your access pattern.

### Do I understand single-block CUDA programming?
**A**: ✅ Yes. I can:
- Write kernels with `__global__`
- Launch with `<<<blocks, threads>>>`
- Allocate GPU memory with `cudaMalloc`
- Copy memory with `cudaMemcpy`  
- Use `threadIdx.x` for indexing
- Handle simple if-else logic
- Debug basic memory issues

### What's next for Day 2?
**A**: 
- Multi-block programming
- Global thread ID calculation: `blockIdx.x * blockDim.x + threadIdx.x`
- Vector addition with multiple blocks
- Handling arrays larger than 1024 elements

---

## Quick Reference

```cpp
// Kernel syntax
__global__ void kernel(int *data) { }

// Launch
kernel<<<blocks, threads>>>(args);

// Memory
cudaMalloc(&device_ptr, size);
cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
cudaFree(device_ptr);

// Threading
int tid = threadIdx.x;  // 0, 1, 2, ... within each block
```

**Day 1 Complete** ✅
