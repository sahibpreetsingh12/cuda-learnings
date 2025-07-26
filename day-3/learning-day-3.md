# CUDA Day 3 – Shared Memory and Thread Cooperation

---

## Shared Memory in CUDA

In CUDA, shared memory is an on-chip memory space that is visible to all threads within the same thread block. It is significantly faster than global memory and is often used for thread cooperation.

### How to Declare:
```cpp
__shared__ int temp[BLOCK_SIZE];
```

Each thread in the block can read and write to this array, making it ideal for collaborative tasks such as reductions, data reuse, and partial computations.

---

## Thread Cooperation

Threads in a block often need to work together on a problem — for example, summing an array or finding a maximum. Shared memory is typically used in these cases to allow intermediate results to be shared among threads.

To avoid race conditions, we use:

### Barrier Synchronization:
```cpp
__syncthreads();
```

This ensures that all threads reach a certain point before any of them proceed. It is critical when one thread reads data written by another.

---

## Understanding the Use of Void Kernels

A common question is:

**“If the return type of a kernel is `void`, how do we get values back from the GPU?”**

### Key Point:

Even though CUDA kernels return `void`, they **modify data via pointers** passed to them. 

For example:
```cpp
__global__ void add_two_numbers(int *A, int *B, int *output) {
    int tid = threadIdx.x;
    if (tid == 0) {
        output[0] = A[0] + B[0];
    }
}
```

The kernel modifies the contents of `output`, which points to device memory. After the kernel finishes, this result is copied back to host memory using:

```cpp
cudaMemcpy(&host_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
```

So although kernels don’t “return” in the traditional function sense, they write to memory that persists after execution.

---

## Example: Shared Memory Sum

In the example where we sum 8 values using shared memory:

1. Each thread copies one value to `temp[threadIdx.x]`
2. All threads synchronize using `__syncthreads()`
3. Thread 0 reads from `temp[]`, computes the total sum, and writes to `output[0]`

This shows a practical use of shared memory, synchronization, and cooperative work among threads.

---

## Recap of Concepts Covered

| Topic                   | Description |
|-------------------------|-------------|
| `__shared__` Memory     | Fast block-local memory shared by all threads in a block |
| `__syncthreads()`       | Synchronization barrier to prevent race conditions |
| Kernel with `void` return | Kernels write results to memory using pointer arguments |
| Shared memory example   | Demonstrated how threads cooperate using shared memory to compute a sum |
| Thread ID to Global Index | Used `blockIdx.x * blockDim.x + threadIdx.x` for global indexing |

---

This concludes the key learnings from Day 3.
