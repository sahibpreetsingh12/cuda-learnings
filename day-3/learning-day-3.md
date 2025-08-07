# CUDA Day 3 – Shared Memory and Thread Cooperation



---

## 🔠 Shared Memory in CUDA

In CUDA, **shared memory** is an on-chip memory space accessible by **all threads in a block**. It's much faster than global memory and plays a critical role in **cooperative parallel computing**.

### ✅ Declaration Syntax:

```cpp
__shared__ int temp[BLOCK_SIZE];
```

* Scope: **Per block** (not visible outside the block)
* Lifetime: **For the duration of the block's execution**
* Speed: **\~100x faster than global memory**

Use cases: reduction, tiling, matrix multiplication, prefix scan, histogram.

---

## 🔌 Thread Cooperation via Shared Memory

Threads often work together to solve problems that require **intermediate storage and communication**.

### 📅 Example: Parallel Sum (Simplified Version)

Each thread copies one value from global to shared memory:

```cpp
temp[threadIdx.x] = input[threadIdx.x];
```

Then all threads synchronize:

```cpp
__syncthreads();
```

Finally, thread 0 sums up:

```cpp
if (threadIdx.x == 0) {
    int sum = 0;
    for (int i = 0; i < 8; i++) sum += temp[i];
    output[0] = sum;
}
```

This demonstrates **collaborative memory usage**, though only one thread performs the sum.

---

## ⚖️ Barrier Synchronization: `__syncthreads()`

To prevent **race conditions**, threads must wait for others to finish certain work before continuing.

```cpp
__syncthreads();
```

* Acts as a **barrier** across the block
* Required when **some threads read data written by others**
* Used in every round of parallel reduction

---

## ❓ How Do Kernels Return Results If They're `void`?

CUDA kernels don’t return values like regular functions.

### 🔎 The trick: use **pointers to device memory**

```cpp
__global__ void add_two_numbers(int *A, int *B, int *output) {
    int tid = threadIdx.x;
    if (tid == 0) output[0] = A[0] + B[0];
}
```

Then copy result from GPU to CPU:

```cpp
cudaMemcpy(&host_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
```

This is how you "return" values in CUDA: **by writing to memory**.

---

## 🔢 Parallel Reduction (Sum/Max/Min/Product)

We explored several reduction types using shared memory:

### 🏃‍♂️ Binary Tree Reduction:

```cpp
for (int stride = 1; stride < N; stride *= 2) {
    if (tid % (2 * stride) == 0 && (tid + stride) < N) {
        temp[tid] += temp[tid + stride]; // or max(), *, etc.
    }
    __syncthreads();
}
```

Each thread **participates in combining values**, reducing work in a tree-like manner.

Variants we practiced:

* `reduce_sum`
* `reduce_max` (used ternary operator for device compatibility)
* `reduce_min`
* `reduce_product`

---

## 📝 Mistakes & Fixes You Encountered

| Issue                                                                 | Fix                                          |
| --------------------------------------------------------------------- | -------------------------------------------- |
| Used `max()` but got compile error                                    | Used `std::max()` or ternary `a > b ? a : b` |
| Forgot to store final result to output                                | Added `if (tid == 0) output[0] = temp[0];`   |
| Used wrong variable name `temp[]` instead of declared `shared_data[]` | Matched names                                |

---

## ✅ Recap of Concepts Learned

| Concept                     | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| `__shared__` memory         | Fast, block-level memory for thread cooperation       |
| `__syncthreads()`           | Barrier to avoid race conditions                      |
| Cooperative memory loading  | Each thread loads 1 value into shared memory          |
| One-thread sum              | Simple shared-memory use by thread 0                  |
| Parallel reduction pattern  | Binary tree pattern to distribute work across threads |
| Kernels return via pointers | Use `cudaMemcpy()` after device write                 |
| `max()` fix                 | Use ternary or `std::max()` carefully                 |

---

This concludes the core lessons of **Day 3: Shared Memory & Thread Cooperation**. Great progress!
