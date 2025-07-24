# 🚀 CUDA Learning Journal – Day 2

## ✅ Goals for Day 2

- Understand how to launch **multiple blocks and threads**
- Learn how to compute **global thread ID**
- Safely perform **conditional logic inside CUDA kernels**
- Apply **bounds checking** to avoid memory errors
- Write real GPU programs that perform **vector-style operations**

---

## 📘 Concepts Learned

### 1. Global Thread Index
To uniquely identify each thread across all blocks:
```cpp
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
```

---

### 2. Bounds Check
Always check whether the thread is within range before accessing memory:
```cpp
if (global_id < N) {
    // Safe to access arr[global_id]
}
```

---

### 3. Conditional Logic in Kernels
CUDA kernels support standard `if-else` logic:
```cpp
if (global_id % 2 == 0) {
    B[global_id] = A[global_id];
} else {
    B[global_id] = -1;
}
```

---

### 4. Multi-Block Kernel Launch
You launched kernels like:
```cpp
<<<2, 4>>>  // 2 blocks × 4 threads = 8 threads total
```
and understood how the GPU organizes execution.

---

### 5. Cuda MemCpy
# 📦 Understanding `cudaMemcpy` in CUDA

## 🔍 What is `cudaMemcpy`?

`cudaMemcpy` is a core CUDA API function used to **copy memory between host (CPU) and device (GPU)**.

Since CPU and GPU memory are **separate** (they don’t share RAM), you **must explicitly copy data** back and forth.

---

## 📘 Function Signature

```cpp
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
```

### Parameters:

| Argument  | Description |
|-----------|-------------|
| `dst`     | Destination pointer (where to copy TO) |
| `src`     | Source pointer (where to copy FROM) |
| `count`   | Number of **bytes** to copy |
| `kind`    | Direction of memory copy (see below) |

---

## 🔄 Copy Directions (`cudaMemcpyKind`)

| Value                        | Meaning                            |
|-----------------------------|------------------------------------|
| `cudaMemcpyHostToDevice`    | Copy from CPU (host) to GPU (device) |
| `cudaMemcpyDeviceToHost`    | Copy from GPU (device) to CPU (host) |
| `cudaMemcpyHostToHost`      | Copy between two CPU arrays |
| `cudaMemcpyDeviceToDevice`  | Copy between two GPU arrays |

---

## ✅ Examples

### 🔹 Host → Device (upload data to GPU)

```cpp
cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
```

### 🔹 Device → Host (retrieve result from GPU)

```cpp
cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
```

### 🔹 Device → Device

```cpp
cudaMemcpy(d_B, d_A, N * sizeof(int), cudaMemcpyDeviceToDevice);
```

---

## ⚠️ Common Mistakes

### ❌ Forgetting to multiply by `sizeof(...)`

```cpp
cudaMemcpy(h_arr, d_arr, N, cudaMemcpyDeviceToHost); // wrong!
```

✅ Fix:
```cpp
cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
```

---

### ❌ Getting the direction wrong

```cpp
cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyDeviceToHost); // wrong direction!
```

✅ Should be:
```cpp
cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
```

---

### ❌ Copying uninitialized or invalid pointers

Make sure you've done `cudaMalloc` for device memory **before** calling `cudaMemcpy`.

---

## 🧪 Real Example Flow

```cpp
int h_arr[8] = {1,2,3,4,5,6,7,8};
int *d_arr;

cudaMalloc(&d_arr, 8 * sizeof(int));                      // Allocate GPU memory
cudaMemcpy(d_arr, h_arr, 8 * sizeof(int), cudaMemcpyHostToDevice); // Copy to GPU

your_kernel<<<1, 8>>>(d_arr);                             // Run on GPU

cudaMemcpy(h_arr, d_arr, 8 * sizeof(int), cudaMemcpyDeviceToHost); // Copy back
```

---

## 🔁 Advanced Tip: Asynchronous Copy

You can use `cudaMemcpyAsync()` for non-blocking memory copies if working with CUDA streams.

---

## 🧠 Summary

| You Want To...                  | Use This Kind                   |
|--------------------------------|---------------------------------|
| Upload data to GPU             | `cudaMemcpyHostToDevice`        |
| Download results to CPU        | `cudaMemcpyDeviceToHost`        |
| Copy GPU memory to another GPU | `cudaMemcpyDeviceToDevice`      |

---

## 📅 Where You Used It (So Far)

- ✅ Day 1: Copying results like `arr[tid] = 7`
- ✅ Day 2: Vector operations, conditional writes
- You always use `cudaMemcpy` to validate GPU output on the CPU


--------------------------------

## ✅ Programs Written

### ✔️ 1. Global Thread ID Filler
- Each thread wrote its own global ID into an array.
- Confirmed correct indexing.

### ✔️ 2. Square of Global ID (Even Only)
- Threads stored their square if `global_id % 2 == 0`, else -1.
- Practiced arithmetic and conditional writes.

### ✔️ 3. Copy Only Even Indices from A to B
- Realistic vector filter example.
- Copied only even-indexed elements from A to B.
- Used host-to-device and device-to-host memory flows.

---

## 🧠 Debugging & Best Practices Learned

- Always use bounds check `if (global_id < N)`
- Understand operator precedence (e.g. `(N - 1) * sizeof(int)`)
- Use `cudaMemcpy` carefully to match directions:
  - `cudaMemcpyHostToDevice`
  - `cudaMemcpyDeviceToHost`
- Avoid printing from inside the kernel on LeetGPU (may show garbage)
- Confirm correctness via `cudaMemcpy` back to CPU

---

## 📅 What's Next – Day 3 Preview

- 🔁 Shared memory
- ➕ Reductions (summing inside a block)
- 🔐 Atomic operations
- ⚡ Performance tips (coalescing, occupancy)
- 👀 Implement a small softmax kernel
- 🎯 Start walking toward fused attention block logic

---

🧠 Logged by: Sahibpreet Singh
📅 Date: 2025-07-24
💻 Platform: LeetGPU (tested with safe memory copies and thread indexing)
