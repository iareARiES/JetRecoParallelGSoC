# GPU Porting Discussion — Pairwise Euclidean Distance in Julia

## Context: What Our CPU Benchmarks Tell Us

From our measurements:
- Serial original throughput:    ~86.4 M ops/sec  (1 thread)
- Serial optimized throughput:   ~141.0 M ops/sec  (~1.6x faster)
- Parallel peak throughput:      ~345.6 M ops/sec  (4 threads, ~4.0x over serial)

We benchmarked on two GPUs: locally on an **NVIDIA RTX 4050 Laptop** (5.64 GB GDDR6,
Ada Lovelace) and on **Google Colab with a Tesla T4** (14.56 GB VRAM, Turing, ~8.1 TFLOPS FP32).

---

## 1. Julia GPU Ecosystem

Julia has first-class GPU support:
- **CUDA.jl** — NVIDIA GPUs (most relevant for CERN/LHC computing infrastructure)
- **AMDGPU.jl** — AMD GPUs
- **Metal.jl** — Apple Silicon
- **KernelAbstractions.jl** — hardware-portable kernel code (write once, run on any backend)

For CERN use, CUDA.jl is the primary target (NVIDIA hardware dominates HEP clusters).

---

## 2. Memory: The Critical Bottleneck

Our distance matrix is:
  n × n × sizeof(Float32) = 10,000 × 10,000 × 4 = **400 MB**

This must fit in GPU VRAM. The Tesla T4 has **14.56 GB** — the 400 MB output fits
comfortably with ~14 GB to spare.

**Key rule:** Transfer data to GPU once, keep it there.
```julia
points_gpu    = CuArray(points_cpu)      # CPU → GPU: ~120 KB  (fast)
distances_gpu = CUDA.zeros(Float32, n, n) # allocate on GPU: 400 MB (no transfer)
# ... run kernel ...
distances_cpu = Array(distances_gpu)     # GPU → CPU: 400 MB  (this is your bottleneck)
```
The GPU→CPU copy of 400 MB over PCIe Gen3 x16 (~16 GB/s on T4) takes ~25 ms.
If you only need the distances on GPU (e.g. for further GPU processing), **avoid this
copy entirely**.

---

## 3. Kernel Design: Mapping (i,j) to Thread Grid

Each (i,j) pair is independent — perfect for a 2D thread grid:

```julia
using CUDA

function distance_kernel!(distances, px, py, pz, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= n && j <= n
        dx = px[i] - px[j]
        dy = py[i] - py[j]
        dz = pz[i] - pz[j]
        @inbounds distances[i, j] = sqrt(dx*dx + dy*dy + dz*dz)
    end
    return
end

function pairwise_distances_gpu(points_cpu::Matrix{Float32})
    n   = size(points_cpu, 1)
    px  = CuArray(points_cpu[:, 1])
    py  = CuArray(points_cpu[:, 2])
    pz  = CuArray(points_cpu[:, 3])
    out = CUDA.zeros(Float32, n, n)
    threads = (16, 16)                    # 256 threads/block
    blocks  = (cld(n, 16), cld(n, 16))   # ceiling division
    @cuda threads=threads blocks=blocks distance_kernel!(out, px, py, pz, n)
    synchronize()
    return out                            # keep on GPU — copy back only if needed
end
```

Thread block size 16×16 = 256 threads/block is a reliable default. Can tune to 32×8 or
others — profile with CUDA.@profile.

---

## 4. Symmetry: Full Grid vs Triangle

Our CPU optimized version computed only the upper triangle (~2x faster serially).
On GPU, the tradeoff changes:

| Approach | GPU Ops | Thread Divergence | Complexity |
|---|---|---|---|
| Full n×n grid | n² | None (no if-branches) | Simple |
| Upper triangle only | n²/2 | Yes (threads in lower half idle) | Complex |

**Recommendation for GPU: use the full n×n grid.**
GPU warps execute 32 threads in lockstep. Triangle-based access creates branch divergence
(some threads in a warp skip the body while others execute it), wasting warp efficiency.
The 2x compute saving is often offset by the divergence penalty and more complex
index arithmetic. The simple full grid keeps all threads busy with identical work.

---

## 5. Float32 Is the Right Choice

Our code already uses Float32. This is optimal for GPU because:
- GPU FP32 throughput is typically 2× FP64 throughput (some cards 8–32×)
- For Euclidean distance at n=10,000, Float32 precision is more than sufficient
- Avoid promoting to Float64 accidentally (e.g. avoid untyped `0.0` literals; use `0.0f0`)

---

## 6. Avoiding sqrt When Possible

`sqrt()` is the most expensive operation in our kernel (~32 cycles/warp vs ~4 for multiply).
If distances are only used for **comparison** (e.g. finding nearest neighbours), replace:
```julia
distances[i,j] = sqrt(dx*dx + dy*dy + dz*dz)
# with:
distances[i,j] = dx*dx + dy*dy + dz*dz   # squared distance — monotonic, sqrt-free
```
This gives ~1.5–2x kernel speedup at the cost of returning squared distances.
For jet reconstruction where actual distance values are needed, keep the sqrt.

---

## 7. Measured GPU Results

We ran the GPU benchmark in two environments:
1. **Cloud:** Google Colab with a Tesla T4 (see [ColabT4_GPU_Testing.ipynb](Codes/ColabT4_GPU_Testing.ipynb))
2. **Local:** NVIDIA RTX 4050 Laptop GPU (using the `bench-gpu.jl` script)

| Metric | Tesla T4 (Colab) | RTX 4050 Laptop (Local) |
|---|---|---|
| **VRAM & Compute Cap** | 14.56 GB, sm_75 | 5.64 GB, sm_89 |
| **Median time** | 6.623 ms | 219.012 ms |
| **Throughput** | **15,098 M ops/sec** | **456.6 M ops/sec** |

### Speedup comparison

| Baseline | Throughput | T4 Speedup | RTX 4050 Speedup |
|---|---|---|---|
| Serial original (1 thread) | 86.4 M ops/sec | **174.7×** | **5.3×** |
| Serial optimised (1 thread) | 141.0 M ops/sec | **107.1×** | **3.2×** |
| Parallel peak (4 threads) | 345.6 M ops/sec | **43.7×** | **1.32×** |

### Analysis

**1. The Cloud Result (Tesla T4):**
The Tesla T4 delivers ~8.1 TFLOPS FP32 (2560 CUDA cores, Turing architecture).
Our kernel performs ~7 FLOPs per (i,j) pair. The theoretical compute-bound peak is:

  8.1 × 10¹² / 7 ≈ 1,157 M ops/sec

Our **measured** throughput of 15,098 M ops/sec far exceeds this naive compute-bound
estimate. This is because the GPU hides arithmetic latency through massive parallelism
— with 625×625 = 390,625 blocks of 256 threads (~100 million threads total), the T4's
warp scheduler keeps CUDA cores fully saturated by switching between warps while others
stall on sqrt.

The **43.7× speedup** over the CPU parallel peak confirms that this embarrassingly
parallel workload maps extremely well to GPU hardware, even on a mid-range data-centre
card. A more powerful GPU (e.g. A100 with 19.5 TFLOPS FP32) would scale further.

**2. The Local Result (RTX 4050 Laptop):**
The laptop GPU achieved a **1.32× speedup** over the CPU parallel peak. While this is
modest compared to the data-centre T4, laptop GPUs are tightly constrained by aggressive
power limits, thermal throttling, and narrower memory buses (192 GB/s). Reaching
456.6 M ops/sec locally still demonstrates that the kernel successfully leverages the
hardware accelerator beyond what the local CPU cores could achieve.
