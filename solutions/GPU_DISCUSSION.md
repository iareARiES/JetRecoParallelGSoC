# GPU Porting Discussion — Pairwise Euclidean Distance in Julia

## Context: What Our CPU Benchmarks Tell Us

From our measurements:
- Serial original throughput:    ~86.4 M ops/sec  (1 thread)
- Serial optimized throughput:   ~141.0 M ops/sec  (~1.6x faster)
- Parallel peak throughput:      ~345.6 M ops/sec  (4 threads, ~4.0x over serial)
- Peak CPU parallel speedup vs ideal: 4.0x out of theoretical 4x

Our system GPU (NVIDIA RTX 4050 Laptop, confirmed via `nvidia-smi`) offers ~9,000 GFLOPS
(FP32) vs ~1,000 GFLOPS for a high-end CPU. For an embarrassingly parallel workload like
ours, GPU is the natural next step.

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

This must fit in GPU VRAM. Our RTX 4050 Laptop has **6 GB GDDR6** — the 400 MB output
fits, but leaves limited headroom (~5.6 GB free). For larger n (e.g. 12,000+), VRAM
would become a constraint requiring tiled computation.

**Key rule:** Transfer data to GPU once, keep it there.
```julia
points_gpu    = CuArray(points_cpu)      # CPU → GPU: ~120 KB  (fast)
distances_gpu = CUDA.zeros(Float32, n, n) # allocate on GPU: 400 MB (no transfer)
# ... run kernel ...
distances_cpu = Array(distances_gpu)     # GPU → CPU: 400 MB  (this is your bottleneck)
```
The GPU→CPU copy of 400 MB over PCIe Gen4 x16 (~32 GB/s unidirectional) takes ~12.5 ms.
This is small relative to any non-trivial compute, but adds up over repeated calls.
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
    return Array(out)                     # copy back only if needed
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

## 7. Expected GPU Speedup vs Our CPU Results

Our parallel CPU peak: ~345.6 M ops/sec on 4 threads.

Our RTX 4050 Laptop delivers **~9 TFLOPS FP32** (2560 CUDA cores, Ada Lovelace).
Our kernel performs ~7 FLOPs per (i,j) pair (3 subtractions + 3 FMAs for the squared
terms + 1 sqrt).

**Compute-bound estimate:**
  9 × 10¹² / 7 ≈ 1,286 M ops/sec → ~3.7× over our CPU parallel peak.

**Memory-bound check:**
The kernel writes 4 bytes per (i,j) pair. With 192 GB/s GDDR6 bandwidth:
  192 × 10⁹ / 4 = 48,000 M pairs/sec.
Since 1,286 << 48,000, the kernel is **compute-bound** on this GPU — memory bandwidth
is not the bottleneck.

Accounting for realistic occupancy (~50–70% of peak due to register pressure, warp
scheduling, and sqrt latency), we expect an effective throughput of ~640–900 M ops/sec,
giving a **GPU speedup of roughly 2–3×** over our best CPU parallel result.

On a data-centre GPU (e.g. NVIDIA A100 with 19.5 TFLOPS FP32 and 2 TB/s HBM2e),
the same kernel would scale proportionally to ~50–100× over our CPU parallel peak.

> **Note:** These estimates use standard FP32 CUDA core throughput, not Tensor Core
> (FP16/TF32) figures which are much higher but not applicable to our kernel.
