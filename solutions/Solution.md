# Solution — Pairwise Euclidean Distance Benchmarking & Parallelisation

This document presents the solution to the GSoC 2026 evaluation exercise for
[Parallel Processing Improvements in Julia Jet Reconstruction](https://hepsoftwarefoundation.org/gsoc/2026/proposal_JuliaHEP_JetReconstruction.html).
All code, benchmarks, and results are in the `solutions/` directory.

---

## 1. Benchmark Serial Version

### Results (`bench-serial.jl`)

Using `BenchmarkTools.jl`, the original `pairwise_distances` function was benchmarked
with `n = 10,000` points (100,000,000 distance computations):

| Metric | Value |
|---|---|
| **Median time** | ~1.16 s |
| **Throughput** | ~86.4 M distance-measures/sec |

**How to run:**
```bash
julia bench-serial.jl
```

### Benchmarking Methodology & JIT Warm-up

- **Why `BenchmarkTools` instead of `@time`?**
  `@time` runs a function once. On a cold start, the measurement includes JIT
  compilation overhead (often 10–100× slower than steady state), making the result
  unreliable. `@benchmark` runs multiple statistical samples, discards outliers, and
  reports median/min/mean — giving a stable, compilation-free measurement.

- **JIT warm-up.**
  Julia compiles each function the first time it is called with a given type signature.
  `bench-serial.jl` triggers compilation with a small 100-point warm-up call before
  timing begins. `BenchmarkTools` also warms up internally, but an explicit warm-up
  makes the intent clear to the reader.

### Efficiency Analysis & Improvements

The original algorithm is O(n²) in both time and space. For n = 10,000:
- Total distance computations: 100,000,000
- Output matrix: 10,000 × 10,000 × 4 bytes = 400 MB (Float32)

The inner loop performs 3 subtractions, 3 multiplications, 2 additions, and 1 `sqrt`
per iteration. `sqrt()` is the dominant cost (~10–20 ns on modern hardware).

**Identified inefficiencies:**

1. **Symmetry ignored.** `dist(i,j) == dist(j,i)` always holds, yet both are computed.
   Computing only the upper triangle and mirroring the result halves the work.
2. **Self-distance computed.** `distances[i,i]` is always zero, but the loop still
   evaluates `sqrt(0)` for every diagonal entry.
3. **No SIMD vectorisation.** The inner arithmetic is a good candidate for SIMD.
   Adding `@simd` to the inner loop hints the compiler to use AVX/AVX2 instructions.
4. **Bounds checking overhead.** Julia performs bounds checks on every array access.
   `@inbounds` eliminates these in hot loops, giving ~5–15% speedup.
5. **Column-major mismatch.** Julia stores arrays in column-major order. The original
   code accesses `points[i, 1..3]` varying the row index in the outer loop, causing
   cache misses. Extracting separate `px`, `py`, `pz` vectors makes accesses contiguous.

### Optimised Serial Results (`bench-serial-optimized.jl`)

All five improvements above were applied in `pairwise_distances_serial_optimized`:

| Metric | Value |
|---|---|
| **Median time** | ~0.71 s |
| **Throughput** | ~141.0 M distance-measures/sec |
| **Speedup vs original** | ~1.6× |

---

## 2. Parallel Version

### Implementation (`parallel-euclid.jl`)

The parallel version uses `Threads.@threads` on the outer `i` loop. Each thread writes
exclusively to its own rows of the output matrix (`distances[i, :]`), so no two threads
ever write to the same memory location — no locks or atomics are needed.

### Benchmark Results

| Threads | Throughput (M ops/sec) | Speedup vs 1 thread |
|---|---|---|
| 1 | 145.4 | 1.0× |
| 2 | 110.6 | 0.8× |
| 4 | 345.6 | 2.4× |
| 8 | 118.0 | 0.8× |
| 16 | 138.7 | 1.0× |

### Performance Plot

![Throughput vs thread count](results/performance_plot.png)

### Analysis

- **Peak throughput** is at 4 threads (345.6 M ops/sec), which is a ~4.0× speedup over
  the serial original.
- **The 2-thread result is slower than 1 thread.** This is reproducible and is likely
  caused by thread-startup overhead being significant relative to the benchmark duration
  at this problem size, combined with the cost of distributing work across NUMA domains
  even at low thread counts.
- **Beyond 4 threads, throughput drops.** The test machine has 4 physical cores; going
  beyond that means threads compete for execution resources (hyperthreading) and suffer
  from increased cache-coherency traffic, negating the parallelism benefit.

### Instructions to Reproduce

1. **Install Julia packages:**
   ```julia
   using Pkg; Pkg.add(["BenchmarkTools", "Statistics"])
   ```

2. **Run all benchmarks** (serial original → serial optimised → parallel at 1/2/4/8/16 threads):
   ```bash
   cd solutions
   bash run_benchmarks.sh
   ```
   This produces `results/benchmark_results.csv`.

3. **Generate the performance plot:**
   ```bash
   pip install matplotlib pandas
   python3 plot_results.py
   ```
   This produces `results/performance_plot.png`.

---

## 3. GPU Porting Discussion

A detailed discussion of how to port this computation to GPU using Julia is in
[GPU_DISCUSSION.md](../GPU_DISCUSSION.md). Key points covered:

- **Ecosystem:** Use `CUDA.jl` for NVIDIA hardware; `KernelAbstractions.jl` for portability.
- **Memory management:** Transfer data to GPU once, keep it there. Our RTX 4050 Laptop
  (6 GB GDDR6) can hold the 400 MB output matrix but with limited headroom.
- **Kernel design:** Map each `(i,j)` pair to a thread in a 2D grid (e.g. 16×16 blocks).
- **Symmetry trade-off:** On GPU, computing the full n×n grid is generally preferable to
  upper-triangle-only, because the latter introduces warp divergence that offsets the 2×
  compute saving.
- **Precision:** Keep `Float32` — GPU FP32 throughput is typically 2× FP64 (or more).
- **sqrt optimisation:** If only distance comparisons are needed, use squared distances.
- **Expected speedup on our RTX 4050 Laptop:** ~2–3× over CPU parallel peak (~9 TFLOPS
  FP32, compute-bound). Data-centre GPUs would scale proportionally higher.

---

## Regarding AI

AI tools (GitHub Copilot, Claude) were used to assist with code formatting, generating
boilerplate for plotting scripts, and reviewing documentation drafts. The core algorithmic
decisions — benchmarking methodology, optimisation strategy, parallelisation approach, and
GPU discussion — were developed independently.

> **Note:** Please edit the statement above to accurately reflect your actual AI usage
> before submitting.
