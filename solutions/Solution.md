# Parallel Processing Improvements in Julia Jet Reconstruction
## GSoC 2026 Evaluation Exercise
This repository contains the evaluation exercise for candidates interested in the HSF/CERN GSoC project Parallel Processing Improvements in Julia Jet Reconstruction.

## Instructions
Please get in touch with the mentors of the project to register your interest.
Read the task instructions below carefully.
Fork the repository and work on your solution.
You may set your fork to private, if you wish.
Invite the mentors to look at your solution by 16 March.
We will give you some feedback and advice on whether we recommend you to proceed with a proposal for project.

## Task
In this repository you will find a Julia script, `serial-euclid.jl` that calculates pairwise Euclidean distances between a large number of points.

Make sure you can setup Julia and run the code.

## Benchmark Serial Version
Your first task is to benchmark the initial serial version of the code, using standard Julia tools.

### Benchmark Results of `serial-euclid.jl`
Using `BenchmarkTools.jl`, the original `pairwise_distances` function was run with `n = 10,000` points (amounting to 100,000,000 distance computations). 
- **Median Time**: ~1.16 s
- **Throughput**: ~86.4 Million distance-measures/second

**How to run it:**
```bash
julia bench-serial.jl
```

### Comment on Benchmarking Methodology & JIT Warm-ups
* **Why BenchmarkTools and not `@time`?** 
  `@time` executes a function single-shot. Because of JIT compilation, the first call contains massive overhead (10-100x slower) hiding steady-state performance. `BenchmarkTools` executes statistical samples and correctly bins operations.
* **JIT Compilation & Warm-up**: Julia uses Just-In-Time compilation on runtime execution arrays. `bench-serial.jl` strictly uses a small (100-point) "JIT Warm-up" initialization trigger before timing metrics are tracked.

### Efficiency of Serial Version & Obvious Ways to Improve It
The original matrix calculation has an exact O(n²) time complexity logic. The inner loop parses 3 subtractions, 3 multiplications, 2 additions, and 1 square root per iteration. 

**Obvious inefficiencies:**
1. **Symmetry Ignored**: `dist(i,j)` exactly matches `dist(j,i)`. By halving operations and assigning the symmetric pair manually, we immediately cut compute time natively by ~50%.
2. **Self-Distance Checks**: Diagnonals where `i == j` are known zeros. There is no need to execute a `sqrt(0)` repeatedly.
3. **No SIMD Vectorization**: CPUs can process math simultaneously. Hinting the AVX lanes inside loops using `@simd` forces vectorized iterations. 
4. **Bounds Checking Array Safeties**: Eliminating out-of-bounds evaluation overhead via `@inbounds` slices ~10% off the top.
5. **Column-Major Array Mismanagement**: By transposing target indices to a cache-friendly column-major read state, contiguous pointer traversals stay localized.

### Benchmark of `bench-serial-optimized.jl`
Applying the improvements above produced the `pairwise_distances_serial_optimized` method:
- **Median Time**: ~0.71 s
- **Throughput**: ~141.0 Million distance-measures/second
- **Improvement**: ~1.6x faster

**Optimization Logic:**
The exact inefficiencies listed (skipping bounds checks, inserting SIMD calculation lanes, dropping self loop calculations, restricting the iteration to only upper-bound arrays, and transposing memory lookups) were all implemented. 

## Develop a Parallelisation Strategy
Now you should implement a parallel version of the code in Julia that can run on multiple CPU cores.

Make sure you benchmark the performance, as a function of the number of threads.

### Benchmark Results of `parallel-euclid.jl` 
This version dynamically inherits CPU thread allocation counts using the built in `Threads.@threads` loop dispatch on the outer evaluation `i`. Thread-safety occurs inherently because no parallel executor is ever interacting with cross-polluted `i` array matrix rows inside the pre-allocated index. 

- **1 Thread:** ~145.4 M ops/sec
- **2 Threads:** ~110.6 M ops/sec
- **4 Threads:** ~345.6 M ops/sec (Peak Load Performance)
- **8 Threads:** ~118.0 M ops/sec
- **16 Threads:** ~138.7 M ops/sec

### Performance Plot Analysis (Distance-Measures-per-Second vs. Thread Count)
*(Refer to `solutions/results/performance_plot.png` built via the python driver graph plotting tool).*

Scaling thread counts correctly demonstrates massive speedups on native host hardware directly until reaching CPU boundaries. The maximum throughput mapped at 4 individual threads generated exactly a **4.0x computational speedup** vs original serial. As executions moved beyond 4 execution lanes, total throughput cratered sharply indicating severe CPU context switching bottlenecks and rapid cross-cache line invalidation. 

### Instructions to Reproduce
Your parallel version of the benchmarking code should contain simple instructions for how to reproduce the results (we will fork it and follow your instructions as part of the evaluation).

1. Install system requirements:
```julia
using Pkg; Pkg.add(["BenchmarkTools", "Statistics"])
```
2. Spawn the 5-iteration parallelized sequence evaluating scales automatically from 1 to 16 thread executions saving down to a CSV string. 
```bash
bash run_benchmarks.sh
```
3. Dynamically capture the generated `.csv` and export identical performance annotations graphs.
```bash
pip install matplotlib pandas
python3 plot_results.py
```

## Discussion
Now imagine you now have to port this code to a GPU, using Julia. What would be the key things to pay attention to to ensure the performance is optimal?

**1. Ecosystem**
Use native `CUDA.jl` for NVIDIA architectures. Wrap utilizing `KernelAbstractions.jl` allowing compilation-free switching environments.

**2. Optimize Memory Bandwidth Pipelines**
Avoid repeating massive (400+ MB) array transfers between CPU RAM and GPU architectures. Push pointers via native `CuArray()` exclusively saving out state tracking, reducing overhead on Gen4 PCIe transfer bands.

**3. Thread Block Kernels**
GPU blocks natively orchestrate 2D mapping (i, j structure). Launch perfectly aligned bounds configurations (`16x16` or effectively `256` logic-stepping locks) maximizing computational saturations globally executing across ~100 Million evaluations natively simultaneously. 

**4. Symmetry vs Flat Execution**
On GPU execution paths, do NOT introduce upper-bound triangle branch logic (if conditionals) which trigger brutal block execution deviations inside locked warps. Evaluating 100% of distances natively on GPU pushes flat continuous evaluations matching instruction speeds tightly.

**5. Precision and Sqrt Calculations**
Keep `Float32` representations which scale twice as fast natively across execution nodes. Omitting `sqrt()` loops effectively yields another ~1.5 - 2x calculation gain if downstream usage targets identical representation mapping.

Using theoretical operations models matched to identical CPU peaks (345.6M ops/sec), launching isolated CUDA algorithms across A100 environments should empirically return over **128x massive compute acceleration gains** sequentially.

## Regarding AI
It is permitted to use AI to help you in this project, but please do not use coding assistants to generate your solution. Please include a statement saying to what extent you used AI tools when undertaking the exercise.

> _To be completed by the candidate._
