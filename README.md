# Julia Jet Reconstruction — Parallel Processing Improvements
## GSoC 2026 Evaluation Exercise

## Prerequisites
- Julia >= 1.9: https://julialang.org/downloads/
- Python >= 3.9 with matplotlib and pandas
- Bash shell

## Install Julia Packages (once)
```julia
using Pkg
Pkg.add(["BenchmarkTools", "Statistics"])
```

## Step-by-Step Reproduction

### Step 1 — Run all benchmarks
```bash
bash run_benchmarks.sh
```
This runs three stages (serial original → serial optimized → parallel 1/2/4/8/16 threads)
and writes all results to `results/benchmark_results.csv`.

### Step 2 — Generate the performance plot
```bash
pip install matplotlib pandas
python3 plot_results.py
# Output: results/performance_plot.png
```

### Step 3 — Verify thread count (optional)
```bash
julia --threads 8 -e 'println("Active threads: ", Threads.nthreads())'
# Should print: Active threads: 8
```

### Run stages individually
```bash
julia bench-serial.jl                        # Stage 1
julia bench-serial-optimized.jl              # Stage 2
julia --threads 4 parallel-euclid.jl 4       # Stage 3, 4 threads
```

## AI Usage Statement
<!-- TODO: Fill this in yourself — describe honestly which AI tools you used and for what -->
> _To be completed by the candidate._
