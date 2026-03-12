"""
bench-gpu.jl
Benchmarks the pairwise distance kernel on the local GPU using CUDA.jl.
Compares against the best CPU result from bench-serial-optimized.jl and
parallel-euclid.jl.

Usage:
    julia bench-gpu.jl

Requirements:
    using Pkg; Pkg.add("CUDA")
"""

using CUDA, BenchmarkTools, Statistics

# ── GPU kernel ────────────────────────────────────────────────────────────────
# Each CUDA thread handles one (i,j) pair.
# 2D thread grid: blockIdx and threadIdx cover i and j independently.
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
    # Transfer only the 3 coordinate vectors (~120 KB total, negligible transfer time)
    px  = CuArray(points_cpu[:, 1])
    py  = CuArray(points_cpu[:, 2])
    pz  = CuArray(points_cpu[:, 3])
    # Allocate output on GPU — 400 MB, no CPU<->GPU transfer needed
    out = CUDA.zeros(Float32, n, n)
    # 16×16 = 256 threads/block — reliable default for Ada Lovelace
    threads = (16, 16)
    blocks  = (cld(n, 16), cld(n, 16))
    @cuda threads=threads blocks=blocks distance_kernel!(out, px, py, pz, n)
    CUDA.synchronize()
    return out   # keep on GPU — only move back if needed
end

# ── Device info ───────────────────────────────────────────────────────────────
dev = CUDA.device()
println(stderr, "")
println(stderr, "=== GPU Device ===")
println(stderr, "  Name       : $(CUDA.name(dev))")
println(stderr, "  VRAM       : $(round(CUDA.totalmem(dev) / 1024^3, digits=2)) GB")
println(stderr, "  Compute cap: $(CUDA.capability(dev))")
println(stderr, "")

# ── Setup ─────────────────────────────────────────────────────────────────────
n = 10_000
points_cpu = rand(Float32, (n, 3))

# ── JIT warm-up (first call compiles the kernel) ──────────────────────────────
println(stderr, "=== Warm-up (compiles CUDA kernel) ===")
warmup_pts = rand(Float32, (100, 3))
pairwise_distances_gpu(warmup_pts)
CUDA.synchronize()
println(stderr, "  Warm-up done.")
println(stderr, "")

# ── Benchmark ─────────────────────────────────────────────────────────────────
println(stderr, "=== Benchmarking GPU kernel: n=10,000 ===")
result = @benchmark begin
    pairwise_distances_gpu($points_cpu)
    CUDA.synchronize()
end samples=10 evals=1 seconds=300

# ── Results ───────────────────────────────────────────────────────────────────
med_s      = median(result).time * 1e-9
min_s      = minimum(result).time * 1e-9
mean_s     = mean(result).time    * 1e-9
throughput = Float64(n * n) / med_s

# Best CPU parallel result (from benchmark_results.csv: parallel, 4 threads)
cpu_parallel_peak = 3.4556703293e8   # 345.6 M ops/sec at 4 threads
speedup_vs_cpu    = throughput / cpu_parallel_peak

println(stderr, "")
println(stderr, "============================================================")
println(stderr, "  GPU BENCHMARK RESULTS")
println(stderr, "  GPU       : $(CUDA.name(dev))")
println(stderr, "  Dataset   : n=10,000 points (Float32)")
println(stderr, "============================================================")
println(stderr, "  Median time      : $(round(med_s * 1000, digits=3)) ms")
println(stderr, "  Min time         : $(round(min_s * 1000, digits=3)) ms")
println(stderr, "  Mean time        : $(round(mean_s * 1000, digits=3)) ms")
println(stderr, "  Throughput       : $(round(throughput/1e6, digits=1)) M ops/sec")
println(stderr, "  Speedup vs CPU   : $(round(speedup_vs_cpu, digits=2))x  (over 4-thread parallel peak)")
println(stderr, "============================================================")
println(stderr, "")

# stdout: CSV line for logging
println("4,gpu,1,$(round(med_s, digits=6)),$(round(throughput, digits=2))")
