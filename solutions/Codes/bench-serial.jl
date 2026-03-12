using BenchmarkTools, Statistics

"""
    pairwise_distances(points::AbstractArray)

Calculate the pairwise distances between 3D points in an Nx3 array
"""
function pairwise_distances(points::AbstractArray{T}) where T
    @assert size(points)[2] == 3
    n = size(points)[1]
    distances = zeros(T, (n, n))
    for i in 1:n
        for j in 1:n
            dx = points[i, 1] - points[j, 1]
            dy = points[i, 2] - points[j, 2]
            dz = points[i, 3] - points[j, 3]
            distances[i, j] = sqrt(dx^2 + dy^2 + dz^2)
        end
    end
    return distances
end

# --- JIT Warm-up ---
println(stderr, "=== JIT Warm-up (100 points, includes compilation) ===")
warmup_points = rand(Float32, (100, 3))
warmup_t = @elapsed pairwise_distances(warmup_points)
println(stderr, "Warm-up done in \$(round(warmup_t, digits=4)) s  ← includes JIT compilation overhead")
println(stderr, "")

# --- Main Benchmark ---
n = 10_000
points = rand(Float32, (n, 3))
println(stderr, "=== Benchmarking Serial Original: n=10,000 ===")
result = @benchmark pairwise_distances($points) samples=5 evals=1 seconds=300

# Calculate outputs
med_s = median(result).time * 1e-9
mean_s = mean(result).time * 1e-9
min_s = minimum(result).time * 1e-9
max_s = maximum(result).time * 1e-9
std_s = std(result).time * 1e-9

alloc_mib = BenchmarkTools.memory(result) / (1024^2)
distance_calcs = n * n
throughput = distance_calcs / med_s

println(stderr, "")
println(stderr, "=== Serial Original Results ===")
println(stderr, "  Median time   : $(round(med_s, digits=2)) s")
println(stderr, "  Mean time     : $(round(mean_s, digits=2)) s")
println(stderr, "  Min time      : $(round(min_s, digits=2)) s")
println(stderr, "  Max time      : $(round(max_s, digits=2)) s")
println(stderr, "  Std deviation : $(round(std_s, digits=2)) s")
println(stderr, "  Memory alloc  : $(round(alloc_mib, digits=0)) MiB")
println(stderr, "  Distances computed : $(distance_calcs)  (10000 × 10000)")
println(stderr, "  Throughput    : $(throughput) distance-measures/second")

# Output to CSV string implicitly for the runner to collect via stdout redirects
println("1,serial_original,1,$med_s,$throughput")

#=
BENCHMARKING METHODOLOGY — SERIAL ORIGINAL
===========================================

WHY BenchmarkTools, NOT @time:
  @time runs the function once. On a cold start, this includes JIT compilation
  (often 10–100x slower than steady-state). BenchmarkTools runs multiple samples,
  discards outliers, and reports median/min/mean — giving a stable, compilation-free
  measurement. @btime reports only the minimum (useful but hides variance). @benchmark
  gives the full statistical picture needed for scientific reporting.

JIT COMPILATION AND WARM-UP:
  Julia compiles each function the first time it is called with a given type signature.
  For pairwise_distances(::Matrix{Float32}), the first call triggers LLVM compilation,
  which can take several seconds. The warm-up call on 100 points forces this compilation
  to happen before timing starts. BenchmarkTools also internally warms up, but an
  explicit warm-up makes the intent clear to any reader of the code.

SERIAL EFFICIENCY ANALYSIS:
  The algorithm is O(n²) in both time and space. For n=10,000:
    - Total distance computations: 100,000,000
    - Output matrix size: 10,000 × 10,000 × 4 bytes = 400 MB (Float32)
  The inner loop does 3 subtractions, 3 multiplications, 2 additions, and 1 sqrt per
  iteration. sqrt() is the dominant cost — roughly 10–20 ns on modern hardware.

OBVIOUS INEFFICIENCIES IN THE SERIAL VERSION:
  1. SYMMETRY IGNORED: dist(i,j) == dist(j,i) always. The code computes both, wasting
     ~50% of all work. Only the upper triangle (j >= i) needs to be computed.
  2. SELF-DISTANCE: distances[i,i] == 0 always, yet the loop still calls sqrt(0).
  3. COLUMN-MAJOR MISMATCH: Julia uses column-major storage. Accessing points[i,1],
     points[i,2], points[i,3] varies the row index (i) in the outer loop — this is
     a row-wise traversal, which causes cache misses in column-major layout. Transposing
     the points array to 3×N would make accesses contiguous.
  4. NO @inbounds: Julia performs bounds checking on every array access. @inbounds macro
     eliminates these checks inside hot loops, giving ~5–15% speedup.
  5. NO SIMD: The inner arithmetic is a perfect candidate for SIMD vectorization.
     Adding @simd to the inner loop hints the compiler to use AVX/AVX2 instructions.
=#
