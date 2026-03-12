using BenchmarkTools, Statistics

"""
    pairwise_distances_parallel(points::AbstractArray{T}) where T

Thread-parallel pairwise distance computation using Threads.@threads.

Thread safety: @threads distributes the outer `i` loop across Julia threads.
Each thread writes exclusively to row `i` of the distances matrix (distances[i, :]).
No two threads ever write to the same memory location simultaneously — no mutex or
atomic operations required. This is safe because the output matrix is pre-allocated
and rows are non-overlapping across thread assignments.

Launch with: julia --threads N parallel-euclid.jl N
"""
function pairwise_distances_parallel(points::AbstractArray{T}) where T
    @assert size(points)[2] == 3
    n = size(points)[1]
    distances = zeros(T, (n, n))
    px = points[:, 1]
    py = points[:, 2]
    pz = points[:, 3]
    Threads.@threads for i in 1:n       # outer loop distributed across threads
        @inbounds for j in 1:n
            dx = px[i] - px[j]
            dy = py[i] - py[j]
            dz = pz[i] - pz[j]
            distances[i, j] = sqrt(dx*dx + dy*dy + dz*dz)
        end
    end
    return distances
end

# Read thread count from command line (set externally via julia --threads N)
thread_count = parse(Int, get(ARGS, 1, string(Threads.nthreads())))
@assert Threads.nthreads() == thread_count "Mismatch: Julia started with $(Threads.nthreads()) threads but ARGS says $thread_count. Re-run with julia --threads $thread_count"

n = 10_000
points = rand(Float32, (n, n == 10_000 ? 3 : 3))   # fixed dataset

result = @benchmark pairwise_distances_parallel($points) samples=5 evals=1 seconds=300

med_s      = median(result).time * 1e-9
throughput = (n * n) / med_s

# stdout: single CSV line for run_benchmarks.sh to capture
println("3,parallel,$thread_count,$(round(med_s, digits=6)),$(round(throughput, digits=2))")

# stderr: human-readable
println(stderr, "Threads=$thread_count | Median=$(round(med_s,digits=3))s | Throughput=$(round(throughput/1e6,digits=1))M ops/s")
