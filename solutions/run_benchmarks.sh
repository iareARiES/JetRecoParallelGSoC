#!/bin/bash
# run_benchmarks.sh
# Runs all three benchmark stages and collects results into one CSV.
#
# USAGE:
#   bash run_benchmarks.sh
#
# REQUIREMENTS:
#   julia >= 1.9 on PATH
#   BenchmarkTools and Statistics installed (see README)
#
# OUTPUT:
#   results/benchmark_results.csv

set -e
cd "$(dirname "$0")"
mkdir -p results

OUTPUT=results/benchmark_results.csv
echo "stage,variant,threads,median_s,throughput_per_sec" > "$OUTPUT"

# ── Stage 1: Serial original ──────────────────────────────────────────────────
echo "[1/3] Benchmarking serial original..."
julia bench-serial.jl >> "$OUTPUT"
echo "      Done."

# ── Stage 2: Serial optimized ────────────────────────────────────────────────
echo "[2/3] Benchmarking serial optimized..."
julia bench-serial-optimized.jl >> "$OUTPUT"
echo "      Done."

# ── Stage 3: Parallel across thread counts ───────────────────────────────────
echo "[3/3] Benchmarking parallel (threads: 1 2 4 8 16)..."
for T in 1 2 4 8 16; do
    echo "      Running with $T thread(s)..."
    julia --threads "$T" parallel-euclid.jl "$T" >> "$OUTPUT"
done
echo "      Done."

echo ""
echo "All benchmarks complete. Results saved to: $OUTPUT"
echo "Run: python3 plot_results.py  to generate the performance plot."
