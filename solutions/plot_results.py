"""
plot_results.py
Reads results/benchmark_results.csv and produces results/performance_plot.png

Usage:
    pip install matplotlib pandas
    python3 plot_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

os.makedirs("results", exist_ok=True)
df = pd.read_csv("results/benchmark_results.csv")

# ── Extract data ──────────────────────────────────────────────────────────────
orig      = df[df["variant"] == "serial_original"]["throughput_per_sec"].values[0]
optimized = df[df["variant"] == "serial_optimized"]["throughput_per_sec"].values[0]
par       = df[df["variant"] == "parallel"].sort_values("threads")
threads   = par["threads"].values
throughput = par["throughput_per_sec"].values

# Ideal linear scaling from 1-thread parallel baseline
baseline_parallel = throughput[0]
ideal = baseline_parallel * threads

# ── Figure setup ─────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Pairwise Euclidean Distance — Performance Analysis\n(n=10,000 points, Float32)",
             fontsize=13, fontweight="bold")

# ── LEFT: Serial comparison bar chart ────────────────────────────────────────
ax1 = axes[0]
variants  = ["Serial\nOriginal", "Serial\nOptimized", f"Parallel\n({threads[-1]} threads)"]
values    = [orig, optimized, throughput[-1]]
colours   = ["#d9534f", "#f0ad4e", "#5cb85c"]
bars = ax1.bar(variants, [v / 1e6 for v in values], color=colours, edgecolor="white", width=0.5)
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val/1e6:.1f}M", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_ylabel("Throughput (Million ops/sec)", fontsize=11)
ax1.set_title("Stage Comparison", fontsize=11)
speedup_opt = optimized / orig
speedup_par = throughput[-1] / orig
ax1.annotate(f"+{speedup_opt:.1f}x", xy=(1, optimized/1e6), xytext=(1.3, optimized/1e6 * 0.85),
             fontsize=9, color="#f0ad4e", arrowprops=dict(arrowstyle="->", color="#f0ad4e"))
ax1.annotate(f"+{speedup_par:.1f}x", xy=(2, throughput[-1]/1e6), xytext=(2.15, throughput[-1]/1e6 * 0.85),
             fontsize=9, color="#5cb85c", arrowprops=dict(arrowstyle="->", color="#5cb85c"))

# ── RIGHT: Parallel scaling line chart ───────────────────────────────────────
ax2 = axes[1]
ax2.plot(threads, throughput / 1e6, "o-", color="#2E75B6", linewidth=2.5,
         markersize=8, label="Measured", zorder=3)
ax2.plot(threads, ideal / 1e6, "--", color="gray", linewidth=1.5,
         label="Ideal linear scaling", zorder=2)
ax2.fill_between(threads, ideal / 1e6, throughput / 1e6, alpha=0.08, color="gray",
                 label="Efficiency gap")
# Annotate each point with speedup vs 1-thread
for t, tp in zip(threads, throughput):
    sp = tp / throughput[0]
    ax2.annotate(f"{sp:.1f}x", xy=(t, tp/1e6), xytext=(t, tp/1e6 + 1.5),
                 ha="center", fontsize=8)
ax2.set_xscale("log", base=2)
ax2.set_xticks(threads)
ax2.set_xticklabels([str(t) for t in threads])
ax2.set_xlabel("Number of Threads", fontsize=11)
ax2.set_ylabel("Throughput (Million ops/sec)", fontsize=11)
ax2.set_title("Parallel Scaling", fontsize=11)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig("results/performance_plot.png", dpi=150, bbox_inches="tight")
print("Plot saved to results/performance_plot.png")
