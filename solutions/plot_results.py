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

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)
df = pd.read_csv("results/benchmark_results.csv")

# ── Extract CPU data ──────────────────────────────────────────────────────────
orig      = df[df["variant"] == "serial_original"]["throughput_per_sec"].values[0]
optimized = df[df["variant"] == "serial_optimized"]["throughput_per_sec"].values[0]
par       = df[df["variant"] == "parallel"].sort_values("threads")
threads   = par["threads"].values
throughput = par["throughput_per_sec"].values

# Ideal linear scaling from 1-thread parallel baseline
baseline_parallel = throughput[0]
ideal = baseline_parallel * threads

# ── Extract GPU data (if present) ─────────────────────────────────────────────
gpu_rows = df[df["variant"].str.startswith("gpu_", na=False)]
has_gpu  = len(gpu_rows) > 0

# ── Figure setup ─────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Pairwise Euclidean Distance — Performance Analysis\n(n=10,000 points, Float32)",
             fontsize=13, fontweight="bold")

# ── LEFT: Stage comparison bar chart (CPU + GPU) ─────────────────────────────
ax1 = axes[0]
variants  = ["Serial\nOriginal", "Serial\nOptimized", f"Parallel\n(4 threads)"]
values    = [orig, optimized, throughput[threads == 4][0]]
colours   = ["#d9534f", "#f0ad4e", "#5cb85c"]

# Add GPU bars if data exists
if has_gpu:
    for _, row in gpu_rows.iterrows():
        name = row["variant"].replace("gpu_", "").replace("_", " ").title()
        # Shorten long names
        if "tesla" in row["variant"].lower() or "t4" in row["variant"].lower():
            name = "GPU\nTesla T4"
            colours.append("#9b59b6")
        elif "4050" in row["variant"]:
            name = "GPU\nRTX 4050"
            colours.append("#3498db")
        else:
            name = f"GPU\n{name}"
            colours.append("#1abc9c")
        variants.append(name)
        values.append(row["throughput_per_sec"])

# Use log scale since GPU throughput is orders of magnitude larger
max_val = max(values)
use_log = max_val / min(values) > 50

bars = ax1.bar(variants, [v / 1e6 for v in values], color=colours, edgecolor="white", width=0.55)
for bar, val in zip(bars, values):
    label = f"{val/1e6:.1f}M" if val < 1e9 else f"{val/1e9:.1f}B"
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
             label, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax1.set_ylabel("Throughput (Million ops/sec)", fontsize=10)
ax1.set_title("Stage Comparison (CPU vs GPU)", fontsize=11)
if use_log:
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# ── RIGHT: Parallel scaling + GPU reference lines ────────────────────────────
ax2 = axes[1]
ax2.plot(threads, throughput / 1e6, "o-", color="#2E75B6", linewidth=2.5,
         markersize=8, label="CPU Parallel (measured)", zorder=3)
ax2.plot(threads, ideal / 1e6, "--", color="gray", linewidth=1.5,
         label="Ideal linear scaling", zorder=2)
ax2.fill_between(threads, ideal / 1e6, throughput / 1e6, alpha=0.08, color="gray",
                 label="Efficiency gap")

# Annotate each CPU point with speedup vs 1-thread
for t, tp in zip(threads, throughput):
    sp = tp / throughput[0]
    ax2.annotate(f"{sp:.1f}x", xy=(t, tp/1e6), xytext=(t, tp/1e6 + 5),
                 ha="center", fontsize=8)

# Add GPU horizontal reference lines
gpu_colours = {"tesla": "#9b59b6", "t4": "#9b59b6", "4050": "#3498db"}
if has_gpu:
    for _, row in gpu_rows.iterrows():
        vname = row["variant"].lower()
        tp_gpu = row["throughput_per_sec"]
        # Choose colour
        col = "#1abc9c"
        for key, c in gpu_colours.items():
            if key in vname:
                col = c
                break
        # Determine label
        if "tesla" in vname or "t4" in vname:
            lbl = f"Tesla T4: {tp_gpu/1e6:,.0f}M ops/s"
        elif "4050" in vname:
            lbl = f"RTX 4050: {tp_gpu/1e6:,.0f}M ops/s"
        else:
            lbl = f"GPU: {tp_gpu/1e6:,.0f}M ops/s"
        ax2.axhline(y=tp_gpu / 1e6, color=col, linestyle=":", linewidth=1.8,
                    label=lbl, zorder=1)

ax2.set_xscale("log", base=2)
ax2.set_xticks(threads)
ax2.set_xticklabels([str(t) for t in threads])
ax2.set_xlabel("Number of Threads", fontsize=11)
ax2.set_ylabel("Throughput (Million ops/sec)", fontsize=10)
ax2.set_title("CPU Parallel Scaling + GPU Reference", fontsize=11)
if has_gpu:
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig("results/performance_plot.png", dpi=150, bbox_inches="tight")
print("Plot saved to results/performance_plot.png")
