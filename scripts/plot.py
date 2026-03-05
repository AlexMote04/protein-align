import matplotlib.pyplot as plt
import numpy as np

# Database sizes (X-axis)
db_sizes = [100, 1000, 5000, 10000, 50000]

# Execution Times (in ms)
naive_cpu_time = [7.62, 130.36, 1196.89, 3925.58, 50555.74]
parasail_time = [1.27, 15.14, 123.61, 376.52, 4393.60]
gpu_time = [0.81, 1.26, 9.46, 44.07, 2562.13]

# GCUPS
naive_cpu_gcups = [0.0312, 0.0324, 0.0326, 0.0324, 0.0320]
parasail_gcups = [0.1880, 0.2740, 0.3200, 0.3370, 0.3668]
gpu_gcups = [0.2882, 3.3470, 4.1799, 2.9326, 0.6303]

# --- Plotting Logic ---
# Create a wide figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ==========================================
# CHART 1: Execution Time (Log-Log Scale)
# ==========================================
ax1.plot(
    db_sizes,
    naive_cpu_time,
    marker="o",
    linestyle="-",
    linewidth=2,
    label="Naive CPU",
    color="#e74c3c",
)
ax1.plot(
    db_sizes,
    parasail_time,
    marker="s",
    linestyle="-",
    linewidth=2,
    label="Parasail (SIMD)",
    color="#3498db",
)
ax1.plot(
    db_sizes,
    gpu_time,
    marker="^",
    linestyle="-",
    linewidth=2,
    label="Optimized GPU (CUDA)",
    color="#2ecc71",
)

ax1.set_title(
    "Execution Time (Lower is Better)", fontsize=14, fontweight="bold", pad=15
)
ax1.set_xlabel("Database Size (Number of Sequences)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Execution Time (ms) - Log Scale", fontsize=12, fontweight="bold")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend(fontsize=12, loc="upper left")

# ==========================================
# CHART 2: GCUPS Throughput (Semi-Log Scale)
# ==========================================
ax2.plot(
    db_sizes,
    naive_cpu_gcups,
    marker="o",
    linestyle="-",
    linewidth=2,
    label="Naive CPU",
    color="#e74c3c",
)
ax2.plot(
    db_sizes,
    parasail_gcups,
    marker="s",
    linestyle="-",
    linewidth=2,
    label="Parasail (SIMD)",
    color="#3498db",
)
ax2.plot(
    db_sizes,
    gpu_gcups,
    marker="^",
    linestyle="-",
    linewidth=2,
    label="Optimized GPU (CUDA)",
    color="#2ecc71",
)

ax2.set_title(
    "Throughput in GCUPS (Higher is Better)", fontsize=14, fontweight="bold", pad=15
)
ax2.set_xlabel("Database Size (Number of Sequences)", fontsize=12, fontweight="bold")
ax2.set_ylabel("GCUPS (Giga Cell Updates / Sec)", fontsize=12, fontweight="bold")

ax2.set_xscale("log")
# Note: Keeping the Y-axis linear for GCUPS is usually best because it visually
# emphasizes the massive tower of performance the GPU provides over the CPU.
# If the CPU lines are completely crushed to the floor and you want to see them,
# you can uncomment the next line:
# ax2.set_yscale("log")

ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend(fontsize=12, loc="upper left")

# ==========================================
# Final Formatting & Save
# ==========================================
plt.suptitle(
    "Needleman-Wunsch Performance: CPU vs SIMD vs GPU",
    fontsize=18,
    fontweight="bold",
    y=1.05,
)
plt.tight_layout()

plt.savefig("benchmark_combined_plot.png", dpi=300, bbox_inches="tight")
print("Plot successfully saved as 'benchmark_combined_plot.png'")
