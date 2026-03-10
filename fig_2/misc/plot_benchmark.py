"""Plot benchmark results for CellFlow timing & memory profiling.

Standalone script — only loads the CSV results from the benchmark runs.
No benchmarking is performed here.

Usage
-----
  python plot_benchmark.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "A100 80GB": "#2171b5",
    "V100 32GB": "#cb181d",
}

OUT_DIR = "/lustre/groups/ml01/workspace/ot_perturbation/results/profiling"

# ---------------------------------------------------------------------------
# Load pre-computed results
# ---------------------------------------------------------------------------

df_a100 = pd.read_csv(f"{OUT_DIR}/benchmark_a100.csv")
df_v100 = pd.read_csv(f"{OUT_DIR}/benchmark_v100.csv")

df_a100["gpu_label"] = "A100 80GB"
df_v100["gpu_label"] = "V100 32GB"

df = pd.concat([df_a100, df_v100], ignore_index=True)

train = df[df["phase"] == "training"].copy()
inf = df[df["phase"].str.startswith("inference")].copy()
inf["n_conditions"] = inf["n_conditions"].astype(int)

# ---------------------------------------------------------------------------
# Figure: 2x2 layout
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# ---- Panel A: Training time vs batch size ----
ax = axes[0, 0]
for gpu_label, grp in train.groupby("gpu_label"):
    grp_sorted = grp.sort_values("batch_size")
    ax.plot(
        grp_sorted["batch_size"],
        grp_sorted["per_iter_ms"],
        "o-",
        color=COLORS[gpu_label],
        label=gpu_label,
        markersize=6,
        linewidth=2,
    )
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xlabel("Batch size")
ax.set_ylabel("Time per iteration (ms)")
ax.set_title("A  Training speed")
ax.legend(frameon=False)

# ---- Panel B: Training GPU memory vs batch size ----
ax = axes[0, 1]
for gpu_label, grp in train.groupby("gpu_label"):
    grp_sorted = grp.sort_values("batch_size")
    ax.plot(
        grp_sorted["batch_size"],
        grp_sorted["gpu_peak_mb"] / 1024,
        "s-",
        color=COLORS[gpu_label],
        label=gpu_label,
        markersize=6,
        linewidth=2,
    )
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xlabel("Batch size")
ax.set_ylabel("Peak GPU memory (GB)")
ax.set_title("B  Training GPU memory")
ax.legend(frameon=False)

# ---- Panel C: Inference time per 10k cells/condition ----
ax = axes[1, 0]
inf_single = inf[inf["n_conditions"] == 1].copy()
for gpu_label, grp in inf_single.groupby("gpu_label"):
    grp_sorted = grp.sort_values("n_cells")
    ax.plot(
        grp_sorted["n_cells"] / 1000,
        grp_sorted["per_10k_cells_per_cond_s"],
        "o-",
        color=COLORS[gpu_label],
        label=gpu_label,
        markersize=6,
        linewidth=2,
    )
ax.set_xlabel("Number of cells (thousands)")
ax.set_ylabel("Time per 10k cells (s)")
ax.set_title("C  Inference speed (1 condition)")
ax.legend(frameon=False)

# ---- Panel D: Inference total time, 1 vs 10 conditions ----
ax = axes[1, 1]
bar_width = 0.18
n_cells_vals = sorted(inf["n_cells"].unique())
x = np.arange(len(n_cells_vals))

for i, (gpu_label, color) in enumerate(COLORS.items()):
    for j, n_cond in enumerate([1, 10]):
        subset = inf[(inf["gpu_label"] == gpu_label) & (inf["n_conditions"] == n_cond)]
        subset = subset.set_index("n_cells").loc[n_cells_vals]
        offset = (i * 2 + j - 1.5) * bar_width
        label = f"{gpu_label}, {n_cond} cond"
        ax.bar(
            x + offset,
            subset["total_elapsed_s"],
            bar_width * 0.9,
            color=color,
            alpha=0.9 if n_cond == 1 else 0.5,
            label=label,
        )

ax.set_xticks(x)
ax.set_xticklabels([f"{int(v/1000)}k" for v in n_cells_vals])
ax.set_xlabel("Number of cells")
ax.set_ylabel("Wall-clock time (s)")
ax.set_title("D  Inference wall-clock time")
ax.set_yscale("log")
ax.legend(frameon=False, fontsize=8, ncol=2)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/benchmark_plots.pdf")
fig.savefig(f"{OUT_DIR}/benchmark_plots.png")
print(f"Saved to {OUT_DIR}/benchmark_plots.pdf")
print(f"Saved to {OUT_DIR}/benchmark_plots.png")
plt.show()
