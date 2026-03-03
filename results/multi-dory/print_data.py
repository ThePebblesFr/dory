import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# --- Load CSV file ---
df = pd.read_csv("results.csv", encoding="utf-8-sig")

# --- Matplotlib style setup (LaTeX-like look) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.edgecolor": "black"
})

# --- Define a consistent color palette ---
# You can adjust or expand if you have more L2 sizes
base_colors = ["lightcoral", "seagreen", "mediumvioletred", "royalblue"]

# --- Map each unique L2 size to a consistent color ---
l2_sizes = sorted(df["L2 size (kB)"].unique())
color_map = {l2: base_colors[i % len(base_colors)] for i, l2 in enumerate(l2_sizes)}

# --- Plot all models in one figure ---
models = df["model"].unique()

# Create one row of subplots (no shared Y-axis)
fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), sharey=False)

# Ensure iterable axes even if only one model
if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    sub = df[df["model"] == model]

    # Pivot and convert cycles → Mcycles
    pivot = sub.pivot(index="# cores", columns="L2 size (kB)", values="# cycles").sort_index(ascending=True)
    pivot = pivot / 1e6  # convert to millions of cycles

    # Grouped bar plot with consistent colors across models
    width = 0.6 / len(pivot.columns)
    x = range(len(pivot.index))

    for i, col in enumerate(pivot.columns):
        color = color_map[col]
        edge = to_rgba(color, 1.0)
        fill = to_rgba(color, 0.5)
        ax.bar(
            [xi + (i - (len(pivot.columns) - 1) / 2) * width for xi in x],
            pivot[col],
            width=width,
            edgecolor=edge,
            facecolor=fill,
            linewidth=1.5,
            label=f"{col} kB"
        )

    # Labels and layout
    ax.set_title(model)
    ax.set_xlabel("Number of cores")
    ax.set_ylabel("Total cycles (in Mcycles)")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)

    # Horizontal grid only
    ax.grid(axis='x', visible=False)

# --- Single centered legend above all subplots ---
handles, labels = [], []
for l2 in l2_sizes:
    color = color_map[l2]
    edge = to_rgba(color, 1.0)
    fill = to_rgba(color, 0.5)
    handles.append(plt.Rectangle((0, 0), 1, 1, edgecolor=edge, facecolor=fill, linewidth=1.5))
    labels.append(f"{l2} kB")

fig.legend(
    handles,
    labels,
    title="L2 sizes",
    loc="upper center",
    ncol=len(l2_sizes),    # all colors in one row
    frameon=True,
    edgecolor="black",
    bbox_to_anchor=(0.5, 0.98)  # position just above subplots
)

plt.tight_layout()#rect=[0, 0, 1, 0.88])
plt.subplots_adjust(top=0.80)
plt.savefig("all_models_cycles.png", dpi=300)
plt.close()

print("✅ Combined chart saved as 'all_models_cycles.png' with consistent colors and one legend.")

