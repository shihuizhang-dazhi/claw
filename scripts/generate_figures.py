#!/usr/bin/env python3
"""Generate paper figures from actual experiment results."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUT = Path(__file__).resolve().parent.parent / "figures_paper"
OUT.mkdir(exist_ok=True)

COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3A7D44", "#6A4C93", "#D81159"]
GRAY = "#888888"

# Load real results
results_path = Path(__file__).resolve().parent.parent / "results/entropy_guard/entropy_guard_results.json"
with open(results_path) as f:
    data = json.load(f)


# =========================================================================
# Fig 2: Main results bar chart (from real data)
# =========================================================================
def fig_main_results():
    methods = ["DASGuard\n-only", "Entropy\n-only", "Entropy\nGuard\n(Ours)"]
    precision = [data["DASGuard-only"]["step"]["precision"],
                 data["Entropy-only"]["step"]["precision"],
                 data["EntropyGuard"]["step"]["precision"]]
    recall = [data["DASGuard-only"]["step"]["recall"],
              data["Entropy-only"]["step"]["recall"],
              data["EntropyGuard"]["step"]["recall"]]
    f1 = [data["DASGuard-only"]["step"]["f1"],
          data["Entropy-only"]["step"]["f1"],
          data["EntropyGuard"]["step"]["f1"]]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(methods))
    w = 0.25

    ax.bar(x - w, precision, w, label="Precision", color=COLORS[0], edgecolor="white", linewidth=0.3)
    ax.bar(x, recall, w, label="Recall", color=COLORS[1], edgecolor="white", linewidth=0.3)
    ax.bar(x + w, f1, w, label="F1", color=COLORS[2], edgecolor="white", linewidth=0.3)

    # Value labels
    for i in range(len(methods)):
        ax.text(i - w, precision[i] + 0.02, f"{precision[i]:.2f}", ha="center", fontsize=6.5)
        ax.text(i, recall[i] + 0.02, f"{recall[i]:.2f}", ha="center", fontsize=6.5)
        ax.text(i + w, f1[i] + 0.02, f"{f1[i]:.2f}", ha="center", fontsize=6.5, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=6.5)
    ax.legend(loc="lower left", framealpha=0.9, ncol=3, fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT / "main_results.pdf")
    plt.close()
    print("[OK] main_results.pdf")


# =========================================================================
# Fig 3: Alpha sensitivity (from real ablation data)
# =========================================================================
def fig_alpha_sensitivity():
    sens = data["sensitivity"]
    alphas = [float(k) for k in sens.keys()]
    f1_vals = [sens[k]["f1"] for k in sorted(sens.keys())]
    recall_vals = [sens[k]["recall"] for k in sorted(sens.keys())]

    fig, ax1 = plt.subplots(figsize=(3.5, 2.2))

    color1 = COLORS[0]
    color2 = COLORS[2]
    ax1.set_xlabel(r"$\alpha$ (Entropy Influence)")
    ax1.set_ylabel("F1 / Recall", color=color1)
    l1 = ax1.plot(alphas, f1_vals, "o-", color=color1, label="F1", markersize=4, linewidth=1.2)
    l2 = ax1.plot(alphas, recall_vals, "s--", color=color2, label="Recall", markersize=4, linewidth=1.2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.65, 1.05)

    # Plateau shade (α ≥ 0.3)
    ax1.axvspan(0.3, 1.0, alpha=0.08, color=COLORS[0])
    ax1.text(0.65, 0.69, "Plateau\n(α ≥ 0.3)", fontsize=7, ha="center", color=COLORS[0])

    lines = l1 + l2
    labels = [ll.get_label() for ll in lines]
    ax1.legend(lines, labels, loc="lower right", framealpha=0.9, fontsize=7)

    ax1.spines["top"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT / "alpha_sensitivity.pdf")
    plt.close()
    print("[OK] alpha_sensitivity.pdf")


# =========================================================================
# Fig 4: Entropy distribution — real validation results
# =========================================================================
def fig_entropy_validation():
    """Bar chart showing Self-BLEU (consistency) comparison from real sampling."""
    categories = ["Malicious", "Benign"]
    self_bleu = [0.297, 0.444]
    jaccard = [0.175, 0.246]
    colors_bar = [COLORS[1], COLORS[0]]

    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    x = np.arange(len(categories))
    w = 0.30

    bars1 = ax.bar(x - w/2, self_bleu, w, label="Self-BLEU", color=colors_bar, edgecolor="white", linewidth=0.3)
    bars2 = ax.bar(x + w/2, jaccard, w, label="Jaccard Sim.", color=[COLORS[3], COLORS[2]],
                   edgecolor="white", linewidth=0.3, alpha=0.7)

    for bar, val in zip(bars1, self_bleu):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", fontsize=7, fontweight="bold")
    for bar, val in zip(bars2, jaccard):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", fontsize=7)

    ax.set_ylabel("Consistency Score")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    ax.set_ylim(0, 0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation
    ax.annotate("Malicious responses\n33% more consistent\n→ Lower entropy",
                xy=(0, 0.297), xytext=(0.5, 0.55),
                arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=1.2),
                fontsize=7, color=COLORS[1], ha="center")

    plt.tight_layout()
    fig.savefig(OUT / "entropy_validation.pdf")
    plt.close()
    print("[OK] entropy_validation.pdf")


# =========================================================================
# Fig 5: Escape attack robustness (from real data)
# =========================================================================
def fig_escape_robustness():
    methods = ["DASGuard\n+EntropyEsc", "Entropy\n+EntropyEsc",
               "DASGuard\n+TextEsc", "Entropy\n+TextEsc"]
    f1_vals = [
        data["DASGuard+entropy_escape"]["step"]["f1"],
        data["Entropy+entropy_escape"]["step"]["f1"],
        data["DASGuard+text_escape"]["step"]["f1"],
        data["Entropy+text_escape"]["step"]["f1"],
    ]
    # EntropyGuard is same for both escape types
    eg_f1 = data["EntropyGuard+entropy_escape"]["step"]["f1"]

    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    colors_bar = [GRAY, COLORS[1], GRAY, COLORS[1]]

    x = np.arange(len(methods))
    bars = ax.bar(x, f1_vals, color=colors_bar, edgecolor="white", linewidth=0.3)
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}", ha="center", fontsize=7)

    # Add EG bar separately
    eg_bar = ax.bar(len(methods), eg_f1, color=COLORS[0], edgecolor="white", linewidth=0.3)
    ax.text(len(methods), eg_f1 + 0.01, f"{eg_f1:.2f}", ha="center", fontsize=7, fontweight="bold")

    xtick_labels = methods + ["Entropy\nGuard"]
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, fontsize=6)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Divider
    ax.axvline(1.5, color="black", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.text(0.5, 1.12, "Entropy-Escape", ha="center", fontsize=7, fontstyle="italic")
    ax.text(3.5, 1.12, "Text-Escape", ha="center", fontsize=7, fontstyle="italic")

    plt.tight_layout()
    fig.savefig(OUT / "escape_robustness.pdf")
    plt.close()
    print("[OK] escape_robustness.pdf")


# =========================================================================
# Fig 6: Ablation — base_threshold sensitivity
# =========================================================================
def fig_threshold_ablation():
    abl = data.get("ablation", {})
    thresh_keys = [k for k in abl.keys() if k.startswith("base_threshold_")]
    thresholds = []
    f1s = []
    recalls = []
    for k in sorted(thresh_keys):
        t = float(k.replace("base_threshold_", ""))
        thresholds.append(t)
        f1s.append(abl[k]["step"]["f1"])
        recalls.append(abl[k]["step"]["recall"])

    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.plot(thresholds, f1s, "o-", color=COLORS[0], label="F1", markersize=5, linewidth=1.2)
    ax.plot(thresholds, recalls, "s--", color=COLORS[2], label="Recall", markersize=5, linewidth=1.2)
    ax.set_xlabel(r"Base Threshold $\tau_{base}$")
    ax.set_ylabel("Score")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0.65, 1.05)

    # Mark optimal
    best_idx = np.argmax(f1s)
    ax.annotate(f"τ={thresholds[best_idx]:.2f}\nF1={f1s[best_idx]:.3f}",
                xy=(thresholds[best_idx], f1s[best_idx]),
                xytext=(thresholds[best_idx] + 0.1, f1s[best_idx] - 0.05),
                arrowprops=dict(arrowstyle="->", lw=1), fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT / "threshold_ablation.pdf")
    plt.close()
    print("[OK] threshold_ablation.pdf")


# =========================================================================
# Generate all
# =========================================================================
fig_main_results()
fig_alpha_sensitivity()
fig_entropy_validation()
fig_escape_robustness()
fig_threshold_ablation()
print(f"\nAll figures saved to {OUT.resolve()}/")
