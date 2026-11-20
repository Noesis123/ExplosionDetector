import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DELIV_DIR = PROJECT_ROOT / "experiments" / "final_submission"
DELIV_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = Path("C:/Users/Aman Yazeen/.gemini/antigravity/brain/ad076906-e1fe-4046-9a40-92f5f2339702")

# Set consistent style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
})

# -------------------------------------------------------------
# 1. GENERATE 3-WAY COMPARATIVE CONFUSION MATRIX
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("NOESIS Explosion Detection — Benchmark Confusion Matrix Evolution (92 Files)", fontsize=14, fontweight="bold", y=1.02)

configs = [
    {
        "name": "Config C (Historical Baseline)",
        "cm": np.array([[49, 19], [2, 22]]),
        "prec": 53.66, "rec": 91.67, "f1": 0.6769, "fp": 19,
        "color": "Blues"
    },
    {
        "name": "Config D (Intermediate)",
        "cm": np.array([[57, 11], [2, 22]]),
        "prec": 66.67, "rec": 91.67, "f1": 0.7719, "fp": 11,
        "color": "YlOrBr"
    },
    {
        "name": "Config E (Final Production)",
        "cm": np.array([[64, 4], [2, 22]]),
        "prec": 84.62, "rec": 91.67, "f1": 0.8800, "fp": 4,
        "color": "Greens"
    }
]

labels = ["Non-Explosion", "Explosion"]

for idx, (ax, cfg) in enumerate(zip(axes, configs)):
    cm = cfg["cm"]
    im = ax.imshow(cm, interpolation="nearest", cmap=cfg["color"])
    ax.set_title(f"{cfg['name']}\nPrec: {cfg['prec']:.1f}% | Rec: {cfg['rec']:.1f}% | FP: {cfg['fp']}", fontsize=11, fontweight="bold", pad=10)
    
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    if idx == 0:
        ax.set_ylabel("True Label", fontweight="bold")
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            label_tag = ""
            if i == 0 and j == 0: label_tag = f"TN={val}\n({val/68*100:.1f}%)"
            elif i == 0 and j == 1: label_tag = f"FP={val}\n({val/68*100:.1f}%)"
            elif i == 1 and j == 0: label_tag = f"FN={val}\n({val/24*100:.1f}%)"
            elif i == 1 and j == 1: label_tag = f"TP={val}\n({val/24*100:.1f}%)"
            
            txt_color = "white" if val > thresh else "black"
            ax.text(j, i, label_tag, ha="center", va="center", color=txt_color, fontsize=11, fontweight="bold")

plt.tight_layout()
cm_path_local = DELIV_DIR / "confusion_matrix.png"
cm_path_artifact = ARTIFACTS_DIR / "confusion_matrix.png"
fig.savefig(cm_path_local, dpi=300, bbox_inches="tight")
fig.savefig(cm_path_artifact, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved confusion matrix figure to {cm_path_local} and {cm_path_artifact}")

# -------------------------------------------------------------
# 2. GENERATE PRECISION-RECALL & PERFORMANCE COMPARISON CHART
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("NOESIS Explosion Detector — Objective Optimization Performance", fontsize=14, fontweight="bold", y=1.00)

# Panel 1: Precision, Recall, F1 Percentage Comparison
cfg_names = ["Config C\n(Historical)", "Config D\n(Intermediate)", "Config E\n(Final Target)"]
x = np.arange(len(cfg_names))
width = 0.25

precs = [53.66, 66.67, 84.62]
recs = [91.67, 91.67, 91.67]
f1s = [67.69, 77.19, 88.00]

rects1 = ax1.bar(x - width, precs, width, label="Precision (%)", color="#1f77b4")
rects2 = ax1.bar(x, recs, width, label="Recall (%)", color="#2ca02c")
rects3 = ax1.bar(x + width, f1s, width, label="F1-Score (x100)", color="#ff7f0e")

# Add target threshold lines
ax1.axhline(80.0, color="#d62728", linestyle="--", linewidth=1.5, label="Target Precision (>80%)")
ax1.axhline(87.5, color="#2ca02c", linestyle=":", linewidth=1.5, label="Target Recall (≥87.5%)")

ax1.set_ylabel("Score (%)", fontweight="bold")
ax1.set_title("Metric Comparison Across Iterations", fontweight="bold", pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(cfg_names, fontweight="bold")
ax1.set_ylim(0, 105)
ax1.legend(loc="lower right", framealpha=0.9, fontsize=9)
ax1.grid(axis="y", linestyle="--", alpha=0.3)

# Value labels on bars
for rects in [rects1, rects2, rects3]:
    for r in rects:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%",
                     xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

# Panel 2: False Positive Reduction
fps = [19, 11, 4]
colors = ["#e74c3c", "#f39c12", "#27ae60"]
bars = ax2.bar(cfg_names, fps, width=0.45, color=colors, edgecolor="black", linewidth=1.2)
ax2.axhline(5, color="#d62728", linestyle="--", linewidth=1.5, label="Target Threshold (≤ 5 FPs)")

ax2.set_ylabel("False Positive Count (out of 68)", fontweight="bold")
ax2.set_title("False-Positive Reduction (Benchmark: 68 Negatives)", fontweight="bold", pad=10)
ax2.set_ylim(0, 24)
ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax2.grid(axis="y", linestyle="--", alpha=0.3)

for b in bars:
    h = b.get_height()
    red_pct = f"-{(1 - h/19)*100:.1f}%" if h < 19 else "Baseline"
    ax2.annotate(f"{h} files\n({red_pct})",
                 xy=(b.get_x() + b.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
pr_path_local = DELIV_DIR / "precision_recall_comparison.png"
pr_path_artifact = ARTIFACTS_DIR / "precision_recall_comparison.png"
fig.savefig(pr_path_local, dpi=300, bbox_inches="tight")
fig.savefig(pr_path_artifact, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved precision-recall figure to {pr_path_local} and {pr_path_artifact}")
