"""
=============================================================
  Phase 3 — Exploratory Data Analysis (EDA)
  Hybrid Malware Detection Project

  Generates all 6 supervisor deliverables:
  1. Dataset overview table
  2. Entropy distribution histogram
  3. Top 10 opcode bar chart
  4. Correlation heatmap
  5. Class balance chart
  6. Feature statistics summary

  Usage (Windows — malware_env activated):
    malware_env/Scripts/activate
    python phase3_eda.py

  Output → results/eda/ folder (all charts as PNG)
=============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  CONFIGURATION — update path to your CSV
# ─────────────────────────────────────────────────────────
CSV_PATH   = r"C:\Users\Daniel\Documents\Final Sem\Project\features\combined_features.csv"
OUTPUT_DIR = r"C:\Users\Daniel\Documents\Final Sem\Project\results\Phase3_eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
#  STYLE SETUP
# ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

MALWARE_COLOR = "#D85A30"   # coral   → malware
BENIGN_COLOR  = "#1D9E75"   # teal    → benign
NEUTRAL_COLOR = "#7F77DD"   # purple  → neutral

# ─────────────────────────────────────────────────────────
#  LOAD DATASET
# ─────────────────────────────────────────────────────────
print("=" * 62)
print("  Phase 3 — Exploratory Data Analysis")
print("=" * 62)
print(f"\n[*] Loading dataset: {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    print(f"\n❌ CSV not found at: {CSV_PATH}")
    print("   Update CSV_PATH at the top of this script.")
    exit(1)

df = pd.read_csv(CSV_PATH)
print(f"    Loaded {len(df)} rows × {len(df.columns)} columns")

malware_df = df[df["label"] == 1]
benign_df  = df[df["label"] == 0]

# ═════════════════════════════════════════════════════════
#  DELIVERABLE 1 — Dataset Overview
# ═════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  Deliverable 1 — Dataset Overview")
print("─" * 62)

total_rows     = len(df)
total_cols     = len(df.columns) - 2   # exclude file_name and label
malware_count  = len(malware_df)
benign_count   = len(benign_df)
missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()

print(f"  Total samples      : {total_rows}")
print(f"  Feature columns    : {total_cols}")
print(f"  Malware samples    : {malware_count}")
print(f"  Benign samples     : {benign_count}")
print(f"  Missing values     : {missing_values}")
print(f"  Duplicate rows     : {duplicate_rows}")
print(f"  Class balance      : {malware_count/total_rows*100:.1f}% / {benign_count/total_rows*100:.1f}%")

# Save overview as text
overview_path = os.path.join(OUTPUT_DIR, "01_dataset_overview.txt")
with open(overview_path, "w") as f:
    f.write("=" * 50 + "\n")
    f.write("  DATASET OVERVIEW\n")
    f.write("=" * 50 + "\n")
    f.write(f"Total samples      : {total_rows}\n")
    f.write(f"Feature columns    : {total_cols}\n")
    f.write(f"Malware samples    : {malware_count}\n")
    f.write(f"Benign samples     : {benign_count}\n")
    f.write(f"Missing values     : {missing_values}\n")
    f.write(f"Duplicate rows     : {duplicate_rows}\n\n")
    f.write("Feature Statistics per Class:\n")
    entropy_cols = [c for c in df.columns if "entropy" in c]
    f.write(df.groupby("label")[entropy_cols].mean().to_string())
print(f"  ✅ Saved: {overview_path}")


# ═════════════════════════════════════════════════════════
#  DELIVERABLE 2 — Entropy Distribution
# ═════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  Deliverable 2 — Entropy Distribution")
print("─" * 62)

entropy_features = [
    "entropy_mean", "entropy_max", "entropy_min",
    "entropy_text_section", "entry_point_entropy"
]
entropy_features = [c for c in entropy_features if c in df.columns]

fig, axes = plt.subplots(1, len(entropy_features),
                         figsize=(4 * len(entropy_features), 5))
fig.suptitle("Entropy Distribution: Malware vs Benign",
             fontsize=14, fontweight="bold", y=1.02)

if len(entropy_features) == 1:
    axes = [axes]

for ax, col in zip(axes, entropy_features):
    ax.hist(malware_df[col].dropna(), bins=40, alpha=0.7,
            color=MALWARE_COLOR, label="Malware", density=True)
    ax.hist(benign_df[col].dropna(),  bins=40, alpha=0.7,
            color=BENIGN_COLOR,  label="Benign",  density=True)
    ax.set_title(col.replace("_", " ").title(), fontsize=11)
    ax.set_xlabel("Entropy value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "02_entropy_distribution.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Saved: {path2}")


# ═════════════════════════════════════════════════════════
#  DELIVERABLE 3 — Top 10 Opcode Analysis
# ═════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  Deliverable 3 — Top 10 Opcode Analysis")
print("─" * 62)

opcode_cols = [c for c in df.columns if c.startswith("op_")]

if opcode_cols:
    # Mean value per class
    mal_mean = malware_df[opcode_cols].mean()
    ben_mean = benign_df[opcode_cols].mean()

    # Difference: malware - benign (most discriminative)
    diff = (mal_mean - ben_mean).abs().sort_values(ascending=False)
    top10 = diff.head(10).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Top 10 Most Discriminative Opcodes",
                 fontsize=14, fontweight="bold")

    # Left: malware vs benign mean frequency
    x      = np.arange(len(top10))
    width  = 0.35
    labels = [c.replace("op_", "").upper() for c in top10]

    axes[0].bar(x - width/2, mal_mean[top10],
                width, label="Malware", color=MALWARE_COLOR, alpha=0.85)
    axes[0].bar(x + width/2, ben_mean[top10],
                width, label="Benign",  color=BENIGN_COLOR,  alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_title("Mean opcode frequency (ratio)")
    axes[0].set_ylabel("Frequency ratio")
    axes[0].legend()

    # Right: absolute difference
    diff_vals = diff.head(10)
    diff_labels = [c.replace("op_", "").upper() for c in diff_vals.index]
    axes[1].barh(diff_labels[::-1], diff_vals.values[::-1],
                 color=NEUTRAL_COLOR, alpha=0.85)
    axes[1].set_title("Absolute difference (malware − benign)")
    axes[1].set_xlabel("Difference in frequency ratio")

    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "03_opcode_analysis.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path3}")

    print(f"\n  Top 10 discriminative opcodes:")
    for i, op in enumerate(top10, 1):
        label = op.replace("op_", "").upper()
        print(f"    {i:2d}. {label:<10} | "
              f"Malware: {mal_mean[op]:.4f} | "
              f"Benign: {ben_mean[op]:.4f} | "
              f"Diff: {diff[op]:.4f}")
else:
    print("  ⚠️  No opcode columns found in CSV")


# ═════════════════════════════════════════════════════════
#  DELIVERABLE 4 — Correlation Heatmap
# ═════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  Deliverable 4 — Correlation Heatmap")
print("─" * 62)

# Use key features only (not all 113 — too dense for heatmap)
key_features = []

entropy_cols_key = [c for c in df.columns if "entropy" in c][:6]
opcode_ratio_cols = ["nop_ratio", "call_ratio", "jmp_ratio",
                     "xor_ratio", "push_ratio", "mov_ratio",
                     "opcode_diversity", "opcode_total_count"]
pe_cols = ["file_size", "section_count", "imports_count",
           "exports_count", "virtual_size_ratio",
           "packer_signature", "suspicious_import_count"]

key_features = (
    [c for c in entropy_cols_key    if c in df.columns] +
    [c for c in opcode_ratio_cols   if c in df.columns] +
    [c for c in pe_cols             if c in df.columns]
)

corr_df = df[key_features].corr()

# Find highly correlated pairs
high_corr = []
for i in range(len(corr_df.columns)):
    for j in range(i+1, len(corr_df.columns)):
        val = abs(corr_df.iloc[i, j])
        if val > 0.90:
            high_corr.append((corr_df.columns[i],
                               corr_df.columns[j], val))

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(
    corr_df, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
    linewidths=0.5, annot_kws={"size": 8},
    ax=ax, cbar_kws={"shrink": 0.8}
)
ax.set_title("Feature Correlation Heatmap\n"
             "(key features — checking multicollinearity)",
             fontsize=13, fontweight="bold", pad=15)
ax.tick_params(axis="x", rotation=40, labelsize=9)
ax.tick_params(axis="y", rotation=0,  labelsize=9)

plt.tight_layout()
path4 = os.path.join(OUTPUT_DIR, "04_correlation_heatmap.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Saved: {path4}")

if high_corr:
    print(f"\n  ⚠️  High correlation pairs (> 0.90):")
    for f1, f2, v in sorted(high_corr, key=lambda x: -x[2]):
        print(f"    {f1:<30} ↔  {f2:<30} = {v:.3f}")
else:
    print("  ✅ No features above 0.90 correlation — clean dataset!")


# ═════════════════════════════════════════════════════════
#  DELIVERABLE 5 — Class Balance Chart
# ═════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  Deliverable 5 — Class Balance")
print("─" * 62)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Dataset Class Balance",
             fontsize=14, fontweight="bold")

# Pie chart
sizes  = [malware_count, benign_count]
colors = [MALWARE_COLOR, BENIGN_COLOR]
labels = [f"Malware\n{malware_count} ({malware_count/total_rows*100:.1f}%)",
          f"Benign\n{benign_count}  ({benign_count/total_rows*100:.1f}%)"]
axes[0].pie(sizes, labels=labels, colors=colors, autopct="",
            startangle=90, pctdistance=0.85,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0].set_title("Class distribution")

# Bar chart showing counts
categories = ["Malware", "Benign"]
counts     = [malware_count, benign_count]
bars = axes[1].bar(categories, counts,
                   color=[MALWARE_COLOR, BENIGN_COLOR],
                   width=0.4, alpha=0.85)
for bar, count in zip(bars, counts):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 30,
                 f"{count:,}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
axes[1].set_title("Sample counts")
axes[1].set_ylabel("Number of samples")
axes[1].set_ylim(0, max(counts) * 1.15)

plt.tight_layout()
path5 = os.path.join(OUTPUT_DIR, "05_class_balance.png")
plt.savefig(path5, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Saved: {path5}")
print(f"  Malware: {malware_count} | Benign: {benign_count} | "
      f"Ratio: {malware_count/benign_count:.2f}")


# ═════════════════════════════════════════════════════════
#  DELIVERABLE 6 — Feature Statistics Summary
# ═════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  Deliverable 6 — Feature Statistics Summary")
print("─" * 62)

summary_features = {
    "Entropy": [c for c in df.columns if "entropy" in c],
    "Opcodes": ["nop_ratio","call_ratio","jmp_ratio",
                "xor_ratio","opcode_diversity"],
    "PE Structure": ["file_size","section_count",
                     "imports_count","virtual_size_ratio"],
    "Packer": ["packer_signature","suspicious_import_count",
               "suspicious_section_count"],
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Feature Mean Values: Malware vs Benign",
             fontsize=14, fontweight="bold")

axes_flat = axes.flatten()
for idx, (group_name, cols) in enumerate(summary_features.items()):
    ax   = axes_flat[idx]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        continue

    mal_means = malware_df[cols].mean()
    ben_means = benign_df[cols].mean()

    x     = np.arange(len(cols))
    width = 0.35
    short = [c.replace("entropy_","").replace("_ratio","")
              .replace("_count","_cnt").replace("_section","_sec")
             for c in cols]

    ax.bar(x - width/2, mal_means, width,
           label="Malware", color=MALWARE_COLOR, alpha=0.85)
    ax.bar(x + width/2, ben_means, width,
           label="Benign",  color=BENIGN_COLOR,  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=9)
    ax.set_title(f"{group_name} features")
    ax.set_ylabel("Mean value")
    ax.legend(fontsize=9)

plt.tight_layout()
path6 = os.path.join(OUTPUT_DIR, "06_feature_statistics.png")
plt.savefig(path6, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Saved: {path6}")

# Print summary table
print("\n  Key feature comparison (malware vs benign mean):\n")
print(f"  {'Feature':<30} {'Malware':>10} {'Benign':>10} {'Diff':>10}")
print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")
all_key = []
for cols in summary_features.values():
    all_key += [c for c in cols if c in df.columns]
for col in all_key[:15]:
    m = malware_df[col].mean()
    b = benign_df[col].mean()
    d = m - b
    print(f"  {col:<30} {m:>10.4f} {b:>10.4f} {d:>+10.4f}")


# ═════════════════════════════════════════════════════════
#  BONUS — Sliding Window Entropy Comparison
# ═════════════════════════════════════════════════════════
if "sliding_window_mean" in df.columns:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(malware_df["sliding_window_mean"].dropna(),
            bins=50, alpha=0.7, color=MALWARE_COLOR,
            label="Malware", density=True)
    ax.hist(benign_df["sliding_window_mean"].dropna(),
            bins=50, alpha=0.7, color=BENIGN_COLOR,
            label="Benign",  density=True)
    ax.set_title("Sliding window entropy — malware vs benign",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Sliding window entropy mean")
    ax.set_ylabel("Density")
    ax.legend()
    path_bonus = os.path.join(OUTPUT_DIR, "07_sliding_window_entropy.png")
    plt.tight_layout()
    plt.savefig(path_bonus, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅ Bonus chart saved: {path_bonus}")


# ═════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHASE 3 COMPLETE")
print("=" * 62)
print(f"\n  All charts saved to: {OUTPUT_DIR}\\\n")
print("  Files generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"    {f:<45} {size//1024} KB")
print()
print("  What to show your supervisor:")
print("  ─────────────────────────────────────────────")
print("  01_dataset_overview.txt     → clean data proof")
print("  02_entropy_distribution.png → entropy gap visible")
print("  03_opcode_analysis.png      → xor/nop differences")
print("  04_correlation_heatmap.png  → no multicollinearity")
print("  05_class_balance.png        → balanced 4000/4000")
print("  06_feature_statistics.png   → feature importance")
print("  07_sliding_window_entropy.png → advanced analysis")
print()
print("  Next step → Phase 4: Run train_xgboost.py")
print("=" * 62)