# -*- coding: utf-8 -*-
"""
Dataset statistics for NarrativeBench.
Generates descriptive tables and figures for the paper.

Output:
  - Console: all statistics printed in paper-ready format
  - results/dataset_stats/ : tables (csv + latex) and figures (png)

Usage:
    python dataset_stats.py
    python dataset_stats.py --data essay_total_data_4500_260417.xlsx
    python dataset_stats.py --no-figures   # skip figure generation
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SCORE_DIMS = ["total_score", "relevance_score", "content_score", "expression_score"]
SCORE_DIM_LABELS = {
    "total_score": "Total Score",
    "relevance_score": "Relevance",
    "content_score": "Content",
    "expression_score": "Expression",
}
HW_TITLE_LABELS = {
    "我学会了____": "I Learned ___",
    "我的奇思妙想": "My Wild Imagination",
    "我的“自画像”": "My Self-Portrait",
    "我和____过一天": "A Day with ___",
    "我们眼中的缤纷世界": "The Colorful World in Our Eyes",
    "生活万花筒": "Life Kaleidoscope",
    "游____": "Visit to ___",
    "故事新编": "Retold Fables",
    "他_____了": "He/She ___",
    "奇妙的想象": "Wonderful Imagination",
    "漫画的启示": "Inspiration from Comics",
    "我有一个想法": "I Have an Idea",
    "那一刻，我长大了": "That Moment, I Grew Up",
    "____即景": "___ Scene",
}


def load_data(path):
    df = pd.read_excel(path)
    print(f"Loaded {len(df)} essays from {path}")
    return df


# ──────────────────────────────────────────────
# 1. Overview statistics
# ──────────────────────────────────────────────
def stat_overview(df):
    print("\n" + "=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Total essays:        {len(df)}")
    print(f"  Assignment types:    {df['hw_title'].nunique()}")
    print(f"  Classes:             {df['class_id'].nunique()}")
    print(f"  Score dimensions:    {len(SCORE_DIMS)} (0–8 each)")
    print()


# ──────────────────────────────────────────────
# 2. Assignment type distribution
# ──────────────────────────────────────────────
def stat_assignment_types(df, outdir):
    print("\n" + "=" * 60)
    print("2. ASSIGNMENT TYPE DISTRIBUTION")
    print("=" * 60)
    counts = df["hw_title"].value_counts().sort_index()
    total = len(df)

    rows = []
    for title, count in counts.items():
        label = HW_TITLE_LABELS.get(title, title)
        pct = count / total * 100
        rows.append({"Assignment Type": label, "N": count, "%": f"{pct:.1f}"})
        print(f"  {label:<40} {count:>5}  ({pct:5.1f}%)")

    rows.append({"Assignment Type": "Total", "N": total, "%": "100.0"})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(f"{outdir}/assignment_types.csv", index=False)

    # LaTeX table
    with open(f"{outdir}/assignment_types.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Distribution of assignment types in NarrativeBench.}\n")
        f.write("\\begin{tabular}{lrr}\n\\toprule\n")
        f.write("Assignment Type & N & \\% \\\\\n\\midrule\n")
        for _, r in tbl[tbl["Assignment Type"] != "Total"].iterrows():
            f.write(f"  {r['Assignment Type']} & {r['N']} & {r['%']} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"  Total & {total} & 100.0 \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    labels_short = [HW_TITLE_LABELS.get(t, t) for t in counts.index]
    bars = ax.barh(range(len(labels_short)), counts.values, color="#4C72B0")
    ax.set_yticks(range(len(labels_short)))
    ax.set_yticklabels(labels_short, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Essays")
    ax.set_title("Assignment Type Distribution", fontsize=12)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{outdir}/assignment_types.png", dpi=150)
    plt.close()
    print(f"\n  Saved: assignment_types.csv, .tex, .png")


# ──────────────────────────────────────────────
# 3. Text length statistics
# ──────────────────────────────────────────────
def _remove_punct(text):
    """Remove whitespace and punctuation, count only content characters."""
    import re
    return len(re.sub(r'[\s\W]', '', str(text)))


def stat_text_length(df, outdir):
    print("\n" + "=" * 60)
    print("3. TEXT LENGTH STATISTICS (Chinese Characters)")
    print("=" * 60)

    char_len = df["content"].str.len()
    clean_len = df["content"].apply(_remove_punct)

    for name, series in [("With punctuation", char_len), ("Without punctuation", clean_len)]:
        print(f"\n  {name}:")
        print(f"    Mean:   {series.mean():.1f}")
        print(f"    Std:    {series.std():.1f}")
        print(f"    Min:    {series.min()}")
        print(f"    Max:    {series.max()}")
        for q, label in [(0.05, "p5"), (0.1, "p10"), (0.25, "p25"), (0.5, "p50/median"),
                         (0.75, "p75"), (0.9, "p90"), (0.95, "p95")]:
            print(f"    {label:>8}: {series.quantile(q):.0f}")

    # Length histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(char_len.values, bins=50, color="#55A868", edgecolor="white", alpha=0.85)
    ax1.axvline(char_len.median(), color="red", linestyle="--", label=f"Median={char_len.median():.0f}")
    ax1.axvline(char_len.mean(), color="blue", linestyle="--", label=f"Mean={char_len.mean():.0f}")
    ax1.set_xlabel("Character Count (with punctuation)")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    ax2.hist(clean_len.values, bins=50, color="#C44E52", edgecolor="white", alpha=0.85)
    ax2.axvline(clean_len.median(), color="red", linestyle="--", label=f"Median={clean_len.median():.0f}")
    ax2.axvline(clean_len.mean(), color="blue", linestyle="--", label=f"Mean={clean_len.mean():.0f}")
    ax2.set_xlabel("Content Character Count (no punctuation)")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{outdir}/text_length.png", dpi=150)
    plt.close()

    # Summary table
    summary = pd.DataFrame({
        "Metric": ["Mean", "Std", "Min", "p25", "p50", "p75", "Max"],
        "With Punctuation": [f"{char_len.mean():.1f}", f"{char_len.std():.1f}", char_len.min(),
                             f"{char_len.quantile(0.25):.0f}", f"{char_len.quantile(0.5):.0f}",
                             f"{char_len.quantile(0.75):.0f}", char_len.max()],
        "Without Punctuation": [f"{clean_len.mean():.1f}", f"{clean_len.std():.1f}", clean_len.min(),
                                f"{clean_len.quantile(0.25):.0f}", f"{clean_len.quantile(0.5):.0f}",
                                f"{clean_len.quantile(0.75):.0f}", clean_len.max()],
    })
    summary.to_csv(f"{outdir}/text_length.csv", index=False)
    print(f"\n  Saved: text_length.csv, .png")


# ──────────────────────────────────────────────
# 4. Score distributions
# ──────────────────────────────────────────────
def stat_score_distributions(df, outdir):
    print("\n" + "=" * 60)
    print("4. SCORE DISTRIBUTIONS")
    print("=" * 60)

    all_rows = []
    for dim in SCORE_DIMS:
        scores = df[dim]
        print(f"\n  {SCORE_DIM_LABELS[dim]}:")
        print(f"    Mean={scores.mean():.2f}, Std={scores.std():.2f}, "
              f"Min={scores.min()}, Max={scores.max()}, Median={scores.median():.1f}")

        dist = scores.value_counts().sort_index()
        for score in range(0, 9):
            count = dist.get(score, 0)
            pct = count / len(df) * 100
            all_rows.append({"Dimension": SCORE_DIM_LABELS[dim], "Score": score, "N": count, "%": f"{pct:.1f}"})

    pd.DataFrame(all_rows).to_csv(f"{outdir}/score_distributions.csv", index=False)

    # Score distribution stacked/overlaid bars
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for idx, dim in enumerate(SCORE_DIMS):
        ax = axes[idx]
        dist = df[dim].value_counts().sort_index()
        # Ensure all 0-8 present
        full = pd.Series(0, index=range(9))
        full.update(dist)
        colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, 9))
        bars = ax.bar(full.index, full.values, color=colors, edgecolor="white", width=0.8)
        ax.set_xticks(range(9))
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_title(SCORE_DIM_LABELS[dim], fontsize=11)
        for bar, val in zip(bars, full.values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                        str(val), ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{outdir}/score_distributions.png", dpi=150)
    plt.close()

    # Score correlation matrix
    print(f"\n  Score correlation matrix (Pearson):")
    corr = df[SCORE_DIMS].corr()
    print("  " + " " * 15 + "  ".join([f"{SCORE_DIM_LABELS[d]:>12}" for d in SCORE_DIMS]))
    for d in SCORE_DIMS:
        vals = "  ".join([f"{corr.loc[d, od]:>12.4f}" for od in SCORE_DIMS])
        print(f"  {SCORE_DIM_LABELS[d]:>15}  {vals}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(SCORE_DIMS)))
    ax.set_yticks(range(len(SCORE_DIMS)))
    ax.set_xticklabels([SCORE_DIM_LABELS[d] for d in SCORE_DIMS], rotation=45, ha="right")
    ax.set_yticklabels([SCORE_DIM_LABELS[d] for d in SCORE_DIMS])
    for i in range(len(SCORE_DIMS)):
        for j in range(len(SCORE_DIMS)):
            ax.text(j, i, f"{corr.values[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, color="black" if abs(corr.values[i, j]) < 0.7 else "white")
    plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{outdir}/score_correlation.png", dpi=150)
    plt.close()

    # LaTeX correlation table
    with open(f"{outdir}/score_correlation.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Pearson correlation matrix among score dimensions.}\n")
        f.write("\\begin{tabular}{l" + "r" * len(SCORE_DIMS) + "}\n\\toprule\n")
        f.write(" & " + " & ".join([SCORE_DIM_LABELS[d] for d in SCORE_DIMS]) + " \\\\\n\\midrule\n")
        for d in SCORE_DIMS:
            vals = " & ".join([f"{corr.loc[d, od]:.3f}" for od in SCORE_DIMS])
            f.write(f"{SCORE_DIM_LABELS[d]} & {vals} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\n  Saved: score_distributions.csv, .png; score_correlation.png, .tex")


# ──────────────────────────────────────────────
# 5. Score distribution per assignment type
# ──────────────────────────────────────────────
def stat_scores_by_type(df, outdir):
    print("\n" + "=" * 60)
    print("5. SCORE DISTRIBUTION BY ASSIGNMENT TYPE (total_score)")
    print("=" * 60)

    rows = []
    for hw in sorted(df["hw_title"].unique()):
        sub = df[df["hw_title"] == hw]
        label = HW_TITLE_LABELS.get(hw, hw)
        for dim in SCORE_DIMS:
            s = sub[dim]
            rows.append({
                "Assignment": label, "Dimension": SCORE_DIM_LABELS[dim],
                "Mean": f"{s.mean():.2f}", "Std": f"{s.std():.2f}",
                "Min": int(s.min()), "Max": int(s.max()),
            })
        print(f"  {label}: total_mean={sub['total_score'].mean():.2f}, "
              f"total_std={sub['total_score'].std():.2f}")

    pd.DataFrame(rows).to_csv(f"{outdir}/scores_by_type.csv", index=False)

    # Heatmap: assignment type × dimension × mean score
    pivot = pd.pivot_table(df[["hw_title"] + SCORE_DIMS], index="hw_title", aggfunc="mean")
    pivot.index = [HW_TITLE_LABELS.get(t, t) for t in pivot.index]
    pivot.columns = [SCORE_DIM_LABELS[d] for d in SCORE_DIMS]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=8)
    ax.set_xticks(range(len(SCORE_DIMS)))
    ax.set_yticks(range(len(pivot)))
    ax.set_xticklabels([SCORE_DIM_LABELS[d] for d in SCORE_DIMS], fontsize=10)
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot)):
        for j in range(len(SCORE_DIMS)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if pivot.values[i, j] > 5.5 else "black")
    plt.colorbar(im, shrink=0.8, label="Mean Score")
    plt.tight_layout()
    plt.savefig(f"{outdir}/scores_by_type_heatmap.png", dpi=150)
    plt.close()

    # LaTeX table
    with open(f"{outdir}/scores_by_type.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Mean scores by assignment type.}\n")
        f.write("\\begin{tabular}{l" + "r" * len(SCORE_DIMS) + "}\n\\toprule\n")
        f.write("Assignment & " + " & ".join([SCORE_DIM_LABELS[d] for d in SCORE_DIMS]) + " \\\\\n\\midrule\n")
        for label in pivot.index:
            vals = " & ".join([f"{pivot.loc[label, SCORE_DIM_LABELS[d]]:.2f}" for d in SCORE_DIMS])
            f.write(f"{label} & {vals} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\n  Saved: scores_by_type.csv, .png, .tex")


# ──────────────────────────────────────────────
# 6. Summary table (paper-ready)
# ──────────────────────────────────────────────
def stat_summary_table(df, outdir):
    print("\n" + "=" * 60)
    print("6. PAPER-READY SUMMARY TABLE")
    print("=" * 60)

    char_len = df["content"].str.len()
    clean_len = df["content"].apply(_remove_punct)

    lines = [
        f"NarrativeBench: {len(df)} Chinese student essays across {df['hw_title'].nunique()} assignment types",
        f"from {df['class_id'].nunique()} classes.",
        "",
        "Text Length (characters):",
        f"  Mean={char_len.mean():.0f}, Std={char_len.std():.0f}, Range=[{char_len.min()}, {char_len.max()}]",
        f"  Content chars (no punctuation): Mean={clean_len.mean():.0f}, Std={clean_len.std():.0f}",
        "",
        "Score Statistics (0–8 scale):",
    ]
    for dim in SCORE_DIMS:
        s = df[dim]
        lines.append(f"  {SCORE_DIM_LABELS[dim]:>12}: μ={s.mean():.2f}, σ={s.std():.2f}, "
                     f"median={s.median():.1f}, range=[{s.min()}, {s.max()}]")

    text = "\n".join(lines)
    print(text)
    with open(f"{outdir}/summary.txt", "w") as f:
        f.write(text)
    print(f"\n  Saved: summary.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="essay_total_data_4500_260417.xlsx")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation (for headless servers)")
    args = parser.parse_args()

    outdir = "results/dataset_stats"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)

    if args.no_figures:
        plt = None  # disable figures

    stat_overview(df)
    if not args.no_figures:
        stat_assignment_types(df, outdir)
    else:
        stat_assignment_types_no_fig(df, outdir)

    if not args.no_figures:
        stat_text_length(df, outdir)
    else:
        stat_text_length_no_fig(df, outdir)

    if not args.no_figures:
        stat_score_distributions(df, outdir)
    else:
        stat_score_distributions_no_fig(df, outdir)

    if not args.no_figures:
        stat_scores_by_type(df, outdir)
    else:
        stat_scores_by_type_no_fig(df, outdir)

    stat_summary_table(df, outdir)


def _assignment_types_table(df, outdir):
    counts = df["hw_title"].value_counts().sort_index()
    total = len(df)
    rows = []
    for title, count in counts.items():
        label = HW_TITLE_LABELS.get(title, title)
        pct = count / total * 100
        rows.append({"Assignment Type": label, "N": count, "%": f"{pct:.1f}"})
    rows.append({"Assignment Type": "Total", "N": total, "%": "100.0"})
    pd.DataFrame(rows).to_csv(f"{outdir}/assignment_types.csv", index=False)

    with open(f"{outdir}/assignment_types.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Distribution of assignment types in NarrativeBench.}\n")
        f.write("\\begin{tabular}{lrr}\n\\toprule\n")
        f.write("Assignment Type & N & \\% \\\\\n\\midrule\n")
        for _, r in pd.DataFrame(rows)[pd.DataFrame(rows)["Assignment Type"] != "Total"].iterrows():
            f.write(f"  {r['Assignment Type']} & {r['N']} & {r['%']} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"  Total & {total} & 100.0 \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def stat_assignment_types_no_fig(df, outdir):
    print("\n" + "=" * 60)
    print("2. ASSIGNMENT TYPE DISTRIBUTION")
    print("=" * 60)
    counts = df["hw_title"].value_counts().sort_index()
    total = len(df)
    for title, count in counts.items():
        label = HW_TITLE_LABELS.get(title, title)
        pct = count / total * 100
        print(f"  {label:<40} {count:>5}  ({pct:5.1f}%)")
    _assignment_types_table(df, outdir)
    print(f"\n  Saved: assignment_types.csv, .tex (no figure)")


def _text_length_table(df, outdir):
    char_len = df["content"].str.len()
    clean_len = df["content"].apply(_remove_punct)
    summary = pd.DataFrame({
        "Metric": ["Mean", "Std", "Min", "p25", "p50", "p75", "Max"],
        "With Punctuation": [f"{char_len.mean():.1f}", f"{char_len.std():.1f}", char_len.min(),
                             f"{char_len.quantile(0.25):.0f}", f"{char_len.quantile(0.5):.0f}",
                             f"{char_len.quantile(0.75):.0f}", char_len.max()],
        "Without Punctuation": [f"{clean_len.mean():.1f}", f"{clean_len.std():.1f}", clean_len.min(),
                                f"{clean_len.quantile(0.25):.0f}", f"{clean_len.quantile(0.5):.0f}",
                                f"{clean_len.quantile(0.75):.0f}", clean_len.max()],
    })
    summary.to_csv(f"{outdir}/text_length.csv", index=False)


def stat_text_length_no_fig(df, outdir):
    print("\n" + "=" * 60)
    print("3. TEXT LENGTH STATISTICS")
    print("=" * 60)
    char_len = df["content"].str.len()
    clean_len = df["content"].apply(_remove_punct)
    for name, series in [("With punctuation", char_len), ("Without punctuation", clean_len)]:
        print(f"\n  {name}:")
        print(f"    Mean={series.mean():.1f}, Std={series.std():.1f}, "
              f"Min={series.min()}, Max={series.max()}")
    _text_length_table(df, outdir)
    print(f"\n  Saved: text_length.csv (no figure)")


def _score_dist_tables(df, outdir):
    all_rows = []
    for dim in SCORE_DIMS:
        dist = df[dim].value_counts().sort_index()
        for score in range(0, 9):
            count = dist.get(score, 0)
            pct = count / len(df) * 100
            all_rows.append({"Dimension": SCORE_DIM_LABELS[dim], "Score": score, "N": count, "%": f"{pct:.1f}"})
    pd.DataFrame(all_rows).to_csv(f"{outdir}/score_distributions.csv", index=False)

    corr = df[SCORE_DIMS].corr()
    with open(f"{outdir}/score_correlation.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Pearson correlation matrix among score dimensions.}\n")
        f.write("\\begin{tabular}{l" + "r" * len(SCORE_DIMS) + "}\n\\toprule\n")
        f.write(" & " + " & ".join([SCORE_DIM_LABELS[d] for d in SCORE_DIMS]) + " \\\\\n\\midrule\n")
        for d in SCORE_DIMS:
            vals = " & ".join([f"{corr.loc[d, od]:.3f}" for od in SCORE_DIMS])
            f.write(f"{SCORE_DIM_LABELS[d]} & {vals} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def stat_score_distributions_no_fig(df, outdir):
    print("\n" + "=" * 60)
    print("4. SCORE DISTRIBUTIONS")
    print("=" * 60)
    for dim in SCORE_DIMS:
        scores = df[dim]
        print(f"\n  {SCORE_DIM_LABELS[dim]}:")
        print(f"    Mean={scores.mean():.2f}, Std={scores.std():.2f}, "
              f"Min={scores.min()}, Max={scores.max()}, Median={scores.median():.1f}")
    print(f"\n  Score correlation matrix (Pearson):")
    corr = df[SCORE_DIMS].corr()
    for d in SCORE_DIMS:
        vals = "  ".join([f"{corr.loc[d, od]:>10.4f}" for od in SCORE_DIMS])
        print(f"    {SCORE_DIM_LABELS[d]:>12}  {vals}")
    _score_dist_tables(df, outdir)
    print(f"\n  Saved: score_distributions.csv, score_correlation.tex (no figure)")


def _scores_by_type_tables(df, outdir):
    rows = []
    for hw in sorted(df["hw_title"].unique()):
        sub = df[df["hw_title"] == hw]
        label = HW_TITLE_LABELS.get(hw, hw)
        for dim in SCORE_DIMS:
            s = sub[dim]
            rows.append({
                "Assignment": label, "Dimension": SCORE_DIM_LABELS[dim],
                "Mean": f"{s.mean():.2f}", "Std": f"{s.std():.2f}",
                "Min": int(s.min()), "Max": int(s.max()),
            })
    pd.DataFrame(rows).to_csv(f"{outdir}/scores_by_type.csv", index=False)

    pivot = pd.pivot_table(df[["hw_title"] + SCORE_DIMS], index="hw_title", aggfunc="mean")
    pivot.index = [HW_TITLE_LABELS.get(t, t) for t in pivot.index]
    pivot.columns = [SCORE_DIM_LABELS[d] for d in SCORE_DIMS]
    with open(f"{outdir}/scores_by_type.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Mean scores by assignment type.}\n")
        f.write("\\begin{tabular}{l" + "r" * len(SCORE_DIMS) + "}\n\\toprule\n")
        f.write("Assignment & " + " & ".join([SCORE_DIM_LABELS[d] for d in SCORE_DIMS]) + " \\\\\n\\midrule\n")
        for label in pivot.index:
            vals = " & ".join([f"{pivot.loc[label, SCORE_DIM_LABELS[d]]:.2f}" for d in SCORE_DIMS])
            f.write(f"{label} & {vals} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def stat_scores_by_type_no_fig(df, outdir):
    print("\n" + "=" * 60)
    print("5. SCORE DISTRIBUTION BY ASSIGNMENT TYPE")
    print("=" * 60)
    for hw in sorted(df["hw_title"].unique()):
        sub = df[df["hw_title"] == hw]
        label = HW_TITLE_LABELS.get(hw, hw)
        print(f"  {label}: total_mean={sub['total_score'].mean():.2f}, "
              f"total_std={sub['total_score'].std():.2f}")
    _scores_by_type_tables(df, outdir)
    print(f"\n  Saved: scores_by_type.csv, .tex (no figure)")


if __name__ == "__main__":
    main()
