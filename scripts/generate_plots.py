"""
Generate all publication-quality figures from saved experiment results.

Figures produced:
  fig1_lqs_by_algorithm.png          — Box plot: LQS per algorithm
  fig2_lqs_by_topology.png           — Grouped bar: LQS × topology
  fig3_convergence_vs_n.png          — Line plot: rounds vs swarm size
  fig4_messages_vs_n.png             — Line plot: messages vs swarm size
  fig5_recovery_comparison.png       — Bar: recovery LQS, single vs cascade
  fig6_churn_stability.png           — Line: avg LQS vs churn rate
  fig7_weight_ternary.png            — Ternary heatmap (matplotlib)
  fig8_lqs_heatmap.png               — Heatmap: algorithm × topology LQS

All figures are saved to results/figures/.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
PALETTE = {
    "Fitness":       "#2196F3",
    "Bully":         "#F44336",
    "Random":        "#9E9E9E",
    "HighBattery":   "#FF9800",
    "MostConnected": "#4CAF50",
    "Raft":          "#9C27B0",
}
ALG_ORDER = ["Fitness", "Bully", "Random", "HighBattery", "MostConnected", "Raft"]
TOPO_ORDER = ["mesh", "random", "scale_free", "geometric"]
DPI = 150


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


def fig1_lqs_boxplot():
    """Figure 1: LQS distribution per algorithm (box plot)."""
    path = RESULTS_DIR / "exp1_election_quality.csv"
    if not path.exists():
        print("  ✗ exp1 data missing — skipping fig1")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    algs_present = [a for a in ALG_ORDER if a in df["algorithm"].unique()]
    sns.boxplot(
        data=df, x="algorithm", y="lqs", order=algs_present,
        hue="algorithm", palette=PALETTE, ax=ax, width=0.6,
        fliersize=3, legend=False,
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="Optimal (LQS=1)")
    ax.set_xlabel("Algorithm", fontsize=13)
    ax.set_ylabel("Leadership Quality Score (LQS)", fontsize=13)
    ax.set_title("Election Quality: LQS Distribution by Algorithm", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.08)
    _save(fig, "fig1_lqs_by_algorithm.png")


def fig2_lqs_topology():
    """Figure 2: Mean LQS per algorithm × topology (grouped bar)."""
    path = RESULTS_DIR / "exp1_election_quality.csv"
    if not path.exists():
        print("  ✗ exp1 data missing — skipping fig2")
        return
    df = pd.read_csv(path)

    summary = (
        df.groupby(["algorithm", "topology"])["lqs"]
        .mean()
        .reset_index()
    )
    algs_present = [a for a in ALG_ORDER if a in summary["algorithm"].unique()]

    fig, ax = plt.subplots(figsize=(12, 5))
    topos_present = [t for t in TOPO_ORDER if t in summary["topology"].unique()]

    x = np.arange(len(algs_present))
    width = 0.8 / len(topos_present)
    topo_colors = sns.color_palette("Set2", len(topos_present))

    for i, topo in enumerate(topos_present):
        sub = summary[summary["topology"] == topo].set_index("algorithm")
        vals = [sub.loc[a, "lqs"] if a in sub.index else 0 for a in algs_present]
        offset = (i - len(topos_present) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9, label=topo.replace("_", "-"),
               color=topo_colors[i], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(algs_present, fontsize=12)
    ax.set_ylabel("Mean LQS", fontsize=13)
    ax.set_title("Mean LQS by Algorithm and Topology", fontsize=14, fontweight="bold")
    ax.legend(title="Topology", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    _save(fig, "fig2_lqs_by_topology.png")


def fig3_convergence():
    """Figure 3: Rounds to converge vs swarm size."""
    path = RESULTS_DIR / "exp2_convergence.csv"
    if not path.exists():
        print("  ✗ exp2 data missing — skipping fig3")
        return
    df = pd.read_csv(path)

    summary = (
        df.groupby(["algorithm", "n_agents"])["rounds_to_converge"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for alg in ALG_ORDER:
        sub = summary[summary["algorithm"] == alg]
        if sub.empty:
            continue
        ax.plot(sub["n_agents"], sub["rounds_to_converge"],
                marker="o", label=alg, color=PALETTE.get(alg), linewidth=2, markersize=5)

    ax.set_xlabel("Swarm Size (N agents)", fontsize=13)
    ax.set_ylabel("Mean Rounds to Converge", fontsize=13)
    ax.set_title("Convergence Speed vs Swarm Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, ncol=2)
    ax.set_yscale("log")
    _save(fig, "fig3_convergence_vs_n.png")


def fig4_messages():
    """Figure 4: Messages sent vs swarm size."""
    path = RESULTS_DIR / "exp2_convergence.csv"
    if not path.exists():
        print("  ✗ exp2 data missing — skipping fig4")
        return
    df = pd.read_csv(path)

    summary = (
        df.groupby(["algorithm", "n_agents"])["messages_sent"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for alg in ALG_ORDER:
        sub = summary[summary["algorithm"] == alg]
        if sub.empty:
            continue
        ax.plot(sub["n_agents"], sub["messages_sent"],
                marker="s", label=alg, color=PALETTE.get(alg), linewidth=2, markersize=5)

    ax.set_xlabel("Swarm Size (N agents)", fontsize=13)
    ax.set_ylabel("Mean Messages Sent", fontsize=13)
    ax.set_title("Communication Overhead vs Swarm Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, ncol=2)
    ax.set_yscale("log")
    _save(fig, "fig4_messages_vs_n.png")


def fig5_recovery():
    """Figure 5: Recovery LQS by failure type."""
    path = RESULTS_DIR / "exp3_failure_recovery.csv"
    if not path.exists():
        print("  ✗ exp3 data missing — skipping fig5")
        return
    df = pd.read_csv(path)

    summary = (
        df.groupby(["algorithm", "failure_type"])["recovery_lqs"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    failure_types = ["leader_only", "leader_and_secondary"]
    titles = ["Single Failure (Leader Only)", "Cascaded Failure (Leader + Secondary)"]

    algs_present = [a for a in ALG_ORDER if a in summary["algorithm"].unique()]
    colors = [PALETTE.get(a, "#888") for a in algs_present]

    for ax, ft, title in zip(axes, failure_types, titles):
        sub = summary[summary["failure_type"] == ft].set_index("algorithm")
        vals = [sub.loc[a, "recovery_lqs"] if a in sub.index else 0 for a in algs_present]
        bars = ax.bar(algs_present, vals, color=colors, edgecolor="white", alpha=0.87)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Recovery LQS" if ax == axes[0] else "", fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", rotation=25, labelsize=10)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Post-Failure Recovery Quality", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig5_recovery_comparison.png")


def fig6_churn():
    """Figure 6: LQS stability under different churn rates."""
    path = RESULTS_DIR / "exp4_churn.csv"
    if not path.exists():
        print("  ✗ exp4 data missing — skipping fig6")
        return
    df = pd.read_csv(path)

    summary = (
        df.groupby(["algorithm", "churn_rate"])["avg_lqs"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for alg in ALG_ORDER:
        sub = summary[summary["algorithm"] == alg]
        if sub.empty:
            continue
        ax.errorbar(
            sub["churn_rate"], sub["mean"], yerr=sub["std"],
            label=alg, color=PALETTE.get(alg), marker="D",
            linewidth=2, markersize=6, capsize=4,
        )

    ax.set_xlabel("Churn Rate (fraction of agents per round)", fontsize=13)
    ax.set_ylabel("Mean LQS (across all rounds)", fontsize=13)
    ax.set_title("Leadership Quality Under Node Churn", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, ncol=2)
    ax.set_ylim(0, 1.1)
    _save(fig, "fig6_churn_stability.png")


def fig7_ternary_heatmap():
    """
    Figure 7: Ternary heatmap of mean LQS over weight space.

    The fitness algorithm always achieves LQS≈1.0 for any positive weights
    (it picks the best agent by definition). So we visualise the *standard
    deviation* of LQS across random trials — a proxy for election stability
    under different emphasis weights.  A low std means the algorithm reliably
    finds a good leader regardless of the random agent configuration.
    """
    path = RESULTS_DIR / "exp5_weight_sensitivity.csv"
    if not path.exists():
        print("  ✗ exp5 data missing — skipping fig7")
        return
    df = pd.read_csv(path)

    w1 = df["w_ir"].values   # IR weight  → bottom-left corner
    w2 = df["w_cc"].values   # CC weight  → bottom-right corner
    w3 = df["w_mc"].values   # MC weight  → top corner

    # Standard ternary → 2-D Cartesian:
    #   bottom-left  = (0, 0)  ← pure IR
    #   bottom-right = (1, 0)  ← pure CC
    #   top-centre   = (0.5, √3/2)  ← pure MC
    x = w2 + w3 / 2.0
    y = w3 * np.sqrt(3) / 2.0

    # We show std_lqs (trial-to-trial variance in LQS).
    # Lower std = more consistent leader selection.
    z = df["std_lqs"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (values, label, cmap, vmin, vmax) in zip(
        axes,
        [
            (df["mean_lqs"].values, "Mean LQS", "RdYlGn", 0.85, 1.0),
            (df["std_lqs"].values,  "Std LQS (lower = more stable)", "RdYlGn_r", 0.0, 0.15),
        ],
    ):
        triang = tri.Triangulation(x, y)
        ax.set_aspect("equal")
        cf = ax.tricontourf(triang, values, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.tricontour(triang, values, levels=8, colors="black", linewidths=0.25, alpha=0.35)
        plt.colorbar(cf, ax=ax, label=label, fraction=0.04, shrink=0.85)

        # Triangle boundary
        bx = [0, 1, 0.5, 0]
        by = [0, 0, np.sqrt(3) / 2, 0]
        ax.plot(bx, by, "k-", linewidth=1.5)

        # Corner labels
        off = 0.07
        ax.text(0.0, -off,             "IR=1\nCC=0\nMC=0", ha="center", va="top",  fontsize=8, color="navy")
        ax.text(1.0, -off,             "IR=0\nCC=1\nMC=0", ha="center", va="top",  fontsize=8, color="navy")
        ax.text(0.5, np.sqrt(3)/2+off, "IR=0\nCC=0\nMC=1", ha="center", va="bottom", fontsize=8, color="navy")

        # Patent default star (0.40, 0.35, 0.25)
        px = 0.35 + 0.25 / 2.0
        py = 0.25 * np.sqrt(3) / 2.0
        ax.scatter([px], [py], color="blue", s=180, zorder=6, marker="*",
                   label="Patent default\n(0.40, 0.35, 0.25)")
        ax.legend(fontsize=8, loc="lower center")
        ax.axis("off")

    axes[0].set_title("Mean LQS\n(higher = better election quality)",
                      fontsize=12, fontweight="bold")
    axes[1].set_title("Std LQS across trials\n(lower = more stable)",
                      fontsize=12, fontweight="bold")

    fig.suptitle("Fitness Weight Sensitivity — Ternary Analysis (IR / CC / MC)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "fig7_weight_ternary.png")


def fig8_lqs_heatmap():
    """Figure 8: Heatmap of mean LQS — algorithm × topology."""
    path = RESULTS_DIR / "exp1_election_quality.csv"
    if not path.exists():
        print("  ✗ exp1 data missing — skipping fig8")
        return
    df = pd.read_csv(path)

    pivot = (
        df.groupby(["algorithm", "topology"])["lqs"]
        .mean()
        .unstack("topology")
        .reindex(index=ALG_ORDER, columns=TOPO_ORDER)
        .dropna(how="all")
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Mean LQS"},
    )
    ax.set_xlabel("Topology", fontsize=13)
    ax.set_ylabel("Algorithm", fontsize=13)
    ax.set_title("Mean LQS — Algorithm × Topology", fontsize=14, fontweight="bold")
    _save(fig, "fig8_lqs_heatmap.png")


def main():
    print("Generating figures...")
    fig1_lqs_boxplot()
    fig2_lqs_topology()
    fig3_convergence()
    fig4_messages()
    fig5_recovery()
    fig6_churn()
    fig7_ternary_heatmap()
    fig8_lqs_heatmap()
    print(f"\n✓ All figures saved → {FIG_DIR}/")


if __name__ == "__main__":
    main()
