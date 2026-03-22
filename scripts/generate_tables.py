"""
Generate LaTeX summary tables from experiment results.

Tables produced:
  table1_lqs_summary.tex         — Mean LQS ± std per algorithm × topology
  table2_convergence.tex         — Rounds and messages vs N
  table3_recovery.tex            — Recovery metrics by failure type
  table4_churn.tex               — Churn stability metrics
  table5_top_weights.tex         — Top-10 weight combinations

All tables are saved to results/tables/.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
TABLE_DIR = RESULTS_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

ALG_ORDER = ["Fitness", "Bully", "Random", "HighBattery", "MostConnected", "Raft"]
TOPO_ORDER = ["mesh", "random", "scale_free", "geometric"]
TOPO_LABELS = {
    "mesh": "Mesh",
    "random": "Erdős–Rényi",
    "scale_free": "Scale-Free",
    "geometric": "Geometric",
}


def _save_tex(content: str, name: str):
    path = TABLE_DIR / name
    path.write_text(content)
    print(f"  ✓ {name}")


def table1_lqs_summary():
    """Table 1: Mean LQS ± std, algorithm × topology."""
    path = RESULTS_DIR / "exp1_election_quality.csv"
    if not path.exists():
        print("  ✗ exp1 missing — skipping table1")
        return

    df = pd.read_csv(path)
    summary = df.groupby(["algorithm", "topology"])["lqs"].agg(["mean", "std"]).reset_index()
    summary["cell"] = summary.apply(
        lambda r: f"{r['mean']:.3f} $\\pm$ {r['std']:.3f}", axis=1
    )
    pivot = (
        summary.pivot(index="algorithm", columns="topology", values="cell")
        .reindex(index=ALG_ORDER, columns=TOPO_ORDER)
    )

    col_headers = " & ".join([TOPO_LABELS.get(t, t) for t in TOPO_ORDER])
    rows = []
    for alg in pivot.index:
        vals = " & ".join([str(pivot.loc[alg, t]) if t in pivot.columns else "--"
                           for t in TOPO_ORDER])
        marker = " \\textbf{*}" if alg == "Fitness" else ""
        rows.append(f"  {alg}{marker} & {vals} \\\\")

    tex = "\n".join([
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Mean LQS $\\pm$ Std by Algorithm and Topology (higher is better; $n=30$ trials per cell)}",
        "\\label{tab:lqs_summary}",
        "\\begin{tabular}{l" + "c" * len(TOPO_ORDER) + "}",
        "\\toprule",
        f"  Algorithm & {col_headers} \\\\",
        "\\midrule",
        "\n".join(rows),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    _save_tex(tex, "table1_lqs_summary.tex")


def table2_convergence():
    """Table 2: Rounds to converge and messages vs swarm size."""
    path = RESULTS_DIR / "exp2_convergence.csv"
    if not path.exists():
        print("  ✗ exp2 missing — skipping table2")
        return

    df = pd.read_csv(path)
    summary = (
        df.groupby(["algorithm", "n_agents"])[["rounds_to_converge", "messages_sent"]]
        .mean()
        .round(1)
        .reset_index()
    )

    sizes = sorted(df["n_agents"].unique())
    algs = [a for a in ALG_ORDER if a in summary["algorithm"].unique()]

    size_headers = " & ".join([f"$N={n}$" for n in sizes])
    rows = []

    rows.append("\\multicolumn{" + str(len(sizes) + 1) + "}{l}{\\textit{Rounds to Converge}} \\\\")
    rows.append("\\midrule")
    for alg in algs:
        sub = summary[summary["algorithm"] == alg].set_index("n_agents")
        vals = " & ".join([f"{sub.loc[n, 'rounds_to_converge']:.1f}" if n in sub.index else "--"
                           for n in sizes])
        rows.append(f"  {alg} & {vals} \\\\")

    rows.append("\\midrule")
    rows.append("\\multicolumn{" + str(len(sizes) + 1) + "}{l}{\\textit{Messages Sent}} \\\\")
    rows.append("\\midrule")
    for alg in algs:
        sub = summary[summary["algorithm"] == alg].set_index("n_agents")
        vals = " & ".join([f"{sub.loc[n, 'messages_sent']:.0f}" if n in sub.index else "--"
                           for n in sizes])
        rows.append(f"  {alg} & {vals} \\\\")

    tex = "\n".join([
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Convergence: Rounds and Messages by Algorithm and Swarm Size (mean across topologies and trials)}",
        "\\label{tab:convergence}",
        "\\begin{tabular}{l" + "r" * len(sizes) + "}",
        "\\toprule",
        f"  Algorithm & {size_headers} \\\\",
        "\\midrule",
        "\n".join(rows),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    _save_tex(tex, "table2_convergence.tex")


def table3_recovery():
    """Table 3: Recovery metrics by algorithm and failure type."""
    path = RESULTS_DIR / "exp3_failure_recovery.csv"
    if not path.exists():
        print("  ✗ exp3 missing — skipping table3")
        return

    df = pd.read_csv(path)
    summary = (
        df.groupby(["algorithm", "failure_type"])[["recovery_lqs", "recovery_rounds", "recovery_messages"]]
        .mean()
        .round(3)
        .reset_index()
    )

    failure_types = ["leader_only", "leader_and_secondary"]
    algs = [a for a in ALG_ORDER if a in summary["algorithm"].unique()]

    rows = []
    for ft in failure_types:
        label = "Single Failure" if ft == "leader_only" else "Cascaded Failure"
        rows.append(f"  \\multicolumn{{4}}{{l}}{{\\textit{{{label}}}}} \\\\")
        rows.append("  \\midrule")
        sub = summary[summary["failure_type"] == ft].set_index("algorithm")
        for alg in algs:
            if alg not in sub.index:
                continue
            row = sub.loc[alg]
            rows.append(
                f"  {alg} & {row['recovery_lqs']:.3f} & {row['recovery_rounds']:.1f} "
                f"& {row['recovery_messages']:.0f} \\\\"
            )
        rows.append("  \\addlinespace")

    tex = "\n".join([
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Failure Recovery: LQS, Rounds, and Messages after Leader Failure}",
        "\\label{tab:recovery}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "  Algorithm & Recovery LQS & Recovery Rounds & Recovery Messages \\\\",
        "\\midrule",
        "\n".join(rows),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    _save_tex(tex, "table3_recovery.tex")


def table4_churn():
    """Table 4: Churn stability metrics."""
    path = RESULTS_DIR / "exp4_churn.csv"
    if not path.exists():
        print("  ✗ exp4 missing — skipping table4")
        return

    df = pd.read_csv(path)
    summary = (
        df.groupby(["algorithm", "churn_rate"])[["avg_lqs", "leader_changes", "lqs_std"]]
        .mean()
        .round(3)
        .reset_index()
    )

    churn_rates = sorted(df["churn_rate"].unique())
    algs = [a for a in ALG_ORDER if a in summary["algorithm"].unique()]

    rate_headers = " & ".join([f"$\\lambda={r}$" for r in churn_rates])
    rows = []
    rows.append("\\multicolumn{" + str(len(churn_rates) + 1) + "}{l}{\\textit{Mean LQS per Round}} \\\\")
    rows.append("\\midrule")
    for alg in algs:
        sub = summary[summary["algorithm"] == alg].set_index("churn_rate")
        vals = " & ".join([f"{sub.loc[r, 'avg_lqs']:.3f}" if r in sub.index else "--"
                           for r in churn_rates])
        rows.append(f"  {alg} & {vals} \\\\")

    tex = "\n".join([
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Churn Stability: Mean LQS per Round under Different Churn Rates ($N=20$, $T=50$ rounds)}",
        "\\label{tab:churn}",
        "\\begin{tabular}{l" + "c" * len(churn_rates) + "}",
        "\\toprule",
        f"  Algorithm & {rate_headers} \\\\",
        "\\midrule",
        "\n".join(rows),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    _save_tex(tex, "table4_churn.tex")


def table5_top_weights():
    """Table 5: Top-10 weight combinations for fitness algorithm."""
    path = RESULTS_DIR / "exp5_weight_sensitivity.csv"
    if not path.exists():
        print("  ✗ exp5 missing — skipping table5")
        return

    df = pd.read_csv(path).sort_values("mean_lqs", ascending=False)
    top10 = df.head(10)

    rows = []
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        is_default = (row["w_ir"] == 0.4 and row["w_cc"] == 0.35 and row["w_mc"] == 0.25)
        marker = " $\\dagger$" if is_default else ""
        rows.append(
            f"  {rank} & {row['w_ir']:.1f} & {row['w_cc']:.1f} & {row['w_mc']:.1f}"
            f" & {row['mean_lqs']:.4f} & {row['std_lqs']:.4f}{marker} \\\\"
        )

    tex = "\n".join([
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Top-10 Weight Combinations by Mean LQS ($\\dagger$ = patent default)}",
        "\\label{tab:top_weights}",
        "\\begin{tabular}{crrrcc}",
        "\\toprule",
        "  Rank & $w_{IR}$ & $w_{CC}$ & $w_{MC}$ & Mean LQS & Std LQS \\\\",
        "\\midrule",
        "\n".join(rows),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    _save_tex(tex, "table5_top_weights.tex")


def main():
    print("Generating LaTeX tables...")
    table1_lqs_summary()
    table2_convergence()
    table3_recovery()
    table4_churn()
    table5_top_weights()
    print(f"\n✓ All tables saved → {TABLE_DIR}/")


if __name__ == "__main__":
    main()
