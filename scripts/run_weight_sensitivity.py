"""
Experiment 5: Weight Sensitivity Analysis.

Exhaustively explores all (w_IR, w_CC, w_MC) weight combinations where:
  - Each weight ∈ {0.0, 0.1, 0.2, ..., 1.0}
  - w_IR + w_CC + w_MC = 1.0

For each valid triplet:
  - Run N_SAMPLES elections with random agent configurations.
  - Record mean LQS of the fitness algorithm with those weights.

This reveals which weight combinations produce the highest leadership
quality and validates the patent's default (0.40, 0.35, 0.25).

Output includes:
  - results/exp5_weight_sensitivity.csv
  - Ternary heatmap visualization

Saves: results/exp5_weight_sensitivity.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    import contextlib
    class tqdm:
        def __init__(self, iterable=None, **kw):
            self._it = iterable
            self.n = 0
        def __iter__(self):
            return iter(self._it) if self._it is not None else iter([])
        def update(self, n=1):
            self.n += n
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass


from src.agents.swarm import Swarm
from src.network.topology import TopologyGenerator
from src.election.fitness import FitnessElection

# ── Configuration ─────────────────────────────────────────────────────
N_SAMPLES = 10          # elections per weight combination
N_AGENTS = 20
TOPOLOGY = "random"
WEIGHT_STEP = 0.1       # resolution of weight grid
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Enumerate all valid weight triplets
def generate_weight_triplets(step: float = 0.1):
    """
    Generate all (w1, w2, w3) where each ∈ {0.0, step, 2*step, ...}
    and w1 + w2 + w3 = 1.0 (within floating-point tolerance).
    """
    values = [round(v * step, 10) for v in range(int(1.0 / step) + 1)]
    triplets = []
    for w1, w2 in itertools.product(values, repeat=2):
        w3 = round(1.0 - w1 - w2, 10)
        if 0.0 <= w3 <= 1.0:
            triplets.append((w1, w2, w3))
    return triplets


def main():
    alg = FitnessElection()
    triplets = generate_weight_triplets(WEIGHT_STEP)

    print(f"Testing {len(triplets)} weight combinations × {N_SAMPLES} samples "
          f"= {len(triplets) * N_SAMPLES} elections")

    records = []

    with tqdm(total=len(triplets), desc="Exp 5: Weight Sensitivity") as pbar:
        for idx, (w_ir, w_cc, w_mc) in enumerate(triplets):
            weights = (w_ir, w_cc, w_mc)
            lqs_values = []

            for sample in range(N_SAMPLES):
                seed = hash(("weight", idx, sample)) % (2**31)
                topo_gen = TopologyGenerator(seed=seed)
                graph = topo_gen.random(N_AGENTS, p=0.3, ensure_connected=True)
                swarm = Swarm.from_graph(graph, seed=seed)

                result = alg.elect(swarm, graph, weights=weights)
                lqs_values.append(result.lqs)

            records.append({
                "w_ir": w_ir,
                "w_cc": w_cc,
                "w_mc": w_mc,
                "mean_lqs": np.mean(lqs_values),
                "std_lqs": np.std(lqs_values),
                "min_lqs": np.min(lqs_values),
                "max_lqs": np.max(lqs_values),
            })
            pbar.update(1)

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "exp5_weight_sensitivity.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(df)} weight combinations → {out_path}")

    # Report top-10 and patent default
    df_sorted = df.sort_values("mean_lqs", ascending=False)
    print("\n── Top 10 Weight Combinations by Mean LQS ──")
    print(df_sorted.head(10)[["w_ir", "w_cc", "w_mc", "mean_lqs", "std_lqs"]].to_string(index=False))

    patent_default = df[
        (df["w_ir"] == 0.4) & (df["w_cc"] == 0.35) & (df["w_mc"] == 0.25)
    ]
    if not patent_default.empty:
        row = patent_default.iloc[0]
        rank = df_sorted.index.get_loc(patent_default.index[0]) + 1
        print(f"\n── Patent Default (0.40, 0.35, 0.25) ──")
        print(f"  Mean LQS: {row['mean_lqs']:.4f} ± {row['std_lqs']:.4f}")
        print(f"  Rank: {rank} / {len(df)}")
    else:
        print("\nNote: Patent default (0.40, 0.35, 0.25) not in grid "
              "(adjust WEIGHT_STEP to include non-round values).")

    return df


if __name__ == "__main__":
    main()
