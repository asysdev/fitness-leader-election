"""
Experiment 1: Election Quality (LQS) Across Algorithms and Topologies.

For each combination of (algorithm × topology × swarm_size):
  - Run N_TRIALS independent elections with random agent configurations.
  - Record LQS, rounds_to_converge, and messages_sent.
  - Compute mean ± std across trials.

Saves: results/exp1_election_quality.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
from itertools import product
from pathlib import Path

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
from src.election import (
    FitnessElection, BullyElection, RandomElection,
    BatteryElection, DegreeElection, RaftElection,
)

# ── Experiment configuration ──────────────────────────────────────────
N_TRIALS = 30
SWARM_SIZES = [10, 20, 50, 100]
TOPOLOGIES = ["mesh", "random", "scale_free", "geometric"]
WEIGHTS = (0.4, 0.35, 0.25)
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ALGORITHMS = [
    FitnessElection(),
    BullyElection(),
    RandomElection(),
    BatteryElection(),
    DegreeElection(),
    RaftElection(),
]


def run_trial(algorithm, graph, swarm):
    """Run a single election trial and return the result dict."""
    result = algorithm.elect(swarm, graph, weights=WEIGHTS)
    return result.to_dict()


def main():
    records = []

    combos = list(product(SWARM_SIZES, TOPOLOGIES))
    total = len(combos) * len(ALGORITHMS) * N_TRIALS

    with tqdm(total=total, desc="Exp 1: Election Quality") as pbar:
        for n, topo_name in combos:
            for trial in range(N_TRIALS):
                seed = hash((n, topo_name, trial)) % (2**31)
                topo_gen = TopologyGenerator(seed=seed)

                # Generate topology
                topos = topo_gen.all_topologies(n)
                graph = topos[topo_name]
                swarm = Swarm.from_graph(graph, seed=seed)

                for alg in ALGORITHMS:
                    result = alg.elect(swarm, graph, weights=WEIGHTS)
                    records.append({
                        "algorithm": alg.name,
                        "topology": topo_name,
                        "n_agents": n,
                        "trial": trial,
                        "lqs": result.lqs,
                        "leader_fitness": result.leader_fitness,
                        "rounds_to_converge": result.rounds_to_converge,
                        "messages_sent": result.messages_sent,
                        "election_time_ms": result.election_time_ms,
                    })
                    pbar.update(1)

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "exp1_election_quality.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(df)} records → {out_path}")

    # Print summary table
    summary = (
        df.groupby(["algorithm", "topology"])["lqs"]
        .agg(["mean", "std"])
        .round(4)
        .reset_index()
    )
    print("\n── LQS Summary (mean ± std) ──")
    print(summary.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
