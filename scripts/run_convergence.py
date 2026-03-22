"""
Experiment 2: Convergence Speed.

Measures the number of communication rounds and messages required for
each algorithm to reach a stable leader decision, across swarm sizes
and topologies.

Key insight: Fitness requires O(diameter) rounds vs O(N) for Bully.
Random and Battery require O(1) rounds but at cost of LQS.

Saves: results/exp2_convergence.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

# ── Configuration ─────────────────────────────────────────────────────
N_TRIALS = 30
SWARM_SIZES = [5, 10, 20, 30, 50, 75, 100]
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


def main():
    records = []
    total = len(SWARM_SIZES) * len(TOPOLOGIES) * len(ALGORITHMS) * N_TRIALS

    with tqdm(total=total, desc="Exp 2: Convergence") as pbar:
        for n, topo_name in product(SWARM_SIZES, TOPOLOGIES):
            for trial in range(N_TRIALS):
                seed = hash(("conv", n, topo_name, trial)) % (2**31)
                topo_gen = TopologyGenerator(seed=seed)
                graph = topo_gen.all_topologies(n)[topo_name]
                swarm = Swarm.from_graph(graph, seed=seed)

                # Compute graph properties once (used for context)
                import networkx as nx
                alive_ids = swarm.alive_ids()
                subgraph = graph.subgraph(alive_ids)
                try:
                    diameter = nx.diameter(subgraph) if nx.is_connected(subgraph) else -1
                except Exception:
                    diameter = -1
                density = nx.density(subgraph)

                for alg in ALGORITHMS:
                    result = alg.elect(swarm, graph, weights=WEIGHTS)
                    records.append({
                        "algorithm": alg.name,
                        "topology": topo_name,
                        "n_agents": n,
                        "trial": trial,
                        "rounds_to_converge": result.rounds_to_converge,
                        "messages_sent": result.messages_sent,
                        "lqs": result.lqs,
                        "graph_diameter": diameter,
                        "graph_density": round(density, 4),
                        "election_time_ms": result.election_time_ms,
                    })
                    pbar.update(1)

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "exp2_convergence.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(df)} records → {out_path}")

    # Print summary
    summary = (
        df.groupby(["algorithm", "n_agents"])["rounds_to_converge"]
        .agg(["mean", "std"])
        .round(2)
        .reset_index()
    )
    print("\n── Rounds to Converge (mean ± std) ──")
    print(summary.pivot(index="n_agents", columns="algorithm", values="mean").to_string())
    return df


if __name__ == "__main__":
    main()
