"""
Experiment 3: Failure Recovery.

Simulates leader failure mid-mission and measures:
  - Recovery time (rounds from failure to new stable leader)
  - Post-recovery LQS
  - Number of cascading elections triggered
  - Whether succession (secondary promotion) vs full re-election occurred

Protocol:
  1. Run initial election on a random swarm.
  2. Kill the elected leader (remove from graph).
  3. Measure how many rounds / messages each algorithm needs to re-elect.
  4. Repeat for cascaded failures (kill both leader and secondary).

Saves: results/exp3_failure_recovery.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
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
SWARM_SIZES = [10, 20, 50, 100]
TOPOLOGIES = ["random", "scale_free", "geometric"]
FAILURE_TYPES = ["leader_only", "leader_and_secondary"]
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


def run_failure_scenario(alg, swarm, graph, failure_type, weights):
    """
    Run initial election, apply failure(s), re-elect, return metrics.
    """
    import networkx as nx

    # Step 1: Initial election
    initial = alg.elect(swarm, graph, weights=weights)
    if initial.leader_id is None:
        return None

    # Step 2: Apply failure
    killed = [initial.leader_id]
    swarm.remove_agent(initial.leader_id, graph)

    if failure_type == "leader_and_secondary" and initial.secondary_id is not None:
        swarm.remove_agent(initial.secondary_id, graph)
        killed.append(initial.secondary_id)

    # Step 3: Re-election
    if not swarm.alive_agents():
        return None

    recovery = alg.elect(swarm, graph, weights=weights)

    return {
        "initial_leader": initial.leader_id,
        "initial_lqs": initial.lqs,
        "initial_rounds": initial.rounds_to_converge,
        "killed_agents": len(killed),
        "recovery_leader": recovery.leader_id,
        "recovery_lqs": recovery.lqs,
        "recovery_rounds": recovery.rounds_to_converge,
        "recovery_messages": recovery.messages_sent,
        "recovery_notes": recovery.notes,
    }


def main():
    records = []
    total = len(SWARM_SIZES) * len(TOPOLOGIES) * len(FAILURE_TYPES) * len(ALGORITHMS) * N_TRIALS

    with tqdm(total=total, desc="Exp 3: Failure Recovery") as pbar:
        for n, topo_name, failure_type in product(SWARM_SIZES, TOPOLOGIES, FAILURE_TYPES):
            for trial in range(N_TRIALS):
                seed = hash(("fail", n, topo_name, failure_type, trial)) % (2**31)
                topo_gen = TopologyGenerator(seed=seed)
                base_graph = topo_gen.all_topologies(n)[topo_name]

                for alg in ALGORITHMS:
                    # Work on a fresh copy for each algorithm
                    import networkx as nx
                    graph = nx.Graph(base_graph)
                    swarm = Swarm.from_graph(graph, seed=seed)

                    metrics = run_failure_scenario(alg, swarm, graph, failure_type, WEIGHTS)
                    if metrics is not None:
                        records.append({
                            "algorithm": alg.name,
                            "topology": topo_name,
                            "n_agents": n,
                            "failure_type": failure_type,
                            "trial": trial,
                            **metrics,
                        })
                    pbar.update(1)

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "exp3_failure_recovery.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(df)} records → {out_path}")

    summary = (
        df.groupby(["algorithm", "failure_type"])[["recovery_lqs", "recovery_rounds"]]
        .mean()
        .round(3)
        .reset_index()
    )
    print("\n── Recovery Summary (mean) ──")
    print(summary.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
