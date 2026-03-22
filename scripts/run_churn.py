"""
Experiment 4: High-Churn Environment.

Simulates continuous node join/leave events over N_ROUNDS rounds and
measures leadership stability across algorithms.

Metrics per algorithm:
  - leader_changes:       Number of times the leader changed.
  - avg_lqs_per_round:    Mean LQS across all rounds.
  - re_elections:         Number of re-election events triggered.
  - avg_alive:            Mean number of alive agents per round.
  - lqs_std:              Variance in LQS (stability indicator).

Churn rates tested: low (0.05), medium (0.10), high (0.20).

Saves: results/exp4_churn.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from itertools import product
from pathlib import Path

import networkx as nx
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
from src.network.dynamics import ChurnTimeline, NetworkDynamics
from src.metrics.leadership_quality import compute_all_fitness, leadership_quality_score
from src.election import (
    FitnessElection, BullyElection, RandomElection,
    BatteryElection, DegreeElection, RaftElection,
)

# ── Configuration ─────────────────────────────────────────────────────
N_TRIALS = 10
N_ROUNDS = 50
INITIAL_N = 20
CHURN_RATES = [0.05, 0.10, 0.20]
TOPOLOGIES = ["random", "geometric"]
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


def evaluate_round_lqs(swarm, graph, weights):
    """Compute LQS for the current leader in the current state."""
    alive = swarm.alive_agents()
    if not alive or swarm.leader_id is None:
        return 0.0
    if swarm.leader_id not in {a.agent_id for a in alive}:
        return 0.0
    subgraph = graph.subgraph({a.agent_id for a in alive})
    fm = compute_all_fitness(alive, subgraph, weights=weights)
    return leadership_quality_score(swarm.leader_id, alive, subgraph, weights=weights, fitness_map=fm)


def run_churn_trial(alg, initial_graph, initial_swarm_snapshot, timeline, seed, weights):
    """
    Run one full churn trial.

    We re-create the swarm from snapshot to ensure fair comparison.
    """
    graph = nx.Graph(initial_graph)

    # Re-create swarm from scratch (snapshot = list of agent dicts)
    from src.agents.agent import Agent
    agents = [Agent.from_dict(d) for d in initial_swarm_snapshot]
    swarm = Swarm(agents)

    dyn = NetworkDynamics(
        swarm=swarm,
        graph=graph,
        timeline=timeline,
        algorithm=alg,
        heartbeat_interval=5,
        succession_threshold=3,
        weights=weights,
        seed=seed,
    )

    history = dyn.run()

    leader_changes = sum(1 for s in history if s.leader_changed)
    re_elections = sum(1 for s in history if s.re_election_triggered)
    avg_alive = sum(s.n_alive for s in history) / len(history) if history else 0

    # Compute per-round LQS
    lqs_values = [evaluate_round_lqs(swarm, graph, weights) for _ in history]
    avg_lqs = sum(lqs_values) / len(lqs_values) if lqs_values else 0.0

    import numpy as np
    lqs_std = float(np.std(lqs_values)) if lqs_values else 0.0

    return {
        "leader_changes": leader_changes,
        "re_elections": re_elections,
        "avg_alive": avg_alive,
        "avg_lqs": avg_lqs,
        "lqs_std": lqs_std,
    }


def main():
    records = []
    total = len(CHURN_RATES) * len(TOPOLOGIES) * len(ALGORITHMS) * N_TRIALS

    with tqdm(total=total, desc="Exp 4: Churn") as pbar:
        for churn_rate, topo_name in product(CHURN_RATES, TOPOLOGIES):
            for trial in range(N_TRIALS):
                seed = hash(("churn", churn_rate, topo_name, trial)) % (2**31)
                topo_gen = TopologyGenerator(seed=seed)
                base_graph = topo_gen.all_topologies(INITIAL_N)[topo_name]
                base_swarm = Swarm.from_graph(base_graph, seed=seed)
                snapshot = base_swarm.snapshot()

                # Build the churn timeline once — shared across algorithms
                timeline = ChurnTimeline(
                    n_rounds=N_ROUNDS,
                    initial_agents=base_swarm.alive_ids(),
                    churn_rate=churn_rate,
                    join_prob=0.5,
                    min_alive=3,
                    seed=seed,
                )

                for alg in ALGORITHMS:
                    metrics = run_churn_trial(
                        alg, base_graph, snapshot, timeline, seed=seed + 1, weights=WEIGHTS
                    )
                    records.append({
                        "algorithm": alg.name,
                        "topology": topo_name,
                        "churn_rate": churn_rate,
                        "trial": trial,
                        "n_initial": INITIAL_N,
                        "n_rounds": N_ROUNDS,
                        **metrics,
                    })
                    pbar.update(1)

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "exp4_churn.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(df)} records → {out_path}")

    summary = (
        df.groupby(["algorithm", "churn_rate"])[["avg_lqs", "leader_changes", "lqs_std"]]
        .mean()
        .round(3)
        .reset_index()
    )
    print("\n── Churn Summary (mean across trials) ──")
    print(summary.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
