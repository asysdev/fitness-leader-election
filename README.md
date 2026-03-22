# Fitness-Based Dynamic Leader Election for Heterogeneous Autonomous Agent Swarms

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python simulation framework for evaluating **fitness-based leader election** in heterogeneous autonomous agent swarms. The algorithm elects the most capable agent as leader by combining three normalized, orthogonal metrics via gossip-based dissemination, achieving provably optimal convergence in O(log N) rounds.

> **Paper:** A. Brahmbhatt, "Fitness-Based Dynamic Leader Election for Heterogeneous Autonomous Agent Swarms: An Information-Theoretic Approach," 2026.
> ORCID: [0000-0001-5649-1265](https://orcid.org/0000-0001-5649-1265)

---

## Overview

In autonomous swarms — drone fleets, robot teams, vehicle platoons, or distributed sensor networks — selecting the right coordinator is critical for mission success. Classical algorithms like Bully and Raft elect leaders based on arbitrary IDs or random timeouts, ignoring agent capabilities entirely.

This framework implements and evaluates a **composite fitness function** that elects the most capable leader by jointly optimizing three dimensions:

| Dimension | What It Captures | How It's Measured |
|-----------|-----------------|-------------------|
| **Information Richness (IR)** | What the agent knows | Normalized Shannon entropy of knowledge categories |
| **Communication Centrality (CC)** | How well it can disseminate decisions | Eigenvector centrality in the network graph |
| **Mission Capacity (MC)** | How long it can sustain leadership | Weighted average of remaining resources |

## Fitness Function

```
F(i) = w₁ · IR(i) + w₂ · CC(i) + w₃ · MC(i)
```

Where:

- **IR(i)** = −(1/log m) Σ p(i,j) log p(i,j) — normalized Shannon entropy over m = 8 knowledge categories
- **CC(i)** = x(i) / max(x) — normalized eigenvector centrality from the adjacency matrix
- **MC(i)** = (bat(i) + stor(i) + sq(i)) / 3 — equal-weighted average of battery, storage, and sensor quality

Default experimental weights: **w₁ = 0.4, w₂ = 0.3, w₃ = 0.3** (the algorithm achieves perfect LQS = 1.000 for all 66 tested weight combinations on the simplex — see paper Section 5.5).

## Key Results

Results from the accompanying paper (30 trials per configuration):

| Metric | Fitness (Ours) | Best Baseline | Bully | Raft |
|--------|---------------|---------------|-------|------|
| **Election Quality (LQS)** | **1.000 ± 0.000** | 0.933 (MostConnected) | 0.740 | 0.802 |
| **Convergence (N=100)** | 8.6 rounds | 2.0 (centralized) | 100.0 | 6.0 |
| **Recovery LQS (single failure)** | **0.998** | 0.958 (MostConnected) | 0.751 | 0.798 |
| **Churn Stability (λ=0.10)** | **0.911** | 0.876 (HighBattery) | 0.798 | 0.799 |
| **Weight Robustness** | **66/66 = 1.000** | — | — | — |

## Repository Structure

```
fitness-leader-election/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── src/
│   ├── agents/
│   │   ├── agent.py              # Agent class with heterogeneous capabilities
│   │   └── swarm.py              # Swarm manager: creation, querying, churn
│   ├── election/
│   │   ├── base.py               # Abstract ElectionAlgorithm interface
│   │   ├── fitness.py            # Fitness-based gossip election (Algorithm 1)
│   │   ├── bully.py              # Bully algorithm [Garcia-Molina 1982]
│   │   ├── random_elect.py       # Uniform random selection
│   │   ├── battery_elect.py      # Highest-battery baseline
│   │   ├── degree_elect.py       # Most-connected (degree centrality) baseline
│   │   └── raft_elect.py         # Simplified Raft-style timeout election
│   ├── network/
│   │   ├── topology.py           # Graph generators: mesh, Erdős–Rényi, BA, geometric
│   │   └── dynamics.py           # Node churn, link failure, partition simulation
│   └── metrics/
│       ├── information_richness.py   # Shannon entropy computation
│       ├── centrality.py             # Eigenvector centrality (NetworkX wrapper)
│       ├── mission_capacity.py       # Resource-weighted capacity score
│       └── leadership_quality.py     # LQS = elected_fitness / max_fitness
├── scripts/
│   ├── run_all_experiments.py        # Run all 5 experiments sequentially
│   ├── run_election_quality.py       # Exp 1: LQS across algorithms & topologies
│   ├── run_convergence.py            # Exp 2: Rounds & messages vs. swarm size
│   ├── run_failure_recovery.py       # Exp 3: Single & cascaded leader failure
│   ├── run_churn.py                  # Exp 4: Stability under node churn
│   ├── run_weight_sensitivity.py     # Exp 5: Sweep over weight simplex
│   ├── generate_plots.py            # Produce publication-quality figures (8 PNGs)
│   └── generate_tables.py           # Produce LaTeX summary tables (5 .tex files)
└── results/
    ├── exp1_election_quality.csv
    ├── exp2_convergence.csv
    ├── exp3_failure_recovery.csv
    ├── exp4_churn.csv
    ├── exp5_weight_sensitivity.csv
    ├── figures/                      # 8 publication-ready PNG figures
    └── tables/                       # 5 LaTeX table files
```

## Requirements

- Python 3.9+
- NetworkX
- NumPy
- SciPy
- Pandas
- Matplotlib

## Installation

```bash
git clone https://github.com/asysdev/fitness-leader-election.git
cd fitness-leader-election
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

Run a single election:

```python
from src.agents.swarm import Swarm
from src.network.topology import TopologyGenerator
from src.election.fitness import FitnessElection

# Create a 20-agent swarm on a random geometric graph
topo = TopologyGenerator(seed=42)
graph = topo.geometric(n=20, radius=0.35)
swarm = Swarm.from_graph(graph, seed=42)

# Run fitness-based election
alg = FitnessElection(weights=(0.4, 0.3, 0.3))
result = alg.elect(swarm, graph)

print(f"Leader: Agent {result.leader_id}")
print(f"Fitness: {result.leader_fitness:.4f}")
print(f"LQS:     {result.lqs:.4f}")
print(f"Rounds:  {result.rounds_to_converge}")
```

## Running Experiments

Reproduce all results from the paper:

```bash
# Run all 5 experiments (saves CSVs to results/)
python scripts/run_all_experiments.py

# Or run individual experiments
python scripts/run_election_quality.py       # Exp 1: ~2,880 trials
python scripts/run_convergence.py            # Exp 2: ~5,040 trials
python scripts/run_failure_recovery.py       # Exp 3: ~4,320 trials
python scripts/run_churn.py                  # Exp 4: ~360 trials
python scripts/run_weight_sensitivity.py     # Exp 5: 66 weight configs

# Generate publication figures and LaTeX tables
python scripts/generate_plots.py
python scripts/generate_tables.py
```

## Experimental Configuration

Parameters used in the paper (see Section 4.5):

| Parameter | Value |
|-----------|-------|
| Swarm sizes (election quality) | N ∈ {10, 20, 50, 100} |
| Swarm sizes (convergence) | N ∈ {5, 10, 20, 30, 50, 75, 100} |
| Topologies | Mesh, Erdős–Rényi, Barabási–Albert (m₀=3), Geometric |
| Trials per configuration | 30 |
| Fitness weights (default) | w₁=0.4, w₂=0.3, w₃=0.3 |
| Knowledge categories (m) | 8 |
| Churn rates | λ ∈ {0.05, 0.10, 0.20} |
| Churn horizon | 50 rounds, N₀=20 |
| Heartbeat timeout (τ) | 3 missed heartbeats |

## Algorithms Compared

| Algorithm | Description | Distributed | Capability-Aware |
|-----------|-------------|:-----------:|:----------------:|
| **Fitness (Ours)** | Gossip-based composite IR + CC + MC | ✅ | ✅ |
| Bully | Highest node ID wins | ✅ | ❌ |
| Random | Uniform random selection | ✅ | ❌ |
| HighBattery | Highest battery level | Centralized | Partial |
| MostConnected | Highest degree centrality | Centralized | Partial |
| Raft | Randomized timeout + majority vote | ✅ | ❌ |

## Metrics

- **Leadership Quality Score (LQS):** `F(elected) / max_i F(i)` ∈ [0, 1]. A value of 1.0 means the omniscient-optimal agent was elected.
- **Convergence Rounds:** Number of gossip rounds until all agents agree on the leader. Scales as O(log N) for the fitness algorithm.
- **Recovery LQS:** Quality of the replacement leader after the current leader fails.
- **Messages Sent:** Total message count during election, capturing communication overhead.
- **Churn Stability:** Mean LQS maintained over time under continuous node arrivals and departures.

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{brahmbhatt2026fitness,
  author  = {Brahmbhatt, Apurv},
  title   = {Fitness-Based Dynamic Leader Election for Heterogeneous
             Autonomous Agent Swarms: An Information-Theoretic Approach},
  year    = {2026},
  note    = {Independent Researcher, Reston, VA, USA},
  url     = {https://github.com/asysdev/fitness-leader-election}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
