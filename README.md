# Fitness-Based Leader Election for Drone Swarms

A Python simulation framework implementing and evaluating the **Fitness-Based Leader Election** algorithm for autonomous drone swarms, as specified in the patent specification. The algorithm elects the most capable agent as leader by combining three normalized metrics: Information Richness (IR), Communication Centrality (CC), and Mission Capacity (MC).

## Overview

In a drone swarm, selecting the right leader is critical for mission success. This framework:

- Implements the fitness-based election algorithm (Algorithm 1 from the patent)
- Provides five baseline comparisons: Bully, Random, Highest-Battery, Most-Connected, and Raft
- Simulates realistic network topologies and dynamics (churn, partitions, failures)
- Evaluates leadership quality, convergence speed, failure recovery, and weight sensitivity

## Fitness Function

```
F(a) = w₁·IR(a) + w₂·CC(a) + w₃·MC(a)

where default weights: w₁=0.40, w₂=0.35, w₃=0.25

IR  = Information Richness   (Shannon entropy of knowledge topics)
CC  = Communication Centrality (eigenvector centrality in network)
MC  = Mission Capacity       (0.4·battery + 0.3·sensors + 0.2·storage + 0.1·payload)
```

## Repository Structure

```
fitness-leader-election/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── agents/
│   │   ├── agent.py          # Agent class with capabilities
│   │   └── swarm.py          # Swarm class managing agent collection
│   ├── election/
│   │   ├── base.py           # Abstract ElectionAlgorithm interface
│   │   ├── fitness.py        # Fitness-based election (Algorithm 1)
│   │   ├── bully.py          # Bully algorithm baseline
│   │   ├── random_elect.py   # Random selection baseline
│   │   ├── battery_elect.py  # Highest-battery baseline
│   │   ├── degree_elect.py   # Most-connected baseline
│   │   └── raft_elect.py     # Simplified Raft baseline
│   ├── network/
│   │   ├── topology.py       # Graph generation (mesh, random, scale-free, geometric)
│   │   └── dynamics.py       # Node join/leave, link failure, partition simulation
│   └── metrics/
│       ├── information_richness.py   # Shannon entropy metric
│       ├── centrality.py             # Eigenvector centrality wrapper
│       ├── mission_capacity.py       # Weighted resource score
│       └── leadership_quality.py    # LQS evaluation metric
├── scripts/
│   ├── run_all_experiments.py        # Run all 5 experiments sequentially
│   ├── run_election_quality.py       # Experiment 1: LQS across algorithms
│   ├── run_convergence.py            # Experiment 2: Rounds to convergence
│   ├── run_failure_recovery.py       # Experiment 3: Recovery after leader failure
│   ├── run_churn.py                  # Experiment 4: High-churn environment
│   ├── run_weight_sensitivity.py     # Experiment 5: Weight sensitivity analysis
│   ├── generate_plots.py             # Produce all publication-quality figures
│   └── generate_tables.py            # Produce LaTeX summary tables
└── results/                          # Output CSVs, PNGs, and LaTeX tables
```

## Installation

```bash
git clone https://github.com/yourname/fitness-leader-election
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
alg = FitnessElection()
result = alg.elect(swarm, graph)

print(f"Leader: Agent {result.leader_id}")
print(f"Fitness: {result.leader_fitness:.4f}")
print(f"LQS:     {result.lqs:.4f}")
print(f"Rounds:  {result.rounds_to_converge}")
```

## Running Experiments

```bash
# Run all experiments (saves CSVs to results/)
python scripts/run_all_experiments.py

# Run individual experiments
python scripts/run_election_quality.py
python scripts/run_convergence.py
python scripts/run_failure_recovery.py
python scripts/run_churn.py
python scripts/run_weight_sensitivity.py

# Generate figures and tables from saved results
python scripts/generate_plots.py
python scripts/generate_tables.py
```

## Algorithms Compared

| Algorithm | Description | Capability-Aware |
|-----------|-------------|-----------------|
| **Fitness** | Weighted IR + CC + MC | ✅ |
| Bully | Highest node ID wins | ❌ |
| Random | Uniform random selection | ❌ |
| Battery | Highest battery wins | Partial |
| Degree | Most connections wins | Partial |
| Raft | Randomized timeout + majority vote | ❌ |

## Metrics

- **Leadership Quality Score (LQS)**: `elected_fitness / max_possible_fitness` ∈ [0, 1]
- **Rounds to Converge**: Number of communication rounds until stable leader
- **Recovery Time**: Rounds from leader failure to new stable leader
- **Churn Stability**: LQS maintained under continuous node join/leave events

## Citation

If you use this simulation framework, please cite:

```bibtex
@misc{fitness_leader_election_2026,
  title  = {Fitness-Based Leader Election for Autonomous Drone Swarms},
  note   = {Patent Specification Implementation},
  year   = {2026}
}
```
