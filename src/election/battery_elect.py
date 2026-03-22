"""
Highest-Battery Election Baseline.

Elects the agent with the highest battery level as leader.
This is a common heuristic in energy-constrained drone swarms but
ignores communication position and knowledge richness.

Expected behavior:
  - Good for mission duration (battery-dominated tasks).
  - Poor if the highest-battery agent is poorly connected or knows little.
  - LQS will be high when w_MC dominates, lower when w_IR or w_CC matters.

Message complexity: O(N) — collect battery levels, broadcast winner.
Round complexity: O(log N) for tree-based aggregation, simulated as O(1).
"""

from __future__ import annotations

import time
from typing import Tuple

import networkx as nx

from src.agents.swarm import Swarm
from src.metrics.leadership_quality import compute_all_fitness, leadership_quality_score

from .base import ElectionAlgorithm, ElectionResult


class BatteryElection(ElectionAlgorithm):
    """
    Battery-based election: highest battery level wins.
    Tie-breaking: lower agent_id wins.
    """

    @property
    def name(self) -> str:
        return "HighBattery"

    def elect(
        self,
        swarm: Swarm,
        graph: nx.Graph,
        weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
    ) -> ElectionResult:
        t0 = time.perf_counter()
        swarm.reset_election_state()

        alive_agents = swarm.alive_agents()
        if not alive_agents:
            return ElectionResult(algorithm=self.name, notes="No alive agents")

        # Sort by battery desc, then agent_id asc for tie-breaking
        sorted_agents = sorted(
            alive_agents, key=lambda a: (-a.battery, a.agent_id)
        )
        leader = sorted_agents[0]
        secondary = sorted_agents[1] if len(sorted_agents) > 1 else None

        swarm.set_leader(leader.agent_id)
        if secondary:
            swarm.set_secondary(secondary.agent_id)

        n = len(alive_agents)
        messages = n      # each broadcasts its battery
        rounds = 2        # 1 broadcast + 1 announce

        subgraph = graph.subgraph(set(a.agent_id for a in alive_agents))
        fitness_map = compute_all_fitness(alive_agents, subgraph, weights=weights)
        lqs = leadership_quality_score(
            leader.agent_id, alive_agents, subgraph,
            weights=weights, fitness_map=fitness_map
        )

        return ElectionResult(
            algorithm=self.name,
            leader_id=leader.agent_id,
            secondary_id=secondary.agent_id if secondary else None,
            leader_fitness=fitness_map.get(leader.agent_id, 0.0),
            lqs=lqs,
            rounds_to_converge=rounds,
            messages_sent=messages,
            election_time_ms=(time.perf_counter() - t0) * 1000,
            fitness_map=fitness_map,
        )
