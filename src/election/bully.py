"""
Bully Algorithm Baseline.

The Bully algorithm (Garcia-Molina 1982) elects the node with the
highest ID as leader. When a node detects a leader failure, it sends
an ELECTION message to all nodes with higher IDs. If no response is
received within a timeout, it declares itself leader.

Simplified simulation:
  - Nodes are sorted by ID descending.
  - The highest-ID alive node is the leader.
  - Message complexity: O(N²) in the worst case.
  - Round complexity: O(N) in the worst case.

Note: The bully algorithm is completely capability-agnostic; it makes
no use of battery, sensors, or knowledge state.
"""

from __future__ import annotations

import time
from typing import Tuple

import networkx as nx

from src.agents.swarm import Swarm
from src.metrics.leadership_quality import compute_all_fitness, leadership_quality_score

from .base import ElectionAlgorithm, ElectionResult


class BullyElection(ElectionAlgorithm):
    """
    Bully algorithm: highest agent_id wins.
    """

    @property
    def name(self) -> str:
        return "Bully"

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

        # Sort by ID descending; highest ID wins
        sorted_agents = sorted(alive_agents, key=lambda a: a.agent_id, reverse=True)
        leader = sorted_agents[0]

        # Message simulation: each lower-ID node sends ELECTION to higher
        # then COORDINATOR is broadcast by the winner
        n = len(alive_agents)
        messages = (n * (n - 1)) // 2 + n  # elections + coordinator broadcast

        # Rounds: in worst case, node 0 starts election → O(N) rounds
        rounds = n

        secondary = sorted_agents[1] if len(sorted_agents) > 1 else None

        swarm.set_leader(leader.agent_id)
        if secondary:
            swarm.set_secondary(secondary.agent_id)

        # Omniscient LQS evaluation
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
