"""
Most-Connected (Highest Degree) Election Baseline.

Elects the agent with the most communication links (highest degree)
as leader. This captures communication centrality but ignores
agent capabilities (battery, sensors) and knowledge richness.

Message complexity: O(E) — topology discovery.
Round complexity: O(diameter) for distributed computation, O(1) centralized.
"""

from __future__ import annotations

import time
from typing import Tuple

import networkx as nx

from src.agents.swarm import Swarm
from src.metrics.leadership_quality import compute_all_fitness, leadership_quality_score

from .base import ElectionAlgorithm, ElectionResult


class DegreeElection(ElectionAlgorithm):
    """
    Degree-based election: most connected node wins.
    Tie-breaking: lower agent_id wins.
    """

    @property
    def name(self) -> str:
        return "MostConnected"

    def elect(
        self,
        swarm: Swarm,
        graph: nx.Graph,
        weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
    ) -> ElectionResult:
        t0 = time.perf_counter()
        swarm.reset_election_state()

        alive_agents = swarm.alive_agents()
        alive_ids = set(a.agent_id for a in alive_agents)
        if not alive_agents:
            return ElectionResult(algorithm=self.name, notes="No alive agents")

        subgraph = graph.subgraph(alive_ids)
        degree_map = dict(subgraph.degree())

        # Sort by degree desc, then agent_id asc
        sorted_agents = sorted(
            alive_agents,
            key=lambda a: (-degree_map.get(a.agent_id, 0), a.agent_id)
        )
        leader = sorted_agents[0]
        secondary = sorted_agents[1] if len(sorted_agents) > 1 else None

        swarm.set_leader(leader.agent_id)
        if secondary:
            swarm.set_secondary(secondary.agent_id)

        messages = len(subgraph.edges()) * 2  # topology discovery
        rounds = 2

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
