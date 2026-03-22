"""
Random Election Baseline.

Selects a leader uniformly at random from all alive agents.
This provides the lower bound on expected LQS for comparison.

Expected LQS = 1/N (for uniform random agent capabilities).
Message complexity: O(N) — one broadcast to announce the winner.
Round complexity: O(1).
"""

from __future__ import annotations

import random
import time
from typing import Optional, Tuple

import networkx as nx

from src.agents.swarm import Swarm
from src.metrics.leadership_quality import compute_all_fitness, leadership_quality_score

from .base import ElectionAlgorithm, ElectionResult


class RandomElection(ElectionAlgorithm):
    """
    Random election: uniform random selection among alive agents.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "Random"

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

        # Random selection
        leader = self._rng.choice(alive_agents)

        # Secondary: random from remaining
        remaining = [a for a in alive_agents if a.agent_id != leader.agent_id]
        secondary = self._rng.choice(remaining) if remaining else None

        swarm.set_leader(leader.agent_id)
        if secondary:
            swarm.set_secondary(secondary.agent_id)

        # O(N) messages to broadcast winner
        messages = len(alive_agents)

        # Omniscient LQS
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
            rounds_to_converge=1,
            messages_sent=messages,
            election_time_ms=(time.perf_counter() - t0) * 1000,
            fitness_map=fitness_map,
        )
