"""
Simplified Raft Election Baseline.

Raft is a consensus algorithm designed for replicated log management.
Its leader election uses randomized timeouts to avoid split votes:

  1. Each node starts as a Follower with a random election timeout
     drawn uniformly from [150ms, 300ms] (simulated as rounds).
  2. The node whose timeout fires first becomes a Candidate.
  3. The Candidate increments its term and sends RequestVote RPCs to all peers.
  4. Nodes grant their vote to the first Candidate they hear from in a term.
  5. The Candidate that collects a majority (⌈N/2⌉ + 1) wins and sends
     AppendEntries (heartbeat) to establish leadership.

Simulation notes:
  - Timeout is modeled as a discrete integer in [3, 6] rounds.
  - Nodes process one message per round (sequential simulation).
  - If a split vote occurs (no majority), timeouts are re-randomized
    and the process repeats (new term).
  - Raft does NOT consider agent capabilities → LQS is determined
    purely by which agent happened to timeout first.

This baseline demonstrates that pure consensus mechanisms without
capability-awareness produce lower LQS than the fitness algorithm.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from src.agents.swarm import Swarm
from src.metrics.leadership_quality import compute_all_fitness, leadership_quality_score

from .base import ElectionAlgorithm, ElectionResult


# Simulated timeout range in discrete rounds (150–300ms at 50ms/round)
_TIMEOUT_MIN = 3
_TIMEOUT_MAX = 6
_MAX_TERMS = 20  # Safety limit to prevent infinite loops


class RaftElection(ElectionAlgorithm):
    """
    Simplified Raft leader election.

    Parameters
    ----------
    seed : int or None
        RNG seed for reproducible timeout sampling.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "Raft"

    def elect(
        self,
        swarm: Swarm,
        graph: nx.Graph,
        weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
    ) -> ElectionResult:
        t0 = time.perf_counter()
        swarm.reset_election_state()

        alive_agents = swarm.alive_agents()
        alive_ids = [a.agent_id for a in alive_agents]

        if not alive_ids:
            return ElectionResult(algorithm=self.name, notes="No alive agents")

        if len(alive_ids) == 1:
            leader_id = alive_ids[0]
            swarm.set_leader(leader_id)
            subgraph = graph.subgraph(set(alive_ids))
            fitness_map = compute_all_fitness(alive_agents, subgraph, weights=weights)
            return ElectionResult(
                algorithm=self.name,
                leader_id=leader_id,
                leader_fitness=fitness_map.get(leader_id, 0.0),
                lqs=1.0,
                rounds_to_converge=1,
                messages_sent=0,
                election_time_ms=(time.perf_counter() - t0) * 1000,
                fitness_map=fitness_map,
            )

        majority = len(alive_ids) // 2 + 1
        subgraph = graph.subgraph(set(alive_ids))

        total_rounds = 0
        total_messages = 0
        term = 0
        leader_id: Optional[int] = None

        for _attempt in range(_MAX_TERMS):
            term += 1

            # Assign random election timeouts
            timeouts: Dict[int, int] = {
                aid: self._rng.randint(_TIMEOUT_MIN, _TIMEOUT_MAX)
                for aid in alive_ids
            }

            # Find the candidate with the earliest timeout
            # Ties broken by lower agent_id
            min_timeout = min(timeouts.values())
            candidates = sorted(
                [aid for aid, t in timeouts.items() if t == min_timeout]
            )
            candidate_id = candidates[0]

            total_rounds += min_timeout  # rounds until first candidate fires

            # Candidate sends RequestVote to all peers (via graph neighbors only
            # for realism — can reach all via multi-hop, but we count direct edges)
            vote_requests = len(list(subgraph.neighbors(candidate_id)))
            total_messages += vote_requests

            # Nodes grant votes: each node votes for the first candidate
            # it hears from (simplified: all hear from this candidate since
            # they haven't voted yet this term)
            votes_received: int = 1  # self-vote
            voted: Set[int] = {candidate_id}

            for aid in alive_ids:
                if aid == candidate_id:
                    continue
                # In a real network, the vote request propagates via graph.
                # We simulate: if the candidate can reach the node (connected
                # component), the node votes; otherwise it doesn't.
                if nx.has_path(subgraph, candidate_id, aid):
                    # Node hasn't voted this term
                    votes_received += 1
                    voted.add(aid)
                    total_messages += 1  # vote reply

            total_rounds += 2  # vote collection + announcement

            if votes_received >= majority:
                leader_id = candidate_id
                total_messages += len(alive_ids)  # AppendEntries broadcast
                total_rounds += 1
                break
            # Split vote — retry with new term

        # Update swarm state
        if leader_id is not None:
            swarm.set_leader(leader_id)
            # Secondary: random from remaining (Raft doesn't define a deputy)
            remaining = [aid for aid in alive_ids if aid != leader_id]
            secondary_id = self._rng.choice(remaining) if remaining else None
            if secondary_id is not None:
                swarm.set_secondary(secondary_id)
        else:
            secondary_id = None

        # Omniscient LQS evaluation
        fitness_map = compute_all_fitness(alive_agents, subgraph, weights=weights)
        lqs = leadership_quality_score(
            leader_id, alive_agents, subgraph,
            weights=weights, fitness_map=fitness_map
        ) if leader_id is not None else 0.0

        notes = f"term={term}" if leader_id else f"No consensus after {_MAX_TERMS} terms"

        return ElectionResult(
            algorithm=self.name,
            leader_id=leader_id,
            secondary_id=secondary_id,
            leader_fitness=fitness_map.get(leader_id, 0.0) if leader_id else 0.0,
            lqs=lqs,
            rounds_to_converge=total_rounds,
            messages_sent=total_messages,
            election_time_ms=(time.perf_counter() - t0) * 1000,
            fitness_map=fitness_map,
            notes=notes,
        )
