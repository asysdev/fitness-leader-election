"""
Fitness-Based Leader Election — Algorithm 1 from the patent specification.

Algorithm overview:
  Phase 1 – Broadcast: Each agent broadcasts its capability vector to
             all neighbors. (1 round, O(E) messages)
  Phase 2 – Local Fitness Computation: Each agent computes the fitness
             of itself and all its neighbors.
  Phase 3 – Candidate Propagation: Each agent forwards the highest-fitness
             candidate it knows about. This repeats for diameter(G) rounds
             until every agent has seen the global maximum. (O(D) rounds)
  Phase 4 – Consensus: The agent with the globally highest fitness is
             declared leader. Ties broken by agent_id (lower wins).
  Phase 5 – Secondary Election: Repeat Phase 3–4 excluding the leader
             to elect a secondary.
  Phase 6 – Heartbeat Setup: Leader begins emitting heartbeats; secondary
             begins monitoring.

Message complexity: O(E + N·D)  where D = graph diameter
Round complexity:   O(D)
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import networkx as nx

from src.agents.swarm import Swarm
from src.metrics.centrality import eigenvector_centrality_map
from src.metrics.information_richness import information_richness
from src.metrics.mission_capacity import mission_capacity
from src.metrics.leadership_quality import compute_all_fitness, leadership_quality_score

from .base import ElectionAlgorithm, ElectionResult


class FitnessElection(ElectionAlgorithm):
    """
    Patent-specified fitness-based leader election.

    Parameters
    ----------
    heartbeat_interval : int
        Rounds between leader heartbeats.
    succession_threshold : int
        Missed heartbeats before secondary promotes itself.
    """

    def __init__(self, heartbeat_interval: int = 5, succession_threshold: int = 3):
        self.heartbeat_interval = heartbeat_interval
        self.succession_threshold = succession_threshold

    @property
    def name(self) -> str:
        return "Fitness"

    # ------------------------------------------------------------------
    # Core fitness function — MUST match patent specification exactly
    # ------------------------------------------------------------------

    @staticmethod
    def compute_fitness(
        agent,
        graph: nx.Graph,
        weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
        centrality_cache: Optional[Dict] = None,
    ) -> float:
        """
        Compute composite fitness score (Algorithm 1, line 3).

            F(a) = w₁·IR(a) + w₂·CC(a) + w₃·MC(a)

        Parameters
        ----------
        agent : Agent
        graph : nx.Graph
        weights : (w_IR, w_CC, w_MC), default (0.40, 0.35, 0.25)
        centrality_cache : dict or None

        Returns
        -------
        float ∈ [0, 1]
        """
        w_ir, w_cc, w_mc = weights

        ir = information_richness(agent)

        if centrality_cache is not None:
            cc = centrality_cache.get(agent.agent_id, 0.0)
        else:
            centrality_map = eigenvector_centrality_map(graph)
            cc = centrality_map.get(agent.agent_id, 0.0)

        mc = mission_capacity(agent)

        return w_ir * ir + w_cc * cc + w_mc * mc

    # ------------------------------------------------------------------
    # Election protocol
    # ------------------------------------------------------------------

    def elect(
        self,
        swarm: Swarm,
        graph: nx.Graph,
        weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
    ) -> ElectionResult:
        """Run the full fitness-based election protocol."""
        t0 = time.perf_counter()
        swarm.reset_election_state()

        alive_agents = swarm.alive_agents()
        alive_ids = set(a.agent_id for a in alive_agents)

        if not alive_ids:
            return ElectionResult(algorithm=self.name, notes="No alive agents")

        if len(alive_ids) == 1:
            sole = alive_agents[0]
            swarm.set_leader(sole.agent_id)
            return ElectionResult(
                algorithm=self.name,
                leader_id=sole.agent_id,
                leader_fitness=self.compute_fitness(sole, graph, weights),
                lqs=1.0,
                rounds_to_converge=1,
                messages_sent=0,
                election_time_ms=(time.perf_counter() - t0) * 1000,
            )

        # ----------------------------------------------------------
        # Phase 1: Broadcast capability vectors (1 round)
        # Message count = number of directed edges for alive nodes
        # ----------------------------------------------------------
        messages = 0
        for u in alive_ids:
            for v in graph.neighbors(u):
                if v in alive_ids:
                    messages += 1  # u → v: capability broadcast

        # ----------------------------------------------------------
        # Phase 2: Compute fitness for all agents (centrality once)
        # ----------------------------------------------------------
        subgraph = graph.subgraph(alive_ids)
        centrality_cache = eigenvector_centrality_map(subgraph)

        local_fitness: Dict[int, float] = {}
        for agent in alive_agents:
            local_fitness[agent.agent_id] = self.compute_fitness(
                agent, subgraph, weights=weights, centrality_cache=centrality_cache
            )

        # ----------------------------------------------------------
        # Phase 3: Candidate propagation — flood the best candidate
        # Each round: each node forwards best known candidate to neighbors
        # Runs for graph diameter rounds to guarantee convergence
        # ----------------------------------------------------------
        if len(subgraph.nodes()) > 1:
            try:
                diameter = nx.diameter(subgraph) if nx.is_connected(subgraph) else len(subgraph)
            except nx.NetworkXError:
                diameter = len(subgraph)
        else:
            diameter = 1

        # Each agent starts with its own best candidate
        best_candidate: Dict[int, Tuple[float, int]] = {
            aid: (local_fitness[aid], aid) for aid in alive_ids
        }

        rounds = 1  # Phase 1 counts as round 1
        for _ in range(diameter):
            rounds += 1
            updated = False
            new_best = dict(best_candidate)
            for aid in alive_ids:
                my_fitness, my_id = best_candidate[aid]
                for neighbor in subgraph.neighbors(aid):
                    if neighbor in alive_ids:
                        messages += 1  # propagation message
                        n_fitness, n_id = best_candidate[neighbor]
                        # Compare: higher fitness wins; tie-break: lower ID
                        if (n_fitness > my_fitness) or (
                            n_fitness == my_fitness and n_id < my_id
                        ):
                            new_best[aid] = (n_fitness, n_id)
                            my_fitness, my_id = new_best[aid]
                            updated = True
            best_candidate = new_best
            if not updated:
                break  # Early termination

        # ----------------------------------------------------------
        # Phase 4: Consensus — majority check
        # Leader is the candidate that most agents converged to
        # ----------------------------------------------------------
        candidate_votes: Dict[int, int] = {}
        for fitness_val, candidate_id in best_candidate.values():
            candidate_votes[candidate_id] = candidate_votes.get(candidate_id, 0) + 1

        leader_id = max(candidate_votes, key=lambda cid: (candidate_votes[cid], -cid))
        swarm.set_leader(leader_id)

        # ----------------------------------------------------------
        # Phase 5: Secondary election (same protocol, exclude leader)
        # ----------------------------------------------------------
        secondary_candidates = {aid: local_fitness[aid] for aid in alive_ids if aid != leader_id}
        secondary_id: Optional[int] = None

        if secondary_candidates:
            # Simple: pick highest fitness among non-leaders
            secondary_id = max(secondary_candidates, key=lambda k: (secondary_candidates[k], -k))
            swarm.set_secondary(secondary_id)
            rounds += 1  # Secondary election round

        # ----------------------------------------------------------
        # Phase 6: Heartbeat setup (counted as 1 round)
        # ----------------------------------------------------------
        rounds += 1

        # ----------------------------------------------------------
        # Evaluate LQS (omniscient view)
        # ----------------------------------------------------------
        fitness_map = compute_all_fitness(alive_agents, subgraph, weights=weights)
        lqs = leadership_quality_score(
            leader_id, alive_agents, subgraph, weights=weights, fitness_map=fitness_map
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return ElectionResult(
            algorithm=self.name,
            leader_id=leader_id,
            secondary_id=secondary_id,
            leader_fitness=local_fitness[leader_id],
            lqs=lqs,
            rounds_to_converge=rounds,
            messages_sent=messages,
            election_time_ms=elapsed_ms,
            fitness_map=fitness_map,
        )

    # ------------------------------------------------------------------
    # Failure handling with succession
    # ------------------------------------------------------------------

    def handle_failure(
        self,
        swarm: Swarm,
        graph: nx.Graph,
        failed_id: int,
        weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
    ) -> ElectionResult:
        """
        Handle leader failure using the heartbeat/succession mechanism.

        If the secondary is still alive, it promotes itself immediately
        (fast path, O(1) rounds) and then elects a new secondary.
        Otherwise, a full re-election is triggered.
        """
        # Kill the failed agent
        swarm.remove_agent(failed_id, graph)

        secondary = swarm.get_secondary()
        if secondary is not None and secondary.alive:
            # Fast succession
            old_secondary_id = secondary.agent_id
            swarm.set_leader(old_secondary_id)

            # Elect new secondary from remaining agents
            alive_agents = swarm.alive_agents()
            remaining = [a for a in alive_agents if a.agent_id != old_secondary_id]

            new_secondary_id = None
            if remaining:
                subgraph = graph.subgraph(set(a.agent_id for a in alive_agents))
                fitness_map = compute_all_fitness(remaining, subgraph, weights=weights)
                new_secondary_id = max(fitness_map, key=fitness_map.__getitem__)
                swarm.set_secondary(new_secondary_id)

            # Evaluate LQS for new leader
            alive_agents = swarm.alive_agents()
            subgraph = graph.subgraph(set(a.agent_id for a in alive_agents))
            fitness_map = compute_all_fitness(alive_agents, subgraph, weights=weights)
            lqs = leadership_quality_score(
                old_secondary_id, alive_agents, subgraph,
                weights=weights, fitness_map=fitness_map
            )

            return ElectionResult(
                algorithm=self.name,
                leader_id=old_secondary_id,
                secondary_id=new_secondary_id,
                leader_fitness=fitness_map.get(old_secondary_id, 0.0),
                lqs=lqs,
                rounds_to_converge=2,  # 1 succession + 1 new secondary
                messages_sent=len(alive_agents),  # notify all
                fitness_map=fitness_map,
                notes=f"Succession: {failed_id} → {old_secondary_id}",
            )

        # No alive secondary → full re-election
        result = self.elect(swarm, graph, weights=weights)
        result.notes = f"Full re-election after failure of agent {failed_id}"
        return result
