"""
Abstract base class for all leader election algorithms.

Every algorithm must implement:
    elect(swarm, graph) -> ElectionResult

and optionally:
    handle_failure(swarm, graph) -> ElectionResult
    handle_churn(swarm, graph, events) -> ElectionResult
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from src.agents.swarm import Swarm


@dataclass
class ElectionResult:
    """
    Container for the outcome of a single leader election run.

    Attributes
    ----------
    algorithm : str
        Name of the algorithm that produced this result.
    leader_id : int or None
        Agent ID of the elected leader. None if election failed.
    secondary_id : int or None
        Agent ID of the elected secondary leader.
    leader_fitness : float
        Fitness score of the elected leader (using patent weights).
    lqs : float
        Leadership Quality Score ∈ [0, 1].
    rounds_to_converge : int
        Number of communication rounds until consensus.
    messages_sent : int
        Total number of messages exchanged during election.
    election_time_ms : float
        Wall-clock time for the election in milliseconds.
    fitness_map : dict[int, float]
        Fitness scores for all alive agents (omniscient evaluation).
    notes : str
        Optional diagnostic notes.
    """
    algorithm: str
    leader_id: Optional[int] = None
    secondary_id: Optional[int] = None
    leader_fitness: float = 0.0
    lqs: float = 0.0
    rounds_to_converge: int = 0
    messages_sent: int = 0
    election_time_ms: float = 0.0
    fitness_map: Dict[int, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "leader_id": self.leader_id,
            "secondary_id": self.secondary_id,
            "leader_fitness": self.leader_fitness,
            "lqs": self.lqs,
            "rounds_to_converge": self.rounds_to_converge,
            "messages_sent": self.messages_sent,
            "election_time_ms": self.election_time_ms,
            "notes": self.notes,
        }


class ElectionAlgorithm(ABC):
    """
    Abstract interface for a leader election algorithm.

    Subclasses implement `elect()` and optionally override
    `handle_failure()` and `handle_churn()`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name."""
        ...

    @abstractmethod
    def elect(
        self,
        swarm: Swarm,
        graph: nx.Graph,
        weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
    ) -> ElectionResult:
        """
        Run a full leader election.

        Parameters
        ----------
        swarm : Swarm
            The current swarm of agents.
        graph : nx.Graph
            Network topology (nodes = agent IDs, edges = communication links).
        weights : (w_IR, w_CC, w_MC)
            Fitness weights used for LQS evaluation.

        Returns
        -------
        ElectionResult
        """
        ...

    def handle_failure(
        self,
        swarm: Swarm,
        graph: nx.Graph,
        failed_id: int,
        weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
    ) -> ElectionResult:
        """
        Handle a leader failure event.

        Default implementation: remove the failed agent and re-run elect().
        Override for algorithms with built-in succession mechanisms.
        """
        swarm.remove_agent(failed_id, graph)
        return self.elect(swarm, graph, weights=weights)

    def handle_churn(
        self,
        swarm: Swarm,
        graph: nx.Graph,
        events: List[Tuple[str, int]],
        weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
    ) -> ElectionResult:
        """
        Process a sequence of churn events and re-elect.

        Parameters
        ----------
        events : list of ("join"|"leave", agent_id)
        """
        return self.elect(swarm, graph, weights=weights)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _compute_lqs_and_fitness(
        self,
        elected_id: Optional[int],
        swarm: Swarm,
        graph: nx.Graph,
        weights: Tuple[float, float, float],
    ) -> Tuple[float, float, Dict[int, float]]:
        """
        Compute LQS, elected fitness, and full fitness map.

        Returns (leader_fitness, lqs, fitness_map).
        """
        from src.metrics.leadership_quality import compute_all_fitness, leadership_quality_score

        alive_agents = swarm.alive_agents()
        fitness_map = compute_all_fitness(alive_agents, graph, weights=weights)

        if elected_id is None or elected_id not in fitness_map:
            return 0.0, 0.0, fitness_map

        leader_fitness = fitness_map[elected_id]
        lqs = leadership_quality_score(
            elected_id, alive_agents, graph, weights=weights, fitness_map=fitness_map
        )
        return leader_fitness, lqs, fitness_map

    def _timed_elect(
        self,
        swarm: Swarm,
        graph: nx.Graph,
        weights: Tuple[float, float, float],
    ) -> ElectionResult:
        """Wrapper that times the election."""
        t0 = time.perf_counter()
        result = self.elect(swarm, graph, weights=weights)
        result.election_time_ms = (time.perf_counter() - t0) * 1000
        return result
