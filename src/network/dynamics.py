"""
Network dynamics simulation.

Provides:
  - ChurnTimeline:   Generate a reproducible sequence of join/leave events.
  - NetworkDynamics: Apply events to a live (swarm, graph) pair and track metrics.
  - LinkFailure:     Randomly disable/re-enable edges.
  - Partition:       Split graph into isolated components and heal.

All operations are applied in-place to the nx.Graph for efficiency;
callers should copy the graph if they need the original.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from src.agents.agent import Agent
from src.agents.swarm import Swarm


# ──────────────────────────────────────────────────────────────────────
# Churn event representation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ChurnEvent:
    """
    A single churn event in the simulation timeline.

    Attributes
    ----------
    round : int
        The simulation round at which this event fires.
    action : str
        "join" or "leave".
    agent_id : int
        The agent involved in the event.
    """
    round: int
    action: str   # "join" | "leave"
    agent_id: int


# ──────────────────────────────────────────────────────────────────────
# Churn timeline generator
# ──────────────────────────────────────────────────────────────────────

class ChurnTimeline:
    """
    Generate a reproducible timeline of node join/leave events.

    The timeline guarantees:
      - At least `min_alive` agents are alive at any point.
      - Leaves are only scheduled for agents currently alive.
      - Joins introduce agents with IDs > existing max.

    Parameters
    ----------
    n_rounds : int
        Total number of simulation rounds.
    initial_agents : list[int]
        Agent IDs present at round 0.
    churn_rate : float
        Expected fraction of agents that change state per round.
        E.g. 0.1 → expect 10% of N agents to join or leave each round.
    join_prob : float
        Probability that a churn event is a join (vs leave).
    min_alive : int
        Minimum number of alive agents to maintain at all times.
    seed : int or None
    """

    def __init__(
        self,
        n_rounds: int,
        initial_agents: List[int],
        churn_rate: float = 0.1,
        join_prob: float = 0.5,
        min_alive: int = 3,
        seed: Optional[int] = None,
    ):
        self.n_rounds = n_rounds
        self.churn_rate = churn_rate
        self.join_prob = join_prob
        self.min_alive = min_alive
        self._rng = random.Random(seed)

        self.events: List[ChurnEvent] = []
        self._generate(initial_agents)

    def _generate(self, initial_agents: List[int]) -> None:
        """Generate the event timeline."""
        alive: List[int] = list(initial_agents)
        next_id = max(initial_agents) + 1 if initial_agents else 0

        for r in range(self.n_rounds):
            # Expected number of events this round
            n_events = max(0, int(self._rng.gauss(len(alive) * self.churn_rate, 0.5)))

            for _ in range(n_events):
                if self._rng.random() < self.join_prob:
                    # Join event
                    self.events.append(ChurnEvent(round=r, action="join", agent_id=next_id))
                    alive.append(next_id)
                    next_id += 1
                else:
                    # Leave event — only if enough agents remain
                    if len(alive) > self.min_alive:
                        victim = self._rng.choice(alive)
                        self.events.append(ChurnEvent(round=r, action="leave", agent_id=victim))
                        alive.remove(victim)

    def events_for_round(self, r: int) -> List[ChurnEvent]:
        """Return all events scheduled for round r."""
        return [e for e in self.events if e.round == r]

    def summary(self) -> Dict:
        joins = sum(1 for e in self.events if e.action == "join")
        leaves = sum(1 for e in self.events if e.action == "leave")
        return {
            "n_rounds": self.n_rounds,
            "total_events": len(self.events),
            "joins": joins,
            "leaves": leaves,
            "events_per_round": len(self.events) / self.n_rounds if self.n_rounds else 0,
        }


# ──────────────────────────────────────────────────────────────────────
# Network dynamics engine
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RoundState:
    """State snapshot after processing one simulation round."""
    round: int
    n_alive: int
    leader_id: Optional[int]
    secondary_id: Optional[int]
    events_this_round: List[ChurnEvent]
    re_election_triggered: bool
    leader_changed: bool


class NetworkDynamics:
    """
    Apply churn events and heartbeat monitoring to a live swarm+graph.

    Usage
    -----
    dyn = NetworkDynamics(swarm, graph, timeline, election_algorithm)
    for state in dyn.run():
        print(state)

    Parameters
    ----------
    swarm : Swarm
    graph : nx.Graph
        Modified in-place as nodes join/leave.
    timeline : ChurnTimeline
    algorithm : ElectionAlgorithm
        Used to re-elect when triggered.
    heartbeat_interval : int
        Rounds between leader heartbeats.
    succession_threshold : int
        Missed heartbeats before secondary promotes itself.
    weights : tuple
        Fitness weights for evaluation.
    seed : int or None
    """

    def __init__(
        self,
        swarm: Swarm,
        graph: nx.Graph,
        timeline: ChurnTimeline,
        algorithm,
        heartbeat_interval: int = 5,
        succession_threshold: int = 3,
        weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
        seed: Optional[int] = None,
    ):
        self.swarm = swarm
        self.graph = graph
        self.timeline = timeline
        self.algorithm = algorithm
        self.heartbeat_interval = heartbeat_interval
        self.succession_threshold = succession_threshold
        self.weights = weights
        self._rng = random.Random(seed)

        self.history: List[RoundState] = []

    def run(self) -> List[RoundState]:
        """
        Execute all rounds in the timeline.

        Returns
        -------
        list[RoundState]: one entry per round.
        """
        # Initial election
        result = self.algorithm.elect(self.swarm, self.graph, weights=self.weights)
        self.swarm.set_leader(result.leader_id)
        if result.secondary_id:
            self.swarm.set_secondary(result.secondary_id)

        for r in range(self.timeline.n_rounds):
            events = self.timeline.events_for_round(r)
            re_elected = False
            prev_leader = self.swarm.leader_id

            # Apply churn events
            for event in events:
                if event.action == "join":
                    self.swarm.add_agent(self.graph, agent_id=event.agent_id, rng=self._rng)
                elif event.action == "leave":
                    self.swarm.remove_agent(event.agent_id, self.graph)
                    if event.agent_id == self.swarm.leader_id:
                        re_elected = True
                    if event.agent_id == self.swarm.secondary_id:
                        # Secondary departed — need a new secondary after election
                        self.swarm.secondary_id = None

            # Heartbeat check
            if not re_elected:
                needs_election = self.swarm.simulate_heartbeat(self.heartbeat_interval, r)
                promoted = self.swarm.secondary_promotes(self.succession_threshold)
                if needs_election or (self.swarm.leader_id is None):
                    re_elected = True

            # Re-election if needed
            if re_elected and self.swarm.alive_agents():
                result = self.algorithm.elect(self.swarm, self.graph, weights=self.weights)
                if result.leader_id is not None:
                    self.swarm.set_leader(result.leader_id)
                    if result.secondary_id:
                        self.swarm.set_secondary(result.secondary_id)

            state = RoundState(
                round=r,
                n_alive=len(self.swarm.alive_agents()),
                leader_id=self.swarm.leader_id,
                secondary_id=self.swarm.secondary_id,
                events_this_round=events,
                re_election_triggered=re_elected,
                leader_changed=(self.swarm.leader_id != prev_leader),
            )
            self.history.append(state)

        return self.history


# ──────────────────────────────────────────────────────────────────────
# Link failure simulation
# ──────────────────────────────────────────────────────────────────────

class LinkFailure:
    """
    Simulate random link failures and recoveries.

    Links are randomly disabled each round with probability `fail_prob`
    and re-enabled with probability `recover_prob`.

    Parameters
    ----------
    graph : nx.Graph
    fail_prob : float
        Per-round probability that each active edge fails.
    recover_prob : float
        Per-round probability that each failed edge recovers.
    seed : int or None
    """

    def __init__(
        self,
        graph: nx.Graph,
        fail_prob: float = 0.05,
        recover_prob: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.graph = graph
        self.fail_prob = fail_prob
        self.recover_prob = recover_prob
        self._rng = random.Random(seed)
        self._failed_edges: List[Tuple[int, int]] = []

    def tick(self) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Advance one round of link dynamics.

        Returns
        -------
        (newly_failed_edges, newly_recovered_edges)
        """
        # Fail some active edges
        newly_failed = []
        active_edges = list(self.graph.edges())
        for u, v in active_edges:
            if self._rng.random() < self.fail_prob:
                self.graph.remove_edge(u, v)
                self._failed_edges.append((u, v))
                newly_failed.append((u, v))

        # Recover some failed edges
        newly_recovered = []
        still_failed = []
        for u, v in self._failed_edges:
            if self._rng.random() < self.recover_prob:
                if self.graph.has_node(u) and self.graph.has_node(v):
                    self.graph.add_edge(u, v)
                    newly_recovered.append((u, v))
            else:
                still_failed.append((u, v))
        self._failed_edges = still_failed

        return newly_failed, newly_recovered

    @property
    def n_failed(self) -> int:
        return len(self._failed_edges)


# ──────────────────────────────────────────────────────────────────────
# Network partition simulation
# ──────────────────────────────────────────────────────────────────────

class PartitionSimulator:
    """
    Simulate a network partition by removing a cut set of edges,
    then healing after a specified number of rounds.

    Parameters
    ----------
    graph : nx.Graph
    partition_fraction : float
        Fraction of edges to remove (approximates a partition).
    heal_after : int
        Number of rounds before the partition heals.
    seed : int or None
    """

    def __init__(
        self,
        graph: nx.Graph,
        partition_fraction: float = 0.4,
        heal_after: int = 10,
        seed: Optional[int] = None,
    ):
        self.graph = graph
        self.partition_fraction = partition_fraction
        self.heal_after = heal_after
        self._rng = random.Random(seed)
        self._removed_edges: List[Tuple[int, int]] = []
        self._partition_round: Optional[int] = None
        self.is_partitioned: bool = False

    def partition(self, current_round: int) -> List[Tuple[int, int]]:
        """Apply the partition. Returns list of removed edges."""
        edges = list(self.graph.edges())
        n_remove = max(1, int(len(edges) * self.partition_fraction))
        to_remove = self._rng.sample(edges, min(n_remove, len(edges)))

        for u, v in to_remove:
            self.graph.remove_edge(u, v)
            self._removed_edges.append((u, v))

        self._partition_round = current_round
        self.is_partitioned = True
        return to_remove

    def maybe_heal(self, current_round: int) -> bool:
        """
        Heal the partition if enough rounds have elapsed.

        Returns True if healed this round.
        """
        if (
            self.is_partitioned
            and self._partition_round is not None
            and current_round >= self._partition_round + self.heal_after
        ):
            for u, v in self._removed_edges:
                if self.graph.has_node(u) and self.graph.has_node(v):
                    self.graph.add_edge(u, v)
            self._removed_edges = []
            self.is_partitioned = False
            return True
        return False
