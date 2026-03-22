"""
Swarm class: manages the collection of Agent objects and provides
helpers for bulk capability generation, leader/secondary tracking,
heartbeat simulation, and churn event processing.
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional, Tuple

import networkx as nx

from .agent import Agent, DEFAULT_TOPICS


class Swarm:
    """
    A collection of Agent objects corresponding to nodes in a network graph.

    Agents are stored keyed by agent_id. The Swarm does NOT own the graph
    topology; that is managed by the caller (experiments / election algorithms).
    The Swarm is the authoritative store for agent state.

    Parameters
    ----------
    agents : list[Agent]
        Initial agent population.
    topics : list[str]
        Topic universe (used when generating new agents during churn).
    """

    def __init__(self, agents: List[Agent], topics: Optional[List[str]] = None):
        self._agents: Dict[int, Agent] = {a.agent_id: a for a in agents}
        self.topics = topics or DEFAULT_TOPICS

        self.leader_id: Optional[int] = None
        self.secondary_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._agents.values())

    def __contains__(self, agent_id: int) -> bool:
        return agent_id in self._agents

    def get(self, agent_id: int) -> Optional[Agent]:
        return self._agents.get(agent_id)

    def agent_ids(self) -> List[int]:
        return list(self._agents.keys())

    def alive_agents(self) -> List[Agent]:
        return [a for a in self._agents.values() if a.alive]

    def alive_ids(self) -> List[int]:
        return [a.agent_id for a in self.alive_agents()]

    # ------------------------------------------------------------------
    # Leader / secondary management
    # ------------------------------------------------------------------

    def set_leader(self, agent_id: int) -> None:
        """Mark agent_id as leader; clear previous leader flag."""
        # Clear old flags
        for a in self._agents.values():
            a.is_leader = False
            a.is_secondary = False

        self.leader_id = agent_id
        if agent_id in self._agents:
            self._agents[agent_id].is_leader = True

    def set_secondary(self, agent_id: int) -> None:
        """Mark agent_id as secondary leader."""
        if self.secondary_id and self.secondary_id in self._agents:
            self._agents[self.secondary_id].is_secondary = False
        self.secondary_id = agent_id
        if agent_id in self._agents:
            self._agents[agent_id].is_secondary = True

    def get_leader(self) -> Optional[Agent]:
        if self.leader_id is not None:
            return self._agents.get(self.leader_id)
        return None

    def get_secondary(self) -> Optional[Agent]:
        if self.secondary_id is not None:
            return self._agents.get(self.secondary_id)
        return None

    # ------------------------------------------------------------------
    # Heartbeat simulation
    # ------------------------------------------------------------------

    def simulate_heartbeat(self, heartbeat_interval: int, current_round: int) -> bool:
        """
        Simulate a heartbeat tick.

        The leader emits a heartbeat every `heartbeat_interval` rounds.
        If the leader is dead, the secondary's missed-heartbeat counter
        is incremented. Returns True if a new leader election is needed.

        Parameters
        ----------
        heartbeat_interval : int
            Rounds between heartbeats.
        current_round : int
            Current simulation round.

        Returns
        -------
        bool
            True if the secondary has promoted itself (re-election needed).
        """
        leader = self.get_leader()
        secondary = self.get_secondary()

        if leader is None or not leader.alive:
            # Leader is gone — increment secondary's missed count
            if secondary is not None and secondary.alive:
                secondary.heartbeat_missed += 1
                return False  # caller decides threshold
            return True  # no leader and no secondary → full re-election

        if current_round % heartbeat_interval == 0:
            # Leader is alive → reset secondary counter
            if secondary is not None:
                secondary.heartbeat_missed = 0

        return False

    def secondary_promotes(self, threshold: int = 3) -> bool:
        """
        Check if the secondary should promote itself to leader.

        Returns True if promotion occurred.
        """
        secondary = self.get_secondary()
        if secondary is not None and secondary.alive and secondary.heartbeat_missed >= threshold:
            old_secondary_id = self.secondary_id
            self.set_leader(old_secondary_id)
            secondary.heartbeat_missed = 0
            return True
        return False

    # ------------------------------------------------------------------
    # Churn operations
    # ------------------------------------------------------------------

    def add_agent(
        self,
        graph: nx.Graph,
        agent_id: Optional[int] = None,
        rng: Optional[random.Random] = None,
        edge_prob: float = 0.3,
    ) -> Tuple[Agent, List[int]]:
        """
        Add a new random agent to the swarm and connect it to the graph.

        Parameters
        ----------
        graph : nx.Graph
            The live network graph (modified in-place).
        agent_id : int or None
            ID to assign; defaults to max(existing)+1.
        rng : random.Random or None
        edge_prob : float
            Probability of connecting to each existing alive node.

        Returns
        -------
        (new_agent, connected_to_ids)
        """
        if rng is None:
            rng = random.Random()

        if agent_id is None:
            agent_id = max(self._agents.keys(), default=-1) + 1

        new_agent = Agent.random(agent_id, topics=self.topics, rng=rng)
        self._agents[agent_id] = new_agent

        graph.add_node(agent_id)
        connected = []
        for existing_id in self.alive_ids():
            if existing_id != agent_id and rng.random() < edge_prob:
                graph.add_edge(agent_id, existing_id)
                connected.append(existing_id)

        # Ensure at least one connection if graph is non-empty
        if not connected and len(self.alive_ids()) > 1:
            target = rng.choice([aid for aid in self.alive_ids() if aid != agent_id])
            graph.add_edge(agent_id, target)
            connected.append(target)

        return new_agent, connected

    def remove_agent(
        self,
        agent_id: int,
        graph: nx.Graph,
    ) -> Optional[Agent]:
        """
        Mark agent as dead and remove it from the graph.

        Returns the removed Agent or None if not found.
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            return None

        agent.alive = False
        if graph.has_node(agent_id):
            graph.remove_node(agent_id)

        # Clear leadership flags
        if self.leader_id == agent_id:
            self.leader_id = None
        if self.secondary_id == agent_id:
            self.secondary_id = None

        return agent

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_graph(
        cls,
        graph: nx.Graph,
        topics: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> "Swarm":
        """
        Create a Swarm where agents correspond 1-to-1 with graph nodes.

        Node attributes in the graph (battery, sensor_health, etc.) are
        respected if present; otherwise capabilities are random.
        """
        rng = random.Random(seed)
        agents = []
        for node_id in sorted(graph.nodes()):
            attrs = graph.nodes[node_id]
            if "battery" in attrs:
                agent = Agent(
                    agent_id=node_id,
                    battery=attrs.get("battery", rng.uniform(0.1, 1.0)),
                    sensor_health=attrs.get("sensor_health", rng.uniform(0.1, 1.0)),
                    storage=attrs.get("storage", rng.uniform(0.1, 1.0)),
                    payload=attrs.get("payload", rng.uniform(0.1, 1.0)),
                    knowledge_topics=attrs.get("knowledge_topics", {}),
                )
            else:
                agent = Agent.random(node_id, topics=topics or DEFAULT_TOPICS, rng=rng)
            agents.append(agent)
        return cls(agents, topics=topics or DEFAULT_TOPICS)

    @classmethod
    def random_swarm(
        cls,
        n: int,
        topics: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> "Swarm":
        """Create a swarm of n randomly-initialized agents."""
        rng = random.Random(seed)
        agents = [Agent.random(i, topics=topics or DEFAULT_TOPICS, rng=rng) for i in range(n)]
        return cls(agents, topics=topics or DEFAULT_TOPICS)

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> List[dict]:
        """Serialize all agents to a list of dicts (for checkpointing)."""
        return [a.to_dict() for a in self._agents.values()]

    def reset_election_state(self) -> None:
        """Clear all leader/secondary/vote state for a fresh election."""
        self.leader_id = None
        self.secondary_id = None
        for a in self._agents.values():
            a.is_leader = False
            a.is_secondary = False
            a.voted_for = None
            a.current_term = 0
            a.heartbeat_missed = 0

    def __repr__(self) -> str:
        n_alive = len(self.alive_agents())
        return (
            f"Swarm(total={len(self)}, alive={n_alive}, "
            f"leader={self.leader_id}, secondary={self.secondary_id})"
        )
