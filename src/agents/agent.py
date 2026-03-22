"""
Agent class representing an individual drone/node in the swarm.

Each agent has four capability dimensions:
  - battery:        remaining charge level          ∈ [0, 1]
  - sensor_health:  operational sensor fraction     ∈ [0, 1]
  - storage:        available storage fraction      ∈ [0, 1]
  - payload:        payload capacity fraction       ∈ [0, 1]
  - knowledge_topics: dict mapping topic → observation count

These feed into the three fitness sub-metrics (IR, CC, MC).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional


# Default topic universe used for knowledge generation
DEFAULT_TOPICS = [
    "terrain_map",
    "obstacle_positions",
    "target_locations",
    "weather_data",
    "friendly_positions",
    "threat_assessment",
    "mission_status",
    "resource_inventory",
    "communication_links",
    "navigation_waypoints",
]


@dataclass
class AgentCapabilities:
    """Normalized [0,1] capability scores for one agent."""
    battery: float = 1.0
    sensor_health: float = 1.0
    storage: float = 1.0
    payload: float = 1.0

    def __post_init__(self):
        for attr in ("battery", "sensor_health", "storage", "payload"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr} must be in [0, 1], got {val}")


class Agent:
    """
    A single drone/node participating in leader election.

    Parameters
    ----------
    agent_id : int
        Unique identifier within the swarm.
    battery : float
        Battery level ∈ [0, 1].
    sensor_health : float
        Fraction of sensors operational ∈ [0, 1].
    storage : float
        Available storage fraction ∈ [0, 1].
    payload : float
        Payload capacity fraction ∈ [0, 1].
    knowledge_topics : dict[str, int] or None
        Mapping of topic name → number of observations.
        If None, the agent has no domain knowledge.
    alive : bool
        Whether this agent is currently active in the network.
    """

    def __init__(
        self,
        agent_id: int,
        battery: float = 1.0,
        sensor_health: float = 1.0,
        storage: float = 1.0,
        payload: float = 1.0,
        knowledge_topics: Optional[Dict[str, int]] = None,
        alive: bool = True,
    ):
        self.agent_id = agent_id
        self.capabilities = AgentCapabilities(
            battery=battery,
            sensor_health=sensor_health,
            storage=storage,
            payload=payload,
        )
        self.knowledge_topics: Dict[str, int] = knowledge_topics if knowledge_topics is not None else {}
        self.alive = alive

        # Election state
        self.is_leader: bool = False
        self.is_secondary: bool = False
        self.voted_for: Optional[int] = None
        self.current_term: int = 0  # used by Raft baseline
        self.heartbeat_missed: int = 0

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def battery(self) -> float:
        return self.capabilities.battery

    @battery.setter
    def battery(self, v: float):
        self.capabilities.battery = max(0.0, min(1.0, v))

    @property
    def sensor_health(self) -> float:
        return self.capabilities.sensor_health

    @sensor_health.setter
    def sensor_health(self, v: float):
        self.capabilities.sensor_health = max(0.0, min(1.0, v))

    @property
    def storage(self) -> float:
        return self.capabilities.storage

    @storage.setter
    def storage(self, v: float):
        self.capabilities.storage = max(0.0, min(1.0, v))

    @property
    def payload(self) -> float:
        return self.capabilities.payload

    @payload.setter
    def payload(self, v: float):
        self.capabilities.payload = max(0.0, min(1.0, v))

    # ------------------------------------------------------------------
    # Knowledge management
    # ------------------------------------------------------------------

    def observe(self, topic: str, count: int = 1) -> None:
        """Record observations about a topic."""
        self.knowledge_topics[topic] = self.knowledge_topics.get(topic, 0) + count

    def total_observations(self) -> int:
        return sum(self.knowledge_topics.values())

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "battery": self.battery,
            "sensor_health": self.sensor_health,
            "storage": self.storage,
            "payload": self.payload,
            "knowledge_topics": dict(self.knowledge_topics),
            "alive": self.alive,
            "is_leader": self.is_leader,
            "is_secondary": self.is_secondary,
        }

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.agent_id}, bat={self.battery:.2f}, "
            f"sens={self.sensor_health:.2f}, stor={self.storage:.2f}, "
            f"pay={self.payload:.2f}, topics={len(self.knowledge_topics)}, "
            f"alive={self.alive})"
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        agent_id: int,
        topics: Optional[list] = None,
        rng: Optional[random.Random] = None,
        min_topics: int = 3,
        max_topics: int = len(DEFAULT_TOPICS),
    ) -> "Agent":
        """
        Create an agent with uniformly-random capabilities and knowledge.

        Parameters
        ----------
        agent_id : int
        topics : list[str] or None
            Topic universe. Defaults to DEFAULT_TOPICS.
        rng : random.Random or None
            Optional seeded RNG for reproducibility.
        min_topics : int
            Minimum number of topics the agent will have knowledge about.
        max_topics : int
            Maximum number of topics.
        """
        if rng is None:
            rng = random.Random()
        if topics is None:
            topics = DEFAULT_TOPICS

        battery = rng.uniform(0.1, 1.0)
        sensor_health = rng.uniform(0.1, 1.0)
        storage = rng.uniform(0.1, 1.0)
        payload = rng.uniform(0.1, 1.0)

        # Sample a random subset of topics and assign observation counts
        n_topics = rng.randint(min_topics, max(min_topics, min(max_topics, len(topics))))
        chosen_topics = rng.sample(topics, n_topics)
        knowledge = {t: rng.randint(1, 50) for t in chosen_topics}

        return cls(
            agent_id=agent_id,
            battery=battery,
            sensor_health=sensor_health,
            storage=storage,
            payload=payload,
            knowledge_topics=knowledge,
            alive=True,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Agent":
        return cls(
            agent_id=d["agent_id"],
            battery=d["battery"],
            sensor_health=d["sensor_health"],
            storage=d["storage"],
            payload=d["payload"],
            knowledge_topics=d.get("knowledge_topics", {}),
            alive=d.get("alive", True),
        )
