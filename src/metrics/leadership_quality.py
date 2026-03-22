"""
Leadership Quality Score (LQS).

LQS provides an omniscient evaluation of how good the elected leader
is relative to the theoretical best possible choice:

    LQS = fitness(elected_leader) / max(fitness(a) for a in all_alive_agents)

    LQS = 1.0  → perfect selection (elected the best possible agent)
    LQS = 0.0  → worst possible selection
    LQS ∈ (0,1) → suboptimal but not worst-case

This metric requires an omniscient view of all agents' capabilities and
is used only for offline evaluation — it is NOT computed by the
algorithms themselves during the election.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx

from .information_richness import information_richness
from .centrality import eigenvector_centrality_map, communication_centrality
from .mission_capacity import mission_capacity


def compute_fitness(
    agent,
    graph: nx.Graph,
    weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
    centrality_cache: Optional[Dict] = None,
) -> float:
    """
    Compute the composite fitness score for a single agent.

    This is the core fitness function matching Algorithm 1 from the patent:

        F(a) = w₁·IR(a) + w₂·CC(a) + w₃·MC(a)

    Parameters
    ----------
    agent : Agent
    graph : nx.Graph
        Current network topology.
    weights : (w_IR, w_CC, w_MC)
        Must sum to 1.0. Default (0.40, 0.35, 0.25).
    centrality_cache : dict or None
        Pre-computed centrality map to avoid recomputation.

    Returns
    -------
    float in [0, 1]
    """
    w_ir, w_cc, w_mc = weights

    ir = information_richness(agent)
    cc = communication_centrality(agent, graph, _cache=centrality_cache)
    mc = mission_capacity(agent)

    return w_ir * ir + w_cc * cc + w_mc * mc


def compute_all_fitness(
    agents: List,
    graph: nx.Graph,
    weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
) -> Dict[int, float]:
    """
    Compute fitness for all agents in one pass (efficient: computes
    centrality only once).

    Returns
    -------
    dict[agent_id, fitness_score]
    """
    # Compute centrality once for the whole graph
    centrality_cache = eigenvector_centrality_map(graph)

    return {
        a.agent_id: compute_fitness(a, graph, weights=weights, centrality_cache=centrality_cache)
        for a in agents
        if a.alive
    }


def leadership_quality_score(
    elected_id: int,
    agents: List,
    graph: nx.Graph,
    weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
    fitness_map: Optional[Dict[int, float]] = None,
) -> float:
    """
    Compute the Leadership Quality Score for an elected leader.

    Parameters
    ----------
    elected_id : int
        The agent_id of the elected leader.
    agents : list[Agent]
        All alive agents (omniscient view).
    graph : nx.Graph
        Current topology.
    weights : tuple
        Fitness weights (w_IR, w_CC, w_MC).
    fitness_map : dict or None
        Pre-computed {agent_id: fitness} map to avoid recomputation.

    Returns
    -------
    float in [0, 1]
        1.0 = optimal selection, 0.0 = worst possible.
    """
    if fitness_map is None:
        fitness_map = compute_all_fitness(agents, graph, weights=weights)

    if not fitness_map:
        return 0.0

    elected_fitness = fitness_map.get(elected_id, 0.0)
    max_fitness = max(fitness_map.values())

    if max_fitness == 0.0:
        return 1.0  # degenerate: all zero fitness, any choice is equally good

    return elected_fitness / max_fitness


def optimal_leader_id(
    agents: List,
    graph: nx.Graph,
    weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
) -> Tuple[int, float]:
    """
    Return the agent_id and fitness of the theoretically optimal leader.

    Used to identify the ground-truth best agent for LQS evaluation.

    Returns
    -------
    (optimal_id, optimal_fitness)
    """
    fitness_map = compute_all_fitness(agents, graph, weights=weights)
    if not fitness_map:
        raise ValueError("No alive agents to elect from.")
    optimal_id = max(fitness_map, key=fitness_map.__getitem__)
    return optimal_id, fitness_map[optimal_id]
