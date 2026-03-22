"""
Communication Centrality (CC) metric.

Uses NetworkX eigenvector centrality, normalized to [0, 1] across
all agents in the graph at the time of election.

Eigenvector centrality measures how well-connected a node is to other
well-connected nodes — capturing communication influence in the swarm.

Normalization:
    CC(a) = raw_centrality(a) / max(raw_centrality for all nodes)

Edge cases:
    - Disconnected graph: use degree centrality as fallback.
    - Single-node graph: CC = 1.0.
    - Node not in graph: CC = 0.0.
"""

from __future__ import annotations

from typing import Dict, Optional

import networkx as nx


def eigenvector_centrality_map(graph: nx.Graph, max_iter: int = 1000) -> Dict[int, float]:
    """
    Compute normalized eigenvector centrality for all nodes in `graph`.

    Returns a dict mapping node_id → normalized centrality ∈ [0, 1].

    Falls back to degree centrality if eigenvector centrality fails to
    converge (common in disconnected or near-disconnected graphs).
    """
    if len(graph) == 0:
        return {}

    if len(graph) == 1:
        node = next(iter(graph.nodes()))
        return {node: 1.0}

    # Try eigenvector centrality on the largest connected component
    # to avoid convergence failures on disconnected graphs.
    try:
        raw = nx.eigenvector_centrality(graph, max_iter=max_iter, tol=1e-6)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
        # Fallback: degree centrality (always converges)
        raw = nx.degree_centrality(graph)

    if not raw:
        return {n: 0.0 for n in graph.nodes()}

    max_val = max(raw.values())
    if max_val == 0.0:
        return {n: 0.0 for n in graph.nodes()}

    normalized = {n: v / max_val for n, v in raw.items()}
    # Ensure all graph nodes are present (isolated nodes get 0)
    for n in graph.nodes():
        normalized.setdefault(n, 0.0)
    return normalized


def communication_centrality(agent, graph: nx.Graph, _cache: Optional[Dict] = None) -> float:
    """
    Return the normalized eigenvector centrality of `agent` in `graph`.

    Parameters
    ----------
    agent : Agent
        Must have `.agent_id` attribute.
    graph : nx.Graph
        Current network topology.
    _cache : dict or None
        Optional pre-computed centrality map (avoids recomputing for every
        agent during a single election round).

    Returns
    -------
    float in [0, 1]
    """
    if not graph.has_node(agent.agent_id):
        return 0.0

    centrality_map = _cache if _cache is not None else eigenvector_centrality_map(graph)
    return centrality_map.get(agent.agent_id, 0.0)
