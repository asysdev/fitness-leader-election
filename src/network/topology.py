"""
Network topology generators for drone swarm simulation.

Provides four graph types via NetworkX:
  1. mesh       — regular grid graph (deterministic, uniform connectivity)
  2. random     — Erdős–Rényi random graph G(n, p)
  3. scale_free — Barabási–Albert preferential attachment
  4. geometric  — Random geometric graph (models physical proximity)

All generators return an nx.Graph with integer node IDs 0..n-1.
"""

from __future__ import annotations

import math
from typing import Optional

import networkx as nx
import numpy as np


class TopologyGenerator:
    """
    Factory for generating swarm network topologies.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Mesh topology
    # ------------------------------------------------------------------

    def mesh(self, n: int) -> nx.Graph:
        """
        Generate an approximately square grid (mesh) graph with n nodes.

        Nodes at the boundary have degree 2–3; interior nodes have degree 4.
        This models a regularly spaced drone formation.

        Parameters
        ----------
        n : int
            Target number of nodes. The actual grid may have slightly
            more nodes (rounded up to the nearest rectangle).

        Returns
        -------
        nx.Graph with nodes relabeled 0..N-1
        """
        rows = math.isqrt(n)
        cols = math.ceil(n / rows)
        G = nx.grid_2d_graph(rows, cols)
        # Remove excess nodes
        nodes = list(G.nodes())
        excess = len(nodes) - n
        for node in nodes[-excess:] if excess > 0 else []:
            G.remove_node(node)
        # Relabel to integers
        mapping = {old: i for i, old in enumerate(G.nodes())}
        return nx.relabel_nodes(G, mapping)

    # ------------------------------------------------------------------
    # Erdős–Rényi random graph
    # ------------------------------------------------------------------

    def random(self, n: int, p: float = 0.3, ensure_connected: bool = True) -> nx.Graph:
        """
        Generate an Erdős–Rényi random graph G(n, p).

        Parameters
        ----------
        n : int
            Number of nodes.
        p : float
            Edge probability ∈ (0, 1].
        ensure_connected : bool
            If True, keep regenerating until the graph is connected
            (or add a spanning tree if needed).

        Returns
        -------
        nx.Graph
        """
        seed_val = int(self._rng.randint(0, 2**31))
        G = nx.erdos_renyi_graph(n, p, seed=seed_val)

        if ensure_connected and not nx.is_connected(G):
            G = self._make_connected(G, seed_val)

        return G

    # ------------------------------------------------------------------
    # Barabási–Albert scale-free graph
    # ------------------------------------------------------------------

    def scale_free(self, n: int, m: int = 2) -> nx.Graph:
        """
        Generate a Barabási–Albert preferential attachment graph.

        Scale-free graphs model hub-and-spoke communication patterns
        common in real swarms (e.g., relay drones with high degree).

        Parameters
        ----------
        n : int
            Number of nodes.
        m : int
            Number of edges to attach from a new node to existing nodes.
            Must satisfy 1 ≤ m < n.

        Returns
        -------
        nx.Graph (always connected)
        """
        seed_val = int(self._rng.randint(0, 2**31))
        m = max(1, min(m, n - 1))
        return nx.barabasi_albert_graph(n, m, seed=seed_val)

    # ------------------------------------------------------------------
    # Random geometric graph
    # ------------------------------------------------------------------

    def geometric(self, n: int, radius: float = 0.35, ensure_connected: bool = True) -> nx.Graph:
        """
        Generate a random geometric graph in the unit square.

        Two nodes are connected iff their Euclidean distance ≤ radius.
        This models physical proximity-based communication (e.g., WiFi/RF).

        Parameters
        ----------
        n : int
            Number of nodes.
        radius : float
            Communication radius ∈ (0, √2]. Larger → denser graph.
            Connectivity threshold for large n ≈ √(ln(n)/πn).
        ensure_connected : bool
            If True, add minimum edges to connect isolated components.

        Returns
        -------
        nx.Graph
        """
        seed_val = int(self._rng.randint(0, 2**31))
        G = nx.random_geometric_graph(n, radius, seed=seed_val)

        if ensure_connected and not nx.is_connected(G):
            G = self._make_connected(G, seed_val)

        # Relabel to ensure 0-based integer IDs
        if set(G.nodes()) != set(range(n)):
            mapping = {old: i for i, old in enumerate(sorted(G.nodes()))}
            G = nx.relabel_nodes(G, mapping)

        return G

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_connected(self, G: nx.Graph, seed_val: int) -> nx.Graph:
        """
        Connect all components of G by adding minimum-weight spanning edges
        between components (using a deterministic chain of component roots).
        """
        components = list(nx.connected_components(G))
        if len(components) <= 1:
            return G

        rng = np.random.RandomState(seed_val)
        # Sort components by smallest node id for determinism
        components.sort(key=lambda c: min(c))

        for i in range(len(components) - 1):
            u = min(components[i])
            v = min(components[i + 1])
            G.add_edge(u, v)

        return G

    # ------------------------------------------------------------------
    # Convenience: generate all four topologies for a given n
    # ------------------------------------------------------------------

    def all_topologies(self, n: int) -> dict:
        """
        Return all four topology types as a dict keyed by name.

        Returns
        -------
        {"mesh": G, "random": G, "scale_free": G, "geometric": G}
        """
        return {
            "mesh": self.mesh(n),
            "random": self.random(n),
            "scale_free": self.scale_free(n),
            "geometric": self.geometric(n),
        }

    # ------------------------------------------------------------------
    # Graph diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def describe(G: nx.Graph) -> dict:
        """Return a dict of key graph statistics."""
        degrees = [d for _, d in G.degree()]
        try:
            diameter = nx.diameter(G) if nx.is_connected(G) else float("inf")
        except Exception:
            diameter = float("inf")
        return {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_connected(G),
            "diameter": diameter,
            "avg_degree": sum(degrees) / len(degrees) if degrees else 0.0,
            "max_degree": max(degrees) if degrees else 0,
            "min_degree": min(degrees) if degrees else 0,
            "n_components": nx.number_connected_components(G),
        }
