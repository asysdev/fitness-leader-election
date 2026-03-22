"""
Information Richness (IR) metric.

IR is the Shannon entropy of an agent's knowledge topic distribution,
normalized to [0, 1] by dividing by log2(num_topics).

Formula:
    IR(a) = H(a) / log2(K)

where:
    H(a)  = -Σ p_i * log2(p_i)    for all topics i with p_i > 0
    p_i   = count_i / Σ count_j    (fraction of observations on topic i)
    K     = total number of distinct topics in the knowledge dict

Edge cases:
    - If the agent knows nothing (empty dict) → IR = 0.0
    - If the agent knows only one topic        → IR = 0.0 (zero entropy)
    - Normalization denominator is log2(K) where K ≥ 2
"""

from __future__ import annotations

import math
from typing import Dict


def information_richness(agent) -> float:
    """
    Compute the normalized Shannon entropy of an agent's knowledge.

    Parameters
    ----------
    agent : Agent
        Must have a `.knowledge_topics` attribute of type dict[str, int].

    Returns
    -------
    float in [0, 1]
        0.0 → no knowledge or all knowledge concentrated in one topic.
        1.0 → perfectly uniform distribution across all known topics.
    """
    return _compute_ir(agent.knowledge_topics)


def _compute_ir(knowledge_topics: Dict[str, int]) -> float:
    """Pure-function implementation for testability."""
    if not knowledge_topics:
        return 0.0

    counts = [c for c in knowledge_topics.values() if c > 0]
    if not counts:
        return 0.0

    total = sum(counts)
    if total == 0:
        return 0.0

    k = len(counts)
    if k == 1:
        return 0.0  # single topic → zero entropy

    # Shannon entropy
    entropy = 0.0
    for c in counts:
        p = c / total
        entropy -= p * math.log2(p)

    # Normalize by maximum possible entropy log2(k)
    max_entropy = math.log2(k)
    return entropy / max_entropy


def information_richness_from_dict(knowledge_topics: Dict[str, int]) -> float:
    """Compute IR directly from a dict (no Agent object needed)."""
    return _compute_ir(knowledge_topics)
