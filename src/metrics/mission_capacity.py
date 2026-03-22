"""
Mission Capacity (MC) metric.

MC is a weighted sum of an agent's hardware resources:

    MC(a) = 0.4 * battery
          + 0.3 * sensor_health
          + 0.2 * storage
          + 0.1 * payload

All inputs and the output are in [0, 1].

The weights reflect typical mission priorities:
  - Battery dominates because it limits operational duration.
  - Sensor health determines data quality.
  - Storage determines how much data can be retained.
  - Payload matters for physical mission tasks (dropping packages, etc.).
"""

from __future__ import annotations

from typing import Tuple


# Default weights (must sum to 1.0)
_DEFAULT_WEIGHTS = (0.4, 0.3, 0.2, 0.1)


def mission_capacity(
    agent,
    weights: Tuple[float, float, float, float] = _DEFAULT_WEIGHTS,
) -> float:
    """
    Compute the Mission Capacity score for an agent.

    Parameters
    ----------
    agent : Agent
        Must have `.battery`, `.sensor_health`, `.storage`, `.payload`.
    weights : tuple of 4 floats
        (w_battery, w_sensor_health, w_storage, w_payload).
        Must sum to 1.0. Defaults to (0.4, 0.3, 0.2, 0.1).

    Returns
    -------
    float in [0, 1]
    """
    w_bat, w_sens, w_stor, w_pay = weights
    return (
        w_bat  * agent.battery
        + w_sens * agent.sensor_health
        + w_stor * agent.storage
        + w_pay  * agent.payload
    )


def mission_capacity_from_values(
    battery: float,
    sensor_health: float,
    storage: float,
    payload: float,
    weights: Tuple[float, float, float, float] = _DEFAULT_WEIGHTS,
) -> float:
    """Compute MC directly from raw values (no Agent object needed)."""
    w_bat, w_sens, w_stor, w_pay = weights
    return w_bat * battery + w_sens * sensor_health + w_stor * storage + w_pay * payload
