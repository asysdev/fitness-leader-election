from .base import ElectionAlgorithm, ElectionResult
from .fitness import FitnessElection
from .bully import BullyElection
from .random_elect import RandomElection
from .battery_elect import BatteryElection
from .degree_elect import DegreeElection
from .raft_elect import RaftElection

ALL_ALGORITHMS = [
    FitnessElection,
    BullyElection,
    RandomElection,
    BatteryElection,
    DegreeElection,
    RaftElection,
]

__all__ = [
    "ElectionAlgorithm",
    "ElectionResult",
    "FitnessElection",
    "BullyElection",
    "RandomElection",
    "BatteryElection",
    "DegreeElection",
    "RaftElection",
    "ALL_ALGORITHMS",
]
