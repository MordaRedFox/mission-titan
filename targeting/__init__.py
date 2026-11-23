"""
Пакет прицеливания: B-плоскость и дифференциальная коррекция
"""

from .bplane import aim_offset, b_plane_basis
from .corrector import CorrectionResult, RendezvousCorrector, TrajectoryImpact

__all__ = [
    "aim_offset",
    "b_plane_basis",
    "CorrectionResult",
    "RendezvousCorrector",
    "TrajectoryImpact",
]
