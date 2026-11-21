"""
Опорные системы координат: экваториальная J2000 и эклиптическая J2000
"""

import math
from typing import Final
import numpy as np
import numpy.typing as npt


# Наклон эклиптики к экватору на эпоху J2000.0 (IAU 2006), рад.
OBLIQUITY_J2000: Final[float] = math.radians(23.439291111)
_COS_E: Final[float] = math.cos(OBLIQUITY_J2000)
_SIN_E: Final[float] = math.sin(OBLIQUITY_J2000)
# Нормаль эклиптики, выраженная в экваториальных координатах J2000
ECLIPTIC_NORMAL: Final[npt.NDArray[np.float64]] = np.array([0.0, -_SIN_E, _COS_E])
# Матрица поворота экваториальные -> эклиптические координаты
EQ_TO_ECL: Final[npt.NDArray[np.float64]] = np.array([
    [1.0, 0.0, 0.0],
    [0.0, _COS_E, _SIN_E],
    [0.0, -_SIN_E, _COS_E],
])


def to_ecliptic(vectors: np.ndarray) -> np.ndarray:
    """
    Переводит векторы из экваториальной J2000 в эклиптическую J2000
    Args:
        vectors: Массив формы (..., 3) в экваториальных координатах
    Returns:
        Массив той же формы в эклиптических координатах
    """
    return np.asarray(vectors, dtype=float) @ EQ_TO_ECL.T


def to_equatorial(vectors: np.ndarray) -> np.ndarray:
    """
    Переводит векторы из эклиптической J2000 в экваториальную J2000
    Args:
        vectors: Массив формы (..., 3) в эклиптических координатах
    Returns:
        Массив той же формы в экваториальных координатах
    """
    return np.asarray(vectors, dtype=float) @ EQ_TO_ECL
