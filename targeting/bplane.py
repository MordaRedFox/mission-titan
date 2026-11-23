"""
Геометрия B-плоскости и построение точки прицеливания
"""

import math
import numpy as np
from dynamics.frames import ECLIPTIC_NORMAL


def b_plane_basis(v_inf: np.ndarray, normal: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Строит ортонормированный базис B-плоскости (S, T, R)
    Args:
        v_inf: Вектор входящей асимптотической скорости, м/с
        normal: Опорная нормаль; по умолчанию нормаль эклиптики J2000
    Returns:
        Тройка единичных векторов (S, T, R)
    """
    s = np.asarray(v_inf, float) / np.linalg.norm(v_inf)
    n = ECLIPTIC_NORMAL if normal is None else np.asarray(normal, float)
    t = np.cross(s, n)
    if np.linalg.norm(t) < 1.0e-8:
        t = np.cross(s, np.array([1.0, 0.0, 0.0]))
    t /= np.linalg.norm(t)
    return s, t, np.cross(s, t)


def aim_offset(v_inf: np.ndarray, periapsis: float, mu_target: float,
               angle: float = 0.0, normal: np.ndarray | None = None
               ) -> np.ndarray:
    """
    Смещение точки прицеливания для заданного радиуса перицентра.
    Модуль вектора B: |B| = r_p sqrt(1 + 2 mu / (r_p v_inf^2))
    Args:
        v_inf: Входящая асимптотическая скорость относительно цели, м/с
        periapsis: Желаемый радиус перицентра, м
        mu_target: Гравитационный параметр цели, м^3/с^2
        angle: Угол в B-плоскости от оси T, рад
        normal: Опорная нормаль B-плоскости
    Returns:
        Вектор смещения в инерциальной СК J2000, м
    """
    _, t, r = b_plane_basis(v_inf, normal)
    v = float(np.linalg.norm(v_inf))
    b = periapsis * math.sqrt(1.0 + 2.0 * mu_target / (periapsis * v * v))
    return b * (math.cos(angle) * t + math.sin(angle) * r)
