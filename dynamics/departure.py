"""
Построение состояния КА на перилюнии отлетной гиперболы.
Исправляет ошибку исходной модели: смещение вдоль delta_v и присвоение
ламбертовской скорости в точке, где она физически недостижима
"""

import math
from dataclasses import dataclass
import numpy as np
from .frames import ECLIPTIC_NORMAL


@dataclass(frozen=True, slots=True)
class Departure:
    """
    Состояние КА в момент отлетного импульса
    Args:
        r0: Гелиоцентрическое положение на перилюнии, м
        v0: Гелиоцентрическая скорость на перилюнии, м/с
        v_inf: Избыточная скорость относительно тела отлета, м/с
        delta_v: Импульс от круговой парковочной орбиты, м/с
        normal: Нормаль плоскости отлетной гиперболы
        eccentricity: Эксцентриситет отлетной гиперболы
        v_circular: Скорость на круговой парковочной орбите, м/с
    Returns:
        None
    """

    r0: np.ndarray
    v0: np.ndarray
    v_inf: np.ndarray
    delta_v: float
    normal: np.ndarray
    eccentricity: float
    v_circular: float


def rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Поворачивает вектор вокруг единичной оси (формула Родрига)
    Args:
        v: Поворачиваемый вектор
        axis: Единичная ось поворота
        angle: Угол поворота, рад
    Returns:
        Повернутый вектор
    """
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(axis @ v) * (1 - c)


def lunar_departure(r_body: np.ndarray, v_body: np.ndarray,
                    v_helio_target: np.ndarray, mu_body: float, r_park: float,
                    normal: np.ndarray | None = None) -> Departure:
    """
    Строит отлетную гиперболу с заданной асимптотической скоростью.
    Соотношения: e = 1 + r_p v_inf^2 / mu, delta = asin(1/e),
    v_p = sqrt(v_inf^2 + 2 mu / r_p). Направление v_inf повернуто
    относительно скорости на перилюнии на угол +delta, поэтому
    v_p_hat = Rot(n, -delta) v_inf_hat, r_p_hat = Rot(n, -pi/2) v_p_hat
    Args:
        r_body: Гелиоцентрическое положение тела отлета, м
        v_body: Гелиоцентрическая скорость тела отлета, м/с
        v_helio_target: Требуемая гелиоцентрическая скорость (Ламберт), м/с
        mu_body: Гравитационный параметр тела отлета, м^3/с^2
        r_park: Радиус парковочной орбиты от центра тела, м
        normal: Нормаль плоскости отлета; None - выбор по умолчанию
    Returns:
        Объект Departure
    Raises:
        ValueError: Если избыточная скорость нулевая
    """
    v_inf = np.asarray(v_helio_target, float) - np.asarray(v_body, float)
    v_inf_norm = float(np.linalg.norm(v_inf))
    if v_inf_norm <= 0.0:
        raise ValueError("Нулевая избыточная скорость: отлет не определен")
    s_hat = v_inf / v_inf_norm
    n_hat = (np.cross(s_hat, ECLIPTIC_NORMAL) if normal is None
             else np.asarray(normal, float).copy())
    if np.linalg.norm(n_hat) < 1.0e-8:
        n_hat = np.cross(s_hat, np.array([1.0, 0.0, 0.0]))
    n_hat /= np.linalg.norm(n_hat)
    n_hat -= s_hat * float(n_hat @ s_hat)
    n_hat /= np.linalg.norm(n_hat)
    ecc = 1.0 + r_park * v_inf_norm ** 2 / mu_body
    delta = math.asin(1.0 / ecc)
    v_p_hat = rotate(s_hat, n_hat, -delta)
    r_p_hat = rotate(v_p_hat, n_hat, -math.pi / 2.0)
    v_p = math.sqrt(v_inf_norm ** 2 + 2.0 * mu_body / r_park)
    v_circ = math.sqrt(mu_body / r_park)
    return Departure(
        r0=np.asarray(r_body, float) + r_park * r_p_hat,
        v0=np.asarray(v_body, float) + v_p * v_p_hat,
        v_inf=v_inf,
        delta_v=v_p - v_circ,
        normal=n_hat,
        eccentricity=ecc,
        v_circular=v_circ,
    )
