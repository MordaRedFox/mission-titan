"""
Импульсные маневры: захват в орбиту спутника и бюджет Delta-v
"""

import math
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class CapturePlan:
    """
    План импульсного выхода на орбиту целевого тела
    Args:
        periapsis: Радиус перицентра, м
        apoapsis: Радиус апоцентра целевой орбиты, м
        v_inf: Скорость подлета относительно тела, м/с
        v_periapsis: Скорость на перицентре подлетной гиперболы, м/с
        v_target: Требуемая скорость на перицентре целевой орбиты, м/с
        delta_v: Модуль тормозного импульса, м/с
        hill_radius: Радиус сферы Хилла тела, м
        stable: Апоцентр внутри половины сферы Хилла
    Returns:
        None
    """

    periapsis: float
    apoapsis: float
    v_inf: float
    v_periapsis: float
    v_target: float
    delta_v: float
    hill_radius: float
    stable: bool

    def as_dict(self) -> dict[str, float | bool]:
        """
        Представление плана в виде словаря для отчета
        Returns:
            Словарь полей плана
        """
        return asdict(self)


def hill_sphere(mu_target: float, mu_primary: float,
                semi_major_axis: float) -> float:
    """
    Вычисляет радиус сферы Хилла спутника
    Args:
        mu_target: GM спутника, м^3/с^2
        mu_primary: GM центрального тела, м^3/с^2
        semi_major_axis: Большая полуось орбиты спутника, м
    Returns:
        Радиус сферы Хилла, м
    """
    return float(semi_major_axis) * (mu_target / (3.0 * mu_primary)) ** (1 / 3)


def capture_plan(v_inf: np.ndarray | float, periapsis: float, mu_target: float,
                 apoapsis: float | None = None, hill_radius: float = math.inf
                 ) -> CapturePlan:
    """
    Считает импульс выхода на орбиту вокруг целевого тела
    Args:
        v_inf: Вектор или модуль подлетной скорости относительно тела, м/с
        periapsis: Радиус перицентра целевой орбиты, м
        mu_target: Гравитационный параметр целевого тела, м^3/с^2
        apoapsis: Апоцентр целевой орбиты, м; None - круговая орбита
        hill_radius: Радиус сферы Хилла для оценки устойчивости, м
    Returns:
        Объект CapturePlan
    """
    v_inf_norm = float(np.linalg.norm(v_inf))
    v_p = math.sqrt(v_inf_norm ** 2 + 2.0 * mu_target / periapsis)
    r_apo = float(periapsis if apoapsis is None else apoapsis)
    a = 0.5 * (periapsis + r_apo)
    v_target = math.sqrt(mu_target * (2.0 / periapsis - 1.0 / a))
    return CapturePlan(
        periapsis=float(periapsis), apoapsis=r_apo, v_inf=v_inf_norm,
        v_periapsis=v_p, v_target=v_target, delta_v=v_p - v_target,
        hill_radius=float(hill_radius), stable=r_apo < 0.5 * hill_radius,
    )
