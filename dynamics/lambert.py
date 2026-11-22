"""
Задача Ламберта в универсальных переменных (Bate-Mueller-White).
Заменяет архивный poliastro: нет зависимости от astropy, решение
занимает доли миллисекунды. Ветвь переноса определяется геометрией,
а не жестко заданным флагом
"""

import math
import numpy as np
from scipy.optimize import brentq
from .frames import ECLIPTIC_NORMAL


_Z_MAX: float = 4.0 * math.pi ** 2 - 1.0e-3


def stumpff(z: float) -> tuple[float, float]:
    """
    Вычисляет функции Штумпфа C(z) и S(z)
    Args:
        z: Универсальная переменная (безразмерная)
    Returns:
        Пара (C(z), S(z))
    """
    if z > 1.0e-6:
        root = math.sqrt(z)
        return ((1.0 - math.cos(root)) / z,
                (root - math.sin(root)) / root ** 3)
    if z < -1.0e-6:
        root = math.sqrt(-z)
        return ((math.cosh(root) - 1.0) / (-z),
                (math.sinh(root) - root) / root ** 3)
    c = 0.5 - z / 24.0 + z * z / 720.0 - z ** 3 / 40320.0
    s = 1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z ** 3 / 362880.0
    return c, s


def _y(z: float, n1: float, n2: float, area: float) -> float:
    """
    Вспомогательная величина y(z) в универсальных переменных
    Args:
        z: Универсальная переменная
        n1: Модуль первого радиус-вектора, м
        n2: Модуль второго радиус-вектора, м
        area: Параметр A со знаком направления обхода, м
    Returns:
        Значение y, м
    """
    c, s = stumpff(z)
    return n1 + n2 + area * (z * s - 1.0) / math.sqrt(c)


def _transfer_time(z: float, n1: float, n2: float, area: float,
                   mu: float) -> float:
    """
    Время переноса как функция универсальной переменной
    Args:
        z: Универсальная переменная
        n1: Модуль первого радиус-вектора, м
        n2: Модуль второго радиус-вектора, м
        area: Параметр A со знаком направления обхода, м
        mu: Гравитационный параметр центра, м^3/с^2
    Returns:
        Время переноса, с; NaN при недопустимом z
    """
    y = _y(z, n1, n2, area)
    if y <= 0.0:
        return math.nan
    c, s = stumpff(z)
    x = math.sqrt(y / c)
    return (x ** 3 * s + area * math.sqrt(y)) / math.sqrt(mu)


def solve(r1: np.ndarray, r2: np.ndarray, tof: float, mu: float, *,
          prograde: bool = True, normal: np.ndarray | None = None
          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Решает задачу Ламберта для однооборотного переноса
    Args:
        r1: Начальный радиус-вектор, м
        r2: Конечный радиус-вектор, м
        tof: Время переноса, с (> 0)
        mu: Гравитационный параметр центра, м^3/с^2
        prograde: True - прямое движение вокруг normal
        normal: Нормаль опорной плоскости; по умолчанию нормаль эклиптики
    Returns:
        Пара (v1, v2) - скорости в начальной и конечной точках, м/с
    Raises:
        ValueError: Если геометрия вырождена или решение не существует
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    n1 = float(np.linalg.norm(r1))
    n2 = float(np.linalg.norm(r2))
    n_hat = ECLIPTIC_NORMAL if normal is None else np.asarray(normal, float)
    cos_dnu = float(np.clip(r1 @ r2 / (n1 * n2), -1.0, 1.0))
    if 1.0 + cos_dnu < 1.0e-12:
        raise ValueError(
            "Угол переноса ~180 градусов: плоскость не определена. "
            "Сдвиньте эпоху прицеливания на несколько часов."
        )
    sign = 1.0 if float(np.cross(r1, r2) @ n_hat) >= 0.0 else -1.0
    if not prograde:
        sign = -sign
    area = sign * math.sqrt(n1 * n2 * (1.0 + cos_dnu))

    def residual(z: float) -> float:
        """
        Невязка по времени переноса
        Args:
            z: Универсальная переменная
        Returns:
            Разность расчетного и заданного времени, с
        """
        return _transfer_time(z, n1, n2, area, mu) - tof

    grid = np.linspace(-4.0 * math.pi ** 2, _Z_MAX, 400)
    values = [residual(float(z)) for z in grid]
    bracket: tuple[float, float] | None = None
    for k in range(len(grid) - 1):
        f_a, f_b = values[k], values[k + 1]
        if math.isfinite(f_a) and math.isfinite(f_b) and f_a * f_b < 0.0:
            bracket = (float(grid[k]), float(grid[k + 1]))
            break
    if bracket is None:
        raise ValueError(
            f"Ламберт: однооборотное решение не найдено для tof="
            f"{tof / 86400.0:.2f} сут. Проверьте ветвь переноса или "
            f"требуется многооборотное решение."
        )
    z = brentq(residual, *bracket, xtol=1.0e-12, rtol=1.0e-14)
    y = _y(z, n1, n2, area)
    f = 1.0 - y / n1
    g = area * math.sqrt(y / mu)
    g_dot = 1.0 - y / n2
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    return v1, v2


def invariants_error(r1: np.ndarray, r2: np.ndarray, v1: np.ndarray,
                     v2: np.ndarray, mu: float) -> dict[str, float]:
    """
    Проверяет решение по независимым интегралам движения и конике.
    Нормировка энергии выполняется на 1/|r1|, а не на 1/a: для почти
    параболических перелетов 1/a близка к нулю и относительная невязка
    теряет смысл.
    Args:
        r1: Начальный радиус-вектор, м
        r2: Конечный радиус-вектор, м
        v1: Начальная скорость, м/с
        v2: Конечная скорость, м/с
        mu: Гравитационный параметр центра, м^3/с^2
    Returns:
        Словарь безразмерных невязок момента, энергии и радиуса коники
    """
    n1 = float(np.linalg.norm(r1))
    n2 = float(np.linalg.norm(r2))
    h1 = np.cross(r1, v1)
    h2 = np.cross(r2, v2)
    inv_a1 = 2.0 / n1 - float(v1 @ v1) / mu
    inv_a2 = 2.0 / n2 - float(v2 @ v2) / mu
    e_vec = ((float(v1 @ v1) - mu / n1) * r1 - float(r1 @ v1) * v1) / mu
    p = float(h1 @ h1) / mu
    r2_pred = p / (1.0 + float(e_vec @ r2) / n2)
    return {
        "momentum": float(np.linalg.norm(h1 - h2)
                          / max(np.linalg.norm(h1), 1e-30)),
        "energy": float(abs(inv_a1 - inv_a2) * n1),
        "conic_radius": float(abs(r2_pred - n2) / n2),
    }
