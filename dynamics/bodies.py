"""
Каталог притягивающих тел и их физические параметры.
Для Меркурия, Венеры, Марса, Юпитера, Урана и Нептуна используются
барицентры (1, 2, 4, 5, 7, 8): в DE440 отдельных сегментов центров этих
планет может не быть, а их спутники в модели отдельно не учитываются.
Сатурн берется как центр 699 (из sat441) вместе с отдельными спутниками,
чтобы исключить двойной учет масс
"""

import logging
from dataclasses import dataclass
from typing import Sequence, Final
import numpy as np
import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError
from .kernels import bodvrd


logger = logging.getLogger(__name__)

# Alias для поиска физических констант: барицентр -> центр планеты
_PARAM_ALIAS: Final[dict[int, int]] = {
    1: 199,
    2: 299,
    4: 499,
    5: 599,
    7: 799,
    8: 899,
}
# Второй зональный коэффициент J2 (JPL SSD / Cassini gravity field)
_J2: Final[dict[int, float]] = {
    399: 1.08262668e-3,
    5:   1.4696430e-2,
    699: 1.6290710e-2,
}
# Имя телофиксированной системы для ориентации полюса
_POLE_FRAME: Final[dict[int, str]] = {
    399: "IAU_EARTH",
    5:   "IAU_JUPITER",
    699: "IAU_SATURN",
}
# Резервные GM, м^3/с^2 (DE440 + sat441)
_GM_FALLBACK: Final[dict[int, float]] = {
    10:  1.3271244004193938e20, 1:   2.2031868551e13,
    2:   3.2485859200e14,       399: 3.98600435507e14,
    301: 4.902800066e12,        4:   4.2828375816e13,
    5:   1.26686531900e17,      699: 3.7931206234e16,
    603: 4.1213528e10,          604: 7.3116366e10,     605: 1.5394246e11,
    606: 8.978138376543e12,     608: 1.2051510e11,
    7:   5.793951256e15,        8:   6.83509997e15,
}
# Резервные экваториальные радиусы, м.
_RADIUS_FALLBACK: Final[dict[int, float]] = {
    10:  6.957e8,   1:   2.4397e6,   2:   6.0518e6,  399: 6.3781366e6,
    301: 1.7374e6,  4:   3.39619e6,  5:   7.1492e7,  699: 6.0268e7,
    603: 5.311e5,   604: 5.614e5,    605: 7.635e5,   606: 2.57473e6,
    608: 7.343e5,   7:   2.5559e7,   8:   2.4764e7,
}
# (NAIF ID, имя, цвет) в порядке вычисления ускорения
DEFAULT_SPECS: Final[tuple[tuple[int, str, str], ...]] = (
    (10,  "Sun",     "#FFD700"),
    (1,   "Mercury", "#A9A9A9"),
    (2,   "Venus",   "#F0E68C"),
    (399, "Earth",   "#4B9CD3"),
    (301, "Moon",    "#C0C0C0"),
    (4,   "Mars",    "#CD5C5C"),
    (5,   "Jupiter", "#CD853F"),
    (699, "Saturn",  "#F4E6C3"),
    (603, "Tethys",  "#DDDDDD"),
    (604, "Dione",   "#DDDDDD"),
    (605, "Rhea",    "#DDDDDD"),
    (606, "Titan",   "#B87333"),
    (608, "Iapetus", "#DDDDDD"),
    (7,   "Uranus",  "#7FDBFF"),
    (8,   "Neptune", "#4169E1"),
)


@dataclass(frozen=True, slots=True)
class Body:
    """
    Небесное тело как источник гравитационного поля
    Args:
        naif_id: NAIF ID центра тела, используемый в динамике
        name: Человекочитаемое имя
        gm: Гравитационный параметр GM, м^3/с^2
        radius: Экваториальный радиус, м
        color: Цвет для визуализации
        j2: Коэффициент J2 (0.0 - точечная масса)
        pole: Единичный вектор полюса в J2000 либо None
    Returns:
        None
    """
    naif_id: int
    name: str
    gm: float
    radius: float
    color: str
    j2: float = 0.0
    pole: np.ndarray | None = None


def _lookup(naif_id: int, item: str, maxn: int) -> list[float] | None:
    """
    Ищет параметр тела по его ID и по alias центра планеты
    Args:
        naif_id: NAIF ID тела
        item: Имя параметра SPICE
        maxn: Максимальное число значений
    Returns:
        Список значений либо None
    """
    values = bodvrd(naif_id, item, maxn)
    if values is None and naif_id in _PARAM_ALIAS:
        values = bodvrd(_PARAM_ALIAS[naif_id], item, maxn)
    return values


def _pole(naif_id: int, et_ref: float) -> np.ndarray | None:
    """
    Возвращает единичный вектор полюса тела в J2000.
    Ориентация берется на опорную эпоху: прецессия полюсов планет за
    время миссии пренебрежимо мала для члена J2
    Args:
        naif_id: NAIF ID тела
        et_ref: Опорная эпоха TDB, с
    Returns:
        Единичный вектор полюса либо None, если frame недоступен
    """
    frame = _POLE_FRAME.get(naif_id)
    if frame is None:
        return None
    try:
        rot = np.asarray(spice.pxform(frame, "J2000", et_ref), float)
        pole = rot[:, 2].copy()
        return pole / np.linalg.norm(pole)
    except SpiceyError:
        return None


def build_catalog(specs: Sequence[tuple[int, str, str]], et_ref: float,
                  oblateness: bool = True) -> tuple[Body, ...]:
    """Собирает каталог тел с параметрами из SPICE и резервными значениями
    Args:
        specs: Тройки (naif_id, имя, цвет) в порядке вычислений
        et_ref: Эпоха TDB для ориентации полюсов, с
        oblateness: Учитывать ли J2 планет
    Returns:
        Кортеж объектов Body
    """
    catalog: list[Body] = []
    for naif_id, name, color in specs:
        gm_values = _lookup(naif_id, "GM", 1)
        gm = gm_values[0] * 1.0e9 if gm_values else _GM_FALLBACK[naif_id]
        radii = _lookup(naif_id, "RADII", 3)
        radius = (radii[0] * 1000.0 if radii
                  else _RADIUS_FALLBACK[naif_id])
        j2 = _J2.get(naif_id, 0.0) if oblateness else 0.0
        pole = _pole(naif_id, et_ref) if j2 > 0.0 else None
        if j2 > 0.0 and pole is None:
            logger.warning("%s: полюс недоступен, J2 отключён", name)
            j2 = 0.0
        catalog.append(Body(naif_id, name, gm, radius, color, j2, pole))
    return tuple(catalog)


def find(catalog: Sequence[Body], naif_id: int) -> Body:
    """Находит тело в каталоге по NAIF ID
    Args:
        catalog: Каталог тел
        naif_id: Искомый NAIF ID
    Returns:
        Объект Body
    Raises:
        KeyError: Если тело отсутствует в каталоге
    """
    for body in catalog:
        if body.naif_id == naif_id:
            return body
    raise KeyError(f"Тело {naif_id} отсутствует в каталоге")
