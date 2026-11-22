"""
Правая часть уравнений движения КА в гелиоцентрической СК J2000.
Модель: N точечных масс (прямые + непрямые члены) + J2 планет
+ 1PN (Шварцшильд) поправка Солнца.
Гелиоцентрическая СК НЕ инерциальна: Солнце ускоряется полем планет.
Поэтому для каждого тела, кроме Солнца, учитывается непрямой член
-mu_i * rho_i / |rho_i|^3, равный минус ускорению начала координат.
Тела с нулевым GM (исключенные) не участвуют в вычислении геометрии:
это нужно и для корректности (иначе 0 * inf = nan при совпадении
положений), и для скорости в тестах с урезанной моделью
"""

import math
from typing import Sequence
import numpy as np
from .bodies import Body
from .ephemeris import EphemerisProvider

C_LIGHT: float = 299792458.0
# Нижний порог квадрата расстояния, м^2: защита от деления на нуль
# при точном совпадении с центром тела (физически недостижимо)
_R2_FLOOR: float = 1.0


class NBodyField:
    """
    Гравитационное поле системы тел как правая часть ОДУ
    """

    def __init__(self, ephemeris: EphemerisProvider, bodies: Sequence[Body],
                 relativity: bool = True, indirect: bool = True,
                 exclude: Sequence[int] = ()) -> None:
        """
        Инициализирует модель поля
        Args:
            ephemeris: Провайдер положений тел
            bodies: Каталог тел; порядок обязан совпадать с провайдером
            relativity: Учитывать 1PN поправку Солнца
            indirect: Учитывать непрямые члены (ускорение центра Солнца)
            exclude: NAIF ID тел, гравитацию которых нужно отключить
        Returns:
            None
        Raises:
            ValueError: Если порядок тел не совпадает или все тела исключены
        """
        if tuple(b.naif_id for b in bodies) != ephemeris.naif_ids:
            raise ValueError("Порядок тел в каталоге и провайдере различен")
        self._eph = ephemeris
        excluded = {int(i) for i in exclude}
        gm_all = np.array([b.gm for b in bodies], dtype=float)
        for k, body in enumerate(bodies):
            if body.naif_id in excluded:
                gm_all[k] = 0.0
        self._active = np.flatnonzero(gm_all > 0.0)
        if self._active.size == 0:
            raise ValueError("Все тела исключены из модели")
        self._all_active = self._active.size == len(bodies)
        self._gm = gm_all[self._active]
        self._sun = int(np.argmax(self._gm))
        self._relativity = relativity
        self._indirect = indirect
        self._eye = np.eye(3)
        self._others = np.array(
            [k for k in range(self._active.size) if k != self._sun],
            dtype=int)
        self._gm_others = self._gm[self._others]
        local = {int(g): k for k, g in enumerate(self._active)}
        self._obl = [
            (local[k], gm_all[k], b.j2, b.radius ** 2, b.pole)
            for k, b in enumerate(bodies)
            if b.j2 > 0.0 and b.pole is not None and k in local
        ]

    def _positions(self, t: float) -> np.ndarray:
        """
        Положения только активных тел
        Args:
            t: Эфемеридное время TDB, с
        Returns:
            Массив формы (n_active, 3), м
        """
        pos = self._eph.positions(t)
        return pos if self._all_active else pos[self._active]

    def _geometry(self, t: float, r: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Общая геометрия задачи: положения, разности, квадраты расстояний
        Args:
            t: Эфемеридное время TDB, с
            r: Радиус-вектор КА, м
        Returns:
            Тройка (положения тел, векторы КА минус тело, квадраты модулей)
        """
        pos = self._positions(t)
        d = r - pos
        r2 = np.einsum("ij,ij->i", d, d)
        np.maximum(r2, _R2_FLOOR, out=r2)
        return pos, d, r2

    def _point_mass(self, d: np.ndarray, r2: np.ndarray, with_gradient: bool
                    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Прямые члены ускорения и их градиент
        Args:
            d: Векторы КА минус тело, (n, 3), м
            r2: Квадраты модулей этих векторов, (n,), м^2
            with_gradient: Вычислять ли матрицу da/dr
        Returns:
            Пара (ускорение, градиент 3x3 либо None)
        """
        inv3 = r2 ** -1.5
        weights = self._gm * inv3
        acc = -np.einsum("i,ij->j", weights, d)
        if not with_gradient:
            return acc, None
        grad = -float(np.sum(weights)) * self._eye
        grad += 3.0 * np.einsum("i,ij,ik->jk", weights / r2, d, d)
        return acc, grad

    def _frame_acceleration(self, pos: np.ndarray) -> np.ndarray:
        """
        Ускорение центра Солнца под действием остальных тел
        Args:
            pos: Гелиоцентрические положения активных тел, (n, 3), м
        Returns:
            Вектор ускорения начала координат, м/с^2
        """
        rho = pos[self._others]
        rho2 = np.einsum("ij,ij->i", rho, rho)
        return np.einsum("i,ij->j", self._gm_others * rho2 ** -1.5, rho)

    def _extras(self, acc: np.ndarray, pos: np.ndarray, d: np.ndarray,
                r2: np.ndarray, v: np.ndarray) -> None:
        """
        Добавляет непрямые члены, J2 и 1PN на месте.
        Непрямые члены не зависят от положения КА, поэтому в градиент
        da/dr их вклад строго нулевой
        Args:
            acc: Накопленное ускорение, м/с^2 (модифицируется)
            pos: Гелиоцентрические положения активных тел, (n, 3), м
            d: Векторы КА минус тело, (n, 3), м
            r2: Квадраты модулей этих векторов, (n,), м^2
            v: Скорость КА, м/с
        Returns:
            None
        """
        if self._indirect:
            acc -= self._frame_acceleration(pos)
        for k, gm, j2, rad2, pole in self._obl:
            dk = d[k]
            rn = math.sqrt(float(r2[k]))
            z = float(dk @ pole)
            s = z / rn
            coef = 1.5 * gm * j2 * rad2 / rn ** 5
            acc -= coef * ((1.0 - 5.0 * s * s) * dk + 2.0 * z * pole)
        if self._relativity:
            dk = d[self._sun]
            rn = math.sqrt(float(r2[self._sun]))
            mu = self._gm[self._sun]
            rv = float(dk @ v)
            acc += (mu / (C_LIGHT ** 2 * rn ** 3)) * (
                (4.0 * mu / rn - float(v @ v)) * dk + 4.0 * rv * v
            )

    def acceleration(self, t: float, r: np.ndarray, v: np.ndarray
                     ) -> np.ndarray:
        """
        Полное ускорение КА в гелиоцентрической СК
        Args:
            t: Эфемеридное время TDB, с
            r: Радиус-вектор КА, м
            v: Скорость КА, м/с
        Returns:
            Вектор ускорения, м/с^2
        """
        pos, d, r2 = self._geometry(t, r)
        acc, _ = self._point_mass(d, r2, with_gradient=False)
        self._extras(acc, pos, d, r2, v)
        return acc

    def acceleration_and_gradient(self, t: float, r: np.ndarray, v: np.ndarray
                                  ) -> tuple[np.ndarray, np.ndarray]:
        """
        Ускорение и градиент da/dr за один запрос эфемерид.
        В градиент не входят вклады J2 и 1PN: это влияет лишь на
        скорость сходимости Ньютона, но не на найденный корень,
        поскольку невязка считается полной моделью
        Args:
            t: Эфемеридное время TDB, с
            r: Радиус-вектор КА, м
            v: Скорость КА, м/с
        Returns:
            Пара (ускорение м/с^2, матрица da/dr 3x3 в 1/с^2)
        """
        pos, d, r2 = self._geometry(t, r)
        acc, grad = self._point_mass(d, r2, with_gradient=True)
        self._extras(acc, pos, d, r2, v)
        assert grad is not None
        return acc, grad

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Производная состояния для интегратора ОДУ
        Args:
            t: Эфемеридное время TDB, с
            state: Вектор [x, y, z, vx, vy, vz], (м, м/с)
        Returns:
            Вектор [vx, vy, vz, ax, ay, az], (м/с, м/с^2)
        """
        acc = self.acceleration(t, state[:3], state[3:])
        return np.concatenate((state[3:], acc))
