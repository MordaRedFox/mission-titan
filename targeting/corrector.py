"""
Дифференциальная коррекция начальной скорости по фиксированной эпохе.
Невязка берется в ЗАДАННУЮ эпоху прицеливания, а не в точке минимума
расстояния: в последнем случае якобиан вырожден (ранг 2), и Ньютон
принципиально не может устранить промах.
Якобиан получается интегрированием матрицы чувствительности (STM),
что исключает конечные разности и дает квадратичную сходимость
"""

import logging
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from dynamics.ephemeris import EphemerisProvider
from dynamics.gravity import NBodyField
from dynamics.propagation import DEFAULT_ATOL, distance_event


logger = logging.getLogger(__name__)


class TrajectoryImpact(RuntimeError):
    """
    Траектория столкнулась с целью до эпохи прицеливания
    """


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    """
    Итог работы корректора
    Args:
        v0: Скорректированная начальная скорость, м/с
        residual: Итоговая норма невязки по положению, м
        iterations: Число выполненных итераций Ньютона
        converged: Достигнут ли заданный допуск
    Returns:
        None
    """

    v0: np.ndarray
    residual: float
    iterations: int
    converged: bool


class StmPropagator:
    """
    Совместное интегрирование состояния и матрицы чувствительности
    """

    def __init__(self, field: NBodyField, ephemeris: EphemerisProvider | None = None,
                 target_id: int | None = None, guard_radius: float = 0.0) -> None:
        """
        Инициализирует пропагатор с_optional защитным событием
        Args:
            field: Модель гравитационного поля
            ephemeris: Провайдер состояний цели для защитного события
            target_id: NAIF ID цели
            guard_radius: Радиус сферы, вход в которую считается аварией, м
        Returns:
            None
        """
        self._field = field
        self._a = np.zeros((6, 6))
        self._a[:3, 3:] = np.eye(3)
        self._guard = None
        if ephemeris is not None and target_id is not None and guard_radius > 0:
            self._guard = distance_event(
                ephemeris, target_id, guard_radius, terminal=True)

    def _rhs(self, t: float, z: np.ndarray) -> np.ndarray:
        """
        Правая часть расширенной системы (6 + 36 уравнений)
        Args:
            t: Эфемеридное время TDB, с
            z: Вектор [r, v, vec(Phi)]
        Returns:
            Производная вектора z
        """
        r, v = z[:3], z[3:6]
        phi = z[6:].reshape(6, 6)
        acc, grad = self._field.acceleration_and_gradient(t, r, v)
        self._a[3:, :3] = grad
        return np.concatenate((v, acc, (self._a @ phi).ravel()))

    def run(self, t0: float, state0: np.ndarray, t1: float, rtol: float = 1e-12,
            atol: np.ndarray = DEFAULT_ATOL) -> tuple[np.ndarray, np.ndarray]:
        """
        Интегрирует состояние и STM до заданной эпохи
        Args:
            t0: Начальная эпоха TDB, с
            state0: Начальное состояние [r, v], (м, м/с)
            t1: Эпоха прицеливания TDB, с
            rtol: Относительный допуск интегратора
            atol: Абсолютные допуски для 6 компонент состояния
        Returns:
            Пара (состояние в t1, матрица Phi 6x6)
        Raises:
            TrajectoryImpact: Если сработало защитное событие
            RuntimeError: Если интегратор не завершился
        """
        z0 = np.concatenate((state0, np.eye(6).ravel()))
        tol = np.concatenate((atol, np.full(36, 1e-12)))
        sol = solve_ivp(
            self._rhs, (t0, t1), z0, method="DOP853", rtol=rtol,
            atol=tol, events=[self._guard] if self._guard else None,
            dense_output=False,
        )
        if sol.status == 1:
            raise TrajectoryImpact(
                f"Столкновение с целью на ET {sol.t[-1]:.1f} "
                f"до эпохи прицеливания {t1:.1f}"
            )
        if not sol.success:
            raise RuntimeError(f"STM: {sol.message}")
        return sol.y[:6, -1], sol.y[6:, -1].reshape(6, 6)


class RendezvousCorrector:
    """
    Ньютоновская пристрелка к фиксированной точке в фиксированную эпоху
    """

    def __init__(self, field: NBodyField, ephemeris: EphemerisProvider,
                 target_id: int, t0: float, t_aim: float, r0: np.ndarray,
                 aim_offset: np.ndarray | None = None, rtol: float = 1e-12,
                 guard_radius: float = 0.0) -> None:
        """
        Инициализирует корректор
        Args:
            field: Модель гравитационного поля
            ephemeris: Провайдер состояний цели
            target_id: NAIF ID целевого тела
            t0: Эпоха старта TDB, с
            t_aim: Эпоха прицеливания TDB, с
            r0: Фиксированное начальное положение КА, м
            aim_offset: Смещение точки прицеливания от центра цели, м
            rtol: Относительный допуск интегратора
            guard_radius: Радиус защитной сферы вокруг цели, м
        Returns:
            None
        """
        self._stm = StmPropagator(field, ephemeris, target_id, guard_radius)
        self._eph = ephemeris
        self._target = target_id
        self._t0 = t0
        self._t_aim = t_aim
        self._r0 = np.asarray(r0, dtype=float)
        self._offset = (np.zeros(3) if aim_offset is None
                        else np.asarray(aim_offset, dtype=float))
        self._rtol = rtol

    def _residual(self, v0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Невязка и якобиан для текущего приближения скорости
        Args:
            v0: Начальная гелиоцентрическая скорость, м/с
        Returns:
            Пара (невязка 3 в м, якобиан 3x3 в м на м/с)
        """
        state, phi = self._stm.run(
            self._t0, np.concatenate((self._r0, v0)),
            self._t_aim, self._rtol,
        )
        aim = self._eph.state(self._t_aim, self._target)[:3] + self._offset
        return state[:3] - aim, phi[:3, 3:]

    def solve(self, v0_guess: np.ndarray, tol_m: float = 1.0e3,
              max_iter: int = 12, max_dv: float = 1_000.0) -> CorrectionResult:
        """
        Решает задачу пристрелки демпфированным методом Ньютона
        Args:
            v0_guess: Начальное приближение скорости, м/с
            tol_m: Допустимая норма невязки по положению, м
            max_iter: Максимум итераций Ньютона
            max_dv: Ограничение нормы поправки за итерацию, м/с
        Returns:
            Объект CorrectionResult
        """
        v0 = np.asarray(v0_guess, dtype=float).copy()
        res, jac = self._residual(v0)
        norm = float(np.linalg.norm(res))
        logger.info("Пристрелка: начальная невязка %.4e м", norm)
        for k in range(max_iter):
            if norm < tol_m:
                return CorrectionResult(v0, norm, k, True)
            scale = np.maximum(np.linalg.norm(jac, axis=0), 1e-30)
            u, s, vt = np.linalg.svd(jac / scale)
            s = np.maximum(s, s[0] * 1e-12)
            dv = -(vt.T @ ((u.T @ res) / s)) / scale
            dv *= min(1.0, max_dv / max(float(np.linalg.norm(dv)), 1e-12))
            accepted = False
            for _ in range(10):
                try:
                    res_t, jac_t = self._residual(v0 + dv)
                except TrajectoryImpact:
                    dv *= 0.5
                    continue
                norm_t = float(np.linalg.norm(res_t))
                if norm_t < norm:
                    accepted = True
                    break
                dv *= 0.5
            if not accepted:
                logger.warning("Пристрелка: шаг не улучшил невязку")
                return CorrectionResult(v0, norm, k + 1, False)
            v0, res, jac, norm = v0 + dv, res_t, jac_t, norm_t
            logger.info(
                "  итерация %d: |dv|=%.3f м/с, невязка %.4e м",
                k + 1, np.linalg.norm(dv), norm,
            )
        return CorrectionResult(v0, norm, max_iter, norm < tol_m)
