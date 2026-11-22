"""
Интегрирование траектории: DOP853, события, точное сближение.
DOP853 (8-й порядок со встроенной оценкой погрешности) при той же
точности дает шаг на 2-3 порядка больше, чем RK4
"""

from dataclasses import dataclass
from typing import Callable, Sequence, TypeAlias
import numpy as np
from scipy.integrate import OdeSolution, solve_ivp
from .ephemeris import EphemerisProvider


Rhs: TypeAlias = Callable[[float, np.ndarray], np.ndarray]
Event: TypeAlias = Callable[[float, np.ndarray], float]
DEFAULT_ATOL: np.ndarray = np.array([1e-4, 1e-4, 1e-4, 1e-11, 1e-11, 1e-11])


@dataclass(frozen=True, slots=True)
class Propagation:
    """
    Результат интегрирования траектории
    Args:
        t: Узлы времени, с TDB
        y: Состояния в узлах, (n, 6), (м, м/с)
        dense: Непрерывное решение либо None
        t_events: Моменты срабатывания событий по порядку их передачи
        n_eval: Число вычислений правой части
        n_steps: Число принятых шагов
        terminated: Интегрирование прервано терминальным событием
    Returns:
        None
    """

    t: np.ndarray
    y: np.ndarray
    dense: OdeSolution | None
    t_events: tuple[np.ndarray, ...]
    n_eval: int
    n_steps: int
    terminated: bool

    def state_at(self, et: float) -> np.ndarray:
        """
        Состояние КА в произвольный момент внутри покрытия
        Args:
            et: Эфемеридное время TDB, с
        Returns:
            Вектор состояния [r, v], (м, м/с)
        """
        if self.dense is None:
            idx = int(np.argmin(np.abs(self.t - et)))
            return self.y[idx]
        return self.dense(et)


def apsis_event(ephemeris: EphemerisProvider, target_id: int) -> Event:
    """
    Создает событие экстремума расстояния (r_rel . v_rel = 0)
    Args:
        ephemeris: Провайдер состояний цели
        target_id: NAIF ID целевого тела
    Returns:
        Функция события с direction = +1 (минимум расстояния)
    """

    def event(t: float, y: np.ndarray) -> float:
        """
        Скалярное произведение относительных радиуса и скорости
        Args:
            t: Эфемеридное время TDB, с
            y: Состояние КА, (м, м/с)
        Returns:
            Значение, обращающееся в нуль в апсиде, м^2/с
        """
        st = ephemeris.state(t, target_id)
        return float((y[:3] - st[:3]) @ (y[3:] - st[3:]))
    
    event.direction = 1.0
    event.terminal = False
    return event


def distance_event(ephemeris: EphemerisProvider, target_id: int, radius: float,
                   terminal: bool = True) -> Event:
    """
    Создает событие входа в сферу заданного радиуса вокруг тела
    Args:
        ephemeris: Провайдер состояний цели
        target_id: NAIF ID целевого тела
        radius: Радиус сферы, м
        terminal: Прерывать ли интегрирование
    Returns:
        Функция события с direction = -1 (вход в сферу)
    """

    def event(t: float, y: np.ndarray) -> float:
        """
        Отклонение расстояния до тела от заданного радиуса
        Args:
            t: Эфемеридное время TDB, с
            y: Состояние КА, (м, м/с)
        Returns:
            Разность расстояния и радиуса, м
        """
        st = ephemeris.state(t, target_id)
        return float(np.linalg.norm(y[:3] - st[:3])) - radius

    event.direction = -1.0
    event.terminal = terminal
    return event


def propagate(rhs: Rhs, t0: float, state0: np.ndarray, t1: float, *,
              rtol: float = 1e-13, atol: np.ndarray = DEFAULT_ATOL,
              events: Sequence[Event] = (), dense: bool = True,
              max_step: float = np.inf) -> Propagation:
    """
    Интегрирует траекторию методом DOP853 с раздельными допусками
    Args:
        rhs: Правая часть ОДУ
        t0: Начальная эпоха TDB, с
        state0: Начальное состояние [r, v], (м, м/с)
        t1: Конечная эпоха TDB, с; может быть меньше t0 (обратный ход)
        rtol: Относительный допуск (не ниже 2.22e-14 для DOP853)
        atol: Вектор абсолютных порогов по 6 компонентам
        events: Функции событий; пустой кортеж означает отсутствие событий
        dense: Строить непрерывное решение
        max_step: Ограничение шага, с
    Returns:
        Объект Propagation
    Raises:
        RuntimeError: Если интегратор не смог завершить работу
    """
    event_list = list(events)
    sol = solve_ivp(
        rhs, (t0, t1), np.asarray(state0, dtype=float),
        method="DOP853", rtol=rtol, atol=atol,
        events=event_list or None, dense_output=dense,
        max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f"Интегрирование не завершено: {sol.message}")
    fired = tuple(np.asarray(e) for e in (sol.t_events or ()))
    return Propagation(
        t=sol.t, y=sol.y.T, dense=sol.sol, t_events=fired,
        n_eval=int(sol.nfev), n_steps=len(sol.t) - 1,
        terminated=bool(sol.status == 1),
    )


def closest_approach(prop: Propagation, ephemeris: EphemerisProvider,
                     target_id: int, candidates: Sequence[float] | None = None
                     ) -> tuple[float, float, np.ndarray]:
    """
    Находит истинный минимум расстояния по событиям, а не по сетке
    Args:
        prop: Результат интегрирования
        ephemeris: Провайдер состояний цели
        target_id: NAIF ID целевого тела
        candidates: Моменты-кандидаты; по умолчанию концы интервала
    Returns:
        Тройка (эпоха сближения, расстояние в м, состояние КА)
    """
    times = list(candidates) if candidates else [prop.t[0], prop.t[-1]]
    best: tuple[float, float, np.ndarray] | None = None
    for et in times:
        state = prop.state_at(float(et))
        dist = float(np.linalg.norm(
            state[:3] - ephemeris.state(float(et), target_id)[:3]))
        if best is None or dist < best[1]:
            best = (float(et), dist, state)
    assert best is not None
    return best
