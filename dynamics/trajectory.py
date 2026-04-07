"""
trajectory.py – модуль для численного интегрирования траектории космического
аппарата. Интегрирование может выполняться с помощью scipy.integrate.solve_ivp
(RK45) или собственного адаптивного метода Рунге-Кутты 4-го порядка
"""
from typing import Tuple
import numpy as np
from scipy.integrate import solve_ivp
from gravity import acceleration, get_distance_to_body
from integrator import integrate_adaptive


def integrate_trajectory(start_et: float, end_et: float, state0: np.ndarray,
    method: str = "rk45", rtol: float = 1e-9, atol: float = 1e-12,
    adaptive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Выполняет интегрирование траектории КА Поддерживаются два метода: 'rk45':
    метод Рунге-Кутты 5(4) из scipy.integrate.solve_ivp, 'rk4_adaptive':
    собственный адаптивный метод Рунге-Кутты 4-го порядка
    Параметры:
        start_et: начальное эфемеридное время, с
        end_et: конечное время интегрирования, с
        state0: начальное состояние КА [x, y, z, vx, vy, vz], (м, м/с)
        method: метод интегрирования - 'rk45' или 'rk4_adaptive'
        rtol: относительная допустимая погрешность
        atol: абсолютная допустимая погрешность
        adaptive: если True, используется адаптивный шаг (для 'rk4_adaptive')
    Возвращает:
        t: массив моментов времени, с
        states: массив состояний [x, y, z, vx, vy, vz] для каждого момента времени
    """
    MAX_YEARS = 10
    MAX_TIME = MAX_YEARS * 365.25 * 86400.0
    TITAN_DISTANCE_THRESHOLD = 1e6
    TITAN_ID = 606

    # Функция события для solve_ivp
    def titan_distance_event(t: float, y: np.ndarray) -> float:
        """
        Функция события для определения момента достижения заданного
        расстояния до Титана
        Параметры:
            t: текущее эфемеридное время, с
            y: вектор состояния КА [x, y, z, vx, vy, vz] (м, м/с)
        Возвращает:
            Разность между текущим расстоянием до Титана и пороговым значением.
            Событие происходит при пересечении нуля (direction = -1 означает,
            что событие срабатывает, когда значение убывает)
        """
        dist = get_distance_to_body(t, y, TITAN_ID)
        return dist - TITAN_DISTANCE_THRESHOLD
    titan_distance_event.direction = -1

    def time_limit_event(t: float, y: np.ndarray) -> float:
        """
        Функция события для ограничения максимального времени интегрирования
        Параметры:
            t: текущее эфемеридное время, с
            y: вектор состояния КА [x, y, z, vx, vy, vz] (м, м/с)
        Возвращает:
            Разность между текущим временем и предельным временем.
            Событие срабатывает при пересечении нуля (direction = 1 означает,
            что событие срабатывает при возрастании)
        """
        return t - start_et - MAX_TIME
    time_limit_event.direction = 1

    events = [titan_distance_event, time_limit_event]
    if method == "rk45":
        sol = solve_ivp(acceleration, [start_et, end_et], state0,
                        method="RK45", rtol=rtol, atol=atol,
                        dense_output=False, max_step=3600.0,
                        events=events)
        t = sol.t
        states = sol.y.T
    elif method == "rk4_adaptive":
        if adaptive:
            def stop_condition(t, y):
                dist = get_distance_to_body(t, y, TITAN_ID)
                days = (t - start_et) / 86400
                print(f"DEBUG: t={t:.1f} с, дни={days:.2f}, " + \
                      f"дистанция={dist/1000:.0f} км")
                if t - start_et >= MAX_TIME:
                    print(f"Остановка по времени: t={t:.1f}, " + \
                          f"MAX_TIME={MAX_TIME}")
                    return True
                if dist <= TITAN_DISTANCE_THRESHOLD:
                    print(f"Остановка по расстоянию: dist={dist/1000:.0f} " + \
                          f"км <= {TITAN_DISTANCE_THRESHOLD/1000:.0f} км")
                    return True
                return False
            t, states = integrate_adaptive(
                f=acceleration,
                t0=start_et,
                y0=state0,
                t_end=end_et,
                h0=3600.0,
                rtol=rtol,
                atol=atol,
                h_min=0.1,
                h_max=86400.0,
                max_steps=10**6,
                stop_condition=stop_condition
            )
        else:
            raise NotImplementedError("Неадаптивный алгоритм RK4 не реализован")
    else:
        raise ValueError("Неподдерживаемый метод. Выберите 'rk45' или 'rk4_adaptive'")
    return t, states
