"""
integrator.py – реализация метода Рунге-Кутты 4-го порядка с адаптивным шагом
(правило Рунге для оценки локальной погрешности)
"""
import numpy as np
from typing import Callable, Optional


def rk4_step(f: Callable[[float, np.ndarray], np.ndarray], t: float,
             y: np.ndarray, h: float) -> np.ndarray:
    """
    Выполняет один шаг метода Рунге-Кутты 4-го порядка
    Параметры:
        f: функция правой части системы dy/dt = f(t, y)
        t: текущее время
        y: текущее состояние (вектор)
        h: шаг интегрирования
    Возвращает:
        y_next: новое состояние после шага h
    """
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)
    y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_next

def integrate_adaptive(
        f: Callable[[float, np.ndarray], np.ndarray], t0: float,
        y0: np.ndarray, t_end: float, h0: float = 1e-4, rtol: float = 1e-9,
        atol: float = 1e-12, h_min: float = 1e-8, h_max: float = 86400.0,
        max_steps: int = 10**6, dense_output: bool = False,
        progress_callback: Optional[Callable[[float, np.ndarray], None]] = None,
        stop_condition: Optional[Callable[[float, np.ndarray], bool]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Интегрирует систему ОДУ от t0 до t_end с адаптивным выбором шага.
    Используется метод Рунге-Кутты 4-го порядка с контролем локальной
    погрешности по правилу Рунге (сравнение одного шага h и двух шагов h/2)
    Параметры:
        f: функция правой части: производная = f(t, y)
        t0: начальное время, с
        y0: начальное состояние (массив)
        t_end: конечное время, с
        h0: начальный шаг, с
        rtol: относительная допустимая погрешность
        atol: абсолютная допустимая погрешность
        h_min: минимально допустимый шаг, с
        h_max: максимально допустимый шаг, с
        max_steps: максимальное количество шагов
        dense_output: если True, сохраняет все промежуточные точки
        progress_callback: функция для вывода прогресса. Принимает текущие (t, y)
        stop_condition : функция stop_condition(t, y), возвращающая True, если
                         интегрирование нужно остановить. По умолчанию None.
    Возвращает:
        t_list: массив моментов времени, в которые сохранено состояние
        y_list: массив состояний (каждая строка соответствует моменту времени)
    """
    t = t0
    y = y0.copy()
    h = h0
    t_list = [t0]
    y_list = [y0.copy()]
    steps = 0
    while t < t_end - 1e-12 and steps < max_steps:
        # Проверяет условие остановки перед каждым шагом
        if stop_condition is not None and stop_condition(t, y):
            print(f"Остановка по условию stop_condition в t = {t:.1f} с")
            break
        # Ограничивает шаг, чтобы не перескочить конечное время
        if t + h > t_end:
            h = t_end - t
        y1 = rk4_step(f, t, y, h)
        y_half = rk4_step(f, t, y, h/2)
        y2 = rk4_step(f, t + h/2, y_half, h/2)
        error = np.linalg.norm(y2 - y1) / (2**4 - 1)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y2))
        tol = np.linalg.norm(scale)
        if error <= tol:
            # Шаг принят
            t += h
            y = y2.copy()
            t_list.append(t)
            y_list.append(y.copy())
            if progress_callback is not None:
                progress_callback(t, y)
            # Увеличивает шаг (но не более h_max)
            h = min(h_max, h * 2.0)
        else:
            # Шаг не принят, уменьшает его
            h = max(h_min, h * 0.5)

        steps += 1
        if steps % 1000 == 0:
            print(f"t = {t/86400:.1f} дни, h = {h:.1f} s, ошибка = " + \
                  f"{error:.2e}, шаги = {steps}")

        # Предотвращает зацикливание из-за слишком маленького шага
        if h < h_min:
            h = h_min
            t += h
            y = rk4_step(f, t - h, y, h)
            t_list.append(t)
            y_list.append(y.copy())
            if progress_callback is not None:
                progress_callback(t, y)
            steps += 1
    if steps >= max_steps:
        print(f"Предупреждение: достигнуто максимальное число шагов " + \
              f"({max_steps}). Интегрирование остановлено")
    return np.array(t_list), np.array(y_list)
