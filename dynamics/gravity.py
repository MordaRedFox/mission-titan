"""
gravity.py – модуль для расчёта ускорения космического аппарата под действием
гравитации тел Солнечной системы с использованием эфемерид SPICE (ИПА РАН)
"""
import numpy as np
import spiceypy as spice


G = 6.67430e-11
BODIES = {
    10:   ("Sun",     1.9885e30,  "yellow"),
    1:    ("Mercury", 3.3011e23,  "gray"),
    2:    ("Venus",   4.8675e24,  "gold"),
    3:    ("Earth",   5.9722e24,  "blue"),
    4:    ("Mars",    6.4171e23,  "red"),
    5:    ("Jupiter", 1.8982e27,  "orange"),
    6:    ("Saturn",  5.6834e26,  "khaki"),
    301:  ("Moon",    7.3477e22,  "gray"),
    606:  ("Titan",   1.3452e23,  "lightblue"),
}

def load_kernel(kernel_path: str) -> None:
    """
    Загружает SPICE-кернел
    Параметры:
        kernel_path: путь к файлу кернела (bsp, bpc, tpc и т.д.)
    Возвращает:
        None
    """
    spice.furnsh(kernel_path)

def get_body_position_helio(et: float, body_id: int) -> np.ndarray:
    """
    Возвращает гелиоцентрическую позицию тела в метрах
    Параметры:
        et: эфемеридное время TDB, с
        body_id: SPICE ID небесного тела
    Возвращает:
        Вектор положения [x, y, z] в метрах
    """
    state_km, _ = spice.spkezr(str(body_id), et, "J2000", "NONE", "10")
    pos_km = state_km[:3]
    return np.array(pos_km) * 1000.0

def get_body_velocity_helio(et: float, body_id: int) -> np.ndarray:
    """
    Возвращает гелиоцентрическую скорость тела в м/с
    Параметры
        et: эфемеридное время TDB, с
        body_id: SPICE ID небесного тела
    Возвращает:
        Вектор скорости [vx, vy, vz] в м/с
    """
    state_km, _ = spice.spkezr(str(body_id), et, "J2000", "NONE", "10")
    vel_km = state_km[3:]
    return np.array(vel_km) * 1000.0

def acceleration(t: float, state: np.ndarray) -> np.ndarray:
    """
    Вычисляет производную состояния КА: [vx, vy, vz, ax, ay, az]
    Параметры:
        t: эфемеридное время TDB, с
        state: вектор состояния КА [x, y, z, vx, vy, vz] (м, м/с)
    Возвращает:
        Производная состояния [vx, vy, vz, ax, ay, az] (м/с, м/с²)
    """
    r_sc = state[:3]
    a_total = np.zeros(3)
    for body_id, (_, mass, _) in BODIES.items():
        r_body = get_body_position_helio(t, body_id)
        r_rel = r_sc - r_body
        dist = np.linalg.norm(r_rel)
        if dist > 1e-3:
            a_total -= G * mass * r_rel / (dist ** 3)
    return np.concatenate([state[3:], a_total])

def get_distance_to_body(et: float, state: np.ndarray, body_id: int) -> float:
    """
    Возвращает расстояние от КА до тела с заданным ID в метрах
    Параметры:
        et: эфемеридное время TDB, с
        state: вектор состояния КА [x, y, z, vx, vy, vz] (м, м/с)
        body_id: SPICE ID небесного тела
    Возвращает:
        Расстояние от КА до тела в метрах
    """
    r_sc = state[:3]
    r_body = get_body_position_helio(et, body_id)
    return np.linalg.norm(r_sc - r_body)
