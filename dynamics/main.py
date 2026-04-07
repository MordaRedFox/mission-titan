import os
import sys
import glob
import inspect
import numpy as np
import spiceypy as spice
import astropy.units as u
from typing import Tuple
from scipy.optimize import root
from poliastro.iod import lambert
from trajectory import integrate_trajectory
from gravity import (load_kernel, get_body_position_helio,
                     get_body_velocity_helio)


# =============================================================================
# 1. Константные параметры и загрузка кернелов
# =============================================================================
KERNEL_DIR = "data/ephemeris"
START_UTC = "2027-08-01T00:00:00"
FLIGHT_DAYS = 870
POST_FLIGHT_DAYS = 30
METHOD = "rk4_adaptive"
RTOL = 1e-15
ATOL = 1e-17
OUTPUT_FILE = "data/trajectory/trajectory.npz"

kernel_files = glob.glob(os.path.join(KERNEL_DIR, "*.bsp")) + \
               glob.glob(os.path.join(KERNEL_DIR, "*.bpc")) + \
               glob.glob(os.path.join(KERNEL_DIR, "*.tcp")) + \
               glob.glob(os.path.join(KERNEL_DIR, "*.tpc")) + \
               glob.glob(os.path.join(KERNEL_DIR, "*.tf")) + \
               glob.glob(os.path.join(KERNEL_DIR, "*.tls"))
if not kernel_files:
    raise FileNotFoundError(f"Не найдены файлы кернелов в {KERNEL_DIR}")
for kfile in kernel_files:
    load_kernel(kfile)
    print(f"Загружен: {kfile}")


# =============================================================================
# 2. Функция Ламберта (адаптивная)
# =============================================================================
def lambert_solution(r1: np.ndarray, r2: np.ndarray, dt: float,
                     mu_sun: float = 1.32712440018e20, use_long_way: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Решает задачу Ламберта для двух позиций и времени перелёта
    Параметры:
        r1: начальный гелиоцентрический вектор положения, м
        r2: конечный гелиоцентрический вектор положения, м
        dt: время перелёта, с
        mu_sun: гравитационный параметр Солнца, м³/с²
        use_long_way: если False, используется короткая дуга (быстрее),
                      если True - длинная дуга
    Возвращает:
        v1: гелиоцентрическая скорость в начальной точке, м/с
        v2: гелиоцентрическая скорость в конечной точке, м/с
    """
    r1_km = (r1 / 1000.0) * u.km
    r2_km = (r2 / 1000.0) * u.km
    dt_day = (dt / 86400.0) * u.day
    mu_sun_km3_s2 = (mu_sun / 1e9) * (u.km**3 / u.s**2)
    sig = inspect.signature(lambert)
    params = sig.parameters
    if "short" in params:
        gen = lambert(mu_sun_km3_s2, r1_km, r2_km, dt_day,
                      short=not use_long_way)
    elif "shortway" in params:
        gen = lambert(mu_sun_km3_s2, r1_km, r2_km, dt_day,
                      shortway=not use_long_way)
    else:
        gen = lambert(mu_sun_km3_s2, r1_km, r2_km, dt_day)
    result = next(gen)
    v1_km = result[0]
    v2_km = result[1]
    v1 = v1_km.to(u.m / u.s).value
    v2 = v2_km.to(u.m / u.s).value
    return v1, v2


# =============================================================================
# 3. Начальные условия (старт от Луны)
# =============================================================================
start_et = spice.str2et(START_UTC)
# Скорость и положение луны
r_moon = get_body_position_helio(start_et, 301)
v_moon = get_body_velocity_helio(start_et, 301)
# Время прибытия к Титану
tof_seconds = FLIGHT_DAYS * 86400.0
arrival_et = start_et + tof_seconds
r_titan_arrival = get_body_position_helio(arrival_et, 606)

print(f"Начальное время: {START_UTC} (TDB {start_et:.2f} с)")
print(f"Положение Луны: {r_moon/1000} км")
print(f"Скорость Луны: {np.linalg.norm(v_moon):.1f} м/с")
print(f"Время перелёта: {FLIGHT_DAYS} дней ({FLIGHT_DAYS/365.25:.2f} лет)")
print(f"Положение Титана на момент прибытия: {r_titan_arrival/1000} км")

# Решение задачи Ламберта от Луны к Титану
try:
    v_sc_departure, v_sc_arrival = lambert_solution(
        r_moon, r_titan_arrival,tof_seconds, use_long_way=False)
    print(f"Требуемая гелиоцентрическая скорость отлёта: {np.linalg.norm(v_sc_departure):.1f} м/с")
    delta_v = v_sc_departure - v_moon
    print(f"Необходимый импульс относительно Луны: {np.linalg.norm(delta_v):.1f} м/с")
    print(f"Вектор импульса (в гелиоцентрической системе): {delta_v} м/с")
except Exception as e:
    print(f"Ошибка решения Ламберта: {e}")
    sys.exit(1)

# Стартовое положение: от центра Луны на высоте 10 000 км
R_moon = 1737e3
h_start = 10000e3
offset_distance = R_moon + h_start
if np.linalg.norm(delta_v) > 0:
    dir_offset = delta_v / np.linalg.norm(delta_v)
else:
    dir_offset = np.array([1.0, 0.0, 0.0])
offset = dir_offset * offset_distance
r0_sc = r_moon + offset
v0_sc = v_sc_departure.copy()
state0 = np.concatenate([r0_sc, v0_sc])
print(f"Начальное смещение от центра Луны: {offset_distance/1000:.0f} км")
print(f"Стартовая гелиоцентрическая скорость КА: {np.linalg.norm(v0_sc):.1f} м/с")


# =============================================================================
# 4. Итерационная коррекция начальной скорости
# =============================================================================
def compute_residual(v0_guess: np.ndarray) -> np.ndarray:
    """
    Интегрирует траекторию с заданной начальной скоростью и возвращает
    вектор невязки: разность между положением КА и положением Титана
    в момент минимального расстояния
    Параметры:
        v0_guess: предполагаемая начальная гелиоцентрическая скорость, м/с
    Возвращает:
        Вектор невязки (dx, dy, dz), м
    """
    state0_guess = np.concatenate([r0_sc, v0_guess])
    t_guess, states_guess = integrate_trajectory(
        start_et=start_et,
        end_et=arrival_et + POST_FLIGHT_DAYS * 86400.0,
        state0=state0_guess,
        method=METHOD,
        rtol=RTOL,
        atol=ATOL,
        adaptive=True
    )
    min_dist2 = np.inf
    idx_min = 0
    for i, t in enumerate(t_guess):
        r_titan = get_body_position_helio(t, 606)
        dr = states_guess[i, :3] - r_titan
        dist2 = np.dot(dr, dr)
        if dist2 < min_dist2:
            min_dist2 = dist2
            idx_min = i
    r_sc_min = states_guess[idx_min, :3]
    r_titan_min = get_body_position_helio(t_guess[idx_min], 606)
    return r_sc_min - r_titan_min

print("\nНачинаем итерационную коррекцию начальной скорости...")
v0_current = v0_sc.copy()
for iter_num in range(5):
    print(f"\nИтерация {iter_num+1}:")
    res = compute_residual(v0_current)
    norm_res = np.linalg.norm(res)
    print(f"  Невязка по положению: {norm_res/1000:.0f} км")
    if norm_res < 1e5:
        print("  Достигнута требуемая точность.")
        break

    def residual_func(delta_v):
        return compute_residual(v0_current + delta_v)

    sol = root(residual_func, np.zeros(3), method='hybr', options={'xtol': 1e-12})
    delta_v_opt = sol.x
    v0_current += delta_v_opt
    print(f"  Поправка скорости: {np.linalg.norm(delta_v_opt):.2f} м/с")

# Обновление начальной скорости
v0_sc = v0_current
state0 = np.concatenate([r0_sc, v0_sc])
print(f"\nИтоговая стартовая скорость: {np.linalg.norm(v0_sc):.1f} м/с")


# =============================================================================
# 5. Финальное интегрирование траектории
# =============================================================================
end_et = arrival_et + POST_FLIGHT_DAYS * 86400.0
print(f"\nФинальное интегрирование траектории до {spice.et2utc(end_et, 'C', 0)}...")
t_arr, states_arr = integrate_trajectory(
    start_et=start_et,
    end_et=end_et,
    state0=state0,
    method=METHOD,
    rtol=RTOL,
    atol=ATOL,
    adaptive=True
)
print(f"Интегрирование завершено. Получено {len(t_arr)} точек")
print(f"Конечное время: {spice.et2utc(t_arr[-1], 'C', 0)}")
print()


# =============================================================================
# 6. Поиск максимального сближения с Титаном
# =============================================================================
distances = []
for i, t in enumerate(t_arr):
    dist = np.linalg.norm(states_arr[i, :3] - get_body_position_helio(t, 606))
    distances.append(dist)
min_dist = min(distances)
min_idx = distances.index(min_dist)
print(f"\n=== РЕЗУЛЬТАТЫ ===")
print(f"Минимальное расстояние до Титана: {min_dist/1000:.0f} км")
print(f"Время минимального сближения: {spice.et2utc(t_arr[min_idx], 'C', 0)}")
r_sc_min = states_arr[min_idx, :3]
v_sc_min = states_arr[min_idx, 3:]
r_titan_min = get_body_position_helio(t_arr[min_idx], 606)
v_titan_min = get_body_velocity_helio(t_arr[min_idx], 606)
v_rel = v_sc_min - v_titan_min
v_rel_norm = np.linalg.norm(v_rel)
print(f"Относительная скорость: {v_rel_norm/1000:.2f} км/с")


# =============================================================================
# 7. Сохранение результатов
# =============================================================================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
np.savez_compressed(
    OUTPUT_FILE,
    t=t_arr,
    states=states_arr,
    start_et=start_et,
    end_et=t_arr[-1],
    start_utc=START_UTC,
    delta_v=delta_v,
    flight_days=FLIGHT_DAYS,
    min_distance=min_dist,
    relative_speed=v_rel_norm,
    method=METHOD,
    rtol=RTOL,
    atol=ATOL
)
print(f"\nТраектория сохранена в {OUTPUT_FILE}")


# =============================================================================
# 8. Сохранение отчёта о полёте
# =============================================================================
report_file = "data/titan_flight_report.txt"
with open(report_file, "w", encoding="utf-8") as f:
    f.write("=== ОТЧЕТ О ПОЛЕТЕ К ТИТАНУ ===\n\n")
    f.write(f"Стартовое тело: Луна\n")
    f.write(f"Высота старта: {h_start/1000:.0f} км\n")
    f.write(f"Необходимый импульс (delta_v): {delta_v} м/с\n")
    f.write(f"Норма delta_v: {np.linalg.norm(delta_v):.1f} м/с\n")
    f.write(f"Время полета (по Ламберту): {FLIGHT_DAYS} дней ({FLIGHT_DAYS/365.25:.2f} лет)\n")
    actual_flight_days = (t_arr[min_idx] - start_et) / 86400.0
    f.write(f"Фактическое время до минимального сближения: {actual_flight_days:.2f} дней\n")
    f.write(f"Минимальное расстояние до Титана: {min_dist/1000:.0f} км\n")
    # Скорости
    v_magnitudes = np.linalg.norm(states_arr[:, 3:], axis=1)
    avg_speed = np.mean(v_magnitudes)
    min_speed = np.min(v_magnitudes)
    max_speed = np.max(v_magnitudes)
    f.write(f"Средняя скорость КА за весь полет: {avg_speed/1000:.3f} км/с\n")
    f.write(f"Минимальная скорость КА: {min_speed/1000:.3f} км/с\n")
    f.write(f"Максимальная скорость КА: {max_speed/1000:.3f} км/с\n")
    f.write(f"Относительная скорость при сближении: {v_rel_norm/1000:.3f} км/с\n")
    f.write(f"\nДата старта: {START_UTC}\n")
    f.write(f"Дата минимального сближения: {spice.et2utc(t_arr[min_idx], 'C', 0)}\n")

print(f"\nОтчёт о полёте сохранён в {report_file}")
print("Готово.")
