"""
animation_trajectory_titan.py – 2D анимация перелета зонда от Луны к Титану
с вычислением положений планет на лету. Орбиты планет отображаются как
реальные траектории, вычисленные через SPICE, с пунктирной линией
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple, Any
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamics.gravity import load_kernel, get_body_position_helio


# =============================================================================
# 1. Константы и загрузка кернелов
# =============================================================================
AU = 1.496e11
DAYS_TO_SEC = 86400.0
BODIES_INFO = [
    (10,   "Sun",      "#FFD700", 200, 0.0,    0.0),
    (1,    "Mercury",  "#A9A9A9",  70, 0.387,  87.97),
    (2,    "Venus",    "#F0E68C",  90, 0.723,  224.7),
    (3,    "Earth",    "#4B9CD3", 100, 1.0,    365.25),
    (4,    "Mars",     "#CD5C5C",  80, 1.524,  687.0),
    (5,    "Jupiter",  "#CD853F", 140, 5.2,    4331.0),
    (6,    "Saturn",   "#F4E6C3", 130, 9.58,   10759.0),
    (301,  "Moon",     "#C0C0C0",  80, 0.00257, 27.3),
    (606,  "Titan",    "#B87333",  85, 0.00817, 15.945),
]
NAME_TRANSLATION = {
    "Sun": "Солнце",
    "Mercury": "Меркурий",
    "Venus": "Венера",
    "Earth": "Земля",
    "Mars": "Марс",
    "Jupiter": "Юпитер",
    "Saturn": "Сатурн",
    "Moon": "Луна",
    "Titan": "Титан",
}
KERNEL_DIR = "data/ephemeris"

def load_spice_kernels() -> None:
    """
    Загружает все кернелы из папки KERNEL_DIR
    Параметры:
        None
    Возвращает:
        None
    """
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
# 2. Необходимые для работы функции
# =============================================================================
def load_spacecraft_trajectory(trajectory_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает времена и позиции космического аппарата из .npz файла.
    Параметры:
        trajectory_file: путь к файлу траектории (формат .npz)
    Возвращает:
        t: массив времен, с
        sc_pos: массив позиций [x, y, z], м
    """
    with np.load(trajectory_file) as data:
        t = data["t"]
        states = data["states"]
    sc_pos = states[:, :3]
    return t, sc_pos

def get_planet_positions(et: float) -> Dict[str, Tuple[float, float]]:
    """
    Получает позиции всех планет и спутников для заданного времени
    Параметры:
        et: эфемеридное время TDB, с
    Возвращает:
        Словарь, где ключ - имя тела, значение - кортеж (x_au, y_au)
        координат в астрономических единицах (проекция на эклиптику)
    """
    positions = {}
    for naif_id, name, _, _, _, _ in BODIES_INFO:
        pos_m = get_body_position_helio(et, naif_id)
        x_au = pos_m[0] / AU
        y_au = pos_m[1] / AU
        positions[name] = (x_au, y_au)
    return positions

def compute_full_orbits(t_start: float, n_points_base: int = 1500
                        ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Вычисляет реальные орбиты планет (кроме Солнца, Луны и Титана) на
    интервале, охватывающем один полный оборот
    Параметры:
        t_start: начальное эфемеридное время, с
        n_points_base: базовое количество точек для орбиты
    Возвращает:
        Словарь, где ключ - имя планеты, значение - кортеж (xs, ys)
        массивов координат в а.е.
    """
    orbits = {}
    for naif_id, name, _, _, _, period_days in BODIES_INFO:
        if name in ("Sun", "Moon", "Titan"):
            continue
        if period_days <= 0:
            continue
        # У дальних планет большее количество точек (для пунктира)
        if period_days > 1000:
            n_points = int(n_points_base * 1.5)
        elif period_days > 300:
            n_points = int(n_points_base * 1.2)
        else:
            n_points = n_points_base
        mult = 1.0
        period_sec = period_days * DAYS_TO_SEC
        dt_half = mult * period_sec / 2.0
        t_min = t_start - dt_half
        t_max = t_start + dt_half
        t_orbit = np.linspace(t_min, t_max, n_points)
        xs = []
        ys = []
        for et in t_orbit:
            pos_m = get_body_position_helio(et, naif_id)
            xs.append(pos_m[0] / AU)
            ys.append(pos_m[1] / AU)
        orbits[name] = (np.array(xs), np.array(ys))
        print(f"Вычислена орбита {name} ({len(xs)} точек)")
    return orbits


# =============================================================================
# 3. Создание анимации
# =============================================================================
def animate_trajectory(t: np.ndarray, sc_pos_m: np.ndarray) -> FuncAnimation:
    """
    Создаёт анимацию движения КА с отображением орбит планет и их 
    текущих положений
    Параметры:
        t: массив эфемеридных времен, с
        sc_pos_m: массив позиций КА [x, y, z], м
    Возвращает:
        FuncAnimation: объект анимации matplotlib
    """
    t0 = t[0]
    sc_pos_au = sc_pos_m / AU
    sc_xy = sc_pos_au[:, :2]
    print("Вычисление орбит планет (это может занять некоторое время)...")
    orbits = compute_full_orbits(t0)
    print("Орбиты вычислены")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlabel("X (а.е.)", fontsize=12)
    ax.set_ylabel("Y (а.е.)", fontsize=12)
    ax.set_title("Траектория космического аппарата к Титану\n"
                 "Проекция на плоскость эклиптики", fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal")
    ax.set_facecolor("#0a0a2a")

    # Границы осей определяются по траектории КА и орбитам планет
    max_radius = 0
    for name, (xs, ys) in orbits.items():
        r_max = max(np.sqrt(xs**2 + ys**2))
        if r_max > max_radius:
            max_radius = r_max
    margin = 0.5 * max_radius
    ax.set_xlim(-max_radius - margin, max_radius + margin)
    ax.set_ylim(-max_radius - margin, max_radius + margin)

    # Рисует орбиты планет
    for name, (xs, ys) in orbits.items():
        color = next(
            (c for _, n, c, _, _, _ in BODIES_INFO if n == name), "white")
        ax.plot(xs, ys, color=color, alpha=0.6, linewidth=1.5, linestyle="--")
    # Рисует планеты и спутники
    planet_scatters = {}
    for _, name, color, size, _, _ in BODIES_INFO:
        scat = ax.scatter([], [], c=color, s=size, marker="o", label=name,
                          alpha=0.9, edgecolors="black", linewidth=1.0, zorder=3)
        planet_scatters[name] = scat
    # Рисует КА и его траекторию
    spacecraft_point = ax.scatter([], [], c="red", s=60, marker="^",
                                 label="КА", alpha=1.0, edgecolors="white",
                                 linewidth=1.5, zorder=5)
    trajectory_line, = ax.plot([], [], c="#FF6B6B", linewidth=2,
                              alpha=0.8, label="Траектория")
    # Отображает текст с информацией
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                       fontsize=10, color="white", verticalalignment="top",
                       bbox=dict(boxstyle="round", facecolor="black",
                                 alpha=0.8, edgecolor="white"), zorder=6)
    # Отображает легенду графика
    legend_elements = []
    for _, name, color, _, _, _ in BODIES_INFO:
        label = NAME_TRANSLATION.get(name, name)
        legend_elements.append(plt.Line2D([0], [0], marker="o", color="w",
                                          markerfacecolor=color, markersize=8,
                                          label=label))
    legend_elements.append(plt.Line2D([0], [0], marker="^", color="w",
                                      markerfacecolor="red", markersize=10,
                                      label="КА"))
    legend_elements.append(plt.Line2D([0], [0], color="#FF6B6B", linewidth=2,
                                      label="Траектория"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
             framealpha=0.9, facecolor="black", edgecolor="white",
             labelcolor="white")
    ani_ref = [None]

    def update(frame: int) -> List[Any]:
        """
        Обновляет кадр анимации
        Параметры:
            frame: номер текущего кадра (индекс в массиве времени)
        Возвращает:
            Список графических объектов, которые необходимо перерисовать
        """
        et = t[frame]
        delta_sec = et - t0
        days = delta_sec / DAYS_TO_SEC
        planet_positions = get_planet_positions(et)

        # Обновление данных на графике
        for name, (x_au, y_au) in planet_positions.items():
            planet_scatters[name].set_offsets([[x_au, y_au]])
        sc_x, sc_y = sc_xy[frame]
        spacecraft_point.set_offsets([[sc_x, sc_y]])
        trajectory_line.set_data(sc_xy[:frame+1, 0], sc_xy[:frame+1, 1])
        if "Titan" in planet_positions:
            titan_pos = np.array(planet_positions["Titan"])
            sc_pos = np.array([sc_x, sc_y])
            dist_au = np.linalg.norm(sc_pos - titan_pos)
            dist_km = dist_au * AU / 1000
        else:
            dist_km = 0
            dist_au = 0
        # Проверка условия остановки анимации
        if dist_km <= 130000 and ani_ref[0] is not None:
            ani_ref[0].event_source.stop()
        # Текстовая информация
        info_lines = [
            f"Время: {days:.1f} дней ({days/365.25:.1f} лет)",
            f"Расстояние до Титана: {dist_km/1e6:.2f} млн км ({dist_au:.3f} а.е.)"
        ]
        info_text.set_text('\n'.join(info_lines))
        return list(planet_scatters.values()) + [spacecraft_point,
                                                 trajectory_line, info_text]

    n_frames = len(t)
    print(f"Всего кадров: {n_frames}")
    ani = FuncAnimation(fig, update, frames=range(n_frames),
                       interval=50, blit=False, repeat=False)
    ani_ref[0] = ani
    plt.tight_layout()
    plt.show()
    return ani


# =============================================================================
# 4. Главная функция и запуск скрипта
# =============================================================================
def main() -> None:
    print("Загрузка SPICE кернелов...")
    load_spice_kernels()
    print("Загрузка траектории КА...")
    t, sc_pos = load_spacecraft_trajectory("data/trajectory/trajectory.npz")
    print(f"Загружено {len(t)} точек траектории")
    print("Запуск анимации...")
    animate_trajectory(t, sc_pos)

if __name__ == "__main__":
    main()
