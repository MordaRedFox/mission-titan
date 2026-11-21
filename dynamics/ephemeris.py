"""
Провайдеры эфемерид: прямой SPICE и быстрая таблица Чебышева.
Паттерн Strategy: динамика зависит только от интерфейса
EphemerisProvider. Таблица строится один раз по SPICE и далее
вычисляется чистым NumPy без обращений к SPICE.
Степень полинома подбирается эмпирически: ряд усекается до нескольких
степеней, и для каждой измеряется фактическая невязка против SPICE.
Рост степени прекращается, когда улучшение исчезает - это шумовой пол
самих эфемерид (микроразрывы между записями SPK), который принципиально
не устраняется повышением степени
"""

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Protocol, Sequence
import numpy as np
import spiceypy as spice
from numpy.polynomial import chebyshev as cheb


logger = logging.getLogger(__name__)
_FRAME: str = "J2000"
_ABERRATION: str = "NONE"
_OBSERVER: str = "10"
# Версия формата кэша, меняется при изменении алгоритма построения
_CACHE_VERSION: int = 3
# Максимальная (пробная) степень и лестница кандидатов на усечение
DEGREE_MAX: int = 28
DEGREE_LADDER: tuple[int, ...] = (10, 14, 18, 22, 26, 28)
# Целевая точность (стремление) и пороги приемки (обязательство)
# Пороги обоснованы допуском пристрелки 1000 м: даже 20 м дают 2 % вклада
POS_TARGET_M: float = 5.0e-2
VEL_TARGET_M_S: float = 1.0e-5
POS_LIMIT_M: float = 20.0
VEL_LIMIT_M_S: float = 5.0e-3
# Относительное улучшение, ниже которого считаем достигнутым шумовой пол
_SATURATION: float = 0.90


class EphemerisProvider(Protocol):
    """
    Интерфейс доступа к гелиоцентрическим состояниям тел
    """

    @property
    def naif_ids(self) -> tuple[int, ...]:
        """
        Упорядоченные NAIF ID тел каталога
        Returns:
            Кортеж идентификаторов
        """
        ...

    def positions(self, et: float) -> np.ndarray:
        """
        Положения всех тел каталога в фиксированном порядке
        Args:
            et: Эфемеридное время TDB, с
        Returns:
            Массив формы (n_bodies, 3), м
        """
        ...

    def state(self, et: float, naif_id: int) -> np.ndarray:
        """
        Полное состояние одного тела
        Args:
            et: Эфемеридное время TDB, с
            naif_id: NAIF ID тела
        Returns:
            Вектор [x, y, z, vx, vy, vz], (м, м/с)
        """
        ...


class SpiceEphemeris:
    """
    Прямой доступ к SPICE: эталон точности, медленный провайдер
    """

    def __init__(self, naif_ids: Sequence[int]) -> None:
        """
        Инициализирует провайдер списком тел
        Args:
            naif_ids: Упорядоченные NAIF ID тел
        Returns:
            None
        """
        self._ids = tuple(int(i) for i in naif_ids)

    @property
    def naif_ids(self) -> tuple[int, ...]:
        """
        Упорядоченные NAIF ID тел каталога
        Returns:
            Кортеж идентификаторов
        """
        return self._ids

    def state(self, et: float, naif_id: int) -> np.ndarray:
        """
        Возвращает состояние тела напрямую из SPICE
        Args:
            et: Эфемеридное время TDB, с
            naif_id: NAIF ID тела
        Returns:
            Вектор [x, y, z, vx, vy, vz], (м, м/с)
        """
        raw, _ = spice.spkezr(str(naif_id), et, _FRAME,
                              _ABERRATION, _OBSERVER)
        return np.asarray(raw, dtype=float) * 1000.0

    def positions(self, et: float) -> np.ndarray:
        """
        Положения всех тел каталога
        Args:
            et: Эфемеридное время TDB, с
        Returns:
            Массив формы (n_bodies, 3), м
        """
        return np.array([self.state(et, i)[:3] for i in self._ids])


class ChebyshevEphemeris:
    """
    Табличные эфемериды на полиномах Чебышева.
    Покрытие разбито на равные интервалы. На каждом положение тела
    аппроксимировано рядом Чебышева. Скорость получается аналитическим
    дифференцированием ряда, а не конечными разностями
    """

    def __init__(self, naif_ids: Sequence[int], t0: float, segment: float,
                 pos_coef: np.ndarray, vel_coef: np.ndarray) -> None:
        """
        Инициализирует таблицу готовыми коэффициентами
        Args:
            naif_ids: Упорядоченные NAIF ID тел
            t0: Начало покрытия, с TDB
            segment: Длина интервала аппроксимации, с
            pos_coef: Коэффициенты положений, (n_seg, degree + 1, 3n)
            vel_coef: Коэффициенты скоростей, (n_seg, degree + 1, 3n)
        Returns:
            None
        """
        self._ids = tuple(int(i) for i in naif_ids)
        self._t0 = float(t0)
        self._segment = float(segment)
        self._pos = pos_coef
        self._vel = vel_coef
        self._degree = int(pos_coef.shape[1]) - 1
        self._n = len(self._ids)
        self._t1 = t0 + segment * pos_coef.shape[0]
        self._order = np.arange(self._degree + 1, dtype=float)
        self._index = {b: k for k, b in enumerate(self._ids)}

    @property
    def naif_ids(self) -> tuple[int, ...]:
        """
        Упорядоченные NAIF ID тел каталога
        Returns:
            Кортеж идентификаторов
        """
        return self._ids

    @property
    def degree(self) -> int:
        """
        Степень аппроксимирующего полинома
        Returns:
            Степень полинома Чебышева
        """
        return self._degree

    @property
    def segment(self) -> float:
        """
        Длина интервала аппроксимации
        Returns:
            Длина интервала, с
        """
        return self._segment

    @property
    def coverage(self) -> tuple[float, float]:
        """
        Интервал покрытия таблицы
        Returns:
            Пара (t_start, t_end), с TDB
        """
        return self._t0, self._t1

    @property
    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Коэффициенты положений и скоростей
        Returns:
            Пара массивов (pos_coef, vel_coef)
        """
        return self._pos, self._vel

    def _basis(self, et: float) -> tuple[int, np.ndarray]:
        """
        Вычисляет индекс интервала и базис T_j(x)
        Args:
            et: Эфемеридное время TDB, с
        Returns:
            Пара (индекс интервала, вектор базиса длины degree + 1)
        Raises:
            ValueError: Если эпоха вне покрытия таблицы
        """
        u = (et - self._t0) / self._segment
        k = int(math.floor(u))
        if not 0 <= k < self._pos.shape[0]:
            raise ValueError(
                f"ET {et:.3f} вне покрытия таблицы "
                f"[{self._t0:.3f}, {self._t1:.3f}]"
            )
        x = min(1.0, max(-1.0, 2.0 * (u - k) - 1.0))
        return k, np.cos(self._order * math.acos(x))

    def positions(self, et: float) -> np.ndarray:
        """
        Положения всех тел одним матричным умножением
        Args:
            et: Эфемеридное время TDB, с
        Returns:
            Массив формы (n_bodies, 3), м
        """
        k, basis = self._basis(et)
        return (basis @ self._pos[k]).reshape(self._n, 3)

    def state(self, et: float, naif_id: int) -> np.ndarray:
        """
        Полное состояние одного тела из таблицы
        Args:
            et: Эфемеридное время TDB, с
            naif_id: NAIF ID тела
        Returns:
            Вектор [x, y, z, vx, vy, vz], (м, м/с)
        """
        k, basis = self._basis(et)
        base = 3 * self._index[naif_id]
        col = slice(base, base + 3)
        pos = basis @ self._pos[k][:, col]
        vel = (basis @ self._vel[k][:, col]) * (2.0 / self._segment)
        return np.concatenate((pos, vel))

    @classmethod
    def build(cls, source: EphemerisProvider, t0: float, t1: float,
              segment: float = 2.0 * 86400.0, degree: int = DEGREE_MAX
              ) -> "ChebyshevEphemeris":
        """
        Строит таблицу по эталонному провайдеру
        Args:
            source: Эталонный провайдер (обычно SpiceEphemeris)
            t0: Начало покрытия, с TDB
            t1: Конец покрытия, с TDB
            segment: Длина интервала аппроксимации, с
            degree: Степень полинома Чебышева
        Returns:
            Готовая таблица ChebyshevEphemeris
        """
        n_seg = int(math.ceil((t1 - t0) / segment))
        n_val = 3 * len(source.naif_ids)
        nodes = np.cos(np.pi * np.arange(degree + 1) / degree)
        vander = cheb.chebvander(nodes, degree)
        pos = np.empty((n_seg, degree + 1, n_val), dtype=float)
        logger.info(
            "Построение таблицы эфемерид: %d интервалов по %.3f сут, "
            "степень %d", n_seg, segment / 86400.0, degree,
        )
        for s in range(n_seg):
            mid = t0 + (s + 0.5) * segment
            half = 0.5 * segment
            samples = np.empty((degree + 1, n_val), dtype=float)
            for j, x in enumerate(nodes):
                samples[j] = source.positions(mid + half * x).ravel()
            pos[s] = np.linalg.solve(vander, samples)
        vel = np.zeros_like(pos)
        vel[:, :degree, :] = cheb.chebder(pos, m=1, axis=1)
        return cls(source.naif_ids, t0, segment, pos, vel)

    def truncated(self, degree: int) -> "ChebyshevEphemeris":
        """
        Строит таблицу меньшей степени усечением ряда Чебышева.
        Усечение ряда Чебышева близко к оптимальному равномерному
        приближению, поэтому повторное обращение к SPICE не требуется
        Args:
            degree: Желаемая степень (не выше текущей)
        Returns:
            Новая таблица с усечёнными коэффициентами
        """
        degree = min(degree, self._degree)
        pos = self._pos[:, :degree + 1, :].copy()
        vel = np.zeros_like(pos)
        if degree > 0:
            vel[:, :degree, :] = cheb.chebder(pos, m=1, axis=1)
        return ChebyshevEphemeris(
            self._ids, self._t0, self._segment, pos, vel)

    def errors_by_body(self, source: EphemerisProvider, n_probe: int = 200
                       ) -> dict[int, tuple[float, float]]:
        """
        Максимальные невязки таблицы против эталона по каждому телу
        Args:
            source: Эталонный провайдер
            n_probe: Число случайных контрольных эпох
        Returns:
            Словарь naif_id -> (невязка положения м, невязка скорости м/с)
        """
        rng = np.random.default_rng(42)
        times = rng.uniform(self._t0, self._t1 - 1.0, n_probe)
        pos_err = np.zeros(self._n, dtype=float)
        vel_err = np.zeros(self._n, dtype=float)
        scale = 2.0 / self._segment
        for raw_et in times:
            et = float(raw_et)
            k, basis = self._basis(et)
            pos = (basis @ self._pos[k]).reshape(self._n, 3)
            vel = (basis @ self._vel[k]).reshape(self._n, 3) * scale
            for i, body in enumerate(self._ids):
                ref = source.state(et, body)
                pos_err[i] = max(
                    pos_err[i], float(np.max(np.abs(pos[i] - ref[:3]))))
                vel_err[i] = max(
                    vel_err[i], float(np.max(np.abs(vel[i] - ref[3:]))))
        return {b: (float(pos_err[i]), float(vel_err[i]))
                for i, b in enumerate(self._ids)}

    def errors(self, source: EphemerisProvider, n_probe: int = 200
               ) -> tuple[float, float]:
        """
        Максимальные невязки таблицы по всем телам
        Args:
            source: Эталонный провайдер
            n_probe: Число случайных контрольных эпох
        Returns:
            Пара (невязка положения м, невязка скорости м/с)
        """
        per_body = self.errors_by_body(source, n_probe)
        return (max(v[0] for v in per_body.values()),
                max(v[1] for v in per_body.values()))

    def max_error(self, source: EphemerisProvider, n_probe: int = 200
                  ) -> float:
        """
        Максимальное отклонение таблицы от эталона по положению
        Args:
            source: Эталонный провайдер
            n_probe: Число случайных контрольных эпох
        Returns:
            Максимальная невязка по положению, м
        """
        return self.errors(source, n_probe)[0]


def _choose_degree(full: ChebyshevEphemeris, source: EphemerisProvider,
                   degree_min: int) -> ChebyshevEphemeris:
    """
    Подбирает степень по фактической невязке с остановкой на шумовом полу
    Args:
        full: Таблица максимальной степени
        source: Эталонный провайдер
        degree_min: Минимально допустимая степень
    Returns:
        Таблица выбранной степени
    """
    best: ChebyshevEphemeris | None = None
    best_err = math.inf
    prev: ChebyshevEphemeris | None = None
    prev_err = math.inf
    for degree in DEGREE_LADDER:
        if degree < degree_min or degree > full.degree:
            continue
        candidate = full.truncated(degree)
        pos_err, vel_err = candidate.errors(source, n_probe=120)
        logger.info(
            "  проба степени %2d: положение %.3e м, скорость %.3e м/с",
            degree, pos_err, vel_err,
        )
        if pos_err < best_err:
            best, best_err = candidate, pos_err
        if pos_err <= POS_TARGET_M and vel_err <= VEL_TARGET_M_S:
            logger.info("Достигнута целевая точность на степени %d", degree)
            return candidate
        if prev is not None and pos_err > _SATURATION * prev_err:
            logger.info(
                "Шумовой пол эфемерид ~%.2f м; выбрана степень %d",
                pos_err, prev.degree,
            )
            return prev
        prev, prev_err = candidate, pos_err
    if best is None:
        return full.truncated(max(degree_min, DEGREE_LADDER[0]))
    return best


def load_or_build(source: EphemerisProvider,kernel_paths: Sequence[Path],
                  t0: float, t1: float, cache_dir: Path,
                  segment: float = 2.0 * 86400.0, degree: int = 8
                  ) -> ChebyshevEphemeris:
    """
    Загружает таблицу из кэша или строит ее с подбором степени.
    Таблица строится один раз при максимальной степени, затем усекается
    до наименьшей степени, обеспечивающей целевую точность либо шумовой
    пол эфемерид. Готовая таблица обязательно валидируется против SPICE
    Args:
        source: Эталонный провайдер для построения
        kernel_paths: Файлы кернелов (для ключа кэша)
        t0: Начало покрытия, с TDB
        t1: Конец покрытия, с TDB
        cache_dir: Каталог кэша
        segment: Длина интервала аппроксимации, с
        degree: Минимальная степень полинома Чебышева
    Returns:
        Таблица ChebyshevEphemeris
    Raises:
        RuntimeError: Если невязки превышают пороги приемки
    """
    payload = {
        "version": _CACHE_VERSION,
        "kernels": sorted([[p.name, p.stat().st_size] for p in kernel_paths]),
        "ids": list(source.naif_ids),
        "t0": t0,
        "t1": t1,
        "segment": segment,
        "degree_min": degree,
        "degree_max": DEGREE_MAX,
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"eph_{key}.npz"
    if path.exists():
        with np.load(path) as data:
            table = ChebyshevEphemeris(
                data["ids"], float(data["t0"]), float(data["segment"]),
                data["pos"], data["vel"],
            )
        logger.info("Таблица эфемерид загружена из кэша %s", path.name)
    else:
        full = ChebyshevEphemeris.build(
            source, t0, t1, segment, DEGREE_MAX)
        logger.info("Подбор степени по фактической невязке:")
        table = _choose_degree(full, source, degree)
        pos, vel = table.coefficients
        np.savez(path, ids=np.array(table.naif_ids), t0=t0,
                 segment=table.segment, pos=pos, vel=vel)
        logger.info("Таблица эфемерид сохранена в кэш %s", path.name)
    per_body = table.errors_by_body(source, n_probe=200)
    pos_err = max(v[0] for v in per_body.values())
    vel_err = max(v[1] for v in per_body.values())
    logger.info(
        "Валидация таблицы (степень %d): положение %.3e м, "
        "скорость %.3e м/с", table.degree, pos_err, vel_err,
    )
    worst = sorted(per_body.items(), key=lambda kv: -kv[1][0])[:3]
    for body, (pe, ve) in worst:
        logger.info("  тело %3d: положение %.3e м, скорость %.3e м/с",
                    body, pe, ve)
    if pos_err > POS_LIMIT_M or vel_err > VEL_LIMIT_M_S:
        raise RuntimeError(
            f"Таблица не прошла валидацию: положение {pos_err:.3e} м "
            f"(лимит {POS_LIMIT_M} м), скорость {vel_err:.3e} м/с "
            f"(лимит {VEL_LIMIT_M_S} м/с). Удалите кэш и уменьшите "
            f"segment_days."
        )
    return table
