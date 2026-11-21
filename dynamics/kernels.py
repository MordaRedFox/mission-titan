"""
Управление SPICE-кернелами: детерминированная загрузка и аудит.
Мета-кернелы не используются: порядок загрузки задается явно в Python
абсолютными путями, что исключает ошибки раскрытия PATH_SYMBOLS и
зависимость от текущей рабочей директории
"""

import logging
from typing import Final
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError


logger = logging.getLogger(__name__)
# Приоритет типов: LSK -> PCK -> GM -> FK -> binary PCK -> SPK
_PRIORITY: Final[dict[str, int]] = {
    ".tls": 0,
    ".tpc": 1,
    ".tcp": 1,
    ".tf":  2,
    ".bpc": 3,
    ".bsp": 4
}


def _sort_key(path: Path) -> tuple[int, int, str]:
    """
    Строит ключ сортировки, ставящий GM-кернел после обычного PCK
    Args:
        path: Путь к файлу кернела
    Returns:
        Кортеж (приоритет типа, флаг GM-переопределения, имя файла)
    """
    is_gm = int("gm" in path.stem.lower())
    return _PRIORITY.get(path.suffix.lower(), 9), is_gm, path.name


def load_kernels(kernel_dir: Path) -> tuple[Path, ...]:
    """
    Загружает все кернелы каталога в физически корректном порядке
    Args:
        kernel_dir: Каталог с файлами кернелов
    Returns:
        Кортеж абсолютных путей загруженных файлов
    Raises:
        FileNotFoundError: Если каталог пуст или отсутствует LSK/SPK
    """
    kernel_dir = Path(kernel_dir).resolve()
    if not kernel_dir.is_dir():
        raise FileNotFoundError(f"Нет каталога кернелов: {kernel_dir}")
    files = sorted(
        (p for p in kernel_dir.iterdir()
         if p.suffix.lower() in _PRIORITY),
        key=_sort_key,
    )
    if not files:
        raise FileNotFoundError(f"Кернелы не найдены в {kernel_dir}")
    exts = {p.suffix.lower() for p in files}
    if ".tls" not in exts:
        raise FileNotFoundError("Отсутствует LSK (*.tls), нужен str2et")
    if ".bsp" not in exts:
        raise FileNotFoundError("Отсутствует SPK (*.bsp), нужны эфемериды")
    if not any("gm" in p.stem.lower() for p in files):
        logger.warning(
            "GM-кернел не найден: GM будут взяты из PCK эпохи DE405 "
            "(ошибка ~1e-5 в поле Титана). Создайте mission_gm.tpc"
        )
    spice.kclear()
    for path in files:
        if path.suffix.lower() == ".tcp":
            logger.warning(
                "%s: расширение .tcp вместо .tpc — переименуйте файл",
                path.name,
            )
        spice.furnsh(str(path))
        logger.info("Загружен кернел %s", path.name)
    return tuple(files)


def bodvrd(naif_id: int, item: str, maxn: int) -> list[float] | None:
    """
    Читает параметр тела из пула SPICE по числовому ID или имени
    Args:
        naif_id: NAIF ID тела
        item: Имя параметра ('GM', 'RADII', ...)
        maxn: Максимальное число запрашиваемых значений
    Returns:
        Список значений либо None, если параметр не найден
    """
    keys = [str(naif_id)]
    try:
        raw: Any = spice.bodc2n(naif_id)
        name = raw[0] if isinstance(raw, tuple) else raw
        if isinstance(name, str) and name.strip():
            keys.append(name.strip())
    except SpiceyError:
        pass
    for key in keys:
        try:
            raw = spice.bodvrd(key, item, maxn)
            values = raw[1] if isinstance(raw, tuple) else raw
            return [float(v) for v in values]
        except SpiceyError:
            continue
    return None


def check_coverage(naif_ids: Sequence[int], t0: float, t1: float,
                   n_probe: int = 7) -> None:
    """Проверяет доступность состояний всех тел на всем интервале миссии
    Args:
        naif_ids: NAIF ID проверяемых тел
        t0: Начало интервала, с TDB
        t1: Конец интервала, с TDB
        n_probe: Число контрольных эпох
    Returns:
        None
    Raises:
        RuntimeError: Если хотя бы одно тело недоступно
    """
    missing: dict[int, str] = {}
    for et in np.linspace(t0, t1, n_probe):
        for body in naif_ids:
            if body in missing:
                continue
            try:
                spice.spkezr(str(body), float(et), "J2000", "NONE", "10")
            except SpiceyError as exc:
                missing[body] = str(exc).splitlines()[0]
    if missing:
        details = "\n".join(f"  {k}: {v}" for k, v in missing.items())
        raise RuntimeError(
            "Не хватает эфемеридного покрытия для тел:\n" + details
        )
    logger.info("Покрытие эфемерид подтверждено для %d тел", len(naif_ids))
