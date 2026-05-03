"""日志配置。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGER_NAME = "voice_agent"


def _to_level(level: int | str | None, default: int) -> int:
    if level is None:
        return default
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), default)


def setup_logging(
    debug: bool = False,
    console: bool = True,
    console_level: int | str | None = None,
    log_file: str | None = None,
) -> logging.Logger:
    """配置 voice_agent 日志。

    Parameters
    ----------
    debug : bool
        是否开启 DEBUG 级别（同时影响 console 和 file 的默认级别）。
    console : bool
        是否向终端输出日志。
    console_level : int | str | None
        终端日志级别，None 则 debug=True 时 DEBUG，否则 INFO。
    log_file : str | None
        日志文件路径。设置后所有日志（DEBUG/INFO 以上）写入该文件。
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if console:
        c_level = _to_level(
            console_level,
            logging.DEBUG if debug else logging.INFO,
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(c_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)
