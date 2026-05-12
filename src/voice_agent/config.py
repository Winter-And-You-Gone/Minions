"""配置系统：读取 config.yaml，支持 ${ENV_NAME} 环境变量替换。"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(value: Any) -> Any:
    """递归替换字符串中的 ${ENV_NAME} 为环境变量值。"""
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            return os.environ.get(m.group(1), "")
        return _ENV_VAR_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


_config_cache: dict[str, dict] = {}


def get_config(config_path: str = "config.yaml") -> dict:
    """读取并返回配置字典。结果按路径缓存，环境变量已替换。"""
    global _config_cache
    if config_path in _config_cache:
        return _config_cache[config_path]

    load_dotenv(".env")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    resolved = _resolve_env_vars(raw)
    _config_cache[config_path] = resolved
    return resolved


def save_config(config: dict, config_path: str = "config.yaml") -> None:
    """保存配置到 YAML 文件，并清除对应路径的缓存。"""
    global _config_cache

    path = Path(config_path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

    _config_cache.pop(config_path, None)


def reload_config(config_path: str = "config.yaml") -> dict:
    """强制重新加载配置（绕过缓存）。"""
    global _config_cache
    _config_cache.pop(config_path, None)
    return get_config(config_path)
