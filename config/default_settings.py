"""
Модуль для работы с настройками приложения
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_settings(filepath: Path) -> Dict[str, Any]:
    """
    Загружает настройки из YAML файла.

    Args:
        filepath: Путь к файлу настроек

    Returns:
        Dict: Загруженные настройки
    """
    try:
        if not filepath.exists():
            logger.warning(f"Файл настроек не найден: {filepath}")
            return create_default_settings()

        with open(filepath, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)

        # Преобразуем в плоскую структуру для удобства доступа
        flat_settings = flatten_dict(settings)

        logger.info(f"Настройки загружены из {filepath}")
        return flat_settings

    except Exception as e:
        logger.error(f"Ошибка загрузки настроек: {e}")
        return create_default_settings()

def save_settings(filepath: Path, settings: Dict[str, Any]):
    """
    Сохраняет настройки в YAML файл.

    Args:
        filepath: Путь к файлу настроек
        settings: Настройки для сохранения
    """
    try:
        # Преобразуем из плоской структуры обратно во вложенную
        nested_settings = unflatten_dict(settings)

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(nested_settings, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Настройки сохранены в {filepath}")

    except Exception as e:
        logger.error(f"Ошибка сохранения настроек: {e}")

def create_default_settings() -> Dict[str, Any]:
    """
    Создает настройки по умолчанию.

    Returns:
        Dict: Настройки по умолчанию
    """
    # Используем настройки из default_settings.yaml
    default_file = Path(__file__).parent / "default_settings.yaml"

    if default_file.exists():
        return load_settings(default_file)

    # Резервные настройки если файл не найден
    return {
        "application.name": "Mocap Pro",
        "application.version": "1.0.0",
        "application.language": "ru",
        "application.auto_save": True,
        "application.auto_save_interval": 300,

        "interface.theme": "dark",
        "interface.font_family": "Segoe UI",
        "interface.font_size": 10,

        "camera.default_resolution": "1280x720",
        "camera.default_fps": 30,

        "tracking.mode": "PRECISE",
        "tracking.enable_kalman_filter": True,

        "recording.default_fps": 30,
        "recording.default_format": "bvh",

        "logging.log_level": "INFO",
        "logging.log_to_file": True
    }

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Преобразует вложенный словарь в плоский.

    Args:
        d: Вложенный словарь
        parent_key: Префикс для ключей
        sep: Разделитель для ключей

    Returns:
        Dict: Плоский словарь
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Преобразует плоский словарь во вложенный.

    Args:
        d: Плоский словарь
        sep: Разделитель в ключах

    Returns:
        Dict: Вложенный словарь
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return result

def get_setting(settings: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Получает значение настройки по ключу.

    Args:
        settings: Словарь настроек
        key: Ключ настройки
        default: Значение по умолчанию

    Returns:
        Any: Значение настройки
    """
    return settings.get(key, default)

def set_setting(settings: Dict[str, Any], key: str, value: Any):
    """
    Устанавливает значение настройки.

    Args:
        settings: Словарь настроек
        key: Ключ настройки
        value: Значение
    """
    settings[key] = value

def merge_settings(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """
    Объединяет две группы настроек.

    Args:
        base: Базовые настройки
        overlay: Настройки для перезаписи

    Returns:
        Dict: Объединенные настройки
    """
    result = base.copy()
    result.update(overlay)
    return result