# config/skeleton_presets.py
"""
Загрузка пресетов скелетов из JSON файла
"""

import json
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_skeleton_preset(preset_name: str = "humanoid_mediapipe") -> Optional[Dict[str, Any]]:
    """
    Загрузка пресета скелета из JSON файла

    Args:
        preset_name: Имя пресета (humanoid_mediapipe, simplified_humanoid, quadruped, facial)

    Returns:
        Конфигурация пресета или None если не найден
    """
    try:
        # Путь к JSON файлу
        json_path = os.path.join(os.path.dirname(__file__), "skeleton_presets.json")

        if not os.path.exists(json_path):
            logger.error(f"JSON файл не найден: {json_path}")
            # Создаем fallback данные
            return _create_fallback_preset(preset_name)

        # Загрузка JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Поиск пресета
        presets = data.get('skeleton_presets', {})

        if preset_name in presets:
            logger.info(f"Пресет '{preset_name}' загружен")
            return presets[preset_name]
        else:
            logger.warning(f"Пресет '{preset_name}' не найден. Доступные пресеты: {list(presets.keys())}")
            # Возвращаем первый доступный пресет
            if presets:
                first_preset = next(iter(presets.values()))
                logger.info(f"Используем пресет '{list(presets.keys())[0]}'")
                return first_preset
            else:
                return _create_fallback_preset(preset_name)

    except Exception as e:
        logger.error(f"Ошибка загрузки пресета: {e}")
        return _create_fallback_preset(preset_name)


def get_all_presets() -> Dict[str, Dict[str, Any]]:
    """Получение всех пресетов"""
    try:
        json_path = os.path.join(os.path.dirname(__file__), "skeleton_presets.json")

        if not os.path.exists(json_path):
            return {"humanoid": _create_fallback_preset("humanoid")}

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get('skeleton_presets', {})

    except Exception as e:
        logger.error(f"Ошибка загрузки всех пресетов: {e}")
        return {"humanoid": _create_fallback_preset("humanoid")}


def get_rest_pose(pose_name: str = "T_POSE") -> Optional[Dict[str, Any]]:
    """Получение позы отдыха"""
    try:
        json_path = os.path.join(os.path.dirname(__file__), "skeleton_presets.json")

        if not os.path.exists(json_path):
            return None

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        rest_poses = data.get('rest_poses', {})
        return rest_poses.get(pose_name)

    except Exception as e:
        logger.error(f"Ошибка загрузки позы отдыха: {e}")
        return None


def get_export_preset(preset_name: str = "blender_humanoid") -> Optional[Dict[str, Any]]:
    """Получение пресета экспорта"""
    try:
        json_path = os.path.join(os.path.dirname(__file__), "skeleton_presets.json")

        if not os.path.exists(json_path):
            return None

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        export_presets = data.get('export_presets', {})
        return export_presets.get(preset_name)

    except Exception as e:
        logger.error(f"Ошибка загрузки пресета экспорта: {e}")
        return None


def _create_fallback_preset(preset_name: str) -> Dict[str, Any]:
    """Создание fallback пресета если JSON не найден"""
    fallback_presets = {
        "humanoid_mediapipe": {
            "name": "Humanoid (MediaPipe)",
            "description": "Стандартный человеческий скелет (fallback)",
            "joint_count": 33,
            "hierarchy": {
                "Hips": {"parent": None, "joint_type": "ROOT"},
                "Spine": {"parent": "Hips", "joint_type": "SPINE"},
                "Chest": {"parent": "Spine", "joint_type": "SPINE"},
                "Neck": {"parent": "Chest", "joint_type": "NECK"},
                "Head": {"parent": "Neck", "joint_type": "HEAD"},
                "LeftShoulder": {"parent": "Chest", "joint_type": "SHOULDER"},
                "LeftUpperArm": {"parent": "LeftShoulder", "joint_type": "UPPER_ARM"},
                "LeftLowerArm": {"parent": "LeftUpperArm", "joint_type": "LOWER_ARM"},
                "LeftHand": {"parent": "LeftLowerArm", "joint_type": "HAND"},
                "RightShoulder": {"parent": "Chest", "joint_type": "SHOULDER"},
                "RightUpperArm": {"parent": "RightShoulder", "joint_type": "UPPER_ARM"},
                "RightLowerArm": {"parent": "RightUpperArm", "joint_type": "LOWER_ARM"},
                "RightHand": {"parent": "RightLowerArm", "joint_type": "HAND"},
                "LeftUpperLeg": {"parent": "Hips", "joint_type": "UPPER_LEG"},
                "LeftLowerLeg": {"parent": "LeftUpperLeg", "joint_type": "LOWER_LEG"},
                "LeftFoot": {"parent": "LeftLowerLeg", "joint_type": "FOOT"},
                "RightUpperLeg": {"parent": "Hips", "joint_type": "UPPER_LEG"},
                "RightLowerLeg": {"parent": "RightUpperLeg", "joint_type": "LOWER_LEG"},
                "RightFoot": {"parent": "RightLowerLeg", "joint_type": "FOOT"}
            }
        }
    }

    if preset_name in fallback_presets:
        return fallback_presets[preset_name]
    else:
        return fallback_presets["humanoid_mediapipe"]


# Экспорт
__all__ = [
    'load_skeleton_preset',
    'get_all_presets',
    'get_rest_pose',
    'get_export_preset'
]