"""
Модуль: BVH Exporter (Экспорт анимации в BVH формат)
Версия: 1.0.0
Автор: Mocap Pro Team

Экспорт данных скелетной анимации в стандартный формат BVH (BioVision Hierarchy).
Поддержка версий BVH 1.0 и 2.0, оптимизация для Blender, Maya, Unity.
"""

import os
import numpy as np
import json
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from core.skeleton import Skeleton, Bone, BoneType as JointType
from core.animation_recorder import AnimationLayer, Keyframe


# Поддерживаемые версии BVH
class BVHVersion(Enum):
    BVH_1_0 = "1.0"  # Стандартный BVH
    BVH_2_0 = "2.0"  # Расширенный BVH с поддержкой масштаба
    MOTION_BUILDER = "mb"  # Совместимость с MotionBuilder


# Режимы вращения
class RotationOrder(Enum):
    XYZ = "XYZ"
    XZY = "XZY"
    YXZ = "YXZ"
    YZX = "YZX"
    ZXY = "ZXY"
    ZYX = "ZYX"


# Типы данных каналов
class ChannelType(Enum):
    XPOSITION = "Xposition"
    YPOSITION = "Yposition"
    ZPOSITION = "Zposition"
    XROTATION = "Xrotation"
    YROTATION = "Yrotation"
    ZROTATION = "Zrotation"
    XSCALE = "Xscale"  # Только для BVH 2.0
    YSCALE = "Yscale"
    ZSCALE = "Zscale"


@dataclass
class BVHExportSettings:
    """Настройки экспорта BVH"""
    version: BVHVersion = BVHVersion.BVH_1_0
    rotation_order: RotationOrder = RotationOrder.ZYX
    frame_rate: float = 30.0
    scale_factor: float = 1.0
    apply_root_motion: bool = True
    convert_to_cm: bool = True  # Конвертация метров в сантиметры
    optimize_channels: bool = True
    compression_threshold: float = 0.001
    include_metadata: bool = True
    target_software: str = "blender"  # blender, maya, unity, unreal

    def to_dict(self) -> Dict:
        return {
            "version": self.version.value,
            "rotation_order": self.rotation_order.value,
            "frame_rate": self.frame_rate,
            "scale_factor": self.scale_factor,
            "apply_root_motion": self.apply_root_motion,
            "convert_to_cm": self.convert_to_cm,
            "optimize_channels": self.optimize_channels,
            "compression_threshold": self.compression_threshold,
            "include_metadata": self.include_metadata,
            "target_software": self.target_software
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BVHExportSettings':
        settings = cls()
        if "version" in data:
            settings.version = BVHVersion(data["version"])
        if "rotation_order" in data:
            settings.rotation_order = RotationOrder(data["rotation_order"])
        if "frame_rate" in data:
            settings.frame_rate = float(data["frame_rate"])
        if "scale_factor" in data:
            settings.scale_factor = float(data["scale_factor"])
        if "apply_root_motion" in data:
            settings.apply_root_motion = bool(data["apply_root_motion"])
        if "convert_to_cm" in data:
            settings.convert_to_cm = bool(data["convert_to_cm"])
        if "optimize_channels" in data:
            settings.optimize_channels = bool(data["optimize_channels"])
        if "compression_threshold" in data:
            settings.compression_threshold = float(data["compression_threshold"])
        if "include_metadata" in data:
            settings.include_metadata = bool(data["include_metadata"])
        if "target_software" in data:
            settings.target_software = data["target_software"]
        return settings


@dataclass
class BVHJoint:
    """Представление сустава в BVH формате"""
    name: str
    offset: List[float]  # [x, y, z]
    channels: List[ChannelType]
    children: List['BVHJoint'] = field(default_factory=list)
    is_end_site: bool = False
    end_site_offset: Optional[List[float]] = None

    def get_channel_count(self) -> int:
        """Возвращает количество каналов"""
        return len(self.channels)

    def get_all_joints(self) -> List['BVHJoint']:
        """Рекурсивно возвращает все суставы"""
        joints = [self]
        for child in self.children:
            joints.extend(child.get_all_joints())
        return joints

    def to_bvh_hierarchy(self, indent: int = 0) -> List[str]:
        """Генерирует строки для секции HIERARCHY"""
        lines = []
        indent_str = " " * indent

        if self.is_end_site:
            lines.append(f"{indent_str}End Site")
            lines.append(f"{indent_str}{{")
            lines.append(
                f"{indent_str}  OFFSET {self.end_site_offset[0]:.6f} {self.end_site_offset[1]:.6f} {self.end_site_offset[2]:.6f}")
            lines.append(f"{indent_str}}}")
        else:
            lines.append(f"{indent_str}JOINT {self.name}" if indent > 0 else f"{indent_str}ROOT {self.name}")
            lines.append(f"{indent_str}{{")
            lines.append(f"{indent_str}  OFFSET {self.offset[0]:.6f} {self.offset[1]:.6f} {self.offset[2]:.6f}")

            if self.channels:
                channels_str = " ".join([c.value for c in self.channels])
                lines.append(f"{indent_str}  CHANNELS {len(self.channels)} {channels_str}")

            for child in self.children:
                lines.extend(child.to_bvh_hierarchy(indent + 2))

            lines.append(f"{indent_str}}}")

        return lines


class BVHExporter:
    """Основной класс экспорта в BVH формат"""

    def __init__(self, settings: Optional[BVHExportSettings] = None):
        self.settings = settings or BVHExportSettings()
        self.warnings: List[str] = []
        self.metadata: Dict[str, Any] = {}

        # Кэширование для производительности
        self._joint_cache: Dict[str, BVHJoint] = {}
        self._channel_indices: Dict[str, Tuple[int, int]] = {}

    def export(self,
               skeleton: Skeleton,
               animation_layer: AnimationLayer,
               output_path: str,
               settings: Optional[BVHExportSettings] = None) -> bool:
        """
        Экспортирует анимацию в BVH файл.

        Args:
            skeleton: Скелет для экспорта
            animation_layer: Слой анимации
            output_path: Путь для сохранения файла
            settings: Настройки экспорта (если None, используются текущие)

        Returns:
            bool: Успешность экспорта
        """
        try:
            if settings:
                self.settings = settings

            # Подготовка данных
            self._prepare_export(skeleton, animation_layer)

            # Создание директории если нужно
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Генерация BVH
            bvh_content = self._generate_bvh_content(skeleton, animation_layer)

            # Запись в файл
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(bvh_content)

            # Сохранение метаданных в отдельный файл
            if self.settings.include_metadata:
                self._save_metadata(output_path, skeleton, animation_layer)

            print(f"✅ BVH успешно экспортирован: {output_path}")
            if self.warnings:
                print("⚠️ Предупреждения:")
                for warning in self.warnings:
                    print(f"   - {warning}")

            return True

        except Exception as e:
            print(f"❌ Ошибка экспорта BVH: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _prepare_export(self, skeleton: Skeleton, animation_layer: AnimationLayer):
        """Подготовка данных к экспорту"""
        self.warnings.clear()
        self.metadata.clear()
        self._joint_cache.clear()
        self._channel_indices.clear()

        # Сбор метаданных
        self.metadata = {
            "source": "Mocap Pro",
            "version": "1.0.0",
            "export_time": np.datetime64('now').astype(str),
            "skeleton_name": skeleton.name,
            "joint_count": len(skeleton.bones),
            "animation_name": animation_layer.name,
            "frame_count": animation_layer.frame_count,
            "duration": animation_layer.duration,
            "original_frame_rate": animation_layer.frame_rate,
            "export_settings": self.settings.to_dict()
        }

        # Проверка совместимости
        self._validate_export_data(skeleton, animation_layer)

    def _validate_export_data(self, skeleton: Skeleton, animation_layer: AnimationLayer):
        """Проверяет данные на совместимость с BVH"""
        # Проверка наличия корневого сустава
        root_bone = skeleton.get_root_bone()
        if not root_bone:
            raise ValueError("Скелет не имеет корневого сустава")

        # Проверка вращений (BVH использует Euler углы)
        for joint_name, keyframes in animation_layer.keyframes.items():
            for i, kf in enumerate(keyframes):
                # Проверяем, что вращение нормализовано
                if kf.rotation is not None:
                    norm = np.linalg.norm(kf.rotation)
                    if abs(norm - 1.0) > 0.001:
                        self.warnings.append(f"Ключевой кадр {i} сустава {joint_name} имеет ненормализованное вращение")

        # Проверка масштаба для BVH 1.0
        if self.settings.version == BVHVersion.BVH_1_0:
            for bone in skeleton.bones:
                if bone.has_scale:
                    self.warnings.append(
                        f"BVH 1.0 не поддерживает масштаб. Масштаб сустава {bone.name} будет проигнорирован")

    def _generate_bvh_content(self, skeleton: Skeleton, animation_layer: AnimationLayer) -> str:
        """Генерирует содержимое BVH файла"""
        lines = []

        # 1. Секция HIERARCHY
        lines.append("HIERARCHY")

        # Создание BVH иерархии из скелета
        bvh_root = self._convert_skeleton_to_bvh(skeleton)
        hierarchy_lines = bvh_root.to_bvh_hierarchy()
        lines.extend(hierarchy_lines)

        # 2. Секция MOTION
        lines.append("MOTION")

        # Информация о кадрах
        frame_count = animation_layer.frame_count
        if self.settings.frame_rate != animation_layer.frame_rate:
            self.warnings.append(
                f"Частота кадров изменена с {animation_layer.frame_rate} на {self.settings.frame_rate}")

        lines.append(f"Frames: {frame_count}")
        lines.append(f"Frame Time: {1.0 / self.settings.frame_rate:.6f}")

        # 3. Данные анимации
        motion_data = self._generate_motion_data(skeleton, animation_layer, bvh_root)
        lines.extend(motion_data)

        return "\n".join(lines)

    def _convert_skeleton_to_bvh(self, skeleton: Skeleton) -> BVHJoint:
        """Конвертирует скелет в BVH иерархию"""
        # Находим корневой сустав
        root_bone = skeleton.get_root_bone()
        if not root_bone:
            raise ValueError("Скелет не имеет корневого сустава")

        # Рекурсивное преобразование
        return self._convert_bone_to_bvh_joint(root_bone, skeleton)

    def _convert_bone_to_bvh_joint(self, bone: Bone, skeleton: Skeleton, parent_matrix=None) -> BVHJoint:
        """Рекурсивно конвертирует кость в BVH сустав"""
        # Вычисляем offset относительно родителя
        if parent_matrix is None:
            # Корневой сустав - используем абсолютную позицию
            offset = bone.rest_position.tolist() if hasattr(bone.rest_position, 'tolist') else list(bone.rest_position)
        else:
            # Дочерний сустав - вычисляем относительный offset
            bone_world = skeleton.get_bone_world_matrix(bone.name)
            offset_matrix = np.linalg.inv(parent_matrix) @ bone_world
            offset = offset_matrix[:3, 3].tolist()

        # Масштабируем offset если нужно
        if self.settings.convert_to_cm:
            offset = [coord * 100.0 * self.settings.scale_factor for coord in offset]
        else:
            offset = [coord * self.settings.scale_factor for coord in offset]

        # Определяем каналы для сустава
        channels = self._get_channels_for_bone(bone)

        # Создаем BVHJoint
        bvh_joint = BVHJoint(
            name=bone.name,
            offset=offset,
            channels=channels
        )

        # Сохраняем в кэш
        self._joint_cache[bone.name] = bvh_joint

        # Обрабатываем детей
        children = skeleton.get_children(bone.name)
        for child_bone in children:
            child_joint = self._convert_bone_to_bvh_joint(
                child_bone,
                skeleton,
                skeleton.get_bone_world_matrix(bone.name)
            )
            bvh_joint.children.append(child_joint)

        # Добавляем End Site если нет детей
        if not children and not bone.name.lower().endswith(('_end', '_tip')):
            # Создаем End Site с небольшим offset
            end_offset = [0.0, 0.0, bone.length * self.settings.scale_factor]
            if self.settings.convert_to_cm:
                end_offset = [coord * 100.0 for coord in end_offset]

            end_site = BVHJoint(
                name=f"{bone.name}_end",
                offset=[],
                channels=[],
                is_end_site=True,
                end_site_offset=end_offset
            )
            bvh_joint.children.append(end_site)

        return bvh_joint

    def _get_channels_for_bone(self, bone: Bone) -> List[ChannelType]:
        """Определяет каналы для кости в зависимости от её типа и настроек"""
        channels = []

        if bone.is_root:
            # Корневой сустав имеет позицию и вращение
            channels.extend([
                ChannelType.XPOSITION,
                ChannelType.YPOSITION,
                ChannelType.ZPOSITION
            ])

        # Все суставы имеют вращение
        if self.settings.rotation_order == RotationOrder.XYZ:
            channels.extend([
                ChannelType.XROTATION,
                ChannelType.YROTATION,
                ChannelType.ZROTATION
            ])
        elif self.settings.rotation_order == RotationOrder.XZY:
            channels.extend([
                ChannelType.XROTATION,
                ChannelType.ZROTATION,
                ChannelType.YROTATION
            ])
        elif self.settings.rotation_order == RotationOrder.YXZ:
            channels.extend([
                ChannelType.YROTATION,
                ChannelType.XROTATION,
                ChannelType.ZROTATION
            ])
        elif self.settings.rotation_order == RotationOrder.YZX:
            channels.extend([
                ChannelType.YROTATION,
                ChannelType.ZROTATION,
                ChannelType.XROTATION
            ])
        elif self.settings.rotation_order == RotationOrder.ZXY:
            channels.extend([
                ChannelType.ZROTATION,
                ChannelType.XROTATION,
                ChannelType.YROTATION
            ])
        else:  # ZYX (по умолчанию)
            channels.extend([
                ChannelType.ZROTATION,
                ChannelType.YROTATION,
                ChannelType.XROTATION
            ])

        # Добавляем масштаб только для BVH 2.0
        if self.settings.version == BVHVersion.BVH_2_0 and bone.has_scale:
            channels.extend([
                ChannelType.XSCALE,
                ChannelType.YSCALE,
                ChannelType.ZSCALE
            ])

        return channels

    def _generate_motion_data(self,
                              skeleton: Skeleton,
                              animation_layer: AnimationLayer,
                              bvh_root: BVHJoint) -> List[str]:
        """Генерирует данные анимации для секции MOTION"""
        lines = []

        # Получаем все суставы в порядке обхода
        all_joints = bvh_root.get_all_joints()

        # Вычисляем индексы каналов для каждого сустава
        self._calculate_channel_indices(all_joints)

        # Подготавливаем данные кадров
        frame_count = animation_layer.frame_count

        for frame_idx in range(frame_count):
            frame_data = []
            frame_time = frame_idx / self.settings.frame_rate

            for joint in all_joints:
                if joint.is_end_site:
                    continue

                # Получаем данные для сустава в текущем кадре
                joint_data = self._get_joint_data_for_frame(
                    skeleton,
                    animation_layer,
                    joint.name,
                    frame_time
                )
                frame_data.extend(joint_data)

            # Добавляем кадр как строку
            frame_line = " ".join([f"{value:.6f}" for value in frame_data])
            lines.append(frame_line)

        return lines

    def _calculate_channel_indices(self, all_joints: List[BVHJoint]):
        """Вычисляет индексы каналов для каждого сустава"""
        current_idx = 0
        for joint in all_joints:
            if joint.is_end_site:
                continue

            channel_count = joint.get_channel_count()
            self._channel_indices[joint.name] = (current_idx, current_idx + channel_count)
            current_idx += channel_count

        self.metadata["total_channels"] = current_idx

    def _get_joint_data_for_frame(self,
                                  skeleton: Skeleton,
                                  animation_layer: AnimationLayer,
                                  joint_name: str,
                                  frame_time: float) -> List[float]:
        """Получает данные сустава для конкретного кадра"""
        data = []

        # Получаем соответствующий Bone
        bone = skeleton.get_bone(joint_name)
        if not bone:
            raise ValueError(f"Сустав {joint_name} не найден в скелете")

        # Получаем ключевые кадры для этого сустава
        keyframes = animation_layer.keyframes.get(joint_name, [])

        if not keyframes:
            # Если нет ключевых кадров, используем rest pose
            if bone.is_root:
                # Позиция корневого сустава
                if self.settings.apply_root_motion:
                    pos = bone.rest_position
                else:
                    pos = np.zeros(3)

                if self.settings.convert_to_cm:
                    pos = pos * 100.0 * self.settings.scale_factor
                else:
                    pos = pos * self.settings.scale_factor

                data.extend(pos.tolist() if hasattr(pos, 'tolist') else list(pos))

            # Вращение
            rot = bone.rest_rotation
            euler = self._quaternion_to_euler(rot, self.settings.rotation_order)
            data.extend(euler)

            # Масштаб (только для BVH 2.0)
            if self.settings.version == BVHVersion.BVH_2_0 and bone.has_scale:
                scale = bone.rest_scale * self.settings.scale_factor
                data.extend(scale.tolist() if hasattr(scale, 'tolist') else list(scale))

        else:
            # Интерполируем между ключевыми кадрами
            interpolated_data = self._interpolate_keyframes(keyframes, frame_time, bone)
            data.extend(interpolated_data)

        # Оптимизация каналов (удаление незначительных изменений)
        if self.settings.optimize_channels:
            data = self._optimize_channels(data, joint_name, frame_time)

        return data

    def _interpolate_keyframes(self,
                               keyframes: List[Keyframe],
                               frame_time: float,
                               bone: Bone) -> List[float]:
        """Интерполирует значение между ключевыми кадрами"""
        if not keyframes:
            return []

        # Находим ключевые кадры вокруг текущего времени
        prev_kf = None
        next_kf = None

        for kf in keyframes:
            if kf.timestamp <= frame_time:
                prev_kf = kf
            if kf.timestamp >= frame_time:
                next_kf = kf
                break

        data = []

        # Если только один ключевой кадр или время точно совпадает
        if prev_kf and (next_kf is None or prev_kf.timestamp == frame_time):
            return self._extract_data_from_keyframe(prev_kf, bone)

        # Если только следующий ключевой кадр
        if next_kf and prev_kf is None:
            return self._extract_data_from_keyframe(next_kf, bone)

        # Интерполяция между двумя ключевыми кадрами
        if prev_kf and next_kf:
            t = (frame_time - prev_kf.timestamp) / (next_kf.timestamp - prev_kf.timestamp)
            t = max(0.0, min(1.0, t))

            # В зависимости от типа интерполяции
            if prev_kf.interpolation == "linear":
                return self._linear_interpolate(prev_kf, next_kf, t, bone)
            elif prev_kf.interpolation == "bezier":
                return self._bezier_interpolate(prev_kf, next_kf, t, bone)
            elif prev_kf.interpolation == "slerp":
                return self._slerp_interpolate(prev_kf, next_kf, t, bone)
            else:
                # По умолчанию линейная интерполяция
                return self._linear_interpolate(prev_kf, next_kf, t, bone)

        return []

    def _extract_data_from_keyframe(self, keyframe: Keyframe, bone: Bone) -> List[float]:
        """Извлекает данные из ключевого кадра"""
        data = []

        # Позиция (только для корневого сустава)
        if bone.is_root and self.settings.apply_root_motion:
            pos = keyframe.position
            if self.settings.convert_to_cm:
                pos = pos * 100.0 * self.settings.scale_factor
            else:
                pos = pos * self.settings.scale_factor
            data.extend(pos.tolist() if hasattr(pos, 'tolist') else list(pos))
        elif bone.is_root:
            # Если не применять root motion, позиция всегда 0
            data.extend([0.0, 0.0, 0.0])

        # Вращение
        if keyframe.rotation is not None:
            euler = self._quaternion_to_euler(keyframe.rotation, self.settings.rotation_order)
            data.extend(euler)
        else:
            # Если нет вращения, используем rest rotation
            euler = self._quaternion_to_euler(bone.rest_rotation, self.settings.rotation_order)
            data.extend(euler)

        # Масштаб
        if self.settings.version == BVHVersion.BVH_2_0 and bone.has_scale:
            if keyframe.scale is not None:
                scale = keyframe.scale * self.settings.scale_factor
                data.extend(scale.tolist() if hasattr(scale, 'tolist') else list(scale))
            else:
                scale = bone.rest_scale * self.settings.scale_factor
                data.extend(scale.tolist() if hasattr(scale, 'tolist') else list(scale))

        return data

    def _linear_interpolate(self,
                            prev_kf: Keyframe,
                            next_kf: Keyframe,
                            t: float,
                            bone: Bone) -> List[float]:
        """Линейная интерполяция"""
        data = []

        # Позиция
        if bone.is_root and self.settings.apply_root_motion:
            pos = prev_kf.position * (1 - t) + next_kf.position * t
            if self.settings.convert_to_cm:
                pos = pos * 100.0 * self.settings.scale_factor
            else:
                pos = pos * self.settings.scale_factor
            data.extend(pos.tolist() if hasattr(pos, 'tolist') else list(pos))
        elif bone.is_root:
            data.extend([0.0, 0.0, 0.0])

        # Вращение (SLERP для кватернионов)
        if prev_kf.rotation is not None and next_kf.rotation is not None:
            from utils.math_utils import quaternion_slerp
            rot = quaternion_slerp(prev_kf.rotation, next_kf.rotation, t)
            euler = self._quaternion_to_euler(rot, self.settings.rotation_order)
            data.extend(euler)
        else:
            # Если нет вращения в одном из ключевых кадров
            euler = self._quaternion_to_euler(bone.rest_rotation, self.settings.rotation_order)
            data.extend(euler)

        # Масштаб
        if self.settings.version == BVHVersion.BVH_2_0 and bone.has_scale:
            if prev_kf.scale is not None and next_kf.scale is not None:
                scale = prev_kf.scale * (1 - t) + next_kf.scale * t
                scale = scale * self.settings.scale_factor
                data.extend(scale.tolist() if hasattr(scale, 'tolist') else list(scale))
            else:
                scale = bone.rest_scale * self.settings.scale_factor
                data.extend(scale.tolist() if hasattr(scale, 'tolist') else list(scale))

        return data

    def _bezier_interpolate(self,
                            prev_kf: Keyframe,
                            next_kf: Keyframe,
                            t: float,
                            bone: Bone) -> List[float]:
        """Интерполяция Безье"""
        # Для простоты используем линейную интерполяцию
        # Полная реализация Безье потребует контрольных точек
        return self._linear_interpolate(prev_kf, next_kf, t, bone)

    def _slerp_interpolate(self,
                           prev_kf: Keyframe,
                           next_kf: Keyframe,
                           t: float,
                           bone: Bone) -> List[float]:
        """SLERP интерполяция для вращений"""
        data = []

        # Позиция (линейная)
        if bone.is_root and self.settings.apply_root_motion:
            pos = prev_kf.position * (1 - t) + next_kf.position * t
            if self.settings.convert_to_cm:
                pos = pos * 100.0 * self.settings.scale_factor
            else:
                pos = pos * self.settings.scale_factor
            data.extend(pos.tolist() if hasattr(pos, 'tolist') else list(pos))
        elif bone.is_root:
            data.extend([0.0, 0.0, 0.0])

        # Вращение (SLERP)
        if prev_kf.rotation is not None and next_kf.rotation is not None:
            from utils.math_utils import quaternion_slerp
            rot = quaternion_slerp(prev_kf.rotation, next_kf.rotation, t)
            euler = self._quaternion_to_euler(rot, self.settings.rotation_order)
            data.extend(euler)

        # Масштаб (линейный)
        if self.settings.version == BVHVersion.BVH_2_0 and bone.has_scale:
            if prev_kf.scale is not None and next_kf.scale is not None:
                scale = prev_kf.scale * (1 - t) + next_kf.scale * t
                scale = scale * self.settings.scale_factor
                data.extend(scale.tolist() if hasattr(scale, 'tolist') else list(scale))

        return data

    def _quaternion_to_euler(self, quaternion: np.ndarray, rotation_order: RotationOrder) -> List[float]:
        """
        Конвертирует кватернион в углы Эйлера в указанном порядке.

        Args:
            quaternion: Кватернион [x, y, z, w]
            rotation_order: Порядок вращения

        Returns:
            List[float]: Углы Эйлера в градусах
        """
        # Нормализуем кватернион
        q = quaternion / np.linalg.norm(quaternion)
        x, y, z, w = q

        # Преобразуем в углы Эйлера в радианах
        # Используем преобразования для разных порядков

        if rotation_order == RotationOrder.XYZ:
            # XYZ order
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (w * y - z * x)
            pitch = np.arcsin(sinp)

            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            angles = [roll, pitch, yaw]

        elif rotation_order == RotationOrder.XZY:
            # XZY order
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + z * z)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            sinp = 2 * (w * y - z * x)
            pitch = np.arcsin(sinp)

            angles = [roll, yaw, pitch]

        elif rotation_order == RotationOrder.YXZ:
            # YXZ order
            sinp = 2 * (w * x - y * z)
            pitch = np.arcsin(sinp)

            sinr_cosp = 2 * (w * y + z * x)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (x * x + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            angles = [pitch, roll, yaw]

        elif rotation_order == RotationOrder.YZX:
            # YZX order
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            sinr_cosp = 2 * (w * y + z * x)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (w * x - y * z)
            pitch = np.arcsin(sinp)

            angles = [yaw, roll, pitch]

        elif rotation_order == RotationOrder.ZXY:
            # ZXY order
            sinp = 2 * (w * x - y * z)
            pitch = np.arcsin(sinp)

            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (x * x + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            sinr_cosp = 2 * (w * y + z * x)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            angles = [pitch, yaw, roll]

        else:  # ZYX (по умолчанию)
            # ZYX order (yaw-pitch-roll)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (w * y - z * x)
            pitch = np.arcsin(sinp)

            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            angles = [yaw, pitch, roll]

        # Конвертируем в градусы
        angles_deg = [np.degrees(angle) for angle in angles]

        # Применяем специфичные для ПО преобразования
        if self.settings.target_software == "maya":
            # Maya использует другой порядок или знаки
            if rotation_order == RotationOrder.ZYX:
                # В Maya обычно ZYX, но с другими знаками
                angles_deg = [-angles_deg[0], -angles_deg[1], angles_deg[2]]

        elif self.settings.target_software == "unity":
            # Unity использует левостороннюю систему
            angles_deg = [-angles_deg[0], -angles_deg[1], angles_deg[2]]

        elif self.settings.target_software == "unreal":
            # Unreal Engine специфичные преобразования
            angles_deg = [angles_deg[0], -angles_deg[1], -angles_deg[2]]

        return angles_deg

    def _optimize_channels(self, data: List[float], joint_name: str, frame_time: float) -> List[float]:
        """Оптимизирует каналы, удаляя незначительные изменения"""
        if not self.settings.compression_threshold:
            return data

        optimized_data = []

        # Проверяем каждый канал
        for i, value in enumerate(data):
            # Для вращений проверяем близость к 0, 90, 180, 270, 360 градусов
            if joint_name != "root" or i >= 3:  # Не позиционные каналы
                # Нормализуем угол к диапазону [-180, 180]
                normalized = ((value + 180) % 360) - 180

                # Проверяем близость к круглым значениям
                round_values = [-180, -90, 0, 90, 180]
                for round_val in round_values:
                    if abs(normalized - round_val) < self.settings.compression_threshold:
                        value = round_val if round_val != -180 else 180
                        break

                # Если значение очень близко к 0
                if abs(value) < self.settings.compression_threshold:
                    value = 0.0

            optimized_data.append(value)

        return optimized_data

    def _save_metadata(self, output_path: str, skeleton: Skeleton, animation_layer: AnimationLayer):
        """Сохраняет метаданные в отдельный JSON файл"""
        metadata_path = output_path.replace('.bvh', '_metadata.json')

        # Добавляем дополнительную информацию
        self.metadata["joint_hierarchy"] = self._extract_joint_hierarchy(skeleton)
        self.metadata["animation_stats"] = self._calculate_animation_stats(animation_layer)
        self.metadata["export_warnings"] = self.warnings

        # Сохраняем в файл
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def _extract_joint_hierarchy(self, skeleton: Skeleton) -> Dict:
        """Извлекает иерархию суставов для метаданных"""
        hierarchy = {}

        def add_bone_to_hierarchy(bone_name: str, parent_name: Optional[str] = None):
            bone = skeleton.get_bone(bone_name)
            if not bone:
                return

            bone_info = {
                "name": bone.name,
                "type": bone.joint_type.value if hasattr(bone.joint_type, 'value') else str(bone.joint_type),
                "parent": parent_name,
                "children": [],
                "rest_position": bone.rest_position.tolist() if hasattr(bone.rest_position, 'tolist') else list(
                    bone.rest_position),
                "length": float(bone.length),
                "has_scale": bool(bone.has_scale)
            }

            if parent_name:
                hierarchy[parent_name]["children"].append(bone.name)

            hierarchy[bone.name] = bone_info

            # Рекурсивно добавляем детей
            children = skeleton.get_children(bone.name)
            for child in children:
                add_bone_to_hierarchy(child.name, bone.name)

        # Начинаем с корня
        root = skeleton.get_root_bone()
        if root:
            add_bone_to_hierarchy(root.name)

        return hierarchy

    def _calculate_animation_stats(self, animation_layer: AnimationLayer) -> Dict:
        """Вычисляет статистику анимации"""
        stats = {
            "total_keyframes": 0,
            "joints_with_keyframes": 0,
            "average_keyframes_per_joint": 0,
            "duration_seconds": animation_layer.duration,
            "frame_count": animation_layer.frame_count,
            "original_frame_rate": animation_layer.frame_rate,
            "export_frame_rate": self.settings.frame_rate
        }

        # Подсчет ключевых кадров
        total_kf = 0
        joints_with_kf = 0

        for joint_name, keyframes in animation_layer.keyframes.items():
            if keyframes:
                total_kf += len(keyframes)
                joints_with_kf += 1

        stats["total_keyframes"] = total_kf
        stats["joints_with_keyframes"] = joints_with_kf

        if joints_with_kf > 0:
            stats["average_keyframes_per_joint"] = total_kf / joints_with_kf

        return stats

    def batch_export(self,
                     skeleton: Skeleton,
                     animation_layers: List[AnimationLayer],
                     output_dir: str,
                     settings: Optional[BVHExportSettings] = None) -> Dict[str, bool]:
        """
        Пакетный экспорт нескольких анимаций.

        Args:
            skeleton: Скелет для экспорта
            animation_layers: Список слоев анимации
            output_dir: Директория для сохранения
            settings: Настройки экспорта

        Returns:
            Dict[str, bool]: Словарь с результатами экспорта для каждого слоя
        """
        results = {}

        for layer in animation_layers:
            output_path = os.path.join(output_dir, f"{layer.name}.bvh")
            success = self.export(skeleton, layer, output_path, settings)
            results[layer.name] = success

        return results


# Утилитарные функции для удобства использования
def create_bvh_exporter(settings: Optional[Dict] = None) -> BVHExporter:
    """
    Создает экземпляр BVHExporter с указанными настройками.

    Args:
        settings: Словарь с настройками

    Returns:
        BVHExporter: Экземпляр экспортера
    """
    if settings:
        export_settings = BVHExportSettings.from_dict(settings)
    else:
        export_settings = BVHExportSettings()

    return BVHExporter(export_settings)


def export_animation_to_bvh(skeleton: Skeleton,
                            animation_layer: AnimationLayer,
                            output_path: str,
                            settings: Optional[Dict] = None) -> bool:
    """
    Утилитарная функция для быстрого экспорта анимации.

    Args:
        skeleton: Скелет для экспорта
        animation_layer: Слой анимации
        output_path: Путь для сохранения
        settings: Настройки экспорта

    Returns:
        bool: Успешность экспорта
    """
    exporter = create_bvh_exporter(settings)
    return exporter.export(skeleton, animation_layer, output_path)


def validate_bvh_file(filepath: str) -> Dict[str, Any]:
    """
    Валидирует BVH файл.

    Args:
        filepath: Путь к BVH файлу

    Returns:
        Dict: Результаты валидации
    """
    validation_result = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "info": {}
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Проверяем основные секции
        hierarchy_found = False
        motion_found = False
        frames_found = False
        frame_time_found = False

        for i, line in enumerate(lines):
            line = line.strip()

            if line.startswith("HIERARCHY"):
                hierarchy_found = True

            elif line.startswith("MOTION"):
                motion_found = True

            elif line.startswith("Frames:"):
                frames_found = True
                try:
                    frame_count = int(line.split(":")[1].strip())
                    validation_result["info"]["frame_count"] = frame_count
                except:
                    validation_result["errors"].append(f"Invalid frame count at line {i}")

            elif line.startswith("Frame Time:"):
                frame_time_found = True
                try:
                    frame_time = float(line.split(":")[1].strip())
                    validation_result["info"]["frame_time"] = frame_time
                    validation_result["info"]["frame_rate"] = 1.0 / frame_time
                except:
                    validation_result["errors"].append(f"Invalid frame time at line {i}")

        # Проверяем наличие всех необходимых секций
        if not hierarchy_found:
            validation_result["errors"].append("HIERARCHY section not found")

        if not motion_found:
            validation_result["errors"].append("MOTION section not found")

        if not frames_found:
            validation_result["warnings"].append("Frames count not specified")

        if not frame_time_found:
            validation_result["warnings"].append("Frame time not specified")

        # Проверяем формат данных
        if hierarchy_found and motion_found:
            # Проверяем количество строк данных
            data_start_index = -1
            for i, line in enumerate(lines):
                if line.strip().startswith("Frame Time:"):
                    data_start_index = i + 1
                    break

            if data_start_index != -1:
                data_lines = lines[data_start_index:]
                data_line_count = sum(1 for line in data_lines if line.strip())

                if "frame_count" in validation_result["info"]:
                    expected_frames = validation_result["info"]["frame_count"]
                    if data_line_count != expected_frames:
                        validation_result["warnings"].append(
                            f"Frame count mismatch: expected {expected_frames}, found {data_line_count}"
                        )

        validation_result["valid"] = len(validation_result["errors"]) == 0

    except Exception as e:
        validation_result["errors"].append(f"File reading error: {str(e)}")

    return validation_result


# Пример использования
if __name__ == "__main__":
    # Пример создания тестового экспортера
    settings = BVHExportSettings(
        version=BVHVersion.BVH_2_0,
        rotation_order=RotationOrder.ZYX,
        frame_rate=30.0,
        scale_factor=1.0,
        convert_to_cm=True,
        target_software="blender"
    )

    exporter = BVHExporter(settings)

    print("BVH Exporter initialized with settings:")
    for key, value in settings.to_dict().items():
        print(f"  {key}: {value}")

    # Пример валидации файла
    test_file = "test_animation.bvh"
    if os.path.exists(test_file):
        validation = validate_bvh_file(test_file)
        print("\nFile validation:")
        print(f"  Valid: {validation['valid']}")
        if validation['errors']:
            print("  Errors:", validation['errors'])
        if validation['warnings']:
            print("  Warnings:", validation['warnings'])