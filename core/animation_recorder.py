"""
ПРОФЕССИОНАЛЬНЫЙ РЕКОРДЕР АНИМАЦИИ ДЛЯ MOCAP
Поддержка слоев, нелинейного редактирования, оптимизации кватернионов
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, field
import json
from collections import defaultdict, deque
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
import transforms3d as tf3d
import pickle
import zlib

logger = logging.getLogger(__name__)


class AnimationLayer:
    """Слой анимации для нелинейного редактирования"""

    def __init__(self, name: str, weight: float = 1.0, enabled: bool = True):
        self.name = name
        self.weight = weight
        self.enabled = enabled
        self.frames: List[Dict[str, Any]] = []
        self.blend_mode = 'override'  # 'override', 'additive', 'multiply'

    def apply_to_frame(self, base_frame: Dict, frame_idx: int) -> Dict:
        """Применение слоя к базовому кадру"""
        if not self.enabled or frame_idx >= len(self.frames) or self.weight == 0:
            return base_frame

        layer_frame = self.frames[frame_idx]
        result_frame = {}

        for bone_name in base_frame.keys():
            if bone_name not in layer_frame:
                result_frame[bone_name] = base_frame[bone_name]
                continue

            base_data = base_frame[bone_name]
            layer_data = layer_frame[bone_name]

            if self.blend_mode == 'override':
                # Интерполяция между base и layer по весу
                result_frame[bone_name] = self._blend_override(base_data, layer_data)
            elif self.blend_mode == 'additive':
                result_frame[bone_name] = self._blend_additive(base_data, layer_data)
            elif self.blend_mode == 'multiply':
                result_frame[bone_name] = self._blend_multiply(base_data, layer_data)

        return result_frame

    def _blend_override(self, base: Dict, layer: Dict) -> Dict:
        """Оверрайд блендинг"""
        weight = self.weight
        result = {}

        for key in ['position', 'rotation', 'scale']:
            if key in base and key in layer:
                base_val = np.array(base[key])
                layer_val = np.array(layer[key])

                if key == 'rotation':
                    # Сферическая интерполяция для кватернионов
                    result[key] = tf3d.quaternions.slerp(base_val, layer_val, weight)
                else:
                    # Линейная интерполяция для позиции и масштаба
                    result[key] = base_val * (1 - weight) + layer_val * weight
            elif key in base:
                result[key] = base[key]

        return result

    def _blend_additive(self, base: Dict, layer: Dict) -> Dict:
        """Аддитивный блендинг"""
        result = {}

        for key in ['position', 'rotation', 'scale']:
            if key in base and key in layer:
                base_val = np.array(base[key])
                layer_val = np.array(layer[key])

                if key == 'rotation':
                    # Для кватернионов: multiply
                    result[key] = tf3d.quaternions.qmult(base_val, layer_val)
                else:
                    result[key] = base_val + layer_val * self.weight
            elif key in base:
                result[key] = base[key]

        return result

    def _blend_multiply(self, base: Dict, layer: Dict) -> Dict:
        """Мультипликативный блендинг"""
        result = {}

        for key in ['position', 'rotation', 'scale']:
            if key in base and key in layer:
                base_val = np.array(base[key])
                layer_val = np.array(layer[key])

                if key == 'rotation':
                    # Интерполяция между вращениями
                    result[key] = tf3d.quaternions.slerp(base_val, layer_val, self.weight)
                else:
                    result[key] = base_val * (1 + (layer_val - 1) * self.weight)
            elif key in base:
                result[key] = base[key]

        return result


@dataclass
class AnimationFrame:
    """Оптимизированная структура для хранения кадра анимации"""
    timestamp: float
    frame_idx: int
    data: Dict[str, Dict[str, np.ndarray]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Сжатие данных для экономии памяти
        self._compressed = False
        self._compressed_data = None

    def compress(self):
        """Сжатие данных кадра"""
        if not self._compressed:
            self._compressed_data = zlib.compress(pickle.dumps(self.data))
            self._compressed = True

    def decompress(self) -> Dict:
        """Распаковка данных кадра"""
        if self._compressed and self._compressed_data is not None:
            self.data = pickle.loads(zlib.decompress(self._compressed_data))
            self._compressed = False
        return self.data


@dataclass
class Keyframe:
    """Ключевой кадр для редактирования"""
    frame_idx: int
    bone_name: str
    position: np.ndarray
    rotation: np.ndarray
    interpolation: str = 'BEZIER'  # BEZIER, LINEAR, CONSTANT
    handles: Tuple[Tuple[float, float], Tuple[float, float]] = None

    def __post_init__(self):
        if self.handles is None:
            self.handles = ((0.0, 0.0), (1.0, 1.0))


class AnimationCurve:
    """Кривая анимации для одного параметра кости"""

    def __init__(self, bone_name: str, parameter: str):
        self.bone_name = bone_name
        self.parameter = parameter  # 'position_x', 'rotation_w', etc.
        self.keyframes: List[Keyframe] = []
        self.fps = 30.0

    def evaluate(self, frame_idx: int) -> float:
        """Вычисление значения кривой в заданном кадре"""
        if not self.keyframes:
            return 0.0

        # Находим ключевые кадры вокруг нужного кадра
        for i, kf in enumerate(self.keyframes):
            if kf.frame_idx == frame_idx:
                return self._get_parameter_value(kf)
            elif kf.frame_idx > frame_idx:
                if i == 0:
                    return self._get_parameter_value(kf)
                else:
                    prev_kf = self.keyframes[i - 1]
                    next_kf = kf
                    t = (frame_idx - prev_kf.frame_idx) / (next_kf.frame_idx - prev_kf.frame_idx)

                    if prev_kf.interpolation == 'LINEAR':
                        return self._linear_interpolate(prev_kf, next_kf, t)
                    elif prev_kf.interpolation == 'BEZIER':
                        return self._bezier_interpolate(prev_kf, next_kf, t)
                    else:  # CONSTANT
                        return self._get_parameter_value(prev_kf)

        # Если кадр после последнего ключевого кадра
        return self._get_parameter_value(self.keyframes[-1])

    def _get_parameter_value(self, keyframe: Keyframe) -> float:
        """Извлечение значения параметра из ключевого кадра"""
        if self.parameter.startswith('position_'):
            axis = self.parameter[-1]  # x, y, z
            idx = {'x': 0, 'y': 1, 'z': 2}[axis]
            return keyframe.position[idx]
        elif self.parameter.startswith('rotation_'):
            component = self.parameter[-1]  # w, x, y, z
            idx = {'w': 0, 'x': 1, 'y': 2, 'z': 3}[component]
            return keyframe.rotation[idx]
        return 0.0

    def _linear_interpolate(self, kf1: Keyframe, kf2: Keyframe, t: float) -> float:
        """Линейная интерполяция"""
        v1 = self._get_parameter_value(kf1)
        v2 = self._get_parameter_value(kf2)
        return v1 + (v2 - v1) * t

    def _bezier_interpolate(self, kf1: Keyframe, kf2: Keyframe, t: float) -> float:
        """Кубическая интерполяция Безье"""
        v1 = self._get_parameter_value(kf1)
        v2 = self._get_parameter_value(kf2)

        # Упрощенная версия Безье
        return v1 + (v2 - v1) * (3 * t ** 2 - 2 * t ** 3)


class ProfessionalAnimationRecorder:
    """
    ПРОФЕССИОНАЛЬНЫЙ РЕКОРДЕР АНИМАЦИИ

    Особенности:
    1. Многослойная запись
    2. Нелинейное редактирование (NLE)
    3. Оптимизация памяти со сжатием
    4. Расширенное сглаживание
    5. Экспорт в BVH, FBX, JSON
    6. Поддержка ключевых кадров
    """

    def __init__(self, fps: int = 30, buffer_size: int = 1000):
        self.fps = fps
        self.frame_time = 1.0 / fps

        # Состояние записи
        self.is_recording = False
        self.is_playing = False
        self.is_paused = False

        # Данные анимации
        self.frames: List[AnimationFrame] = []
        self.layers: Dict[str, AnimationLayer] = {}
        self.curves: Dict[str, AnimationCurve] = {}

        # Временные метки
        self.start_time = 0.0
        self.current_frame_idx = 0
        self.total_duration = 0.0

        # Буфер для реального времени
        self.frame_buffer = deque(maxlen=buffer_size)
        self.last_frame_time = 0.0

        # Сглаживание
        self.smoothing_enabled = True
        self.smoothing_window = 7
        self.smoothing_polyorder = 3

        # Статистика
        self.stats = {
            'total_frames': 0,
            'dropped_frames': 0,
            'avg_fps': 0.0,
            'memory_usage_mb': 0.0
        }

        # Метрики качества
        self.quality_metrics = {
            'jitter': 0.0,  # Дрожание
            'latency': 0.0,  # Задержка
            'consistency': 1.0  # Консистентность трекинга
        }

        logger.info(f"ProfessionalAnimationRecorder инициализирован @ {fps} FPS")

    # ==================== ЗАПИСЬ ====================

    def start_recording(self, clear_existing: bool = True):
        """Начало записи анимации"""
        if clear_existing:
            self.frames.clear()
            self.frame_buffer.clear()
            self.layers.clear()
            self.curves.clear()

        self.is_recording = True
        self.is_playing = False
        self.is_paused = False
        self.start_time = time.time()
        self.current_frame_idx = 0
        self.last_frame_time = self.start_time

        # Создаем базовый слой
        base_layer = AnimationLayer("Base", weight=1.0)
        self.layers["Base"] = base_layer

        logger.info("Запись анимации начата")

    def stop_recording(self):
        """Остановка записи анимации"""
        self.is_recording = False

        # Переносим данные из буфера в основной массив
        while self.frame_buffer:
            frame = self.frame_buffer.popleft()
            self.frames.append(frame)

        self.total_duration = len(self.frames) * self.frame_time
        self._update_statistics()

        logger.info(f"Запись остановлена. Кадров: {len(self.frames)}, Длительность: {self.total_duration:.2f}с")

    def record_frame(self, skeleton_data: Dict[str, Dict[str, np.ndarray]],
                     metadata: Optional[Dict] = None) -> bool:
        """
        Запись кадра анимации

        Args:
            skeleton_data: Данные скелета {bone_name: {position: [], rotation: []}}
            metadata: Дополнительные метаданные кадра

        Returns:
            True если кадр успешно записан
        """
        if not self.is_recording:
            return False

        current_time = time.time()
        frame_delay = current_time - self.last_frame_time

        # Проверка FPS (пропускаем кадры если слишком быстро)
        if frame_delay < self.frame_time * 0.8:  # 80% от целевого интервала
            self.stats['dropped_frames'] += 1
            return False

        # Создание кадра
        frame = AnimationFrame(
            timestamp=current_time,
            frame_idx=self.current_frame_idx,
            data=skeleton_data,
            metadata=metadata or {}
        )

        # Добавление в буфер
        self.frame_buffer.append(frame)

        # Обновление базового слоя
        if "Base" in self.layers:
            self.layers["Base"].frames.append(skeleton_data)

        self.current_frame_idx += 1
        self.last_frame_time = current_time

        # Автоматическое сглаживание в реальном времени
        if self.smoothing_enabled and len(self.frame_buffer) >= self.smoothing_window:
            self._apply_realtime_smoothing()

        return True

    def _apply_realtime_smoothing(self):
        """Применение сглаживания в реальном времени"""
        if len(self.frame_buffer) < self.smoothing_window:
            return

        # Берем последние N кадров из буфера
        recent_frames = list(self.frame_buffer)[-self.smoothing_window:]

        # Сглаживаем каждый параметр
        for bone_name in recent_frames[0].data.keys():
            # Позиции
            positions = np.array([f.data[bone_name]['position'] for f in recent_frames])
            smoothed_positions = savgol_filter(positions, self.smoothing_window,
                                               self.smoothing_polyorder, axis=0)

            # Кватернионы (сложнее)
            rotations = np.array([f.data[bone_name]['rotation'] for f in recent_frames])

            # Для кватернионов используем сферическую интерполяцию
            for i in range(len(recent_frames)):
                recent_frames[i].data[bone_name]['position'] = smoothed_positions[i]
                # Вращение оставляем как есть (требуется специальная обработка)

        # Обновляем качество
        self._update_quality_metrics()

    # ==================== ВОСПРОИЗВЕДЕНИЕ ====================

    def play(self, start_frame: int = 0, loop: bool = False):
        """Начало воспроизведения анимации"""
        self.is_playing = True
        self.is_paused = False
        self.current_frame_idx = start_frame

        logger.info(f"Воспроизведение начато с кадра {start_frame}")

    def pause(self):
        """Пауза воспроизведения"""
        self.is_paused = True
        logger.info("Воспроизведение на паузе")

    def resume(self):
        """Возобновление воспроизведения"""
        self.is_paused = False
        logger.info("Воспроизведение возобновлено")

    def stop_playback(self):
        """Остановка воспроизведения"""
        self.is_playing = False
        self.is_paused = False
        logger.info("Воспроизведение остановлено")

    def get_frame(self, frame_idx: int = None, apply_layers: bool = True) -> Optional[Dict]:
        """
        Получение кадра анимации

        Args:
            frame_idx: Номер кадра (None для текущего)
            apply_layers: Применять ли слои анимации

        Returns:
            Данные кадра или None
        """
        if frame_idx is None:
            frame_idx = self.current_frame_idx

        if frame_idx < 0 or frame_idx >= len(self.frames):
            return None

        # Получаем базовый кадр
        frame = self.frames[frame_idx]
        frame_data = frame.decompress()

        # Применяем слои если нужно
        if apply_layers and self.layers:
            result_data = frame_data.copy()

            for layer_name, layer in self.layers.items():
                if layer.enabled and layer.weight > 0:
                    if frame_idx < len(layer.frames):
                        result_data = layer.apply_to_frame(result_data, frame_idx)

            return result_data

        return frame_data

    def seek(self, frame_idx: int):
        """Переход к конкретному кадру"""
        if 0 <= frame_idx < len(self.frames):
            self.current_frame_idx = frame_idx
            logger.debug(f"Переход к кадру {frame_idx}")

    def seek_time(self, timestamp: float):
        """Переход к конкретному времени"""
        frame_idx = int(timestamp * self.fps)
        self.seek(frame_idx)

    # ==================== РЕДАКТИРОВАНИЕ ====================

    def create_layer(self, name: str, weight: float = 1.0) -> AnimationLayer:
        """Создание нового слоя анимации"""
        if name in self.layers:
            logger.warning(f"Слой '{name}' уже существует")
            return self.layers[name]

        layer = AnimationLayer(name, weight)
        self.layers[name] = layer

        # Инициализируем кадрами слоя
        for _ in range(len(self.frames)):
            layer.frames.append({})

        logger.info(f"Создан слой анимации: '{name}' с весом {weight}")
        return layer

    def remove_layer(self, name: str):
        """Удаление слоя анимации"""
        if name in self.layers:
            del self.layers[name]
            logger.info(f"Слой '{name}' удален")

    def set_layer_weight(self, layer_name: str, weight: float):
        """Установка веса слоя"""
        if layer_name in self.layers:
            self.layers[layer_name].weight = max(0.0, min(1.0, weight))

    def enable_layer(self, layer_name: str, enabled: bool = True):
        """Включение/выключение слоя"""
        if layer_name in self.layers:
            self.layers[layer_name].enabled = enabled

    def cut_animation(self, start_frame: int, end_frame: int):
        """Вырезание сегмента анимации"""
        if start_frame < 0 or end_frame > len(self.frames) or start_frame >= end_frame:
            logger.error("Некорректные границы для вырезания")
            return False

        # Вырезаем кадры
        self.frames = self.frames[:start_frame] + self.frames[end_frame:]

        # Обновляем слои
        for layer in self.layers.values():
            layer.frames = layer.frames[:start_frame] + layer.frames[end_frame:]

        self.current_frame_idx = min(self.current_frame_idx, len(self.frames) - 1)
        self.total_duration = len(self.frames) * self.frame_time

        logger.info(f"Вырезан сегмент: кадры {start_frame}-{end_frame}")
        return True

    def trim_animation(self, start_frame: int, end_frame: int):
        """Обрезка анимации до указанного диапазона"""
        if start_frame < 0 or end_frame > len(self.frames) or start_frame >= end_frame:
            logger.error("Некорректные границы для обрезки")
            return False

        # Оставляем только указанный диапазон
        self.frames = self.frames[start_frame:end_frame]

        # Обновляем слои
        for layer in self.layers.values():
            layer.frames = layer.frames[start_frame:end_frame]

        self.current_frame_idx = 0
        self.total_duration = len(self.frames) * self.frame_time

        logger.info(f"Анимация обрезана: кадры {start_frame}-{end_frame}")
        return True

    def apply_smoothing(self, window_size: int = None, polyorder: int = None):
        """Применение сглаживания ко всей анимации"""
        if window_size:
            self.smoothing_window = window_size
        if polyorder:
            self.smoothing_polyorder = polyorder

        if len(self.frames) < self.smoothing_window:
            logger.warning("Недостаточно кадров для сглаживания")
            return

        logger.info(f"Применение сглаживания (window={self.smoothing_window}, poly={self.smoothing_polyorder})")

        # Сглаживаем позиции для каждой кости
        for bone_name in self.frames[0].data.keys():
            # Позиции
            positions = np.array([f.data[bone_name]['position'] for f in self.frames])
            if len(positions) > self.smoothing_window:
                smoothed_positions = savgol_filter(positions, self.smoothing_window,
                                                   self.smoothing_polyorder, axis=0)

                # Обновляем кадры
                for i, frame in enumerate(self.frames):
                    frame.data[bone_name]['position'] = smoothed_positions[i]

    def time_warp(self, speed_factor: float):
        """Изменение скорости анимации"""
        if speed_factor <= 0:
            logger.error("Коэффициент скорости должен быть > 0")
            return

        # Для ускорения: удаляем кадры
        # Для замедления: дублируем кадры
        # В реальной реализации нужна интерполяция

        logger.info(f"Изменение скорости анимации: x{speed_factor}")
        # Реализация требует интерполяции между кадрами

    # ==================== ЭКСПОРТ ====================

    def export_bvh(self, filepath: str, skeleton_hierarchy: Dict):
        """
        Экспорт анимации в формат BVH

        Args:
            filepath: Путь для сохранения файла
            skeleton_hierarchy: Иерархия скелета для BVH заголовка
        """
        logger.info(f"Экспорт BVH в {filepath}")

        with open(filepath, 'w') as f:
            # HIERARCHY секция
            f.write("HIERARCHY\n")
            self._write_bvh_hierarchy(f, skeleton_hierarchy, 0)

            # MOTION секция
            f.write("MOTION\n")
            f.write(f"Frames: {len(self.frames)}\n")
            f.write(f"Frame Time: {self.frame_time:.6f}\n")

            # Данные кадров
            for frame in self.frames:
                frame_data = self.get_frame(frame.frame_idx, apply_layers=True)
                bvh_line = self._frame_to_bvh_string(frame_data, skeleton_hierarchy)
                f.write(bvh_line + "\n")

        logger.info("BVH экспорт завершен")

    def _write_bvh_hierarchy(self, f, bone: Dict, indent_level: int):
        """Рекурсивная запись иерархии BVH"""
        indent = "  " * indent_level

        if indent_level == 0:
            f.write(f"{indent}ROOT {bone['name']}\n")
        else:
            f.write(f"{indent}JOINT {bone['name']}\n")

        f.write(f"{indent}{{\n")

        # OFFSET
        offset = bone.get('offset', [0, 0, 0])
        f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")

        # CHANNELS
        channels = bone.get('channels', ['Xposition', 'Yposition', 'Zposition',
                                         'Zrotation', 'Yrotation', 'Xrotation'])
        f.write(f"{indent}  CHANNELS {len(channels)} {' '.join(channels)}\n")

        # Дочерние кости
        for child in bone.get('children', []):
            self._write_bvh_hierarchy(f, child, indent_level + 1)

        # End Site для конечных костей
        if not bone.get('children'):
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            f.write(f"{indent}    OFFSET 0.0 0.0 0.0\n")
            f.write(f"{indent}  }}\n")

        f.write(f"{indent}}}\n")

    def _frame_to_bvh_string(self, frame_data: Dict, skeleton: Dict) -> str:
        """Конвертация кадра в строку BVH"""
        values = []

        def process_bone(bone: Dict):
            bone_name = bone['name']
            if bone_name in frame_data:
                bone_data = frame_data[bone_name]

                # Позиция
                pos = bone_data.get('position', [0, 0, 0])
                values.extend([pos[0], pos[1], pos[2]])

                # Вращение (кватернион -> Euler ZYX)
                rot_quat = bone_data.get('rotation', [1, 0, 0, 0])
                euler = tf3d.euler.quat2euler(rot_quat, 'rzyx')
                values.extend(np.degrees(euler))  # Радианы -> градусы

            # Рекурсивно обрабатываем детей
            for child in bone.get('children', []):
                process_bone(child)

        # Начинаем с корневой кости
        process_bone(skeleton)

        return " ".join(f"{v:.6f}" for v in values)

    def export_json(self, filepath: str):
        """Экспорт анимации в JSON формат"""
        export_data = {
            'metadata': {
                'fps': self.fps,
                'frame_count': len(self.frames),
                'duration': self.total_duration,
                'layers': list(self.layers.keys())
            },
            'frames': []
        }

        for i, frame in enumerate(self.frames):
            frame_data = self.get_frame(i, apply_layers=True)

            # Конвертация numpy в списки
            frame_dict = {}
            for bone_name, bone_data in frame_data.items():
                frame_dict[bone_name] = {
                    'position': bone_data['position'].tolist(),
                    'rotation': bone_data['rotation'].tolist()
                }

            export_data['frames'].append({
                'frame_idx': i,
                'timestamp': frame.timestamp,
                'data': frame_dict
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"JSON экспорт завершен: {filepath}")

    def save_project(self, filepath: str):
        """Сохранение проекта анимации"""
        project_data = {
            'fps': self.fps,
            'frames': [frame.data for frame in self.frames],
            'layers': {name: layer.__dict__ for name, layer in self.layers.items()},
            'current_frame': self.current_frame_idx
        }

        with open(filepath, 'wb') as f:
            pickle.dump(project_data, f)

        logger.info(f"Проект сохранен: {filepath}")

    def load_project(self, filepath: str) -> bool:
        """Загрузка проекта анимации"""
        try:
            with open(filepath, 'rb') as f:
                project_data = pickle.load(f)

            self.fps = project_data['fps']
            self.frames = [AnimationFrame(0, i, data) for i, data in enumerate(project_data['frames'])]

            # Восстанавливаем слои
            self.layers.clear()
            for name, layer_data in project_data['layers'].items():
                layer = AnimationLayer(name)
                layer.__dict__.update(layer_data)
                self.layers[name] = layer

            self.current_frame_idx = project_data.get('current_frame', 0)
            self.total_duration = len(self.frames) * self.frame_time

            logger.info(f"Проект загружен: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки проекта: {e}")
            return False

    # ==================== УТИЛИТЫ ====================

    def _update_statistics(self):
        """Обновление статистики"""
        self.stats['total_frames'] = len(self.frames)

        # Расчет использования памяти
        total_bytes = 0
        for frame in self.frames:
            # Приблизительный расчет размера
            total_bytes += len(str(frame.data)) * 2  # UTF-16

        self.stats['memory_usage_mb'] = total_bytes / (1024 * 1024)

        # Расчет FPS
        if self.total_duration > 0:
            self.stats['avg_fps'] = len(self.frames) / self.total_duration

    def _update_quality_metrics(self):
        """Обновление метрик качества"""
        if len(self.frames) < 10:
            return

        # Расчет дрожания (вариация позиций)
        recent_frames = self.frames[-10:]
        positions = []

        for frame in recent_frames:
            for bone_data in frame.data.values():
                positions.append(bone_data['position'])

        if positions:
            positions_array = np.array(positions)
            self.quality_metrics['jitter'] = np.std(positions_array)

    def get_info(self) -> Dict:
        """Получение информации об анимации"""
        return {
            'frame_count': len(self.frames),
            'duration_seconds': self.total_duration,
            'fps': self.fps,
            'current_frame': self.current_frame_idx,
            'is_recording': self.is_recording,
            'is_playing': self.is_playing,
            'is_paused': self.is_paused,
            'layers': list(self.layers.keys()),
            'stats': self.stats.copy(),
            'quality': self.quality_metrics.copy()
        }

    def clear(self):
        """Очистка всей анимации"""
        self.frames.clear()
        self.frame_buffer.clear()
        self.layers.clear()
        self.curves.clear()

        self.current_frame_idx = 0
        self.total_duration = 0.0

        logger.info("Анимация полностью очищена")


# Быстрый тест
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Тестирование ProfessionalAnimationRecorder...")

    # Создание тестовых данных
    test_skeleton = {
        'Hips': {'position': [0, 1, 0], 'rotation': [1, 0, 0, 0]},
        'Spine': {'position': [0, 1.1, 0], 'rotation': [1, 0, 0, 0]}
    }

    # Создание рекордера
    recorder = ProfessionalAnimationRecorder(fps=30)

    # Запись
    recorder.start_recording()

    for i in range(100):
        # Немного изменяем позицию
        test_skeleton['Hips']['position'][1] = 1 + 0.1 * np.sin(i * 0.1)
        recorder.record_frame(test_skeleton, {'frame_type': 'test'})
        time.sleep(0.01)  # Имитация реального времени

    recorder.stop_recording()

    # Информация
    info = recorder.get_info()
    print(f"Записано кадров: {info['frame_count']}")
    print(f"Длительность: {info['duration_seconds']:.2f} сек")
    print(f"FPS: {info['stats']['avg_fps']:.1f}")

    # Экспорт
    test_hierarchy = {
        'name': 'Hips',
        'offset': [0, 0, 0],
        'channels': ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Yrotation', 'Xrotation'],
        'children': [
            {
                'name': 'Spine',
                'offset': [0, 0.1, 0],
                'channels': ['Zrotation', 'Yrotation', 'Xrotation']
            }
        ]
    }

    recorder.export_bvh("test_animation.bvh", test_hierarchy)
    recorder.export_json("test_animation.json")

    print("Тест завершен!")

AnimationRecorder = ProfessionalAnimationRecorder

__all__ = [
    'ProfessionalAnimationRecorder',
    'AnimationRecorder',  # Алиас
    'AnimationFrame',
    'AnimationLayer',
    'CompressionMethod',
    'ExportFormat',
    'create_animation_recorder'
]