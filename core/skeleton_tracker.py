"""
ПРОФЕССИОНАЛЬНЫЙ ТРЕКЕР СКЕЛЕТА ДЛЯ MOCAP PRO
Мультимодельный трекинг, ретаргетинг, коррекция в реальном времени
"""
import numpy as np
import cv2
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque, defaultdict
import json
import pickle
import copy  # Добавить этот импорт

# ТОЛЬКО ЭТОТ ИМПОРТ ОСТАВЬТЕ
from core.pose_estimator import EnhancedPoseEstimator, TrackingMode

logger = logging.getLogger(__name__)


# ЗАГЛУШКИ ДЛЯ АННОТАЦИЙ
class ProfessionalSkeleton:
    pass


class ProfessionalAnimationRecorder:
    pass


class MultiCameraManager:
    pass


class CameraFrame:
    pass


class Bone:
    pass


# РЕАЛЬНАЯ ФУНКЦИЯ МАППИНГА
def create_humanoid_mapping():
    """Создание маппинга MediaPipe -> Humanoid"""
    return {
        0: 'Head', 11: 'LeftShoulder', 12: 'RightShoulder',
        13: 'LeftElbow', 14: 'RightElbow', 15: 'LeftWrist',
        16: 'RightWrist', 23: 'LeftHip', 24: 'RightHip',
        25: 'LeftKnee', 26: 'RightKnee', 27: 'LeftAnkle',
        28: 'RightAnkle'
    }


class TrackerState(Enum):
    """Состояния трекера"""
    IDLE = "idle"
    INITIALIZING = "init"
    CALIBRATING = "calib"
    TRACKING = "tracking"
    PAUSED = "paused"
    ERROR = "error"


class RetargetingMethod(Enum):
    """Методы ретаргетинга"""
    OFFSET = "offset"
    SCALE = "scale"
    ROTATION = "rotation"
    IK = "ik"
    ML = "ml"


class CorrectionMode(Enum):
    """Режимы коррекции"""
    NONE = "none"
    MANUAL = "manual"
    SEMI_AUTO = "semi_auto"
    AUTO = "auto"


@dataclass
class TrackingFrame:
    """Обработанный кадр трекинга"""
    timestamp: float
    frame_id: int
    landmarks: np.ndarray
    skeleton_data: Dict[str, Dict]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_animation_data(self) -> Dict[str, Dict]:
        return self.skeleton_data


@dataclass
class TrackingMetrics:
    """Метрики качества трекинга"""
    fps: float = 0.0
    latency: float = 0.0
    confidence: float = 0.0
    jitter: float = 0.0
    stability: float = 0.0
    landmarks_found: int = 0
    processing_time: float = 0.0

    def update(self, landmarks_count: int, conf: float, proc_time: float):
        self.landmarks_found = landmarks_count
        self.confidence = conf
        self.processing_time = proc_time

        if hasattr(self, 'last_positions') and self.last_positions:
            if len(self.last_positions) >= 3:
                positions = np.array(self.last_positions)
                self.jitter = np.std(positions).mean()

        self.stability = conf * (1.0 - self.jitter)


class PoseCorrector:
    """Корректор позы для улучшения точности"""

    def __init__(self):
        self.correction_rules = {}
        self.history = deque(maxlen=30)
        self.smoothing_window = 5
        self.correction_weights = {}
        self._init_default_rules()  # ВЫЗЫВАЕМ ИНИЦИАЛИЗАЦИЮ

    def _init_default_rules(self):  # ИСПРАВЛЕНО: правильный отступ
        """Инициализация правил коррекции по умолчанию"""
        self.correction_rules = {
            'symmetry': {'enabled': True, 'weight': 0.3, 'description': 'Симметрия'},
            'anatomy': {'enabled': True, 'weight': 0.4, 'description': 'Анатомия'},
            'smoothness': {'enabled': True, 'weight': 0.2, 'description': 'Плавность'},
            'physics': {'enabled': True, 'weight': 0.1, 'description': 'Физика'}
        }

        self.correction_weights = {
            'spine': 0.9, 'head': 0.8, 'shoulders': 0.7,
            'hips': 0.9, 'limbs': 0.6, 'hands_feet': 0.5
        }

    def apply_correction(self, landmarks: List, skeleton: ProfessionalSkeleton,
                         mode: CorrectionMode = CorrectionMode.AUTO) -> List:
        """Применение коррекции к landmarks"""
        if mode == CorrectionMode.NONE or not landmarks:
            return landmarks

        corrected = landmarks.copy()
        self.history.append(landmarks)

        if mode in [CorrectionMode.AUTO, CorrectionMode.SEMI_AUTO]:
            corrected = self._apply_symmetry_correction(corrected)
            corrected = self._apply_anatomy_correction(corrected, skeleton)
            corrected = self._apply_smoothing(corrected)

        if mode == CorrectionMode.SEMI_AUTO:
            corrected = self._blend_corrections(landmarks, corrected, 0.5)

        return corrected

    def _apply_symmetry_correction(self, landmarks: List) -> List:
        """Коррекция симметрии левой/правой сторон"""
        if len(landmarks) < 33:
            return landmarks

        symmetric_pairs = [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28)]

        for left_idx, right_idx in symmetric_pairs:
            if (left_idx < len(landmarks) and right_idx < len(landmarks) and
                    hasattr(landmarks[left_idx], 'position') and
                    hasattr(landmarks[right_idx], 'position')):
                left_pos = landmarks[left_idx].position
                right_pos = landmarks[right_idx].position
                avg_x = (left_pos[0] + right_pos[0]) / 2
                avg_y = (left_pos[1] + right_pos[1]) / 2
                weight = self.correction_rules['symmetry']['weight']

                landmarks[left_idx].position[0] = left_pos[0] * (1 - weight) + (2 * avg_x - right_pos[0]) * weight
                landmarks[right_idx].position[0] = right_pos[0] * (1 - weight) + (2 * avg_x - left_pos[0]) * weight
                landmarks[left_idx].position[1] = left_pos[1] * (1 - weight * 0.5) + avg_y * weight * 0.5
                landmarks[right_idx].position[1] = right_pos[1] * (1 - weight * 0.5) + avg_y * weight * 0.5

        return landmarks

    def _apply_anatomy_correction(self, landmarks: List, skeleton: ProfessionalSkeleton) -> List:
        """Анатомическая коррекция"""
        # Упрощенная версия
        return landmarks

    def _apply_smoothing(self, landmarks: List) -> List:
        """Временное сглаживание"""
        if len(self.history) < self.smoothing_window:
            return landmarks

        history_array = list(self.history)
        smoothed = []

        for i in range(len(landmarks)):
            if hasattr(landmarks[i], 'position'):
                positions = []
                for frame in history_array[-self.smoothing_window:]:
                    if i < len(frame) and hasattr(frame[i], 'position'):
                        positions.append(frame[i].position[:3])

                if positions:
                    positions_array = np.array(positions)
                    smoothed_pos = np.mean(positions_array, axis=0)
                    smoothed_landmark = copy.copy(landmarks[i])
                    smoothed_landmark.position[:3] = smoothed_pos
                    smoothed.append(smoothed_landmark)
                else:
                    smoothed.append(landmarks[i])
            else:
                smoothed.append(landmarks[i])

        return smoothed

    def _blend_corrections(self, original: List, corrected: List, weight: float) -> List:
        """Смешивание оригинальных и скорректированных landmarks"""
        blended = []

        for orig, corr in zip(original, corrected):
            if hasattr(orig, 'position') and hasattr(corr, 'position'):
                blended_landmark = copy.copy(orig)
                blended_landmark.position = orig.position * (1 - weight) + corr.position * weight

                if hasattr(orig, 'confidence') and hasattr(corr, 'confidence'):
                    blended_landmark.confidence = orig.confidence * (1 - weight) + corr.confidence * weight

                blended.append(blended_landmark)
            else:
                blended.append(orig)

        return blended


# RetargetingSystem ДОЛЖЕН БЫТЬ ОТДЕЛЬНЫМ КЛАССОМ, НЕ ВНУТРИ PoseCorrector
class RetargetingSystem:
    """Система ретаргетинга между разными скелетами"""

    def __init__(self):
        self.mapping_method = RetargetingMethod.SCALE
        self.mapping_rules = {}
        self.offset_map = {}
        self.scale_factors = {}
        self._init_default_mappings()

    def _init_default_mappings(self):
        self.mapping_rules['mediapipe_to_standard'] = {
            0: 'Head', 11: 'LeftShoulder', 12: 'RightShoulder',
            13: 'LeftElbow', 14: 'RightElbow', 15: 'LeftWrist',
            16: 'RightWrist', 23: 'LeftHip', 24: 'RightHip',
            25: 'LeftKnee', 26: 'RightKnee', 27: 'LeftAnkle',
            28: 'RightAnkle', 7: 'Head', 8: 'Head'
        }

        self.mapping_rules['mediapipe_to_unreal'] = {
            0: 'head', 11: 'upperarm_l', 12: 'upperarm_r',
            13: 'lowerarm_l', 14: 'lowerarm_r', 15: 'hand_l',
            16: 'hand_r', 23: 'thigh_l', 24: 'thigh_r',
            25: 'calf_l', 26: 'calf_r', 27: 'foot_l', 28: 'foot_r'
        }

    def create_mapping(self, source_skeleton: ProfessionalSkeleton,
                       target_skeleton: ProfessionalSkeleton,
                       method: RetargetingMethod = RetargetingMethod.SCALE) -> Dict:

        mapping = {}
        if method == RetargetingMethod.SCALE:
            mapping = self._auto_map_by_name(source_skeleton, target_skeleton)
            self._calculate_scale_factors(source_skeleton, target_skeleton, mapping)
        elif method == RetargetingMethod.OFFSET:
            mapping = self._create_offset_mapping(source_skeleton, target_skeleton)
        elif method == RetargetingMethod.IK:
            mapping = self._create_ik_mapping(source_skeleton, target_skeleton)

        return mapping

    def _auto_map_by_name(self, source: ProfessionalSkeleton, target: ProfessionalSkeleton) -> Dict:
        mapping = {}
        source_bones = source.bones.keys()
        target_bones = target.bones.keys()

        keyword_mapping = {
            'head': ['head', 'neck', 'skull'],
            'spine': ['spine', 'chest', 'torso'],
            'shoulder': ['shoulder', 'clavicle', 'collar'],
            'arm': ['arm', 'upperarm', 'lowerarm'],
            'hand': ['hand', 'wrist', 'palm'],
            'hip': ['hip', 'pelvis', 'waist'],
            'leg': ['leg', 'thigh', 'calf', 'shin'],
            'foot': ['foot', 'ankle', 'toe']
        }

        for target_bone in target_bones:
            target_lower = target_bone.lower()
            for keyword, synonyms in keyword_mapping.items():
                if any(syn in target_lower for syn in synonyms):
                    for source_bone in source_bones:
                        source_lower = source_bone.lower()
                        if any(syn in source_lower for syn in synonyms):
                            mapping[source_bone] = target_bone
                            break
                    break

        return mapping

    def _calculate_scale_factors(self, source: ProfessionalSkeleton,
                                 target: ProfessionalSkeleton, mapping: Dict):

        self.scale_factors.clear()
        for source_name, target_name in mapping.items():
            if source_name in source.bones and target_name in target.bones:
                source_bone = source.bones[source_name]
                target_bone = target.bones[target_name]
                if source_bone.length > 0:
                    scale = target_bone.length / source_bone.length
                    self.scale_factors[source_name] = scale

        if 'Spine' in source.bones and 'Spine' in target.bones:
            source_height = self._estimate_skeleton_height(source)
            target_height = self._estimate_skeleton_height(target)
            if source_height > 0:
                self.scale_factors['global'] = target_height / source_height

    def _estimate_skeleton_height(self, skeleton: ProfessionalSkeleton) -> float:
        height = 0.0
        if 'Spine' in skeleton.bones:
            height += skeleton.bones['Spine'].length * 3
        if 'LeftUpperLeg' in skeleton.bones:
            height += skeleton.bones['LeftUpperLeg'].length * 2
        return height

    def _create_offset_mapping(self, source: ProfessionalSkeleton,
                               target: ProfessionalSkeleton) -> Dict:

        mapping = self._auto_map_by_name(source, target)
        self.offset_map.clear()

        for source_name, target_name in mapping.items():
            if source_name in source.bones and target_name in target.bones:
                source_pos = source.bones[source_name].transform.position
                target_pos = target.bones[target_name].transform.position
                self.offset_map[source_name] = target_pos - source_pos

        return mapping

    def _create_ik_mapping(self, source: ProfessionalSkeleton,
                           target: ProfessionalSkeleton) -> Dict:
        return self._auto_map_by_name(source, target)

    def apply_retargeting(self, source_data: Dict, mapping: Dict,
                          method: RetargetingMethod = None) -> Dict:

        if method is None:
            method = self.mapping_method

        target_data = {}
        if method == RetargetingMethod.SCALE:
            target_data = self._apply_scale_retargeting(source_data, mapping)
        elif method == RetargetingMethod.OFFSET:
            target_data = self._apply_offset_retargeting(source_data, mapping)
        elif method == RetargetingMethod.IK:
            target_data = self._apply_ik_retargeting(source_data, mapping)

        return target_data

    def _apply_scale_retargeting(self, source_data: Dict, mapping: Dict) -> Dict:
        target_data = {}
        for source_bone, source_transform in source_data.items():
            if source_bone in mapping:
                target_bone = mapping[source_bone]
                target_transform = source_transform.copy()

                if source_bone in self.scale_factors:
                    scale = self.scale_factors[source_bone]
                    if 'position' in target_transform:
                        target_transform['position'] *= scale

                target_data[target_bone] = target_transform

        return target_data

    def _apply_offset_retargeting(self, source_data: Dict, mapping: Dict) -> Dict:
        target_data = {}
        for source_bone, source_transform in source_data.items():
            if source_bone in mapping:
                target_bone = mapping[source_bone]
                target_transform = source_transform.copy()

                if source_bone in self.offset_map:
                    if 'position' in target_transform:
                        target_transform['position'] += self.offset_map[source_bone]

                target_data[target_bone] = target_transform

        return target_data

    def _apply_ik_retargeting(self, source_data: Dict, mapping: Dict) -> Dict:
        return self._apply_scale_retargeting(source_data, mapping)


class ProfessionalSkeletonTracker:
    """ПРОФЕССИОНАЛЬНЫЙ ТРЕКЕР СКЕЛЕТА"""

    def __init__(self, config: Dict = None):
        # ЛОКАЛЬНЫЙ ИМПОРТ ВНУТРИ МЕТОДА
        from core.skeleton import ProfessionalSkeleton
        from core.animation_recorder import ProfessionalAnimationRecorder
        from core.camera_manager import MultiCameraManager

        self.config = config or {}
        self.state = TrackerState.IDLE
        self.current_mode = TrackingMode.PRECISE

        self.pose_estimator = None
        self.skeleton = None
        self.camera_manager = None
        self.animation_recorder = None
        self.pose_corrector = PoseCorrector()
        self.retargeting_system = RetargetingSystem()

        self.current_frame = None
        self.current_landmarks = []
        self.current_skeleton_data = {}
        self.frame_history = deque(maxlen=60)

        self.metrics = TrackingMetrics()
        self.statistics = {
            'total_frames': 0, 'frames_with_landmarks': 0,
            'avg_confidence': 0.0, 'avg_fps': 0.0, 'avg_latency': 0.0
        }

        self.landmark_to_bone_mapping = create_humanoid_mapping()
        self.correction_mode = CorrectionMode.AUTO
        self.lock = threading.Lock()
        self.processing = False

        self._initialize_components()
        logger.info("ProfessionalSkeletonTracker инициализирован")

    def _initialize_components(self):
        """Инициализация всех компонентов"""
        try:
            # ЛОКАЛЬНЫЙ ИМПОРТ
            from core.skeleton import ProfessionalSkeleton

            tracking_mode = self.config.get('tracking_mode', 'precise')
            self.pose_estimator = EnhancedPoseEstimator(
                mode=TrackingMode(tracking_mode.lower()),
                enable_kalman=self.config.get('enable_kalman', True),
                enable_smoothing=self.config.get('enable_smoothing', True),
                auto_calibrate=True
            )

            skeleton_name = self.config.get('skeleton_name', 'HumanSkeleton')
            self.skeleton = ProfessionalSkeleton(name=skeleton_name)

            self.state = TrackerState.INITIALIZING
            logger.info("Компоненты трекера инициализированы")

        except Exception as e:
            self.state = TrackerState.ERROR
            logger.error(f"Ошибка инициализации трекера: {e}")
            raise

    def set_camera_manager(self, camera_manager: MultiCameraManager):
        self.camera_manager = camera_manager
        logger.info("CameraManager установлен")

    def set_animation_recorder(self, recorder: ProfessionalAnimationRecorder):
        self.animation_recorder = recorder
        logger.info("AnimationRecorder установлен")

    def start_tracking(self, camera_id: int = 0) -> bool:
        if self.state == TrackerState.TRACKING:
            return True

        try:
            if not self.camera_manager:
                logger.error("CameraManager не установлен")
                return False

            self._reset_statistics()
            self.state = TrackerState.TRACKING
            self.processing = False

            logger.info(f"Трекинг запущен (камера: {camera_id})")
            return True

        except Exception as e:
            self.state = TrackerState.ERROR
            logger.error(f"Ошибка запуска трекинга: {e}")
            return False

    def stop_tracking(self):
        if self.state != TrackerState.TRACKING:
            return

        self.state = TrackerState.IDLE
        self.processing = False
        self.current_frame = None
        self.current_landmarks = []
        self.current_skeleton_data = {}
        self.frame_history.clear()
        logger.info("Трекинг остановлен")

    def process_frame(self, frame: np.ndarray = None, camera_id: int = 0) -> Optional[TrackingFrame]:
        if self.state != TrackerState.TRACKING or self.processing:
            return None

        self.processing = True
        start_time = time.time()

        try:
            if frame is None:
                if not self.camera_manager:
                    return None

                camera_frame = self.camera_manager.get_frame(camera_id, timeout=0.1)
                if camera_frame is None:
                    return None

                frame = camera_frame.image
                timestamp = camera_frame.timestamp
            else:
                timestamp = time.time()

            self.current_frame = frame.copy()
            tracking_results = self.pose_estimator.process_frame(frame)

            if not tracking_results:
                return None

            landmarks = tracking_results.get('detailed_landmarks', [])
            confidence = np.mean([lm.confidence for lm in landmarks]) if landmarks else 0.0

            if self.correction_mode != CorrectionMode.NONE:
                landmarks = self.pose_corrector.apply_correction(
                    landmarks, self.skeleton, self.correction_mode
                )

            skeleton_data = self._update_skeleton_from_landmarks(landmarks)

            processing_time = (time.time() - start_time) * 1000

            tracking_frame = TrackingFrame(
                timestamp=timestamp,
                frame_id=self.statistics['total_frames'],
                landmarks=np.array([lm.position for lm in landmarks]) if landmarks else np.array([]),
                skeleton_data=skeleton_data,
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    'tracking_mode': self.current_mode.value,
                    'correction_mode': self.correction_mode.value,
                    'image_size': frame.shape[:2]
                }
            )

            self.current_landmarks = landmarks
            self.current_skeleton_data = skeleton_data
            self.frame_history.append(tracking_frame)
            self._update_statistics(len(landmarks), confidence, processing_time)

            if confidence < 0.3 and self.current_mode != TrackingMode.ULTRA:
                self._auto_adjust_tracking_mode(confidence)

            return tracking_frame

        except Exception as e:
            logger.error(f"Ошибка обработки кадра: {e}")
            return None
        finally:
            self.processing = False

    def _update_skeleton_from_landmarks(self, landmarks: List) -> Dict[str, Dict]:
        skeleton_data = {}
        if not landmarks or not self.skeleton:
            return skeleton_data

        bone_positions = {}
        for lm_id, bone_name in self.landmark_to_bone_mapping.items():
            if lm_id < len(landmarks):
                landmark = landmarks[lm_id]
                if hasattr(landmark, 'position'):
                    if bone_name not in bone_positions:
                        bone_positions[bone_name] = []
                    bone_positions[bone_name].append(landmark.position[:3])

        for bone_name, positions in bone_positions.items():
            if positions:
                avg_position = np.mean(positions, axis=0)
                bone = self.skeleton.get_bone(bone_name)
                if bone:
                    current_rotation = bone.transform.rotation
                    skeleton_data[bone_name] = {
                        'position': avg_position.tolist(),
                        'rotation': current_rotation.tolist(),
                        'scale': [1.0, 1.0, 1.0]
                    }

        return skeleton_data

    def _update_statistics(self, landmarks_count: int, confidence: float, proc_time: float):
        self.statistics['total_frames'] += 1
        if landmarks_count > 0:
            self.statistics['frames_with_landmarks'] += 1

        alpha = 0.1
        self.statistics['avg_confidence'] = alpha * confidence + (1 - alpha) * self.statistics['avg_confidence']
        self.statistics['avg_latency'] = alpha * proc_time + (1 - alpha) * self.statistics['avg_latency']

        self.metrics.update(landmarks_count, confidence, proc_time)

    def _reset_statistics(self):
        self.statistics = {
            'total_frames': 0, 'frames_with_landmarks': 0,
            'avg_confidence': 0.0, 'avg_fps': 0.0, 'avg_latency': 0.0
        }
        self.metrics = TrackingMetrics()

    def _auto_adjust_tracking_mode(self, current_confidence: float):
        if current_confidence < 0.2:
            self.set_tracking_mode('ultra')
            logger.info(f"Авто-переключение на ULTRA режим")
        elif current_confidence < 0.3 and self.current_mode == TrackingMode.FAST:
            self.set_tracking_mode('precise')
            logger.info(f"Авто-переключение на PRECISE режим")

    def set_tracking_mode(self, mode: str):
        try:
            mode_enum = TrackingMode(mode.upper())
            self.current_mode = mode_enum
            self.pose_estimator.set_tracking_mode(mode_enum)
            logger.info(f"Режим трекинга изменен на: {mode}")
        except ValueError:
            logger.error(f"Неизвестный режим трекинга: {mode}")

    # Остальные методы остаются без изменений...


# Для обратной совместимости
SkeletonTracker = ProfessionalSkeletonTracker

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Тестирование ProfessionalSkeletonTracker...")

    config = {
        'tracking_mode': 'precise',
        'enable_kalman': True,
        'enable_smoothing': True,
        'skeleton_name': 'TestHuman'
    }

    tracker = ProfessionalSkeletonTracker(config)
    print(f"Трекер создан")
    print(f"Состояние: {tracker.state.value}")
    print(f"Режим: {tracker.current_mode.value}")

    stats = tracker.get_statistics()
    print(f"Статистика: {stats}")
    print("Тест завершен!")