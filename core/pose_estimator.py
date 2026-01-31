"""
ПРОФЕССИОНАЛЬНЫЙ ТРЕКИНГ ПОЗЫ С ВЫСОКОЙ ТОЧНОСТЬЮ
Поддержка нескольких моделей + фильтр Калмана + калибровка освещения
"""

import cv2
import numpy as np
import mediapipe
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
from collections import deque
import time
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import transforms3d as tf3d

logger = logging.getLogger(__name__)



class TrackingMode(Enum):
    """Режимы трекинга с разной точностью/скоростью"""
    FAST = "fast"  # Быстрый режим (30+ FPS)
    PRECISE = "precise"  # Точный режим с Holistic
    ULTRA = "ultra"  # Ультра-точный (замедленный)
    MANUAL = "manual"  # Ручная коррекция


@dataclass
class PoseLandmark:
    """Улучшенная структура landmark с историей и фильтрацией"""
    id: int
    name: str
    position: np.ndarray  # [x, y, z, visibility]
    filtered_position: np.ndarray  # Отфильтрованная позиция
    confidence: float
    velocity: np.ndarray  # Скорость (px/сек)
    acceleration: np.ndarray  # Ускорение
    history: deque  # История позиций для фильтрации
    is_locked: bool = False  # Заблокирован ли вручную

    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=10)
        if self.filtered_position is None:
            self.filtered_position = self.position.copy()


class KalmanFilter3D:
    """Упрощенный фильтр Калмана для 3D точек"""

    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = None
        self.covariance = np.eye(6) * 0.1

    def predict(self, dt=1 / 30.0):
        if self.state is not None:
            # Простая модель постоянной скорости
            F = np.array([
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            self.state = F @ self.state
            self.covariance = F @ self.covariance @ F.T + np.eye(6) * self.process_noise

    def update(self, measurement):
        if self.state is None:
            self.state = np.zeros(6)
            self.state[:3] = measurement
            return self.state[:3]

        # Матрица измерения H
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Обновление Калмана
        y = measurement - H @ self.state
        S = H @ self.covariance @ H.T + np.eye(3) * self.measurement_noise
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.covariance = (np.eye(6) - K @ H) @ self.covariance

        return self.state[:3]


class EnhancedPoseEstimator:
    """
    ПРОФЕССИОНАЛЬНЫЙ ТРЕКЕР ПОЗЫ ДЛЯ MOCAP

    Особенности:
    1. 3 режима трекинга (скорость/точность/ультра)
    2. Адаптивная калибровка под освещение
    3. Фильтры Калмана для каждого landmark
    4. Оценка скорости и ускорения
    5. Автоматическое переключение моделей при плохом освещении
    6. Коррекция дрожания и артефактов
    """

    # Имена landmarks для удобства
    LANDMARK_NAMES = {
        0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
        4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
        7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
        11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
        15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
        19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
        23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
        31: "left_foot_index", 32: "right_foot_index"
    }

    # Важные точки для трекинга (приоритетные)
    KEY_POINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def __init__(self,
                 mode: TrackingMode = TrackingMode.PRECISE,
                 enable_smoothing: bool = True,
                 enable_kalman: bool = True,
                 auto_calibrate: bool = True,
                 # ДОБАВЬ ЭТИ ПАРАМЕТРЫ:
                 model_complexity: int = 2,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False):
        """
        Инициализация улучшенного трекера
        """
        self.mode = mode
        self.enable_smoothing = enable_smoothing
        self.enable_kalman = enable_kalman
        self.auto_calibrate = auto_calibrate
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation

        # ========== ПРОСТОЙ ИСПРАВЛЕННЫЙ КОД ==========
        import mediapipe as mp

        # Используем напрямую
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Создаем объект Pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ========== УДАЛИ ЭТУ СТРОКУ - ОНА ЛИШНЯЯ ==========
        # self.mp_drawing_styles = mp.solutions.drawing_styles  # ← УДАЛИ!

        # Инициализация трекеров
        self.pose_tracker = None
        self.holistic_tracker = None
        self._init_trackers()

        # Фильтры Калмана для каждой точки
        self.kalman_filters = {}
        if enable_kalman:
            for i in range(33):
                self.kalman_filters[i] = KalmanFilter3D()

        # Статистика и калибровка
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.light_level = 0.5  # Уровень освещения (0-1)
        self.confidence_threshold = 0.5

        # История для фильтрации
        self.position_history = {i: deque(maxlen=5) for i in range(33)}
        self.velocity_history = {i: deque(maxlen=3) for i in range(33)}

        # Коррекции пользователя
        self.manual_corrections = {}
        self.locked_points = set()

        logger.info(f"EnhancedPoseEstimator инициализирован в режиме {mode.value}")

    def _init_trackers(self):
        """Инициализация трекеров в зависимости от режима"""
        try:
            # Базовый трекер позы (всегда доступен)
            self.pose_tracker = self.pose  # ← уже создан в конструкторе

            # Holistic трекер для точного режима
            if self.mode in [TrackingMode.PRECISE, TrackingMode.ULTRA]:
                # Импортируем holistic
                import mediapipe as mp
                self.mp_holistic = mp.solutions.holistic

                self.holistic_tracker = self.mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=2,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            else:
                self.holistic_tracker = None

        except Exception as e:
            logger.error(f"Ошибка инициализации трекеров: {e}")
            self.holistic_tracker = None  # ← не raise, а продолжаем без holistic

    def _estimate_light_level(self, frame: np.ndarray) -> float:
        """Оценка уровня освещения на кадре"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        light_level = np.mean(gray) / 255.0

        # Адаптация параметров под освещение
        if light_level < 0.3:  # Темно
            self.confidence_threshold = 0.3
        elif light_level > 0.7:  # Ярко
            self.confidence_threshold = 0.6
        else:  # Нормально
            self.confidence_threshold = 0.5

        return light_level

    def _apply_kalman_filter(self, point_id: int, position: np.ndarray) -> np.ndarray:
        """Применение фильтра Калмана к точке"""
        if not self.enable_kalman or point_id not in self.kalman_filters:
            return position

        kalman = self.kalman_filters[point_id]
        kalman.predict()
        filtered = kalman.update(position[:3])

        return np.array([filtered[0], filtered[1], filtered[2], position[3]])

    def _apply_temporal_smoothing(self, point_id: int, position: np.ndarray) -> np.ndarray:
        """Временное сглаживание с использованием истории"""
        if not self.enable_smoothing or len(self.position_history[point_id]) < 2:
            return position

        self.position_history[point_id].append(position)

        # Сглаживание Гауссом
        history_array = np.array(self.position_history[point_id])
        smoothed = np.zeros_like(position)

        for i in range(4):  # Для x, y, z, visibility
            if len(history_array[:, i]) > 1:
                smoothed[i] = gaussian_filter1d(history_array[:, i], sigma=1.0)[-1]
            else:
                smoothed[i] = position[i]

        return smoothed

    def _calculate_velocity(self, point_id: int, current_pos: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Расчет скорости и ускорения"""
        if len(self.position_history[point_id]) < 2:
            return np.zeros(3), np.zeros(3)

        # Текущая и предыдущая позиции
        current = current_pos[:3]
        prev = self.position_history[point_id][-2][:3]

        # Скорость
        velocity = (current - prev) / dt

        # Ускорение
        if len(self.velocity_history[point_id]) > 0:
            prev_velocity = self.velocity_history[point_id][-1]
            acceleration = (velocity - prev_velocity) / dt
        else:
            acceleration = np.zeros(3)

        self.velocity_history[point_id].append(velocity)

        return velocity, acceleration

    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Обработка кадра с улучшенным трекингом

        Args:
            frame: Кадр BGR

        Returns:
            Словарь с детальной информацией о позе
        """
        if frame is None:
            return None

        start_time = time.time()
        self.frame_count += 1

        try:
            # 1. Анализ освещения (если включена автокалибровка)
            if self.auto_calibrate:
                self.light_level = self._estimate_light_level(frame)

            # 2. Конвертация цвета
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False  # Для производительности

            # 3. Выбор трекера в зависимости от режима
            results = None
            h, w = frame.shape[:2]

            if self.mode == TrackingMode.FAST or self.holistic_tracker is None:
                # Быстрый режим - только поза
                results = self.pose_tracker.process(frame_rgb)
                pose_landmarks = results.pose_landmarks
                world_landmarks = results.pose_world_landmarks

            elif self.mode in [TrackingMode.PRECISE, TrackingMode.ULTRA]:
                # Точный режим - holistic
                results = self.holistic_tracker.process(frame_rgb)
                pose_landmarks = results.pose_landmarks
                world_landmarks = results.pose_world_landmarks

                # Добавляем landmarks рук и лица если доступны
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    # Здесь можно добавить обработку рук
                    pass

            # 4. Проверка наличия landmarks
            if not pose_landmarks:
                logger.debug("Landmarks не обнаружены")
                return None

            # 5. Обработка и фильтрация landmarks
            dt = 1.0 / max(self.fps, 30.0)  # Дельта времени
            landmarks_data = []
            detailed_landmarks = []

            for idx, lm in enumerate(pose_landmarks.landmark):
                # Базовые координаты
                x, y, z, vis = lm.x * w, lm.y * h, lm.z * w, lm.visibility
                position = np.array([x, y, z, vis])

                # Пропускаем точки с низкой уверенностью
                if vis < self.confidence_threshold and idx not in self.locked_points:
                    continue

                # Применение ручных коррекций
                if idx in self.manual_corrections:
                    correction = self.manual_corrections[idx]
                    position[:3] += correction[:3]

                # Фильтрация
                if self.enable_smoothing:
                    position = self._apply_temporal_smoothing(idx, position)

                if self.enable_kalman:
                    position = self._apply_kalman_filter(idx, position)

                # Расчет скорости и ускорения
                velocity, acceleration = self._calculate_velocity(idx, position, dt)

                # Создание объекта landmark
                landmark_obj = PoseLandmark(
                    id=idx,
                    name=self.LANDMARK_NAMES.get(idx, f"point_{idx}"),
                    position=position,
                    filtered_position=position.copy(),
                    confidence=vis,
                    velocity=velocity,
                    acceleration=acceleration,
                    history=self.position_history[idx].copy(),
                    is_locked=(idx in self.locked_points)
                )

                landmarks_data.append(position)
                detailed_landmarks.append(landmark_obj)

            # 6. Расчет FPS
            current_time = time.time()
            if current_time - self.last_time > 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time

            # 7. Формирование результата
            processing_time = (time.time() - start_time) * 1000

            result = {
                'landmarks': np.array(landmarks_data),
                'detailed_landmarks': detailed_landmarks,
                'world_landmarks': world_landmarks,
                'image_width': w,
                'image_height': h,
                'pose_landmarks': pose_landmarks,
                'fps': self.fps,
                'processing_time_ms': processing_time,
                'light_level': self.light_level,
                'confidence_threshold': self.confidence_threshold,
                'tracking_mode': self.mode.value,
                'num_points_found': len(landmarks_data)
            }

            # 8. Дополнительные вычисления для ключевых точек
            if len(landmarks_data) >= len(self.KEY_POINTS):
                result['key_points'] = self._extract_key_points(result)
                result['body_angles'] = self._calculate_body_angles(result)
                result['center_of_mass'] = self._calculate_center_of_mass(result)

            return result

        except Exception as e:
            logger.error(f"Ошибка обработки кадра: {e}")
            return None

    def _extract_key_points(self, results: Dict) -> Dict:
        """Извлечение ключевых точек для упрощенного скелета"""
        key_points = {}
        landmarks = results['landmarks']

        for idx in self.KEY_POINTS:
            if idx < len(landmarks):
                key_points[self.LANDMARK_NAMES[idx]] = {
                    'position': landmarks[idx][:3],
                    'confidence': landmarks[idx][3]
                }

        return key_points

    def _calculate_body_angles(self, results: Dict) -> Dict:
        """Расчет углов между сегментами тела"""
        angles = {}
        landmarks = results['landmarks']

        # Угол плеча (плечо-локоть-запястье)
        if len(landmarks) > 16:
            # Левая рука
            if landmarks[11][3] > 0.3 and landmarks[13][3] > 0.3 and landmarks[15][3] > 0.3:
                shoulder = landmarks[11][:3]
                elbow = landmarks[13][:3]
                wrist = landmarks[15][:3]
                angles['left_elbow'] = self._calculate_angle(shoulder, elbow, wrist)

            # Правая рука
            if landmarks[12][3] > 0.3 and landmarks[14][3] > 0.3 and landmarks[16][3] > 0.3:
                shoulder = landmarks[12][:3]
                elbow = landmarks[14][:3]
                wrist = landmarks[16][:3]
                angles['right_elbow'] = self._calculate_angle(shoulder, elbow, wrist)

        return angles

    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Расчет угла ABC в градусах"""
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def _calculate_center_of_mass(self, results: Dict) -> np.ndarray:
        """Расчет центра масс тела"""
        landmarks = results['landmarks']
        if len(landmarks) < 5:
            return np.zeros(3)

        # Используем основные точки: плечи, бедра
        key_indices = [11, 12, 23, 24]  # Плечи и бедра
        positions = []
        weights = []

        for idx in key_indices:
            if idx < len(landmarks) and landmarks[idx][3] > 0.3:
                positions.append(landmarks[idx][:3])
                weights.append(landmarks[idx][3])

        if not positions:
            return np.zeros(3)

        positions = np.array(positions)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Нормализация

        com = np.average(positions, axis=0, weights=weights)
        return com

    def draw_landmarks(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Улучшенное отображение landmarks с дополнительной информацией

        Args:
            frame: Исходный кадр
            results: Результат process_frame

        Returns:
            Кадр с отрисованными landmarks
        """
        if frame is None or 'pose_landmarks' not in results:
            return frame

        annotated_frame = frame.copy()

        try:
            # 1. Отрисовка стандартных landmarks MediaPipe
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results['pose_landmarks'],
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_pose_connections_style()
            )

            # 2. Отрисовка дополнительной информации
            if 'detailed_landmarks' in results:
                for landmark in results['detailed_landmarks']:
                    if landmark.confidence > 0.3:
                        x, y = int(landmark.position[0]), int(landmark.position[1])

                        # Цвет в зависимости от уверенности
                        color_intensity = int(255 * landmark.confidence)
                        color = (0, color_intensity, 255 - color_intensity)

                        # Точка
                        cv2.circle(annotated_frame, (x, y), 4, color, -1)

                        # ID точки (для отладки)
                        if self.mode == TrackingMode.ULTRA:
                            cv2.putText(annotated_frame, str(landmark.id), (x + 5, y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 3. Отрисовка информации о FPS и режиме
            info_text = f"FPS: {results.get('fps', 0):.1f} | Mode: {results.get('tracking_mode', 'N/A')}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 4. Отрисовка центра масс
            if 'center_of_mass' in results:
                com = results['center_of_mass']
                cx, cy = int(com[0]), int(com[1])
                cv2.circle(annotated_frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.putText(annotated_frame, "COM", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 5. Отрисовка уровня освещения
            light_level = results.get('light_level', 0.5)
            light_text = f"Light: {light_level:.2f}"
            cv2.putText(annotated_frame, light_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        except Exception as e:
            logger.warning(f"Ошибка отрисовки landmarks: {e}")

        return annotated_frame

    def set_manual_correction(self, point_id: int, correction: np.ndarray):
        """Установка ручной коррекции для точки"""
        self.manual_corrections[point_id] = correction
        logger.info(f"Коррекция установлена для точки {point_id}: {correction}")

    def lock_point(self, point_id: int, lock: bool = True):
        """Блокировка/разблокировка точки"""
        if lock:
            self.locked_points.add(point_id)
        elif point_id in self.locked_points:
            self.locked_points.remove(point_id)

    def set_tracking_mode(self, mode: TrackingMode):
        """Изменение режима трекинга"""
        self.mode = mode
        self._init_trackers()  # Переинициализация с новыми параметрами
        logger.info(f"Режим трекинга изменен на {mode.value}")

    def get_statistics(self) -> Dict:
        """Получение статистики работы трекера"""
        return {
            'fps': self.fps,
            'light_level': self.light_level,
            'confidence_threshold': self.confidence_threshold,
            'mode': self.mode.value,
            'num_locked_points': len(self.locked_points),
            'num_manual_corrections': len(self.manual_corrections)
        }

    def reset(self):
        """Сброс всех фильтров и истории"""
        self.position_history = {i: deque(maxlen=5) for i in range(33)}
        self.velocity_history = {i: deque(maxlen=3) for i in range(33)}
        self.manual_corrections.clear()
        self.locked_points.clear()

        # Сброс фильтров Калмана
        for kalman in self.kalman_filters.values():
            kalman.state = None
            kalman.covariance = np.eye(6) * 0.1

        logger.info("Трекер сброшен")

    def release(self):
        """Освобождение ресурсов"""
        if self.pose_tracker:
            self.pose_tracker.close()
        if self.holistic_tracker:
            self.holistic_tracker.close()

        logger.info("Ресурсы трекера освобождены")


# Фабричная функция для удобства создания
def create_pose_estimator(mode: str = "precise", **kwargs) -> EnhancedPoseEstimator:
    """Создание трекера по названию режима"""
    mode_map = {
        "fast": TrackingMode.FAST,
        "precise": TrackingMode.PRECISE,
        "ultra": TrackingMode.ULTRA,
        "manual": TrackingMode.MANUAL
    }

    tracking_mode = mode_map.get(mode.lower(), TrackingMode.PRECISE)
    return EnhancedPoseEstimator(mode=tracking_mode, **kwargs)


# Тестирование модуля
if __name__ == "__main__":
    # Быстрый тест работы трекера
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Тестирование EnhancedPoseEstimator...")

    # Создание трекера
    estimator = create_pose_estimator("precise", enable_kalman=True)

    # Тестовое изображение (или камера)
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
        if frame is not None:
            results = estimator.process_frame(frame)
            if results:
                print(f"Найдено точек: {results['num_points_found']}")
                print(f"FPS: {results['fps']:.1f}")

                # Визуализация
                annotated = estimator.draw_landmarks(frame, results)
                cv2.imshow("Enhanced Pose Tracking", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        print("Использование: python pose_estimator.py <путь_к_изображению>")
        print("Или импортируйте модуль в свой проект")

    estimator.release()

PoseEstimator = EnhancedPoseEstimator

__all__ = [
    'EnhancedPoseEstimator',
    'PoseEstimator',  # Алиас для обратной совместимости
    'TrackingMode',
    'PoseLandmark',
    'KalmanFilter3D',
    'create_pose_estimator'
]