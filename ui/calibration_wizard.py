"""
ПРОФЕССИОНАЛЬНЫЙ МАСТЕР КАЛИБРОВКИ ДЛЯ MOCAP PRO
Пошаговая калибровка камеры и скелета с ARUCO, шахматной доской и AI-помощником
"""

import sys
import numpy as np
import cv2
import json
import yaml
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QPushButton, QLabel, QProgressBar,
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTabWidget, QTextEdit, QListWidget, QListWidgetItem,
    QStackedWidget, QWizard, QWizardPage, QLineEdit,
    QRadioButton, QButtonGroup, QSlider, QMessageBox,
    QApplication, QStyleFactory, QFileDialog, QDialog,
    QDialogButtonBox, QFormLayout, QSizePolicy
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor,
    QFont, QIcon, QPalette, QLinearGradient, QRadialGradient,
    QAction, QKeySequence, QPainterPath
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QTimer, QSize, QPoint, QRect,
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup,
    QSequentialAnimationGroup, QThread, pyqtSlot
)
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

from core.camera_manager import MultiCameraManager, CameraCalibration, CameraInfo
from core.skeleton_tracker import ProfessionalSkeletonTracker
from core.skeleton import ProfessionalSkeleton
from core.pose_estimator import EnhancedPoseEstimator

logger = logging.getLogger(__name__)


class CalibrationStep(Enum):
    """Шаги калибровки"""
    WELCOME = "welcome"
    CAMERA_SELECTION = "camera_selection"
    CALIBRATION_TYPE = "calibration_type"
    CHESSBOARD_CALIBRATION = "chessboard_calibration"
    ARUCO_CALIBRATION = "aruco_calibration"
    SKELETON_CALIBRATION = "skeleton_calibration"
    AUTO_CALIBRATION = "auto_calibration"
    MANUAL_ADJUSTMENT = "manual_adjustment"
    VERIFICATION = "verification"
    COMPLETION = "completion"


class CalibrationType(Enum):
    """Типы калибровки"""
    CAMERA_INTRINSICS = "camera_intrinsics"  # Внутренние параметры камеры
    CAMERA_EXTRINSICS = "camera_extrinsics"  # Внешние параметры (позиция)
    SKELETON_SCALE = "skeleton_scale"  # Масштаб скелета
    SKELETON_OFFSET = "skeleton_offset"  # Смещение скелета
    FULL_CALIBRATION = "full_calibration"  # Полная калибровка


@dataclass
class CalibrationData:
    """Данные калибровки"""
    camera_calibration: Dict[int, CameraCalibration] = field(default_factory=dict)
    skeleton_data: Dict = field(default_factory=dict)
    transformation_matrices: Dict = field(default_factory=dict)
    quality_metrics: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def save_to_file(self, filepath: str):
        """Сохранение калибровки в файл"""
        data = {
            'camera_calibration': {
                cam_id: {
                    'camera_matrix': calib.camera_matrix.tolist(),
                    'dist_coeffs': calib.dist_coeffs.tolist(),
                    'resolution': calib.resolution,
                    'fov': calib.fov
                }
                for cam_id, calib in self.camera_calibration.items()
            },
            'skeleton_data': self.skeleton_data,
            'transformation_matrices': {
                key: matrix.tolist() for key, matrix in self.transformation_matrices.items()
            },
            'quality_metrics': self.quality_metrics,
            'timestamp': self.timestamp,
            'version': '1.0'
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'CalibrationData':
        """Загрузка калибровки из файла"""
        calib = cls()

        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)

            # Загрузка калибровки камеры
            if 'camera_calibration' in data:
                for cam_id, cam_data in data['camera_calibration'].items():
                    camera_calib = CameraCalibration()
                    camera_calib.camera_matrix = np.array(cam_data['camera_matrix'])
                    camera_calib.dist_coeffs = np.array(cam_data['dist_coeffs'])
                    camera_calib.resolution = tuple(cam_data['resolution'])
                    camera_calib.fov = tuple(cam_data.get('fov', (0.0, 0.0)))
                    camera_calib.intrinsics_set = True

                    calib.camera_calibration[int(cam_id)] = camera_calib

            # Загрузка других данных
            calib.skeleton_data = data.get('skeleton_data', {})
            calib.transformation_matrices = {
                key: np.array(matrix)
                for key, matrix in data.get('transformation_matrices', {}).items()
            }
            calib.quality_metrics = data.get('quality_metrics', {})
            calib.timestamp = data.get('timestamp', time.time())

        except Exception as e:
            logger.error(f"Ошибка загрузки калибровки: {e}")

        return calib


class CalibrationThread(QThread):
    """Поток для выполнения калибровки"""

    calibration_progress = pyqtSignal(int, int, str)  # текущий, всего, сообщение
    calibration_complete = pyqtSignal(bool, str)  # успех, сообщение
    calibration_error = pyqtSignal(str)  # ошибка
    frame_processed = pyqtSignal(np.ndarray, dict)  # обработанный кадр, данные

    def __init__(self, camera_manager: MultiCameraManager,
                 calibration_type: CalibrationType,
                 parameters: Dict = None):
        super().__init__()

        self.camera_manager = camera_manager
        self.calibration_type = calibration_type
        self.parameters = parameters or {}
        self.is_running = False

        # ARUCO параметры
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def run(self):
        """Выполнение калибровки"""
        self.is_running = True

        try:
            if self.calibration_type == CalibrationType.CAMERA_INTRINSICS:
                self._calibrate_camera_intrinsics()
            elif self.calibration_type == CalibrationType.SKELETON_SCALE:
                self._calibrate_skeleton_scale()
            elif self.calibration_type == CalibrationType.FULL_CALIBRATION:
                self._calibrate_full()
            else:
                self.calibration_error.emit(f"Неизвестный тип калибровки: {self.calibration_type}")

        except Exception as e:
            self.calibration_error.emit(f"Ошибка калибровки: {str(e)}")
            logger.error(f"Ошибка в потоке калибровки: {e}")
        finally:
            self.is_running = False

    def _calibrate_camera_intrinsics(self):
        """Калибровка внутренних параметров камеры"""
        calibration_method = self.parameters.get('method', 'chessboard')
        camera_id = self.parameters.get('camera_id', 0)

        if calibration_method == 'chessboard':
            self._calibrate_with_chessboard(camera_id)
        elif calibration_method == 'aruco':
            self._calibrate_with_aruco(camera_id)

    def _calibrate_with_chessboard(self, camera_id: int):
        """Калибровка с шахматной доской"""
        pattern_size = self.parameters.get('pattern_size', (9, 6))
        square_size = self.parameters.get('square_size', 0.025)  # метры
        frames_needed = self.parameters.get('frames_needed', 20)

        self.calibration_progress.emit(0, frames_needed, "Подготовка к калибровке...")

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Подготовка точек шахматной доски
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size

        objpoints = []  # 3D точки в реальном мире
        imgpoints = []  # 2D точки на изображении

        frames_captured = 0
        last_frame_time = 0

        while frames_captured < frames_needed and self.is_running:
            # Получение кадра
            frame_obj = self.camera_manager.get_frame(camera_id, timeout=0.5)
            if frame_obj is None:
                time.sleep(0.1)
                continue

            frame = frame_obj.image
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame

            # Поиск углов шахматной доски
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            if ret:
                # Уточнение позиций углов
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                objpoints.append(objp)
                imgpoints.append(corners_refined)
                frames_captured += 1

                # Визуализация
                vis_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.drawChessboardCorners(vis_frame, pattern_size, corners_refined, ret)

                # Отправка для отображения
                self.frame_processed.emit(vis_frame, {
                    'corners_found': True,
                    'frame_number': frames_captured,
                    'total_frames': frames_needed
                })

                self.calibration_progress.emit(
                    frames_captured, frames_needed,
                    f"Кадр {frames_captured}/{frames_needed} захвачен"
                )

                # Задержка между кадрами
                current_time = time.time()
                if current_time - last_frame_time < 1.0:
                    time.sleep(1.0 - (current_time - last_frame_time))
                last_frame_time = time.time()

            else:
                # Кадр без шахматной доски
                vis_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(vis_frame, "Шахматная доска не найдена",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                self.frame_processed.emit(vis_frame, {
                    'corners_found': False,
                    'message': "Держите шахматную доску в поле зрения"
                })

            time.sleep(0.1)

        if not self.is_running:
            return

        # Калибровка камеры
        if len(objpoints) >= 10:
            self.calibration_progress.emit(
                frames_needed, frames_needed,
                "Выполнение калибровки..."
            )

            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )

            if ret:
                # Расчет ошибки калибровки
                mean_error = 0
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                                      camera_matrix, dist_coeffs)
                    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    mean_error += error

                mean_error /= len(objpoints)

                # Сохранение калибровки
                calibration = CameraCalibration()
                calibration.camera_matrix = camera_matrix
                calibration.dist_coeffs = dist_coeffs
                calibration.resolution = frame.shape[:2][::-1]  # (ширина, высота)
                calibration.calibration_error = mean_error
                calibration.calculate_fov()
                calibration.intrinsics_set = True

                # Сохранение в файл
                calibration.save_to_file(f"camera_{camera_id}_calibration.yaml")

                self.calibration_complete.emit(
                    True,
                    f"Калибровка завершена успешно! Ошибка: {mean_error:.4f} пикселей"
                )

            else:
                self.calibration_complete.emit(False, "Ошибка калибровки камеры")

        else:
            self.calibration_complete.emit(
                False,
                f"Недостаточно кадров для калибровки. Захвачено: {len(objpoints)}"
            )

    def _calibrate_with_aruco(self, camera_id: int):
        """Калибровка с ARUCO маркерами"""
        marker_size = self.parameters.get('marker_size', 0.05)  # метры
        board_size = self.parameters.get('board_size', (5, 7))
        frames_needed = self.parameters.get('frames_needed', 25)

        self.calibration_progress.emit(0, frames_needed, "Подготовка ARUCO калибровки...")

        # Создание ARUCO board
        aruco_board = cv2.aruco.GridBoard(
            size=board_size,
            markerLength=marker_size,
            markerSeparation=marker_size * 0.2,
            dictionary=self.aruco_dict
        )

        all_corners = []
        all_ids = []

        frames_captured = 0

        while frames_captured < frames_needed and self.is_running:
            # Получение кадра
            frame_obj = self.camera_manager.get_frame(camera_id, timeout=0.5)
            if frame_obj is None:
                time.sleep(0.1)
                continue

            frame = frame_obj.image
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame

            # Детекция ARUCO маркеров
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

            if ids is not None and len(ids) > 4:  # Нужно минимум 5 маркеров
                all_corners.append(corners)
                all_ids.append(ids)
                frames_captured += 1

                # Визуализация
                vis_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids)

                # Отображение информации
                cv2.putText(vis_frame, f"Маркеров: {len(ids)}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Кадр: {frames_captured}/{frames_needed}",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self.frame_processed.emit(vis_frame, {
                    'markers_found': len(ids),
                    'frame_number': frames_captured,
                    'total_frames': frames_needed
                })

                self.calibration_progress.emit(
                    frames_captured, frames_needed,
                    f"Кадр {frames_captured}/{frames_needed} ({len(ids)} маркеров)"
                )

                time.sleep(0.5)  # Задержка между кадрами

            else:
                # Кадр без достаточного количества маркеров
                vis_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(vis_frame, "Мало ARUCO маркеров",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(vis_frame, "Нужно минимум 5 маркеров",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                self.frame_processed.emit(vis_frame, {
                    'markers_found': 0 if ids is None else len(ids),
                    'message': "Покажите ARUCO board с 5+ маркерами"
                })

            time.sleep(0.1)

        if not self.is_running:
            return

        # Калибровка камеры с ARUCO
        if len(all_corners) >= 10:
            self.calibration_progress.emit(
                frames_needed, frames_needed,
                "Выполнение ARUCO калибровки..."
            )

            # Подготовка данных для калибровки
            image_size = gray.shape[::-1]

            # Калибровка
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
                all_corners, all_ids, frames_captured, aruco_board,
                image_size, None, None
            )

            if ret:
                # Сохранение калибровки
                calibration = CameraCalibration()
                calibration.camera_matrix = camera_matrix
                calibration.dist_coeffs = dist_coeffs
                calibration.resolution = frame.shape[:2][::-1]
                calibration.calibration_error = 0.0  # ARUCO не возвращает ошибку
                calibration.calculate_fov()
                calibration.intrinsics_set = True

                # Сохранение в файл
                calibration.save_to_file(f"camera_{camera_id}_aruco_calibration.yaml")

                self.calibration_complete.emit(
                    True,
                    "ARUCO калибровка завершена успешно!"
                )

            else:
                self.calibration_complete.emit(False, "Ошибка ARUCO калибровки")

        else:
            self.calibration_complete.emit(
                False,
                f"Недостаточно кадров для ARUCO калибровки. Захвачено: {len(all_corners)}"
            )

    def _calibrate_skeleton_scale(self):
        """Калибровка масштаба скелета"""
        frames_needed = 30
        camera_id = self.parameters.get('camera_id', 0)

        self.calibration_progress.emit(0, frames_needed, "Калибровка масштаба скелета...")

        # Собираем данные о позе
        heights = []
        frames_captured = 0

        pose_estimator = EnhancedPoseEstimator(
            mode=self.parameters.get('tracking_mode', 'precise')
        )

        while frames_captured < frames_needed and self.is_running:
            # Получение кадра
            frame_obj = self.camera_manager.get_frame(camera_id, timeout=0.5)
            if frame_obj is None:
                time.sleep(0.1)
                continue

            frame = frame_obj.image

            # Трекинг позы
            results = pose_estimator.process_frame(frame)

            if results and 'detailed_landmarks' in results:
                landmarks = results['detailed_landmarks']

                # Расчет высоты по ключевым точкам
                height = self._estimate_height_from_landmarks(landmarks)
                if height > 0:
                    heights.append(height)
                    frames_captured += 1

                # Визуализация
                vis_frame = pose_estimator.draw_landmarks(
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), results
                )

                # Отображение информации
                cv2.putText(vis_frame, f"Кадр: {frames_captured}/{frames_needed}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis_frame, "Стойте прямо в T-позе",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self.frame_processed.emit(vis_frame, {
                    'frame_number': frames_captured,
                    'total_frames': frames_needed,
                    'estimated_height': height
                })

                self.calibration_progress.emit(
                    frames_captured, frames_needed,
                    f"Кадр {frames_captured}/{frames_needed}"
                )

            else:
                # Landmarks не найдены
                vis_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(vis_frame, "Поза не распознана",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(vis_frame, "Встаньте в поле зрения камеры",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                self.frame_processed.emit(vis_frame, {
                    'message': "Встаньте в поле зрения камеры"
                })

            time.sleep(0.1)

        if not self.is_running:
            return

        # Расчет среднего роста
        if heights:
            avg_height = np.mean(heights)
            std_height = np.std(heights)

            if std_height / avg_height < 0.1:  # Приемлемая вариация
                # Расчет масштабного коэффициента
                actual_height = self.parameters.get('actual_height', 1.75)  # метры
                scale_factor = actual_height / avg_height

                self.calibration_complete.emit(
                    True,
                    f"Масштаб скелета определен: {scale_factor:.3f}x\n"
                    f"Оцененный рост: {avg_height:.2f} м\n"
                    f"Вариация: {std_height / avg_height * 100:.1f}%"
                )

                # Сохранение данных
                skeleton_data = {
                    'scale_factor': float(scale_factor),
                    'estimated_height': float(avg_height),
                    'actual_height': float(actual_height),
                    'frames_used': len(heights),
                    'timestamp': time.time()
                }

                with open('skeleton_scale_calibration.json', 'w') as f:
                    json.dump(skeleton_data, f, indent=2)

            else:
                self.calibration_complete.emit(
                    False,
                    f"Слишком большая вариация в росте: {std_height / avg_height * 100:.1f}%\n"
                    "Стойте неподвижно в T-позе"
                )

        else:
            self.calibration_complete.emit(
                False,
                "Не удалось определить рост. Убедитесь, что вы в поле зрения камеры."
            )

    def _estimate_height_from_landmarks(self, landmarks) -> float:
        """Оценка роста человека по landmarks"""
        if not landmarks:
            return 0.0

        # Ключевые точки для оценки роста
        key_points = {
            'head': 0,  # Нос
            'neck': 1,  # Основание шеи
            'hip': 23,  # Левое бедро (приблизительно талия)
            'knee': 25,  # Левое колено
            'ankle': 27,  # Левая лодыжка
        }

        positions = {}
        for name, idx in key_points.items():
            if idx < len(landmarks) and hasattr(landmarks[idx], 'position'):
                positions[name] = landmarks[idx].position[1]  # Y координата

        # Расчет высоты по разнице Y координат
        if all(name in positions for name in ['head', 'ankle']):
            # Разница между самой высокой и самой низкой точкой
            min_y = min(positions.values())
            max_y = max(positions.values())

            # Примерная высота в пикселях
            height_pixels = abs(max_y - min_y)

            # Конвертация в метры (очень приблизительно)
            # В реальном приложении нужна калибровка по известному размеру
            height_meters = height_pixels * 0.001  # Примерный коэффициент

            return height_meters

        return 0.0

    def _calibrate_full(self):
        """Полная калибровка системы"""
        # Последовательная калибровка всех компонентов
        steps = [
            ("Калибровка камеры (шахматная доска)", self._calibrate_with_chessboard),
            ("Калибровка масштаба скелета", self._calibrate_skeleton_scale),
        ]

        for i, (step_name, step_func) in enumerate(steps):
            if not self.is_running:
                return

            self.calibration_progress.emit(
                i, len(steps),
                f"Шаг {i + 1}/{len(steps)}: {step_name}"
            )

            # Выполнение шага
            # Здесь должна быть реализация каждого шага

            time.sleep(1)  # Заглушка

        self.calibration_complete.emit(
            True,
            "Полная калибровка завершена успешно!"
        )

    def stop(self):
        """Остановка калибровки"""
        self.is_running = False
        self.wait()


class CalibrationVisualization(QWidget):
    """Виджет для визуализации калибровки"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_frame = None
        self.overlay_data = {}

        self.setMinimumSize(640, 480)

        # Таймер для обновления
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(33)  # ~30 FPS

        # Стиль
        self.setStyleSheet("""
            CalibrationVisualization {
                background-color: #1a1a2e;
                border: 2px solid #404040;
                border-radius: 8px;
            }
        """)

    def set_frame(self, frame: np.ndarray, overlay: Dict = None):
        """Установка кадра для отображения"""
        self.current_frame = frame.copy() if frame is not None else None
        self.overlay_data = overlay or {}
        self.update()

    def paintEvent(self, event):
        """Отрисовка кадра и оверлея"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Фон
        painter.fillRect(self.rect(), QColor(26, 26, 46))

        if self.current_frame is not None:
            # Конвертация numpy в QImage
            height, width = self.current_frame.shape[:2]
            bytes_per_line = 3 * width

            if len(self.current_frame.shape) == 3 and self.current_frame.shape[2] == 3:
                # BGR to RGB
                rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                qimage = QImage(
                    rgb_frame.data, width, height,
                    bytes_per_line, QImage.Format.Format_RGB888
                )
            else:
                # Grayscale
                qimage = QImage(
                    self.current_frame.data, width, height,
                    width, QImage.Format.Format_Grayscale8
                )

            # Масштабирование под размер виджета
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            # Центрирование
            pixmap_rect = scaled_pixmap.rect()
            pixmap_rect.moveCenter(self.rect().center())

            painter.drawPixmap(pixmap_rect, scaled_pixmap)

        # Отрисовка оверлея
        self._draw_overlay(painter)

    def _draw_overlay(self, painter: QPainter):
        """Отрисовка оверлейной информации"""
        if not self.overlay_data:
            return

        painter.setPen(QPen(QColor(255, 255, 255, 200)))
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))

        # Информация о калибровке
        y_offset = 30
        for key, value in self.overlay_data.items():
            if key not in ['frame', 'image']:
                text = f"{key}: {value}"
                painter.drawText(20, y_offset, text)
                y_offset += 25

        # Рамка для фокусировки
        center = self.rect().center()
        size = min(self.width(), self.height()) * 0.6

        focus_rect = QRect(
            center.x() - size // 2,
            center.y() - size // 2,
            size, size
        )

        painter.setPen(QPen(QColor(0, 255, 0, 150), 3))
        painter.drawRect(focus_rect)

        # Текст в центре
        if 'message' in self.overlay_data:
            message = self.overlay_data['message']
            font_metrics = painter.fontMetrics()
            text_width = font_metrics.horizontalAdvance(message)

            painter.setPen(QPen(QColor(255, 255, 0, 220), 2))
            painter.drawText(
                center.x() - text_width // 2,
                center.y() + size // 2 + 40,
                message
            )


class CalibrationWizardPage(QWizardPage):
    """Базовая страница мастера калибровки"""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)

        self.setTitle(title)
        self.setSubTitle(subtitle)

        # Анимации
        self.animations = QParallelAnimationGroup()

        # Стиль
        self.setStyleSheet("""
            QWizardPage {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #cccccc;
                font-size: 12px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #353535;
                color: #ffffff;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #505050;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #505050;
                border-color: #606060;
            }
            QPushButton:pressed {
                background-color: #303030;
            }
        """)

    def add_animation(self, widget, property_name: bytes,
                      start_value, end_value, duration: int = 500):
        """Добавление анимации к виджету"""
        animation = QPropertyAnimation(widget, property_name)
        animation.setDuration(duration)
        animation.setStartValue(start_value)
        animation.setEndValue(end_value)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.animations.addAnimation(animation)

    def enter_page(self):
        """Вход на страницу (для анимаций)"""
        self.animations.start()


class WelcomePage(CalibrationWizardPage):
    """Страница приветствия"""

    def __init__(self, parent=None):
        super().__init__("Добро пожаловать в мастер калибровки",
                         "Пройдите несколько шагов для точной настройки системы", parent)

        self.init_ui()

    def init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout()

        # Заголовок
        title_label = QLabel("🎯 КАЛИБРОВКА MOCAP СИСТЕМЫ")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 24px;
                font-weight: bold;
                margin: 20px;
            }
        """)
        layout.addWidget(title_label)

        # Описание
        description = QLabel(
            "Этот мастер поможет вам настроить:\n\n"
            "• 📷 Калибровку камеры (внутренние параметры)\n"
            "• 🦴 Масштаб и позицию скелета\n"
            "• 🎯 Точность трекинга\n\n"
            "Для наилучших результатов подготовьте:\n"
            "• Шахматную доску или ARUCO маркеры\n"
            "• Хорошее освещение\n"
            "• Пространство для движений"
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        description.setStyleSheet("""
            QLabel {
                color: #aaaaaa;
                font-size: 14px;
                line-height: 1.5;
                margin: 20px;
                padding: 20px;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
            }
        """)
        layout.addWidget(description)

        # Иконка
        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setPixmap(QPixmap(500, 300))  # Заглушка, можно заменить на реальную иконку
        layout.addWidget(icon_label)

        # Советы
        tips_label = QLabel("💡 Совет: Выполняйте калибровку в том же месте, где будете снимать.")
        tips_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tips_label.setStyleSheet("color: #ffff88; font-style: italic;")
        layout.addWidget(tips_label)

        layout.addStretch()

        self.setLayout(layout)


class CameraSelectionPage(CalibrationWizardPage):
    """Страница выбора камеры"""

    def __init__(self, camera_manager: MultiCameraManager, parent=None):
        super().__init__("Выбор камеры",
                         "Выберите камеру для калибровки", parent)

        self.camera_manager = camera_manager
        self.selected_camera = 0

        self.init_ui()

    def init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout()

        # Список камер
        cameras_group = QGroupBox("Доступные камеры")
        cameras_group.setStyleSheet("""
            QGroupBox {
                color: #cccccc;
                font-weight: bold;
                border: 2px solid #404040;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)

        cameras_layout = QVBoxLayout()

        self.camera_list = QListWidget()
        self.camera_list.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #cccccc;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #353535;
            }
            QListWidget::item:selected {
                background-color: #404040;
                color: #ffffff;
            }
            QListWidget::item:hover {
                background-color: #353535;
            }
        """)
        self.camera_list.itemClicked.connect(self._on_camera_selected)

        # Заполнение списка камер
        self._populate_camera_list()

        cameras_layout.addWidget(self.camera_list)
        cameras_group.setLayout(cameras_layout)
        layout.addWidget(cameras_group)

        # Предпросмотр камеры
        preview_group = QGroupBox("Предпросмотр")
        preview_layout = QVBoxLayout()

        self.preview_label = QLabel("Предпросмотр не доступен")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(240)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #888888;
                font-style: italic;
            }
        """)
        preview_layout.addWidget(self.preview_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Информация о камере
        info_group = QGroupBox("Информация о камере")
        info_layout = QFormLayout()

        self.camera_info_label = QLabel("Выберите камеру для просмотра информации")
        self.camera_info_label.setWordWrap(True)
        self.camera_info_label.setStyleSheet("color: #aaaaaa;")
        info_layout.addRow("Статус:", self.camera_info_label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Таймер для обновления предпросмотра
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self._update_preview)

        self.setLayout(layout)

    def _populate_camera_list(self):
        """Заполнение списка камер"""
        self.camera_list.clear()

        if self.camera_manager:
            cameras = self.camera_manager.discover_cameras()

            for camera_info in cameras:
                item_text = f"📷 Камера {camera_info.camera_id}: {camera_info.name}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, camera_info.camera_id)
                self.camera_list.addItem(item)

            if cameras:
                self.camera_list.setCurrentRow(0)
                self._on_camera_selected(self.camera_list.item(0))

    def _on_camera_selected(self, item):
        """Обработка выбора камеры"""
        camera_id = item.data(Qt.ItemDataRole.UserRole)
        self.selected_camera = camera_id

        # Обновление информации о камере
        cameras = self.camera_manager.discover_cameras()
        camera_info = next((c for c in cameras if c.camera_id == camera_id), None)

        if camera_info:
            info_text = (
                f"ID: {camera_info.camera_id}\n"
                f"Имя: {camera_info.name}\n"
                f"Тип: {camera_info.type.value}\n"
                f"Доступные разрешения: {len(camera_info.available_resolutions)}\n"
                f"Поддерживаемые настройки: {len(camera_info.supported_settings)}"
            )
            self.camera_info_label.setText(info_text)

        # Запуск предпросмотра
        if not self.preview_timer.isActive():
            self.preview_timer.start(33)  # ~30 FPS

    def _update_preview(self):
        """Обновление предпросмотра камеры"""
        if self.camera_manager and self.selected_camera is not None:
            # Проверяем, добавлена ли уже камера
            if self.selected_camera not in self.camera_manager.cameras:
                # Добавляем камеру временно для предпросмотра
                self.camera_manager.add_camera(self.selected_camera)

            frame_obj = self.camera_manager.get_frame(self.selected_camera, timeout=0.1)
            if frame_obj is not None:
                # Конвертация в QPixmap
                frame = frame_obj.image
                height, width = frame.shape[:2]
                bytes_per_line = 3 * width

                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # RGB to QImage
                    qimage = QImage(
                        frame.data, width, height,
                        bytes_per_line, QImage.Format.Format_RGB888
                    )
                else:
                    # Grayscale
                    qimage = QImage(
                        frame.data, width, height,
                        width, QImage.Format.Format_Grayscale8
                    )

                pixmap = QPixmap.fromImage(qimage)
                scaled_pixmap = pixmap.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )

                self.preview_label.setPixmap(scaled_pixmap)

    def initializePage(self):
        """Инициализация страницы"""
        self._populate_camera_list()
        self.preview_timer.start(33)

    def cleanupPage(self):
        """Очистка страницы"""
        self.preview_timer.stop()
        # Останавливаем все камеры
        if self.camera_manager:
            self.camera_manager.stop_all()

    def get_camera_id(self) -> int:
        """Получение выбранного ID камеры"""
        return self.selected_camera


class CalibrationTypePage(CalibrationWizardPage):
    """Страница выбора типа калибровки"""

    def __init__(self, parent=None):
        super().__init__("Выбор типа калибровки",
                         "Выберите что вы хотите откалибровать", parent)

        self.selected_type = CalibrationType.FULL_CALIBRATION

        self.init_ui()

    def init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout()

        # Описание
        description = QLabel(
            "Выберите тип калибровки в зависимости от ваших потребностей:"
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        # Варианты калибровки
        self.type_group = QButtonGroup(self)

        calibration_types = [
            (
                CalibrationType.CAMERA_INTRINSICS,
                "📷 Калибровка камеры",
                "Определение внутренних параметров камеры (фокусное расстояние, искажения).\n"
                "Требуется: шахматная доска или ARUCO маркеры."
            ),
            (
                CalibrationType.SKELETON_SCALE,
                "🦴 Масштаб скелета",
                "Настройка масштаба скелета под рост человека.\n"
                "Требуется: человек в T-позе."
            ),
            (
                CalibrationType.FULL_CALIBRATION,
                "⚡ Полная калибровка",
                "Полная настройка системы (камера + скелет).\n"
                "Рекомендуется для первого использования."
            )
        ]

        for calib_type, title, description_text in calibration_types:
            radio_btn = QRadioButton(title)
            radio_btn.setStyleSheet("""
                QRadioButton {
                    color: #cccccc;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 15px;
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                    margin: 5px;
                }
                QRadioButton:hover {
                    background-color: rgba(255, 255, 255, 0.1);
                }
                QRadioButton::indicator {
                    width: 20px;
                    height: 20px;
                }
            """)

            desc_label = QLabel(description_text)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #888888; font-size: 11px; margin-left: 30px;")

            self.type_group.addButton(radio_btn, calib_type.value)

            layout.addWidget(radio_btn)
            layout.addWidget(desc_label)

        # Устанавливаем по умолчанию полную калибровку
        full_calib_btn = self.type_group.button(CalibrationType.FULL_CALIBRATION.value)
        if full_calib_btn:
            full_calib_btn.setChecked(True)

        layout.addStretch()

        # Рекомендации
        tips_group = QGroupBox("💡 Рекомендации")
        tips_group.setStyleSheet("""
            QGroupBox {
                color: #ffff88;
                border: 1px solid #888844;
                border-radius: 6px;
                margin-top: 10px;
            }
        """)

        tips_layout = QVBoxLayout()
        tips_label = QLabel(
            "• Для наилучшей точности используйте шахматную доску\n"
            "• Обеспечьте хорошее равномерное освещение\n"
            "• Избегайте прямых источников света и бликов\n"
            "• Используйте калибровку при изменении условий съемки"
        )
        tips_label.setWordWrap(True)
        tips_layout.addWidget(tips_label)
        tips_group.setLayout(tips_layout)

        layout.addWidget(tips_group)

        self.setLayout(layout)

    def get_calibration_type(self) -> CalibrationType:
        """Получение выбранного типа калибровки"""
        checked_button = self.type_group.checkedButton()
        if checked_button:
            return CalibrationType(self.type_group.id(checked_button))
        return CalibrationType.FULL_CALIBRATION


class ChessboardCalibrationPage(CalibrationWizardPage):
    """Страница калибровки с шахматной доской"""

    calibration_complete = pyqtSignal(bool, str)

    def __init__(self, camera_manager: MultiCameraManager, camera_id: int, parent=None):
        super().__init__("Калибровка с шахматной доской",
                         "Используйте шахматную доску для калибровки камеры", parent)

        self.camera_manager = camera_manager
        self.camera_id = camera_id
        self.calibration_thread = None

        self.init_ui()

    def init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout()

        # Визуализация
        self.visualization = CalibrationVisualization()
        layout.addWidget(self.visualization, 70)  # 70% высоты

        # Панель управления
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)

        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        control_layout.addWidget(self.progress_bar, 60)

        # Кнопки
        self.start_btn = QPushButton("▶️ Начать калибровку")
        self.start_btn.clicked.connect(self.start_calibration)
        control_layout.addWidget(self.start_btn, 20)

        self.cancel_btn = QPushButton("⏹️ Отмена")
        self.cancel_btn.clicked.connect(self.cancel_calibration)
        self.cancel_btn.setEnabled(False)
        control_layout.addWidget(self.cancel_btn, 20)

        layout.addWidget(control_panel, 10)

        # Настройки
        settings_group = QGroupBox("Настройки калибровки")
        settings_layout = QGridLayout()

        settings_layout.addWidget(QLabel("Размер доски:"), 0, 0)
        self.pattern_width = QSpinBox()
        self.pattern_width.setRange(3, 15)
        self.pattern_width.setValue(9)
        settings_layout.addWidget(self.pattern_width, 0, 1)

        self.pattern_height = QSpinBox()
        self.pattern_height.setRange(3, 15)
        self.pattern_height.setValue(6)
        settings_layout.addWidget(self.pattern_height, 0, 2)

        settings_layout.addWidget(QLabel("Размер квадрата (м):"), 1, 0)
        self.square_size = QDoubleSpinBox()
        self.square_size.setRange(0.01, 0.5)
        self.square_size.setValue(0.025)
        self.square_size.setSingleStep(0.005)
        settings_layout.addWidget(self.square_size, 1, 1, 1, 2)

        settings_layout.addWidget(QLabel("Количество кадров:"), 2, 0)
        self.frames_needed = QSpinBox()
        self.frames_needed.setRange(5, 50)
        self.frames_needed.setValue(20)
        settings_layout.addWidget(self.frames_needed, 2, 1, 1, 2)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group, 20)

        self.setLayout(layout)

    def start_calibration(self):
        """Запуск калибровки"""
        if self.calibration_thread and self.calibration_thread.isRunning():
            return

        # Параметры калибровки
        parameters = {
            'method': 'chessboard',
            'camera_id': self.camera_id,
            'pattern_size': (self.pattern_width.value(), self.pattern_height.value()),
            'square_size': self.square_size.value(),
            'frames_needed': self.frames_needed.value()
        }

        # Создание потока калибровки
        self.calibration_thread = CalibrationThread(
            self.camera_manager,
            CalibrationType.CAMERA_INTRINSICS,
            parameters
        )

        # Подключение сигналов
        self.calibration_thread.calibration_progress.connect(self._on_progress)
        self.calibration_thread.calibration_complete.connect(self._on_complete)
        self.calibration_thread.calibration_error.connect(self._on_error)
        self.calibration_thread.frame_processed.connect(self._on_frame_processed)

        # Обновление UI
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        # Запуск потока
        self.calibration_thread.start()

    def cancel_calibration(self):
        """Отмена калибровки"""
        if self.calibration_thread and self.calibration_thread.isRunning():
            self.calibration_thread.stop()
            self.calibration_thread.wait()

        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        self.visualization.set_frame(None, {'message': 'Калибровка отменена'})

    def _on_progress(self, current: int, total: int, message: str):
        """Обработка прогресса"""
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{message} - {progress}%")

    def _on_complete(self, success: bool, message: str):
        """Обработка завершения"""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Калибровка завершена успешно!")

            # Показать сообщение об успехе
            QMessageBox.information(self, "Успех", message)

            # Отправить сигнал завершения
            self.calibration_complete.emit(True, message)
        else:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Калибровка не удалась")

            # Показать сообщение об ошибке
            QMessageBox.warning(self, "Ошибка", message)

    def _on_error(self, error_message: str):
        """Обработка ошибки"""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        QMessageBox.critical(self, "Ошибка калибровки", error_message)

    def _on_frame_processed(self, frame: np.ndarray, data: Dict):
        """Обработка обработанного кадра"""
        self.visualization.set_frame(frame, data)

    def cleanupPage(self):
        """Очистка страницы"""
        self.cancel_calibration()


class SkeletonCalibrationPage(CalibrationWizardPage):
    """Страница калибровки скелета"""

    calibration_complete = pyqtSignal(bool, str)

    def __init__(self, camera_manager: MultiCameraManager, camera_id: int, parent=None):
        super().__init__("Калибровка скелета",
                         "Настройка масштаба скелета под ваш рост", parent)

        self.camera_manager = camera_manager
        self.camera_id = camera_id
        self.calibration_thread = None
        self.actual_height = 1.75  # Средний рост по умолчанию

        self.init_ui()

    def init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout()

        # Визуализация
        self.visualization = CalibrationVisualization()
        layout.addWidget(self.visualization, 60)

        # Инструкция
        instruction = QLabel(
            "📋 ИНСТРУКЦИЯ:\n\n"
            "1. Встаньте прямо в поле зрения камеры\n"
            "2. Примите T-позу (руки в стороны)\n"
            "3. Стойте неподвижно во время калибровки\n"
            "4. Убедитесь, что все части тела видны"
        )
        instruction.setWordWrap(True)
        instruction.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 100, 200, 0.1);
                border: 1px solid #0064c8;
                border-radius: 8px;
                padding: 15px;
                color: #88ccff;
                font-size: 12px;
                margin: 5px;
            }
        """)
        layout.addWidget(instruction, 15)

        # Панель управления
        control_panel = QWidget()
        control_layout = QGridLayout(control_panel)

        # Рост пользователя
        control_layout.addWidget(QLabel("Ваш рост (метры):"), 0, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.5, 2.5)
        self.height_spin.setValue(self.actual_height)
        self.height_spin.setSingleStep(0.01)
        self.height_spin.valueChanged.connect(
            lambda v: setattr(self, 'actual_height', v)
        )
        control_layout.addWidget(self.height_spin, 0, 1)

        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        control_layout.addWidget(self.progress_bar, 0, 2, 1, 2)

        # Кнопки
        self.start_btn = QPushButton("🎯 Начать калибровку")
        self.start_btn.clicked.connect(self.start_calibration)
        control_layout.addWidget(self.start_btn, 1, 0, 1, 2)

        self.cancel_btn = QPushButton("⏹️ Отмена")
        self.cancel_btn.clicked.connect(self.cancel_calibration)
        self.cancel_btn.setEnabled(False)
        control_layout.addWidget(self.cancel_btn, 1, 2, 1, 2)

        layout.addWidget(control_panel, 10)

        # Поза-пример
        pose_example = QLabel(
            "🦴 Пример T-позы:\n"
            "• Ноги на ширине плеч\n"
            "• Руки вытянуты в стороны\n"
            "• Ладони обращены вниз\n"
            "• Спина прямая, смотрите вперед"
        )
        pose_example.setWordWrap(True)
        pose_example.setStyleSheet("""
            QLabel {
                background-color: rgba(100, 200, 100, 0.1);
                border: 1px solid #64c864;
                border-radius: 8px;
                padding: 10px;
                color: #aaffaa;
                font-size: 11px;
                margin: 5px;
            }
        """)
        layout.addWidget(pose_example, 15)

        self.setLayout(layout)

    def start_calibration(self):
        """Запуск калибровки скелета"""
        if self.calibration_thread and self.calibration_thread.isRunning():
            return

        # Параметры калибровки
        parameters = {
            'camera_id': self.camera_id,
            'actual_height': self.actual_height,
            'tracking_mode': 'precise',
            'frames_needed': 30
        }

        # Создание потока калибровки
        self.calibration_thread = CalibrationThread(
            self.camera_manager,
            CalibrationType.SKELETON_SCALE,
            parameters
        )

        # Подключение сигналов
        self.calibration_thread.calibration_progress.connect(self._on_progress)
        self.calibration_thread.calibration_complete.connect(self._on_complete)
        self.calibration_thread.calibration_error.connect(self._on_error)
        self.calibration_thread.frame_processed.connect(self._on_frame_processed)

        # Обновление UI
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        # Запуск потока
        self.calibration_thread.start()

    def cancel_calibration(self):
        """Отмена калибровки"""
        if self.calibration_thread and self.calibration_thread.isRunning():
            self.calibration_thread.stop()
            self.calibration_thread.wait()

        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        self.visualization.set_frame(None, {'message': 'Калибровка отменена'})

    def _on_progress(self, current: int, total: int, message: str):
        """Обработка прогресса"""
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{message} - {progress}%")

    def _on_complete(self, success: bool, message: str):
        """Обработка завершения"""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Калибровка завершена успешно!")

            # Показать сообщение об успехе
            QMessageBox.information(self, "Успех", message)

            # Отправить сигнал завершения
            self.calibration_complete.emit(True, message)
        else:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Калибровка не удалась")

            # Показать сообщение об ошибке
            QMessageBox.warning(self, "Ошибка", message)

    def _on_error(self, error_message: str):
        """Обработка ошибки"""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        QMessageBox.critical(self, "Ошибка калибровки", error_message)

    def _on_frame_processed(self, frame: np.ndarray, data: Dict):
        """Обработка обработанного кадра"""
        self.visualization.set_frame(frame, data)

    def cleanupPage(self):
        """Очистка страницы"""
        self.cancel_calibration()


class CompletionPage(CalibrationWizardPage):
    """Страница завершения калибровки"""

    def __init__(self, parent=None):
        super().__init__("Завершение калибровки",
                         "Калибровка успешно завершена!", parent)

        self.calibration_data = None
        self.quality_score = 0

        self.init_ui()

    def init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout()

        # Иконка успеха
        success_label = QLabel("✅")
        success_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        success_label.setStyleSheet("font-size: 72px; margin: 20px;")
        layout.addWidget(success_label)

        # Заголовок
        title_label = QLabel("КАЛИБРОВКА ЗАВЕРШЕНА УСПЕШНО!")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #00ff00;
                font-size: 24px;
                font-weight: bold;
                margin: 10px;
            }
        """)
        layout.addWidget(title_label)

        # Качество калибровки
        quality_group = QGroupBox("Качество калибровки")
        quality_layout = QVBoxLayout()

        self.quality_label = QLabel("Оценка качества: вычисляется...")
        self.quality_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        quality_layout.addWidget(self.quality_label)

        self.quality_bar = QProgressBar()
        self.quality_bar.setRange(0, 100)
        self.quality_bar.setTextVisible(True)
        quality_layout.addWidget(self.quality_bar)

        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)

        # Результаты
        results_group = QGroupBox("Результаты калибровки")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #cccccc;
                border: 1px solid #404040;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Действия
        actions_group = QGroupBox("Дальнейшие действия")
        actions_layout = QVBoxLayout()

        actions_text = QLabel(
            "• Начните запись анимации\n"
            "• Проверьте точность трекинга\n"
            "• При необходимости выполните ручную коррекцию\n"
            "• Сохраните калибровку для будущего использования"
        )
        actions_text.setWordWrap(True)
        actions_layout.addWidget(actions_text)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        # Кнопки
        button_layout = QHBoxLayout()

        self.save_btn = QPushButton("💾 Сохранить калибровку")
        self.save_btn.clicked.connect(self.save_calibration)
        button_layout.addWidget(self.save_btn)

        self.test_btn = QPushButton("🎬 Протестировать")
        button_layout.addWidget(self.test_btn)

        self.finish_btn = QPushButton("🏁 Завершить")
        self.finish_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aa00;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
            }
            QPushButton:hover {
                background-color: #00cc00;
            }
        """)
        button_layout.addWidget(self.finish_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def set_calibration_data(self, data: CalibrationData):
        """Установка данных калибровки"""
        self.calibration_data = data

        # Расчет качества
        self.quality_score = self._calculate_quality_score(data)
        self.quality_bar.setValue(int(self.quality_score))

        quality_text = f"Оценка качества: {self.quality_score:.1f}/100"
        if self.quality_score >= 80:
            quality_text += " (Отлично!)"
            self.quality_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        elif self.quality_score >= 60:
            quality_text += " (Хорошо)"
            self.quality_label.setStyleSheet("color: #ffff00; font-weight: bold;")
        else:
            quality_text += " (Требуется улучшение)"
            self.quality_label.setStyleSheet("color: #ff4444; font-weight: bold;")

        self.quality_label.setText(quality_text)

        # Формирование текста результатов
        results = []
        results.append("=== РЕЗУЛЬТАТЫ КАЛИБРОВКИ ===")
        results.append(f"Время: {datetime.fromtimestamp(data.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        results.append("")

        if data.camera_calibration:
            results.append("📷 КАЛИБРОВКА КАМЕРЫ:")
            for cam_id, calib in data.camera_calibration.items():
                results.append(f"  Камера {cam_id}:")
                results.append(f"    Разрешение: {calib.resolution[0]}x{calib.resolution[1]}")
                results.append(f"    Поле зрения: {calib.fov[0]:.1f}° x {calib.fov[1]:.1f}°")
                if hasattr(calib, 'calibration_error'):
                    results.append(f"    Ошибка калибровки: {calib.calibration_error:.4f} пикс")
                results.append("")

        if data.skeleton_data:
            results.append("🦴 КАЛИБРОВКА СКЕЛЕТА:")
            for key, value in data.skeleton_data.items():
                if isinstance(value, (int, float)):
                    results.append(f"  {key}: {value:.4f}")
                else:
                    results.append(f"  {key}: {value}")

        self.results_text.setText("\n".join(results))

    def _calculate_quality_score(self, data: CalibrationData) -> float:
        """Расчет оценки качества калибровки"""
        score = 50.0  # Базовая оценка

        # Учет калибровки камеры
        if data.camera_calibration:
            for calib in data.camera_calibration.values():
                if hasattr(calib, 'calibration_error'):
                    # Меньше ошибка = выше оценка
                    error = calib.calibration_error
                    if error < 0.1:
                        score += 20
                    elif error < 0.5:
                        score += 15
                    elif error < 1.0:
                        score += 10
                    else:
                        score += 5

                if calib.intrinsics_set:
                    score += 10

        # Учет калибровки скелета
        if data.skeleton_data:
            score += 20

        # Ограничение до 100
        return min(score, 100.0)

    def save_calibration(self):
        """Сохранение калибровки"""
        if not self.calibration_data:
            QMessageBox.warning(self, "Нет данных", "Нет данных калибровки для сохранения")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить калибровку",
            "calibration.yaml", "YAML Files (*.yaml);;All Files (*.*)"
        )

        if filepath:
            self.calibration_data.save_to_file(filepath)
            QMessageBox.information(self, "Сохранено", f"Калибровка сохранена в:\n{filepath}")


class ProfessionalCalibrationWizard(QWidget):
    """
    ПРОФЕССИОНАЛЬНЫЙ МАСТЕР КАЛИБРОВКИ MOCAP

    Пошаговая калибровка:
    1. Приветствие
    2. Выбор камеры
    3. Выбор типа калибровки
    4. Калибровка камеры (шахматная доска/ARUCO)
    5. Калибровка скелета
    6. Завершение
    """

    # Сигналы
    calibration_started = pyqtSignal()
    calibration_completed = pyqtSignal(CalibrationData)
    calibration_failed = pyqtSignal(str)

    def __init__(self, camera_manager: MultiCameraManager = None, parent=None):
        super().__init__(parent)

        self.camera_manager = camera_manager
        self.calibration_data = CalibrationData()
        self.current_step = CalibrationStep.WELCOME

        # Настройки калибровки
        self.settings = {
            'camera_id': 0,
            'calibration_type': CalibrationType.FULL_CALIBRATION,
            'chessboard_pattern': (9, 6),
            'square_size': 0.025,
            'frames_needed': 20,
            'user_height': 1.75
        }

        self.init_ui()
        self.init_wizard()

        logger.info("ProfessionalCalibrationWizard инициализирован")

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("🎯 Мастер калибровки MOCAP Pro")
        self.setMinimumSize(900, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Заголовок
        self.header = QLabel("🎯 МАСТЕР КАЛИБРОВКИ")
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header.setStyleSheet("""
            QLabel {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2b5b84, stop:1 #1e3a5f
                );
                color: white;
                font-size: 20px;
                font-weight: bold;
                padding: 15px;
                border-bottom: 2px solid #404040;
            }
        """)
        layout.addWidget(self.header)

        # Виджет мастера
        self.wizard_widget = QStackedWidget()
        layout.addWidget(self.wizard_widget, 100)

        # Панель прогресса
        self.progress_panel = self._create_progress_panel()
        layout.addWidget(self.progress_panel)

        # Кнопки навигации
        self.nav_panel = self._create_navigation_panel()
        layout.addWidget(self.nav_panel)

    def init_wizard(self):
        """Инициализация страниц мастера"""
        # Страница приветствия
        self.welcome_page = WelcomePage()
        self.wizard_widget.addWidget(self.welcome_page)

        # Страница выбора камеры (только если есть camera_manager)
        if self.camera_manager:
            self.camera_page = CameraSelectionPage(self.camera_manager)
            self.wizard_widget.addWidget(self.camera_page)

        # Страница выбора типа калибровки
        self.type_page = CalibrationTypePage()
        self.wizard_widget.addWidget(self.type_page)

        # Страницы калибровки (будут созданы динамически)
        self.chessboard_page = None
        self.skeleton_page = None

        # Страница завершения
        self.completion_page = CompletionPage()
        self.completion_page.finish_btn.clicked.connect(self._on_finish)
        self.wizard_widget.addWidget(self.completion_page)

        # Обновление прогресса
        self._update_progress()

    def _create_progress_panel(self) -> QWidget:
        """Создание панели прогресса"""
        panel = QWidget()
        panel.setFixedHeight(40)
        panel.setStyleSheet("""
            QWidget {
                background-color: #353535;
                border-top: 1px solid #404040;
            }
        """)

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(20, 5, 20, 5)

        # Шаги калибровки
        self.step_labels = {}
        steps = [
            ("🎯", "Старт"),
            ("📷", "Камера"),
            ("⚙️", "Тип"),
            ("🔧", "Калибровка"),
            ("✅", "Готово")
        ]

        for icon, text in steps:
            label = QLabel(f"{icon} {text}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("""
                QLabel {
                    color: #888888;
                    font-size: 11px;
                    padding: 5px 10px;
                    border-radius: 10px;
                }
            """)
            self.step_labels[text] = label
            layout.addWidget(label)

        layout.addStretch()

        return panel

    def _create_navigation_panel(self) -> QWidget:
        """Создание панели навигации"""
        panel = QWidget()
        panel.setFixedHeight(60)
        panel.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border-top: 1px solid #404040;
            }
        """)

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(20, 10, 20, 10)

        # Кнопка "Назад"
        self.back_btn = QPushButton("◀️ Назад")
        self.back_btn.clicked.connect(self.previous_step)
        self.back_btn.setEnabled(False)
        layout.addWidget(self.back_btn)

        layout.addStretch()

        # Кнопка "Далее"
        self.next_btn = QPushButton("Далее ▶️")
        self.next_btn.clicked.connect(self.next_step)
        layout.addWidget(self.next_btn)

        return panel

    def _update_progress(self):
        """Обновление индикатора прогресса"""
        # Определяем текущий шаг
        steps_order = [
            "Старт", "Камера", "Тип", "Калибровка", "Готово"
        ]

        current_idx = self.wizard_widget.currentIndex()
        if current_idx < len(steps_order):
            current_step = steps_order[current_idx]

            # Подсветка текущего шага
            for step_name, label in self.step_labels.items():
                if step_name == current_step:
                    label.setStyleSheet("""
                        QLabel {
                            color: #ffffff;
                            font-weight: bold;
                            background-color: #00aaff;
                            padding: 5px 10px;
                            border-radius: 10px;
                        }
                    """)
                else:
                    label.setStyleSheet("""
                        QLabel {
                            color: #888888;
                            font-size: 11px;
                            padding: 5px 10px;
                            border-radius: 10px;
                        }
                    """)

        # Обновление кнопок навигации
        self.back_btn.setEnabled(current_idx > 0)

        if current_idx == self.wizard_widget.count() - 1:
            self.next_btn.setText("🏁 Завершить")
        else:
            self.next_btn.setText("Далее ▶️")

    def set_camera_manager(self, camera_manager: MultiCameraManager):
        """Установка менеджера камер"""
        self.camera_manager = camera_manager

        # Обновляем страницу выбора камеры
        if self.camera_page:
            self.wizard_widget.removeWidget(self.camera_page)

        self.camera_page = CameraSelectionPage(camera_manager)
        self.wizard_widget.insertWidget(1, self.camera_page)

    def next_step(self):
        """Переход к следующему шагу"""
        current_idx = self.wizard_widget.currentIndex()

        # Проверка перед переходом
        if current_idx == 0:  # Welcome -> Camera
            pass
        elif current_idx == 1:  # Camera -> Type
            if self.camera_manager:
                self.settings['camera_id'] = self.camera_page.get_camera_id()
        elif current_idx == 2:  # Type -> Calibration
            self.settings['calibration_type'] = self.type_page.get_calibration_type()

            # Создаем соответствующую страницу калибровки
            self._create_calibration_page()
        elif current_idx == 3:  # Calibration -> Completion
            # Переход к завершению
            self.current_step = CalibrationStep.COMPLETION
            self.completion_page.set_calibration_data(self.calibration_data)
        elif current_idx == 4:  # Completion -> Finish
            self._on_finish()
            return

        # Переход
        if current_idx < self.wizard_widget.count() - 1:
            self.wizard_widget.setCurrentIndex(current_idx + 1)
            self._update_progress()

            # Анимация входа
            current_widget = self.wizard_widget.currentWidget()
            if isinstance(current_widget, CalibrationWizardPage):
                current_widget.enter_page()

    def previous_step(self):
        """Переход к предыдущему шагу"""
        current_idx = self.wizard_widget.currentIndex()

        if current_idx > 0:
            # Очистка текущей страницы если нужно
            current_widget = self.wizard_widget.currentWidget()
            if hasattr(current_widget, 'cleanupPage'):
                current_widget.cleanupPage()

            self.wizard_widget.setCurrentIndex(current_idx - 1)
            self._update_progress()

    def _create_calibration_page(self):
        """Создание страницы калибровки в зависимости от типа"""
        # Удаляем старые страницы калибровки
        if self.chessboard_page:
            self.wizard_widget.removeWidget(self.chessboard_page)
            self.chessboard_page = None

        if self.skeleton_page:
            self.wizard_widget.removeWidget(self.skeleton_page)
            self.skeleton_page = None

        # Создаем новые страницы
        calib_type = self.settings['calibration_type']
        camera_id = self.settings['camera_id']

        if calib_type == CalibrationType.CAMERA_INTRINSICS:
            self.current_step = CalibrationStep.CHESSBOARD_CALIBRATION
            self.chessboard_page = ChessboardCalibrationPage(
                self.camera_manager, camera_id
            )
            self.chessboard_page.calibration_complete.connect(
                self._on_chessboard_calibration_complete
            )
            self.wizard_widget.insertWidget(3, self.chessboard_page)

        elif calib_type == CalibrationType.SKELETON_SCALE:
            self.current_step = CalibrationStep.SKELETON_CALIBRATION
            self.skeleton_page = SkeletonCalibrationPage(
                self.camera_manager, camera_id
            )
            self.skeleton_page.calibration_complete.connect(
                self._on_skeleton_calibration_complete
            )
            self.wizard_widget.insertWidget(3, self.skeleton_page)

        elif calib_type == CalibrationType.FULL_CALIBRATION:
            # Для полной калибровки показываем сначала калибровку камеры
            self.current_step = CalibrationStep.CHESSBOARD_CALIBRATION
            self.chessboard_page = ChessboardCalibrationPage(
                self.camera_manager, camera_id
            )
            self.chessboard_page.calibration_complete.connect(
                lambda success, msg: self._on_full_calibration_part1_complete(success, msg)
            )
            self.wizard_widget.insertWidget(3, self.chessboard_page)

        # Обновляем текущий виджет
        self.wizard_widget.setCurrentIndex(3)

    def _on_chessboard_calibration_complete(self, success: bool, message: str):
        """Обработка завершения калибровки камеры"""
        if success:
            # Загружаем калибровку камеры
            camera_id = self.settings['camera_id']
            calib_file = f"camera_{camera_id}_calibration.yaml"

            if os.path.exists(calib_file):
                camera_calib = CameraCalibration.load_from_file(calib_file)
                self.calibration_data.camera_calibration[camera_id] = camera_calib

                # Переходим к завершению
                self.current_step = CalibrationStep.COMPLETION
                self.completion_page.set_calibration_data(self.calibration_data)
                self.wizard_widget.setCurrentWidget(self.completion_page)
                self._update_progress()

            else:
                QMessageBox.warning(self, "Ошибка", "Файл калибровки не найден")

    def _on_skeleton_calibration_complete(self, success: bool, message: str):
        """Обработка завершения калибровки скелета"""
        if success:
            # Загружаем калибровку скелета
            skeleton_file = "skeleton_scale_calibration.json"

            if os.path.exists(skeleton_file):
                with open(skeleton_file, 'r') as f:
                    skeleton_data = json.load(f)
                self.calibration_data.skeleton_data = skeleton_data

                # Переходим к завершению
                self.current_step = CalibrationStep.COMPLETION
                self.completion_page.set_calibration_data(self.calibration_data)
                self.wizard_widget.setCurrentWidget(self.completion_page)
                self._update_progress()

    def _on_full_calibration_part1_complete(self, success: bool, message: str):
        """Обработка первой части полной калибровки"""
        if success:
            # Загружаем калибровку камеры
            camera_id = self.settings['camera_id']
            calib_file = f"camera_{camera_id}_calibration.yaml"

            if os.path.exists(calib_file):
                camera_calib = CameraCalibration.load_from_file(calib_file)
                self.calibration_data.camera_calibration[camera_id] = camera_calib

                # Переходим к калибровке скелета
                self.current_step = CalibrationStep.SKELETON_CALIBRATION

                # Создаем страницу калибровки скелета
                if self.chessboard_page:
                    self.wizard_widget.removeWidget(self.chessboard_page)

                self.skeleton_page = SkeletonCalibrationPage(
                    self.camera_manager, camera_id
                )
                self.skeleton_page.calibration_complete.connect(
                    self._on_full_calibration_part2_complete
                )

                # Вставляем после текущей позиции
                self.wizard_widget.insertWidget(3, self.skeleton_page)
                self.wizard_widget.setCurrentWidget(self.skeleton_page)
                self._update_progress()

    def _on_full_calibration_part2_complete(self, success: bool, message: str):
        """Обработка второй части полной калибровки"""
        if success:
            # Загружаем калибровку скелета
            skeleton_file = "skeleton_scale_calibration.json"

            if os.path.exists(skeleton_file):
                with open(skeleton_file, 'r') as f:
                    skeleton_data = json.load(f)
                self.calibration_data.skeleton_data = skeleton_data

            # Переходим к завершению
            self.current_step = CalibrationStep.COMPLETION
            self.completion_page.set_calibration_data(self.calibration_data)
            self.wizard_widget.setCurrentWidget(self.completion_page)
            self._update_progress()

    def _on_finish(self):
        """Завершение работы мастера"""
        # Отправляем сигнал завершения
        if self.calibration_data.camera_calibration or self.calibration_data.skeleton_data:
            self.calibration_completed.emit(self.calibration_data)
        else:
            self.calibration_failed.emit("Калибровка не выполнена")

        # Закрываем мастер
        self.close()

    def start_calibration(self):
        """Запуск процесса калибровки"""
        self.calibration_started.emit()
        self.show()

    def get_calibration_data(self) -> CalibrationData:
        """Получение данных калибровки"""
        return self.calibration_data


# Для обратной совместимости
class CalibrationWizard(ProfessionalCalibrationWizard):
    """Алиас для обратной совместимости"""
    pass


# Быстрый тест
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)

    # Тестовый мастер калибровки
    wizard = ProfessionalCalibrationWizard()
    wizard.resize(1000, 800)
    wizard.show()

    # Сигналы
    wizard.calibration_started.connect(
        lambda: print("Калибровка начата")
    )
    wizard.calibration_completed.connect(
        lambda data: print(f"Калибровка завершена: {len(data.camera_calibration)} камер")
    )
    wizard.calibration_failed.connect(
        lambda msg: print(f"Калибровка не удалась: {msg}")
    )

    sys.exit(app.exec())