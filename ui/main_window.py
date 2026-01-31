"""
ПРОФЕССИОНАЛЬНОЕ ГЛАВНОЕ ОКНО MOCAP PRO
Модульный дизайн, поддержка плагинов, темная тема, расширенные функции
"""

import sys
import logging
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QStatusBar, QMenuBar, QMenu, QFileDialog, QMessageBox,
    QLabel, QDockWidget, QToolBar, QTabWidget, QApplication,
    QToolButton, QPushButton, QGroupBox, QStyleFactory,
    QDialog, QProgressDialog, QSystemTrayIcon, QStyle
)
from PyQt6.QtGui import (
    QAction, QIcon, QPixmap, QFont, QKeySequence, QPalette,
    QColor, QPainter, QPen, QBrush, QFontMetrics
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QSize, QThread, QObject,
    QEvent, QSettings, QPoint, QRect, QPropertyAnimation
)
import json
import yaml
import os
from datetime import datetime

from ui.video_panel import VideoPanel
from ui.controls_panel import ControlsPanel
from ui.timeline_editor import TimelineEditor
from ui.skeleton_editor import SkeletonEditor
from ui.calibration_wizard import CalibrationWizard

from core.skeleton_tracker import SkeletonTracker
from core.animation_recorder import ProfessionalAnimationRecorder
from core.camera_manager import MultiCameraManager
from export.bvh_exporter import BVHExporter
from export.blender_bridge import BlenderBridge
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)


class CustomTitleBar(QWidget):
    """Кастомная панель заголовка для Windows 11 стиля"""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        self.setFixedHeight(40)
        self.setStyleSheet("""
            CustomTitleBar {
                background-color: #2b2b2b;
                border-bottom: 1px solid #404040;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
                padding-left: 12px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Иконка приложения
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(24, 24)
        layout.addWidget(self.icon_label)

        # Название приложения
        self.title_label = QLabel("Motion Capture Pro")
        self.title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(self.title_label)

        layout.addStretch()

        # Кнопки управления окном
        self.minimize_btn = self._create_button("━", "Минимизировать")
        self.maximize_btn = self._create_button("□", "Развернуть")
        self.close_btn = self._create_button("✕", "Закрыть", True)

        self.minimize_btn.clicked.connect(parent.showMinimized)
        self.maximize_btn.clicked.connect(self._toggle_maximize)
        self.close_btn.clicked.connect(parent.close)

        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.maximize_btn)
        layout.addWidget(self.close_btn)

    def _create_button(self, text, tooltip, is_close=False):
        """Создание кастомной кнопки"""
        btn = QPushButton(text)
        btn.setFixedSize(46, 40)
        btn.setToolTip(tooltip)

        if is_close:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #ffffff;
                    font-size: 14px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #e81123;
                }
                QPushButton:pressed {
                    background-color: #f1707a;
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #ffffff;
                    font-size: 14px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #404040;
                }
                QPushButton:pressed {
                    background-color: #505050;
                }
            """)

        return btn

    def _toggle_maximize(self):
        """Переключение режима максимизации"""
        if self.parent_window.isMaximized():
            self.parent_window.showNormal()
            self.maximize_btn.setText("□")
        else:
            self.parent_window.showMaximized()
            self.maximize_btn.setText("❐")


class StatusIndicator(QLabel):
    """Индикатор статуса с анимацией"""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.status = "idle"  # idle, recording, tracking, error
        self.blink_animation = QPropertyAnimation(self, b"color")

    def set_status(self, status, blink=False):
        """Установка статуса с визуальными эффектами"""
        self.status = status

        colors = {
            "idle": "#808080",
            "recording": "#ff4444",
            "tracking": "#44ff44",
            "calibrating": "#ffff44",
            "error": "#ff4444"
        }

        text = {
            "idle": "● Готов",
            "recording": "● Запись",
            "tracking": "● Трекинг",
            "calibrating": "● Калибровка",
            "error": "● Ошибка"
        }

        self.setText(text.get(status, "● Неизвестно"))

        color = QColor(colors.get(status, "#808080"))
        self.setStyleSheet(f"color: {color.name()}; font-weight: bold;")

        if blink:
            self.start_blink(color)

    def start_blink(self, color):
        """Запуск мигающей анимации"""
        self.blink_animation.stop()
        self.blink_animation.setDuration(500)
        self.blink_animation.setLoopCount(-1)
        self.blink_animation.setStartValue(color)
        self.blink_animation.setEndValue(QColor(color).lighter(150))
        self.blink_animation.start()


class PerformanceMonitor(QWidget):
    """Виджет мониторинга производительности"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(80)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        # FPS
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #44ff44; font-weight: bold;")
        layout.addWidget(self.fps_label)

        # CPU
        self.cpu_label = QLabel("CPU: --%")
        self.cpu_label.setStyleSheet("color: #4488ff; font-weight: bold;")
        layout.addWidget(self.cpu_label)

        # Память
        self.memory_label = QLabel("RAM: -- MB")
        self.memory_label.setStyleSheet("color: #ff8844; font-weight: bold;")
        layout.addWidget(self.memory_label)

        # Задержка
        self.latency_label = QLabel("Latency: -- ms")
        self.latency_label.setStyleSheet("color: #ff44ff; font-weight: bold;")
        layout.addWidget(self.latency_label)

        layout.addStretch()

    def update_metrics(self, fps, cpu, memory, latency):
        """Обновление метрик"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.cpu_label.setText(f"CPU: {cpu:.1f}%")
        self.memory_label.setText(f"RAM: {memory:.1f} MB")
        self.latency_label.setText(f"Latency: {latency:.1f} ms")


class ProfessionalMainWindow(QMainWindow):
    """Профессиональное главное окно Mocap Pro"""

    # Сигналы
    aboutToClose = pyqtSignal()
    trackingStarted = pyqtSignal()
    trackingStopped = pyqtSignal()
    recordingStarted = pyqtSignal()
    recordingStopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        print("🚀 Инициализация ProfessionalMainWindow...")

        # Флаги состояния
        self.is_tracking = False
        self.is_recording = False
        self.is_calibrated = False
        self.is_paused = False
        self.test_mode = False

        # Инициализация компонентов
        self._init_components()
        self._init_ui()
        self._init_signals()

        print("✅ ProfessionalMainWindow инициализирован")

    def _init_components(self):
        """Инициализация компонентов"""
        print("🔧 Инициализация компонентов...")

        try:
            # 1. Камера
            from core.camera_manager import MultiCameraManager
            self.camera_manager = MultiCameraManager()
            print(f"✅ Камер найдено: {len(self.camera_manager.list_cameras())}")
        except Exception as e:
            print(f"⚠️ Ошибка инициализации камеры: {e}")
            self.camera_manager = None

        # 2. UI компоненты
        try:
            from ui.video_panel import VideoPanel
            from ui.controls_panel import ControlsPanel

            self.video_panel = VideoPanel()
            self.controls_panel = ControlsPanel()
            print("✅ UI компоненты созданы")
        except Exception as e:
            print(f"⚠️ Ошибка создания UI компонентов: {e}")
            self.video_panel = None
            self.controls_panel = None

        # 3. Настройки
        self.settings = None
        try:
            from PyQt6.QtCore import QSettings
            self.settings = QSettings("MocapPro", "MotionCapturePro")
            self._load_settings()
            print("✅ Настройки загружены")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки настроек: {e}")

    def _init_ui(self):
        """Инициализация интерфейса"""
        print("🖥️ Инициализация интерфейса...")

        # Установка размеров окна
        self.setWindowTitle("Motion Capture Pro")
        self.setGeometry(100, 100, 1280, 720)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 1. Видео панель
        if self.video_panel:
            main_layout.addWidget(self.video_panel, stretch=3)

        # 2. Панель управления
        if self.controls_panel:
            main_layout.addWidget(self.controls_panel, stretch=1)

        # 3. Кнопки управления
        button_layout = QHBoxLayout()

        self.start_tracking_btn = QPushButton("▶️ Запустить трекинг")
        self.start_tracking_btn.clicked.connect(self.start_tracking)

        self.stop_tracking_btn = QPushButton("⏹️ Остановить трекинг")
        self.stop_tracking_btn.clicked.connect(self.stop_tracking)
        self.stop_tracking_btn.setEnabled(False)

        self.start_recording_btn = QPushButton("🔴 Начать запись")
        self.start_recording_btn.clicked.connect(self.start_recording)

        self.stop_recording_btn = QPushButton("⏹️ Остановить запись")
        self.stop_recording_btn.clicked.connect(self.stop_recording)
        self.stop_recording_btn.setEnabled(False)

        button_layout.addWidget(self.start_tracking_btn)
        button_layout.addWidget(self.stop_tracking_btn)
        button_layout.addWidget(self.start_recording_btn)
        button_layout.addWidget(self.stop_recording_btn)

        main_layout.addLayout(button_layout)

        # 4. Статус бар
        self.statusBar().showMessage("Готов к работе")

        # Применение темы
        self._apply_theme()

        # Запуск обновления видео
        self._start_video_update()

    def _init_signals(self):
        """Инициализация сигналов"""
        print("🔌 Инициализация сигналов...")

        # Подключение сигналов кнопок
        if self.controls_panel:
            # Если в ControlsPanel есть сигналы, подключаем их
            pass

    def _load_settings(self):
        """Загрузка настроек"""
        if self.settings:
            defaults = {
                "ui/theme": "dark",
                "tracking/mode": "precise",
                "tracking/kalman": True,
                "tracking/smoothing": True
            }

            for key, value in defaults.items():
                if not self.settings.contains(key):
                    self.settings.setValue(key, value)

    def _apply_theme(self):
        """Применение темы"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #404040;
                color: white;
                border: 1px solid #505050;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #303030;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #777777;
            }
            QStatusBar {
                background-color: #2b2b2b;
                color: white;
            }
        """)

    def _start_video_update(self):
        """Запуск обновления видео"""
        print("📹 Запуск обновления видео...")

        # Проверяем есть ли камера и добавляем если нет
        if self.camera_manager:
            # Получаем список добавленных камер
            num_cameras = len(self.camera_manager.list_cameras())
            print(f"📷 Добавлено камер в систему: {num_cameras}")

            # Если камер нет в системе, пробуем найти и добавить
            if num_cameras == 0:
                print("🔄 Ищу доступные камеры...")

                try:
                    # Сначала пробуем камеру с индексом 0
                    if self.camera_manager.add_camera(0, (640, 480), 30):
                        print("✅ Камера 0 добавлена (640x480 @ 30FPS)")
                    else:
                        # Пробуем камеру с индексом 1
                        print("⚠️ Камера 0 не найдена, пробую камеру 1...")
                        if self.camera_manager.add_camera(1, (640, 480), 30):
                            print("✅ Камера 1 добавлена (640x480 @ 30FPS)")
                        else:
                            # Показываем тестовый режим
                            print("❌ Не удалось найти камеру. Включаю тестовый режим.")
                            self.test_mode = True
                except Exception as e:
                    print(f"⚠️ Ошибка добавления камеры: {e}")
                    self.test_mode = True
            else:
                print("✅ Камеры готовы к работе")

            # Получаем статистику
            try:
                stats = self.camera_manager.get_all_stats()
                if stats and 'cameras' in stats:
                    for cam_id, cam_stats in stats['cameras'].items():
                        print(f"📊 Камера {cam_id}: {cam_stats.get('avg_fps', 0):.1f} FPS")
            except:
                pass
        else:
            print("❌ CameraManager не инициализирован")
            self.test_mode = True

        # Таймер для обновления видео
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self._update_video_frame)
        self.video_timer.start(33)  # 30 FPS
        print("✅ Таймер видео запущен")

    def _update_video_frame(self):
        """Обновление кадра видео"""
        try:
            if not self.video_panel:
                print("❌ VideoPanel не доступен")
                return

            # Режим тестирования (без камеры)
            if hasattr(self, 'test_mode') and self.test_mode:
                self._show_test_frame()
                return

            # Режим с камерой
            if not self.camera_manager:
                print("❌ CameraManager не доступен")
                self._show_test_frame()
                return

            # Получаем кадр с камеры
            camera_frame = self.camera_manager.get_frame(0, timeout=0.1)

            if camera_frame:
                frame = camera_frame.image

                # Дополнительная отладка (можно убрать позже)
                if hasattr(self, 'debug_counter'):
                    self.debug_counter += 1
                    if self.debug_counter % 30 == 0:  # Каждые 30 кадров
                        print(f"📹 Кадр {self.debug_counter}: {frame.shape}")
                else:
                    self.debug_counter = 1

                # Обновляем видео-панель
                self.video_panel.update_frame(frame)

                # Обновляем FPS в статусе
                if hasattr(camera_frame, 'fps') and camera_frame.fps > 0:
                    self.statusBar().showMessage(
                        f"Камера: {camera_frame.fps:.1f} FPS | Разрешение: {frame.shape[1]}x{frame.shape[0]}")
            else:
                # Если кадр не получен, переключаемся в тестовый режим
                print("⚠️ Кадр не получен, включаю тестовый режим")
                self.test_mode = True
                self._show_test_frame()

        except Exception as e:
            print(f"⚠️ Ошибка обновления видео: {e}")
            self.test_mode = True
            import traceback
            traceback.print_exc()

    def _show_test_frame(self):
        """Показать тестовый кадр"""
        try:
            import cv2
            import numpy as np
            import time

            # Создаем тестовое изображение
            height, width = 480, 640
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Градиентный фон
            for i in range(height):
                color = int((np.sin(time.time() * 2 + i / height * np.pi) + 1) * 127.5)
                frame[i, :] = [color, 255 - color, color // 2]

            # Текст
            cv2.putText(frame, "MOCAP PRO - ТЕСТОВЫЙ РЕЖИМ",
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

            cv2.putText(frame, "Камера не обнаружена",
                        (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)

            cv2.putText(frame, "FPS: 30.0 | Разрешение: 640x480",
                        (160, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 255), 2)

            # Анимированный круг
            circle_radius = int(30 + 20 * np.sin(time.time() * 3))
            cv2.circle(frame, (width // 2, height // 2 + 80),
                       circle_radius, (255, 0, 0), -1)

            cv2.putText(frame, "LIVE",
                        (width // 2 - 25, height // 2 + 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            # Обновляем видео-панель
            self.video_panel.update_frame(frame)

            # Обновляем статус
            self.statusBar().showMessage("Тестовый режим | Подключите камеру")

        except Exception as e:
            print(f"⚠️ Ошибка создания тестового кадра: {e}")
    def start_tracking(self):
        """Запуск трекинга"""
        if not self.is_tracking:
            self.is_tracking = True
            self.trackingStarted.emit()
            self.statusBar().showMessage("Трекинг запущен")

            self.start_tracking_btn.setEnabled(False)
            self.stop_tracking_btn.setEnabled(True)

            print("✅ Трекинг запущен")

    def stop_tracking(self):
        """Остановка трекинга"""
        if self.is_tracking:
            self.is_tracking = False
            self.trackingStopped.emit()
            self.statusBar().showMessage("Трекинг остановлен")

            self.start_tracking_btn.setEnabled(True)
            self.stop_tracking_btn.setEnabled(False)

            print("✅ Трекинг остановлен")

    def start_recording(self):
        """Начало записи"""
        if not self.is_recording:
            self.is_recording = True
            self.recordingStarted.emit()
            self.statusBar().showMessage("Запись начата")

            self.start_recording_btn.setEnabled(False)
            self.stop_recording_btn.setEnabled(True)

            print("✅ Запись начата")

    def stop_recording(self):
        """Остановка записи"""
        if self.is_recording:
            self.is_recording = False
            self.recordingStopped.emit()
            self.statusBar().showMessage("Запись остановлена")

            self.start_recording_btn.setEnabled(True)
            self.stop_recording_btn.setEnabled(False)

            print("✅ Запись остановлена")

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        print("🚪 Закрытие окна...")

        # Остановка трекинга если активен
        if self.is_tracking:
            self.stop_tracking()

        # Остановка записи если активна
        if self.is_recording:
            self.stop_recording()

        # Остановка таймера видео
        if hasattr(self, 'video_timer'):
            self.video_timer.stop()

        # Освобождение камеры
        if self.camera_manager:
            try:
                self.camera_manager.release()
            except:
                pass

        # Сохранение настроек
        if self.settings:
            try:
                self.settings.sync()
            except:
                pass

        self.aboutToClose.emit()
        event.accept()
        print("✅ Окно закрыто")

    # Методы-заглушки для будущей реализации
    def init_menu(self):
        pass

    def init_toolbars(self):
        pass

    def init_docks(self):
        pass

    def connect_signals(self):
        pass

    def apply_theme(self):
        pass

    def init_tray_icon(self):
        pass

    def new_project(self):
        pass

    def save_project(self):
        pass

    def save_project_as(self):
        pass

    def open_project(self):
        pass

    def calibrate_camera(self):
        pass

    def calibrate_skeleton(self):
        pass

    def toggle_pause(self):
        pass

    def set_tracking_mode(self, mode):
        pass

    def export_animation(self, format_type):
        pass

    def send_to_blender(self):
        pass

    def show_about(self):
        pass

    def open_documentation(self):
        pass

    def open_tutorials(self):
        pass

    def open_camera_settings(self):
        pass

    def open_skeleton_settings(self):
        pass

    def open_tracking_settings(self):
        pass

    def open_export_settings(self):
        pass

    def reset_settings(self):
        pass


# Для обратной совместимости
MainWindow = ProfessionalMainWindow

if __name__ == "__main__":
    # Тестовый запуск
    import sys

    app = QApplication(sys.argv)
    window = ProfessionalMainWindow()
    window.show()
    sys.exit(app.exec())