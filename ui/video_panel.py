"""
ПРОФЕССИОНАЛЬНАЯ ВИДЕО ПАНЕЛЬ ДЛЯ MOCAP
3D просмотр, наложение сеток, инструменты анализа, multiple viewports
"""

import sys
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolBar,
    QToolButton, QComboBox, QSlider, QSplitter, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsLineItem, QMenu, QRubberBand,
    QStyle, QSizePolicy, QGroupBox, QCheckBox, QSpinBox
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor, QFont,
    QPainterPath, QTransform, QPolygonF, QRadialGradient,
    QAction, QKeySequence, QMouseEvent, QCursor, QIcon
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QPoint, QRect, QPointF,
    QLineF, QSize, QEvent, QPropertyAnimation, QEasingCurve,
    QParallelAnimationGroup, QSequentialAnimationGroup
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


class ViewMode(Enum):
    """Режимы отображения"""
    VIDEO_ONLY = "video_only"  # Только видео
    SKELETON_OVERLAY = "skeleton"  # Скелет поверх видео
    SKELETON_ONLY = "skeleton_only"  # Только скелет
    HEATMAP = "heatmap"  # Тепловая карта уверенности
    DEPTH = "depth"  # Карта глубины
    MULTI_VIEW = "multi_view"  # Мульти-вью (4 камеры)
    SIDE_BY_SIDE = "side_by_side"  # Видео и 3D бок о бок


class VisualizationStyle(Enum):
    """Стили визуализации скелета"""
    SIMPLE = "simple"  # Простые линии
    ANATOMICAL = "anatomical"  # Анатомически точный
    GAMING = "gaming"  # Стиль игровых движков
    SCIENTIFIC = "scientific"  # Научная визуализация
    WIREFRAME = "wireframe"  # Каркасный
    VOLUMETRIC = "volumetric"  # Объемные кости


@dataclass
class CameraView:
    """Настройки вида камеры"""
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    rotation: float = 0.0
    grid_enabled: bool = True
    hud_enabled: bool = True


class SkeletonRenderer:
    """Продвинутый рендерер скелета с разными стилями"""

    # Цветовые схемы
    COLOR_SCHEMES = {
        "default": {
            "joints": QColor(0, 255, 0),
            "bones": QColor(255, 165, 0),
            "selected": QColor(255, 0, 0),
            "root": QColor(0, 0, 255)
        },
        "anatomical": {
            "head": QColor(255, 200, 200),
            "spine": QColor(200, 255, 200),
            "arms": QColor(200, 200, 255),
            "legs": QColor(255, 255, 200)
        },
        "gaming": {
            "joints": QColor(0, 255, 255),
            "bones": QColor(255, 0, 255),
            "selected": QColor(255, 255, 0)
        }
    }

    # Соединения костей (MediaPipe Pose)
    BONE_CONNECTIONS = [
        (0, 1), (0, 4), (1, 2), (2, 3), (3, 7),  # Голова
        (4, 5), (5, 6), (6, 8),  # Голова (правая сторона)
        (9, 10),  # Рот
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Левая рука
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Правая рука
        (11, 23), (12, 24),  # Плечи к бедрам
        (23, 24),  # Таз
        (23, 25), (25, 27), (27, 29), (27, 31),  # Левая нога
        (24, 26), (26, 28), (28, 30), (28, 32)  # Правая нога
    ]

    @staticmethod
    def render_simple(painter: QPainter, landmarks: List, scale: float = 1.0):
        """Простой рендеринг (точки + линии)"""
        if not landmarks:
            return

        # Настройка пера
        bone_pen = QPen(QColor(255, 165, 0, 200))
        bone_pen.setWidthF(2.0 * scale)
        bone_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        joint_pen = QPen(QColor(0, 255, 0, 220))
        joint_pen.setWidthF(4.0 * scale)

        # Рисуем кости
        painter.setPen(bone_pen)
        for start_idx, end_idx in SkeletonRenderer.BONE_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                if hasattr(start, 'position') and hasattr(end, 'position'):
                    painter.drawLine(
                        QPointF(start.position[0], start.position[1]),
                        QPointF(end.position[0], end.position[1])
                    )

        # Рисуем суставы
        painter.setPen(joint_pen)
        for landmark in landmarks:
            if hasattr(landmark, 'position'):
                pos = landmark.position
                confidence = getattr(landmark, 'confidence', 1.0)

                # Размер точки зависит от уверенности
                radius = 4.0 * scale * confidence
                painter.drawEllipse(
                    QPointF(pos[0], pos[1]),
                    radius, radius
                )

    @staticmethod
    def render_anatomical(painter: QPainter, landmarks: List, scale: float = 1.0):
        """Анатомически точный рендеринг"""
        if not landmarks:
            return

        colors = SkeletonRenderer.COLOR_SCHEMES["anatomical"]

        # Голова
        head_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        SkeletonRenderer._render_body_part(
            painter, landmarks, head_indices,
            colors["head"], "head", scale
        )

        # Позвоночник и торс
        spine_indices = [11, 12, 23, 24]
        SkeletonRenderer._render_body_part(
            painter, landmarks, spine_indices,
            colors["spine"], "torso", scale
        )

        # Руки
        left_arm_indices = [11, 13, 15, 17, 19, 21]
        right_arm_indices = [12, 14, 16, 18, 20, 22]

        SkeletonRenderer._render_body_part(
            painter, landmarks, left_arm_indices,
            colors["arms"], "left_arm", scale
        )
        SkeletonRenderer._render_body_part(
            painter, landmarks, right_arm_indices,
            colors["arms"], "right_arm", scale
        )

        # Ноги
        left_leg_indices = [23, 25, 27, 29, 31]
        right_leg_indices = [24, 26, 28, 30, 32]

        SkeletonRenderer._render_body_part(
            painter, landmarks, left_leg_indices,
            colors["legs"], "left_leg", scale
        )
        SkeletonRenderer._render_body_part(
            painter, landmarks, right_leg_indices,
            colors["legs"], "right_leg", scale
        )

    @staticmethod
    def _render_body_part(painter: QPainter, landmarks: List,
                          indices: List[int], color: QColor,
                          part_name: str, scale: float):
        """Рендеринг части тела"""
        pen = QPen(color)
        pen.setWidthF(3.0 * scale)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        brush = QBrush(color)
        brush.setStyle(Qt.BrushStyle.SolidPattern)

        painter.setPen(pen)
        painter.setBrush(brush)

        # Рисуем линии между точками
        points = []
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1]

            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                    hasattr(landmarks[start_idx], 'position') and
                    hasattr(landmarks[end_idx], 'position')):

                start = landmarks[start_idx].position
                end = landmarks[end_idx].position

                painter.drawLine(
                    QPointF(start[0], start[1]),
                    QPointF(end[0], end[1])
                )

                # Сохраняем точки для заливки
                points.append(QPointF(start[0], start[1]))
                if i == len(indices) - 2:
                    points.append(QPointF(end[0], end[1]))

        # Рисуем суставы
        joint_radius = 5.0 * scale
        for idx in indices:
            if idx < len(landmarks) and hasattr(landmarks[idx], 'position'):
                pos = landmarks[idx].position
                painter.drawEllipse(
                    QPointF(pos[0], pos[1]),
                    joint_radius, joint_radius
                )

    @staticmethod
    def render_wireframe(painter: QPainter, landmarks: List, scale: float = 1.0):
        """Каркасный рендеринг"""
        pen = QPen(QColor(0, 255, 255, 180))
        pen.setWidthF(1.5 * scale)
        pen.setStyle(Qt.PenStyle.DashLine)

        painter.setPen(pen)

        # Рисуем все соединения
        for start_idx, end_idx in SkeletonRenderer.BONE_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                if hasattr(start, 'position') and hasattr(end, 'position'):
                    painter.drawLine(
                        QPointF(start.position[0], start.position[1]),
                        QPointF(end.position[0], end.position[1])
                    )

        # Точки вершин
        point_brush = QBrush(QColor(255, 255, 255, 200))
        painter.setBrush(point_brush)

        for landmark in landmarks:
            if hasattr(landmark, 'position'):
                pos = landmark.position
                painter.drawEllipse(
                    QPointF(pos[0], pos[1]),
                    3.0 * scale, 3.0 * scale
                )


class ProfessionalVideoPanel(QWidget):
    """
    ПРОФЕССИОНАЛЬНАЯ ПАНЕЛЬ ВИДЕО ДЛЯ MOCAP

    Особенности:
    1. Multiple viewports (видео, 3D, скелет, heatmap)
    2. Продвинутый рендеринг скелета (5+ стилей)
    3. Инструменты измерения и анализа
    4. Наложение сеток и направляющих
    5. Запись видео и скриншоты
    6. Анимации и переходы
    """

    # Сигналы
    mouse_clicked = pyqtSignal(QPoint, int)  # позиция, кнопка мыши
    mouse_moved = pyqtSignal(QPoint)
    key_pressed = pyqtSignal(int)
    view_changed = pyqtSignal(str)
    screenshot_saved = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_frame = None
        self.current_skeleton = None
        self.landmarks = []
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video)
        self.video_timer.start(33)

        # Настройки отображения
        self.view_mode = ViewMode.SKELETON_OVERLAY
        self.visualization_style = VisualizationStyle.ANATOMICAL
        self.color_scheme = "default"

        # Настройки камеры/вида
        self.camera_view = CameraView()
        self.show_grid = True
        self.show_hud = True
        self.show_measurements = False

        # Инструменты
        self.active_tool = "select"  # select, measure, calibrate, annotate
        self.measurements = []
        self.annotations = []

        # Для масштабирования и панорамирования
        self.is_panning = False
        self.last_pan_point = QPoint()
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)

        # Кэширование
        self.cached_pixmap = None
        self.cached_skeleton = None

        # Таймеры
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._smooth_update)
        self.update_timer.start(16)  # 60 FPS

        # Анимации
        #self.animations = QParallelAnimationGroup()

        self.init_ui()
        #self.init_toolbar()
        self.init_shortcuts()


        logger.info("ProfessionalVideoPanel инициализирован")

    def update_video(self):
        """Обновление видео с камеры"""
        try:
            # Получаем кадр с камеры
            from core.camera_manager import MultiCameraManager

            # Временный код для теста
            if hasattr(self, 'test_frame') and self.test_frame is not None:
                self.update_frame(self.test_frame)
            else:
                # Создаем тестовый кадр
                test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(test_frame, "MOCAP PRO", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(test_frame, "КАМЕРА: 1280x720", (180, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                self.update_frame(test_frame)

        except Exception as e:
            print(f"Ошибка обновления видео: {e}")

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setMinimumSize(640, 480)

        # Основной layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Панель инструментов
        self.toolbar = self.create_toolbar()
        main_layout.addWidget(self.toolbar)

        # Основная область отображения
        self.view_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Главный вид (видео + скелет)
        self.main_view = GraphicsView(self)
        self.main_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.main_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.main_view.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )

        # Включение OpenGL для аппаратного ускорения
        gl_widget = QOpenGLWidget()
        self.main_view.setViewport(gl_widget)

        self.scene = QGraphicsScene()
        self.main_view.setScene(self.scene)

        # Дополнительные виды (опционально)
        if self.view_mode == ViewMode.MULTI_VIEW:
            self._setup_multi_view()
        elif self.view_mode == ViewMode.SIDE_BY_SIDE:
            self._setup_side_by_side()

        self.view_splitter.addWidget(self.main_view)
        self.view_splitter.setSizes([800, 200])

        main_layout.addWidget(self.view_splitter)

        # Статус бар
        self.status_bar = QLabel()
        self.status_bar.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                color: #cccccc;
                padding: 4px;
                font-size: 11px;
            }
        """)
        main_layout.addWidget(self.status_bar)

        # Градиентный фон
        self.setStyleSheet("""
            ProfessionalVideoPanel {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #1a1a2e, stop: 1 #16213e
                );
                border: 1px solid #404040;
            }
        """)

    def create_toolbar(self) -> QToolBar:
        """Создание панели инструментов"""
        toolbar = QToolBar("Video Tools")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)

        # Выбор режима просмотра
        view_combo = QComboBox()
        view_combo.addItem("🎥 Видео + Скелет", ViewMode.SKELETON_OVERLAY)
        view_combo.addItem("🎥 Только видео", ViewMode.VIDEO_ONLY)
        view_combo.addItem("🦴 Только скелет", ViewMode.SKELETON_ONLY)
        view_combo.addItem("🔥 Тепловая карта", ViewMode.HEATMAP)
        view_combo.addItem("📐 Мульти-вид", ViewMode.MULTI_VIEW)
        view_combo.currentIndexChanged.connect(self._on_view_mode_changed)
        toolbar.addWidget(view_combo)

        toolbar.addSeparator()

        # Стиль визуализации
        style_combo = QComboBox()
        style_combo.addItem("🔵 Анатомический", VisualizationStyle.ANATOMICAL)
        style_combo.addItem("⚪ Простой", VisualizationStyle.SIMPLE)
        style_combo.addItem("🎮 Игровой", VisualizationStyle.GAMING)
        style_combo.addItem("📐 Каркасный", VisualizationStyle.WIREFRAME)
        style_combo.currentIndexChanged.connect(self._on_style_changed)
        toolbar.addWidget(style_combo)

        toolbar.addSeparator()

        # Инструменты
        tools_group = QToolButton()
        tools_group.setText("🛠️ Инструменты")
        tools_group.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        tools_menu = QMenu()
        tools_menu.addAction("✏️ Аннотации").triggered.connect(
            lambda: self.set_tool("annotate")
        )
        tools_menu.addAction("📏 Измерения").triggered.connect(
            lambda: self.set_tool("measure")
        )
        tools_menu.addAction("🎯 Калибровка").triggered.connect(
            lambda: self.set_tool("calibrate")
        )
        tools_menu.addSeparator()
        tools_menu.addAction("🗑️ Очистить всё").triggered.connect(
            self.clear_annotations
        )

        tools_group.setMenu(tools_menu)
        toolbar.addWidget(tools_group)

        toolbar.addSeparator()

        # Настройки отображения
        self.grid_toggle = QCheckBox("Сетка")
        self.grid_toggle.setChecked(True)
        self.grid_toggle.stateChanged.connect(self.toggle_grid)
        toolbar.addWidget(self.grid_toggle)

        self.hud_toggle = QCheckBox("HUD")
        self.hud_toggle.setChecked(True)
        self.hud_toggle.stateChanged.connect(self.toggle_hud)
        toolbar.addWidget(self.hud_toggle)

        toolbar.addSeparator()

        # Масштаб
        toolbar.addWidget(QLabel(" Масштаб:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 400)  # 10% - 400%
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(100)
        self.zoom_slider.valueChanged.connect(self.set_zoom)
        toolbar.addWidget(self.zoom_slider)

        self.zoom_label = QLabel("100%")
        toolbar.addWidget(self.zoom_label)

        toolbar.addSeparator()

        # Действия
        toolbar.addAction("📸 Скриншот").triggered.connect(self.take_screenshot)
        toolbar.addAction("🎥 Запись").triggered.connect(self.toggle_recording)

        return toolbar

    def init_shortcuts(self):
        """Инициализация горячих клавиш"""
        shortcuts = {
            Qt.Key.Key_Plus: self.zoom_in,
            Qt.Key.Key_Minus: self.zoom_out,
            Qt.Key.Key_0: self.zoom_reset,
            Qt.Key.Key_G: self.toggle_grid,
            Qt.Key.Key_H: self.toggle_hud,
            Qt.Key.Key_Space: self.toggle_playback,
            Qt.Key.Key_F: self.toggle_fullscreen,
            Qt.Key.Key_F11: self.take_screenshot,
            Qt.Key.Key_R: self.toggle_recording
        }

        for key, callback in shortcuts.items():
            # В PyQt6 нужно создать действие с шорткатом
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(callback)
            self.addAction(action)

    def _setup_multi_view(self):
        """Настройка мульти-вью режима"""
        # 4 вьюпорта для разных ракурсов
        self.top_left_view = GraphicsView(self)
        self.top_right_view = GraphicsView(self)
        self.bottom_left_view = GraphicsView(self)
        self.bottom_right_view = GraphicsView(self)

        # Собираем в grid layout
        # (реализация зависит от требований)

    def _setup_side_by_side(self):
        """Настройка бок-о-бок режима"""
        self.video_view = GraphicsView(self)
        self.skeleton_3d_view = GraphicsView(self)

    def update_frame(self, frame: np.ndarray, skeleton_data: Dict = None):
        """
        Обновление кадра и скелета

        Args:
            frame: Кадр видео (RGB или BGR)
            skeleton_data: Данные скелета
        """
        if frame is not None:
            self.current_frame = frame.copy()

        if skeleton_data is not None:
            self.current_skeleton = skeleton_data
            if 'detailed_landmarks' in skeleton_data:
                self.landmarks = skeleton_data['detailed_landmarks']

        # Помечаем для обновления
        self.cached_pixmap = None
        self.update()

    def paintEvent(self, event):
        """Отрисовка компонентов"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Очистка фона
        painter.fillRect(self.rect(), QColor(30, 30, 40))

        # Вычисляем область отрисовки с учетом масштаба и панорамирования
        draw_rect = self._calculate_draw_rect()

        # Отрисовка сетки если включена
        if self.show_grid and self.view_mode != ViewMode.SKELETON_ONLY:
            self._draw_grid(painter, draw_rect)

        # Отрисовка кадра если есть
        if self.current_frame is not None and self.view_mode != ViewMode.SKELETON_ONLY:
            self._draw_frame(painter, draw_rect)

        # Отрисовка скелета если есть
        if self.landmarks and self.view_mode != ViewMode.VIDEO_ONLY:
            self._draw_skeleton(painter, draw_rect)

        # Отрисовка измерений и аннотаций
        if self.measurements:
            self._draw_measurements(painter, draw_rect)

        if self.annotations:
            self._draw_annotations(painter, draw_rect)

        # Отрисовка HUD если включен
        if self.show_hud:
            self._draw_hud(painter)

        # Отрисовка активного инструмента
        self._draw_active_tool(painter)

    def _calculate_draw_rect(self) -> QRect:
        """Вычисление прямоугольника отрисовки с учетом трансформаций"""
        rect = self.rect().adjusted(10, 10, -10, -10)

        # Применяем масштаб
        if self.zoom_factor != 1.0:
            center = rect.center()
            new_width = rect.width() * self.zoom_factor
            new_height = rect.height() * self.zoom_factor

            # ИСПРАВЬТЕ ЭТУ СТРОКУ:
            # БЫЛО: QRect(center.x() - new_width // 2, ...)
            # СТАЛО:
            rect = QRect(
                int(center.x() - new_width // 2),  # ← int()
                int(center.y() - new_height // 2),  # ← int()
                int(new_width),  # ← int()
                int(new_height)  # ← int()
            )

        # Применяем панорамирование
        if self.pan_offset != QPointF(0, 0):
            rect.translate(int(self.pan_offset.x()), int(self.pan_offset.y()))  # ← int()

        return rect
    def _draw_grid(self, painter: QPainter, draw_rect: QRect):
        """Отрисовка сетки"""
        pen = QPen(QColor(255, 255, 255, 30))
        pen.setWidthF(0.5)
        painter.setPen(pen)

        # Основные линии сетки
        grid_size = 50
        start_x = draw_rect.left() - (draw_rect.left() % grid_size)
        start_y = draw_rect.top() - (draw_rect.top() % grid_size)

        # Вертикальные линии
        x = start_x
        while x <= draw_rect.right():
            painter.drawLine(x, draw_rect.top(), x, draw_rect.bottom())
            x += grid_size

        # Горизонтальные линии
        y = start_y
        while y <= draw_rect.bottom():
            painter.drawLine(draw_rect.left(), y, draw_rect.right(), y)
            y += grid_size

        # Центральные оси
        center_pen = QPen(QColor(255, 100, 100, 150))
        center_pen.setWidthF(1.5)
        painter.setPen(center_pen)

        center_x = draw_rect.center().x()
        center_y = draw_rect.center().y()

        painter.drawLine(center_x, draw_rect.top(), center_x, draw_rect.bottom())
        painter.drawLine(draw_rect.left(), center_y, draw_rect.right(), center_y)

        # Подписи осей
        font = QFont("Arial", 8)
        painter.setFont(font)
        painter.setPen(QColor(200, 200, 200, 180))

        painter.drawText(center_x + 5, draw_rect.top() + 15, "Y")
        painter.drawText(draw_rect.right() - 15, center_y - 5, "X")

    def _draw_frame(self, painter: QPainter, draw_rect: QRect):
        """Отрисовка видеокадра"""
        if self.current_frame is None:
            return

        # Конвертация numpy в QImage
        height, width = self.current_frame.shape[:2]
        bytes_per_line = 3 * width

        if len(self.current_frame.shape) == 3 and self.current_frame.shape[2] == 3:
            # RGB
            qimage = QImage(
                self.current_frame.data, width, height,
                bytes_per_line, QImage.Format.Format_RGB888
            )
        elif len(self.current_frame.shape) == 2:
            # Grayscale
            qimage = QImage(
                self.current_frame.data, width, height,
                width, QImage.Format.Format_Grayscale8
            )
        else:
            # BGR (OpenCV по умолчанию)
            rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            qimage = QImage(
                rgb_frame.data, width, height,
                bytes_per_line, QImage.Format.Format_RGB888
            )

        # Масштабирование и отрисовка
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            draw_rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Центрирование
        pixmap_rect = scaled_pixmap.rect()
        pixmap_rect.moveCenter(draw_rect.center())

        painter.drawPixmap(pixmap_rect, scaled_pixmap)

        # Сохранение для повторного использования
        self.cached_pixmap = scaled_pixmap

    def _draw_skeleton(self, painter: QPainter, draw_rect: QRect):
        """Отрисовка скелета"""
        if not self.landmarks:
            return

        # Сохраняем состояние painter
        painter.save()

        # Масштабирование координат landmarks под текущий вид
        scale_x = draw_rect.width() / 1280  # Предполагаем 1280x720
        scale_y = draw_rect.height() / 720

        # Создаем трансформацию
        transform = QTransform()
        transform.translate(draw_rect.x(), draw_rect.y())
        transform.scale(scale_x, scale_y)
        painter.setTransform(transform)

        # Выбор стиля рендеринга
        if self.visualization_style == VisualizationStyle.SIMPLE:
            SkeletonRenderer.render_simple(painter, self.landmarks)
        elif self.visualization_style == VisualizationStyle.ANATOMICAL:
            SkeletonRenderer.render_anatomical(painter, self.landmarks)
        elif self.visualization_style == VisualizationStyle.WIREFRAME:
            SkeletonRenderer.render_wireframe(painter, self.landmarks)
        elif self.visualization_style == VisualizationStyle.GAMING:
            # Можно добавить специальный рендеринг
            SkeletonRenderer.render_simple(painter, self.landmarks)

        # Восстанавливаем состояние
        painter.restore()

        # Если тепловая карта уверенности
        if self.view_mode == ViewMode.HEATMAP:
            self._draw_confidence_heatmap(painter, draw_rect)

    def _draw_confidence_heatmap(self, painter: QPainter, draw_rect: QRect):
        """Отрисовка тепловой карты уверенности"""
        if not self.landmarks:
            return

        # Создаем радиальные градиенты для каждой точки
        for landmark in self.landmarks:
            if hasattr(landmark, 'position') and hasattr(landmark, 'confidence'):
                pos = landmark.position
                confidence = landmark.confidence

                # Пропускаем точки с низкой уверенностью
                if confidence < 0.3:
                    continue

                # Цвет от зеленого (высокая уверенность) к красному (низкая)
                color = QColor()
                color.setHsv(int(confidence * 120), 255, 255, 150)  # 120° = зеленый, 0° = красный

                # Радиальный градиент
                gradient = QRadialGradient(
                    QPointF(pos[0], pos[1]),  # центр
                    30 * confidence  # радиус
                )
                gradient.setColorAt(0, color)
                gradient.setColorAt(1, QColor(255, 255, 255, 0))

                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(
                    QPointF(pos[0], pos[1]),
                    30 * confidence, 30 * confidence
                )

    def _draw_measurements(self, painter: QPainter, draw_rect: QRect):
        """Отрисовка измерений"""
        pen = QPen(QColor(0, 255, 255, 200))
        pen.setWidthF(2.0)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)

        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)

        for measurement in self.measurements:
            if len(measurement) >= 2:
                start, end = measurement[:2]
                painter.drawLine(start, end)

                # Подпись расстояния
                mid_point = (start + end) / 2
                distance = np.sqrt(
                    (end.x() - start.x()) ** 2 +
                    (end.y() - start.y()) ** 2
                )

                painter.drawText(
                    mid_point.x() + 5, mid_point.y() - 5,
                    f"{distance:.1f}px"
                )

    def _draw_annotations(self, painter: QPainter, draw_rect: QRect):
        """Отрисовка аннотаций"""
        for annotation in self.annotations:
            if isinstance(annotation, dict):
                text = annotation.get('text', '')
                pos = annotation.get('position', QPointF())
                color = annotation.get('color', QColor(255, 255, 0, 200))

                font = QFont("Arial", 12)
                painter.setFont(font)
                painter.setPen(QPen(color))

                painter.drawText(pos, text)

    def _draw_hud(self, painter: QPainter):
        """Отрисовка HUD (Heads-Up Display)"""
        hud_rect = QRect(10, 10, 300, 120)

        # Полупрозрачный фон HUD
        painter.fillRect(hud_rect, QColor(0, 0, 0, 150))

        # Текст HUD
        font = QFont("Consolas", 9)
        painter.setFont(font)
        painter.setPen(QColor(220, 220, 220))

        lines = [
            f"Режим: {self.view_mode.value}",
            f"Стиль: {self.visualization_style.value}",
            f"Масштаб: {self.zoom_factor * 100:.0f}%",
            f"Landmarks: {len(self.landmarks)}",
            f"Инструмент: {self.active_tool}",
            f"Сетка: {'Вкл' if self.show_grid else 'Выкл'}",
            f"HUD: {'Вкл' if self.show_hud else 'Выкл'}"
        ]

        y_offset = 25
        for line in lines:
            painter.drawText(20, y_offset, line)
            y_offset += 18

        # Индикатор записи
        if hasattr(self, 'is_recording') and self.is_recording:
            painter.setBrush(QColor(255, 0, 0, 200))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(15, 130, 10, 10)
            painter.drawText(30, 140, "REC")

    def _draw_active_tool(self, painter: QPainter):
        """Отрисовка активного инструмента"""
        if self.active_tool == "measure" and hasattr(self, 'measure_start'):
            # Показываем линию измерения
            pen = QPen(QColor(0, 255, 255, 200))
            pen.setWidthF(2.0)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)

            current_pos = self.mapFromGlobal(QCursor.pos())
            painter.drawLine(self.measure_start, current_pos)

    def _on_view_mode_changed(self, index: int):
        """Обработка изменения режима просмотра"""
        if index >= 0:
            self.view_mode = self.sender().itemData(index)
            self.view_changed.emit(self.view_mode.value)
            self.update()
            logger.info(f"Режим просмотра изменен на: {self.view_mode.value}")

    def _on_style_changed(self, index: int):
        """Обработка изменения стиля визуализации"""
        if index >= 0:
            self.visualization_style = self.sender().itemData(index)
            self.update()
            logger.info(f"Стиль визуализации изменен на: {self.visualization_style.value}")

    # ==================== ИНТЕРФЕЙСНЫЕ МЕТОДЫ ====================

    def set_tool(self, tool_name: str):
        """Установка активного инструмента"""
        self.active_tool = tool_name

        # Изменение курсора
        cursors = {
            "select": Qt.CursorShape.ArrowCursor,
            "measure": Qt.CursorShape.CrossCursor,
            "calibrate": Qt.CursorShape.CrossCursor,
            "annotate": Qt.CursorShape.IBeamCursor
        }

        self.setCursor(cursors.get(tool_name, Qt.CursorShape.ArrowCursor))
        self.status_bar.setText(f"Инструмент: {tool_name}")

    def clear_annotations(self):
        """Очистка всех аннотаций и измерений"""
        self.measurements.clear()
        self.annotations.clear()
        self.update()

    def toggle_grid(self, checked=None):
        """Переключение сетки"""
        if checked is not None:
            self.show_grid = bool(checked)
        else:
            self.show_grid = not self.show_grid

        if hasattr(self, 'grid_toggle'):
            self.grid_toggle.setChecked(self.show_grid)

        self.update()

    def toggle_hud(self, checked=None):
        """Переключение HUD"""
        if checked is not None:
            self.show_hud = bool(checked)
        else:
            self.show_hud = not self.show_hud

        if hasattr(self, 'hud_toggle'):
            self.hud_toggle.setChecked(self.show_hud)

        self.update()

    def set_zoom(self, value: int):
        """Установка масштаба"""
        self.zoom_factor = value / 100.0
        if hasattr(self, 'zoom_label'):
            self.zoom_label.setText(f"{value}%")
        self.update()

    def zoom_in(self):
        """Увеличение масштаба"""
        current = self.zoom_slider.value()
        self.zoom_slider.setValue(min(current + 10, 400))

    def zoom_out(self):
        """Уменьшение масштаба"""
        current = self.zoom_slider.value()
        self.zoom_slider.setValue(max(current - 10, 10))

    def zoom_reset(self):
        """Сброс масштаба"""
        self.zoom_slider.setValue(100)
        self.pan_offset = QPointF(0, 0)
        self.update()

    def take_screenshot(self):
        """Сохранение скриншота"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"

        # Создаем изображение текущего вида
        screenshot = QPixmap(self.size())
        self.render(screenshot)

        # Сохраняем
        screenshot.save(filename, "PNG")

        self.screenshot_saved.emit(filename)
        self.status_bar.setText(f"Скриншот сохранен: {filename}")
        logger.info(f"Скриншот сохранен: {filename}")

    def toggle_recording(self):
        """Переключение записи видео"""
        if not hasattr(self, 'is_recording'):
            self.is_recording = False

        self.is_recording = not self.is_recording

        if self.is_recording:
            self.status_bar.setText("Запись видео начата")
            # Здесь можно начать запись в видеофайл
        else:
            self.status_bar.setText("Запись видео остановлена")
            # Остановка записи

        self.update()

    def toggle_playback(self):
        """Переключение воспроизведения"""
        # Для работы с таймлайном
        pass

    def toggle_fullscreen(self):
        """Переключение полноэкранного режима"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _smooth_update(self):
        """Плавное обновление"""
        self.update()

        # ==================== СОБЫТИЯ МЫШИ И КЛАВИАТУРЫ ====================

    def mousePressEvent(self, event: QMouseEvent):
        """Обработка нажатия мыши"""
        pos = event.pos()

        if event.button() == Qt.MouseButton.LeftButton:
            if self.active_tool == "measure":
                self.measure_start = pos
            elif self.active_tool == "annotate":
                # Добавление текстовой аннотации
                pass

            self.mouse_clicked.emit(pos, 1)

        elif event.button() == Qt.MouseButton.RightButton:
            # Контекстное меню
            self._show_context_menu(pos)
            self.mouse_clicked.emit(pos, 3)

        elif event.button() == Qt.MouseButton.MiddleButton:
            # Начало панорамирования
            self.is_panning = True
            self.last_pan_point = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Обработка движения мыши"""
        pos = event.pos()

        if self.is_panning:
            # Панорамирование
            delta = pos - self.last_pan_point
            self.pan_offset += delta
            self.last_pan_point = pos
            self.update()

        self.mouse_moved.emit(pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Обработка отпускания мыши"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

        if event.button() == Qt.MouseButton.LeftButton:
            if self.active_tool == "measure" and hasattr(self, 'measure_start'):
                # Завершение измерения
                self.measurements.append([self.measure_start, event.pos()])
                delattr(self, 'measure_start')
                self.update()

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Обработка колесика мыши (зум)"""
        delta = event.angleDelta().y()

        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

        event.accept()

    def keyPressEvent(self, event):
        """Обработка нажатия клавиш"""
        self.key_pressed.emit(event.key())

        # Проброс в родительский виджет
        super().keyPressEvent(event)

    def _show_context_menu(self, pos: QPoint):
        """Показ контекстного меню"""
        menu = QMenu(self)

        # Действия с кадром
        menu.addAction("📸 Скриншот").triggered.connect(self.take_screenshot)
        menu.addAction("📋 Копировать кадр").triggered.connect(
            lambda: self._copy_frame_to_clipboard()
        )

        menu.addSeparator()

        # Инструменты
        tools_menu = menu.addMenu("🛠️ Инструменты")
        tools_menu.addAction("📏 Измерить расстояние").triggered.connect(
            lambda: self.set_tool("measure")
        )
        tools_menu.addAction("✏️ Добавить аннотацию").triggered.connect(
            lambda: self.set_tool("annotate")
        )

        menu.addSeparator()

        # Настройки отображения
        display_menu = menu.addMenu("👁️ Отображение")
        display_menu.addAction("Сетка").triggered.connect(
            lambda: self.toggle_grid()
        ).setCheckable(True).setChecked(self.show_grid)

        display_menu.addAction("HUD").triggered.connect(
            lambda: self.toggle_hud()
        ).setCheckable(True).setChecked(self.show_hud)

        menu.exec(self.mapToGlobal(pos))

    def _copy_frame_to_clipboard(self):
        """Копирование текущего кадра в буфер обмена"""
        if self.current_frame is not None:
            # Конвертация в QPixmap и копирование
            height, width = self.current_frame.shape[:2]
            bytes_per_line = 3 * width

            if len(self.current_frame.shape) == 3 and self.current_frame.shape[2] == 3:
                qimage = QImage(
                    self.current_frame.data, width, height,
                    bytes_per_line, QImage.Format.Format_RGB888
                )
            else:
                rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                qimage = QImage(
                    rgb_frame.data, width, height,
                    bytes_per_line, QImage.Format.Format_RGB888
                )

            pixmap = QPixmap.fromImage(qimage)
            QApplication.clipboard().setPixmap(pixmap)

            self.status_bar.setText("Кадр скопирован в буфер обмена")
            logger.info("Кадр скопирован в буфер обмена")


# Вспомогательный класс для GraphicsView
class GraphicsView(QGraphicsView):
    """Кастомный GraphicsView с улучшенной обработкой событий"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event):
        """Обработка колесика для зума"""
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)

        event.accept()


# Для обратной совместимости
class VideoPanel(ProfessionalVideoPanel):
    """Алиас для обратной совместимости"""
    pass


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    panel = ProfessionalVideoPanel()
    panel.resize(800, 600)
    panel.show()

    # Тестовые данные
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (100, 100), (300, 300), (0, 255, 0), 2)

    panel.update_frame(test_frame)

    sys.exit(app.exec())