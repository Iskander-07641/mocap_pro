"""
Модуль: Timeline Editor (Нелинейный редактор анимации)
Версия: 1.0.0
Автор: Mocap Pro Team

Многотрековый редактор временной шкалы для нелинейного редактирования анимации.
Поддерживает ключевые кадры, кривые Безье, синхронизацию с видео и аудио.
"""

import sys
import json
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from core.animation_recorder import AnimationLayer, Keyframe, AnimationRecorder
from utils.math_utils import bezier_interpolation, quaternion_slerp


# Типы треков
class TrackType(Enum):
    SKELETON_ANIMATION = "skeleton"
    VIDEO = "video"
    AUDIO = "audio"
    MARKER = "marker"
    EFFECT = "effect"


# Состояния редактирования
class EditMode(Enum):
    SELECT = "select"
    CUT = "cut"
    TRIM = "trim"
    KEYFRAME = "keyframe"
    PAN = "pan"
    ZOOM = "zoom"


# Типы интерполяции
class InterpolationType(Enum):
    LINEAR = "linear"
    BEZIER = "bezier"
    STEP = "step"
    SLERP = "slerp"
    EASE_IN_OUT = "ease_in_out"


@dataclass
class TimelineTrack:
    """Представляет один трек на временной шкале"""
    id: str
    name: str
    type: TrackType
    color: QColor
    visible: bool = True
    locked: bool = False
    muted: bool = False
    height: int = 60
    data: Any = None
    segments: List['TimelineSegment'] = field(default_factory=list)

    def add_segment(self, segment: 'TimelineSegment'):
        """Добавляет сегмент на трек"""
        # Проверка пересечений
        for existing in self.segments:
            if existing.intersects(segment):
                raise ValueError(f"Segment intersects with existing segment {existing.id}")
        self.segments.append(segment)
        segment.parent_track = self

    def get_segment_at(self, time_sec: float) -> Optional['TimelineSegment']:
        """Возвращает сегмент в указанное время"""
        for segment in self.segments:
            if segment.start_time <= time_sec <= segment.end_time:
                return segment
        return None


@dataclass
class TimelineSegment:
    """Сегмент данных на треке (клип)"""
    id: str
    name: str
    start_time: float  # секунды
    duration: float  # секунды
    data_ref: Any  # Ссылка на оригинальные данные
    parent_track: Optional[TimelineTrack] = None
    properties: Dict = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def intersects(self, other: 'TimelineSegment') -> bool:
        """Проверяет пересечение с другим сегментом"""
        return not (self.end_time <= other.start_time or
                    other.end_time <= self.start_time)

    def split(self, split_time: float) -> Tuple['TimelineSegment', 'TimelineSegment']:
        """Разделяет сегмент в указанное время"""
        if not self.start_time < split_time < self.end_time:
            raise ValueError("Split time must be inside segment")

        # Первая часть
        part1 = TimelineSegment(
            id=f"{self.id}_part1",
            name=f"{self.name} (Part 1)",
            start_time=self.start_time,
            duration=split_time - self.start_time,
            data_ref=self.data_ref,
            properties=self.properties.copy()
        )

        # Вторая часть
        part2 = TimelineSegment(
            id=f"{self.id}_part2",
            name=f"{self.name} (Part 2)",
            start_time=split_time,
            duration=self.end_time - split_time,
            data_ref=self.data_ref,
            properties=self.properties.copy()
        )

        return part1, part2


class KeyframeCurveEditor(QWidget):
    """Редактор кривых ключевых кадров"""

    curve_updated = pyqtSignal(str, list)  # joint_name, keyframes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.joint_name = ""
        self.keyframes: List[Keyframe] = []
        self.selected_keyframes = set()
        self.view_scale = QPointF(1.0, 1.0)
        self.view_offset = QPointF(0, 0)
        self.is_panning = False
        self.last_mouse_pos = QPoint()
        self.hovered_keyframe = -1
        self.tangent_handles = {}  # keyframe_idx -> (in_tangent, out_tangent)

        # Настройки отображения
        self.grid_size = QPointF(50, 50)
        self.keyframe_radius = 6
        self.handle_length = 40

        self.setMinimumSize(600, 300)
        self.setMouseTracking(True)

    def set_data(self, joint_name: str, keyframes: List[Keyframe]):
        """Устанавливает данные для редактирования"""
        self.joint_name = joint_name
        self.keyframes = sorted(keyframes, key=lambda k: k.timestamp)
        self.selected_keyframes.clear()
        self.update_tangents()
        self.update()

    def update_tangents(self):
        """Обновляет касательные для Безье-интерполяции"""
        self.tangent_handles.clear()
        if len(self.keyframes) < 2:
            return

        for i, kf in enumerate(self.keyframes):
            in_tangent = QPointF(0, 0)
            out_tangent = QPointF(0, 0)

            if kf.interpolation == InterpolationType.BEZIER:
                # Автоматическое вычисление касательных
                if i > 0:
                    prev = self.keyframes[i - 1]
                    delta = (kf.timestamp - prev.timestamp) * 0.3
                    in_tangent = QPointF(-delta, 0)

                if i < len(self.keyframes) - 1:
                    next_kf = self.keyframes[i + 1]
                    delta = (next_kf.timestamp - kf.timestamp) * 0.3
                    out_tangent = QPointF(delta, 0)

            self.tangent_handles[i] = (in_tangent, out_tangent)

    def paintEvent(self, event):
        """Отрисовка редактора кривых"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Фон
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        # Сетка
        self.draw_grid(painter)

        # Оси
        self.draw_axes(painter)

        # Кривая
        if len(self.keyframes) >= 2:
            self.draw_curve(painter)

        # Ключевые кадры
        for i, kf in enumerate(self.keyframes):
            self.draw_keyframe(painter, i, kf)

        # Информация
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(10, 20, f"Joint: {self.joint_name} | Keyframes: {len(self.keyframes)}")

    def draw_grid(self, painter: QPainter):
        """Рисует сетку"""
        painter.setPen(QPen(QColor(60, 60, 60), 1))

        width = self.width()
        height = self.height()

        # Вертикальные линии (время)
        for x in range(0, width, int(self.grid_size.x() * self.view_scale.x())):
            line_x = x + self.view_offset.x() % (self.grid_size.x() * self.view_scale.x())
            painter.drawLine(int(line_x), 0, int(line_x), height)

        # Горизонтальные линии (значение)
        for y in range(0, height, int(self.grid_size.y() * self.view_scale.y())):
            line_y = y + self.view_offset.y() % (self.grid_size.y() * self.view_scale.y())
            painter.drawLine(0, int(line_y), width, int(line_y))

    def draw_axes(self, painter: QPainter):
        """Рисует оси координат"""
        center_x = -self.view_offset.x()
        center_y = self.height() / 2 - self.view_offset.y()

        # Ось X (время)
        painter.setPen(QPen(QColor(100, 150, 255), 2))
        painter.drawLine(0, int(center_y), self.width(), int(center_y))

        # Ось Y (значение)
        painter.setPen(QPen(QColor(255, 150, 100), 2))
        painter.drawLine(int(center_x), 0, int(center_x), self.height())

        # Подписи
        painter.setPen(QColor(150, 150, 150))
        for i in range(0, self.width(), 100):
            time = (i - center_x) / self.view_scale.x()
            painter.drawText(i, int(center_y) + 15, f"{time:.2f}s")

    def draw_curve(self, painter: QPainter):
        """Рисует кривую интерполяции"""
        if not self.keyframes:
            return

        path = QPainterPath()

        for i in range(len(self.keyframes) - 1):
            kf1 = self.keyframes[i]
            kf2 = self.keyframes[i + 1]

            x1 = self.time_to_x(kf1.timestamp)
            y1 = self.value_to_y(kf1.position[0])  # Берем X-компоненту для примера

            x2 = self.time_to_x(kf2.timestamp)
            y2 = self.value_to_y(kf2.position[0])

            if kf1.interpolation == InterpolationType.BEZIER and i in self.tangent_handles:
                _, out_tangent = self.tangent_handles[i]
                in_tangent, _ = self.tangent_handles.get(i + 1, (QPointF(0, 0), QPointF(0, 0)))

                # Контрольные точки Безье
                cp1_x = x1 + out_tangent.x() * self.view_scale.x()
                cp1_y = y1 + out_tangent.y() * self.view_scale.y()
                cp2_x = x2 + in_tangent.x() * self.view_scale.x()
                cp2_y = y2 + in_tangent.y() * self.view_scale.y()

                if i == 0:
                    path.moveTo(x1, y1)
                path.cubicTo(cp1_x, cp1_y, cp2_x, cp2_y, x2, y2)
            else:
                if i == 0:
                    path.moveTo(x1, y1)
                path.lineTo(x2, y2)

        painter.setPen(QPen(QColor(0, 255, 150), 2))
        painter.drawPath(path)

    def draw_keyframe(self, painter: QPainter, idx: int, keyframe: Keyframe):
        """Рисует ключевой кадр и его касательные"""
        x = self.time_to_x(keyframe.timestamp)
        y = self.value_to_y(keyframe.position[0])  # X-компонента

        # Выбор цвета
        if idx in self.selected_keyframes:
            color = QColor(255, 100, 100)
        elif idx == self.hovered_keyframe:
            color = QColor(255, 200, 100)
        else:
            color = QColor(100, 200, 255)

        # Ключевой кадр
        painter.setBrush(color)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawEllipse(QPointF(x, y), self.keyframe_radius, self.keyframe_radius)

        # Касательные для Безье
        if keyframe.interpolation == InterpolationType.BEZIER and idx in self.tangent_handles:
            in_tangent, out_tangent = self.tangent_handles[idx]

            # Исходящая касательная
            if out_tangent:
                tx = x + out_tangent.x() * self.view_scale.x()
                ty = y + out_tangent.y() * self.view_scale.y()
                painter.setPen(QPen(QColor(200, 200, 100), 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(x), int(y), int(tx), int(ty))
                painter.setBrush(QColor(200, 200, 100))
                painter.drawEllipse(QPointF(tx, ty), 4, 4)

            # Входящая касательная
            if in_tangent:
                tx = x + in_tangent.x() * self.view_scale.x()
                ty = y + in_tangent.y() * self.view_scale.y()
                painter.setPen(QPen(QColor(100, 200, 200), 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(x), int(y), int(tx), int(ty))
                painter.setBrush(QColor(100, 200, 200))
                painter.drawEllipse(QPointF(tx, ty), 4, 4)

    def time_to_x(self, time: float) -> float:
        """Конвертирует время в координату X"""
        return time * self.view_scale.x() + self.view_offset.x()

    def value_to_y(self, value: float) -> float:
        """Конвертирует значение в координату Y"""
        return self.height() / 2 - value * self.view_scale.y() + self.view_offset.y()

    def mousePressEvent(self, event):
        """Обработка нажатия мыши"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Проверяем клик по ключевому кадру
            pos = event.position()
            clicked_idx = self.get_keyframe_at(pos.x(), pos.y())

            if clicked_idx != -1:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    # Добавляем/удаляем из выделения
                    if clicked_idx in self.selected_keyframes:
                        self.selected_keyframes.remove(clicked_idx)
                    else:
                        self.selected_keyframes.add(clicked_idx)
                else:
                    # Новое выделение
                    self.selected_keyframes = {clicked_idx}
                self.update()
            else:
                # Начало панорамирования
                self.is_panning = True
                self.last_mouse_pos = event.pos()

        elif event.button() == Qt.MouseButton.RightButton:
            # Контекстное меню
            self.show_context_menu(event.pos())

    def mouseMoveEvent(self, event):
        """Обработка движения мыши"""
        pos = event.position()

        # Обновляем hovered ключевой кадр
        self.hovered_keyframe = self.get_keyframe_at(pos.x(), pos.y())

        # Панорамирование
        if self.is_panning:
            delta = event.pos() - self.last_mouse_pos
            self.view_offset += delta
            self.last_mouse_pos = event.pos()
            self.update()

        # Обновляем курсор
        if self.hovered_keyframe != -1:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif self.is_panning:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        self.update()

    def mouseReleaseEvent(self, event):
        """Обработка отпускания мыши"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):
        """Обработка колесика мыши для масштабирования"""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9

        # Масштабирование относительно позиции курсора
        mouse_pos = event.position()
        old_x = (mouse_pos.x() - self.view_offset.x()) / self.view_scale.x()
        old_y = (mouse_pos.y() - self.view_offset.y()) / self.view_scale.y()

        self.view_scale *= zoom_factor

        new_x = mouse_pos.x() - old_x * self.view_scale.x()
        new_y = mouse_pos.y() - old_y * self.view_scale.y()
        self.view_offset = QPointF(new_x, new_y)

        self.update()

    def get_keyframe_at(self, x: float, y: float) -> int:
        """Находит ключевой кадр по координатам"""
        for i, kf in enumerate(self.keyframes):
            kf_x = self.time_to_x(kf.timestamp)
            kf_y = self.value_to_y(kf.position[0])

            distance = ((x - kf_x) ** 2 + (y - kf_y) ** 2) ** 0.5
            if distance <= self.keyframe_radius:
                return i
        return -1

    def show_context_menu(self, pos: QPoint):
        """Показывает контекстное меню"""
        menu = QMenu(self)

        # Действия для ключевых кадров
        if self.selected_keyframes:
            change_interp_menu = menu.addMenu("Change Interpolation")

            for interp in InterpolationType:
                action = change_interp_menu.addAction(interp.value.title())
                action.triggered.connect(
                    lambda checked, i=interp: self.change_interpolation(i)
                )

            menu.addSeparator()

            delete_action = menu.addAction("Delete Keyframes")
            delete_action.triggered.connect(self.delete_selected_keyframes)

        # Общие действия
        menu.addSeparator()
        reset_view_action = menu.addAction("Reset View")
        reset_view_action.triggered.connect(self.reset_view)

        menu.exec(self.mapToGlobal(pos))

    def change_interpolation(self, interp_type: InterpolationType):
        """Изменяет тип интерполяции выделенных ключевых кадров"""
        for idx in self.selected_keyframes:
            if 0 <= idx < len(self.keyframes):
                self.keyframes[idx].interpolation = interp_type

        self.update_tangents()
        self.update()
        self.curve_updated.emit(self.joint_name, self.keyframes)

    def delete_selected_keyframes(self):
        """Удаляет выделенные ключевые кадры"""
        # Сортируем в обратном порядке для безопасного удаления
        for idx in sorted(self.selected_keyframes, reverse=True):
            if 0 <= idx < len(self.keyframes):
                del self.keyframes[idx]

        self.selected_keyframes.clear()
        self.update_tangents()
        self.update()
        self.curve_updated.emit(self.joint_name, self.keyframes)

    def reset_view(self):
        """Сбрасывает вид редактора"""
        self.view_scale = QPointF(1.0, 1.0)
        self.view_offset = QPointF(0, 0)
        self.update()


class TimelineWidget(QWidget):
    """Виджет временной шкалы с треками"""

    # Сигналы
    time_changed = pyqtSignal(float)  # Текущее время изменилось
    selection_changed = pyqtSignal(list)  # Изменилось выделение
    edit_performed = pyqtSignal(str, dict)  # Выполнено редактирование

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracks: List[TimelineTrack] = []
        self.current_time = 0.0
        self.duration = 60.0  # секунд
        self.playhead_visible = True
        self.playhead_color = QColor(255, 100, 100)
        self.selection_rect = QRect()
        self.is_selecting = False
        self.edit_mode = EditMode.SELECT
        self.snap_to_grid = True
        self.grid_spacing = 0.1  # 100ms

        # Настройки отображения
        self.time_scale = 100.0  # пикселей в секунду
        self.track_header_width = 150
        self.timeline_height = 400
        self.minimum_track_height = 40

        self.setMinimumHeight(self.timeline_height)
        self.setMouseTracking(True)

        # Инициализация треков по умолчанию
        self.init_default_tracks()

    def init_default_tracks(self):
        """Инициализирует треки по умолчанию"""
        # Трек анимации скелета
        anim_track = TimelineTrack(
            id="skeleton_anim",
            name="Skeleton Animation",
            type=TrackType.SKELETON_ANIMATION,
            color=QColor(100, 150, 255)
        )
        self.tracks.append(anim_track)

        # Трек видео
        video_track = TimelineTrack(
            id="video_source",
            name="Video Source",
            type=TrackType.VIDEO,
            color=QColor(255, 150, 100)
        )
        self.tracks.append(video_track)

        # Трек аудио
        audio_track = TimelineTrack(
            id="audio_track",
            name="Audio",
            type=TrackType.AUDIO,
            color=QColor(150, 255, 100)
        )
        self.tracks.append(audio_track)

    def add_track(self, track: TimelineTrack):
        """Добавляет трек на временную шкалу"""
        self.tracks.append(track)
        self.update()

    def remove_track(self, track_id: str):
        """Удаляет трек по ID"""
        self.tracks = [t for t in self.tracks if t.id != track_id]
        self.update()

    def get_track_by_id(self, track_id: str) -> Optional[TimelineTrack]:
        """Возвращает трек по ID"""
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None

    def paintEvent(self, event):
        """Отрисовка временной шкалы"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Фон
        painter.fillRect(self.rect(), QColor(50, 50, 50))

        # Заголовки треков
        self.draw_track_headers(painter)

        # Область временной шкалы
        timeline_rect = QRect(
            self.track_header_width, 0,
            self.width() - self.track_header_width, self.height()
        )
        painter.setClipRect(timeline_rect)

        # Сетка времени
        self.draw_time_grid(painter, timeline_rect)

        # Треки и сегменты
        self.draw_tracks(painter, timeline_rect)

        # Playhead (линия текущего времени)
        if self.playhead_visible:
            self.draw_playhead(painter, timeline_rect)

        # Область выделения
        if not self.selection_rect.isEmpty():
            self.draw_selection_rect(painter)

    def draw_track_headers(self, painter: QPainter):
        """Рисует заголовки треков"""
        header_rect = QRect(0, 0, self.track_header_width, self.height())
        painter.fillRect(header_rect, QColor(40, 40, 40))

        y_offset = 0
        for track in self.tracks:
            if not track.visible:
                continue

            track_header = QRect(0, y_offset, self.track_header_width, track.height)

            # Фон заголовка
            painter.fillRect(track_header, track.color.darker(200))

            # Название трека
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                track_header.adjusted(10, 10, -10, -10),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                track.name
            )

            # Иконки состояния
            icon_size = 16
            icon_x = self.track_header_width - icon_size - 5

            # Видимость
            visibility_icon = "👁" if track.visible else "👁‍🗨"
            painter.drawText(icon_x, y_offset + 20, visibility_icon)

            # Блокировка
            lock_icon = "🔒" if track.locked else "🔓"
            painter.drawText(icon_x - 20, y_offset + 20, lock_icon)

            y_offset += track.height

    def draw_time_grid(self, painter: QPainter, timeline_rect: QRect):
        """Рисует сетку времени"""
        painter.setPen(QPen(QColor(70, 70, 70), 1))

        # Основные деления (каждую секунду)
        for second in range(0, int(self.duration) + 1):
            x = self.time_to_pixel(second)
            if timeline_rect.left() <= x <= timeline_rect.right():
                painter.drawLine(int(x), timeline_rect.top(), int(x), timeline_rect.bottom())

                # Подпись времени
                painter.setPen(QColor(150, 150, 150))
                time_text = f"{second}s"
                painter.drawText(int(x) + 5, 20, time_text)
                painter.setPen(QPen(QColor(70, 70, 70), 1))

        # Мелкие деления (каждые 100ms)
        painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.PenStyle.DashLine))
        for ms in range(0, int(self.duration * 10) + 1):
            if ms % 10 == 0:  # Пропускаем секунды
                continue
            x = self.time_to_pixel(ms / 10.0)
            if timeline_rect.left() <= x <= timeline_rect.right():
                painter.drawLine(int(x), timeline_rect.top(), int(x), timeline_rect.bottom())

    def draw_tracks(self, painter: QPainter, timeline_rect: QRect):
        """Рисует треки и сегменты"""
        y_offset = 0

        for track in self.tracks:
            if not track.visible:
                continue

            track_rect = QRect(
                timeline_rect.left(),
                y_offset,
                timeline_rect.width(),
                track.height
            )

            # Фон трека (чередование)
            if y_offset % (track.height * 2) == 0:
                painter.fillRect(track_rect, QColor(45, 45, 45))
            else:
                painter.fillRect(track_rect, QColor(55, 55, 55))

            # Сегменты трека
            for segment in track.segments:
                self.draw_segment(painter, segment, track_rect, y_offset)

            # Границы трека
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.drawLine(
                track_rect.left(), track_rect.bottom(),
                track_rect.right(), track_rect.bottom()
            )

            y_offset += track.height

    def draw_segment(self, painter: QPainter, segment: TimelineSegment,
                     track_rect: QRect, track_y: int):
        """Рисует сегмент на треке"""
        start_x = self.time_to_pixel(segment.start_time)
        end_x = self.time_to_pixel(segment.end_time)
        width = max(10, end_x - start_x)

        segment_rect = QRect(
            int(start_x),
            track_y + 5,
            int(width),
            track_rect.height() - 10
        )

        # Прямоугольник сегмента
        color = segment.parent_track.color if segment.parent_track else QColor(150, 150, 150)
        painter.fillRect(segment_rect, color)

        # Тень
        painter.setPen(QPen(color.darker(150), 2))
        painter.drawRect(segment_rect)

        # Название сегмента
        painter.setPen(QColor(255, 255, 255))
        text_rect = segment_rect.adjusted(5, 5, -5, -5)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            segment.name
        )

        # Длительность
        duration_text = f"{segment.duration:.2f}s"
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            duration_text
        )

        # Ручки изменения размера (если достаточно широкий)
        if width > 30:
            # Левая ручка
            left_handle = QRect(
                segment_rect.left() - 3,
                segment_rect.top() + segment_rect.height() // 2 - 10,
                6, 20
            )
            painter.fillRect(left_handle, QColor(200, 200, 200))

            # Правая ручка
            right_handle = QRect(
                segment_rect.right() - 3,
                segment_rect.top() + segment_rect.height() // 2 - 10,
                6, 20
            )
            painter.fillRect(right_handle, QColor(200, 200, 200))

    def draw_playhead(self, painter: QPainter, timeline_rect: QRect):
        """Рисует линию текущего времени"""
        playhead_x = self.time_to_pixel(self.current_time)

        painter.setPen(QPen(self.playhead_color, 2))
        painter.drawLine(
            int(playhead_x), timeline_rect.top(),
            int(playhead_x), timeline_rect.bottom()
        )

        # Треугольник сверху
        triangle = QPolygon([
            QPoint(int(playhead_x) - 6, timeline_rect.top()),
            QPoint(int(playhead_x) + 6, timeline_rect.top()),
            QPoint(int(playhead_x), timeline_rect.top() + 12)
        ])
        painter.setBrush(self.playhead_color)
        painter.drawPolygon(triangle)

    def draw_selection_rect(self, painter: QPainter):
        """Рисует прямоугольник выделения"""
        painter.setBrush(QColor(100, 150, 255, 50))
        painter.setPen(QPen(QColor(100, 150, 255), 1, Qt.PenStyle.DashLine))
        painter.drawRect(self.selection_rect)

    def time_to_pixel(self, time_sec: float) -> float:
        """Конвертирует время в пиксели"""
        return self.track_header_width + time_sec * self.time_scale

    def pixel_to_time(self, pixel_x: float) -> float:
        """Конвертирует пиксели во время"""
        return (pixel_x - self.track_header_width) / self.time_scale

    def mousePressEvent(self, event):
        """Обработка нажатия мыши"""
        pos = event.position()
        timeline_x = pos.x() - self.track_header_width

        if event.button() == Qt.MouseButton.LeftButton:
            if timeline_x >= 0:
                # Клик по временной шкале
                if self.edit_mode == EditMode.SELECT:
                    # Начало выделения
                    self.selection_rect = QRect(pos.toPoint(), QSize())
                    self.is_selecting = True
                elif self.edit_mode == EditMode.CUT:
                    # Разрезание сегментов
                    self.cut_at_time(self.pixel_to_time(pos.x()))
                else:
                    # Установка времени
                    new_time = self.pixel_to_time(pos.x())
                    if self.snap_to_grid:
                        new_time = round(new_time / self.grid_spacing) * self.grid_spacing
                    self.set_current_time(new_time)

        elif event.button() == Qt.MouseButton.RightButton:
            self.show_timeline_context_menu(pos.toPoint())

    def mouseMoveEvent(self, event):
        """Обработка движения мыши"""
        pos = event.position()

        if self.is_selecting:
            # Обновление прямоугольника выделения
            self.selection_rect.setBottomRight(pos.toPoint())
            self.update()
        elif event.buttons() & Qt.MouseButton.LeftButton:
            # Перетаскивание playhead
            timeline_x = pos.x() - self.track_header_width
            if timeline_x >= 0:
                new_time = self.pixel_to_time(pos.x())
                if self.snap_to_grid:
                    new_time = round(new_time / self.grid_spacing) * self.grid_spacing
                self.set_current_time(new_time)

        # Обновляем курсор
        self.update_cursor(pos)

    def mouseReleaseEvent(self, event):
        """Обработка отпускания мыши"""
        if event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            self.is_selecting = False
            self.process_selection()
            self.selection_rect = QRect()
            self.update()

    def wheelEvent(self, event):
        """Обработка колесика мыши для масштабирования"""
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Масштабирование времени
            zoom_factor = 1.1 if delta > 0 else 0.9
            self.time_scale *= zoom_factor
            self.time_scale = max(10.0, min(1000.0, self.time_scale))
            self.update()
        else:
            # Вертикальная прокрутка
            super().wheelEvent(event)

    def update_cursor(self, pos: QPointF):
        """Обновляет курсор в зависимости от позиции"""
        timeline_x = pos.x() - self.track_header_width

        if timeline_x >= 0:
            # Проверяем, находимся ли над ручкой сегмента
            for track in self.tracks:
                for segment in track.segments:
                    if self.is_over_segment_handle(pos, segment):
                        self.setCursor(Qt.CursorShape.SizeHorCursor)
                        return

            self.setCursor(Qt.CursorShape.ArrowCursor)

    def is_over_segment_handle(self, pos: QPointF, segment: TimelineSegment) -> bool:
        """Проверяет, находится ли курсор над ручкой сегмента"""
        if not segment.parent_track:
            return False

        track_idx = self.tracks.index(segment.parent_track)
        track_y = sum(t.height for t in self.tracks[:track_idx])

        start_x = self.time_to_pixel(segment.start_time)
        end_x = self.time_to_pixel(segment.end_time)

        # Левая ручка
        left_handle = QRectF(
            start_x - 3,
            track_y + segment.parent_track.height // 2 - 10,
            6, 20
        )

        # Правая ручка
        right_handle = QRectF(
            end_x - 3,
            track_y + segment.parent_track.height // 2 - 10,
            6, 20
        )

        return left_handle.contains(pos) or right_handle.contains(pos)

    def set_current_time(self, time_sec: float):
        """Устанавливает текущее время"""
        self.current_time = max(0.0, min(self.duration, time_sec))
        self.time_changed.emit(self.current_time)
        self.update()

    def cut_at_time(self, cut_time: float):
        """Разрезает сегменты в указанное время"""
        for track in self.tracks:
            if track.locked:
                continue

            for segment in track.segments[:]:  # Копируем список
                if segment.start_time < cut_time < segment.end_time:
                    # Разрезаем сегмент
                    part1, part2 = segment.split(cut_time)

                    # Удаляем оригинальный и добавляем части
                    track.segments.remove(segment)
                    track.add_segment(part1)
                    track.add_segment(part2)

        self.edit_performed.emit("cut", {"time": cut_time})
        self.update()

    def process_selection(self):
        """Обрабатывает область выделения"""
        if self.selection_rect.isEmpty():
            return

        # Конвертируем прямоугольник во время
        start_time = self.pixel_to_time(self.selection_rect.left())
        end_time = self.pixel_to_time(self.selection_rect.right())

        # Находим сегменты в выделенной области
        selected_segments = []
        for track in self.tracks:
            for segment in track.segments:
                if (segment.start_time <= end_time and
                        segment.end_time >= start_time):
                    selected_segments.append(segment)

        self.selection_changed.emit(selected_segments)

    def show_timeline_context_menu(self, pos: QPoint):
        """Показывает контекстное меню временной шкалы"""
        menu = QMenu(self)

        # Режимы редактирования
        mode_menu = menu.addMenu("Edit Mode")
        for mode in EditMode:
            action = mode_menu.addAction(mode.value.title())
            action.setCheckable(True)
            action.setChecked(self.edit_mode == mode)
            action.triggered.connect(
                lambda checked, m=mode: self.set_edit_mode(m)
            )

        menu.addSeparator()

        # Привязка к сетке
        snap_action = menu.addAction("Snap to Grid")
        snap_action.setCheckable(True)
        snap_action.setChecked(self.snap_to_grid)
        snap_action.triggered.connect(
            lambda checked: setattr(self, 'snap_to_grid', checked)
        )

        # Настройки сетки
        grid_menu = menu.addMenu("Grid Spacing")
        spacings = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        for spacing in spacings:
            action = grid_menu.addAction(f"{spacing * 1000:.0f} ms")
            action.triggered.connect(
                lambda checked, s=spacing: setattr(self, 'grid_spacing', s)
            )

        menu.addSeparator()

        # Сброс масштаба
        reset_zoom_action = menu.addAction("Reset Zoom")
        reset_zoom_action.triggered.connect(self.reset_zoom)

        menu.exec(self.mapToGlobal(pos))

    def set_edit_mode(self, mode: EditMode):
        """Устанавливает режим редактирования"""
        self.edit_mode = mode
        self.update()

    def reset_zoom(self):
        """Сбрасывает масштаб временной шкалы"""
        self.time_scale = 100.0
        self.update()


class TimelineEditor(QMainWindow):
    """Главное окно редактора временной шкалы"""

    def __init__(self, animation_recorder: AnimationRecorder = None):
        super().__init__()
        self.animation_recorder = animation_recorder
        self.timeline_widget = TimelineWidget()
        self.curve_editor = KeyframeCurveEditor()
        self.property_editor = QTreeWidget()
        self.preview_widget = QWidget()

        self.current_joint = ""
        self.keyframe_cache = {}

        self.init_ui()
        self.connect_signals()

        if animation_recorder:
            self.load_animation_data()

    def init_ui(self):
        """Инициализирует интерфейс"""
        self.setWindowTitle("Mocap Pro - Timeline Editor")
        self.setGeometry(100, 100, 1400, 800)

        # Центральный виджет с разделителями
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Панель инструментов
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)

        # Основная область с разделителями
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Верхняя часть: таймлайн и свойства
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(self.timeline_widget)
        top_splitter.addWidget(self.property_editor)
        top_splitter.setSizes([800, 400])

        # Нижняя часть: редактор кривых и предпросмотр
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(self.curve_editor)
        bottom_splitter.addWidget(self.preview_widget)
        bottom_splitter.setSizes([700, 300])

        splitter.addWidget(top_splitter)
        splitter.addWidget(bottom_splitter)
        splitter.setSizes([500, 300])

        main_layout.addWidget(splitter)

        # Статус бар
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status_bar()

        # Настройка редактора свойств
        self.setup_property_editor()

        # Настройка виджета предпросмотра
        self.setup_preview_widget()

    def create_toolbar(self) -> QToolBar:
        """Создает панель инструментов"""
        toolbar = QToolBar("Timeline Tools")

        # Кнопки управления воспроизведением
        play_action = QAction("▶ Play", self)
        play_action.triggered.connect(self.play_animation)
        toolbar.addAction(play_action)

        pause_action = QAction("⏸ Pause", self)
        pause_action.triggered.connect(self.pause_animation)
        toolbar.addAction(pause_action)

        stop_action = QAction("⏹ Stop", self)
        stop_action.triggered.connect(self.stop_animation)
        toolbar.addAction(stop_action)

        toolbar.addSeparator()

        # Инструменты редактирования
        tools_group = QActionGroup(self)

        select_tool = QAction(QIcon(), "Select", self)
        select_tool.setCheckable(True)
        select_tool.setChecked(True)
        select_tool.triggered.connect(lambda: self.set_edit_mode(EditMode.SELECT))
        toolbar.addAction(select_tool)
        tools_group.addAction(select_tool)

        cut_tool = QAction(QIcon(), "Cut", self)
        cut_tool.setCheckable(True)
        cut_tool.triggered.connect(lambda: self.set_edit_mode(EditMode.CUT))
        toolbar.addAction(cut_tool)
        tools_group.addAction(cut_tool)

        trim_tool = QAction(QIcon(), "Trim", self)
        trim_tool.setCheckable(True)
        trim_tool.triggered.connect(lambda: self.set_edit_mode(EditMode.TRIM))
        toolbar.addAction(trim_tool)
        tools_group.addAction(trim_tool)

        keyframe_tool = QAction(QIcon(), "Keyframe", self)
        keyframe_tool.setCheckable(True)
        keyframe_tool.triggered.connect(lambda: self.set_edit_mode(EditMode.KEYFRAME))
        toolbar.addAction(keyframe_tool)
        tools_group.addAction(keyframe_tool)

        toolbar.addSeparator()

        # Дополнительные инструменты
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)

        fit_action = QAction("Fit to View", self)
        fit_action.triggered.connect(self.fit_to_view)
        toolbar.addAction(fit_action)

        return toolbar

    def setup_property_editor(self):
        """Настраивает редактор свойств"""
        self.property_editor.setHeaderLabels(["Property", "Value", "Type"])
        self.property_editor.setColumnWidth(0, 200)
        self.property_editor.setColumnWidth(1, 150)

        # Контекстное меню
        self.property_editor.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.property_editor.customContextMenuRequested.connect(
            self.show_property_context_menu
        )

    def setup_preview_widget(self):
        """Настраивает виджет предпросмотра"""
        layout = QVBoxLayout(self.preview_widget)

        # Метка предпросмотра
        preview_label = QLabel("Animation Preview")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(preview_label)

        # Виджет для 3D предпросмотра (заглушка)
        preview_area = QOpenGLWidget()
        preview_area.setMinimumSize(300, 200)
        layout.addWidget(preview_area)

        # Панель управления предпросмотром
        preview_controls = QHBoxLayout()

        loop_checkbox = QCheckBox("Loop Playback")
        loop_checkbox.setChecked(True)
        preview_controls.addWidget(loop_checkbox)

        speed_slider = QSlider(Qt.Orientation.Horizontal)
        speed_slider.setRange(10, 200)
        speed_slider.setValue(100)
        speed_slider.setToolTip("Playback Speed (%)")
        preview_controls.addWidget(speed_slider)

        layout.addLayout(preview_controls)

    def connect_signals(self):
        """Подключает сигналы"""
        # Таймлайн
        self.timeline_widget.time_changed.connect(self.on_time_changed)
        self.timeline_widget.selection_changed.connect(self.on_selection_changed)

        # Редактор кривых
        self.curve_editor.curve_updated.connect(self.on_curve_updated)

        # Редактор свойств
        self.property_editor.itemChanged.connect(self.on_property_changed)

    def load_animation_data(self):
        """Загружает данные анимации из рекордера"""
        if not self.animation_recorder:
            return

        # Создаем сегмент анимации на таймлайне
        skeleton_track = self.timeline_widget.get_track_by_id("skeleton_anim")
        if skeleton_track and self.animation_recorder.animation_layers:
            # Берем первый слой для примера
            main_layer = self.animation_recorder.animation_layers[0]

            segment = TimelineSegment(
                id=f"anim_{main_layer.name}",
                name=main_layer.name,
                start_time=0,
                duration=main_layer.duration,
                data_ref=main_layer
            )

            skeleton_track.add_segment(segment)

            # Обновляем длительность таймлайна
            self.timeline_widget.duration = max(
                self.timeline_widget.duration,
                main_layer.duration
            )

            # Кэшируем ключевые кадры
            self.cache_keyframes(main_layer)

            # Загружаем свойства в редактор
            self.load_properties(main_layer)

            self.update_status_bar()
            self.timeline_widget.update()

    def cache_keyframes(self, animation_layer: AnimationLayer):
        """Кэширует ключевые кадры для быстрого доступа"""
        self.keyframe_cache.clear()

        for joint_name, keyframes in animation_layer.keyframes.items():
            self.keyframe_cache[joint_name] = keyframes

    def load_properties(self, animation_layer: AnimationLayer):
        """Загружает свойства в редактор"""
        self.property_editor.clear()

        # Корневой элемент
        root_item = QTreeWidgetItem(self.property_editor, ["Animation", animation_layer.name, "Layer"])

        # Основные свойства
        basic_props = QTreeWidgetItem(root_item, ["Basic Properties", "", ""])

        duration_item = QTreeWidgetItem(basic_props, ["Duration", f"{animation_layer.duration:.2f}s", "float"])
        duration_item.setFlags(duration_item.flags() | Qt.ItemFlag.ItemIsEditable)

        fps_item = QTreeWidgetItem(basic_props, ["Frame Rate", str(animation_layer.frame_rate), "int"])
        fps_item.setFlags(fps_item.flags() | Qt.ItemFlag.ItemIsEditable)

        joints_item = QTreeWidgetItem(basic_props, ["Joints", str(len(animation_layer.keyframes)), "int"])

        # Свойства сжатия
        compression_item = QTreeWidgetItem(root_item, ["Compression", "", ""])

        if animation_layer.compression_settings:
            for key, value in animation_layer.compression_settings.items():
                comp_item = QTreeWidgetItem(compression_item, [key, str(value), type(value).__name__])
                comp_item.setFlags(comp_item.flags() | Qt.ItemFlag.ItemIsEditable)

        # Разворачиваем
        self.property_editor.expandAll()

    def on_time_changed(self, time_sec: float):
        """Обработка изменения времени"""
        # Обновляем редактор кривых для текущего времени
        if self.current_joint and self.current_joint in self.keyframe_cache:
            keyframes = self.keyframe_cache[self.current_joint]

            # Находим ближайшие ключевые кадры
            for i, kf in enumerate(keyframes):
                if abs(kf.timestamp - time_sec) < 0.033:  # ~1 кадр при 30fps
                    # Показываем информацию о ключевом кадре в статус баре
                    self.status_bar.showMessage(
                        f"Keyframe at {kf.timestamp:.2f}s - "
                        f"Position: {kf.position} - Rotation: {kf.rotation}"
                    )
                    break

        # TODO: Обновить предпросмотр анимации

    def on_selection_changed(self, segments: List[TimelineSegment]):
        """Обработка изменения выделения"""
        if not segments:
            return

        # Для первого выделенного сегмента
        segment = segments[0]

        # Если это сегмент анимации, загружаем его ключевые кадры
        if (segment.parent_track and
                segment.parent_track.type == TrackType.SKELETON_ANIMATION):

            if isinstance(segment.data_ref, AnimationLayer):
                self.cache_keyframes(segment.data_ref)
                self.load_properties(segment.data_ref)

                # Выбираем первый сустав для отображения кривых
                if segment.data_ref.keyframes:
                    self.current_joint = list(segment.data_ref.keyframes.keys())[0]
                    self.curve_editor.set_data(
                        self.current_joint,
                        segment.data_ref.keyframes[self.current_joint]
                    )

    def on_curve_updated(self, joint_name: str, keyframes: List[Keyframe]):
        """Обработка обновления кривой"""
        if self.animation_recorder and self.animation_recorder.animation_layers:
            # Обновляем ключевые кадры в текущем слое
            for layer in self.animation_recorder.animation_layers:
                if joint_name in layer.keyframes:
                    layer.keyframes[joint_name] = keyframes
                    break

        # Обновляем кэш
        self.keyframe_cache[joint_name] = keyframes

    def on_property_changed(self, item: QTreeWidgetItem, column: int):
        """Обработка изменения свойства"""
        if column != 1:  # Только колонка значений
            return

        prop_name = item.text(0)
        prop_value = item.text(1)
        prop_type = item.text(2)

        # Преобразуем значение к правильному типу
        try:
            if prop_type == "int":
                value = int(prop_value)
            elif prop_type == "float":
                value = float(prop_value)
            elif prop_type == "bool":
                value = prop_value.lower() in ("true", "1", "yes")
            else:
                value = prop_value

            # TODO: Применить изменение к анимации
            print(f"Property changed: {prop_name} = {value}")

        except ValueError:
            item.setText(1, "Invalid value")

    def show_property_context_menu(self, pos: QPoint):
        """Показывает контекстное меню редактора свойств"""
        menu = QMenu(self)

        # Получаем выделенный элемент
        item = self.property_editor.itemAt(pos)
        if not item:
            return

        # Действия в зависимости от типа элемента
        if item.parent() is None:
            # Корневой элемент
            add_prop_action = menu.addAction("Add Property")
            add_prop_action.triggered.connect(self.add_custom_property)

        elif item.text(2):  # Есть тип данных - редактируемое свойство
            reset_action = menu.addAction("Reset to Default")
            reset_action.triggered.connect(
                lambda: self.reset_property(item)
            )

        menu.exec(self.property_editor.mapToGlobal(pos))

    def add_custom_property(self):
        """Добавляет пользовательское свойство"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Custom Property")

        layout = QFormLayout(dialog)

        name_edit = QLineEdit()
        value_edit = QLineEdit()
        type_combo = QComboBox()
        type_combo.addItems(["str", "int", "float", "bool", "list", "dict"])

        layout.addRow("Name:", name_edit)
        layout.addRow("Value:", value_edit)
        layout.addRow("Type:", type_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Добавляем новое свойство
            parent_item = self.property_editor.currentItem() or self.property_editor.topLevelItem(0)
            if parent_item:
                new_item = QTreeWidgetItem(parent_item, [
                    name_edit.text(),
                    value_edit.text(),
                    type_combo.currentText()
                ])
                new_item.setFlags(new_item.flags() | Qt.ItemFlag.ItemIsEditable)

    def reset_property(self, item: QTreeWidgetItem):
        """Сбрасывает свойство к значению по умолчанию"""
        # TODO: Реализовать сброс к значениям по умолчанию
        item.setText(1, "Default")

    def set_edit_mode(self, mode: EditMode):
        """Устанавливает режим редактирования"""
        self.timeline_widget.edit_mode = mode
        self.timeline_widget.update()

    def play_animation(self):
        """Запускает воспроизведение анимации"""
        self.status_bar.showMessage("Playing animation...")
        # TODO: Реализовать воспроизведение

    def pause_animation(self):
        """Приостанавливает воспроизведение"""
        self.status_bar.showMessage("Animation paused")
        # TODO: Реализовать паузу

    def stop_animation(self):
        """Останавливает воспроизведение"""
        self.status_bar.showMessage("Animation stopped")
        # TODO: Реализовать остановку

    def zoom_in(self):
        """Увеличивает масштаб временной шкалы"""
        self.timeline_widget.time_scale *= 1.2
        self.timeline_widget.update()

    def zoom_out(self):
        """Уменьшает масштаб временной шкалы"""
        self.timeline_widget.time_scale /= 1.2
        self.timeline_widget.update()

    def fit_to_view(self):
        """Подгоняет анимацию по ширине окна"""
        if self.timeline_widget.duration > 0:
            available_width = self.timeline_widget.width() - self.timeline_widget.track_header_width
            self.timeline_widget.time_scale = available_width / self.timeline_widget.duration
            self.timeline_widget.update()

    def update_status_bar(self):
        """Обновляет статус бар"""
        stats = [
            f"Tracks: {len(self.timeline_widget.tracks)}",
            f"Duration: {self.timeline_widget.duration:.1f}s",
            f"Scale: {self.timeline_widget.time_scale:.1f} px/s"
        ]

        if self.animation_recorder:
            stats.append(f"Layers: {len(self.animation_recorder.animation_layers)}")

        self.status_bar.showMessage(" | ".join(stats))

    def save_timeline(self, filepath: str):
        """Сохраняет проект временной шкалы"""
        project_data = {
            "version": "1.0",
            "duration": self.timeline_widget.duration,
            "tracks": [],
            "current_time": self.timeline_widget.current_time,
            "time_scale": self.timeline_widget.time_scale
        }

        # Сохраняем треки
        for track in self.timeline_widget.tracks:
            track_data = {
                "id": track.id,
                "name": track.name,
                "type": track.type.value,
                "color": [
                    track.color.red(),
                    track.color.green(),
                    track.color.blue(),
                    track.color.alpha()
                ],
                "visible": track.visible,
                "locked": track.locked,
                "height": track.height,
                "segments": []
            }

            # Сохраняем сегменты
            for segment in track.segments:
                segment_data = {
                    "id": segment.id,
                    "name": segment.name,
                    "start_time": segment.start_time,
                    "duration": segment.duration,
                    "properties": segment.properties
                }
                track_data["segments"].append(segment_data)

            project_data["tracks"].append(track_data)

        # Сохраняем в файл
        with open(filepath, 'w') as f:
            json.dump(project_data, f, indent=2)

        self.status_bar.showMessage(f"Project saved to {filepath}")

    def load_timeline(self, filepath: str):
        """Загружает проект временной шкалы"""
        try:
            with open(filepath, 'r') as f:
                project_data = json.load(f)

            # Очищаем текущие треки
            self.timeline_widget.tracks.clear()

            # Загружаем треки
            for track_data in project_data.get("tracks", []):
                track = TimelineTrack(
                    id=track_data["id"],
                    name=track_data["name"],
                    type=TrackType(track_data["type"]),
                    color=QColor(*track_data["color"]),
                    visible=track_data.get("visible", True),
                    locked=track_data.get("locked", False),
                    height=track_data.get("height", 60)
                )

                # Загружаем сегменты
                for segment_data in track_data.get("segments", []):
                    segment = TimelineSegment(
                        id=segment_data["id"],
                        name=segment_data["name"],
                        start_time=segment_data["start_time"],
                        duration=segment_data["duration"],
                        data_ref=None,  # TODO: Восстановить ссылки на данные
                        properties=segment_data.get("properties", {})
                    )
                    track.add_segment(segment)

                self.timeline_widget.add_track(track)

            # Восстанавливаем состояние
            self.timeline_widget.duration = project_data.get("duration", 60.0)
            self.timeline_widget.time_scale = project_data.get("time_scale", 100.0)
            self.timeline_widget.set_current_time(project_data.get("current_time", 0.0))

            self.update_status_bar()
            self.status_bar.showMessage(f"Project loaded from {filepath}")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load project: {str(e)}")


# Экспортируемые функции для интеграции
def create_timeline_editor(animation_recorder=None, parent=None) -> TimelineEditor:
    """
    Создает и возвращает экземпляр редактора временной шкалы.

    Args:
        animation_recorder: Экземпляр AnimationRecorder для загрузки данных
        parent: Родительский виджет

    Returns:
        TimelineEditor: Экземпляр редактора временной шкалы
    """
    editor = TimelineEditor(animation_recorder)
    if parent:
        editor.setParent(parent)
    return editor


def integrate_with_main_window(main_window, animation_recorder):
    """
    Интегрирует редактор временной шкалы с главным окном.

    Args:
        main_window: Главное окно приложения
        animation_recorder: Модуль записи анимации
    """
    editor = create_timeline_editor(animation_recorder)

    # Создаем док-виджет
    dock_widget = QDockWidget("Timeline Editor", main_window)
    dock_widget.setWidget(editor)
    dock_widget.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable
    )

    # Добавляем в главное окно
    main_window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_widget)

    return editor


# Точка входа для тестирования
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Тестовый редактор
    editor = TimelineEditor()
    editor.show()

    sys.exit(app.exec())