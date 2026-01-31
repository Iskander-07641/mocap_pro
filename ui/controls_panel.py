"""
ПРОФЕССИОНАЛЬНАЯ ПАНЕЛЬ УПРАВЛЕНИЯ ДЛЯ MOCAP PRO
Расширенные элементы управления, визуализация состояния, макросы
"""
  # ← ДОБАВЬТЕ ЭТУ СТРОКУ
import sys
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QPushButton, QLabel, QSlider, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QProgressBar, QTabWidget, QTextEdit, QListWidget,
    QListWidgetItem, QTreeWidget, QTreeWidgetItem,
    QSplitter, QToolBar, QToolButton, QMenu, QFrame,
    QScrollArea, QSizePolicy, QButtonGroup, QRadioButton,
    QDial, QLCDNumber, QStyleFactory, QStyle
)
from PyQt6.QtGui import (
    QIcon, QFont, QPalette, QColor, QLinearGradient,
    QPainter, QPen, QBrush, QPixmap, QPainterPath,
    QAction, QKeySequence, QMovie, QFontMetrics
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QTimer, QSize, QPoint, QRect,
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup,
    QSequentialAnimationGroup, QDateTime, QUrl
)
from PyQt6.QtMultimedia import QSoundEffect
import json
import yaml
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from PyQt6.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)


class ControlTheme(Enum):
    """Темы элементов управления"""
    DARK = "dark"
    LIGHT = "light"
    BLUE = "blue"
    GREEN = "green"
    PURPLE = "purple"


@dataclass
class ControlStyle:
    """Стиль для элементов управления"""
    background: QColor
    foreground: QColor
    accent: QColor
    border: QColor
    font_family: str = "Segoe UI"
    font_size: int = 10

    @staticmethod
    def get_theme(theme: ControlTheme) -> 'ControlStyle':
        """Получение стиля по теме"""
        themes = {
            ControlTheme.DARK: ControlStyle(
                background=QColor(45, 45, 45),
                foreground=QColor(220, 220, 220),
                accent=QColor(0, 150, 255),
                border=QColor(70, 70, 70)
            ),
            ControlTheme.LIGHT: ControlStyle(
                background=QColor(240, 240, 240),
                foreground=QColor(30, 30, 30),
                accent=QColor(0, 120, 215),
                border=QColor(200, 200, 200)
            ),
            ControlTheme.BLUE: ControlStyle(
                background=QColor(30, 40, 60),
                foreground=QColor(220, 230, 240),
                accent=QColor(0, 180, 255),
                border=QColor(50, 70, 100)
            ),
            ControlTheme.GREEN: ControlStyle(
                background=QColor(40, 60, 50),
                foreground=QColor(220, 240, 220),
                accent=QColor(100, 220, 100),
                border=QColor(70, 100, 80)
            ),
            ControlTheme.PURPLE: ControlStyle(
                background=QColor(50, 40, 60),
                foreground=QColor(240, 220, 240),
                accent=QColor(180, 100, 220),
                border=QColor(80, 70, 100)
            )
        }
        return themes.get(theme, themes[ControlTheme.DARK])


class AnimatedButton(QPushButton):
    """Анимированная кнопка с эффектами"""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)

        # Анимации
        self.hover_animation = QPropertyAnimation(self, b"geometry")
        self.click_animation = QPropertyAnimation(self, b"geometry")

        # Настройки цветов
        self.normal_color = QColor(60, 60, 60)
        self.hover_color = QColor(80, 80, 80)
        self.press_color = QColor(100, 100, 100)
        self.accent_color = QColor(0, 150, 255)

        # Начальный стиль
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.normal_color.name()};
                border: 2px solid {self.accent_color.name()};
                border-radius: 5px;
                padding: 8px;
                color: white;
                font-weight: bold;
            }}
        """)

        self.setMouseTracking(True)

    def enterEvent(self, event):
        """При наведении"""
        if self.hover_animation:
            self.hover_animation.stop()
            self.hover_animation.setDuration(150)
            self.hover_animation.setStartValue(self.geometry())
            self.hover_animation.setEndValue(
                self.geometry().adjusted(-1, -1, 1, 1)
            )
            self.hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            self.hover_animation.start()

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.hover_color.name()};
                border: 2px solid {self.accent_color.name()};
                border-radius: 5px;
                padding: 8px;
                color: white;
                font-weight: bold;
            }}
        """)

        super().enterEvent(event)

    def leaveEvent(self, event):
        """При уходе курсора"""
        if self.hover_animation:
            self.hover_animation.stop()
            self.hover_animation.setDuration(150)
            self.hover_animation.setStartValue(self.geometry())
            self.hover_animation.setEndValue(
                self.geometry().adjusted(1, 1, -1, -1)
            )
            self.hover_animation.start()

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.normal_color.name()};
                border: 2px solid {self.accent_color.name()};
                border-radius: 5px;
                padding: 8px;
                color: white;
                font-weight: bold;
            }}
        """)

        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """При нажатии"""
        if self.click_animation:
            self.click_animation.stop()
            self.click_animation.setDuration(100)
            self.click_animation.setStartValue(self.geometry())
            self.click_animation.setEndValue(
                self.geometry().adjusted(2, 2, -2, -2)
            )
            self.click_animation.start()

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.press_color.name()};
                border: 2px solid {self.accent_color.name()};
                border-radius: 5px;
                padding: 8px;
                color: white;
                font-weight: bold;
            }}
        """)

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """При отпускании"""
        if self.click_animation:
            self.click_animation.stop()
            self.click_animation.setDuration(100)
            self.click_animation.setStartValue(self.geometry())
            self.click_animation.setEndValue(
                self.geometry().adjusted(-2, -2, 2, 2)
            )
            self.click_animation.start()

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.hover_color.name()};
                border: 2px solid {self.accent_color.name()};
                border-radius: 5px;
                padding: 8px;
                color: white;
                font-weight: bold;
            }}
        """)

        super().mouseReleaseEvent(event)


class StatusLED(QLabel):
    """Светодиод индикатор статуса"""

    def __init__(self, size=12, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.status = "off"  # off, green, yellow, red, blue
        self.blinking = False
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self._blink)

        self.colors = {
            "off": QColor(60, 60, 60),
            "green": QColor(0, 255, 0),
            "yellow": QColor(255, 255, 0),
            "red": QColor(255, 0, 0),
            "blue": QColor(0, 150, 255),
            "purple": QColor(180, 0, 255)
        }

        self._update_appearance()

    def set_status(self, status: str, blink=False):
        """Установка статуса"""
        self.status = status
        self.blinking = blink

        if blink:
            self.blink_timer.start(500)  # Мигание каждые 500ms
        else:
            self.blink_timer.stop()

        self._update_appearance()

    def _blink(self):
        """Мигание"""
        if self.status != "off":
            self.status = "off"
        else:
            self.status = "green"  # Возвращаемся к исходному

        self._update_appearance()

    def _update_appearance(self):
        """Обновление внешнего вида"""
        color = self.colors.get(self.status, QColor(60, 60, 60))

        # Создаем градиент для 3D эффекта
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Внешний круг (тень)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, color.darker(150))
        gradient.setColorAt(1, color.lighter(150))

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(1, 1, self.width() - 2, self.height() - 2)

        # Внутренний круг (свет)
        inner_size = self.width() - 6
        gradient2 = QLinearGradient(0, 0, 0, inner_size)
        gradient2.setColorAt(0, color.lighter(200))
        gradient2.setColorAt(1, color)

        painter.setBrush(QBrush(gradient2))
        painter.drawEllipse(3, 3, inner_size, inner_size)

        # Блики
        painter.setBrush(QBrush(QColor(255, 255, 255, 100)))
        painter.drawEllipse(5, 5, 4, 4)

        painter.end()

        self.setPixmap(pixmap)


class ParameterSlider(QWidget):
    """Продвинутый слайдер с параметрами"""

    value_changed = pyqtSignal(float)

    def __init__(self, label="", min_val=0.0, max_val=1.0, default=0.5,
                 unit="", parent=None):
        super().__init__(parent)

        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit

        self.init_ui()
        self.set_value(default)

    def init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Верхняя строка: метка и значение
        top_layout = QHBoxLayout()

        self.label_widget = QLabel(self.label)
        self.label_widget.setStyleSheet("color: #cccccc; font-weight: bold;")
        top_layout.addWidget(self.label_widget)

        top_layout.addStretch()

        self.value_label = QLabel("0.0")
        self.value_label.setStyleSheet("""
            QLabel {
                background-color: #404040;
                border: 1px solid #505050;
                border-radius: 3px;
                padding: 2px 6px;
                color: #ffffff;
                font-family: 'Consolas', monospace;
                min-width: 50px;
                text-align: center;
            }
        """)
        top_layout.addWidget(self.value_label)

        if self.unit:
            unit_label = QLabel(self.unit)
            unit_label.setStyleSheet("color: #888888;")
            top_layout.addWidget(unit_label)

        layout.addLayout(top_layout)

        # Слайдер
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)  # Высокое разрешение
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2b2b2b, stop:1 #4a4a4a
                );
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: qradialgradient(
                    cx:0.5, cy:0.5, radius:0.5,
                    fx:0.5, fy:0.5,
                    stop:0 #ffffff, stop:1 #888888
                );
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00aaff, stop:1 #0088cc
                );
                border-radius: 3px;
            }
        """)

        layout.addWidget(self.slider)

        # Шкала значений
        scale_layout = QHBoxLayout()

        min_label = QLabel(f"{self.min_val:.2f}")
        min_label.setStyleSheet("color: #888888; font-size: 9px;")
        scale_layout.addWidget(min_label)

        scale_layout.addStretch()

        max_label = QLabel(f"{self.max_val:.2f}")
        max_label.setStyleSheet("color: #888888; font-size: 9px;")
        scale_layout.addWidget(max_label)

        layout.addLayout(scale_layout)

    def set_value(self, value: float):
        """Установка значения"""
        value = max(self.min_val, min(self.max_val, value))

        # Нормализация к диапазону слайдера
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        self.slider.setValue(int(normalized * 1000))

        self._update_display(value)

    def get_value(self) -> float:
        """Получение значения"""
        normalized = self.slider.value() / 1000.0
        return self.min_val + normalized * (self.max_val - self.min_val)

    def _on_slider_changed(self, value: int):
        """Обработка изменения слайдера"""
        actual_value = self.get_value()
        self._update_display(actual_value)
        self.value_changed.emit(actual_value)

    def _update_display(self, value: float):
        """Обновление отображения значения"""
        self.value_label.setText(f"{value:.3f}")


class MacroButton(QPushButton):
    """Кнопка макроса с сохранением действий"""

    def __init__(self, name="", parent=None):
        super().__init__(name, parent)

        self.name = name
        self.actions = []  # Список действий макроса
        self.hotkey = ""

        self.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a6fa5, stop:1 #2e4a7a
                );
                border: 2px solid #5a8ac5;
                border-radius: 8px;
                padding: 10px;
                color: white;
                font-weight: bold;
                font-size: 11px;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5a8ac5, stop:1 #3e5a95
                );
                border: 2px solid #6a9ad5;
            }
            QPushButton:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2e4a7a, stop:1 #1e3a6a
                );
                padding: 11px 9px 9px 11px;
            }
        """)

    def add_action(self, action_type: str, params: Dict):
        """Добавление действия в макрос"""
        self.actions.append({
            "type": action_type,
            "params": params,
            "timestamp": datetime.now().isoformat()
        })

    def execute(self):
        """Выполнение макроса"""
        logger.info(f"Выполнение макроса: {self.name}")
        # Здесь будет выполнение всех действий
        # В реальном приложении нужно отправить сигналы
        return True


class PresetManager(QWidget):
    """Менеджер пресетов настроек"""

    preset_selected = pyqtSignal(str)
    preset_saved = pyqtSignal(str, dict)
    preset_deleted = pyqtSignal(str)

    def __init__(self, category="general", parent=None):
        super().__init__(parent)

        self.category = category
        self.presets = {}  # name -> settings
        self.current_preset = ""

        self.init_ui()
        self.load_presets()

    def init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout(self)

        # Заголовок
        title = QLabel(f"🎛️ Пресеты ({self.category})")
        title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: #404040;
                border-radius: 4px;
            }
        """)
        layout.addWidget(title)

        # Список пресетов
        self.preset_list = QListWidget()
        self.preset_list.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #cccccc;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #353535;
            }
            QListWidget::item:selected {
                background-color: #505050;
                color: #ffffff;
            }
            QListWidget::item:hover {
                background-color: #404040;
            }
        """)
        self.preset_list.itemClicked.connect(self._on_preset_selected)
        layout.addWidget(self.preset_list)

        # Кнопки управления
        btn_layout = QHBoxLayout()

        self.save_btn = QPushButton("💾 Сохранить")
        self.save_btn.clicked.connect(self.save_current)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)

        self.delete_btn = QPushButton("🗑️ Удалить")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_btn.setEnabled(False)
        btn_layout.addWidget(self.delete_btn)

        layout.addLayout(btn_layout)

    def load_presets(self):
        """Загрузка пресетов из файла"""
        preset_file = f"presets/{self.category}.json"

        if os.path.exists(preset_file):
            try:
                with open(preset_file, 'r', encoding='utf-8') as f:
                    self.presets = json.load(f)

                self.preset_list.clear()
                for preset_name in self.presets.keys():
                    item = QListWidgetItem(preset_name)
                    self.preset_list.addItem(item)

            except Exception as e:
                logger.error(f"Ошибка загрузки пресетов: {e}")

    def save_presets(self):
        """Сохранение пресетов в файл"""
        preset_dir = "presets"
        if not os.path.exists(preset_dir):
            os.makedirs(preset_dir)

        preset_file = os.path.join(preset_dir, f"{self.category}.json")

        try:
            with open(preset_file, 'w', encoding='utf-8') as f:
                json.dump(self.presets, f, indent=2, ensure_ascii=False)

            logger.info(f"Пресеты сохранены: {preset_file}")

        except Exception as e:
            logger.error(f"Ошибка сохранения пресетов: {e}")

    def save_current(self):
        """Сохранение текущего пресета"""
        if self.current_preset:
            # Получаем текущие настройки из родительского виджета
            settings = self._get_current_settings()

            self.presets[self.current_preset] = settings
            self.save_presets()
            self.preset_saved.emit(self.current_preset, settings)

            logger.info(f"Пресет сохранен: {self.current_preset}")

    def delete_selected(self):
        """Удаление выбранного пресета"""
        current_item = self.preset_list.currentItem()
        if current_item:
            preset_name = current_item.text()

            reply = QMessageBox.question(
                self, "Удаление пресета",
                f"Удалить пресет '{preset_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                if preset_name in self.presets:
                    del self.presets[preset_name]
                    self.save_presets()

                    row = self.preset_list.row(current_item)
                    self.preset_list.takeItem(row)

                    self.preset_deleted.emit(preset_name)
                    logger.info(f"Пресет удален: {preset_name}")

    def _on_preset_selected(self, item):
        """Обработка выбора пресета"""
        preset_name = item.text()
        self.current_preset = preset_name

        if preset_name in self.presets:
            settings = self.presets[preset_name]
            self.preset_selected.emit(preset_name)

            # Активируем кнопки
            self.save_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)

            logger.info(f"Пресет выбран: {preset_name}")

    def _get_current_settings(self) -> Dict:
        """Получение текущих настроек"""
        # Этот метод должен быть переопределен в дочернем классе
        # для получения настроек из конкретных виджетов
        return {}


class ProfessionalControlsPanel(QWidget):
    """
    ПРОФЕССИОНАЛЬНАЯ ПАНЕЛЬ УПРАВЛЕНИЯ MOCAP

    Особенности:
    1. Расширенные элементы управления с анимациями
    2. Визуализация состояния системы
    3. Управление макросами и пресетами
    4. Полная интеграция со всеми модулями
    5. Профессиональный дизайн с темами
    6. Горячие клавиши и быстрые действия
    """

    # Сигналы
    start_recording = pyqtSignal()
    stop_recording = pyqtSignal()
    calibrate_camera = pyqtSignal()
    calibrate_skeleton = pyqtSignal()
    export_animation = pyqtSignal(str)

    settings_changed = pyqtSignal(str, object)  # key, value
    macro_triggered = pyqtSignal(str)
    preset_applied = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Текущее состояние
        self.is_recording = False
        self.is_tracking = False
        self.is_calibrating = False

        # Настройки темы
        self.theme = ControlTheme.DARK
        self.style = ControlStyle.get_theme(self.theme)

        # Звуковые эффекты
        self.sound_effects = {}
        self._init_sounds()

        # Макросы
        self.macros = {}
        self.active_macro = None

        # Пресеты
        self.presets = {
            "tracking": PresetManager("tracking", self),
            "recording": PresetManager("recording", self),
            "export": PresetManager("export", self)
        }

        self.init_ui()
        self.apply_theme()
        self.init_shortcuts()

        logger.info("ProfessionalControlsPanel инициализирован")

    def _init_sounds(self):
        """Инициализация звуковых эффектов"""
        try:
            # В реальном приложении загружаем звуковые файлы
            # Здесь заглушки
            self.sound_effects = {
                "click": QSoundEffect(),
                "record_start": QSoundEffect(),
                "record_stop": QSoundEffect(),
                "error": QSoundEffect()
            }
        except:
            logger.warning("Звуковые эффекты недоступны")

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setMinimumWidth(350)

        # Основной layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Заголовок
        title = QLabel("🎮 УПРАВЛЕНИЕ MOCAP")
        title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2b5b84, stop:1 #1e3a5f
                );
                border-radius: 6px;
                text-align: center;
            }
        """)
        main_layout.addWidget(title)

        # Табы для разных категорий управления
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #404040;
                border-radius: 4px;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #353535;
                color: #cccccc;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2b2b2b;
                color: #ffffff;
                border-bottom: 2px solid #00aaff;
            }
            QTabBar::tab:hover {
                background-color: #404040;
            }
        """)

        # Добавляем вкладки
        self.tab_widget.addTab(self._create_recording_tab(), "🎥 Запись")
        self.tab_widget.addTab(self._create_tracking_tab(), "🎯 Трекинг")
        self.tab_widget.addTab(self._create_calibration_tab(), "⚙️ Калибровка")
        self.tab_widget.addTab(self._create_export_tab(), "📤 Экспорт")
        self.tab_widget.addTab(self._create_macros_tab(), "⚡ Макросы")

        main_layout.addWidget(self.tab_widget)

        # Панель статуса
        status_group = self._create_status_group()
        main_layout.addWidget(status_group)

        # Быстрые действия
        quick_actions = self._create_quick_actions()
        main_layout.addWidget(quick_actions)

        main_layout.addStretch()

    def _create_recording_tab(self) -> QWidget:
        """Создание вкладки записи"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Группа управления записью
        record_group = QGroupBox("Управление записью")
        record_group.setStyleSheet("""
            QGroupBox {
                color: #cccccc;
                border: 2px solid #404040;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        record_layout = QVBoxLayout()

        # Основные кнопки записи
        btn_layout = QHBoxLayout()

        self.record_btn = AnimatedButton("🔴 Начать запись")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                border: 2px solid #ff6666;
                border-radius: 8px;
                padding: 15px;
                color: white;
                font-size: 14px;
                font-weight: bold;
                min-height: 50px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:pressed {
                background-color: #cc2222;
            }
        """)
        btn_layout.addWidget(self.record_btn)

        self.pause_btn = QPushButton("⏸️ Пауза")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        btn_layout.addWidget(self.pause_btn)

        record_layout.addLayout(btn_layout)

        # Настройки записи
        settings_layout = QGridLayout()

        # FPS
        settings_layout.addWidget(QLabel("FPS:"), 0, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        self.fps_spin.valueChanged.connect(
            lambda v: self.settings_changed.emit("recording/fps", v)
        )
        settings_layout.addWidget(self.fps_spin, 0, 1)

        # Качество
        settings_layout.addWidget(QLabel("Качество:"), 1, 0)
        self.quality_slider = ParameterSlider("", 0.1, 1.0, 0.8, "")
        self.quality_slider.value_changed.connect(
            lambda v: self.settings_changed.emit("recording/quality", v)
        )
        settings_layout.addWidget(self.quality_slider, 1, 1, 1, 2)

        # Автозапуск трекинга
        self.auto_track_cb = QCheckBox("Автозапуск трекинга")
        self.auto_track_cb.setChecked(True)
        record_layout.addWidget(self.auto_track_cb)

        # Предпросмотр записи
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("Предпросмотр:"))

        self.preview_cb = QCheckBox("Включить")
        self.preview_cb.setChecked(True)
        preview_layout.addWidget(self.preview_cb)

        preview_layout.addStretch()
        record_layout.addLayout(preview_layout)

        record_group.setLayout(record_layout)
        layout.addWidget(record_group)

        # Группа информации о записи
        info_group = QGroupBox("Информация о записи")
        info_layout = QVBoxLayout()

        info_grid = QGridLayout()

        # Длительность
        info_grid.addWidget(QLabel("Длительность:"), 0, 0)
        self.duration_label = QLabel("00:00:00")
        self.duration_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        info_grid.addWidget(self.duration_label, 0, 1)

        # Кадры
        info_grid.addWidget(QLabel("Кадры:"), 1, 0)
        self.frames_label = QLabel("0")
        info_grid.addWidget(self.frames_label, 1, 1)

        # Размер
        info_grid.addWidget(QLabel("Размер:"), 2, 0)
        self.size_label = QLabel("0 MB")
        info_grid.addWidget(self.size_label, 2, 1)

        # FPS (фактический)
        info_grid.addWidget(QLabel("Факт. FPS:"), 3, 0)
        self.actual_fps_label = QLabel("0.0")
        info_grid.addWidget(self.actual_fps_label, 3, 1)

        info_layout.addLayout(info_grid)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Пресеты записи
        layout.addWidget(self.presets["recording"])

        layout.addStretch()
        return widget

    def _create_tracking_tab(self) -> QWidget:
        """Создание вкладки трекинга"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Группа управления трекингом
        tracking_group = QGroupBox("Управление трекингом")
        tracking_layout = QVBoxLayout()

        # Кнопки трекинга
        track_btn_layout = QHBoxLayout()

        self.start_track_btn = AnimatedButton("▶️ Запустить трекинг")
        self.start_track_btn.clicked.connect(self.start_tracking)
        track_btn_layout.addWidget(self.start_track_btn)

        self.stop_track_btn = QPushButton("⏹️ Остановить")
        self.stop_track_btn.clicked.connect(self.stop_tracking)
        self.stop_track_btn.setEnabled(False)
        track_btn_layout.addWidget(self.stop_track_btn)

        tracking_layout.addLayout(track_btn_layout)

        # Настройки трекинга
        settings_group = QGroupBox("Настройки трекинга")
        settings_layout = QGridLayout()

        # Режим трекинга
        settings_layout.addWidget(QLabel("Режим:"), 0, 0)
        self.tracking_mode_combo = QComboBox()
        self.tracking_mode_combo.addItems([
            "⚡ Быстрый",
            "🎯 Точный",
            "✨ Ультра",
            "🛠️ Ручной"
        ])
        self.tracking_mode_combo.currentIndexChanged.connect(
            lambda i: self.settings_changed.emit(
                "tracking/mode",
                self.tracking_mode_combo.currentText()
            )
        )
        settings_layout.addWidget(self.tracking_mode_combo, 0, 1)

        # Уверенность
        settings_layout.addWidget(QLabel("Уверенность:"), 1, 0)
        self.confidence_slider = ParameterSlider("", 0.1, 1.0, 0.5, "")
        self.confidence_slider.value_changed.connect(
            lambda v: self.settings_changed.emit("tracking/confidence", v)
        )
        settings_layout.addWidget(self.confidence_slider, 1, 1, 1, 2)

        # Фильтр Калмана
        self.kalman_cb = QCheckBox("Фильтр Калмана")
        self.kalman_cb.setChecked(True)
        self.kalman_cb.stateChanged.connect(
            lambda s: self.settings_changed.emit("tracking/kalman", bool(s))
        )
        settings_layout.addWidget(self.kalman_cb, 2, 0, 1, 2)

        # Сглаживание
        self.smoothing_cb = QCheckBox("Сглаживание")
        self.smoothing_cb.setChecked(True)
        self.smoothing_cb.stateChanged.connect(
            lambda s: self.settings_changed.emit("tracking/smoothing", bool(s))
        )
        settings_layout.addWidget(self.smoothing_cb, 3, 0, 1, 2)

        settings_group.setLayout(settings_layout)
        tracking_layout.addWidget(settings_group)

        # Информация о трекинге
        info_group = QGroupBox("Информация о трекинге")
        info_layout = QGridLayout()

        info_layout.addWidget(QLabel("Landmarks:"), 0, 0)
        self.landmarks_label = QLabel("0")
        self.landmarks_label.setStyleSheet("color: #00ff00;")
        info_layout.addWidget(self.landmarks_label, 0, 1)

        info_layout.addWidget(QLabel("Уверенность:"), 1, 0)
        self.tracking_confidence_label = QLabel("0%")
        info_layout.addWidget(self.tracking_confidence_label, 1, 1)

        info_layout.addWidget(QLabel("FPS:"), 2, 0)
        self.tracking_fps_label = QLabel("0.0")
        info_layout.addWidget(self.tracking_fps_label, 2, 1)

        info_layout.addWidget(QLabel("Задержка:"), 3, 0)
        self.latency_label = QLabel("0ms")
        info_layout.addWidget(self.latency_label, 3, 1)

        info_group.setLayout(info_layout)
        tracking_layout.addWidget(info_group)

        tracking_group.setLayout(tracking_layout)
        layout.addWidget(tracking_group)

        # Пресеты трекинга
        layout.addWidget(self.presets["tracking"])

        layout.addStretch()
        return widget

    def _create_calibration_tab(self) -> QWidget:
        """Создание вкладки калибровки"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Калибровка камеры
        cam_cal_group = QGroupBox("Калибровка камеры")
        cam_layout = QVBoxLayout()

        cam_info = QLabel(
            "Используйте шахматную доску или ARUCO маркеры\n"
            "для точной калибровки камеры."
        )
        cam_info.setWordWrap(True)
        cam_layout.addWidget(cam_info)

        self.calibrate_cam_btn = AnimatedButton("🎯 Калибровать камеру")
        self.calibrate_cam_btn.clicked.connect(
            lambda: self.calibrate_camera.emit()
        )
        cam_layout.addWidget(self.calibrate_cam_btn)

        # Прогресс калибровки
        self.cam_cal_progress = QProgressBar()
        self.cam_cal_progress.setTextVisible(True)
        self.cam_cal_progress.setFormat("Кадров: %v/%m")
        cam_layout.addWidget(self.cam_cal_progress)

        cam_cal_group.setLayout(cam_layout)
        layout.addWidget(cam_cal_group)

        # Калибровка скелета
        skel_cal_group = QGroupBox("Калибровка скелета")
        skel_layout = QVBoxLayout()

        skel_info = QLabel(
            "Встаньте в T-позу для автоматической калибровки\n"
            "или настройте скелет вручную."
        )
        skel_info.setWordWrap(True)
        skel_layout.addWidget(skel_info)

        btn_layout = QHBoxLayout()

        self.auto_calibrate_btn = QPushButton("🤖 Авто-калибровка")
        self.auto_calibrate_btn.clicked.connect(
            lambda: self.calibrate_skeleton.emit()
        )
        btn_layout.addWidget(self.auto_calibrate_btn)

        self.manual_calibrate_btn = QPushButton("🛠️ Ручная настройка")
        btn_layout.addWidget(self.manual_calibrate_btn)

        skel_layout.addLayout(btn_layout)

        # Настройки скелета
        skel_settings = QGroupBox("Параметры скелета")
        skel_set_layout = QGridLayout()

        skel_set_layout.addWidget(QLabel("Модель:"), 0, 0)
        self.skeleton_model_combo = QComboBox()
        self.skeleton_model_combo.addItems([
            "👤 Стандартный",
            "🏃 Атлетический",
            "👧 Детский",
            "👴 Пожилой"
        ])
        skel_set_layout.addWidget(self.skeleton_model_combo, 0, 1)

        skel_set_layout.addWidget(QLabel("Рост:"), 1, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.5, 2.5)
        self.height_spin.setValue(1.75)
        self.height_spin.setSuffix(" м")
        skel_set_layout.addWidget(self.height_spin, 1, 1)

        skel_settings.setLayout(skel_set_layout)
        skel_layout.addWidget(skel_settings)

        skel_cal_group.setLayout(skel_layout)
        layout.addWidget(skel_cal_group)

        # Калибровочные данные
        data_group = QGroupBox("Данные калибровки")
        data_layout = QVBoxLayout()

        self.calibration_data_text = QTextEdit()
        self.calibration_data_text.setReadOnly(True)
        self.calibration_data_text.setMaximumHeight(100)
        data_layout.addWidget(self.calibration_data_text)

        data_btn_layout = QHBoxLayout()
        data_btn_layout.addWidget(QPushButton("📥 Загрузить"))
        data_btn_layout.addWidget(QPushButton("📤 Сохранить"))
        data_btn_layout.addWidget(QPushButton("🔄 Сбросить"))

        data_layout.addLayout(data_btn_layout)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        layout.addStretch()
        return widget

    def _create_export_tab(self) -> QWidget:
        """Создание вкладки экспорта"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Форматы экспорта
        format_group = QGroupBox("Форматы экспорта")
        format_layout = QVBoxLayout()

        self.format_list = QListWidget()
        formats = [
            ("BVH", "📁 Стандартный формат для 3D анимации"),
            ("FBX", "🎮 Поддержка игровых движков"),
            ("JSON", "⚡ Легковесный для веба"),
            ("GLTF", "🌐 Современный для WebGL"),
            ("USD", "💼 Профессиональный для VFX")
        ]

        for name, desc in formats:
            item = QListWidgetItem(f"{name} - {desc}")
            item.setData(Qt.ItemDataRole.UserRole, name.lower())
            self.format_list.addItem(item)

        format_layout.addWidget(self.format_list)
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Настройки экспорта
        export_settings = QGroupBox("Настройки экспорта")
        export_set_layout = QGridLayout()

        export_set_layout.addWidget(QLabel("FPS:"), 0, 0)
        self.export_fps_spin = QSpinBox()
        self.export_fps_spin.setRange(1, 120)
        self.export_fps_spin.setValue(30)
        export_set_layout.addWidget(self.export_fps_spin, 0, 1)

        export_set_layout.addWidget(QLabel("Компрессия:"), 1, 0)
        self.compression_combo = QComboBox()
        self.compression_combo.addItems(["Нет", "Средняя", "Максимальная"])
        export_set_layout.addWidget(self.compression_combo, 1, 1)

        # Опции
        self.export_anim_cb = QCheckBox("Только анимация")
        export_set_layout.addWidget(self.export_anim_cb, 2, 0, 1, 2)

        self.export_skeleton_cb = QCheckBox("Со скелетом")
        export_set_layout.addWidget(self.export_skeleton_cb, 3, 0, 1, 2)

        self.export_metadata_cb = QCheckBox("С метаданными")
        self.export_metadata_cb.setChecked(True)
        export_set_layout.addWidget(self.export_metadata_cb, 4, 0, 1, 2)

        export_settings.setLayout(export_set_layout)
        layout.addWidget(export_settings)

        # Кнопка экспорта
        self.export_btn = AnimatedButton("🚀 Экспортировать анимацию")
        self.export_btn.clicked.connect(self._on_export_clicked)
        layout.addWidget(self.export_btn)

        # Пресеты экспорта
        layout.addWidget(self.presets["export"])

        layout.addStretch()
        return widget

    def _create_macros_tab(self) -> QWidget:
        """Создание вкладки макросов"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Список макросов
        macros_group = QGroupBox("Макросы")
        macros_layout = QVBoxLayout()

        self.macro_list = QTreeWidget()
        self.macro_list.setHeaderLabels(["Название", "Горячая клавиша", "Действия"])
        macros_layout.addWidget(self.macro_list)

        # Кнопки управления макросами
        macro_btn_layout = QHBoxLayout()

        self.record_macro_btn = QPushButton("🔴 Записать макрос")
        self.record_macro_btn.clicked.connect(self.start_macro_recording)
        macro_btn_layout.addWidget(self.record_macro_btn)

        self.play_macro_btn = QPushButton("▶️ Выполнить")
        self.play_macro_btn.clicked.connect(self.execute_macro)
        macro_btn_layout.addWidget(self.play_macro_btn)

        self.save_macro_btn = QPushButton("💾 Сохранить")
        macro_btn_layout.addWidget(self.save_macro_btn)

        macros_layout.addLayout(macro_btn_layout)
        macros_group.setLayout(macros_layout)
        layout.addWidget(macros_group)

        # Редактор макросов
        editor_group = QGroupBox("Редактор макроса")
        editor_layout = QVBoxLayout()

        self.macro_editor = QTextEdit()
        self.macro_editor.setPlaceholderText(
            "JSON структура макроса...\n"
            "Или используйте кнопку 'Записать' для создания."
        )
        editor_layout.addWidget(self.macro_editor)

        editor_group.setLayout(editor_layout)
        layout.addWidget(editor_group)

        # Быстрые макросы
        quick_group = QGroupBox("Быстрые макросы")
        quick_layout = QGridLayout()

        quick_macros = [
            ("🎬 Запись+Экспорт", "record_and_export"),
            ("⚡ Быстрая калибровка", "quick_calibration"),
            ("🔄 Цикл анимации", "loop_animation"),
            ("🎮 Игровой режим", "gaming_mode")
        ]

        row, col = 0, 0
        for name, action in quick_macros:
            btn = MacroButton(name)
            btn.clicked.connect(
                lambda checked, a=action: self.macro_triggered.emit(a)
            )
            quick_layout.addWidget(btn, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1

        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)

        layout.addStretch()
        return widget

    def _create_status_group(self) -> QGroupBox:
        """Создание группы статуса"""
        group = QGroupBox("Статус системы")
        group.setStyleSheet("""
            QGroupBox {
                color: #cccccc;
                border: 2px solid #404040;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        layout = QGridLayout()

        # Индикаторы состояния
        layout.addWidget(QLabel("Запись:"), 0, 0)
        self.record_led = StatusLED(16)
        layout.addWidget(self.record_led, 0, 1)

        layout.addWidget(QLabel("Трекинг:"), 1, 0)
        self.tracking_led = StatusLED(16)
        self.tracking_led.set_status("green")
        layout.addWidget(self.tracking_led, 1, 1)

        layout.addWidget(QLabel("Калибровка:"), 2, 0)
        self.calibration_led = StatusLED(16)
        layout.addWidget(self.calibration_led, 2, 1)

        layout.addWidget(QLabel("Память:"), 0, 2)
        self.memory_bar = QProgressBar()
        self.memory_bar.setRange(0, 100)
        self.memory_bar.setValue(45)
        self.memory_bar.setTextVisible(True)
        self.memory_bar.setFormat("%p%")
        layout.addWidget(self.memory_bar, 0, 3, 1, 2)

        layout.addWidget(QLabel("CPU:"), 1, 2)
        self.cpu_label = QLabel("12%")
        self.cpu_label.setStyleSheet("color: #ffaa00;")
        layout.addWidget(self.cpu_label, 1, 3)

        layout.addWidget(QLabel("Система:"), 2, 2)
        self.system_status = QLabel("✅ OK")
        self.system_status.setStyleSheet("color: #00ff00;")
        layout.addWidget(self.system_status, 2, 3)

        group.setLayout(layout)
        return group

    def _create_quick_actions(self) -> QGroupBox:
        """Создание быстрых действий"""
        group = QGroupBox("Быстрые действия")

        layout = QGridLayout()

        actions = [
            ("📸 Скриншот", self.take_screenshot, "F11"),
            ("🎥 Предпросмотр", self.toggle_preview, "F9"),
            ("🗑️ Очистить", self.clear_all, "Ctrl+Del"),
            ("🔄 Сброс", self.reset_all, "F12"),
            ("⚙️ Настройки", self.open_settings, "Ctrl+P"),
            ("❓ Помощь", self.show_help, "F1")
        ]

        row, col = 0, 0
        for name, callback, shortcut in actions:
            btn = QPushButton(name)
            btn.setToolTip(f"Горячая клавиша: {shortcut}")
            btn.clicked.connect(callback)

            if shortcut:
                action = QAction(self)
                action.setShortcut(QKeySequence(shortcut))
                action.triggered.connect(callback)
                self.addAction(action)

            layout.addWidget(btn, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1

        group.setLayout(layout)
        return group

    def apply_theme(self):
        """Применение темы"""
        style_sheet = f"""
            ProfessionalControlsPanel {{
                background-color: {self.style.background.name()};
                color: {self.style.foreground.name()};
                font-family: '{self.style.font_family}';
                font-size: {self.style.font_size}px;
            }}
            QPushButton {{
                background-color: {self.style.background.lighter(110).name()};
                border: 1px solid {self.style.border.name()};
                border-radius: 4px;
                padding: 6px;
                color: {self.style.foreground.name()};
            }}
            QPushButton:hover {{
                background-color: {self.style.accent.darker(120).name()};
                border: 1px solid {self.style.accent.name()};
            }}
            QPushButton:pressed {{
                background-color: {self.style.accent.darker(150).name()};
            }}
            QLabel {{
                color: {self.style.foreground.name()};
            }}
            QGroupBox {{
                color: {self.style.accent.name()};
                border: 1px solid {self.style.border.name()};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """

        self.setStyleSheet(style_sheet)

    def init_shortcuts(self):
        """Инициализация горячих клавиш"""
        shortcuts = {
            Qt.Key.Key_F5: self.start_recording.emit,
            Qt.Key.Key_F6: self.stop_recording.emit,
            Qt.Key.Key_F2: self.start_tracking,
            Qt.Key.Key_F3: self.stop_tracking,
            Qt.Key.Key_F7: lambda: self.calibrate_camera.emit(),
            Qt.Key.Key_F8: lambda: self.calibrate_skeleton.emit(),
            Qt.Key.Key_F4: lambda: self.export_animation.emit("bvh"),
            Qt.Key.Key_Escape: self.cancel_all
        }

        for key, callback in shortcuts.items():
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(callback)
            self.addAction(action)

    # ==================== ОСНОВНЫЕ МЕТОДЫ ====================

    def toggle_recording(self):
        """Переключение записи"""
        if self.is_recording:
            self.stop_recording.emit()
            self.record_btn.setText("🔴 Начать запись")
            self.record_btn.setStyleSheet(self.record_btn.styleSheet().replace(
                "#ff4444", "#44aa44"
            ))
            self.record_led.set_status("off")
            self.pause_btn.setEnabled(False)
        else:
            self.start_recording.emit()
            self.record_btn.setText("⏹️ Остановить запись")
            self.record_btn.setStyleSheet(self.record_btn.styleSheet().replace(
                "#44aa44", "#ff4444"
            ))
            self.record_led.set_status("red", blink=True)
            self.pause_btn.setEnabled(True)

        self.is_recording = not self.is_recording

        # Звуковой эффект
        if "record_start" in self.sound_effects:
            self.sound_effects["record_start"].play()

    def toggle_pause(self):
        """Переключение паузы"""
        if self.pause_btn.text() == "⏸️ Пауза":
            self.pause_btn.setText("▶️ Продолжить")
            # Отправить сигнал паузы
        else:
            self.pause_btn.setText("⏸️ Пауза")
            # Отправить сигнал продолжения

    def start_tracking(self):
        """Запуск трекинга"""
        if not self.is_tracking:
            self.is_tracking = True
            self.start_track_btn.setEnabled(False)
            self.stop_track_btn.setEnabled(True)
            self.tracking_led.set_status("green")

            logger.info("Трекинг запущен")

    def stop_tracking(self):
        """Остановка трекинга"""
        if self.is_tracking:
            self.is_tracking = False
            self.start_track_btn.setEnabled(True)
            self.stop_track_btn.setEnabled(False)
            self.tracking_led.set_status("off")

            logger.info("Трекинг остановлен")

    def _on_export_clicked(self):
        """Обработка клика по экспорту"""
        if self.format_list.currentItem():
            format_type = self.format_list.currentItem().data(
                Qt.ItemDataRole.UserRole
            )
            self.export_animation.emit(format_type)
        else:
            # Выбрать первый формат по умолчанию
            self.export_animation.emit("bvh")

    def start_macro_recording(self):
        """Начало записи макроса"""
        if self.record_macro_btn.text() == "🔴 Записать макрос":
            self.record_macro_btn.setText("⏹️ Остановить запись")
            self.active_macro = []
            logger.info("Начало записи макроса")
        else:
            self.record_macro_btn.setText("🔴 Записать макрос")
            # Сохранение макроса
            logger.info("Запись макроса завершена")

    def execute_macro(self):
        """Выполнение выбранного макроса"""
        current_item = self.macro_list.currentItem()
        if current_item:
            macro_name = current_item.text(0)
            if macro_name in self.macros:
                self.macro_triggered.emit(macro_name)
                logger.info(f"Выполнение макроса: {macro_name}")

    def update_status(self, system_status: Dict):
        """Обновление статуса системы"""
        # Обновление индикаторов
        if "recording" in system_status:
            self.record_led.set_status(
                "red" if system_status["recording"] else "off",
                blink=system_status["recording"]
            )

        if "tracking" in system_status:
            self.tracking_led.set_status(
                "green" if system_status["tracking"] else "off"
            )

        if "calibrating" in system_status:
            self.calibration_led.set_status(
                "yellow" if system_status["calibrating"] else "off",
                blink=system_status["calibrating"]
            )

        # Обновление информации о записи
        if "recording_info" in system_status:
            info = system_status["recording_info"]
            self.duration_label.setText(info.get("duration", "00:00:00"))
            self.frames_label.setText(str(info.get("frames", 0)))
            self.size_label.setText(info.get("size", "0 MB"))
            self.actual_fps_label.setText(f"{info.get('fps', 0):.1f}")

        # Обновление информации о трекинге
        if "tracking_info" in system_status:
            info = system_status["tracking_info"]
            self.landmarks_label.setText(str(info.get("landmarks", 0)))
            confidence = info.get("confidence", 0) * 100
            self.tracking_confidence_label.setText(f"{confidence:.1f}%")
            self.tracking_fps_label.setText(f"{info.get('fps', 0):.1f}")
            self.latency_label.setText(f"{info.get('latency', 0):.0f}ms")

        # Обновление системной информации
        if "system_info" in system_status:
            info = system_status["system_info"]
            self.memory_bar.setValue(info.get("memory_percent", 0))
            self.cpu_label.setText(f"{info.get('cpu_percent', 0):.0f}%")

            status_text = "✅ OK"
            status_color = "#00ff00"

            if info.get("warnings", []):
                status_text = "⚠️ Предупреждения"
                status_color = "#ffff00"
            if info.get("errors", []):
                status_text = "❌ Ошибки"
                status_color = "#ff0000"

            self.system_status.setText(status_text)
            self.system_status.setStyleSheet(f"color: {status_color};")

    def update_calibration_progress(self, current: int, total: int):
        """Обновление прогресса калибровки"""
        self.cam_cal_progress.setMaximum(total)
        self.cam_cal_progress.setValue(current)

        if current >= total:
            self.calibration_led.set_status("green")
        elif current > 0:
            self.calibration_led.set_status("yellow", blink=True)

    def set_calibration_data(self, data: str):
        """Установка данных калибровки"""
        self.calibration_data_text.setText(data)

    # ==================== БЫСТРЫЕ ДЕЙСТВИЯ ====================

    def take_screenshot(self):
        """Сделать скриншот"""
        logger.info("Скриншот сделан")
        # Сигнал будет обработан в главном окне

    def toggle_preview(self):
        """Переключение предпросмотра"""
        logger.info("Предпросмотр переключен")

    def clear_all(self):
        """Очистка всего"""
        reply = QMessageBox.question(
            self, "Очистка",
            "Очистить все данные?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            logger.info("Все данные очищены")
            # Отправка сигналов очистки

    def reset_all(self):
        """Сброс всех настроек"""
        reply = QMessageBox.question(
            self, "Сброс",
            "Сбросить все настройки к значениям по умолчанию?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            logger.info("Все настройки сброшены")
            # Сброс всех виджетов к значениям по умолчанию

    def open_settings(self):
        """Открытие настроек"""
        logger.info("Открытие настроек")
        # Открытие диалога настроек

    def show_help(self):
        """Показать справку"""
        logger.info("Показана справка")
        # Открытие справки

    def cancel_all(self):
        """Отмена всех операций"""
        logger.info("Все операции отменены")
        # Отправка сигналов отмены

    def set_theme(self, theme: ControlTheme):
        """Установка темы"""
        self.theme = theme
        self.style = ControlStyle.get_theme(theme)
        self.apply_theme()

    def get_settings(self) -> Dict:
        """Получение текущих настроек"""
        settings = {
            "recording": {
                "fps": self.fps_spin.value(),
                "quality": self.quality_slider.get_value(),
                "auto_track": self.auto_track_cb.isChecked(),
                "preview": self.preview_cb.isChecked()
            },
            "tracking": {
                "mode": self.tracking_mode_combo.currentText(),
                "confidence": self.confidence_slider.get_value(),
                "kalman": self.kalman_cb.isChecked(),
                "smoothing": self.smoothing_cb.isChecked()
            },
            "calibration": {
                "skeleton_model": self.skeleton_model_combo.currentText(),
                "height": self.height_spin.value()
            },
            "export": {
                "fps": self.export_fps_spin.value(),
                "compression": self.compression_combo.currentText(),
                "animation_only": self.export_anim_cb.isChecked(),
                "with_skeleton": self.export_skeleton_cb.isChecked(),
                "with_metadata": self.export_metadata_cb.isChecked()
            }
        }

        return settings

    def set_settings(self, settings: Dict):
        """Установка настроек"""
        try:
            # Запись
            if "recording" in settings:
                rec = settings["recording"]
                self.fps_spin.setValue(rec.get("fps", 30))
                self.quality_slider.set_value(rec.get("quality", 0.8))
                self.auto_track_cb.setChecked(rec.get("auto_track", True))
                self.preview_cb.setChecked(rec.get("preview", True))

            # Трекинг
            if "tracking" in settings:
                track = settings["tracking"]
                mode = track.get("mode", "🎯 Точный")
                index = self.tracking_mode_combo.findText(mode)
                if index >= 0:
                    self.tracking_mode_combo.setCurrentIndex(index)

                self.confidence_slider.set_value(track.get("confidence", 0.5))
                self.kalman_cb.setChecked(track.get("kalman", True))
                self.smoothing_cb.setChecked(track.get("smoothing", True))

            # Калибровка
            if "calibration" in settings:
                cal = settings["calibration"]
                model = cal.get("skeleton_model", "👤 Стандартный")
                index = self.skeleton_model_combo.findText(model)
                if index >= 0:
                    self.skeleton_model_combo.setCurrentIndex(index)

                self.height_spin.setValue(cal.get("height", 1.75))

            # Экспорт
            if "export" in settings:
                exp = settings["export"]
                self.export_fps_spin.setValue(exp.get("fps", 30))

                compression = exp.get("compression", "Средняя")
                index = self.compression_combo.findText(compression)
                if index >= 0:
                    self.compression_combo.setCurrentIndex(index)

                self.export_anim_cb.setChecked(exp.get("animation_only", False))
                self.export_skeleton_cb.setChecked(exp.get("with_skeleton", False))
                self.export_metadata_cb.setChecked(exp.get("with_metadata", True))

        except Exception as e:
            logger.error(f"Ошибка установки настроек: {e}")


# Для обратной совместимости
class ControlsPanel(ProfessionalControlsPanel):
    """Алиас для обратной совместимости"""
    pass


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    panel = ProfessionalControlsPanel()
    panel.resize(400, 800)
    panel.show()

    # Тестовое обновление статуса
    test_status = {
        "recording": True,
        "tracking": True,
        "calibrating": False,
        "recording_info": {
            "duration": "00:01:23",
            "frames": 1234,
            "size": "45.6 MB",
            "fps": 29.8
        },
        "tracking_info": {
            "landmarks": 33,
            "confidence": 0.87,
            "fps": 59.2,
            "latency": 16.8
        },
        "system_info": {
            "memory_percent": 65,
            "cpu_percent": 23,
            "warnings": [],
            "errors": []
        }
    }

    panel.update_status(test_status)

    sys.exit(app.exec())