"""
Модуль: Skeleton Editor (Визуальный редактор скелета)
Версия: 1.0.0
Автор: Mocap Pro Team

Интерактивный редактор для создания, редактирования и настройки скелетов.
Визуальное редактирование костей, ограничений, иерархии и поз.
"""

import sys
import json
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import glutInit

from core.skeleton import Skeleton, Bone, BoneType as JointType
from config.skeleton_presets import load_skeleton_preset


# Режимы редактирования
class EditMode(Enum):
    SELECT = "select"
    MOVE = "move"
    ROTATE = "rotate"
    SCALE = "scale"
    CREATE_BONE = "create_bone"
    DELETE_BONE = "delete_bone"
    CONSTRAINT = "constraint"
    PAINT_WEIGHTS = "paint_weights"


# Виды отображения
class ViewMode(Enum):
    SOLID = "solid"
    WIREFRAME = "wireframe"
    POINTS = "points"
    XRAY = "xray"
    SHADED = "shaded"
    CONSTRAINTS = "constraints"
    IK_CHAINS = "ik_chains"


# Типы ограничений
class ConstraintType(Enum):
    LIMIT_LOCATION = "limit_location"
    LIMIT_ROTATION = "limit_rotation"
    LIMIT_SCALE = "limit_scale"
    COPY_LOCATION = "copy_location"
    COPY_ROTATION = "copy_rotation"
    COPY_SCALE = "copy_scale"
    IK = "inverse_kinematics"
    STRETCH_TO = "stretch_to"
    TRACK_TO = "track_to"
    DAMPED_TRACK = "damped_track"
    TRANSFORM = "transform"
    SPLINE_IK = "spline_ik"
    PIVOT = "pivot"
    SHRINKWRAP = "shrinkwrap"


@dataclass
class BoneConstraint:
    """Ограничение для кости"""
    name: str
    constraint_type: ConstraintType
    target_bone: Optional[str] = None
    enabled: bool = True
    influence: float = 1.0
    is_muted: bool = False

    # Параметры ограничений
    use_limit_x: bool = False
    use_limit_y: bool = False
    use_limit_z: bool = False

    min_x: float = -180.0
    max_x: float = 180.0
    min_y: float = -180.0
    max_y: float = 180.0
    min_z: float = -180.0
    max_z: float = 180.0

    # IK параметры
    chain_length: int = 2
    iterations: int = 10
    tolerance: float = 0.01
    use_stretch: bool = False
    use_rotation: bool = True

    # Transform параметры
    from_min: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    from_max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    to_min: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    to_max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Дополнительные данные
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Конвертирует ограничение в словарь"""
        return {
            "name": self.name,
            "type": self.constraint_type.value,
            "target_bone": self.target_bone,
            "enabled": self.enabled,
            "influence": self.influence,
            "is_muted": self.is_muted,
            "use_limit_x": self.use_limit_x,
            "use_limit_y": self.use_limit_y,
            "use_limit_z": self.use_limit_z,
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "min_z": self.min_z,
            "max_z": self.max_z,
            "chain_length": self.chain_length,
            "iterations": self.iterations,
            "tolerance": self.tolerance,
            "use_stretch": self.use_stretch,
            "use_rotation": self.use_rotation,
            "from_min": self.from_min,
            "from_max": self.from_max,
            "to_min": self.to_min,
            "to_max": self.to_max,
            "custom_data": self.custom_data
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BoneConstraint':
        """Создает ограничение из словаря"""
        constraint = cls(
            name=data["name"],
            constraint_type=ConstraintType(data["type"]),
            target_bone=data.get("target_bone"),
            enabled=data.get("enabled", True),
            influence=data.get("influence", 1.0),
            is_muted=data.get("is_muted", False)
        )

        # Загружаем все параметры
        for key in data:
            if hasattr(constraint, key) and key not in ["name", "type", "target_bone", "enabled", "influence",
                                                        "is_muted"]:
                setattr(constraint, key, data[key])

        return constraint

    def is_ik_constraint(self) -> bool:
        """Проверяет, является ли ограничение IK"""
        return self.constraint_type in [
            ConstraintType.IK,
            ConstraintType.SPLINE_IK,
            ConstraintType.STRETCH_TO
        ]

    def is_limit_constraint(self) -> bool:
        """Проверяет, является ли ограничение лимитом"""
        return self.constraint_type in [
            ConstraintType.LIMIT_LOCATION,
            ConstraintType.LIMIT_ROTATION,
            ConstraintType.LIMIT_SCALE
        ]


class SkeletonViewport(QOpenGLWidget):
    """3D вьюпорт для редактирования скелета с аппаратным ускорением"""

    # Сигналы
    bone_selected = pyqtSignal(str, bool)  # Имя кости, additive selection
    bone_moved = pyqtSignal(str, object)  # Имя кости, новая позиция
    bone_rotated = pyqtSignal(str, object)  # Имя кости, новый rotation
    view_changed = pyqtSignal(object)  # Изменение вида камеры
    edit_mode_changed = pyqtSignal(str)  # Изменение режима редактирования

    def __init__(self, parent=None):
        super().__init__(parent)

        # Инициализация OpenGL
        try:
            glutInit()
        except:
            pass

        # Данные скелета
        self.skeleton: Optional[Skeleton] = None
        self.selected_bones: Set[str] = set()
        self.hovered_bone: Optional[str] = None
        self.dragging_bone: Optional[str] = None
        self.drag_start_pos: Optional[QVector3D] = None

        # Режимы
        self.edit_mode = EditMode.SELECT
        self.view_mode = ViewMode.SHADED
        self.coordinate_space = "LOCAL"  # LOCAL, WORLD, VIEW

        # Отображение
        self.show_grid = True
        self.show_axes = True
        self.show_bone_names = True
        self.show_constraints = True
        self.show_ik_chains = True
        self.show_normals = False
        self.show_wireframe = True

        # Камера
        self.camera_distance = 3.0
        self.camera_rotation = QVector3D(30.0, -45.0, 0.0)
        self.camera_target = QVector3D(0.0, 1.0, 0.0)
        self.camera_up = QVector3D(0.0, 1.0, 0.0)
        self.field_of_view = 60.0
        self.near_clip = 0.1
        self.far_clip = 100.0

        # Визуализация
        self.bone_radius = 0.03
        self.joint_radius = 0.05
        self.constraint_size = 0.1
        self.grid_size = 5.0
        self.grid_subdivisions = 20

        # Манипуляторы (гизмо)
        self.manipulator_type = "TRANSLATE"  # TRANSLATE, ROTATE, SCALE
        self.manipulator_size = 1.0
        self.manipulator_visible = True

        # Интерактивность
        self.is_rotating_camera = False
        self.is_panning_camera = False
        self.is_zooming_camera = False
        self.is_dragging_manipulator = False
        self.last_mouse_pos = QPoint()

        # Пикинг (выбор объектов в 3D)
        self.pick_buffer_size = 512
        self.pick_buffer = None

        # Цветовая схема
        self.colors = {
            "background": QColor(25, 25, 30),
            "background_gradient_top": QColor(40, 40, 50),
            "background_gradient_bottom": QColor(20, 20, 25),
            "grid_primary": QColor(60, 60, 70, 150),
            "grid_secondary": QColor(45, 45, 55, 100),
            "axes": {
                "x": QColor(220, 80, 80),
                "y": QColor(80, 220, 80),
                "z": QColor(80, 80, 220)
            },
            "bone": {
                "default": QColor(180, 180, 200, 200),
                "selected": QColor(255, 200, 50, 220),
                "hovered": QColor(100, 180, 255, 200),
                "locked": QColor(150, 150, 150, 150),
                "ik_chain": QColor(255, 150, 50, 180)
            },
            "joint": {
                "default": QColor(200, 200, 220, 220),
                "selected": QColor(255, 220, 100, 240),
                "root": QColor(220, 100, 100, 240)
            },
            "constraint": {
                "limit": QColor(255, 100, 100, 180),
                "copy": QColor(100, 200, 255, 180),
                "ik": QColor(255, 180, 50, 200),
                "track": QColor(100, 255, 100, 180)
            },
            "manipulator": {
                "x": QColor(255, 80, 80),
                "y": QColor(80, 255, 80),
                "z": QColor(80, 80, 255),
                "center": QColor(255, 255, 255),
                "plane_xy": QColor(255, 255, 80, 80),
                "plane_xz": QColor(255, 80, 255, 80),
                "plane_yz": QColor(80, 255, 255, 80)
            },
            "text": QColor(220, 220, 220),
            "selection_rect": QColor(100, 150, 255, 80)
        }

        # Шейдеры (для продвинутой визуализации)
        self.shaders_initialized = False
        self.bone_shader_program = None
        self.grid_shader_program = None

        # Буферы OpenGL
        self.grid_vbo = None
        self.axes_vbo = None
        self.bone_vbos = {}

        # Настройка виджета
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Настройка контекста OpenGL
        format = QSurfaceFormat()
        format.setDepthBufferSize(24)
        format.setStencilBufferSize(8)
        format.setVersion(3, 3)
        format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        format.setSamples(8)  # MSAA антиалиасинг
        self.setFormat(format)

        # Таймеры
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate)
        self.animation_timer.start(16)  # ~60 FPS

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(33)  # ~30 FPS

    def initializeGL(self):
        """Инициализация OpenGL"""
        try:
            # Инициализируем OpenGL
            glClearColor(0.1, 0.1, 0.12, 1.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_MULTISAMPLE)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LINE_SMOOTH)
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

            # Настраиваем освещение
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

            # Создаем шейдеры
            self._init_shaders()

            # Создаем геометрию
            self._init_geometry()

            self.shaders_initialized = True

        except Exception as e:
            print(f"⚠️ Ошибка инициализации OpenGL: {e}")
            # Fallback на обычную отрисовку

    def _init_shaders(self):
        """Инициализирует GLSL шейдеры"""
        try:
            # Простой vertex shader
            vertex_shader_source = """
            #version 330 core
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            layout(location = 2) in vec4 color;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            uniform vec3 light_position;

            out vec4 frag_color;
            out vec3 frag_normal;
            out vec3 frag_position;
            out vec3 frag_light_dir;

            void main() {
                vec4 world_position = model * vec4(position, 1.0);
                gl_Position = projection * view * world_position;

                frag_position = world_position.xyz;
                frag_normal = normalize(mat3(transpose(inverse(model))) * normal);
                frag_light_dir = normalize(light_position - world_position.xyz);
                frag_color = color;
            }
            """

            # Простой fragment shader
            fragment_shader_source = """
            #version 330 core
            in vec4 frag_color;
            in vec3 frag_normal;
            in vec3 frag_position;
            in vec3 frag_light_dir;

            out vec4 out_color;

            void main() {
                float ambient = 0.3;
                float diffuse = max(dot(frag_normal, frag_light_dir), 0.0);
                float specular = pow(max(dot(normalize(reflect(-frag_light_dir, frag_normal)), 
                                           normalize(-frag_position)), 0.0), 32.0) * 0.5;

                vec3 lighting = vec3(ambient + diffuse + specular);
                out_color = vec4(frag_color.rgb * lighting, frag_color.a);
            }
            """

            # Здесь должна быть полная реализация компиляции шейдеров
            # Для простоты оставляем заглушку

        except:
            pass  # Шейдеры не обязательны

    def _init_geometry(self):
        """Инициализирует геометрию для отрисовки"""
        # Создание VBO для сетки
        self._create_grid_geometry()

        # Создание VBO для осей
        self._create_axes_geometry()

    def _create_grid_geometry(self):
        """Создает геометрию для сетки"""
        # Реализация создания VBO для сетки
        pass

    def _create_axes_geometry(self):
        """Создает геометрию для осей координат"""
        # Реализация создания VBO для осей
        pass

    def resizeGL(self, width, height):
        """Обработка изменения размера окна"""
        glViewport(0, 0, width, height)

        # Устанавливаем проекционную матрицу
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        aspect = width / height if height > 0 else 1.0
        gluPerspective(self.field_of_view, aspect, self.near_clip, self.far_clip)

        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """Основная отрисовка"""
        try:
            # Очищаем буферы
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Сохраняем матрицы
            glPushMatrix()

            # Устанавливаем вид камеры
            self._setup_camera()

            # Рисуем фон
            self._draw_background()

            # Рисуем сетку
            if self.show_grid:
                self._draw_grid()

            # Рисуем оси координат
            if self.show_axes:
                self._draw_axes()

            # Рисуем скелет
            if self.skeleton:
                self._draw_skeleton()

                # Рисуем ограничения
                if self.show_constraints:
                    self._draw_constraints()

                # Рисуем IK цепи
                if self.show_ik_chains:
                    self._draw_ik_chains()

                # Рисуем манипуляторы
                if self.manipulator_visible and self.selected_bones:
                    self._draw_manipulators()

            # Рисуем прямоугольник выделения
            if self.is_selecting_rect:
                self._draw_selection_rect()

            # Восстанавливаем матрицы
            glPopMatrix()

        except Exception as e:
            print(f"⚠️ Ошибка отрисовки OpenGL: {e}")
            # Fallback отрисовка
            self._draw_fallback()

    def _setup_camera(self):
        """Настраивает вид камеры"""
        glLoadIdentity()

        # Перемещаем камеру
        glTranslatef(0.0, 0.0, -self.camera_distance)

        # Вращаем камеру
        glRotatef(self.camera_rotation.x(), 1.0, 0.0, 0.0)
        glRotatef(self.camera_rotation.y(), 0.0, 1.0, 0.0)
        glRotatef(self.camera_rotation.z(), 0.0, 0.0, 1.0)

        # Центрируем на цели
        glTranslatef(-self.camera_target.x(), -self.camera_target.y(), -self.camera_target.z())

    def _draw_background(self):
        """Рисует градиентный фон"""
        # Простой фон
        glClearColor(0.1, 0.1, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def _draw_grid(self):
        """Рисует 3D сетку"""
        glPushMatrix()
        glTranslatef(self.camera_target.x(), 0.0, self.camera_target.z())

        glBegin(GL_LINES)

        # Основные линии
        glColor4f(0.4, 0.4, 0.45, 0.6)
        size = self.grid_size

        for i in range(-int(size), int(size) + 1):
            if i % 5 == 0:
                glColor4f(0.5, 0.5, 0.55, 0.8)  # Более яркие линии
            else:
                glColor4f(0.3, 0.3, 0.35, 0.4)  # Более тусклые линии

            # Линии по X
            glVertex3f(-size, 0.0, i)
            glVertex3f(size, 0.0, i)

            # Линии по Z
            glVertex3f(i, 0.0, -size)
            glVertex3f(i, 0.0, size)

        glEnd()

        glPopMatrix()

    def _draw_axes(self):
        """Рисует оси координат"""
        glPushMatrix()
        glTranslatef(-self.grid_size + 0.5, 0.0, -self.grid_size + 0.5)

        axis_length = 1.0

        glBegin(GL_LINES)

        # Ось X (красная)
        glColor3f(0.9, 0.2, 0.2)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(axis_length, 0.0, 0.0)

        # Ось Y (зеленая)
        glColor3f(0.2, 0.9, 0.2)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, axis_length, 0.0)

        # Ось Z (синяя)
        glColor3f(0.2, 0.2, 0.9)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, axis_length)

        glEnd()

        # Стрелки
        arrow_size = 0.1

        # Стрелка X
        glColor3f(0.9, 0.2, 0.2)
        glBegin(GL_TRIANGLES)
        glVertex3f(axis_length, 0.0, 0.0)
        glVertex3f(axis_length - arrow_size, arrow_size / 2, 0.0)
        glVertex3f(axis_length - arrow_size, -arrow_size / 2, 0.0)
        glEnd()

        # Стрелка Y
        glColor3f(0.2, 0.9, 0.2)
        glBegin(GL_TRIANGLES)
        glVertex3f(0.0, axis_length, 0.0)
        glVertex3f(arrow_size / 2, axis_length - arrow_size, 0.0)
        glVertex3f(-arrow_size / 2, axis_length - arrow_size, 0.0)
        glEnd()

        # Стрелка Z
        glColor3f(0.2, 0.2, 0.9)
        glBegin(GL_TRIANGLES)
        glVertex3f(0.0, 0.0, axis_length)
        glVertex3f(arrow_size / 2, 0.0, axis_length - arrow_size)
        glVertex3f(-arrow_size / 2, 0.0, axis_length - arrow_size)
        glEnd()

        glPopMatrix()

    def _draw_skeleton(self):
        """Рисует скелет с костями и суставами"""
        if not self.skeleton:
            return

        # Рекурсивная отрисовка костей
        root_bone = self.skeleton.get_root_bone()
        if root_bone:
            self._draw_bone_recursive(root_bone.name, None)

    def _draw_bone_recursive(self, bone_name: str, parent_name: Optional[str]):
        """Рекурсивно рисует кость и её детей"""
        bone = self.skeleton.get_bone(bone_name)
        if not bone or not bone.enabled:
            return

        # Получаем мировые координаты
        bone_world = self.skeleton.get_bone_world_matrix(bone_name)
        position = bone_world[:3, 3]

        # Цвет кости
        if bone_name in self.selected_bones:
            color = self.colors["bone"]["selected"]
        elif bone_name == self.hovered_bone:
            color = self.colors["bone"]["hovered"]
        elif bone.locked:
            color = self.colors["bone"]["locked"]
        else:
            color = self.colors["bone"]["default"]

        # Рисуем кость
        if parent_name:
            parent_bone = self.skeleton.get_bone(parent_name)
            if parent_bone:
                parent_world = self.skeleton.get_bone_world_matrix(parent_name)
                parent_pos = parent_world[:3, 3]

                # Рисуем линию от родителя к текущей кости
                self._draw_bone_line(parent_pos, position, color, bone.length)

        # Рисуем сустав
        self._draw_joint(position, color, bone.is_root)

        # Рисуем имя кости
        if self.show_bone_names:
            self._draw_bone_name(bone_name, position)

        # Рекурсивно рисуем детей
        children = self.skeleton.get_children(bone_name)
        for child in children:
            self._draw_bone_recursive(child.name, bone_name)

    def _draw_bone_line(self, start: np.ndarray, end: np.ndarray, color: QColor, length: float):
        """Рисует линию кости с объемом"""
        glPushMatrix()

        # Преобразуем цвет
        glColor4f(color.redF(), color.greenF(), color.blueF(), color.alphaF())

        # Вычисляем направление
        direction = end - start
        distance = np.linalg.norm(direction)

        if distance > 0:
            direction = direction / distance

            # Вычисляем rotation для цилиндра
            up = np.array([0.0, 1.0, 0.0])
            axis = np.cross(up, direction)
            axis_len = np.linalg.norm(axis)

            if axis_len > 0:
                axis = axis / axis_len
                angle = np.degrees(np.arccos(np.clip(np.dot(up, direction), -1.0, 1.0)))

                # Рисуем цилиндр
                glTranslatef(start[0], start[1], start[2])
                glRotatef(angle, axis[0], axis[1], axis[2])

                # Простой цилиндр
                radius = self.bone_radius
                slices = 8

                # Боковая поверхность
                glBegin(GL_QUAD_STRIP)
                for i in range(slices + 1):
                    angle = 2 * np.pi * i / slices
                    x = radius * np.cos(angle)
                    z = radius * np.sin(angle)

                    glVertex3f(x, 0.0, z)
                    glVertex3f(x, distance, z)
                glEnd()

                # Крышки
                glBegin(GL_POLYGON)
                for i in range(slices):
                    angle = 2 * np.pi * i / slices
                    x = radius * np.cos(angle)
                    z = radius * np.sin(angle)
                    glVertex3f(x, 0.0, z)
                glEnd()

                glBegin(GL_POLYGON)
                for i in range(slices):
                    angle = 2 * np.pi * i / slices
                    x = radius * np.cos(angle)
                    z = radius * np.sin(angle)
                    glVertex3f(x, distance, z)
                glEnd()

        glPopMatrix()

    def _draw_joint(self, position: np.ndarray, color: QColor, is_root: bool = False):
        """Рисует сустав (сферу)"""
        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])

        # Выбираем цвет
        if is_root:
            joint_color = self.colors["joint"]["root"]
        else:
            joint_color = color

        glColor4f(joint_color.redF(), joint_color.greenF(), joint_color.blueF(), joint_color.alphaF())

        # Рисуем сферу
        radius = self.joint_radius
        slices = 16
        stacks = 16

        # Используем quadric для сферы
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluQuadricDrawStyle(quadric, GLU_FILL)
        gluSphere(quadric, radius, slices, stacks)
        gluDeleteQuadric(quadric)

        glPopMatrix()

    def _draw_bone_name(self, bone_name: str, position: np.ndarray):
        """Рисует имя кости в 3D пространстве"""
        # Это заглушка - настоящая отрисовка текста в OpenGL сложна
        # В реальной реализации нужно использовать текстуры или QPainter поверх OpenGL
        pass

    def _draw_constraints(self):
        """Рисует ограничения на костях"""
        # Здесь должна быть сложная логика отрисовки разных типов ограничений
        # Для простоты рисуем только IK ограничения

        if not self.skeleton:
            return

        # Ищем IK ограничения
        for bone_name in self.selected_bones:
            bone = self.skeleton.get_bone(bone_name)
            if bone and hasattr(bone, 'constraints'):
                for constraint in getattr(bone, 'constraints', []):
                    if constraint.is_ik_constraint() and constraint.target_bone:
                        self._draw_ik_constraint(bone_name, constraint.target_bone)

    def _draw_ik_constraint(self, source_bone: str, target_bone: str):
        """Рисует IK ограничение"""
        source = self.skeleton.get_bone_world_matrix(source_bone)[:3, 3]
        target = self.skeleton.get_bone_world_matrix(target_bone)[:3, 3]

        glColor4f(1.0, 0.7, 0.2, 0.8)
        glLineWidth(2.0)

        glBegin(GL_LINES)
        glVertex3f(source[0], source[1], source[2])
        glVertex3f(target[0], target[1], target[2])
        glEnd()

        glLineWidth(1.0)

    def _draw_ik_chains(self):
        """Рисует IK цепи"""
        if not self.skeleton:
            return

        # Здесь должна быть логика отрисовки IK цепей
        # Для простоты рисуем цепочки костей

        for bone in self.skeleton.bones:
            if hasattr(bone, 'ik_chain_length') and bone.ik_chain_length > 1:
                self._draw_ik_chain(bone.name, bone.ik_chain_length)

    def _draw_ik_chain(self, start_bone: str, chain_length: int):
        """Рисует IK цепь"""
        current_bone = start_bone
        points = []

        for i in range(chain_length):
            bone = self.skeleton.get_bone(current_bone)
            if not bone:
                break

            world_matrix = self.skeleton.get_bone_world_matrix(current_bone)
            points.append(world_matrix[:3, 3])

            # Переходим к родителю
            if bone.parent:
                current_bone = bone.parent
            else:
                break

        if len(points) > 1:
            glColor4f(0.8, 0.5, 0.2, 0.6)
            glLineWidth(3.0)

            glBegin(GL_LINE_STRIP)
            for point in reversed(points):  # От конца к началу
                glVertex3f(point[0], point[1], point[2])
            glEnd()

            glLineWidth(1.0)

    def _draw_manipulators(self):
        """Рисует манипуляторы (гизмо) для редактирования"""
        if not self.selected_bones:
            return

        # Берем первую выбранную кость для манипулятора
        bone_name = next(iter(self.selected_bones))
        bone = self.skeleton.get_bone(bone_name)
        if not bone:
            return

        world_matrix = self.skeleton.get_bone_world_matrix(bone_name)
        position = world_matrix[:3, 3]

        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])

        if self.manipulator_type == "TRANSLATE":
            self._draw_translate_manipulator()
        elif self.manipulator_type == "ROTATE":
            self._draw_rotate_manipulator()
        elif self.manipulator_type == "SCALE":
            self._draw_scale_manipulator()

        glPopMatrix()

    def _draw_translate_manipulator(self):
        """Рисует манипулятор перемещения"""
        size = self.manipulator_size * 0.5

        # Ось X (красная)
        glColor3f(1.0, 0.3, 0.3)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(size, 0.0, 0.0)
        glEnd()

        # Стрелка X
        glBegin(GL_TRIANGLES)
        glVertex3f(size, 0.0, 0.0)
        glVertex3f(size - 0.1, 0.05, 0.0)
        glVertex3f(size - 0.1, -0.05, 0.0)
        glEnd()

        # Ось Y (зеленая)
        glColor3f(0.3, 1.0, 0.3)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, size, 0.0)
        glEnd()

        # Стрелка Y
        glBegin(GL_TRIANGLES)
        glVertex3f(0.0, size, 0.0)
        glVertex3f(0.05, size - 0.1, 0.0)
        glVertex3f(-0.05, size - 0.1, 0.0)
        glEnd()

        # Ось Z (синяя)
        glColor3f(0.3, 0.3, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, size)
        glEnd()

        # Стрелка Z
        glBegin(GL_TRIANGLES)
        glVertex3f(0.0, 0.0, size)
        glVertex3f(0.05, 0.0, size - 0.1)
        glVertex3f(-0.05, 0.0, size - 0.1)
        glEnd()

    def _draw_rotate_manipulator(self):
        """Рисует манипулятор вращения"""
        size = self.manipulator_size

        segments = 32
        glLineWidth(2.0)

        # Кольцо X (красный)
        glColor3f(1.0, 0.3, 0.3)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            glVertex3f(0.0, size * np.cos(angle), size * np.sin(angle))
        glEnd()

        # Кольцо Y (зеленый)
        glColor3f(0.3, 1.0, 0.3)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            glVertex3f(size * np.cos(angle), 0.0, size * np.sin(angle))
        glEnd()

        # Кольцо Z (синий)
        glColor3f(0.3, 0.3, 1.0)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            glVertex3f(size * np.cos(angle), size * np.sin(angle), 0.0)
        glEnd()

        glLineWidth(1.0)

    def _draw_scale_manipulator(self):
        """Рисует манипулятор масштабирования"""
        size = self.manipulator_size * 0.5

        # Ось X (красная)
        glColor3f(1.0, 0.3, 0.3)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(size, 0.0, 0.0)
        glEnd()

        # Кубик на конце X
        glPushMatrix()
        glTranslatef(size, 0.0, 0.0)
        glScalef(0.05, 0.05, 0.05)
        self._draw_cube()
        glPopMatrix()

        # Ось Y (зеленая)
        glColor3f(0.3, 1.0, 0.3)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, size, 0.0)
        glEnd()

        # Кубик на конце Y
        glPushMatrix()
        glTranslatef(0.0, size, 0.0)
        glScalef(0.05, 0.05, 0.05)
        self._draw_cube()
        glPopMatrix()

        # Ось Z (синяя)
        glColor3f(0.3, 0.3, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, size)
        glEnd()

        # Кубик на конце Z
        glPushMatrix()
        glTranslatef(0.0, 0.0, size)
        glScalef(0.05, 0.05, 0.05)
        self._draw_cube()
        glPopMatrix()

    def _draw_cube(self):
        """Рисует простой куб"""
        vertices = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
        ]

        glBegin(GL_QUADS)
        for face in faces:
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()

    def _draw_selection_rect(self):
        """Рисует прямоугольник выделения"""
        if not hasattr(self, 'selection_rect') or self.selection_rect.isEmpty():
            return

        # Рисуем полупрозрачный прямоугольник
        rect = self.selection_rect
        color = self.colors["selection_rect"]

        # Используем 2D отрисовку поверх 3D
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width(), self.height(), 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)

        glColor4f(color.redF(), color.greenF(), color.blueF(), color.alphaF())
        glBegin(GL_QUADS)
        glVertex2f(rect.left(), rect.top())
        glVertex2f(rect.right(), rect.top())
        glVertex2f(rect.right(), rect.bottom())
        glVertex2f(rect.left(), rect.bottom())
        glEnd()

        glColor4f(0.4, 0.6, 1.0, 1.0)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(rect.left(), rect.top())
        glVertex2f(rect.right(), rect.top())
        glVertex2f(rect.right(), rect.bottom())
        glVertex2f(rect.left(), rect.bottom())
        glEnd()

        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _draw_fallback(self):
        """Fallback отрисовка если OpenGL не работает"""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(50, 50, 55))

        if self.skeleton:
            # Простая 2D отрисовка скелета
            painter.setPen(QPen(QColor(200, 200, 220), 2))
            self._draw_skeleton_2d(painter)

        painter.end()

    def _draw_skeleton_2d(self, painter: QPainter):
        """2D отрисовка скелета (fallback)"""
        # Упрощенная 2D визуализация
        transform = QTransform()
        transform.translate(self.width() / 2, self.height() / 2)
        transform.scale(50, -50)  # Масштаб и инвертирование Y

        painter.setTransform(transform)

        # Рисуем кости
        for bone in self.skeleton.bones:
            if bone.parent:
                parent_bone = self.skeleton.get_bone(bone.parent)
                if parent_bone:
                    # Простая линия от родителя к кости
                    start = bone.rest_position
                    end = parent_bone.rest_position

                    color = QColor(200, 200, 220)
                    if bone.name in self.selected_bones:
                        color = QColor(255, 200, 50)

                    painter.setPen(QPen(color, 0.1))
                    painter.drawLine(
                        QPointF(start[0], start[2]),  # Используем XZ плоскость
                        QPointF(end[0], end[2])
                    )

    # === МЕТОДЫ ВЗАИМОДЕЙСТВИЯ ===

    def set_skeleton(self, skeleton: Skeleton):
        """Устанавливает скелет для редактирования"""
        self.skeleton = skeleton
        self.selected_bones.clear()
        self.hovered_bone = None

        # Обновляем геометрию
        self._update_bone_geometry()

        # Обновляем отображение
        self.update()

    def select_bone(self, bone_name: str, additive: bool = False, clear: bool = True):
        """Выбирает кость"""
        if not self.skeleton or bone_name not in self.skeleton.bones:
            return

        if not additive and clear:
            self.selected_bones.clear()

        self.selected_bones.add(bone_name)
        self.bone_selected.emit(bone_name, additive)
        self.update()

    def select_bones(self, bone_names: List[str], additive: bool = False):
        """Выбирает несколько костей"""
        if not additive:
            self.selected_bones.clear()

        for bone_name in bone_names:
            if self.skeleton and bone_name in self.skeleton.bones:
                self.selected_bones.add(bone_name)

        self.update()

    def clear_selection(self):
        """Очищает выделение"""
        self.selected_bones.clear()
        self.update()

    def set_edit_mode(self, mode: EditMode):
        """Устанавливает режим редактирования"""
        self.edit_mode = mode
        self.edit_mode_changed.emit(mode.value)
        self.update()

    def set_view_mode(self, mode: ViewMode):
        """Устанавливает режим отображения"""
        self.view_mode = mode
        self.update()

    def set_manipulator_type(self, manip_type: str):
        """Устанавливает тип манипулятора"""
        self.manipulator_type = manip_type
        self.update()

    def _update_bone_geometry(self):
        """Обновляет геометрию костей для OpenGL"""
        # Здесь должна быть оптимизированная генерация VBO для костей
        pass

    def _screen_to_world(self, screen_pos: QPoint) -> QVector3D:
        """Конвертирует экранные координаты в мировые"""
        # Получаем матрицы
        glGetDoublev(GL_MODELVIEW_MATRIX, self.modelview)
        glGetDoublev(GL_PROJECTION_MATRIX, self.projection)
        glGetIntegerv(GL_VIEWPORT, self.viewport)

        # Конвертируем координаты
        win_x = screen_pos.x()
        win_y = self.viewport[3] - screen_pos.y()  # Инвертируем Y

        # Берем ближнюю и дальнюю точки
        near_point = gluUnProject(win_x, win_y, 0.0,
                                  self.modelview, self.projection, self.viewport)
        far_point = gluUnProject(win_x, win_y, 1.0,
                                 self.modelview, self.projection, self.viewport)

        if near_point and far_point:
            # Возвращаем ближнюю точку
            return QVector3D(near_point[0], near_point[1], near_point[2])

        return QVector3D()

    def _world_to_screen(self, world_pos: QVector3D) -> QPoint:
        """Конвертирует мировые координаты в экранные"""
        # Получаем матрицы
        glGetDoublev(GL_MODELVIEW_MATRIX, self.modelview)
        glGetDoublev(GL_PROJECTION_MATRIX, self.projection)
        glGetIntegerv(GL_VIEWPORT, self.viewport)

        # Конвертируем координаты
        screen_coords = gluProject(world_pos.x(), world_pos.y(), world_pos.z(),
                                   self.modelview, self.projection, self.viewport)

        if screen_coords:
            # Инвертируем Y
            return QPoint(int(screen_coords[0]), int(self.viewport[3] - screen_coords[1]))

        return QPoint()

    def _pick_bone(self, screen_pos: QPoint) -> Optional[str]:
        """Выбирает кость под курсором (пикинг)"""
        if not self.skeleton:
            return None

        # Реализация пикинга через буфер выбора
        # В реальной реализации нужно использовать glSelectBuffer

        # Упрощенная реализация - проверяем ближайшие кости
        closest_bone = None
        closest_distance = float('inf')

        for bone in self.skeleton.bones:
            world_pos = self.skeleton.get_bone_world_matrix(bone.name)[:3, 3]
            screen_point = self._world_to_screen(QVector3D(world_pos[0], world_pos[1], world_pos[2]))

            distance = QLineF(screen_pos, screen_point).length()
            if distance < 20 and distance < closest_distance:  # 20 пикселей радиус
                closest_distance = distance
                closest_bone = bone.name

        return closest_bone

    def mousePressEvent(self, event: QMouseEvent):
        """Обработка нажатия мыши"""
        self.last_mouse_pos = event.pos()

        if event.button() == Qt.MouseButton.LeftButton:
            # Проверяем, попали ли в манипулятор
            if self.manipulator_visible and self.selected_bones:
                bone_name = next(iter(self.selected_bones))
                if self._check_manipulator_hit(event.pos(), bone_name):
                    self.is_dragging_manipulator = True
                    self.drag_start_pos = self._screen_to_world(event.pos())
                    return

            # Проверяем, попали ли в кость
            picked_bone = self._pick_bone(event.pos())
            if picked_bone:
                # Выбираем кость
                additive = event.modifiers() & Qt.KeyboardModifier.ControlModifier
                self.select_bone(picked_bone, additive)

                # Начинаем перетаскивание
                if self.edit_mode == EditMode.MOVE:
                    self.dragging_bone = picked_bone
                    self.drag_start_pos = self._screen_to_world(event.pos())

                return

            # Начинаем выделение прямоугольником
            if self.edit_mode == EditMode.SELECT:
                self.is_selecting_rect = True
                self.selection_rect = QRect(event.pos(), QSize())

        elif event.button() == Qt.MouseButton.MiddleButton:
            # Начинаем панорамирование камеры
            self.is_panning_camera = True

        elif event.button() == Qt.MouseButton.RightButton:
            # Начинаем вращение камеры
            self.is_rotating_camera = True

        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Обработка движения мыши"""
        delta = event.pos() - self.last_mouse_pos

        # Обновляем hovered кость
        hovered_bone = self._pick_bone(event.pos())
        if hovered_bone != self.hovered_bone:
            self.hovered_bone = hovered_bone
            self.update()

        # Обработка перетаскивания манипулятора
        if self.is_dragging_manipulator and self.drag_start_pos is not None:
            current_pos = self._screen_to_world(event.pos())
            delta_world = current_pos - self.drag_start_pos

            if self.selected_bones:
                bone_name = next(iter(self.selected_bones))
                bone = self.skeleton.get_bone(bone_name)
                if bone:
                    # Применяем трансформацию в зависимости от типа манипулятора
                    if self.manipulator_type == "TRANSLATE":
                        new_pos = bone.rest_position + np.array([delta_world.x(), delta_world.y(), delta_world.z()])
                        bone.set_rest_position(new_pos)
                        self.bone_moved.emit(bone_name, new_pos.tolist())

                    elif self.manipulator_type == "ROTATE":
                        # Вращение вокруг выбранной оси
                        pass

                    elif self.manipulator_type == "SCALE":
                        # Масштабирование
                        pass

            self.drag_start_pos = current_pos
            self.update()

        # Обработка перетаскивания кости
        elif self.dragging_bone and self.drag_start_pos is not None:
            if self.edit_mode == EditMode.MOVE:
                current_pos = self._screen_to_world(event.pos())
                delta_world = current_pos - self.drag_start_pos

                bone = self.skeleton.get_bone(self.dragging_bone)
                if bone:
                    new_pos = bone.rest_position + np.array([delta_world.x(), delta_world.y(), delta_world.z()])
                    bone.set_rest_position(new_pos)
                    self.bone_moved.emit(self.dragging_bone, new_pos.tolist())

                self.drag_start_pos = current_pos
                self.update()

        # Обработка вращения камеры
        elif self.is_rotating_camera:
            self.camera_rotation.setX(self.camera_rotation.x() + delta.y() * 0.5)
            self.camera_rotation.setY(self.camera_rotation.y() + delta.x() * 0.5)
            self.view_changed.emit({
                "rotation": [self.camera_rotation.x(), self.camera_rotation.y(), self.camera_rotation.z()],
                "distance": self.camera_distance
            })
            self.update()

        # Обработка панорамирования камеры
        elif self.is_panning_camera:
            pan_speed = 0.01 * self.camera_distance
            self.camera_target.setX(self.camera_target.x() - delta.x() * pan_speed)
            self.camera_target.setY(self.camera_target.y() + delta.y() * pan_speed)  # Инвертируем Y
            self.update()

        # Обработка выделения прямоугольником
        elif self.is_selecting_rect:
            self.selection_rect.setBottomRight(event.pos())
            self.update()

        self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Обработка отпускания мыши"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging_manipulator = False
            self.dragging_bone = None
            self.is_selecting_rect = False

            # Если был прямоугольник выделения, выбираем кости внутри него
            if hasattr(self, 'selection_rect') and not self.selection_rect.isEmpty():
                self._select_bones_in_rect(self.selection_rect)
                self.selection_rect = QRect()

        elif event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning_camera = False

        elif event.button() == Qt.MouseButton.RightButton:
            self.is_rotating_camera = False

        self.update()

    def wheelEvent(self, event: QWheelEvent):
        """Обработка колесика мыши"""
        delta = event.angleDelta().y()

        # Приближение/отдаление камеры
        zoom_factor = 1.1 if delta > 0 else 0.9
        self.camera_distance *= zoom_factor
        self.camera_distance = max(0.1, min(100.0, self.camera_distance))

        self.view_changed.emit({
            "distance": self.camera_distance,
            "target": [self.camera_target.x(), self.camera_target.y(), self.camera_target.z()]
        })

        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        """Обработка нажатия клавиш"""
        # Управление камерой
        if event.key() == Qt.Key.Key_F:
            # Фокусировка на выделенных костях
            self.focus_on_selection()

        elif event.key() == Qt.Key.Key_R:
            # Сброс камеры
            self.reset_camera()

        elif event.key() == Qt.Key.Key_G:
            # Режим перемещения
            self.set_edit_mode(EditMode.MOVE)

        elif event.key() == Qt.Key.Key_R and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Режим вращения
            self.set_edit_mode(EditMode.ROTATE)

        elif event.key() == Qt.Key.Key_S and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Режим масштабирования
            self.set_edit_mode(EditMode.SCALE)

        elif event.key() == Qt.Key.Key_W:
            # Переключение режима отображения
            modes = list(ViewMode)
            current_index = modes.index(self.view_mode)
            next_index = (current_index + 1) % len(modes)
            self.set_view_mode(modes[next_index])

        elif event.key() == Qt.Key.Key_Delete:
            # Удаление выбранных костей
            self.delete_selected_bones()

        elif event.key() == Qt.Key.Key_Escape:
            # Сброс выделения
            self.clear_selection()

        self.update()

    def _check_manipulator_hit(self, screen_pos: QPoint, bone_name: str) -> bool:
        """Проверяет, попал ли курсор в манипулятор"""
        # Упрощенная проверка
        bone = self.skeleton.get_bone(bone_name)
        if not bone:
            return False

        world_pos = self.skeleton.get_bone_world_matrix(bone_name)[:3, 3]
        screen_point = self._world_to_screen(QVector3D(world_pos[0], world_pos[1], world_pos[2]))

        distance = QLineF(screen_pos, screen_point).length()
        return distance < 30  # 30 пикселей радиус

    def _select_bones_in_rect(self, rect: QRect):
        """Выбирает кости внутри прямоугольника"""
        if not self.skeleton:
            return

        selected_bones = []

        for bone in self.skeleton.bones:
            world_pos = self.skeleton.get_bone_world_matrix(bone.name)[:3, 3]
            screen_point = self._world_to_screen(QVector3D(world_pos[0], world_pos[1], world_pos[2]))

            if rect.contains(screen_point):
                selected_bones.append(bone.name)

        if selected_bones:
            self.select_bones(selected_bones, additive=True)

    def focus_on_selection(self):
        """Фокусирует камеру на выделенных костях"""
        if not self.selected_bones or not self.skeleton:
            return

        # Вычисляем центр выделенных костей
        positions = []
        for bone_name in self.selected_bones:
            world_pos = self.skeleton.get_bone_world_matrix(bone_name)[:3, 3]
            positions.append(world_pos)

        if positions:
            center = np.mean(positions, axis=0)
            self.camera_target = QVector3D(center[0], center[1], center[2])

            # Автоматически подбираем дистанцию
            max_distance = 0
            for pos in positions:
                distance = np.linalg.norm(pos - center)
                max_distance = max(max_distance, distance)

            self.camera_distance = max(max_distance * 3.0, 1.0)
            self.update()

    def reset_camera(self):
        """Сбрасывает камеру в положение по умолчанию"""
        self.camera_distance = 3.0
        self.camera_rotation = QVector3D(30.0, -45.0, 0.0)
        self.camera_target = QVector3D(0.0, 1.0, 0.0)
        self.update()

    def delete_selected_bones(self):
        """Удаляет выбранные кости"""
        if not self.skeleton or not self.selected_bones:
            return

        # Удаляем кости (нельзя удалять корневую кость если она единственная)
        for bone_name in list(self.selected_bones):
            if bone_name != self.skeleton.get_root_bone().name or len(self.skeleton.bones) > 1:
                self.skeleton.remove_bone(bone_name)

        self.selected_bones.clear()
        self.update()

    def animate(self):
        """Анимация (для плавных переходов)"""
        # Здесь может быть логика плавных переходов камеры и т.д.
        pass

    def get_view_info(self) -> Dict:
        """Возвращает информацию о текущем виде"""
        return {
            "camera": {
                "distance": self.camera_distance,
                "rotation": [self.camera_rotation.x(), self.camera_rotation.y(), self.camera_rotation.z()],
                "target": [self.camera_target.x(), self.camera_target.y(), self.camera_target.z()],
                "fov": self.field_of_view
            },
            "view_mode": self.view_mode.value,
            "edit_mode": self.edit_mode.value,
            "show_grid": self.show_grid,
            "show_axes": self.show_axes,
            "selected_bones": list(self.selected_bones),
            "hovered_bone": self.hovered_bone
        }

    def save_view(self, filepath: str):
        """Сохраняет настройки вида в файл"""
        view_info = self.get_view_info()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(view_info, f, indent=2)

    def load_view(self, filepath: str):
        """Загружает настройки вида из файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                view_info = json.load(f)

            camera = view_info.get("camera", {})
            self.camera_distance = camera.get("distance", 3.0)
            rotation = camera.get("rotation", [30.0, -45.0, 0.0])
            self.camera_rotation = QVector3D(rotation[0], rotation[1], rotation[2])
            target = camera.get("target", [0.0, 1.0, 0.0])
            self.camera_target = QVector3D(target[0], target[1], target[2])

            self.view_mode = ViewMode(view_info.get("view_mode", "shaded"))
            self.edit_mode = EditMode(view_info.get("edit_mode", "select"))
            self.show_grid = view_info.get("show_grid", True)
            self.show_axes = view_info.get("show_axes", True)

            self.update()

        except Exception as e:
            print(f"❌ Ошибка загрузки вида: {e}")


class BoneHierarchyWidget(QTreeWidget):
    """Виджет иерархии костей"""

    # Сигналы
    bone_selected = pyqtSignal(str, bool)  # Имя кости, additive selection
    bone_visibility_changed = pyqtSignal(str, bool)  # Имя кости, видимость
    bone_lock_changed = pyqtSignal(str, bool)  # Имя кости, заблокированность

    def __init__(self, parent=None):
        super().__init__(parent)

        self.skeleton: Optional[Skeleton] = None
        self.bone_items: Dict[str, BoneTreeItem] = {}

        # Настройка виджета
        self.setHeaderLabels(["Кость", "Тип", "Длина", "Видим.", "Блок."])
        self.setColumnWidth(0, 200)
        self.setColumnWidth(1, 100)
        self.setColumnWidth(2, 80)
        self.setColumnWidth(3, 60)
        self.setColumnWidth(4, 60)

        # Включение редактирования
        self.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.itemChanged.connect(self._on_item_changed)

        # Контекстное меню
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Настройка отображения
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # Поиск
        self.search_box = QLineEdit(self)
        self.search_box.setPlaceholderText("Поиск костей...")
        self.search_box.textChanged.connect(self._filter_bones)

        layout = QVBoxLayout()
        layout.addWidget(self.search_box)
        layout.addWidget(self)
        self.setLayout(layout)

    def set_skeleton(self, skeleton: Skeleton):
        """Устанавливает скелет для отображения"""
        self.skeleton = skeleton
        self._populate_tree()

    def _populate_tree(self):
        """Заполняет дерево костями"""
        self.clear()
        self.bone_items.clear()

        if not self.skeleton:
            return

        # Создаем элементы для всех костей
        for bone in self.skeleton.bones:
            item = BoneTreeItem(bone)
            self.bone_items[bone.name] = item

        # Строим иерархию
        for bone in self.skeleton.bones:
            item = self.bone_items[bone.name]

            if bone.parent and bone.parent in self.bone_items:
                parent_item = self.bone_items[bone.parent]
                parent_item.addChild(item)
            else:
                self.addTopLevelItem(item)

        # Разворачиваем все элементы
        self.expandAll()

    def select_bone(self, bone_name: str, additive: bool = False):
        """Выбирает кость в дереве"""
        if bone_name in self.bone_items:
            item = self.bone_items[bone_name]

            if not additive:
                self.clearSelection()

            item.setSelected(True)
            self.scrollToItem(item)

    def _on_item_changed(self, item, column):
        """Обработка изменения элемента"""
        if column == 0:  # Изменение имени
            old_name = item.bone.name
            new_name = item.text(0)

            if new_name != old_name and self.skeleton:
                # Проверяем уникальность имени
                if new_name not in self.skeleton.bones:
                    self.skeleton.rename_bone(old_name, new_name)
                    # Обновляем словарь
                    self.bone_items[new_name] = self.bone_items.pop(old_name)
                else:
                    # Восстанавливаем старое имя
                    item.setText(0, old_name)

        elif column == 3:  # Изменение видимости
            visible = item.checkState(3) == Qt.CheckState.Checked
            item.bone.visible = visible
            self.bone_visibility_changed.emit(item.bone.name, visible)

        elif column == 4:  # Изменение блокировки
            locked = item.checkState(4) == Qt.CheckState.Checked
            item.bone.locked = locked
            self.bone_lock_changed.emit(item.bone.name, locked)

    def _filter_bones(self, text: str):
        """Фильтрует кости по тексту поиска"""
        if not text:
            # Показываем все кости
            for i in range(self.topLevelItemCount()):
                item = self.topLevelItem(i)
                self._set_item_visible(item, True)
            return

        search_text = text.lower()

        for bone_name, item in self.bone_items.items():
            visible = search_text in bone_name.lower()
            self._set_item_visible(item, visible)

    def _set_item_visible(self, item, visible: bool):
        """Устанавливает видимость элемента и его детей"""
        item.setHidden(not visible)

        # Показываем родителей если виден ребенок
        if visible and item.parent():
            self._set_item_visible(item.parent(), True)

        # Рекурсивно обрабатываем детей
        for i in range(item.childCount()):
            child = item.child(i)
            self._set_item_visible(child, visible)

    def _show_context_menu(self, position: QPoint):
        """Показывает контекстное меню"""
        selected_items = self.selectedItems()
        if not selected_items:
            return

        menu = QMenu(self)

        # Действия для выбранных костей
        rename_action = menu.addAction("Переименовать")
        rename_action.triggered.connect(self._rename_selected)

        menu.addSeparator()

        duplicate_action = menu.addAction("Дублировать")
        duplicate_action.triggered.connect(self._duplicate_selected)

        delete_action = menu.addAction("Удалить")
        delete_action.triggered.connect(self._delete_selected)

        menu.addSeparator()

        # Действия с видимостью
        show_action = menu.addAction("Показать выбранные")
        show_action.triggered.connect(lambda: self._set_selected_visibility(True))

        hide_action = menu.addAction("Скрыть выбранные")
        hide_action.triggered.connect(lambda: self._set_selected_visibility(False))

        menu.addSeparator()

        # Действия с блокировкой
        lock_action = menu.addAction("Заблокировать выбранные")
        lock_action.triggered.connect(lambda: self._set_selected_locked(True))

        unlock_action = menu.addAction("Разблокировать выбранные")
        unlock_action.triggered.connect(lambda: self._set_selected_locked(False))

        menu.exec(self.mapToGlobal(position))

    def _rename_selected(self):
        """Переименовывает выбранную кость"""
        selected = self.selectedItems()
        if selected:
            self.editItem(selected[0], 0)

    def _duplicate_selected(self):
        """Дублирует выбранные кости"""
        # Реализация дублирования
        pass

    def _delete_selected(self):
        """Удаляет выбранные кости"""
        selected_names = [item.bone.name for item in self.selectedItems()]

        # Нельзя удалять корневую кость если она единственная
        if self.skeleton:
            root_bone = self.skeleton.get_root_bone()
            if root_bone and root_bone.name in selected_names and len(self.skeleton.bones) == 1:
                QMessageBox.warning(self, "Ошибка", "Нельзя удалить единственную корневую кость")
                return

            for bone_name in selected_names:
                if bone_name != root_bone.name or len(self.skeleton.bones) > 1:
                    self.skeleton.remove_bone(bone_name)

        self._populate_tree()

    def _set_selected_visibility(self, visible: bool):
        """Устанавливает видимость выбранных костей"""
        for item in self.selectedItems():
            item.setCheckState(3, Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)

    def _set_selected_locked(self, locked: bool):
        """Устанавливает блокировку выбранных костей"""
        for item in self.selectedItems():
            item.setCheckState(4, Qt.CheckState.Checked if locked else Qt.CheckState.Unchecked)


class BoneTreeItem(QTreeWidgetItem):
    """Элемент дерева для кости"""

    def __init__(self, bone: Bone, parent=None):
        super().__init__(parent)
        self.bone = bone

        # Настройка отображения
        self.setText(0, bone.name)
        self.setText(1, bone.joint_type.value)
        self.setText(2, f"{bone.length:.3f}")

        # Чекбоксы
        self.setCheckState(3, Qt.CheckState.Checked if bone.visible else Qt.CheckState.Unchecked)
        self.setCheckState(4, Qt.CheckState.Checked if bone.locked else Qt.CheckState.Unchecked)

        # Цвет в зависимости от состояния
        self._update_appearance()

    def _update_appearance(self):
        """Обновляет внешний вид элемента"""
        if not self.bone.enabled:
            self.setForeground(0, QBrush(QColor(150, 150, 150)))
        elif self.bone.locked:
            self.setForeground(0, QBrush(QColor(200, 150, 150)))
        else:
            self.setForeground(0, QBrush(QColor(220, 220, 220)))


class ConstraintEditorWidget(QWidget):
    """Виджет для редактирования ограничений"""

    constraint_changed = pyqtSignal(str, BoneConstraint)  # Имя кости, ограничение
    constraint_added = pyqtSignal(str, BoneConstraint)  # Имя кости, новое ограничение
    constraint_removed = pyqtSignal(str, str)  # Имя кости, имя ограничения

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_bone: Optional[str] = None
        self.constraints: Dict[str, List[BoneConstraint]] = {}

        self._init_ui()

    def _init_ui(self):
        """Инициализирует интерфейс"""
        layout = QVBoxLayout(self)

        # Заголовок
        title_label = QLabel("Редактор ограничений")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)

        # Выбор кости
        bone_layout = QHBoxLayout()
        bone_layout.addWidget(QLabel("Кость:"))
        self.bone_combo = QComboBox()
        self.bone_combo.currentTextChanged.connect(self._on_bone_changed)
        bone_layout.addWidget(self.bone_combo)
        layout.addLayout(bone_layout)

        # Список ограничений
        self.constraint_list = QListWidget()
        self.constraint_list.itemClicked.connect(self._on_constraint_selected)
        layout.addWidget(self.constraint_list)

        # Кнопки управления ограничениями
        button_layout = QHBoxLayout()

        add_button = QPushButton("+ Добавить")
        add_button.clicked.connect(self._add_constraint)
        button_layout.addWidget(add_button)

        remove_button = QPushButton("- Удалить")
        remove_button.clicked.connect(self._remove_constraint)
        button_layout.addWidget(remove_button)

        layout.addLayout(button_layout)

        # Редактор свойств ограничения
        self.property_editor = self._create_property_editor()
        layout.addWidget(self.property_editor)

        # Информация
        self.info_label = QLabel("Выберите кость и ограничение для редактирования")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

    def _create_property_editor(self) -> QWidget:
        """Создает редактор свойств ограничения"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Название ограничения
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Название:"))
        self.name_edit = QLineEdit()
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)

        # Тип ограничения
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Тип:"))
        self.type_combo = QComboBox()
        for constraint_type in ConstraintType:
            self.type_combo.addItem(constraint_type.value)
        self.type_combo.currentTextChanged.connect(self._on_constraint_type_changed)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        # Target кость
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target кость:"))
        self.target_combo = QComboBox()
        target_layout.addWidget(self.target_combo)
        layout.addLayout(target_layout)

        # Включено
        self.enabled_check = QCheckBox("Включено")
        layout.addWidget(self.enabled_check)

        # Influence
        influence_layout = QHBoxLayout()
        influence_layout.addWidget(QLabel("Влияние:"))
        self.influence_slider = QSlider(Qt.Orientation.Horizontal)
        self.influence_slider.setRange(0, 100)
        self.influence_slider.setValue(100)
        influence_layout.addWidget(self.influence_slider)
        self.influence_label = QLabel("1.00")
        influence_layout.addWidget(self.influence_label)
        layout.addLayout(influence_layout)

        # Дополнительные параметры (зависит от типа)
        self.param_stack = QStackedWidget()
        layout.addWidget(self.param_stack)

        # Кнопка применения
        apply_button = QPushButton("Применить изменения")
        apply_button.clicked.connect(self._apply_changes)
        layout.addWidget(apply_button)

        return widget

    def set_bones(self, bone_names: List[str]):
        """Устанавливает список костей"""
        self.bone_combo.clear()
        self.bone_combo.addItems(bone_names)

        self.target_combo.clear()
        self.target_combo.addItems(bone_names)
        self.target_combo.insertItem(0, "")  # Пустой элемент

    def set_bone_constraints(self, bone_name: str, constraints: List[BoneConstraint]):
        """Устанавливает ограничения для кости"""
        self.constraints[bone_name] = constraints

        if bone_name == self.current_bone:
            self._update_constraint_list()

    def _on_bone_changed(self, bone_name: str):
        """Обработка изменения выбранной кости"""
        self.current_bone = bone_name
        self._update_constraint_list()
        self.property_editor.setEnabled(False)

    def _update_constraint_list(self):
        """Обновляет список ограничений"""
        self.constraint_list.clear()

        if self.current_bone and self.current_bone in self.constraints:
            for constraint in self.constraints[self.current_bone]:
                item = QListWidgetItem(constraint.name)
                item.setData(Qt.ItemDataRole.UserRole, constraint)

                # Цвет в зависимости от типа
                if constraint.is_ik_constraint():
                    item.setForeground(QColor(255, 150, 50))
                elif constraint.is_limit_constraint():
                    item.setForeground(QColor(255, 100, 100))
                else:
                    item.setForeground(QColor(100, 200, 255))

                self.constraint_list.addItem(item)

    def _on_constraint_selected(self, item):
        """Обработка выбора ограничения"""
        constraint = item.data(Qt.ItemDataRole.UserRole)
        if constraint:
            self._load_constraint_properties(constraint)

    def _load_constraint_properties(self, constraint: BoneConstraint):
        """Загружает свойства ограничения в редактор"""
        self.name_edit.setText(constraint.name)
        self.type_combo.setCurrentText(constraint.constraint_type.value)
        self.target_combo.setCurrentText(constraint.target_bone or "")
        self.enabled_check.setChecked(constraint.enabled)
        self.influence_slider.setValue(int(constraint.influence * 100))
        self.influence_label.setText(f"{constraint.influence:.2f}")

        # Показываем соответствующие параметры
        self._show_constraint_params(constraint)

        self.property_editor.setEnabled(True)

    def _show_constraint_params(self, constraint: BoneConstraint):
        """Показывает параметры для конкретного типа ограничения"""
        # Очищаем стек
        while self.param_stack.count() > 0:
            widget = self.param_stack.widget(0)
            self.param_stack.removeWidget(widget)
            widget.deleteLater()

        # Создаем виджет параметров в зависимости от типа
        if constraint.constraint_type in [ConstraintType.LIMIT_ROTATION, ConstraintType.LIMIT_LOCATION]:
            widget = self._create_limit_params_widget(constraint)
        elif constraint.constraint_type == ConstraintType.IK:
            widget = self._create_ik_params_widget(constraint)
        else:
            widget = QLabel("Нет дополнительных параметров")

        self.param_stack.addWidget(widget)
        self.param_stack.setCurrentWidget(widget)

    def _create_limit_params_widget(self, constraint: BoneConstraint) -> QWidget:
        """Создает виджет параметров для ограничений лимита"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Оси
        axes_group = QGroupBox("Ограничения по осям")
        axes_layout = QVBoxLayout()

        for axis, min_val, max_val, use_limit in [
            ("X", constraint.min_x, constraint.max_x, constraint.use_limit_x),
            ("Y", constraint.min_y, constraint.max_y, constraint.use_limit_y),
            ("Z", constraint.min_z, constraint.max_z, constraint.use_limit_z)
        ]:
            axis_widget = self._create_axis_limit_widget(axis, min_val, max_val, use_limit)
            axes_layout.addWidget(axis_widget)

        axes_group.setLayout(axes_layout)
        layout.addWidget(axes_group)

        return widget

    def _create_axis_limit_widget(self, axis: str, min_val: float, max_val: float, use_limit: bool) -> QWidget:
        """Создает виджет ограничения для одной оси"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Чекбокс
        check = QCheckBox(f"Ось {axis}")
        check.setChecked(use_limit)
        layout.addWidget(check)

        # Минимальное значение
        layout.addWidget(QLabel("Min:"))
        min_edit = QDoubleSpinBox()
        min_edit.setRange(-360, 360)
        min_edit.setValue(min_val)
        min_edit.setSuffix("°")
        layout.addWidget(min_edit)

        # Максимальное значение
        layout.addWidget(QLabel("Max:"))
        max_edit = QDoubleSpinBox()
        max_edit.setRange(-360, 360)
        max_edit.setValue(max_val)
        max_edit.setSuffix("°")
        layout.addWidget(max_edit)

        return widget

    def _create_ik_params_widget(self, constraint: BoneConstraint) -> QWidget:
        """Создает виджет параметров для IK ограничения"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Длина цепи
        chain_layout = QHBoxLayout()
        chain_layout.addWidget(QLabel("Длина цепи:"))
        chain_spin = QSpinBox()
        chain_spin.setRange(1, 10)
        chain_spin.setValue(constraint.chain_length)
        chain_layout.addWidget(chain_spin)
        layout.addLayout(chain_layout)

        # Итерации
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Итерации:"))
        iter_spin = QSpinBox()
        iter_spin.setRange(1, 100)
        iter_spin.setValue(constraint.iterations)
        iter_layout.addWidget(iter_spin)
        layout.addLayout(iter_layout)

        # Точность
        tolerance_layout = QHBoxLayout()
        tolerance_layout.addWidget(QLabel("Точность:"))
        tolerance_spin = QDoubleSpinBox()
        tolerance_spin.setRange(0.001, 1.0)
        tolerance_spin.setValue(constraint.tolerance)
        tolerance_spin.setDecimals(3)
        tolerance_layout.addWidget(tolerance_spin)
        layout.addLayout(tolerance_layout)

        # Дополнительные опции
        stretch_check = QCheckBox("Растяжение")
        stretch_check.setChecked(constraint.use_stretch)
        layout.addWidget(stretch_check)

        rotation_check = QCheckBox("Использовать вращение")
        rotation_check.setChecked(constraint.use_rotation)
        layout.addWidget(rotation_check)

        return widget

    def _add_constraint(self):
        """Добавляет новое ограничение"""
        if not self.current_bone:
            return

        # Создаем ограничение по умолчанию
        constraint = BoneConstraint(
            name=f"Constraint_{len(self.constraints.get(self.current_bone, [])) + 1}",
            constraint_type=ConstraintType.LIMIT_ROTATION
        )

        # Добавляем в список
        if self.current_bone not in self.constraints:
            self.constraints[self.current_bone] = []

        self.constraints[self.current_bone].append(constraint)
        self._update_constraint_list()

        # Выбираем новое ограничение
        self.constraint_list.setCurrentRow(self.constraint_list.count() - 1)

        # Сигнализируем
        self.constraint_added.emit(self.current_bone, constraint)

    def _remove_constraint(self):
        """Удаляет выбранное ограничение"""
        current_item = self.constraint_list.currentItem()
        if not current_item or not self.current_bone:
            return

        constraint = current_item.data(Qt.ItemDataRole.UserRole)
        if constraint and self.current_bone in self.constraints:
            self.constraints[self.current_bone].remove(constraint)
            self._update_constraint_list()
            self.property_editor.setEnabled(False)

            # Сигнализируем
            self.constraint_removed.emit(self.current_bone, constraint.name)

    def _on_constraint_type_changed(self, type_str: str):
        """Обработка изменения типа ограничения"""
        # Обновляем отображаемые параметры
        current_item = self.constraint_list.currentItem()
        if current_item:
            constraint = current_item.data(Qt.ItemDataRole.UserRole)
            if constraint:
                constraint.constraint_type = ConstraintType(type_str)
                self._show_constraint_params(constraint)

    def _apply_changes(self):
        """Применяет изменения к текущему ограничению"""
        current_item = self.constraint_list.currentItem()
        if not current_item or not self.current_bone:
            return

        constraint = current_item.data(Qt.ItemDataRole.UserRole)
        if not constraint:
            return

        # Обновляем свойства
        constraint.name = self.name_edit.text()
        constraint.constraint_type = ConstraintType(self.type_combo.currentText())
        constraint.target_bone = self.target_combo.currentText() or None
        constraint.enabled = self.enabled_check.isChecked()
        constraint.influence = self.influence_slider.value() / 100.0

        # Обновляем отображение
        current_item.setText(constraint.name)

        # Сигнализируем
        self.constraint_changed.emit(self.current_bone, constraint)


class PoseLibraryWidget(QWidget):
    """Виджет библиотеки поз"""

    pose_selected = pyqtSignal(str)  # Имя позы
    pose_saved = pyqtSignal(str, dict)  # Имя позы, данные позы

    def __init__(self, parent=None):
        super().__init__(parent)

        self.poses: Dict[str, dict] = {}
        self.current_skeleton: Optional[Skeleton] = None

        self._init_ui()

    def _init_ui(self):
        """Инициализирует интерфейс"""
        layout = QVBoxLayout(self)

        # Заголовок
        title_label = QLabel("Библиотека поз")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)

        # Список поз
        self.pose_list = QListWidget()
        self.pose_list.itemClicked.connect(self._on_pose_selected)
        layout.addWidget(self.pose_list)

        # Предпросмотр позы
        self.preview_widget = QLabel("Предпросмотр")
        self.preview_widget.setMinimumHeight(100)
        self.preview_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_widget.setStyleSheet("border: 1px solid #555; background: #333;")
        layout.addWidget(self.preview_widget)

        # Управление
        button_layout = QHBoxLayout()

        save_button = QPushButton("Сохранить позу")
        save_button.clicked.connect(self._save_current_pose)
        button_layout.addWidget(save_button)

        apply_button = QPushButton("Применить позу")
        apply_button.clicked.connect(self._apply_selected_pose)
        button_layout.addWidget(apply_button)

        delete_button = QPushButton("Удалить позу")
        delete_button.clicked.connect(self._delete_selected_pose)
        button_layout.addWidget(delete_button)

        layout.addLayout(button_layout)

    def set_skeleton(self, skeleton: Skeleton):
        """Устанавливает скелет для работы с позами"""
        self.current_skeleton = skeleton

    def load_poses(self, poses: Dict[str, dict]):
        """Загружает позы"""
        self.poses = poses
        self._update_pose_list()

    def _update_pose_list(self):
        """Обновляет список поз"""
        self.pose_list.clear()

        for pose_name in sorted(self.poses.keys()):
            item = QListWidgetItem(pose_name)
            self.pose_list.addItem(item)

    def _on_pose_selected(self, item):
        """Обработка выбора позы"""
        pose_name = item.text()
        self.pose_selected.emit(pose_name)

        # Показываем предпросмотр
        if pose_name in self.poses:
            self._show_pose_preview(pose_name)

    def _show_pose_preview(self, pose_name: str):
        """Показывает предпросмотр позы"""
        # Здесь должна быть визуализация позы
        # Для простоты показываем текст
        pose_data = self.poses[pose_name]
        preview_text = f"{pose_name}\n{len(pose_data.get('bone_rotations', {}))} костей"
        self.preview_widget.setText(preview_text)

    def _save_current_pose(self):
        """Сохраняет текущую позу"""
        if not self.current_skeleton:
            QMessageBox.warning(self, "Ошибка", "Скелет не установлен")
            return

        # Запрашиваем имя позы
        name, ok = QInputDialog.getText(self, "Сохранение позы", "Введите имя позы:")
        if not ok or not name:
            return

        # Проверяем уникальность имени
        if name in self.poses:
            reply = QMessageBox.question(
                self, "Перезаписать?",
                f"Поза '{name}' уже существует. Перезаписать?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Сохраняем позу
        pose_data = self._capture_current_pose()
        self.poses[name] = pose_data
        self._update_pose_list()

        # Сигнализируем
        self.pose_saved.emit(name, pose_data)

        QMessageBox.information(self, "Успех", f"Поза '{name}' сохранена")

    def _capture_current_pose(self) -> dict:
        """Захватывает текущую позу скелета"""
        pose_data = {
            "timestamp": time.time(),
            "bone_rotations": {},
            "bone_positions": {}
        }

        if self.current_skeleton:
            for bone in self.current_skeleton.bones:
                pose_data["bone_rotations"][bone.name] = bone.rest_rotation.tolist()
                pose_data["bone_positions"][bone.name] = bone.rest_position.tolist()

        return pose_data

    def _apply_selected_pose(self):
        """Применяет выбранную позу"""
        current_item = self.pose_list.currentItem()
        if not current_item:
            return

        pose_name = current_item.text()
        if pose_name in self.poses:
            self._apply_pose_data(self.poses[pose_name])

    def _apply_pose_data(self, pose_data: dict):
        """Применяет данные позы к скелету"""
        if not self.current_skeleton:
            return

        # Применяем вращения
        bone_rotations = pose_data.get("bone_rotations", {})
        for bone_name, rotation in bone_rotations.items():
            bone = self.current_skeleton.get_bone(bone_name)
            if bone:
                bone.set_rest_rotation(np.array(rotation))

        # Применяем позиции (только для корневых костей)
        bone_positions = pose_data.get("bone_positions", {})
        for bone_name, position in bone_positions.items():
            bone = self.current_skeleton.get_bone(bone_name)
            if bone and bone.is_root:
                bone.set_rest_position(np.array(position))

    def _delete_selected_pose(self):
        """Удаляет выбранную позу"""
        current_item = self.pose_list.currentItem()
        if not current_item:
            return

        pose_name = current_item.text()

        reply = QMessageBox.question(
            self, "Удалить позу?",
            f"Удалить позу '{pose_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            del self.poses[pose_name]
            self._update_pose_list()


class SkeletonEditor(QMainWindow):
    """Главное окно редактора скелета"""

    def __init__(self, skeleton: Optional[Skeleton] = None):
        super().__init__()

        self.skeleton = skeleton or Skeleton("NewSkeleton")
        self.current_file: Optional[str] = None
        self.modified = False

        # Инициализация UI
        self._init_ui()
        self._create_menu()
        self._create_toolbars()
        self._connect_signals()

        # Загрузка скелета
        self._load_skeleton()

        # Настройка окна
        self.setWindowTitle("Mocap Pro - Skeleton Editor")
        self.setGeometry(100, 100, 1600, 900)

        # Статус бар
        self.statusBar().showMessage("Готов")

    def _init_ui(self):
        """Инициализирует интерфейс"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Левая панель - иерархия и свойства
        left_panel = QSplitter(Qt.Orientation.Vertical)

        # Иерархия костей
        self.bone_hierarchy = BoneHierarchyWidget()
        left_panel.addWidget(self.bone_hierarchy)

        # Редактор ограничений
        self.constraint_editor = ConstraintEditorWidget()
        left_panel.addWidget(self.constraint_editor)

        # Библиотека поз
        self.pose_library = PoseLibraryWidget()
        left_panel.addWidget(self.pose_library)

        left_panel.setSizes([300, 300, 200])

        # Центральная панель - 3D вьюпорт
        self.viewport = SkeletonViewport()

        # Правая панель - свойства и инструменты
        right_panel = QSplitter(Qt.Orientation.Vertical)

        # Инструменты редактирования
        self.tools_widget = self._create_tools_widget()
        right_panel.addWidget(self.tools_widget)

        # Свойства кости
        self.bone_properties = self._create_bone_properties_widget()
        right_panel.addWidget(self.bone_properties)

        # Статистика
        self.stats_widget = self._create_stats_widget()
        right_panel.addWidget(self.stats_widget)

        right_panel.setSizes([200, 300, 100])

        # Добавляем панели в основной layout
        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(self.viewport, 5)
        main_layout.addWidget(right_panel, 2)

    def _create_tools_widget(self) -> QWidget:
        """Создает виджет инструментов"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Режимы редактирования
        mode_group = QGroupBox("Режим редактирования")
        mode_layout = QVBoxLayout()

        self.mode_buttons = {}
        for mode in EditMode:
            button = QRadioButton(mode.value.title())
            button.setChecked(mode == EditMode.SELECT)
            button.toggled.connect(lambda checked, m=mode: self._on_mode_changed(m) if checked else None)
            mode_layout.addWidget(button)
            self.mode_buttons[mode] = button

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Манипуляторы
        manip_group = QGroupBox("Манипулятор")
        manip_layout = QVBoxLayout()

        self.translate_radio = QRadioButton("Перемещение (G)")
        self.translate_radio.setChecked(True)
        manip_layout.addWidget(self.translate_radio)

        self.rotate_radio = QRadioButton("Вращение (Ctrl+R)")
        manip_layout.addWidget(self.rotate_radio)

        self.scale_radio = QRadioButton("Масштаб (Ctrl+S)")
        manip_layout.addWidget(self.scale_radio)

        manip_group.setLayout(manip_layout)
        layout.addWidget(manip_group)

        # Отображение
        view_group = QGroupBox("Отображение")
        view_layout = QVBoxLayout()

        self.show_grid_check = QCheckBox("Сетка")
        self.show_grid_check.setChecked(True)
        view_layout.addWidget(self.show_grid_check)

        self.show_axes_check = QCheckBox("Оси")
        self.show_axes_check.setChecked(True)
        view_layout.addWidget(self.show_axes_check)

        self.show_names_check = QCheckBox("Имена костей")
        self.show_names_check.setChecked(True)
        view_layout.addWidget(self.show_names_check)

        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        # Создание костей
        create_group = QGroupBox("Создание костей")
        create_layout = QVBoxLayout()

        self.new_bone_button = QPushButton("Новая кость")
        self.new_bone_button.clicked.connect(self._create_new_bone)
        create_layout.addWidget(self.new_bone_button)

        self.connect_bones_button = QPushButton("Соединить кости")
        self.connect_bones_button.clicked.connect(self._connect_bones)
        create_layout.addWidget(self.connect_bones_button)

        create_group.setLayout(create_layout)
        layout.addWidget(create_group)

        layout.addStretch()

        return widget

    def _create_bone_properties_widget(self) -> QWidget:
        """Создает виджет свойств кости"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Заголовок
        self.bone_title = QLabel("Свойства кости")
        self.bone_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.bone_title)

        # Имя кости
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Имя:"))
        self.bone_name_edit = QLineEdit()
        self.bone_name_edit.editingFinished.connect(self._on_bone_name_changed)
        name_layout.addWidget(self.bone_name_edit)
        layout.addLayout(name_layout)

        # Тип сустава
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Тип:"))
        self.bone_type_combo = QComboBox()
        for joint_type in JointType:
            self.bone_type_combo.addItem(joint_type.value)
        self.bone_type_combo.currentTextChanged.connect(self._on_bone_type_changed)
        type_layout.addWidget(self.bone_type_combo)
        layout.addLayout(type_layout)

        # Длина кости
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Длина:"))
        self.bone_length_spin = QDoubleSpinBox()
        self.bone_length_spin.setRange(0.01, 10.0)
        self.bone_length_spin.setDecimals(3)
        self.bone_length_spin.valueChanged.connect(self._on_bone_length_changed)
        length_layout.addWidget(self.bone_length_spin)
        length_layout.addWidget(QLabel("м"))
        layout.addLayout(length_layout)

        # Родительская кость
        parent_layout = QHBoxLayout()
        parent_layout.addWidget(QLabel("Родитель:"))
        self.bone_parent_combo = QComboBox()
        self.bone_parent_combo.currentTextChanged.connect(self._on_bone_parent_changed)
        parent_layout.addWidget(self.bone_parent_combo)
        layout.addLayout(parent_layout)

        # Флаги
        flags_layout = QVBoxLayout()

        self.bone_enabled_check = QCheckBox("Включена")
        self.bone_enabled_check.stateChanged.connect(self._on_bone_enabled_changed)
        flags_layout.addWidget(self.bone_enabled_check)

        self.bone_visible_check = QCheckBox("Видима")
        self.bone_visible_check.stateChanged.connect(self._on_bone_visible_changed)
        flags_layout.addWidget(self.bone_visible_check)

        self.bone_locked_check = QCheckBox("Заблокирована")
        self.bone_locked_check.stateChanged.connect(self._on_bone_locked_changed)
        flags_layout.addWidget(self.bone_locked_check)

        layout.addLayout(flags_layout)

        # Позиция и вращение
        transform_group = QGroupBox("Трансформация")
        transform_layout = QFormLayout()

        self.bone_pos_x = QDoubleSpinBox()
        self.bone_pos_y = QDoubleSpinBox()
        self.bone_pos_z = QDoubleSpinBox()

        for spin, axis in [(self.bone_pos_x, "X"), (self.bone_pos_y, "Y"), (self.bone_pos_z, "Z")]:
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(3)
            spin.valueChanged.connect(self._on_bone_position_changed)
            transform_layout.addRow(f"Позиция {axis}:", spin)

        transform_group.setLayout(transform_layout)
        layout.addWidget(transform_group)

        layout.addStretch()

        return widget

    def _create_stats_widget(self) -> QWidget:
        """Создает виджет статистики"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Заголовок
        stats_title = QLabel("Статистика")
        stats_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(stats_title)

        # Информация
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        # Обновляем статистику
        self._update_stats()

        return widget

    def _create_menu(self):
        """Создает меню"""
        menubar = self.menuBar()

        # Файл
        file_menu = menubar.addMenu("Файл")

        new_action = QAction("Новый", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_skeleton)
        file_menu.addAction(new_action)

        open_action = QAction("Открыть...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_skeleton)
        file_menu.addAction(open_action)

        save_action = QAction("Сохранить", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_skeleton)
        file_menu.addAction(save_action)

        save_as_action = QAction("Сохранить как...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._save_skeleton_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        import_action = QAction("Импорт пресета...", self)
        import_action.triggered.connect(self._import_preset)
        file_menu.addAction(import_action)

        export_action = QAction("Экспорт скелета...", self)
        export_action.triggered.connect(self._export_skeleton)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Правка
        edit_menu = menubar.addMenu("Правка")

        undo_action = QAction("Отменить", self)
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)

        redo_action = QAction("Повторить", self)
        redo_action.setShortcut("Ctrl+Y")
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        select_all_action = QAction("Выделить все", self)
        select_all_action.setShortcut("Ctrl+A")
        select_all_action.triggered.connect(self._select_all_bones)
        edit_menu.addAction(select_all_action)

        deselect_all_action = QAction("Снять выделение", self)
        deselect_all_action.setShortcut("Esc")
        deselect_all_action.triggered.connect(self._deselect_all_bones)
        edit_menu.addAction(deselect_all_action)

        # Вид
        view_menu = menubar.addMenu("Вид")

        reset_view_action = QAction("Сброс вида", self)
        reset_view_action.setShortcut("R")
        reset_view_action.triggered.connect(self.viewport.reset_camera)
        view_menu.addAction(reset_view_action)

        focus_selection_action = QAction("Фокус на выделенное", self)
        focus_selection_action.setShortcut("F")
        focus_selection_action.triggered.connect(self.viewport.focus_on_selection)
        view_menu.addAction(focus_selection_action)

        # Скелет
        skeleton_menu = menubar.addMenu("Скелет")

        mirror_action = QAction("Зеркалить скелет", self)
        mirror_action.triggered.connect(self._mirror_skeleton)
        skeleton_menu.addAction(mirror_action)

        optimize_action = QAction("Оптимизировать иерархию", self)
        optimize_action.triggered.connect(self._optimize_hierarchy)
        skeleton_menu.addAction(optimize_action)

        # Справка
        help_menu = menubar.addMenu("Справка")

        about_action = QAction("О программе", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_toolbars(self):
        """Создает панели инструментов"""
        # Основная панель инструментов
        main_toolbar = QToolBar("Основные инструменты")
        self.addToolBar(main_toolbar)

        # Кнопки режимов
        mode_actions = QActionGroup(self)

        for mode in [EditMode.SELECT, EditMode.MOVE, EditMode.ROTATE, EditMode.SCALE]:
            action = QAction(mode.value.title(), self)
            action.setCheckable(True)
            action.setChecked(mode == EditMode.SELECT)
            action.triggered.connect(lambda checked, m=mode: self._on_mode_changed(m) if checked else None)
            mode_actions.addAction(action)
            main_toolbar.addAction(action)

        main_toolbar.addSeparator()

        # Кнопки манипуляторов
        manip_actions = QActionGroup(self)

        translate_action = QAction("Перемещение", self)
        translate_action.setCheckable(True)
        translate_action.setChecked(True)
        translate_action.triggered.connect(lambda: self.viewport.set_manipulator_type("TRANSLATE"))
        manip_actions.addAction(translate_action)
        main_toolbar.addAction(translate_action)

        rotate_action = QAction("Вращение", self)
        rotate_action.setCheckable(True)
        rotate_action.triggered.connect(lambda: self.viewport.set_manipulator_type("ROTATE"))
        manip_actions.addAction(rotate_action)
        main_toolbar.addAction(rotate_action)

        scale_action = QAction("Масштаб", self)
        scale_action.setCheckable(True)
        scale_action.triggered.connect(lambda: self.viewport.set_manipulator_type("SCALE"))
        manip_actions.addAction(scale_action)
        main_toolbar.addAction(scale_action)

    def _connect_signals(self):
        """Подключает сигналы"""
        # Вьюпорт -> Иерархия
        self.viewport.bone_selected.connect(self.bone_hierarchy.select_bone)

        # Иерархия -> Вьюпорт
        self.bone_hierarchy.itemSelectionChanged.connect(self._on_hierarchy_selection_changed)

        # Свойства отображения
        self.show_grid_check.stateChanged.connect(
            lambda state: setattr(self.viewport, 'show_grid', state == Qt.CheckState.Checked)
        )
        self.show_axes_check.stateChanged.connect(
            lambda state: setattr(self.viewport, 'show_axes', state == Qt.CheckState.Checked)
        )
        self.show_names_check.stateChanged.connect(
            lambda state: setattr(self.viewport, 'show_bone_names', state == Qt.CheckState.Checked)
        )

        # Манипуляторы
        self.translate_radio.toggled.connect(
            lambda checked: self.viewport.set_manipulator_type("TRANSLATE") if checked else None
        )
        self.rotate_radio.toggled.connect(
            lambda checked: self.viewport.set_manipulator_type("ROTATE") if checked else None
        )
        self.scale_radio.toggled.connect(
            lambda checked: self.viewport.set_manipulator_type("SCALE") if checked else None
        )

    def _load_skeleton(self):
        """Загружает скелет во все виджеты"""
        # Вьюпорт
        self.viewport.set_skeleton(self.skeleton)

        # Иерархия
        self.bone_hierarchy.set_skeleton(self.skeleton)

        # Свойства костей
        self._update_bone_properties_combo()

        # Редактор ограничений
        bone_names = [bone.name for bone in self.skeleton.bones]
        self.constraint_editor.set_bones(bone_names)

        # Библиотека поз
        self.pose_library.set_skeleton(self.skeleton)

        # Обновляем статистику
        self._update_stats()

        self.modified = False

    def _update_bone_properties_combo(self):
        """Обновляет комбобоксы выбора костей"""
        bone_names = [bone.name for bone in self.skeleton.bones]

        # Комбобокс выбора родителя
        self.bone_parent_combo.clear()
        self.bone_parent_combo.addItem("")  # Пустой элемент
        self.bone_parent_combo.addItems(bone_names)

    def _update_stats(self):
        """Обновляет статистику"""
        stats_text = f"""
        Костей: {len(self.skeleton.bones)}
        Иерархия: {self._get_hierarchy_depth()} уровней
        Корневая кость: {self.skeleton.get_root_bone().name if self.skeleton.get_root_bone() else 'Нет'}
        Модифицирован: {'Да' if self.modified else 'Нет'}
        """
        self.stats_label.setText(stats_text)

    def _get_hierarchy_depth(self) -> int:
        """Вычисляет глубину иерархии"""

        def get_depth(bone_name: str) -> int:
            bone = self.skeleton.get_bone(bone_name)
            if not bone or not bone.children:
                return 1

            max_depth = 0
            for child_name in bone.children:
                depth = get_depth(child_name)
                max_depth = max(max_depth, depth)

            return max_depth + 1

        root = self.skeleton.get_root_bone()
        return get_depth(root.name) if root else 0

    def _on_mode_changed(self, mode: EditMode):
        """Обработка изменения режима редактирования"""
        self.viewport.set_edit_mode(mode)

        # Обновляем радиокнопки
        for m, button in self.mode_buttons.items():
            button.setChecked(m == mode)

    def _on_hierarchy_selection_changed(self):
        """Обработка изменения выделения в иерархии"""
        selected_items = self.bone_hierarchy.selectedItems()
        if selected_items:
            bone_names = [item.bone.name for item in selected_items]

            # Выделяем в вьюпорте
            self.viewport.select_bones(bone_names, additive=False)

            # Показываем свойства первой выбранной кости
            self._show_bone_properties(selected_items[0].bone)

    def _show_bone_properties(self, bone: Bone):
        """Показывает свойства кости"""
        self.bone_title.setText(f"Свойства: {bone.name}")
        self.bone_name_edit.setText(bone.name)
        self.bone_type_combo.setCurrentText(bone.joint_type.value)
        self.bone_length_spin.setValue(bone.length)

        # Родитель
        if bone.parent:
            self.bone_parent_combo.setCurrentText(bone.parent)
        else:
            self.bone_parent_combo.setCurrentIndex(0)  # Пустой элемент

        # Флаги
        self.bone_enabled_check.setChecked(bone.enabled)
        self.bone_visible_check.setChecked(bone.visible)
        self.bone_locked_check.setChecked(bone.locked)

        # Позиция
        self.bone_pos_x.setValue(bone.rest_position[0])
        self.bone_pos_y.setValue(bone.rest_position[1])
        self.bone_pos_z.setValue(bone.rest_position[2])

    def _on_bone_name_changed(self):
        """Обработка изменения имени кости"""
        selected_items = self.bone_hierarchy.selectedItems()
        if selected_items:
            bone = selected_items[0].bone
            new_name = self.bone_name_edit.text()

            if new_name != bone.name:
                self.skeleton.rename_bone(bone.name, new_name)
                self.modified = True
                self._load_skeleton()  # Перезагружаем для обновления интерфейса

    def _on_bone_type_changed(self, type_str: str):
        """Обработка изменения типа кости"""
        selected_items = self.bone_hierarchy.selectedItems()
        if selected_items:
            bone = selected_items[0].bone
            bone.joint_type = JointType(type_str)
            self.modified = True
            self.bone_hierarchy._populate_tree()  # Обновляем иерархию

    def _on_bone_length_changed(self, value: float):
        """Обработка изменения длины кости"""
        selected_items = self.bone_hierarchy.selectedItems()
        if selected_items:
            bone = selected_items[0].bone
            bone.length = value
            self.modified = True
            self.viewport.update()

    def _on_bone_parent_changed(self, parent_name: str):
        """Обработка изменения родительской кости"""
        selected_items = self.bone_hierarchy.selectedItems()
        if selected_items and parent_name != "":
            bone = selected_items[0].bone

            # Нельзя сделать родителем самого себя или своего ребенка
            if parent_name != bone.name and not self._is_descendant(bone.name, parent_name):
                bone.parent = parent_name
                self.modified = True
                self._load_skeleton()  # Перезагружаем для обновления иерархии

    def _is_descendant(self, bone_name: str, potential_parent: str) -> bool:
        """Проверяет, является ли кость потомком"""
        bone = self.skeleton.get_bone(bone_name)
        while bone and bone.parent:
            if bone.parent == potential_parent:
                return True
            bone = self.skeleton.get_bone(bone.parent)
        return False

    def _on_bone_enabled_changed(self, state: int):
        """Обработка изменения включения кости"""
        selected_items = self.bone_hierarchy.selectedItems()
        if selected_items:
            bone = selected_items[0].bone
            bone.enabled = state == Qt.CheckState.Checked
            self.modified = True
            self.viewport.update()

    def _on_bone_visible_changed(self, state: int):
        """Обработка изменения видимости кости"""
        selected_items = self.bone_hierarchy.selectedItems()
        if selected_items:
            bone = selected_items[0].bone
            bone.visible = state == Qt.CheckState.Checked
            self.modified = True
            self.viewport.update()

    def _on_bone_locked_changed(self, state: int):
        """Обработка изменения блокировки кости"""
        selected_items = self.bone_hierarchy.selectedItems()
        if selected_items:
            bone = selected_items[0].bone
            bone.locked = state == Qt.CheckState.Checked
            self.modified = True
            self.viewport.update()

    def _on_bone_position_changed(self):
        """Обработка изменения позиции кости"""
        selected_items = self.bone_hierarchy.selectedItems()
        if selected_items:
            bone = selected_items[0].bone
            new_position = np.array([
                self.bone_pos_x.value(),
                self.bone_pos_y.value(),
                self.bone_pos_z.value()
            ])

            if not np.array_equal(bone.rest_position, new_position):
                bone.set_rest_position(new_position)
                self.modified = True
                self.viewport.update()

    def _create_new_bone(self):
        """Создает новую кость"""
        # Запрашиваем имя кости
        name, ok = QInputDialog.getText(self, "Новая кость", "Введите имя кости:")
        if not ok or not name:
            return

        # Проверяем уникальность имени
        if name in [bone.name for bone in self.skeleton.bones]:
            QMessageBox.warning(self, "Ошибка", f"Кость с именем '{name}' уже существует")
            return

        # Создаем кость
        bone = Bone(
            name=name,
            joint_type=JointType.SPINE,
            rest_position=np.array([0.0, 0.0, 0.0]),
            rest_rotation=np.array([0.0, 0.0, 0.0, 1.0]),
            length=0.3
        )

        # Если есть выделенная кость, делаем её родителем
        selected_items = self.bone_hierarchy.selectedItems()
        if selected_items:
            parent_bone = selected_items[0].bone
            bone.parent = parent_bone.name

        # Добавляем кость в скелет
        self.skeleton.add_bone(bone)
        self.modified = True

        # Обновляем интерфейс
        self._load_skeleton()

        # Выделяем новую кость
        self.bone_hierarchy.select_bone(bone.name)

    def _connect_bones(self):
        """Соединяет выбранные кости"""
        selected_items = self.bone_hierarchy.selectedItems()
        if len(selected_items) != 2:
            QMessageBox.warning(self, "Ошибка", "Выберите ровно 2 кости для соединения")
            return

        bone1 = selected_items[0].bone
        bone2 = selected_items[1].bone

        # Определяем, какая кость будет родителем
        # (можно добавить диалог выбора)

        # Соединяем кости
        bone2.parent = bone1.name
        self.modified = True

        # Обновляем интерфейс
        self._load_skeleton()

    def _select_all_bones(self):
        """Выделяет все кости"""
        for bone in self.skeleton.bones:
            self.viewport.select_bone(bone.name, additive=True, clear=False)

    def _deselect_all_bones(self):
        """Снимает выделение со всех костей"""
        self.viewport.clear_selection()

    def _mirror_skeleton(self):
        """Зеркалирует скелет"""
        # Реализация зеркалирования
        pass

    def _optimize_hierarchy(self):
        """Оптимизирует иерархию скелета"""
        # Реализация оптимизации
        pass

    def _new_skeleton(self):
        """Создает новый скелет"""
        if self.modified:
            reply = QMessageBox.question(
                self, "Сохранение",
                "Сохранить изменения в текущем скелете?",
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._save_skeleton()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        # Создаем новый скелет
        self.skeleton = Skeleton("NewSkeleton")
        self.current_file = None
        self._load_skeleton()
        self.setWindowTitle("Mocap Pro - Skeleton Editor")

    def _open_skeleton(self):
        """Открывает скелет из файла"""
        if self.modified:
            reply = QMessageBox.question(
                self, "Сохранение",
                "Сохранить изменения в текущем скелете?",
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._save_skeleton()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        # Выбираем файл
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Открыть скелет", "",
            "Skeleton Files (*.json *.skel);;All Files (*)"
        )

        if file_path:
            try:
                # Загружаем скелет
                with open(file_path, 'r', encoding='utf-8') as f:
                    skeleton_data = json.load(f)

                self.skeleton = Skeleton.from_dict(skeleton_data)
                self.current_file = file_path
                self._load_skeleton()

                self.setWindowTitle(f"Mocap Pro - Skeleton Editor - {file_path}")
                self.modified = False

                self.statusBar().showMessage(f"Скелет загружен: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить скелет: {str(e)}")

    def _save_skeleton(self):
        """Сохраняет скелет"""
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self._save_skeleton_as()

    def _save_skeleton_as(self):
        """Сохраняет скелет как..."""
        # Выбираем файл
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить скелет", self.skeleton.name + ".json",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            self._save_to_file(file_path)

    def _save_to_file(self, file_path: str):
        """Сохраняет скелет в файл"""
        try:
            # Сохраняем скелет
            skeleton_data = self.skeleton.to_dict()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(skeleton_data, f, indent=2, ensure_ascii=False)

            self.current_file = file_path
            self.modified = False

            self.setWindowTitle(f"Mocap Pro - Skeleton Editor - {file_path}")
            self.statusBar().showMessage(f"Скелет сохранен: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить скелет: {str(e)}")

    def _import_preset(self):
        """Импортирует пресет скелета"""
        # Выбираем пресет
        presets = {
            "Humanoid (MediaPipe)": "humanoid_mediapipe",
            "Simplified Humanoid": "simplified_humanoid",
            "Quadruped Animal": "quadruped",
            "Facial Rig": "facial"
        }

        preset_name, ok = QInputDialog.getItem(
            self, "Импорт пресета", "Выберите пресет:",
            list(presets.keys()), 0, False
        )

        if ok and preset_name:
            try:
                # Загружаем пресет
                preset_id = presets[preset_name]
                preset_data = load_skeleton_preset(preset_id)

                # Создаем скелет из пресета
                self.skeleton = Skeleton.from_preset(preset_data)
                self.modified = True
                self._load_skeleton()

                self.statusBar().showMessage(f"Пресет загружен: {preset_name}")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить пресет: {str(e)}")

    def _export_skeleton(self):
        """Экспортирует скелет"""
        # Здесь может быть экспорт в различные форматы
        pass

    def _show_about(self):
        """Показывает информацию о программе"""
        QMessageBox.about(
            self, "О программе",
            """<b>Mocap Pro - Skeleton Editor</b><br><br>
            Профессиональный редактор скелетов для Motion Capture Pro.<br><br>
            Версия: 1.0.0<br>
            © 2024 Mocap Pro Team<br><br>
            Функции:<br>
            • Визуальное редактирование скелетов в 3D<br>
            • Иерархическая структура костей<br>
            • Настройка ограничений и IK<br>
            • Библиотека поз<br>
            • Импорт/экспорт в различные форматы"""
        )

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        if self.modified:
            reply = QMessageBox.question(
                self, "Сохранение",
                "Сохранить изменения перед закрытием?",
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._save_skeleton()
                event.accept()
            elif reply == QMessageBox.StandardButton.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# Функции для интеграции с основным приложением
def create_skeleton_editor(skeleton: Optional[Skeleton] = None, parent=None) -> SkeletonEditor:
    """
    Создает и возвращает экземпляр редактора скелета.

    Args:
        skeleton: Существующий скелет для редактирования
        parent: Родительское окно

    Returns:
        SkeletonEditor: Экземпляр редактора скелета
    """
    editor = SkeletonEditor(skeleton)
    if parent:
        editor.setParent(parent)
    return editor


def integrate_with_main_window(main_window, skeleton: Optional[Skeleton] = None):
    """
    Интегрирует редактор скелета с главным окном.

    Args:
        main_window: Главное окно приложения
        skeleton: Скелет для редактирования
    """
    editor = create_skeleton_editor(skeleton)

    # Создаем док-виджет
    dock_widget = QDockWidget("Skeleton Editor", main_window)
    dock_widget.setWidget(editor)
    dock_widget.setFeatures(
        QDockWidget.DockWidgetFeature.DockWidgetMovable |
        QDockWidget.DockWidgetFeature.DockWidgetFloatable
    )

    # Добавляем в главное окно
    main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_widget)

    return editor


# Точка входа для тестирования
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Создаем тестовый скелет
    skeleton = Skeleton("TestSkeleton")

    # Корневая кость
    root_bone = Bone(
        name="Hips",
        joint_type=JointType.SPINE,
        rest_position=np.array([0.0, 1.0, 0.0]),
        rest_rotation=np.array([0.0, 0.0, 0.0, 1.0]),
        length=0.2
    )
    skeleton.add_bone(root_bone)

    # Spine
    spine_bone = Bone(
        name="Spine",
        joint_type=JointType.SPINE,
        rest_position=np.array([0.0, 1.2, 0.0]),
        rest_rotation=np.array([0.0, 0.0, 0.0, 1.0]),
        length=0.2,
        parent="Hips"
    )
    skeleton.add_bone(spine_bone)

    # Запускаем редактор
    editor = SkeletonEditor(skeleton)
    editor.show()

    sys.exit(app.exec())