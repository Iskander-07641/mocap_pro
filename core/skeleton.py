"""
ПРОФЕССИОНАЛЬНАЯ СИСТЕМА СКЕЛЕТА ДЛЯ MOCAP
С поддержкой FK/IK, ретаргетинга, экспорта в BVH/FBX
"""

import numpy as np
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import transforms3d as tf3d
from scipy.spatial.transform import Rotation as R
import warnings

logger = logging.getLogger(__name__)


class BoneType(Enum):
    """Типы костей с приоритетами для IK"""
    ROOT = 1  # Корневая кость (таз)
    SPINE = 2  # Позвоночник
    NECK = 3  # Шея
    HEAD = 4  # Голова
    CLAVICLE = 5  # Ключица
    UPPER_ARM = 6  # Плечо
    LOWER_ARM = 7  # Предплечье
    HAND = 8  # Кисть
    UPPER_LEG = 9  # Бедро
    LOWER_LEG = 10  # Голень
    FOOT = 11  # Стопа
    TOE = 12  # Пальцы ног
    FINGER = 13  # Пальцы рук
    EXTRA = 14  # Дополнительные кости


@dataclass
class BoneTransform:
    """Полная трансформация кости в мировых и локальных координатах"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # quat w,x,y,z
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    local_matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    world_matrix: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self):
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)
        if not isinstance(self.rotation, np.ndarray):
            self.rotation = np.array(self.rotation, dtype=np.float32)
        if not isinstance(self.scale, np.ndarray):
            self.scale = np.array(self.scale, dtype=np.float32)


class Bone:
    """Улучшенная кость с поддержкой FK/IK и ретаргетинга"""

    def __init__(self,
                 name: str,
                 bone_type: BoneType,
                 parent=None,
                 length: float = 0.1,
                 local_offset: np.ndarray = None,
                 rotation_order: str = 'XYZ',
                 ik_enabled: bool = False):

        self.name = name
        self.type = bone_type
        self.parent = parent
        self.children: List['Bone'] = []

        # Трансформации
        self.transform = BoneTransform()
        self.rest_pose = BoneTransform()  # T-поза
        self.bind_pose = BoneTransform()  # Поза привязки (для скининга)

        # Геометрия
        self.length = float(length)
        self.local_offset = local_offset if local_offset is not None else np.zeros(3)
        self.rotation_order = rotation_order

        # FK/IK параметры
        self.ik_enabled = ik_enabled
        self.ik_target = None
        self.ik_chain_length = 3  # Длина IK цепи
        self.ik_weight = 1.0

        # Ограничения (constraints)
        self.constraints = {
            'rotation_limit': None,  # [min_angle, max_angle] в градусах
            'twist_limit': None,
            'swing_limit': None,
            'position_limit': None
        }

        # Для ретаргетинга
        self.retarget_map = None  # Сопоставление с другим скелетом
        self.retarget_offset = np.zeros(3)

        # Связь с landmark
        self.landmark_id = None
        self.landmark_weight = 1.0

        # Иерархия
        if parent:
            parent.add_child(self)

    def add_child(self, child_bone: 'Bone'):
        """Добавление дочерней кости"""
        if child_bone not in self.children:
            self.children.append(child_bone)
            child_bone.parent = self

    def remove_child(self, child_bone: 'Bone'):
        """Удаление дочерней кости"""
        if child_bone in self.children:
            self.children.remove(child_bone)
            child_bone.parent = None

    def get_world_position(self) -> np.ndarray:
        """Получение мировой позиции кости"""
        return self.transform.position.copy()

    def get_world_rotation(self) -> np.ndarray:
        """Получение мирового вращения (кватернион)"""
        return self.transform.rotation.copy()

    def set_local_transform(self,
                            position: np.ndarray = None,
                            rotation: np.ndarray = None,
                            update_children: bool = True):
        """Установка локальной трансформации"""
        if position is not None:
            self.transform.position = np.array(position, dtype=np.float32)
        if rotation is not None:
            self.transform.rotation = np.array(rotation, dtype=np.float32)
            # Нормализация кватерниона
            norm = np.linalg.norm(self.transform.rotation)
            if norm > 0:
                self.transform.rotation /= norm

        # Обновление матриц
        self._update_matrices()

        # Рекурсивное обновление детей
        if update_children:
            for child in self.children:
                child._update_from_parent()

    def set_world_transform(self,
                            position: np.ndarray,
                            rotation: np.ndarray,
                            update_children: bool = True):
        """Установка мировой трансформации"""
        if self.parent is None:
            self.set_local_transform(position, rotation, update_children)
        else:
            # Конвертация в локальные координаты
            parent_inv = np.linalg.inv(self.parent.transform.world_matrix)
            local_pos = parent_inv[:3, :3] @ position + parent_inv[:3, 3]

            # Для вращения нужно использовать кватернионы
            parent_rot_inv = tf3d.quaternions.qinverse(self.parent.transform.rotation)
            local_rot = tf3d.quaternions.qmult(parent_rot_inv, rotation)

            self.set_local_transform(local_pos, local_rot, update_children)

    def _update_matrices(self):
        """Обновление локальной и мировой матриц"""
        # Локальная матрица
        rot_matrix = tf3d.quaternions.quat2mat(self.transform.rotation)
        self.transform.local_matrix = np.eye(4)
        self.transform.local_matrix[:3, :3] = rot_matrix
        self.transform.local_matrix[:3, 3] = self.transform.position

        # Мировая матрица
        if self.parent is None:
            self.transform.world_matrix = self.transform.local_matrix.copy()
        else:
            self.transform.world_matrix = self.parent.transform.world_matrix @ self.transform.local_matrix

        # Извлечение позиции из мировой матрицы
        self.transform.position = self.transform.world_matrix[:3, 3].copy()

        # Извлечение вращения из мировой матрицы
        rot_mat = self.transform.world_matrix[:3, :3]
        self.transform.rotation = tf3d.quaternions.mat2quat(rot_mat)

    def _update_from_parent(self):
        """Обновление трансформации при изменении родителя"""
        self._update_matrices()
        for child in self.children:
            child._update_from_parent()

    def get_direction_vector(self) -> np.ndarray:
        """Вектор направления кости (от родителя к кости)"""
        if self.parent is None:
            return np.array([0, 1, 0])  # По умолчанию вверх

        direction = self.transform.position - self.parent.transform.position
        norm = np.linalg.norm(direction)
        if norm > 0:
            return direction / norm
        return np.array([0, 1, 0])

    def calculate_length_from_children(self) -> float:
        """Автоматический расчет длины кости на основе детей"""
        if not self.children:
            return self.length

        total_length = 0.0
        count = 0

        for child in self.children:
            dist = np.linalg.norm(child.transform.position - self.transform.position)
            if dist > 0.01:  # Игнорируем очень маленькие расстояния
                total_length += dist
                count += 1

        if count > 0:
            self.length = total_length / count

        return self.length

    def apply_ik(self, target_position: np.ndarray, iterations: int = 10) -> bool:
        """Применение инверсной кинематики (CCD алгоритм)"""
        if not self.ik_enabled or self.parent is None:
            return False

        # Получаем цепь костей для IK
        chain = self._get_ik_chain()
        if len(chain) < 2:
            return False

        for _ in range(iterations):
            # Циклическая координатная спуск (CCD)
            for i in range(len(chain) - 1, -1, -1):
                bone = chain[i]

                # Вектор от кости к текущему концу эффектора
                current_end = chain[-1].transform.position
                to_end = current_end - bone.transform.position
                to_end_norm = np.linalg.norm(to_end)
                if to_end_norm < 0.001:
                    continue

                # Вектор от кости к цели
                to_target = target_position - bone.transform.position
                to_target_norm = np.linalg.norm(to_target)
                if to_target_norm < 0.001:
                    continue

                # Вращение для выравнивания векторов
                to_end = to_end / to_end_norm
                to_target = to_target / to_target_norm

                # Ось и угол вращения
                rotation_axis = np.cross(to_end, to_target)
                axis_norm = np.linalg.norm(rotation_axis)

                if axis_norm > 0.001:
                    rotation_axis = rotation_axis / axis_norm
                    dot_product = np.clip(np.dot(to_end, to_target), -1.0, 1.0)
                    rotation_angle = np.arccos(dot_product) * self.ik_weight

                    # Создаем кватернион вращения
                    quat = tf3d.quaternions.axangle2quat(rotation_axis, rotation_angle)

                    # Применяем вращение
                    new_rot = tf3d.quaternions.qmult(bone.transform.rotation, quat)
                    bone.set_local_transform(rotation=new_rot, update_children=True)

        return True

    def _get_ik_chain(self) -> List['Bone']:
        """Получение цепи костей для IK (от текущей до корня)"""
        chain = []
        current = self

        while current is not None and len(chain) < self.ik_chain_length:
            chain.append(current)
            current = current.parent

        return list(reversed(chain))

    def to_dict(self) -> Dict:
        """Сериализация кости в словарь"""
        return {
            'name': self.name,
            'type': self.type.name,
            'parent': self.parent.name if self.parent else None,
            'length': float(self.length),
            'local_offset': self.local_offset.tolist(),
            'position': self.transform.position.tolist(),
            'rotation': self.transform.rotation.tolist(),
            'landmark_id': self.landmark_id,
            'landmark_weight': float(self.landmark_weight),
            'ik_enabled': self.ik_enabled,
            'rotation_order': self.rotation_order
        }

    @classmethod
    def from_dict(cls, data: Dict, parent_map: Dict[str, 'Bone']) -> 'Bone':
        """Десериализация кости из словаря"""
        parent = parent_map.get(data.get('parent'))

        bone = cls(
            name=data['name'],
            bone_type=BoneType[data['type']],
            parent=parent,
            length=data.get('length', 0.1),
            local_offset=np.array(data.get('local_offset', [0, 0, 0])),
            rotation_order=data.get('rotation_order', 'XYZ'),
            ik_enabled=data.get('ik_enabled', False)
        )

        bone.transform.position = np.array(data.get('position', [0, 0, 0]))
        bone.transform.rotation = np.array(data.get('rotation', [1, 0, 0, 0]))
        bone.landmark_id = data.get('landmark_id')
        bone.landmark_weight = data.get('landmark_weight', 1.0)

        # Обновляем матрицы
        bone._update_matrices()

        return bone


class ProfessionalSkeleton:
    """
    ПРОФЕССИОНАЛЬНЫЙ СКЕЛЕТ ДЛЯ MOCAP

    Особенности:
    1. Поддержка FK/IK смешивания
    2. Автоматический ретаргетинг
    3. Экспорт в BVH/FBX
    4. Кэширование матриц для производительности
    5. Поддержка нескольких поз (T-pose, bind pose, текущая)
    """

    def __init__(self, name: str = "HumanSkeleton"):
        self.name = name
        self.bones: Dict[str, Bone] = {}
        self.bone_hierarchy: Dict[str, List[str]] = {}
        self.root_bone: Optional[Bone] = None

        # Для производительности
        self._bone_cache = {}
        self._matrix_cache = {}
        self._dirty = True

        # Создание скелета по умолчанию
        self._create_human_skeleton()
        self.update_rest_pose()

        logger.info(f"ProfessionalSkeleton '{name}' created with {len(self.bones)} bones")

    def _create_human_skeleton(self):
        """Создание анатомически правильного человеческого скелета"""

        # ====== CORE (ЦЕНТРАЛЬНАЯ ОСЬ) ======
        self.root_bone = self.add_bone("Hips", BoneType.ROOT, length=0.15)

        # Позвоночник
        spine1 = self.add_bone("Spine1", BoneType.SPINE, parent="Hips", length=0.1)
        spine2 = self.add_bone("Spine2", BoneType.SPINE, parent="Spine1", length=0.1)
        spine3 = self.add_bone("Spine3", BoneType.SPINE, parent="Spine2", length=0.08)

        # Шея и голова
        neck = self.add_bone("Neck", BoneType.NECK, parent="Spine3", length=0.05)
        head = self.add_bone("Head", BoneType.HEAD, parent="Neck", length=0.1)

        # ====== UPPER BODY (ВЕРХНЯЯ ЧАСТЬ ТЕЛА) ======
        # Ключицы
        left_clavicle = self.add_bone("LeftClavicle", BoneType.CLAVICLE, parent="Spine3",
                                      length=0.05, local_offset=[-0.05, 0, 0])
        right_clavicle = self.add_bone("RightClavicle", BoneType.CLAVICLE, parent="Spine3",
                                       length=0.05, local_offset=[0.05, 0, 0])

        # Руки
        self._create_arm("Left", left_clavicle)
        self._create_arm("Right", right_clavicle)

        # ====== LOWER BODY (НИЖНЯЯ ЧАСТЬ ТЕЛА) ======
        # Ноги
        self._create_leg("Left", self.root_bone)
        self._create_leg("Right", self.root_bone)

        # Устанавливаем T-позу
        self._set_t_pose()

    def _create_arm(self, side: str, clavicle: Bone):
        """Создание руки с пальцами"""
        prefix = "Left" if side == "Left" else "Right"
        offset_x = -0.1 if side == "Left" else 0.1

        # Плечо
        upper_arm = self.add_bone(f"{prefix}UpperArm", BoneType.UPPER_ARM,
                                  parent=clavicle.name, length=0.15)

        # Локоть
        lower_arm = self.add_bone(f"{prefix}LowerArm", BoneType.LOWER_ARM,
                                  parent=upper_arm.name, length=0.12)

        # Кисть
        hand = self.add_bone(f"{prefix}Hand", BoneType.HAND,
                             parent=lower_arm.name, length=0.05)

        # Включаем IK для руки
        hand.ik_enabled = True
        hand.ik_chain_length = 3

        # Пальцы (упрощенно)
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        for finger in finger_names:
            for i in range(1, 4):  # 3 фаланги
                bone_name = f"{prefix}{finger}{i}"
                parent_name = hand.name if i == 1 else f"{prefix}{finger}{i - 1}"

                self.add_bone(bone_name, BoneType.FINGER,
                              parent=parent_name, length=0.02)

    def _create_leg(self, side: str, hips: Bone):
        """Создание ноги"""
        prefix = "Left" if side == "Left" else "Right"
        offset_x = -0.08 if side == "Left" else 0.08

        # Бедро
        upper_leg = self.add_bone(f"{prefix}UpperLeg", BoneType.UPPER_LEG,
                                  parent=hips.name, length=0.2,
                                  local_offset=[offset_x, 0, 0])

        # Колено
        lower_leg = self.add_bone(f"{prefix}LowerLeg", BoneType.LOWER_LEG,
                                  parent=upper_leg.name, length=0.18)

        # Стопа
        foot = self.add_bone(f"{prefix}Foot", BoneType.FOOT,
                             parent=lower_leg.name, length=0.05)

        # Пальцы ног
        toe = self.add_bone(f"{prefix}Toe", BoneType.TOE,
                            parent=foot.name, length=0.03)

        # Включаем IK для ноги
        foot.ik_enabled = True
        foot.ik_chain_length = 3

    def _set_t_pose(self):
        """Установка T-позы (руки в стороны)"""
        # Центральная ось вертикально
        self.root_bone.set_local_transform(position=[0, 1, 0])

        # Руки в стороны
        for bone_name in ["LeftUpperArm", "RightUpperArm"]:
            if bone_name in self.bones:
                bone = self.bones[bone_name]
                # Руки в стороны, слегка вниз
                if "Left" in bone_name:
                    bone.set_local_transform(rotation=[0.707, 0, 0, 0.707])  # 90 градусов
                else:
                    bone.set_local_transform(rotation=[0.707, 0, 0, -0.707])  # -90 градусов

        # Обновляем все трансформации
        self._update_all_transforms()

    def add_bone(self,
                 name: str,
                 bone_type: BoneType,
                 parent: str = None,
                 length: float = 0.1,
                 local_offset: np.ndarray = None,
                 **kwargs) -> Bone:
        """Добавление новой кости в скелет"""

        if name in self.bones:
            logger.warning(f"Кость '{name}' уже существует, перезаписываю")

        parent_bone = self.bones.get(parent) if parent else None

        bone = Bone(
            name=name,
            bone_type=bone_type,
            parent=parent_bone,
            length=length,
            local_offset=local_offset,
            **kwargs
        )

        self.bones[name] = bone

        # Обновление иерархии
        if parent:
            if parent not in self.bone_hierarchy:
                self.bone_hierarchy[parent] = []
            self.bone_hierarchy[parent].append(name)

        # Установка корневой кости если нужно
        if parent is None and self.root_bone is None:
            self.root_bone = bone

        self._dirty = True
        logger.debug(f"Кость '{name}' добавлена")

        return bone

    def remove_bone(self, name: str):
        """Удаление кости из скелета"""
        if name not in self.bones:
            return

        bone = self.bones[name]

        # Перепривязываем детей к родителю удаляемой кости
        for child in bone.children[:]:  # Копируем список
            child.parent = bone.parent
            if bone.parent:
                bone.parent.add_child(child)

        # Удаляем из родителя
        if bone.parent:
            bone.parent.remove_child(bone)

        # Удаляем из словарей
        del self.bones[name]

        # Обновляем иерархию
        for parent, children in self.bone_hierarchy.items():
            if name in children:
                children.remove(name)

        if name in self.bone_hierarchy:
            del self.bone_hierarchy[name]

        self._dirty = True
        logger.info(f"Кость '{name}' удалена")

    def update_from_landmarks(self, landmarks: np.ndarray, landmark_map: Dict[int, str]):
        """
        Обновление позиций костей на основе landmarks

        Args:
            landmarks: Массив landmarks из pose_estimator
            landmark_map: Словарь {landmark_id: bone_name}
        """
        for lm_id, bone_name in landmark_map.items():
            if bone_name in self.bones and lm_id < len(landmarks):
                bone = self.bones[bone_name]
                lm_data = landmarks[lm_id]

                if lm_data[3] > 0.3:  # confidence > 0.3
                    # Позиция в 3D (x, y, z)
                    position = lm_data[:3]

                    # Если это root bone, устанавливаем мировую позицию
                    if bone == self.root_bone:
                        bone.set_world_transform(position, bone.transform.rotation)
                    else:
                        # Для других костей ищем подходящий parent
                        pass

        self._dirty = True

    def calculate_bone_lengths(self):
        """Автоматический расчет длин костей на основе текущей позы"""
        for bone in self.bones.values():
            bone.calculate_length_from_children()

    def update_rest_pose(self):
        """Сохранение текущей позы как rest pose"""
        for bone in self.bones.values():
            bone.rest_pose.position = bone.transform.position.copy()
            bone.rest_pose.rotation = bone.transform.rotation.copy()
            bone.rest_pose.local_matrix = bone.transform.local_matrix.copy()
            bone.rest_pose.world_matrix = bone.transform.world_matrix.copy()

    def reset_to_rest_pose(self):
        """Сброс к rest pose"""
        for bone in self.bones.values():
            bone.transform.position = bone.rest_pose.position.copy()
            bone.transform.rotation = bone.rest_pose.rotation.copy()
            bone._update_matrices()

        self._dirty = True

    def _update_all_transforms(self):
        """Обновление всех трансформаций (вызывать при изменении иерархии)"""
        if self.root_bone is None:
            return

        # Начинаем с корневой кости
        self.root_bone._update_matrices()

        # Рекурсивно обновляем всех детей
        stack = self.root_bone.children[:]
        while stack:
            bone = stack.pop()
            bone._update_matrices()
            stack.extend(bone.children)

        self._dirty = False

    def get_bone(self, name: str) -> Optional[Bone]:
        """Получение кости по имени"""
        return self.bones.get(name)

    def get_bone_chain(self, start_bone: str, end_bone: str) -> List[Bone]:
        """Получение цепи костей между двумя костями"""
        chain = []

        start = self.get_bone(start_bone)
        end = self.get_bone(end_bone)

        if not start or not end:
            return chain

        # Находим путь от end к start через родителей
        current = end
        while current and current != start:
            chain.append(current)
            current = current.parent

        if current == start:
            chain.append(start)
            return list(reversed(chain))

        return []

    def apply_ik_to_chain(self, end_bone_name: str, target_position: np.ndarray):
        """Применение IK к цепи костей"""
        end_bone = self.get_bone(end_bone_name)
        if end_bone and end_bone.ik_enabled:
            end_bone.apply_ik(target_position)
            self._dirty = True

    def to_dict(self) -> Dict:
        """Сериализация скелета в словарь"""
        bone_dicts = {}
        for name, bone in self.bones.items():
            bone_dicts[name] = bone.to_dict()

        return {
            'name': self.name,
            'bones': bone_dicts,
            'bone_hierarchy': self.bone_hierarchy,
            'root_bone': self.root_bone.name if self.root_bone else None
        }

    def save_to_file(self, filepath: str):
        """Сохранение скелета в JSON файл"""
        data = self.to_dict()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Скелет сохранен в {filepath}")

    @classmethod
    def from_dict(cls, data: Dict) -> 'ProfessionalSkeleton':
        """Создание скелета из словаря"""
        skeleton = cls(name=data.get('name', 'ImportedSkeleton'))

        # Сначала создаем все кости без родителей
        parent_map = {}
        for name, bone_data in data['bones'].items():
            parent_name = bone_data.get('parent')
            parent_map[name] = parent_name

        # Создаем кости в правильном порядке (сначала родители)
        created_bones = {}

        def create_bone_recursive(bone_name):
            if bone_name in created_bones:
                return created_bones[bone_name]

            bone_data = data['bones'][bone_name]
            parent_name = bone_data.get('parent')

            # Сначала создаем родителя если нужно
            parent_bone = None
            if parent_name and parent_name in data['bones']:
                parent_bone = create_bone_recursive(parent_name)

            # Создаем кость
            bone = Bone.from_dict(bone_data, created_bones)
            created_bones[bone_name] = bone

            return bone

        # Создаем все кости
        for bone_name in data['bones']:
            create_bone_recursive(bone_name)

        # Обновляем скелет
        skeleton.bones = created_bones
        skeleton.bone_hierarchy = data.get('bone_hierarchy', {})

        root_name = data.get('root_bone')
        if root_name and root_name in created_bones:
            skeleton.root_bone = created_bones[root_name]

        # Обновляем трансформации
        skeleton._update_all_transforms()

        logger.info(f"Скелет загружен из словаря: {len(skeleton.bones)} костей")
        return skeleton

    @classmethod
    def load_from_file(cls, filepath: str) -> 'ProfessionalSkeleton':
        """Загрузка скелета из JSON файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def get_bvh_hierarchy(self) -> Dict:
        """Получение иерархии в формате BVH"""
        hierarchy = {}

        def add_bone_to_hierarchy(bone, parent_offset=None):
            offset = bone.local_offset.tolist() if parent_offset is None else parent_offset

            hierarchy[bone.name] = {
                'offset': offset,
                'channels': ['Xposition', 'Yposition', 'Zposition',
                             'Zrotation', 'Yrotation', 'Xrotation'],
                'children': [child.name for child in bone.children]
            }

            for child in bone.children:
                add_bone_to_hierarchy(child, [0, bone.length, 0])  # BVH offset

        if self.root_bone:
            add_bone_to_hierarchy(self.root_bone)

        return hierarchy

    def retarget_to(self, target_skeleton: 'ProfessionalSkeleton', mapping: Dict[str, str]):
        """
        Ретаргетинг анимации на другой скелет

        Args:
            target_skeleton: Целевой скелет
            mapping: Словарь {source_bone: target_bone}
        """
        for src_name, tgt_name in mapping.items():
            if src_name in self.bones and tgt_name in target_skeleton.bones:
                src_bone = self.bones[src_name]
                tgt_bone = target_skeleton.bones[tgt_name]

                # Устанавливаем ретаргет мап
                src_bone.retarget_map = tgt_bone

                # Вычисляем offset между костями
                if src_bone.parent and tgt_bone.parent:
                    # Разница в локальных пространствах
                    src_local = src_bone.transform.position
                    tgt_local = tgt_bone.transform.position
                    src_bone.retarget_offset = tgt_local - src_local

    def __repr__(self) -> str:
        return f"ProfessionalSkeleton(name='{self.name}', bones={len(self.bones)})"


# Утилитарные функции
def create_humanoid_mapping() -> Dict[int, str]:
    """Стандартное сопоставление MediaPipe landmarks с костями"""
    return {
        # Тело
        0: "Head",  # Нос (используем как Head)
        11: "LeftUpperArm",
        12: "RightUpperArm",
        13: "LeftLowerArm",
        14: "RightLowerArm",
        15: "LeftHand",
        16: "RightHand",
        23: "LeftUpperLeg",
        24: "RightUpperLeg",
        25: "LeftLowerLeg",
        26: "RightLowerLeg",
        27: "LeftFoot",
        28: "RightFoot",

        # Для более точного позиционирования
        7: "Head",  # Левое ухо
        8: "Head",  # Правое ухо
        29: "LeftFoot",  # Левая пятка
        30: "RightFoot",  # Правая пятка
    }


def blend_skeletons(skeleton1: ProfessionalSkeleton,
                    skeleton2: ProfessionalSkeleton,
                    weight: float) -> ProfessionalSkeleton:
    """Смешивание двух скелетов"""
    blended = ProfessionalSkeleton(name=f"Blended_{weight}")

    for bone_name in skeleton1.bones:
        if bone_name in skeleton2.bones:
            bone1 = skeleton1.bones[bone_name]
            bone2 = skeleton2.bones[bone_name]

            # Интерполяция позиции
            pos = bone1.transform.position * (1 - weight) + bone2.transform.position * weight

            # Сферическая интерполяция вращения (SLERP)
            rot = tf3d.quaternions.slerp(bone1.transform.rotation,
                                         bone2.transform.rotation,
                                         weight)

            blended.add_bone(bone_name, bone1.type, parent=bone1.parent.name if bone1.parent else None)
            blended_bone = blended.get_bone(bone_name)
            blended_bone.set_local_transform(position=pos, rotation=rot)

    return blended


if __name__ == "__main__":
    # Тестирование модуля
    logging.basicConfig(level=logging.INFO)

    print("Тестирование ProfessionalSkeleton...")

    # Создание скелета
    skeleton = ProfessionalSkeleton("TestHuman")

    print(f"Создан скелет: {skeleton}")
    print(f"Корневая кость: {skeleton.root_bone.name if skeleton.root_bone else 'None'}")
    print(f"Количество костей: {len(skeleton.bones)}")

    # Тест сериализации
    skeleton_dict = skeleton.to_dict()
    print(f"\nСериализовано {len(skeleton_dict['bones'])} костей")

    # Сохранение в файл
    skeleton.save_to_file("test_skeleton.json")
    print("Скелет сохранен в test_skeleton.json")

    # Загрузка из файла
    loaded_skeleton = ProfessionalSkeleton.load_from_file("test_skeleton.json")
    print(f"Загружен скелет: {loaded_skeleton}")

Skeleton = ProfessionalSkeleton

__all__ = [
    'ProfessionalSkeleton',
    'Skeleton',  # <-- ДОБАВЬТЕ ЭТУ СТРОКУ
    'Bone',
    'BoneType',
    'BoneTransform',
    'create_humanoid_mapping',
    'JointType',
    'blend_skeletons'
]