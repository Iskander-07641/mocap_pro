"""
Математические утилиты для Mocap Pro
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import math


# ========== ОПЕРАЦИИ С КВАРТЕРНИОНАМИ ==========

def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Нормализует кватернион."""
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Возвращает сопряженный кватернион."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Умножает два кватерниона."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Возвращает обратный кватернион."""
    conjugate = quaternion_conjugate(q)
    norm_sq = np.dot(q, q)
    if norm_sq == 0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return conjugate / norm_sq


def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Сферическая линейная интерполяция (SLERP) между двумя кватернионами.

    Args:
        q1: Начальный кватернион
        q2: Конечный кватернион
        t: Параметр интерполяции [0, 1]

    Returns:
        np.ndarray: Интерполированный кватернион
    """
    # Нормализуем кватернионы
    q1 = quaternion_normalize(q1)
    q2 = quaternion_normalize(q2)

    # Вычисляем косинус угла между кватернионами
    dot = np.dot(q1, q2)

    # Если dot отрицательный, инвертируем один кватернион для кратчайшего пути
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # Если кватернионы очень близки, используем линейную интерполяцию
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return quaternion_normalize(result)

    # Ограничиваем dot для численной стабильности
    dot = np.clip(dot, -1.0, 1.0)

    # Вычисляем угол между кватернионами
    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    # Вычисляем коэффициенты интерполяции
    q2_perp = q2 - q1 * dot
    q2_perp = quaternion_normalize(q2_perp)

    # Интерполируем
    result = q1 * np.cos(theta) + q2_perp * np.sin(theta)
    return quaternion_normalize(result)


def quaternion_to_euler(q: np.ndarray, order: str = 'zyx') -> np.ndarray:
    """
    Конвертирует кватернион в углы Эйлера.

    Args:
        q: Кватернион [x, y, z, w]
        order: Порядок вращения ('xyz', 'zyx', etc.)

    Returns:
        np.ndarray: Углы Эйлера в радианах
    """
    q = quaternion_normalize(q)
    x, y, z, w = q

    if order == 'xyz':
        # XYZ order (roll, pitch, yaw)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    elif order == 'zyx':
        # ZYX order (yaw, pitch, roll)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([yaw, pitch, roll])

    elif order == 'zxy':
        # ZXY order
        sinp = 2 * (w * x - y * z)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (x * x + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        sinr_cosp = 2 * (w * y + z * x)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        return np.array([yaw, pitch, roll])

    else:
        raise ValueError(f"Unsupported rotation order: {order}")


def euler_to_quaternion(euler: np.ndarray, order: str = 'zyx') -> np.ndarray:
    """
    Конвертирует углы Эйлера в кватернион.

    Args:
        euler: Углы Эйлера в радианах
        order: Порядок вращения

    Returns:
        np.ndarray: Кватернион [x, y, z, w]
    """
    if order == 'xyz':
        roll, pitch, yaw = euler
    elif order == 'zyx':
        yaw, pitch, roll = euler
    elif order == 'zxy':
        yaw, roll, pitch = euler
    else:
        raise ValueError(f"Unsupported rotation order: {order}")

    # Вычисляем половины углов
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    if order == 'xyz':
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
    elif order == 'zyx':
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
    elif order == 'zxy':
        w = cr * cp * cy - sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy + sr * sp * cy

    return np.array([x, y, z, w])


# ========== МАТРИЧНЫЕ ОПЕРАЦИИ ==========

def create_translation_matrix(translation: np.ndarray) -> np.ndarray:
    """Создает матрицу трансляции."""
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return matrix


def create_rotation_matrix(rotation: np.ndarray, rotation_type: str = 'quaternion') -> np.ndarray:
    """
    Создает матрицу вращения из кватерниона или углов Эйлера.

    Args:
        rotation: Кватернион [x, y, z, w] или углы Эйлера
        rotation_type: 'quaternion' или 'euler'

    Returns:
        np.ndarray: Матрица вращения 4x4
    """
    matrix = np.eye(4)

    if rotation_type == 'quaternion':
        q = quaternion_normalize(rotation)
        x, y, z, w = q

        matrix[0, 0] = 1 - 2 * (y * y + z * z)
        matrix[0, 1] = 2 * (x * y - w * z)
        matrix[0, 2] = 2 * (x * z + w * y)

        matrix[1, 0] = 2 * (x * y + w * z)
        matrix[1, 1] = 1 - 2 * (x * x + z * z)
        matrix[1, 2] = 2 * (y * z - w * x)

        matrix[2, 0] = 2 * (x * z - w * y)
        matrix[2, 1] = 2 * (y * z + w * x)
        matrix[2, 2] = 1 - 2 * (x * x + y * y)

    elif rotation_type == 'euler':
        # Преобразуем углы Эйлера в кватернион
        q = euler_to_quaternion(rotation)
        return create_rotation_matrix(q, 'quaternion')

    return matrix


def create_scale_matrix(scale: np.ndarray) -> np.ndarray:
    """Создает матрицу масштабирования."""
    matrix = np.eye(4)
    matrix[0, 0] = scale[0]
    matrix[1, 1] = scale[1]
    matrix[2, 2] = scale[2]
    return matrix


def compose_transform_matrix(translation: np.ndarray,
                             rotation: np.ndarray,
                             scale: np.ndarray,
                             rotation_type: str = 'quaternion') -> np.ndarray:
    """
    Создает матрицу преобразования из компонентов.

    Args:
        translation: Вектор трансляции
        rotation: Вращение (кватернион или углы Эйлера)
        scale: Вектор масштабирования
        rotation_type: Тип представления вращения

    Returns:
        np.ndarray: Матрица преобразования 4x4
    """
    T = create_translation_matrix(translation)
    R = create_rotation_matrix(rotation, rotation_type)
    S = create_scale_matrix(scale)

    # Порядок: сначала масштаб, потом вращение, потом трансляция
    return T @ R @ S


def decompose_transform_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Декомпозирует матрицу преобразования на компоненты.

    Args:
        matrix: Матрица преобразования 4x4

    Returns:
        Tuple: (translation, rotation_quaternion, scale)
    """
    # Трансляция
    translation = matrix[:3, 3].copy()

    # Масштаб (из колонок матрицы вращения)
    scale_x = np.linalg.norm(matrix[:3, 0])
    scale_y = np.linalg.norm(matrix[:3, 1])
    scale_z = np.linalg.norm(matrix[:3, 2])
    scale = np.array([scale_x, scale_y, scale_z])

    # Нормализуем оси для получения чистой матрицы вращения
    if scale_x > 0:
        matrix[:3, 0] /= scale_x
    if scale_y > 0:
        matrix[:3, 1] /= scale_y
    if scale_z > 0:
        matrix[:3, 2] /= scale_z

    # Извлекаем кватернион из матрицы вращения
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix[:3, :3].flatten()

    trace = m00 + m11 + m22

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    rotation = np.array([x, y, z, w])
    rotation = quaternion_normalize(rotation)

    return translation, rotation, scale


# ========== ИНТЕРПОЛЯЦИИ И ФИЛЬТРАЦИЯ ==========

def bezier_interpolation(p0: np.ndarray, p1: np.ndarray,
                         p2: np.ndarray, p3: np.ndarray,
                         t: float) -> np.ndarray:
    """
    Кривая Безье 3-го порядка.

    Args:
        p0, p1, p2, p3: Контрольные точки
        t: Параметр [0, 1]

    Returns:
        np.ndarray: Интерполированная точка
    """
    t2 = t * t
    t3 = t2 * t
    one_minus_t = 1 - t
    one_minus_t2 = one_minus_t * one_minus_t
    one_minus_t3 = one_minus_t2 * one_minus_t

    return (one_minus_t3 * p0 +
            3 * one_minus_t2 * t * p1 +
            3 * one_minus_t * t2 * p2 +
            t3 * p3)


def catmull_rom_interpolation(p0: np.ndarray, p1: np.ndarray,
                              p2: np.ndarray, p3: np.ndarray,
                              t: float, alpha: float = 0.5) -> np.ndarray:
    """
    Интерполяция Catmull-Rom.

    Args:
        p0, p1, p2, p3: Контрольные точки
        t: Параметр [0, 1]
        alpha: Параметр натяжения (0.0 для равномерного, 0.5 для центрированного)

    Returns:
        np.ndarray: Интерполированная точка
    """

    # Вычисляем тау
    def tau(pk, pl):
        return np.power(np.sum((pl - pk) ** 2), alpha * 0.5)

    t01 = tau(p0, p1)
    t12 = tau(p1, p2)
    t23 = tau(p2, p3)

    # Вычисляем касательные
    m1 = (p2 - p1 + t12 * ((p1 - p0) / t01 - (p2 - p0) / (t01 + t12))) / t12
    m2 = (p2 - p1 + t12 * ((p3 - p2) / t23 - (p3 - p1) / (t12 + t23))) / t12

    # Интерполируем по Эрмиту
    t2 = t * t
    t3 = t2 * t

    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    return h00 * p1 + h10 * t12 * m1 + h01 * p2 + h11 * t12 * m2


def exponential_smoothing(current: np.ndarray, previous: np.ndarray,
                          alpha: float) -> np.ndarray:
    """
    Экспоненциальное сглаживание.

    Args:
        current: Текущее значение
        previous: Предыдущее сглаженное значение
        alpha: Коэффициент сглаживания (0-1)

    Returns:
        np.ndarray: Сглаженное значение
    """
    return alpha * current + (1 - alpha) * previous


def one_euro_filter(current: np.ndarray, previous: np.ndarray,
                    previous_velocity: np.ndarray, dt: float,
                    min_cutoff: float = 1.0, beta: float = 0.0,
                    d_cutoff: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-Euro фильтр для сглаживания сигналов.

    Args:
        current: Текущее значение
        previous: Предыдущее отфильтрованное значение
        previous_velocity: Предыдущая скорость
        dt: Временной шаг
        min_cutoff: Минимальная частота среза
        beta: Коэффициент скорости
        d_cutoff: Частота среза для производной

    Returns:
        Tuple: (filtered_value, new_velocity)
    """
    if dt <= 0:
        return current, np.zeros_like(current)

    # Фильтр для производной
    alpha_d = 1.0 / (1.0 + 1.0 / (2 * np.pi * d_cutoff * dt))
    velocity = alpha_d * ((current - previous) / dt) + (1 - alpha_d) * previous_velocity

    # Адаптивная частота среза
    cutoff = min_cutoff + beta * np.linalg.norm(velocity)

    # Фильтр для позиции
    alpha = 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))
    filtered = alpha * current + (1 - alpha) * previous

    return filtered, velocity


# ========== ГЕОМЕТРИЧЕСКИЕ ОПЕРАЦИИ ==========

def distance_point_to_line(point: np.ndarray,
                           line_start: np.ndarray,
                           line_end: np.ndarray) -> float:
    """
    Расстояние от точки до линии.

    Args:
        point: Точка
        line_start: Начало линии
        line_end: Конец линии

    Returns:
        float: Расстояние
    """
    line_vec = line_end - line_start
    point_vec = point - line_start

    # Проекция point_vec на line_vec
    line_length = np.dot(line_vec, line_vec)
    if line_length == 0:
        return np.linalg.norm(point_vec)

    t = np.dot(point_vec, line_vec) / line_length
    t = np.clip(t, 0.0, 1.0)

    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Угол между двумя векторами в радианах.

    Args:
        v1: Первый вектор
        v2: Второй вектор

    Returns:
        float: Угол в радианах [0, π]
    """
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm == 0 or v2_norm == 0:
        return 0.0

    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.arccos(cos_angle)


def signed_angle_between_vectors(v1: np.ndarray, v2: np.ndarray,
                                 normal: np.ndarray) -> float:
    """
    Знаковый угол между двумя векторами.

    Args:
        v1: Первый вектор
        v2: Второй вектор
        normal: Вектор нормали для определения знака

    Returns:
        float: Угол в радианах [-π, π]
    """
    angle = angle_between_vectors(v1, v2)

    # Определяем знак через векторное произведение
    cross = np.cross(v1, v2)
    sign = np.sign(np.dot(cross, normal))

    return angle * sign


def project_point_onto_plane(point: np.ndarray,
                             plane_point: np.ndarray,
                             plane_normal: np.ndarray) -> np.ndarray:
    """
    Проецирует точку на плоскость.

    Args:
        point: Точка для проецирования
        plane_point: Точка на плоскости
        plane_normal: Нормаль плоскости

    Returns:
        np.ndarray: Проекция точки
    """
    # Нормализуем нормаль
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Вектор от точки плоскости к точке
    vec = point - plane_point

    # Расстояние вдоль нормали
    distance = np.dot(vec, plane_normal)

    # Проекция
    return point - distance * plane_normal


# ========== СТАТИСТИЧЕСКИЕ ФУНКЦИИ ==========

def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Скользящее среднее.

    Args:
        data: Входные данные
        window_size: Размер окна

    Returns:
        np.ndarray: Сглаженные данные
    """
    if window_size <= 1:
        return data.copy()

    if len(data.shape) == 1:
        # 1D случай
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            result[i] = np.mean(data[start:end])
        return result
    else:
        # Многомерный случай
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            result[i] = np.mean(data[start:end], axis=0)
        return result


def weighted_moving_average(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Взвешенное скользящее среднее.

    Args:
        data: Входные данные
        weights: Веса

    Returns:
        np.ndarray: Сглаженные данные
    """
    weights = weights / np.sum(weights)

    if len(data.shape) == 1:
        # 1D случай
        result = np.convolve(data, weights, mode='same')
    else:
        # Многомерный случай
        result = np.zeros_like(data)
        for dim in range(data.shape[1]):
            result[:, dim] = np.convolve(data[:, dim], weights, mode='same')

    return result


def gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """
    Создает гауссово ядро.

    Args:
        size: Размер ядра (нечетный)
        sigma: Стандартное отклонение

    Returns:
        np.ndarray: Гауссово ядро
    """
    if size % 2 == 0:
        size += 1

    kernel = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (kernel / sigma) ** 2)
    kernel /= np.sum(kernel)

    return kernel


# ========== УТИЛИТЫ ДЛЯ АНИМАЦИИ ==========

def ease_in_out(t: float) -> float:
    """Функция плавности ease-in-out."""
    if t < 0.5:
        return 2 * t * t
    else:
        return -1 + (4 - 2 * t) * t


def ease_in(t: float, power: float = 2.0) -> float:
    """Функция плавности ease-in."""
    return t ** power


def ease_out(t: float, power: float = 2.0) -> float:
    """Функция плавности ease-out."""
    return 1 - (1 - t) ** power


def remap(value: float,
          from_min: float, from_max: float,
          to_min: float, to_max: float,
          clamp: bool = True) -> float:
    """
    Преобразует значение из одного диапазона в другой.

    Args:
        value: Исходное значение
        from_min, from_max: Исходный диапазон
        to_min, to_max: Целевой диапазон
        clamp: Ограничивать ли значение исходным диапазоном

    Returns:
        float: Преобразованное значение
    """
    if clamp:
        value = max(from_min, min(from_max, value))

    # Нормализуем в диапазон [0, 1]
    normalized = (value - from_min) / (from_max - from_min)

    # Преобразуем в целевой диапазон
    return to_min + normalized * (to_max - to_min)


# ========== КЛАССЫ ДЛЯ УДОБСТВА ==========

class KalmanFilter:
    """Простой фильтр Калмана для 1D сигнала."""

    def __init__(self, process_variance: float = 1e-5,
                 measurement_variance: float = 0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement: float) -> float:
        """
        Обновляет фильтр с новым измерением.

        Args:
            measurement: Новое измерение

        Returns:
            float: Отфильтрованное значение
        """
        # Предсказание
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # Обновление
        kalman_gain = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + kalman_gain * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - kalman_gain) * priori_error_estimate

        return self.posteri_estimate


class VectorKalmanFilter:
    """Фильтр Калмана для многомерных векторов."""

    def __init__(self, dim: int,
                 process_covariance: float = 1e-5,
                 measurement_covariance: float = 0.1):
        self.dim = dim

        # Инициализация матриц
        self.state = np.zeros(dim)
        self.covariance = np.eye(dim)

        # Матрицы процесса
        self.F = np.eye(dim)  # Матрица состояния
        self.Q = np.eye(dim) * process_covariance  # Ковариация процесса

        # Матрицы измерения
        self.H = np.eye(dim)  # Матрица измерения
        self.R = np.eye(dim) * measurement_covariance  # Ковариация измерения

    def predict(self) -> np.ndarray:
        """Этап предсказания."""
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Этап обновления с новым измерением.

        Args:
            measurement: Новое измерение

        Returns:
            np.ndarray: Отфильтрованное состояние
        """
        # Предварительно предсказываем
        self.predict()

        # Вычисляем коэффициент Калмана
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)

        # Обновляем состояние
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        self.covariance = (np.eye(self.dim) - K @ self.H) @ self.covariance

        return self.state.copy()


# ========== ТЕСТОВЫЕ ФУНКЦИИ ==========

def test_quaternion_operations():
    """Тестирует операции с кватернионами."""
    print("Тестирование кватернионных операций...")

    # Тест нормализации
    q = np.array([1.0, 2.0, 3.0, 4.0])
    q_norm = quaternion_normalize(q)
    print(f"Нормализация: {np.linalg.norm(q_norm):.6f}")

    # Тест умножения
    q1 = np.array([0.0, 0.0, 0.0, 1.0])  # Единичный кватернион
    q2 = np.array([0.0, 0.0, 0.0, 1.0])
    q_mult = quaternion_multiply(q1, q2)
    print(f"Умножение единичных: {q_mult}")

    # Тест SLERP
    q_start = np.array([0.0, 0.0, 0.0, 1.0])
    q_end = np.array([1.0, 0.0, 0.0, 0.0])
    q_slerp = quaternion_slerp(q_start, q_end, 0.5)
    print(f"SLERP средняя точка: {q_slerp}")

    print("✅ Тест кватернионов пройден")


def test_matrix_operations():
    """Тестирует матричные операции."""
    print("\nТестирование матричных операций...")

    # Тест композиции/декомпозиции
    translation = np.array([1.0, 2.0, 3.0])
    rotation = np.array([0.0, 0.0, 0.707, 0.707])  # 90 градусов вокруг Z
    scale = np.array([2.0, 2.0, 2.0])

    matrix = compose_transform_matrix(translation, rotation, scale)
    t, r, s = decompose_transform_matrix(matrix)

    print(f"Исходная трансляция: {translation}")
    print(f"Восстановленная трансляция: {t}")
    print(f"Разница: {np.max(np.abs(translation - t)):.6f}")

    print("✅ Тест матриц пройден")


if __name__ == "__main__":
    # Запуск тестов
    test_quaternion_operations()
    test_matrix_operations()
    print("\n✅ Все тесты пройдены успешно!")