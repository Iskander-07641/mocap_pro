"""
Функции сглаживания для 3D точек и кадров
"""

import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

def moving_average(points: List[np.ndarray], window_size=5) -> List[np.ndarray]:
    """Скользящее среднее для сглаживания траекторий"""
    if len(points) < window_size:
        return points
    smoothed = []
    half = window_size // 2
    for i in range(len(points)):
        start = max(0, i - half)
        end = min(len(points), i + half + 1)
        smoothed.append(np.mean(points[start:end], axis=0))
    return smoothed

def smooth_quaternions(quaternions: List[np.ndarray], window_size=5) -> List[np.ndarray]:
    """Сглаживание кватернионов через SLERP и скользящее среднее"""
    from utils.math_utils import slerp, normalize
    if len(quaternions) < window_size:
        return [normalize(q) for q in quaternions]

    smoothed = []
    half = window_size // 2
    for i in range(len(quaternions)):
        start = max(0, i - half)
        end = min(len(quaternions), i + half + 1)
        q_avg = quaternions[start]
        for q in quaternions[start+1:end]:
            q_avg = slerp(q_avg, q, 1.0/(end-start))
        smoothed.append(normalize(q_avg))
    return smoothed
