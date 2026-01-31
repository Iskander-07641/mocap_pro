"""
Ретаргетинг анимации между скелетами
"""

import numpy as np
from core.skeleton import Skeleton
import logging

logger = logging.getLogger(__name__)

class Retargeting:
    """Ретаргетинг анимации с одного скелета на другой"""

    def __init__(self, source_skeleton: Skeleton, target_skeleton: Skeleton):
        self.source = source_skeleton
        self.target = target_skeleton

    def retarget_frame(self, frame: dict) -> dict:
        """
        Переносит один кадр анимации

        Args:
            frame: {'bone_positions': {}, 'bone_rotations': {}}
        Returns:
            Новый кадр для target_skeleton
        """
        new_frame = {'bone_positions': {}, 'bone_rotations': {}}

        for bone_name in self.target.bones.keys():
            if bone_name in frame['bone_positions']:
                new_frame['bone_positions'][bone_name] = frame['bone_positions'][bone_name].copy()
            if bone_name in frame['bone_rotations']:
                new_frame['bone_rotations'][bone_name] = frame['bone_rotations'][bone_name].copy()

        return new_frame

    def retarget_sequence(self, frames: list) -> list:
        """Переносит последовательность кадров"""
        return [self.retarget_frame(f) for f in frames]
