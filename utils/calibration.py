"""
Калибровка камеры
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CameraCalibration:
    """Класс для калибровки камеры с использованием шахматной доски"""

    def __init__(self, chessboard_size=(9, 6), square_size=0.025):
        """
        Args:
            chessboard_size: (cols, rows) количество внутренних углов
            square_size: размер клетки в метрах
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size

        # Подготовка точек шахматной доски в 3D
        self.objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
        self.objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)
        self.objp *= square_size

        self.objpoints = []  # 3D точки
        self.imgpoints = []  # 2D точки

        self.camera_matrix = None
        self.dist_coeffs = None

    def add_chessboard_frame(self, image):
        """Добавляет кадр шахматной доски для калибровки"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        if ret:
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)
            logger.info("Шахматная доска найдена и добавлена")
            return True
        return False

    def calibrate(self, image_shape):
        """Выполняет калибровку камеры"""
        if len(self.objpoints) < 1:
            logger.warning("Недостаточно кадров для калибровки")
            return None, None
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_shape[::-1], None, None)
        if ret:
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            logger.info("Калибровка завершена")
        else:
            logger.error("Ошибка калибровки")
        return self.camera_matrix, self.dist_coeffs

    def undistort(self, image):
        """Исправляет искажения кадра"""
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
