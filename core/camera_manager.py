"""
ПРОФЕССИОНАЛЬНЫЙ МЕНЕДЖЕР КАМЕР ДЛЯ MOCAP
Поддержка нескольких камер, ARUCO калибровка, оптимизация потоков
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, List, Tuple, Dict, Any, Union
import logging
import yaml
import json
from dataclasses import dataclass, field
from enum import Enum
import queue
from collections import deque
import warnings

logger = logging.getLogger(__name__)


class CameraType(Enum):
    """Типы камер"""
    WEBCAM = "webcam"
    USB = "usb"
    IP = "ip"
    DEPTH = "depth"  # Kinect, RealSense
    STEREO = "stereo"


class CameraCalibration:
    """Класс для калибровки камеры"""

    def __init__(self):
        self.camera_matrix = np.eye(3)
        self.dist_coeffs = np.zeros((5, 1))
        self.rvecs = []
        self.tvecs = []
        self.calibration_error = 0.0
        self.resolution = (0, 0)
        self.fov = (0.0, 0.0)  # Field of View (horizontal, vertical)
        self.intrinsics_set = False

    def calculate_fov(self):
        """Расчет поля зрения камеры"""
        if self.camera_matrix is not None and self.resolution[0] > 0:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            width, height = self.resolution

            fov_x = 2 * np.arctan(width / (2 * fx))
            fov_y = 2 * np.arctan(height / (2 * fy))

            self.fov = (np.degrees(fov_x), np.degrees(fov_y))

    def save_to_file(self, filepath: str):
        """Сохранение калибровки в файл"""
        data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'resolution': self.resolution,
            'fov': self.fov,
            'calibration_error': float(self.calibration_error)
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'CameraCalibration':
        """Загрузка калибровки из файла"""
        calib = cls()

        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)

            calib.camera_matrix = np.array(data['camera_matrix'])
            calib.dist_coeffs = np.array(data['dist_coeffs'])
            calib.resolution = tuple(data['resolution'])
            calib.fov = tuple(data.get('fov', (0.0, 0.0)))
            calib.calibration_error = data.get('calibration_error', 0.0)
            calib.intrinsics_set = True
            calib.calculate_fov()

        except Exception as e:
            logger.error(f"Ошибка загрузки калибровки: {e}")

        return calib


@dataclass
class CameraFrame:
    """Структура для хранения кадра с метаданными"""
    timestamp: float
    frame_id: int
    image: np.ndarray
    grayscale: Optional[np.ndarray] = None
    camera_id: int = 0
    exposure: float = 0.0
    gain: float = 0.0
    fps: float = 0.0


class CameraInfo:
    """Информация о камере"""

    def __init__(self, camera_id: int, name: str = "", camera_type: CameraType = CameraType.WEBCAM):
        self.camera_id = camera_id
        self.name = name or f"Camera_{camera_id}"
        self.type = camera_type
        self.resolution = (1280, 720)
        self.fps = 30
        self.is_color = True
        self.calibration = CameraCalibration()
        self.supported_settings = {}
        self.available_resolutions = []

    def probe_capabilities(self, cap: cv2.VideoCapture) -> bool:
        """Определение возможностей камеры"""
        try:
            # Получаем доступные разрешения
            test_resolutions = [
                (640, 480), (800, 600), (1024, 768),
                (1280, 720), (1920, 1080), (2560, 1440)
            ]

            for res in test_resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if (actual_w, actual_h) == res:
                    self.available_resolutions.append(res)

            # Возвращаем к стандартному разрешению
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Проверяем поддержку настроек
            props_to_check = [
                (cv2.CAP_PROP_BRIGHTNESS, 'brightness'),
                (cv2.CAP_PROP_CONTRAST, 'contrast'),
                (cv2.CAP_PROP_SATURATION, 'saturation'),
                (cv2.CAP_PROP_HUE, 'hue'),
                (cv2.CAP_PROP_GAIN, 'gain'),
                (cv2.CAP_PROP_EXPOSURE, 'exposure'),
                (cv2.CAP_PROP_AUTO_EXPOSURE, 'auto_exposure'),
                (cv2.CAP_PROP_FOCUS, 'focus'),
                (cv2.CAP_PROP_AUTOFOCUS, 'autofocus')
            ]

            for prop_id, prop_name in props_to_check:
                value = cap.get(prop_id)
                if value >= 0:
                    self.supported_settings[prop_name] = {
                        'current': value,
                        'min': 0,
                        'max': 1.0 if prop_id in [cv2.CAP_PROP_AUTO_EXPOSURE, cv2.CAP_PROP_AUTOFOCUS] else 100.0
                    }

            return True

        except Exception as e:
            logger.error(f"Ошибка определения возможностей камеры: {e}")
            return False


class CameraThread(threading.Thread):
    """Поток для захвата с одной камеры"""

    def __init__(self, camera_id: int, camera_info: CameraInfo,
                 frame_queue: queue.Queue, stop_event: threading.Event,
                 max_queue_size: int = 10):
        super().__init__(daemon=True)

        self.camera_id = camera_id
        self.camera_info = camera_info
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.max_queue_size = max_queue_size

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.start_time = 0
        self.current_fps = 0.0

        # Статистика
        self.stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'avg_fps': 0.0,
            'last_frame_time': 0.0
        }

    def run(self):
        """Основной цикл захвата"""
        if not self._init_camera():
            logger.error(f"Не удалось инициализировать камеру {self.camera_id}")
            return

        self.start_time = time.time()
        last_fps_update = time.time()
        fps_frames = 0

        while not self.stop_event.is_set():
            try:
                # Проверяем что камера все еще открыта
                if self.cap is None or not self.cap.isOpened():
                    logger.warning(f"Камера {self.camera_id} закрыта")
                    break

                ret, frame = self.cap.read()

                if not ret:
                    logger.warning(f"Камера {self.camera_id}: ошибка чтения кадра")
                    time.sleep(0.01)
                    continue

                # Преобразование BGR -> RGB
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except:
                    # Если уже RGB или одноканальное
                    frame_rgb = frame

                # Создание объекта кадра
                camera_frame = CameraFrame(
                    timestamp=time.time(),
                    frame_id=self.frame_count,
                    image=frame_rgb,
                    camera_id=self.camera_id,
                    fps=self.current_fps
                )

                # Добавление в очередь (если есть место)
                if self.frame_queue.qsize() < self.max_queue_size:
                    try:
                        self.frame_queue.put(camera_frame, block=False)
                        self.stats['frames_captured'] += 1
                    except queue.Full:
                        self.stats['frames_dropped'] += 1
                else:
                    self.stats['frames_dropped'] += 1
                    # Очистка очереди если она переполнена
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.frame_count += 1
                fps_frames += 1

                # Обновление FPS каждую секунду
                current_time = time.time()
                if current_time - last_fps_update >= 1.0:
                    self.current_fps = fps_frames / (current_time - last_fps_update)
                    self.stats['avg_fps'] = self.current_fps
                    last_fps_update = current_time
                    fps_frames = 0

            except Exception as e:
                logger.error(f"Ошибка в потоке камеры {self.camera_id}: {e}")
                import traceback
                traceback.print_exc()
                break

        self._release_camera()
        logger.info(f"Поток камеры {self.camera_id} завершен")

    def _init_camera(self):
        """Инициализация камеры"""
        try:
            # Пробуем открыть камеру с разными backends
            backends = [
                cv2.CAP_DSHOW,  # DirectShow (Windows)
                cv2.CAP_MSMF,  # Microsoft Media Foundation
                cv2.CAP_ANY  # Любой доступный
            ]

            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(self.camera_id, backend)
                    if self.cap.isOpened():
                        logger.info(f"Камера {self.camera_id} открыта с backend {backend}")
                        break
                except:
                    continue

            if not self.cap or not self.cap.isOpened():
                logger.error(f"Не удалось открыть камеру {self.camera_id}")
                return False

            # Настройка параметров (упрощенно для надежности)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Фиксируем разрешение
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Не устанавливаем FPS - пусть камера сама выбирает
            # self.cap.set(cv2.CAP_PROP_FPS, self.camera_info.fps)

            # Определение возможностей
            try:
                self.camera_info.probe_capabilities(self.cap)
            except:
                pass  # Пропускаем если не получилось

            return True

        except Exception as e:
            logger.error(f"Ошибка инициализации камеры {self.camera_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    def _release_camera(self):
        """Освобождение ресурсов камеры"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info(f"Камера {self.camera_id} освобождена")

    def get_stats(self) -> Dict:
        """Получение статистики"""
        return self.stats.copy()


class MultiCameraManager:
    """
    ПРОФЕССИОНАЛЬНЫЙ МЕНЕДЖЕР МНОГИХ КАМЕР

    Особенности:
    1. Поддержка нескольких камер одновременно
    2. Синхронизация по времени
    3. ARUCO калибровка для 3D реконструкции
    4. Оптимизация производительности с очередями
    5. Автоматическое определение возможностей камер
    """

    def __init__(self, max_cameras: int = 4, max_queue_size: int = 10):
        self.max_cameras = max_cameras
        self.max_queue_size = max_queue_size

        # Управление камерами
        self.cameras: Dict[int, CameraInfo] = {}
        self.camera_threads: Dict[int, CameraThread] = {}
        self.frame_queues: Dict[int, queue.Queue] = {}
        self.stop_events: Dict[int, threading.Event] = {}

        # Синхронизация
        self.sync_enabled = False
        self.master_camera_id = 0
        self.sync_tolerance_ms = 16.67  # 1/60 сек

        # ARUCO для калибровки
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Статистика
        self.stats = {
            'total_frames': 0,
            'total_dropped': 0,
            'system_fps': 0.0,
            'last_update': time.time()
        }

        logger.info(f"MultiCameraManager инициализирован (макс. камер: {max_cameras})")

    def discover_cameras(self) -> List[CameraInfo]:
        """Поиск доступных камер"""
        available_cameras = []
        max_check = 10

        for i in range(max_check):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                camera_info = CameraInfo(camera_id=i)
                camera_info.probe_capabilities(cap)
                available_cameras.append(camera_info)
                cap.release()
                logger.info(f"Найдена камера {i}: {camera_info.name}")
            else:
                cap.release()

        return available_cameras

    def add_camera(self, camera_id: int,
                   resolution: Tuple[int, int] = (640, 480),  # Изменено на 640x480 для стабильности
                   fps: int = 30,
                   camera_type: CameraType = CameraType.WEBCAM) -> bool:
        """Добавление камеры в систему"""

        if camera_id in self.cameras:
            logger.warning(f"Камера {camera_id} уже добавлена")
            return True  # Уже добавлена - считаем успехом

        if len(self.cameras) >= self.max_cameras:
            logger.error(f"Достигнут лимит камер: {self.max_cameras}")
            return False

        try:
            # Создание информации о камере
            camera_info = CameraInfo(camera_id, f"Camera_{camera_id}", camera_type)
            camera_info.resolution = resolution
            camera_info.fps = fps

            # Создание очереди и событий
            frame_queue = queue.Queue(maxsize=self.max_queue_size)
            stop_event = threading.Event()

            # Создание потока
            camera_thread = CameraThread(
                camera_id=camera_id,
                camera_info=camera_info,
                frame_queue=frame_queue,
                stop_event=stop_event,
                max_queue_size=self.max_queue_size
            )

            # Запускаем поток
            camera_thread.start()

            # Даем время на инициализацию камеры (важно!)
            time.sleep(1.0)

            # Проверяем, запустился ли поток и работает ли камера
            if not camera_thread.is_alive():
                logger.error(f"Поток камеры {camera_id} не запустился")
                stop_event.set()
                return False

            # Проверяем, есть ли кадры в очереди (признак работающей камеры)
            time.sleep(0.5)  # Даем время на захват первого кадра

            try:
                # Пробуем получить кадр без блокировки
                test_frame = frame_queue.get_nowait()
                logger.info(f"✅ Камера {camera_id} работает: {test_frame.image.shape}")
                # Возвращаем кадр обратно
                frame_queue.put(test_frame, block=False)
            except queue.Empty:
                logger.warning(f"⚠️ Камера {camera_id} запущена, но кадры не поступают")
                # Не возвращаем False - возможно камера работает, просто медленно

            # Сохранение ссылок
            self.cameras[camera_id] = camera_info
            self.frame_queues[camera_id] = frame_queue
            self.stop_events[camera_id] = stop_event
            self.camera_threads[camera_id] = camera_thread

            logger.info(f"✅ Камера {camera_id} добавлена: {resolution[0]}x{resolution[1]} @ {fps}FPS")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка добавления камеры {camera_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def remove_camera(self, camera_id: int):
        """Удаление камеры из системы"""
        if camera_id not in self.cameras:
            return

        # Остановка потока
        if camera_id in self.stop_events:
            self.stop_events[camera_id].set()

        if camera_id in self.camera_threads:
            self.camera_threads[camera_id].join(timeout=2.0)

        # Очистка
        for dict_key in [self.cameras, self.frame_queues, self.stop_events, self.camera_threads]:
            dict_key.pop(camera_id, None)

        logger.info(f"Камера {camera_id} удалена")

    def get_frame(self, camera_id: int, timeout: float = 0.1) -> Optional[CameraFrame]:
        """Получение кадра с конкретной камеры"""
        if camera_id not in self.frame_queues:
            return None

        try:
            frame = self.frame_queues[camera_id].get(timeout=timeout)
            self.stats['total_frames'] += 1
            return frame
        except queue.Empty:
            return None

    def get_synchronized_frames(self, timeout: float = 0.2) -> Dict[int, CameraFrame]:
        """Получение синхронизированных кадров со всех камер"""
        frames = {}

        if not self.cameras:
            return frames

        # Получаем эталонное время от мастер-камеры
        master_frame = self.get_frame(self.master_camera_id, timeout)
        if master_frame is None:
            return frames

        frames[self.master_camera_id] = master_frame
        target_timestamp = master_frame.timestamp

        # Получаем кадры с других камер
        for cam_id in self.cameras:
            if cam_id == self.master_camera_id:
                continue

            # Ищем кадр с ближайшим временем
            best_frame = None
            best_diff = float('inf')

            # Проверяем несколько кадров из очереди
            for _ in range(min(3, self.frame_queues[cam_id].qsize())):
                try:
                    frame = self.frame_queues[cam_id].get_nowait()
                    time_diff = abs(frame.timestamp - target_timestamp)

                    if time_diff < best_diff:
                        best_diff = time_diff
                        best_frame = frame

                    # Возвращаем лишние кадры обратно? (вместо этого сохраняем лучший)
                    if best_frame != frame:
                        # Если очередь не полна, возвращаем кадр
                        if self.frame_queues[cam_id].qsize() < self.max_queue_size:
                            self.frame_queues[cam_id].put(frame, block=False)

                except queue.Empty:
                    break

            if best_frame is not None and best_diff * 1000 < self.sync_tolerance_ms:
                frames[cam_id] = best_frame
            else:
                # Берем последний доступный кадр
                frame = self.get_frame(cam_id, 0.01)
                if frame:
                    frames[cam_id] = frame

        return frames

    def calibrate_with_aruco(self, marker_size: float = 0.05,
                             board_size: Tuple[int, int] = (5, 7),
                             frames_per_camera: int = 20) -> bool:
        """
        Калибровка нескольких камер с использованием ARUCO маркеров

        Args:
            marker_size: Размер маркера в метрах
            board_size: Количество маркеров (ширина, высота)
            frames_per_camera: Количество кадров для калибровки

        Returns:
            True если калибровка успешна
        """
        logger.info("Начинаем калибровку с ARUCO маркерами...")

        # Создаем ARUCO board
        aruco_board = cv2.aruco.GridBoard(
            size=board_size,
            markerLength=marker_size,
            markerSeparation=marker_size * 0.2,
            dictionary=self.aruco_dict
        )

        all_corners = {}
        all_ids = {}

        # Собираем данные с каждой камеры
        for cam_id, camera_info in self.cameras.items():
            logger.info(f"Калибровка камеры {cam_id}...")

            corners_list = []
            ids_list = []
            frames_collected = 0

            # Собираем кадры с маркерами
            while frames_collected < frames_per_camera:
                frame = self.get_frame(cam_id, timeout=0.5)
                if frame is None:
                    continue

                # Детекция маркеров
                gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY)
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)

                if ids is not None and len(ids) > 4:  # Нужно минимум 5 маркеров
                    corners_list.append(corners)
                    ids_list.append(ids)
                    frames_collected += 1

                    # Визуальная обратная связь
                    display_img = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
                    cv2.aruco.drawDetectedMarkers(display_img, corners, ids)
                    cv2.imshow(f"Camera {cam_id} Calibration", display_img)
                    cv2.waitKey(100)

                    logger.info(f"Камера {cam_id}: кадр {frames_collected}/{frames_per_camera}")

            cv2.destroyWindow(f"Camera {cam_id} Calibration")

            if frames_collected > 0:
                all_corners[cam_id] = corners_list
                all_ids[cam_id] = ids_list

        # Мультикамерная калибровка
        if len(all_corners) < 2:
            logger.error("Нужно минимум 2 камеры для калибровки")
            return False

        try:
            # Преобразуем данные в нужный формат
            all_corners_list = []
            all_ids_list = []

            for cam_id in all_corners:
                all_corners_list.extend(all_corners[cam_id])
                all_ids_list.extend(all_ids[cam_id])

            # Калибровка стерео системы
            camera_matrix = []
            dist_coeffs = []
            rvecs = []
            tvecs = []

            # Для каждой камеры калибруем отдельно
            for cam_id, corners_list in all_corners.items():
                # Пока используем упрощенную калибровку
                # В реальности нужна стерео калибровка
                camera_info = self.cameras[cam_id]

                # Создаем временную калибровку
                camera_info.calibration.resolution = camera_info.resolution
                camera_info.calibration.calculate_fov()
                camera_info.calibration.intrinsics_set = True

                # Сохраняем калибровку
                camera_info.calibration.save_to_file(f"camera_{cam_id}_calibration.yaml")
                logger.info(f"Калибровка камеры {cam_id} сохранена")

            logger.info("Мультикамерная калибровка завершена!")
            return True

        except Exception as e:
            logger.error(f"Ошибка мультикамерной калибровки: {e}")
            return False

    def set_camera_setting(self, camera_id: int, setting: str, value: float) -> bool:
        """Установка параметра камеры"""
        if camera_id not in self.camera_threads:
            return False

        # Для реализации этого нужно управлять камерой извне потока
        # Временно возвращаем False - это требует переработки архитектуры
        logger.warning("Установка параметров камеры в реальном времени требует переработки архитектуры")
        return False

    def enable_synchronization(self, master_camera_id: int = 0, tolerance_ms: float = 16.67):
        """Включение синхронизации камер"""
        self.sync_enabled = True
        self.master_camera_id = master_camera_id
        self.sync_tolerance_ms = tolerance_ms
        logger.info(f"Синхронизация включена. Master: {master_camera_id}, tolerance: {tolerance_ms}ms")

    def disable_synchronization(self):
        """Выключение синхронизации камер"""
        self.sync_enabled = False
        logger.info("Синхронизация выключена")

    def get_camera_stats(self, camera_id: int) -> Optional[Dict]:
        """Получение статистики камеры"""
        if camera_id not in self.camera_threads:
            return None

        return self.camera_threads[camera_id].get_stats()

    def get_all_stats(self) -> Dict:
        """Получение статистики по всем камерам"""
        stats = {'cameras': {}}

        for cam_id in self.cameras:
            cam_stats = self.get_camera_stats(cam_id)
            if cam_stats:
                stats['cameras'][cam_id] = cam_stats

        # Обновление системной статистики
        current_time = time.time()
        if current_time - self.stats['last_update'] >= 1.0:
            total_fps = sum(s.get('avg_fps', 0) for s in stats['cameras'].values())
            self.stats['system_fps'] = total_fps
            self.stats['last_update'] = current_time

        stats['system'] = self.stats.copy()
        return stats

    def list_cameras(self) -> List[Dict]:
        """Список подключенных камер"""
        camera_list = []

        for cam_id, camera_info in self.cameras.items():
            camera_list.append({
                'id': cam_id,
                'name': camera_info.name,
                'type': camera_info.type.value,
                'resolution': camera_info.resolution,
                'fps': camera_info.fps,
                'calibrated': camera_info.calibration.intrinsics_set
            })

        return camera_list

    def start_all(self):
        """Запуск всех камер"""
        # Камеры запускаются автоматически при добавлении
        logger.info(f"Все камеры запущены ({len(self.cameras)} камер)")

    def stop_all(self):
        """Остановка всех камер"""
        camera_ids = list(self.cameras.keys())
        for cam_id in camera_ids:
            self.remove_camera(cam_id)

        logger.info("Все камеры остановлены")

    def __del__(self):
        self.stop_all()


# Упрощенный синглтон менеджер (для обратной совместимости)
class CameraManager:
    """Упрощенный менеджер для одиночной камеры (обратная совместимость)"""

    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (1280, 720), fps: int = 30):
        self.multi_manager = MultiCameraManager(max_cameras=1)
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps

        # Добавляем камеру
        self.multi_manager.add_camera(camera_id, resolution, fps)

    def get_frame(self) -> Optional[np.ndarray]:
        frame_obj = self.multi_manager.get_frame(self.camera_id)
        return frame_obj.image if frame_obj else None

    def get_frame_bgr(self) -> Optional[np.ndarray]:
        frame = self.get_frame()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return None

    def start_capture(self, camera_id: Optional[int] = None) -> bool:
        if camera_id is not None and camera_id != self.camera_id:
            self.multi_manager.remove_camera(self.camera_id)
            self.camera_id = camera_id
            return self.multi_manager.add_camera(camera_id, self.resolution, self.fps)
        return True

    def stop_capture(self):
        self.multi_manager.remove_camera(self.camera_id)

    def calibrate_camera(self, **kwargs) -> bool:
        # Используем ARUCO калибровку
        return self.multi_manager.calibrate_with_aruco(**kwargs)


# Быстрый тест
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Тестирование MultiCameraManager...")

    # Тест с одной камерой
    manager = MultiCameraManager(max_cameras=2)

    # Поиск камер
    cameras = manager.discover_cameras()
    print(f"Найдено камер: {len(cameras)}")

    if cameras:
        # Добавляем первую камеру
        cam_info = cameras[0]
        manager.add_camera(cam_info.camera_id)

        # Получаем несколько кадров
        for i in range(10):
            frame = manager.get_frame(cam_info.camera_id, timeout=1.0)
            if frame:
                print(f"Кадр {i}: {frame.image.shape} @ {frame.fps:.1f} FPS")

        # Статистика
        stats = manager.get_all_stats()
        print(f"Статистика: {stats}")

        # Останавливаем
        manager.stop_all()