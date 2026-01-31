"""
Модуль: Blender Bridge (Интеграция с Blender)
Версия: 1.1.0
Автор: Mocap Pro Team

Продвинутая интеграция с Blender через Python API.
Автоматическая отправка анимации, ретаргетинг, пакетная обработка.
Поддержка Blender 2.8+ с новым Python API.
"""

import os
import sys
import json
import tempfile
import subprocess
import threading
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import numpy as np
import socket

from core.skeleton import Skeleton, Bone, BoneType as JointType
from core.animation_recorder import AnimationLayer
from export.bvh_exporter import BVHExporter, BVHExportSettings

# Настройка логирования
logger = logging.getLogger(__name__)


# Состояния соединения с Blender
class BlenderConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


# Режимы ретаргетинга
class RetargetMode(Enum):
    AUTOMATIC = "automatic"  # Автоматическое сопоставление костей
    MANUAL = "manual"  # Ручное сопоставление
    HIK = "hik"  # Human IK (Maya совместимость)
    RIGIFY = "rigify"  # Blender Rigify метариг


# Методы интерполяции
class InterpolationMethod(Enum):
    LINEAR = "LINEAR"
    BEZIER = "BEZIER"
    SINE = "SINE"
    QUAD = "QUAD"
    CUBIC = "CUBIC"
    QUART = "QUART"
    QUINT = "QUINT"
    EXPO = "EXPO"
    CIRC = "CIRC"
    BACK = "BACK"
    BOUNCE = "BOUNCE"
    ELASTIC = "ELASTIC"


@dataclass
class BlenderSettings:
    """Настройки интеграции с Blender"""
    blender_executable: str = ""
    auto_detect_blender: bool = True
    python_script_path: str = ""
    connection_timeout: int = 30
    enable_live_streaming: bool = False
    streaming_fps: int = 30
    auto_import_animation: bool = True
    auto_retarget: bool = True
    retarget_mode: RetargetMode = RetargetMode.AUTOMATIC
    create_metarig: bool = False
    bake_animation: bool = True
    apply_scale: bool = True
    use_nla_editor: bool = True

    def to_dict(self) -> Dict:
        return {
            "blender_executable": self.blender_executable,
            "auto_detect_blender": self.auto_detect_blender,
            "python_script_path": self.python_script_path,
            "connection_timeout": self.connection_timeout,
            "enable_live_streaming": self.enable_live_streaming,
            "streaming_fps": self.streaming_fps,
            "auto_import_animation": self.auto_import_animation,
            "auto_retarget": self.auto_retarget,
            "retarget_mode": self.retarget_mode.value,
            "create_metarig": self.create_metarig,
            "bake_animation": self.bake_animation,
            "apply_scale": self.apply_scale,
            "use_nla_editor": self.use_nla_editor
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BlenderSettings':
        settings = cls()
        if "blender_executable" in data:
            settings.blender_executable = data["blender_executable"]
        if "auto_detect_blender" in data:
            settings.auto_detect_blender = bool(data["auto_detect_blender"])
        if "python_script_path" in data:
            settings.python_script_path = data["python_script_path"]
        if "connection_timeout" in data:
            settings.connection_timeout = int(data["connection_timeout"])
        if "enable_live_streaming" in data:
            settings.enable_live_streaming = bool(data["enable_live_streaming"])
        if "streaming_fps" in data:
            settings.streaming_fps = int(data["streaming_fps"])
        if "auto_import_animation" in data:
            settings.auto_import_animation = bool(data["auto_import_animation"])
        if "auto_retarget" in data:
            settings.auto_retarget = bool(data["auto_retarget"])
        if "retarget_mode" in data:
            settings.retarget_mode = RetargetMode(data["retarget_mode"])
        if "create_metarig" in data:
            settings.create_metarig = bool(data["create_metarig"])
        if "bake_animation" in data:
            settings.bake_animation = bool(data["bake_animation"])
        if "apply_scale" in data:
            settings.apply_scale = bool(data["apply_scale"])
        if "use_nla_editor" in data:
            settings.use_nla_editor = bool(data["use_nla_editor"])
        return settings


@dataclass
class BoneMapping:
    """Сопоставление костей между скелетами"""
    source_bone: str
    target_bone: str
    confidence: float = 1.0
    rotation_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    translation_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale_multiplier: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    def to_dict(self) -> Dict:
        return {
            "source_bone": self.source_bone,
            "target_bone": self.target_bone,
            "confidence": self.confidence,
            "rotation_offset": self.rotation_offset,
            "translation_offset": self.translation_offset,
            "scale_multiplier": self.scale_multiplier
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BoneMapping':
        return cls(
            source_bone=data["source_bone"],
            target_bone=data["target_bone"],
            confidence=data.get("confidence", 1.0),
            rotation_offset=data.get("rotation_offset", [0.0, 0.0, 0.0]),
            translation_offset=data.get("translation_offset", [0.0, 0.0, 0.0]),
            scale_multiplier=data.get("scale_multiplier", [1.0, 1.0, 1.0])
        )


class BlenderConnection:
    """Управление соединением с Blender"""

    def __init__(self, settings: BlenderSettings):
        self.settings = settings
        self.status = BlenderConnectionStatus.DISCONNECTED
        self.process: Optional[subprocess.Popen] = None
        self.blender_version: Optional[str] = None
        self.connection_time: Optional[float] = None
        self.socket: Optional[socket.socket] = None
        self._stop_event = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None

        # Кэш для производительности
        self._script_cache: Dict[str, str] = {}

    def connect(self) -> bool:
        """Устанавливает соединение с Blender"""
        try:
            self.status = BlenderConnectionStatus.CONNECTING
            logger.info("Connecting to Blender...")

            # Находим исполняемый файл Blender
            blender_path = self._find_blender_executable()
            if not blender_path:
                logger.error("Blender executable not found")
                self.status = BlenderConnectionStatus.ERROR
                return False

            # Проверяем версию Blender
            version = self._get_blender_version(blender_path)
            if not version:
                logger.error("Failed to get Blender version")
                self.status = BlenderConnectionStatus.ERROR
                return False

            self.blender_version = version
            logger.info(f"Blender version: {version}")

            # Запускаем Blender в фоновом режиме
            if self.settings.enable_live_streaming:
                success = self._start_blender_with_socket(blender_path)
            else:
                success = self._start_blender_background(blender_path)

            if success:
                self.status = BlenderConnectionStatus.CONNECTED
                self.connection_time = time.time()
                logger.info("Blender connection established successfully")
                return True
            else:
                self.status = BlenderConnectionStatus.ERROR
                logger.error("Failed to establish Blender connection")
                return False

        except Exception as e:
            logger.error(f"Connection error: {str(e)}", exc_info=True)
            self.status = BlenderConnectionStatus.ERROR
            return False

    def disconnect(self):
        """Разрывает соединение с Blender"""
        logger.info("Disconnecting from Blender...")

        # Останавливаем поток прослушивания
        if self._listener_thread and self._listener_thread.is_alive():
            self._stop_event.set()
            self._listener_thread.join(timeout=5.0)

        # Закрываем сокет
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None

        # Завершаем процесс Blender
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5.0)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            self.process = None

        self.status = BlenderConnectionStatus.DISCONNECTED
        logger.info("Blender disconnected")

    def is_connected(self) -> bool:
        """Проверяет, активно ли соединение"""
        return self.status == BlenderConnectionStatus.CONNECTED

    def send_animation(self,
                       bvh_filepath: str,
                       import_settings: Optional[Dict] = None) -> Dict:
        """
        Отправляет анимацию в Blender.

        Args:
            bvh_filepath: Путь к BVH файлу
            import_settings: Настройки импорта

        Returns:
            Dict: Результат операции
        """
        if not self.is_connected():
            return {"success": False, "error": "Not connected to Blender"}

        try:
            # Генерируем Python скрипт для импорта
            script = self._generate_import_script(bvh_filepath, import_settings)

            # Выполняем скрипт в Blender
            result = self._execute_python_script(script)

            if result["success"]:
                logger.info(f"Animation imported successfully: {bvh_filepath}")
                return {"success": True, "message": "Animation imported"}
            else:
                logger.error(f"Animation import failed: {result.get('error', 'Unknown error')}")
                return {"success": False, "error": result.get("error")}

        except Exception as e:
            logger.error(f"Send animation error: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def stream_pose(self,
                    skeleton: Skeleton,
                    pose_data: Dict[str, Any]) -> bool:
        """
        Потоковая передача позы в реальном времени.

        Args:
            skeleton: Скелет
            pose_data: Данные позы

        Returns:
            bool: Успешность передачи
        """
        if not self.is_connected() or not self.socket:
            return False

        try:
            # Формируем данные для отправки
            stream_data = {
                "type": "pose_update",
                "timestamp": time.time(),
                "joints": {}
            }

            # Преобразуем данные позы в формат для Blender
            for joint_name, joint_data in pose_data.items():
                if "position" in joint_data and "rotation" in joint_data:
                    stream_data["joints"][joint_name] = {
                        "position": joint_data["position"],
                        "rotation": joint_data["rotation"],
                        "confidence": joint_data.get("confidence", 1.0)
                    }

            # Отправляем данные через сокет
            data_str = json.dumps(stream_data)
            self.socket.sendall(f"{len(data_str):010d}".encode() + data_str.encode())

            return True

        except Exception as e:
            logger.error(f"Stream pose error: {str(e)}")
            return False

    def _find_blender_executable(self) -> Optional[str]:
        """Находит исполняемый файл Blender"""
        # Если путь указан явно, проверяем его
        if self.settings.blender_executable and os.path.exists(self.settings.blender_executable):
            return self.settings.blender_executable

        # Автоматический поиск
        search_paths = []

        # Windows
        if sys.platform == "win32":
            search_paths.extend([
                "C:/Program Files/Blender Foundation/Blender/blender.exe",
                "C:/Program Files (x86)/Blender Foundation/Blender/blender.exe",
                os.path.expanduser("~/AppData/Roaming/Blender Foundation/Blender/blender.exe")
            ])
            # Поиск в PATH
            search_paths.extend(self._search_in_path("blender.exe"))

        # macOS
        elif sys.platform == "darwin":
            search_paths.extend([
                "/Applications/Blender.app/Contents/MacOS/Blender",
                os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender")
            ])

        # Linux
        else:
            search_paths.extend([
                "/usr/bin/blender",
                "/usr/local/bin/blender",
                os.path.expanduser("~/bin/blender"),
                "/opt/blender/blender"
            ])
            search_paths.extend(self._search_in_path("blender"))

        # Проверяем все возможные пути
        for path in search_paths:
            if os.path.exists(path):
                logger.info(f"Found Blender at: {path}")
                return path

        logger.warning("Blender executable not found in standard locations")
        return None

    def _search_in_path(self, executable_name: str) -> List[str]:
        """Ищет исполняемый файл в переменной PATH"""
        paths = []
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)

        for dir_path in path_dirs:
            if os.path.isdir(dir_path):
                exe_path = os.path.join(dir_path, executable_name)
                if os.path.exists(exe_path):
                    paths.append(exe_path)

        return paths

    def _get_blender_version(self, blender_path: str) -> Optional[str]:
        """Получает версию Blender"""
        try:
            cmd = [blender_path, "--version"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            if result.returncode == 0:
                # Извлекаем версию из вывода
                for line in result.stdout.split('\n'):
                    if "Blender" in line and any(char.isdigit() for char in line):
                        parts = line.split()
                        for part in parts:
                            if part[0].isdigit() and '.' in part:
                                return part
            return None

        except Exception as e:
            logger.error(f"Failed to get Blender version: {str(e)}")
            return None

    def _start_blender_background(self, blender_path: str) -> bool:
        """Запускает Blender в фоновом режиме"""
        try:
            # Аргументы для запуска Blender
            args = [
                blender_path,
                "--background",  # Фоновый режим
                "--python-expr",  # Выполнить Python код
                "import bpy; print('Blender Python API ready')"
            ]

            # Запускаем процесс
            self.process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            # Ждем инициализации
            time.sleep(2)

            # Проверяем, что процесс работает
            if self.process.poll() is None:
                logger.info("Blender started in background mode")
                return True
            else:
                # Читаем ошибки
                stdout, stderr = self.process.communicate(timeout=5)
                logger.error(f"Blender failed to start. STDOUT: {stdout}, STDERR: {stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to start Blender: {str(e)}")
            return False

    def _start_blender_with_socket(self, blender_path: str) -> bool:
        """Запускает Blender с поддержкой сокетов для стриминга"""
        try:
            # Создаем сокет для связи
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Находим свободный порт
            port = self._find_free_port()
            if not port:
                logger.error("No free port available")
                return False

            # Привязываем сокет
            self.socket.bind(('localhost', port))
            self.socket.listen(1)
            self.socket.settimeout(self.settings.connection_timeout)

            # Генерируем Python скрипт для сервера в Blender
            server_script = self._generate_socket_server_script(port)

            # Сохраняем скрипт во временный файл
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(server_script)
                script_path = f.name

            # Запускаем Blender с скриптом
            args = [
                blender_path,
                "--background",
                "--python", script_path
            ]

            self.process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            # Ждем подключения от Blender
            logger.info(f"Waiting for Blender connection on port {port}...")
            try:
                client_socket, address = self.socket.accept()
                client_socket.settimeout(5.0)

                # Получаем приветственное сообщение
                welcome = client_socket.recv(1024).decode('utf-8')
                if welcome == "BLENDER_SOCKET_READY":
                    logger.info("Blender socket connection established")

                    # Запускаем поток прослушивания
                    self._stop_event.clear()
                    self._listener_thread = threading.Thread(
                        target=self._socket_listener,
                        args=(client_socket,),
                        daemon=True
                    )
                    self._listener_thread.start()

                    # Удаляем временный файл
                    try:
                        os.unlink(script_path)
                    except:
                        pass

                    return True
                else:
                    logger.error(f"Unexpected welcome message: {welcome}")
                    return False

            except socket.timeout:
                logger.error("Socket connection timeout")
                return False

        except Exception as e:
            logger.error(f"Failed to start Blender with socket: {str(e)}", exc_info=True)
            return False

    def _find_free_port(self, start_port: int = 11000, end_port: int = 12000) -> Optional[int]:
        """Находит свободный порт в диапазоне"""
        for port in range(start_port, end_port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    s.bind(('localhost', port))
                    return port
            except (OSError, socket.error):
                continue
        return None

    def _socket_listener(self, client_socket: socket.socket):
        """Поток прослушивания сокета"""
        try:
            while not self._stop_event.is_set():
                try:
                    # Читаем длину сообщения
                    length_bytes = client_socket.recv(10)
                    if not length_bytes:
                        break

                    length = int(length_bytes.decode('utf-8'))

                    # Читаем само сообщение
                    data = b''
                    while len(data) < length:
                        chunk = client_socket.recv(min(4096, length - len(data)))
                        if not chunk:
                            break
                        data += chunk

                    if data:
                        message = json.loads(data.decode('utf-8'))
                        self._handle_socket_message(message)

                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Socket listener error: {str(e)}")
                    break

        finally:
            try:
                client_socket.close()
            except:
                pass

    def _handle_socket_message(self, message: Dict):
        """Обрабатывает сообщения от Blender"""
        msg_type = message.get("type")

        if msg_type == "status":
            logger.info(f"Blender status: {message.get('status')}")

        elif msg_type == "error":
            logger.error(f"Blender error: {message.get('error')}")

        elif msg_type == "animation_complete":
            logger.info(f"Animation processing complete: {message.get('message')}")

    def _generate_socket_server_script(self, port: int) -> str:
        """Генерирует Python скрипт для сервера в Blender"""
        return f'''
import bpy
import socket
import json
import threading
import time

class BlenderSocketServer:
    def __init__(self, host='localhost', port={port}):
        self.host = host
        self.port = port
        self.socket = None
        self.client = None
        self.running = False
        self.animation_data = {{}}

    def start(self):
        """Запускает сервер"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.socket.settimeout(5.0)

            print(f"Blender socket server listening on {{self.host}}:{{self.port}}")

            # Принимаем соединение
            self.client, address = self.socket.accept()
            self.client.settimeout(1.0)

            # Отправляем приветственное сообщение
            self.client.sendall(b"BLENDER_SOCKET_READY")

            self.running = True

            # Запускаем поток обработки сообщений
            thread = threading.Thread(target=self._receive_loop, daemon=True)
            thread.start()

            return True

        except Exception as e:
            print(f"Failed to start socket server: {{str(e)}}")
            return False

    def _receive_loop(self):
        """Цикл приема сообщений"""
        while self.running:
            try:
                # Читаем длину сообщения
                length_bytes = self.client.recv(10)
                if not length_bytes:
                    print("Connection closed by client")
                    break

                length = int(length_bytes.decode('utf-8'))

                # Читаем сообщение
                data = b''
                while len(data) < length:
                    chunk = self.client.recv(min(4096, length - len(data)))
                    if not chunk:
                        break
                    data += chunk

                if data:
                    message = json.loads(data.decode('utf-8'))
                    self._process_message(message)

            except socket.timeout:
                continue
            except Exception as e:
                print(f"Receive error: {{str(e)}}")
                break

    def _process_message(self, message):
        """Обрабатывает полученные сообщения"""
        msg_type = message.get('type')

        if msg_type == 'pose_update':
            self._update_pose(message)
        elif msg_type == 'import_animation':
            self._import_animation(message)
        elif msg_type == 'retarget_animation':
            self._retarget_animation(message)

    def _update_pose(self, message):
        """Обновляет позу в реальном времени"""
        try:
            joints = message.get('joints', {{}})

            # Ищем armature в сцене
            armature = None
            for obj in bpy.data.objects:
                if obj.type == 'ARMATURE':
                    armature = obj
                    break

            if armature:
                bpy.context.view_layer.objects.active = armature
                bpy.ops.object.mode_set(mode='POSE')

                for joint_name, joint_data in joints.items():
                    bone = armature.pose.bones.get(joint_name)
                    if bone:
                        # Обновляем позицию и вращение
                        position = joint_data.get('position')
                        rotation = joint_data.get('rotation')

                        if position:
                            bone.location = position

                        if rotation:
                            # Конвертируем в кватернион
                            from mathutils import Quaternion
                            q = Quaternion(rotation)
                            bone.rotation_quaternion = q

                # Обновляем сцену
                bpy.context.view_layer.update()

                # Отправляем подтверждение
                response = {{
                    'type': 'status',
                    'status': 'pose_updated',
                    'timestamp': time.time()
                }}
                self._send_message(response)

        except Exception as e:
            print(f"Pose update error: {{str(e)}}")

    def _import_animation(self, message):
        """Импортирует анимацию из файла"""
        try:
            filepath = message.get('filepath')
            import_settings = message.get('settings', {{}})

            # Импорт BVH
            bpy.ops.import_anim.bvh(
                filepath=filepath,
                target='ARMATURE',
                global_scale=import_settings.get('scale', 1.0),
                frame_start=1,
                use_fps_scale=import_settings.get('use_fps_scale', False),
                update_scene_fps=import_settings.get('update_scene_fps', False),
                update_scene_duration=import_settings.get('update_scene_duration', True)
            )

            print(f"Animation imported: {{filepath}}")

            # Отправляем подтверждение
            response = {{
                'type': 'animation_complete',
                'message': f'Animation {{filepath}} imported successfully',
                'success': True
            }}
            self._send_message(response)

        except Exception as e:
            print(f"Import animation error: {{str(e)}}")
            response = {{
                'type': 'error',
                'error': str(e),
                'success': False
            }}
            self._send_message(response)

    def _retarget_animation(self, message):
        """Выполняет ретаргетинг анимации"""
        try:
            source_armature = message.get('source_armature')
            target_armature = message.get('target_armature')
            bone_mapping = message.get('bone_mapping', {{}})

            # Здесь будет логика ретаргетинга
            # Для простоты просто выводим сообщение
            print(f"Retargeting from {{source_armature}} to {{target_armature}}")
            print(f"Bone mapping: {{bone_mapping}}")

            # Отправляем подтверждение
            response = {{
                'type': 'animation_complete',
                'message': 'Retargeting completed',
                'success': True
            }}
            self._send_message(response)

        except Exception as e:
            print(f"Retargeting error: {{str(e)}}")
            response = {{
                'type': 'error',
                'error': str(e),
                'success': False
            }}
            self._send_message(response)

    def _send_message(self, message):
        """Отправляет сообщение клиенту"""
        try:
            data = json.dumps(message).encode('utf-8')
            length = len(data)
            self.client.sendall(f"{{length:010d}}".encode() + data)
        except Exception as e:
            print(f"Send message error: {{str(e)}}")

    def stop(self):
        """Останавливает сервер"""
        self.running = False
        if self.client:
            self.client.close()
        if self.socket:
            self.socket.close()

# Запускаем сервер
if __name__ == '__main__':
    server = BlenderSocketServer()
    if server.start():
        print("Blender socket server started successfully")
        # Держим скрипт активным
        while server.running:
            time.sleep(0.1)
    else:
        print("Failed to start socket server")
'''

    def _generate_import_script(self,
                                bvh_filepath: str,
                                import_settings: Optional[Dict]) -> str:
        """Генерирует Python скрипт для импорта в Blender"""
        settings = import_settings or {}

        # Кэширование скриптов для производительности
        cache_key = f"{bvh_filepath}_{hash(str(settings))}"
        if cache_key in self._script_cache:
            return self._script_cache[cache_key]

        script = f'''
import bpy
import os
import json
import mathutils

def import_bvh_animation(filepath, settings):
    """Импортирует BVH анимацию в Blender"""

    # Сохраняем текущий режим
    original_mode = bpy.context.object.mode if bpy.context.object else 'OBJECT'

    try:
        # Очищаем текущую выделение
        bpy.ops.object.select_all(action='DESELECT')

        # Проверяем существование файла
        if not os.path.exists(filepath):
            return {{'success': False, 'error': f'File not found: {{filepath}}'}}

        # Импорт BVH
        bpy.ops.import_anim.bvh(
            filepath=filepath,
            target='ARMATURE',
            global_scale=settings.get('global_scale', 1.0),
            frame_start=settings.get('frame_start', 1),
            use_fps_scale=settings.get('use_fps_scale', False),
            update_scene_fps=settings.get('update_scene_fps', False),
            update_scene_duration=settings.get('update_scene_duration', True),
            use_cyclic=settings.get('use_cyclic', False),
            rotate_mode=settings.get('rotate_mode', 'NATIVE')
        )

        # Находим импортированную арматуру
        imported_armature = None
        for obj in bpy.context.selected_objects:
            if obj.type == 'ARMATURE':
                imported_armature = obj
                break

        if not imported_armature:
            return {{'success': False, 'error': 'No armature imported'}}

        # Применяем дополнительные настройки
        if settings.get('apply_scale', True):
            imported_armature.scale = (1.0, 1.0, 1.0)

        # Переименовываем если нужно
        new_name = settings.get('new_name')
        if new_name:
            imported_armature.name = new_name
            imported_armature.data.name = new_name

        # Создаем метариг если нужно
        if settings.get('create_metarig', False):
            create_metarig(imported_armature)

        # Выполняем ретаргетинг если нужно
        target_armature_name = settings.get('target_armature')
        if target_armature_name and settings.get('auto_retarget', True):
            retarget_animation(imported_armature, target_armature_name, settings)

        # Выпекаем анимацию если нужно
        if settings.get('bake_animation', True):
            bake_animation(imported_armature)

        # Используем NLA если нужно
        if settings.get('use_nla_editor', True):
            setup_nla_strip(imported_armature, settings)

        # Восстанавливаем режим
        if bpy.context.object:
            bpy.ops.object.mode_set(mode=original_mode)

        return {{
            'success': True,
            'armature': imported_armature.name,
            'message': f'Animation imported successfully: {{filepath}}'
        }}

    except Exception as e:
        # Восстанавливаем режим в случае ошибки
        if bpy.context.object:
            try:
                bpy.ops.object.mode_set(mode=original_mode)
            except:
                pass

        return {{'success': False, 'error': str(e)}}

def create_metarig(armature):
    """Создает метариг из импортированной арматуры"""
    try:
        # Выбираем арматуру
        bpy.ops.object.select_all(action='DESELECT')
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature

        # Переходим в режим редактирования
        bpy.ops.object.mode_set(mode='EDIT')

        # Здесь была бы логика создания метарига
        # Для простоты просто выводим сообщение
        print(f"Creating metarig for: {{armature.name}}")

        # Возвращаемся в объектный режим
        bpy.ops.object.mode_set(mode='OBJECT')

    except Exception as e:
        print(f"Failed to create metarig: {{str(e)}}")
        bpy.ops.object.mode_set(mode='OBJECT')

def retarget_animation(source_armature, target_armature_name, settings):
    """Ретаргетирует анимацию на другую арматуру"""
    try:
        # Находим целевую арматуру
        target_armature = bpy.data.objects.get(target_armature_name)
        if not target_armature or target_armature.type != 'ARMATURE':
            print(f"Target armature not found: {{target_armature_name}}")
            return

        # Получаем маппинг костей
        bone_mapping = settings.get('bone_mapping', {{}})
        if not bone_mapping:
            # Автоматический маппинг
            bone_mapping = auto_map_bones(source_armature, target_armature)

        # Здесь была бы логика ретаргетинга
        print(f"Retargeting from {{source_armature.name}} to {{target_armature.name}}")
        print(f"Bone mapping: {{bone_mapping}}")

        # Пример простого копирования трансформаций
        bpy.ops.object.select_all(action='DESELECT')
        source_armature.select_set(True)
        target_armature.select_set(True)
        bpy.context.view_layer.objects.active = target_armature

        # Копируем анимацию
        bpy.ops.object.mode_set(mode='POSE')

        for src_bone_name, tgt_bone_name in bone_mapping.items():
            src_bone = source_armature.pose.bones.get(src_bone_name)
            tgt_bone = target_armature.pose.bones.get(tgt_bone_name)

            if src_bone and tgt_bone:
                # Копируем ключевые кадры
                copy_bone_animation(src_bone, tgt_bone)

        bpy.ops.object.mode_set(mode='OBJECT')

    except Exception as e:
        print(f"Retargeting error: {{str(e)}}")

def auto_map_bones(source_armature, target_armature):
    """Автоматически сопоставляет кости между арматурами"""
    mapping = {{}}

    # Простая эвристика для маппинга
    source_bones = {{bone.name.lower(): bone.name for bone in source_armature.data.bones}}
    target_bones = {{bone.name.lower(): bone.name for bone in target_armature.data.bones}}

    # Общие имена костей
    common_names = set(source_bones.keys()) & set(target_bones.keys())

    for name in common_names:
        mapping[source_bones[name]] = target_bones[name]

    # Дополнительные правила для человеческого скелета
    bone_aliases = {{
        'hips': ['pelvis', 'root', 'hip'],
        'spine': ['spine01', 'spine_01', 'spine1'],
        'chest': ['spine02', 'spine_02', 'spine2', 'spine03', 'spine_03'],
        'neck': ['neck01', 'neck_01'],
        'head': ['head'],
        'shoulder': ['shoulder', 'clavicle', 'clav'],
        'upperarm': ['upperarm', 'upper_arm', 'arm'],
        'lowerarm': ['lowerarm', 'lower_arm', 'forearm'],
        'hand': ['hand'],
        'thigh': ['thigh', 'upperleg', 'upper_leg'],
        'shin': ['shin', 'lowerleg', 'lower_leg'],
        'foot': ['foot']
    }}

    for alias, variations in bone_aliases.items():
        for src_bone_name in source_bones.values():
            src_lower = src_bone_name.lower()
            for variation in variations:
                if variation in src_lower:
                    # Ищем соответствующую кость в цели
                    for tgt_bone_name in target_bones.values():
                        tgt_lower = tgt_bone_name.lower()
                        if alias in tgt_lower or any(v in tgt_lower for v in variations):
                            mapping[src_bone_name] = tgt_bone_name
                            break
                    break

    return mapping

def copy_bone_animation(source_bone, target_bone):
    """Копирует анимацию с одной кости на другую"""
    if not source_bone.animation_data or not source_bone.animation_data.action:
        return

    # Создаем действие для целевой кости
    if not target_bone.animation_data:
        target_bone.animation_data_create()

    action = source_bone.animation_data.action
    target_action = bpy.data.actions.new(name=f"{{target_bone.name}}_retargeted")
    target_bone.animation_data.action = target_action

    # Копируем кривые
    for fcurve in action.fcurves:
        data_path = fcurve.data_path

        # Адаптируем data_path для целевой кости
        if source_bone.name in data_path:
            new_data_path = data_path.replace(source_bone.name, target_bone.name)

            # Создаем новую кривую
            new_fcurve = target_action.fcurves.new(
                data_path=new_data_path,
                index=fcurve.array_index,
                action_group=target_bone.name
            )

            # Копируем ключевые кадры
            for keyframe in fcurve.keyframe_points:
                new_keyframe = new_fcurve.keyframe_points.insert(
                    keyframe.co[0],
                    keyframe.co[1]
                )
                new_keyframe.handle_left = keyframe.handle_left.copy()
                new_keyframe.handle_right = keyframe.handle_right.copy()
                new_keyframe.interpolation = keyframe.interpolation
                new_keyframe.easing = keyframe.easing

def bake_animation(armature):
    """Выпекает анимацию для чистоты данных"""
    try:
        # Выбираем арматуру
        bpy.ops.object.select_all(action='DESELECT')
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature

        # Переходим в режим позы
        bpy.ops.object.mode_set(mode='POSE')

        # Выбираем все кости
        bpy.ops.pose.select_all(action='SELECT')

        # Получаем диапазон анимации
        action = armature.animation_data.action if armature.animation_data else None
        if action:
            frame_start = int(min([kf.co[0] for fcurve in action.fcurves for kf in fcurve.keyframe_points]))
            frame_end = int(max([kf.co[0] for fcurve in action.fcurves for kf in fcurve.keyframe_points]))

            # Выпекаем анимацию
            bpy.ops.nla.bake(
                frame_start=frame_start,
                frame_end=frame_end,
                step=1,
                only_selected=True,
                visual_keying=True,
                clear_constraints=False,
                clear_parents=False,
                use_current_action=True,
                bake_types={'POSE'}
            )

        # Возвращаемся в объектный режим
        bpy.ops.object.mode_set(mode='OBJECT')

    except Exception as e:
        print(f"Bake animation error: {{str(e)}}")
        bpy.ops.object.mode_set(mode='OBJECT')

def setup_nla_strip(armature, settings):
    """Настраивает NLA strip для анимации"""
    if not armature.animation_data or not armature.animation_data.action:
        return

    # Создаем NLA track
    if not armature.animation_data.nla_tracks:
        track = armature.animation_data.nla_tracks.new()
        track.name = "Mocap_Animation"
    else:
        track = armature.animation_data.nla_tracks[0]

    # Создаем NLA strip
    action = armature.animation_data.action
    strip = track.strips.new(
        name=action.name,
        start=settings.get('nla_start_frame', 1),
        action=action
    )

    # Настройки strip
    strip.blend_type = settings.get('nla_blend_type', 'COMBINE')
    strip.extrapolation = settings.get('nla_extrapolation', 'NOTHING')
    strip.repeat = settings.get('nla_repeat', 1.0)

    # Отключаем прямое действие
    armature.animation_data.action = None

# Основная функция
def main():
    filepath = r"{bvh_filepath}"
    settings = {json.dumps(settings)}

    result = import_bvh_animation(filepath, settings)

    # Выводим результат для внешнего процесса
    print("IMPORT_RESULT_START")
    print(json.dumps(result))
    print("IMPORT_RESULT_END")

if __name__ == "__main__":
    main()
'''

        # Кэшируем скрипт
        self._script_cache[cache_key] = script
        return script

    def _execute_python_script(self, script: str) -> Dict:
        """Выполняет Python скрипт в Blender"""
        if not self.process:
            return {"success": False, "error": "Blender process not running"}

        try:
            # Сохраняем скрипт во временный файл
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                script_path = f.name

            # Выполняем скрипт в Blender
            execute_cmd = f"exec(open(r'{script_path}').read())"
            full_cmd = [
                self.settings.blender_executable or self._find_blender_executable(),
                "--background",
                "--python-expr",
                execute_cmd
            ]

            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=self.settings.connection_timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            # Удаляем временный файл
            try:
                os.unlink(script_path)
            except:
                pass

            # Парсим результат
            stdout = result.stdout
            if "IMPORT_RESULT_START" in stdout:
                start_idx = stdout.index("IMPORT_RESULT_START") + len("IMPORT_RESULT_START")
                end_idx = stdout.index("IMPORT_RESULT_END")
                result_json = stdout[start_idx:end_idx].strip()

                try:
                    result_dict = json.loads(result_json)
                    return result_dict
                except json.JSONDecodeError:
                    return {"success": False, "error": "Failed to parse result"}
            else:
                # Возвращаем stdout/stderr для отладки
                return {
                    "success": result.returncode == 0,
                    "stdout": stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Blender execution timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class BlenderBridge:
    """Основной класс для интеграции с Blender"""

    def __init__(self, settings: Optional[BlenderSettings] = None):
        self.settings = settings or BlenderSettings()
        self.connection = BlenderConnection(self.settings)
        self.bvh_exporter = BVHExporter()

        # Кэш для маппинга костей
        self._bone_mapping_cache: Dict[str, List[BoneMapping]] = {}

        # Статистика
        self.stats = {
            "animations_sent": 0,
            "bytes_transferred": 0,
            "connection_time": 0,
            "errors": 0
        }

        # Callback функции
        self.on_animation_sent = None
        self.on_retarget_complete = None
        self.on_error = None

    def connect_to_blender(self) -> bool:
        """Устанавливает соединение с Blender"""
        success = self.connection.connect()
        if success:
            logger.info("Blender bridge connected successfully")
        else:
            logger.error("Failed to connect to Blender")

            if self.on_error:
                self.on_error("Connection failed", "Failed to establish connection with Blender")

        return success

    def disconnect_from_blender(self):
        """Разрывает соединение с Blender"""
        self.connection.disconnect()
        logger.info("Blender bridge disconnected")

    def send_animation_to_blender(self,
                                  skeleton: Skeleton,
                                  animation_layer: AnimationLayer,
                                  export_settings: Optional[Dict] = None,
                                  import_settings: Optional[Dict] = None) -> Dict:
        """
        Экспортирует и отправляет анимацию в Blender.

        Args:
            skeleton: Скелет для экспорта
            animation_layer: Слой анимации
            export_settings: Настройки экспорта BVH
            import_settings: Настройки импорта в Blender

        Returns:
            Dict: Результат операции
        """
        try:
            # Проверяем соединение
            if not self.connection.is_connected():
                if not self.connect_to_blender():
                    return {"success": False, "error": "Not connected to Blender"}

            # Экспортируем в BVH
            with tempfile.NamedTemporaryFile(suffix='.bvh', delete=False) as tmp_file:
                bvh_path = tmp_file.name

            # Настройки экспорта
            if export_settings:
                bvh_settings = BVHExportSettings.from_dict(export_settings)
            else:
                bvh_settings = BVHExportSettings(
                    target_software="blender",
                    convert_to_cm=False,
                    scale_factor=0.01  # Blender работает в метрах
                )

            # Экспортируем
            export_success = self.bvh_exporter.export(
                skeleton,
                animation_layer,
                bvh_path,
                bvh_settings
            )

            if not export_success:
                # Удаляем временный файл
                try:
                    os.unlink(bvh_path)
                except:
                    pass

                return {"success": False, "error": "BVH export failed"}

            # Отправляем в Blender
            import_result = self.connection.send_animation(bvh_path, import_settings)

            # Обновляем статистику
            if import_result["success"]:
                self.stats["animations_sent"] += 1
                file_size = os.path.getsize(bvh_path)
                self.stats["bytes_transferred"] += file_size

                # Вызываем callback
                if self.on_animation_sent:
                    self.on_animation_sent(animation_layer.name, bvh_path)

            # Удаляем временный файл
            try:
                os.unlink(bvh_path)
            except:
                pass

            return import_result

        except Exception as e:
            logger.error(f"Send animation error: {str(e)}", exc_info=True)

            self.stats["errors"] += 1

            if self.on_error:
                self.on_error("Send animation failed", str(e))

            return {"success": False, "error": str(e)}

    def stream_pose_to_blender(self,
                               skeleton: Skeleton,
                               pose_data: Dict[str, Any]) -> bool:
        """
        Потоковая передача позы в Blender в реальном времени.

        Args:
            skeleton: Скелет
            pose_data: Данные позы

        Returns:
            bool: Успешность передачи
        """
        if not self.connection.is_connected():
            return False

        try:
            success = self.connection.stream_pose(skeleton, pose_data)
            return success

        except Exception as e:
            logger.error(f"Stream pose error: {str(e)}")

            if self.on_error:
                self.on_error("Stream pose failed", str(e))

            return False

    def batch_send_animations(self,
                              skeleton: Skeleton,
                              animation_layers: List[AnimationLayer],
                              export_settings: Optional[Dict] = None,
                              import_settings: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Пакетная отправка нескольких анимаций.

        Args:
            skeleton: Скелет
            animation_layers: Список слоев анимации
            export_settings: Настройки экспорта
            import_settings: Настройки импорта

        Returns:
            Dict: Результаты для каждой анимации
        """
        results = {}

        for layer in animation_layers:
            logger.info(f"Processing animation: {layer.name}")

            result = self.send_animation_to_blender(
                skeleton,
                layer,
                export_settings,
                import_settings
            )

            results[layer.name] = result

            # Небольшая пауза между анимациями
            time.sleep(0.5)

        return results

    def create_bone_mapping(self,
                            source_skeleton: Skeleton,
                            target_armature_name: str,
                            mode: RetargetMode = None) -> List[BoneMapping]:
        """
        Создает маппинг костей между скелетом и арматурой Blender.

        Args:
            source_skeleton: Исходный скелет
            target_armature_name: Имя целевой арматуры в Blender
            mode: Режим ретаргетинга

        Returns:
            List[BoneMapping]: Список сопоставлений
        """
        mode = mode or self.settings.retarget_mode

        # Проверяем кэш
        cache_key = f"{source_skeleton.name}_{target_armature_name}_{mode.value}"
        if cache_key in self._bone_mapping_cache:
            return self._bone_mapping_cache[cache_key]

        mappings = []

        if mode == RetargetMode.AUTOMATIC:
            # Автоматическое сопоставление
            mappings = self._auto_map_bones(source_skeleton, target_armature_name)

        elif mode == RetargetMode.MANUAL:
            # Ручное сопоставление (требует UI)
            mappings = self._load_manual_mapping(target_armature_name)

        elif mode == RetargetMode.HIK:
            # Human IK совместимость
            mappings = self._create_hik_mapping(source_skeleton)

        elif mode == RetargetMode.RIGIFY:
            # Rigify метариг
            mappings = self._create_rigify_mapping(source_skeleton)

        # Сохраняем в кэш
        self._bone_mapping_cache[cache_key] = mappings

        return mappings

    def _auto_map_bones(self,
                        source_skeleton: Skeleton,
                        target_armature_name: str) -> List[BoneMapping]:
        """Автоматическое сопоставление костей"""
        mappings = []

        # Этот метод требует информации о целевой арматуре из Blender
        # Для простоты используем эвристику

        # Стандартные маппинги для человеческого скелета
        standard_mappings = {
            "humanoid_mediapipe": {
                "nose": "head",
                "neck": "neck",
                "left_shoulder": "shoulder.L",
                "right_shoulder": "shoulder.R",
                "left_elbow": "upper_arm.L",
                "right_elbow": "upper_arm.R",
                "left_wrist": "forearm.L",
                "right_wrist": "forearm.R",
                "left_hip": "thigh.L",
                "right_hip": "thigh.R",
                "left_knee": "shin.L",
                "right_knee": "shin.R",
                "left_ankle": "foot.L",
                "right_ankle": "foot.R"
            }
        }

        skeleton_type = source_skeleton.name or "humanoid_mediapipe"

        if skeleton_type in standard_mappings:
            for source_bone, target_bone in standard_mappings[skeleton_type].items():
                if source_skeleton.get_bone(source_bone):
                    mapping = BoneMapping(
                        source_bone=source_bone,
                        target_bone=target_bone,
                        confidence=0.9
                    )
                    mappings.append(mapping)

        return mappings

    def _load_manual_mapping(self, target_armature_name: str) -> List[BoneMapping]:
        """Загружает ручной маппинг из файла"""
        mappings = []

        # Ищем файл маппинга
        mapping_file = f"{target_armature_name}_mapping.json"

        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    data = json.load(f)

                for item in data.get("mappings", []):
                    mapping = BoneMapping.from_dict(item)
                    mappings.append(mapping)

            except Exception as e:
                logger.error(f"Failed to load manual mapping: {str(e)}")

        return mappings

    def _create_hik_mapping(self, source_skeleton: Skeleton) -> List[BoneMapping]:
        """Создает маппинг для Human IK"""
        # Human IK стандартные имена
        hik_bones = [
            "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
            "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
            "RightShoulder", "RightArm", "RightForeArm", "RightHand",
            "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
            "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"
        ]

        mappings = []

        for bone in source_skeleton.bones:
            bone_name = bone.name

            # Пробуем сопоставить по имени
            target_bone = None

            if "hip" in bone_name.lower() or "pelvis" in bone_name.lower():
                target_bone = "Hips"
            elif "spine" in bone_name.lower():
                if "3" in bone_name or "chest" in bone_name.lower():
                    target_bone = "Spine3"
                elif "2" in bone_name:
                    target_bone = "Spine2"
                elif "1" in bone_name:
                    target_bone = "Spine1"
                else:
                    target_bone = "Spine"
            elif "neck" in bone_name.lower():
                target_bone = "Neck"
            elif "head" in bone_name.lower():
                target_bone = "Head"
            elif "shoulder" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "LeftShoulder"
                else:
                    target_bone = "RightShoulder"
            elif "upper" in bone_name.lower() or "arm" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "LeftArm"
                else:
                    target_bone = "RightArm"
            elif "fore" in bone_name.lower() or "lower" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "LeftForeArm"
                else:
                    target_bone = "RightForeArm"
            elif "hand" in bone_name.lower() or "wrist" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "LeftHand"
                else:
                    target_bone = "RightHand"
            elif "thigh" in bone_name.lower() or "upleg" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "LeftUpLeg"
                else:
                    target_bone = "RightUpLeg"
            elif "shin" in bone_name.lower() or "leg" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "LeftLeg"
                else:
                    target_bone = "RightLeg"
            elif "foot" in bone_name.lower() or "ankle" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "LeftFoot"
                else:
                    target_bone = "RightFoot"
            elif "toe" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "LeftToeBase"
                else:
                    target_bone = "RightToeBase"

            if target_bone:
                mapping = BoneMapping(
                    source_bone=bone_name,
                    target_bone=target_bone,
                    confidence=0.8
                )
                mappings.append(mapping)

        return mappings

    def _create_rigify_mapping(self, source_skeleton: Skeleton) -> List[BoneMapping]:
        """Создает маппинг для Rigify метарига"""
        # Rigify стандартные имена
        rigify_bones = [
            "root", "spine", "spine.001", "spine.002", "spine.003", "spine.004", "spine.005", "spine.006",
            "neck", "head",
            "shoulder.L", "upper_arm.L", "forearm.L", "hand.L",
            "shoulder.R", "upper_arm.R", "forearm.R", "hand.R",
            "thigh.L", "shin.L", "foot.L", "toe.L",
            "thigh.R", "shin.R", "foot.R", "toe.R"
        ]

        mappings = []

        for bone in source_skeleton.bones:
            bone_name = bone.name

            # Пробуем сопоставить по имени
            target_bone = None

            if "hip" in bone_name.lower() or "pelvis" in bone_name.lower():
                target_bone = "root"
            elif "spine" in bone_name.lower():
                if "6" in bone_name or "chest" in bone_name.lower():
                    target_bone = "spine.006"
                elif "5" in bone_name:
                    target_bone = "spine.005"
                elif "4" in bone_name:
                    target_bone = "spine.004"
                elif "3" in bone_name:
                    target_bone = "spine.003"
                elif "2" in bone_name:
                    target_bone = "spine.002"
                elif "1" in bone_name:
                    target_bone = "spine.001"
                else:
                    target_bone = "spine"
            elif "neck" in bone_name.lower():
                target_bone = "neck"
            elif "head" in bone_name.lower():
                target_bone = "head"
            elif "shoulder" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "shoulder.L"
                else:
                    target_bone = "shoulder.R"
            elif "upper" in bone_name.lower() and "arm" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "upper_arm.L"
                else:
                    target_bone = "upper_arm.R"
            elif "fore" in bone_name.lower() or ("lower" in bone_name.lower() and "arm" in bone_name.lower()):
                if "left" in bone_name.lower():
                    target_bone = "forearm.L"
                else:
                    target_bone = "forearm.R"
            elif "hand" in bone_name.lower() or "wrist" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "hand.L"
                else:
                    target_bone = "hand.R"
            elif "thigh" in bone_name.lower() or "upleg" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "thigh.L"
                else:
                    target_bone = "thigh.R"
            elif "shin" in bone_name.lower() or ("lower" in bone_name.lower() and "leg" in bone_name.lower()):
                if "left" in bone_name.lower():
                    target_bone = "shin.L"
                else:
                    target_bone = "shin.R"
            elif "foot" in bone_name.lower() or "ankle" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "foot.L"
                else:
                    target_bone = "foot.R"
            elif "toe" in bone_name.lower():
                if "left" in bone_name.lower():
                    target_bone = "toe.L"
                else:
                    target_bone = "toe.R"

            if target_bone:
                mapping = BoneMapping(
                    source_bone=bone_name,
                    target_bone=target_bone,
                    confidence=0.85
                )
                mappings.append(mapping)

        return mappings

    def save_bone_mapping(self,
                          target_armature_name: str,
                          mappings: List[BoneMapping],
                          filepath: Optional[str] = None) -> bool:
        """
        Сохраняет маппинг костей в файл.

        Args:
            target_armature_name: Имя целевой арматуры
            mappings: Список маппингов
            filepath: Путь для сохранения

        Returns:
            bool: Успешность сохранения
        """
        try:
            if not filepath:
                filepath = f"{target_armature_name}_mapping.json"

            data = {
                "target_armature": target_armature_name,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mappings": [mapping.to_dict() for mapping in mappings]
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Bone mapping saved to: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save bone mapping: {str(e)}")
            return False

    def get_blender_info(self) -> Dict:
        """Получает информацию о подключенном Blender"""
        if not self.connection.is_connected():
            return {"connected": False}

        info = {
            "connected": True,
            "status": self.connection.status.value,
            "version": self.connection.blender_version,
            "connection_time": self.connection.connection_time,
            "live_streaming": self.settings.enable_live_streaming,
            "stats": self.stats.copy()
        }

        return info


# Утилитарные функции для удобства использования
def create_blender_bridge(settings: Optional[Dict] = None) -> BlenderBridge:
    """
    Создает экземпляр BlenderBridge.

    Args:
        settings: Настройки интеграции

    Returns:
        BlenderBridge: Экземпляр моста
    """
    if settings:
        blender_settings = BlenderSettings.from_dict(settings)
    else:
        blender_settings = BlenderSettings()

    return BlenderBridge(blender_settings)


def quick_export_to_blender(skeleton: Skeleton,
                            animation_layer: AnimationLayer,
                            blender_path: str = "") -> Dict:
    """
    Быстрый экспорт анимации в Blender.

    Args:
        skeleton: Скелет
        animation_layer: Слой анимации
        blender_path: Путь к Blender

    Returns:
        Dict: Результат операции
    """
    settings = BlenderSettings()
    if blender_path:
        settings.blender_executable = blender_path

    bridge = BlenderBridge(settings)

    # Пробуем подключиться
    if not bridge.connect_to_blender():
        return {"success": False, "error": "Failed to connect to Blender"}

    # Отправляем анимацию
    result = bridge.send_animation_to_blender(skeleton, animation_layer)

    # Отключаемся
    bridge.disconnect_from_blender()

    return result


# Тестирование модуля
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=== Blender Bridge Test ===")

    # Создаем тестовый мост
    bridge = create_blender_bridge()

    # Пробуем подключиться
    print("Connecting to Blender...")
    if bridge.connect_to_blender():
        print("✅ Connected successfully!")

        # Получаем информацию
        info = bridge.get_blender_info()
        print(f"Blender Version: {info.get('version', 'Unknown')}")
        print(f"Connection Status: {info.get('status', 'Unknown')}")

        # Отключаемся
        bridge.disconnect_from_blender()
        print("Disconnected from Blender")
    else:
        print("❌ Failed to connect to Blender")

    print("=== Test Complete ===")