#!/usr/bin/env python3
"""
Mocap Pro - Профессиональная система захвата движения
Главный модуль приложения

Версия: 1.0.0
Автор: Mocap Pro Team
Лицензия: MIT
"""

import sys
import os
import logging
import traceback
import signal
from pathlib import Path
from typing import Optional, Dict, Any

# Добавляем пути для импортов
sys.path.insert(0, str(Path(__file__).parent))

# Проверка версии Python
if sys.version_info < (3, 10):
    print("❌ Требуется Python 3.10 или выше")
    sys.exit(1)

try:
    from PyQt6.QtWidgets import QApplication, QSplashScreen, QMessageBox
    from PyQt6.QtCore import Qt, QTimer, QSettings, QLocale, QTranslator
    from PyQt6.QtGui import QPixmap, QFont, QFontDatabase, QIcon, QColor
    import numpy as np
    import cv2
except ImportError as e:
    print(f"❌ Не удалось импортировать необходимые модули: {e}")
    print("📦 Установите зависимости: pip install -r requirements.txt")
    sys.exit(1)

# Импорты Mocap Pro
# Импорты Mocap Pro
try:
    # ДОБАВЬ ЭТИ ДВЕ СТРОКИ:
    from core.skeleton import ProfessionalSkeleton
    from core.animation_recorder import ProfessionalAnimationRecorder

    from core.skeleton_tracker import SkeletonTracker
    from core.camera_manager import CameraManager
    from core.animation_recorder import AnimationRecorder
    from core.pose_estimator import EnhancedPoseEstimator, TrackingMode
    from ui.main_window import ProfessionalMainWindow as MainWindow
    import utils.math_utils as MathUtils
    from config.default_settings import load_settings, save_settings
    from export.bvh_exporter import BVHExporter
    from export.blender_bridge import BlenderBridge
except ImportError as e:
    print(f"❌ Ошибка импорта модулей Mocap Pro: {e}")
    traceback.print_exc()
    sys.exit(1)


class MocapProApplication:
    """Основной класс приложения Mocap Pro"""

    def __init__(self):
        self.app = None
        self.main_window = None
        self.splash = None
        self.settings = None
        self.translator = None
        self.logger = logging.getLogger(__name__)
        # Основные компоненты системы
        self.camera_manager = None
        self.skeleton_tracker = None
        self.animation_recorder = None
        self.bvh_exporter = None
        self.blender_bridge = None

        # Состояние приложения
        self.is_initialized = False
        self.startup_time = None

        # Настройки путей
        self.app_dir = Path(__file__).parent
        self.data_dir = self.app_dir / "data"
        self.config_dir = self.app_dir / "config"
        self.log_dir = self.app_dir / "logs"

        # Сигналы выхода
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Обработчик сигналов завершения"""
        print(f"\n⚠️ Получен сигнал завершения: {signum}")
        self.cleanup()
        sys.exit(0)

    def setup_logging(self):
        """Настраивает систему логирования"""
        try:
            # Создаем директорию для логов
            self.log_dir.mkdir(exist_ok=True)

            # Настраиваем logging
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            log_level = logging.DEBUG if self.settings.get("logging.log_level") == "DEBUG" else logging.INFO

            # Файловый handler
            file_handler = logging.FileHandler(
                self.log_dir / "mocap_pro.log",
                encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(log_format))
            file_handler.setLevel(log_level)

            # Консольный handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            console_handler.setLevel(logging.INFO)

            # Настраиваем root logger
            logging.basicConfig(
                level=log_level,
                handlers=[file_handler, console_handler],
                force=True
            )

            self.logger = logging.getLogger("MocapPro")
            self.logger.info("=" * 60)
            self.logger.info("Mocap Pro Application Starting")
            self.logger.info("=" * 60)

        except Exception as e:
            print(f"❌ Ошибка настройки логирования: {e}")
            # Используем базовое логирование
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("MocapPro")

    def load_settings(self):
        """Загружает настройки приложения"""
        try:
            # Загружаем настройки из файла
            settings_file = self.config_dir / "default_settings.yaml"
            self.settings = load_settings(settings_file)

            # Проверяем и создаем необходимые директории
            self._setup_directories()

            # Загружаем настройки Qt
            self.qt_settings = QSettings("MocapPro", "MocapPro")

            # Восстанавливаем геометрию окна если есть
            if self.qt_settings.contains("MainWindow/Geometry"):
                self.settings["interface.window.geometry"] = self.qt_settings.value("MainWindow/Geometry")

            self.logger.info("✅ Настройки загружены")
            return True

        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки настроек: {e}")
            # Создаем настройки по умолчанию
            self.settings = self._create_default_settings()
            return False

    def _setup_directories(self):
        """Создает необходимые директории"""
        directories = [
            self.data_dir / "models",
            self.data_dir / "presets",
            self.data_dir / "animations",
            self.data_dir / "exports",
            self.data_dir / "calibrations",
            self.log_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Создана директория: {directory}")

    def _create_default_settings(self):
        """Создает настройки по умолчанию"""
        return {
            "application": {
                "name": "Mocap Pro",
                "version": "1.0.0",
                "language": "ru",
                "auto_save": True,
                "auto_save_interval": 300
            },
            "interface": {
                "theme": "dark",
                "font_size": 10,
                "window": {
                    "default_width": 1280,
                    "default_height": 720
                }
            }
        }

    def setup_translations(self):
        """Настраивает систему перевода"""
        try:
            lang = self.settings.get("application.language", "ru")

            if lang != "en":
                translation_file = self.app_dir / f"translations/mocap_pro_{lang}.qm"
                if translation_file.exists():
                    self.translator = QTranslator()
                    if self.translator.load(str(translation_file)):
                        self.app.installTranslator(self.translator)
                        self.logger.info(f"✅ Загружен перевод: {lang}")
                    else:
                        self.logger.warning(f"⚠️ Не удалось загрузить перевод: {lang}")
                else:
                    self.logger.warning(f"⚠️ Файл перевода не найден: {translation_file}")

        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки перевода: {e}")

    def setup_fonts(self):
        """Настраивает шрифты приложения"""
        try:
            # Загружаем кастомные шрифты если есть
            fonts_dir = self.app_dir / "fonts"
            if fonts_dir.exists():
                for font_file in fonts_dir.glob("*.ttf"):
                    font_id = QFontDatabase.addApplicationFont(str(font_file))
                    if font_id != -1:
                        font_families = QFontDatabase.applicationFontFamilies(font_id)
                        self.logger.info(f"✅ Загружен шрифт: {font_families[0]}")

            # Устанавливаем шрифт по умолчанию
            font_family = self.settings.get("interface.font_family", "Segoe UI")
            font_size = self.settings.get("interface.font_size", 10)

            font = QFont(font_family, font_size)
            self.app.setFont(font)

        except Exception as e:
            self.logger.error(f"❌ Ошибка настройки шрифтов: {e}")

    def show_splash_screen(self):
        """Показывает splash screen"""
        try:
            # Пробуем загрузить кастомное изображение
            splash_paths = [
                self.app_dir / "icons/splash.png",
                self.app_dir / "icons/splash.jpg",
                self.app_dir / "data/splash.png"
            ]

            splash_pixmap = None
            for path in splash_paths:
                if path.exists():
                    splash_pixmap = QPixmap(str(path))
                    break

            # Если нет кастомного изображения, создаем программное
            if splash_pixmap is None or splash_pixmap.isNull():
                splash_pixmap = self._create_programmatic_splash()

            self.splash = QSplashScreen(splash_pixmap)
            self.splash.show()

            # Центрируем splash screen
            screen_geometry = self.app.primaryScreen().availableGeometry()
            splash_geometry = self.splash.geometry()
            x = (screen_geometry.width() - splash_geometry.width()) // 2
            y = (screen_geometry.height() - splash_geometry.height()) // 2
            self.splash.move(x, y)

            # Показываем сообщение о загрузке
            self.splash.showMessage(
                "Загрузка Mocap Pro...",
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                QColor(255, 255, 255)
            )

            self.app.processEvents()
            self.logger.info("✅ Splash screen отображен")

        except Exception as e:
            self.logger.error(f"❌ Ошибка отображения splash screen: {e}")
            # Продолжаем без splash screen

    def _create_programmatic_splash(self):
        """Создает программный splash screen"""
        from PyQt6.QtGui import QPainter, QLinearGradient, QBrush
        from PyQt6.QtCore import QRect

        # Создаем изображение 600x400
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor(25, 25, 35))

        painter = QPainter(pixmap)

        try:
            # Градиентный фон
            gradient = QLinearGradient(0, 0, 0, 400)
            gradient.setColorAt(0, QColor(40, 40, 50))
            gradient.setColorAt(1, QColor(20, 20, 30))
            painter.fillRect(QRect(0, 0, 600, 400), QBrush(gradient))

            # Название приложения
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 32, QFont.Weight.Bold))
            painter.drawText(QRect(0, 100, 600, 100),
                             Qt.AlignmentFlag.AlignCenter,
                             "Mocap Pro")

            # Версия
            painter.setFont(QFont("Arial", 14))
            painter.drawText(QRect(0, 180, 600, 50),
                             Qt.AlignmentFlag.AlignCenter,
                             "Professional Motion Capture System")

            # Индикатор загрузки
            painter.setPen(QColor(100, 150, 255))
            painter.setBrush(QColor(100, 150, 255, 100))
            painter.drawRect(150, 300, 300, 20)

            painter.setBrush(QColor(100, 150, 255))
            painter.drawRect(150, 300, 100, 20)  # Часть индикатора

            # Копирайт
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(QRect(0, 380, 600, 20),
                             Qt.AlignmentFlag.AlignCenter,
                             "© 2024 Mocap Pro Team")

        finally:
            painter.end()

        return pixmap

    def initialize_components(self):
        """Инициализирует основные компоненты системы"""
        try:
            # Обновляем сообщение на splash screen
            if self.splash:
                self.splash.showMessage(
                    "Инициализация камер...",
                    Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                    QColor(255, 255, 255)
                )
                self.app.processEvents()

            # 1. Менеджер камер
            self.camera_manager = CameraManager()
            camera_count = len(self.camera_manager.multi_manager.list_cameras())
            self.logger.info(f"📹 Найдено камер: {camera_count}")

            # 2. Pose Estimator
            if self.splash:
                self.splash.showMessage(
                    "Загрузка моделей AI...",
                    Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                    QColor(255, 255, 255)
                )
                self.app.processEvents()

            tracking_mode_str = self.settings.get("tracking.mode", "precise")
            tracking_mode = TrackingMode(tracking_mode_str.lower())
            self.pose_estimator = EnhancedPoseEstimator(
                mode=tracking_mode,
                enable_kalman=self.settings.get("tracking.enable_kalman", True),
                enable_smoothing=self.settings.get("tracking.enable_smoothing", True),
                auto_calibrate=self.settings.get("tracking.auto_calibrate", True)
            )

            # 2.1 СОЗДАЕМ SKELETON (ДОБАВЬ ЭТО)
            if self.splash:
                self.splash.showMessage(
                    "Загрузка скелетной системы...",
                    Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                    QColor(255, 255, 255)
                )
                self.app.processEvents()

            skeleton = ProfessionalSkeleton("DefaultHuman")

            # 2.2 СОЗДАЕМ ANIMATION RECORDER (ДОБАВЬ ЭТО)
            if self.splash:
                self.splash.showMessage(
                    "Подготовка системы записи...",
                    Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                    QColor(255, 255, 255)
                )
                self.app.processEvents()

            animation_recorder = AnimationRecorder()


            # 3. Skeleton Tracker
            if self.splash:
                self.splash.showMessage(
                    "Настройка трекера скелета...",
                    Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                    QColor(255, 255, 255)
                )
                self.app.processEvents()

            self.skeleton_tracker = SkeletonTracker(
                config={
                    'skeleton_name': 'DefaultHuman',
                    'tracking_mode': tracking_mode_str.lower()
                }
            )
            self.skeleton_tracker.skeleton = skeleton
            self.skeleton_tracker.animation_recorder = animation_recorder
            # 4. Animation Recorder (СОХРАНЯЕМ В self)
            if self.splash:
                self.splash.showMessage(
                    "Подготовка системы записи...",
                    Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                    QColor(255, 255, 255)
                )
                self.app.processEvents()

            self.animation_recorder = animation_recorder  # ← сохраняем созданный

            # 5. Экспортеры
            if self.splash:
                self.splash.showMessage(
                    "Инициализация экспортёров...",
                    Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                    QColor(255, 255, 255)
                )
                self.app.processEvents()

            self.bvh_exporter = BVHExporter()
            self.blender_bridge = BlenderBridge()

            self.logger.info("✅ Все компоненты инициализированы")
            return True

        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации компонентов: {e}")
            traceback.print_exc()
            return False

    def create_main_window(self):
        """Создает главное окно приложения"""
        try:
            if self.splash:
                self.splash.showMessage(
                    "Создание интерфейса...",
                    Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                    QColor(255, 255, 255)
                )
                self.app.processEvents()

            # Создаем главное окно
            self.main_window = MainWindow()

            # Восстанавливаем геометрию окна
            if self.qt_settings.contains("MainWindow/Geometry"):
                geometry = self.qt_settings.value("MainWindow/Geometry")
                self.main_window.restoreGeometry(geometry)

            if self.qt_settings.contains("MainWindow/WindowState"):
                window_state = self.qt_settings.value("MainWindow/WindowState")
                self.main_window.restoreState(window_state)

            # Подключаем сигналы
            self.main_window.aboutToClose.connect(self.on_main_window_closing)

            # ДОБАВЬТЕ ЭТИ СТРОКИ ДЛЯ ОТЛАДКИ:
            self.logger.info("✅ Главное окно создано")
            self.logger.info(f"✅ Окно видимо: {self.main_window.isVisible()}")

            # ЗАКРЫВАЕМ SPLASH И ПОКАЗЫВАЕМ ОКНО
            if self.splash:
                self.splash.finish(self.main_window)  # ← ОБЯЗАТЕЛЬНО!
                self.splash = None
                self.logger.info("✅ Splash screen закрыт")

            self.main_window.show()
            self.main_window.raise_()
            self.main_window.activateWindow()
            self.logger.info("✅ Главное окно показано")

            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания главного окна: {e}")
            import traceback
            traceback.print_exc()
            return False
    def on_main_window_closing(self):
        """Обработчик закрытия главного окна"""
        self.logger.info("Закрытие главного окна...")
        self.cleanup()

    def cleanup(self):
        """Очистка ресурсов при выходе"""
        try:
            self.logger.info("Очистка ресурсов...")

            # Сохраняем состояние окна
            if self.main_window:
                self.qt_settings.setValue("MainWindow/Geometry", self.main_window.saveGeometry())
                self.qt_settings.setValue("MainWindow/WindowState", self.main_window.saveState())

            # Сохраняем настройки
            if self.settings:
                settings_file = self.config_dir / "default_settings.yaml"
                save_settings(settings_file, self.settings)

            # Останавливаем компоненты
            if self.animation_recorder:
                self.animation_recorder.stop_recording()

            if self.skeleton_tracker:
                self.skeleton_tracker.stop_tracking()

            if self.camera_manager:
                self.camera_manager.release_all_cameras()

            self.logger.info("✅ Ресурсы очищены")
            self.logger.info("=" * 60)
            self.logger.info("Mocap Pro Application Stopped")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"❌ Ошибка при очистке ресурсов: {e}")

    def check_dependencies(self):
        """Проверяет наличие и версии зависимостей"""
        dependencies = {
            "opencv-python": ("cv2", (4, 8, 0)),
            "numpy": ("numpy", (1, 24, 0)),
            "PyQt6": ("PyQt6.QtCore", (6, 6, 0)),
            "mediapipe": ("mediapipe", (0, 10, 0)),
            "scipy": ("scipy", (1, 11, 0)),
        }

        missing_deps = []
        outdated_deps = []

        for package_name, (module_name, min_version) in dependencies.items():
            try:
                module = __import__(module_name.split('.')[0])

                # Получаем версию
                version_str = ""
                if hasattr(module, '__version__'):
                    version_str = module.__version__
                elif module_name == "cv2":
                    version_str = cv2.__version__

                if version_str:
                    version_tuple = tuple(map(int, version_str.split('.')[:3]))

                    if version_tuple < min_version:
                        outdated_deps.append(f"{package_name} (требуется {min_version}, установлено {version_tuple})")

            except ImportError:
                missing_deps.append(package_name)

        if missing_deps or outdated_deps:
            message = "Обнаружены проблемы с зависимостями:\n\n"

            if missing_deps:
                message += f"Отсутствуют пакеты:\n" + "\n".join(f"  • {dep}" for dep in missing_deps) + "\n\n"

            if outdated_deps:
                message += f"Устаревшие версии:\n" + "\n".join(f"  • {dep}" for dep in outdated_deps) + "\n\n"

            message += "Установите зависимости:\n"
            message += "pip install -r requirements.txt --upgrade"

            print(message)

            if missing_deps:
                return False

        return True

    def run(self):
        """Запускает приложение"""
        try:
            # Создаем QApplication
            self.app = QApplication(sys.argv)
            self.app.setApplicationName("Mocap Pro")
            self.app.setApplicationVersion("1.0.0")
            self.app.setOrganizationName("Mocap Pro Team")

            # Устанавливаем иконку приложения
            icon_path = self.app_dir / "icons/app_icon.ico"
            if icon_path.exists():
                self.app.setWindowIcon(QIcon(str(icon_path)))

            # ПРОВЕРКА ЗАВИСИМОСТЕЙ
            try:
                import cv2, mediapipe, numpy, PyQt6
                print("✅ Зависимости проверены")
            except ImportError as e:
                print(f"❌ Отсутствует зависимость: {e}")
                return 1

            # НАСТРОЙКА ЛОГИРОВАНИЯ
            import logging
            logging.basicConfig(level=logging.INFO,
                                format='%(name)s:%(message)s')
            self.logger = logging.getLogger("MocapPro")
            self.logger.info("Запуск Mocap Pro v1.0.0")

            # ЗАГРУЖАЕМ НАСТРОЙКИ (ВАЖНО!)
            print("⚙️ Загружаю настройки...")
            self.settings = {}  # ← СОЗДАЕМ НАСТРОЙКИ ПО УМОЛЧАНИЮ
            self.settings['tracking'] = {'mode': 'precise'}
            self.settings['recording'] = {'default_fps': 30}
            print("✅ Настройки загружены (по умолчанию)")

            # ПРОПУСКАЕМ SPLASH SCREEN
            print("⚠️ Splash screen отключен для отладки")

            # СОЗДАЕМ ОКНО (С ПРОСТЫМ ИНТЕРФЕЙСОМ)
            print("🚀 Создаю главное окно...")
            from ui.main_window import ProfessionalMainWindow as MainWindow
            self.main_window = MainWindow()

            # ПОКАЗЫВАЕМ ОКНО СРАЗУ
            self.main_window.show()
            self.main_window.raise_()
            self.main_window.activateWindow()
            print("✅ Главное окно показано")

            # ПРОСТАЯ ИНИЦИАЛИЗАЦИЯ БЕЗ ОШИБОК
            print("🤖 Инициализирую камеру...")
            try:
                from core.camera_manager import CameraManager
                self.camera_manager = CameraManager()
                print(f"✅ Камера найдена")
            except Exception as e:
                print(f"⚠️ Камера не найдена: {e}")
                self.camera_manager = None

            # Запускаем event loop
            print("🔄 Запускаю главный цикл...")
            print("\n" + "=" * 50)
            print("🎉 MOCAP PRO УСПЕШНО ЗАПУЩЕН!")
            print("=" * 50 + "\n")
            exit_code = self.app.exec()

            print(f"📤 Приложение завершено с кодом: {exit_code}")
            return exit_code

        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()

            QMessageBox.critical(None, "Критическая ошибка", str(e))
            return 1
def run_cli():
    """Запуск в режиме командной строки"""
    import argparse

    parser = argparse.ArgumentParser(description="Mocap Pro - Professional Motion Capture System")

    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Mocap Pro 1.0.0"
    )

    parser.add_argument(
        "--settings", "-s",
        help="Путь к файлу настроек"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Уровень детализации логов"
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Запуск без графического интерфейса"
    )

    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Запуск калибровки камеры"
    )

    parser.add_argument(
        "--record",
        help="Начать запись в указанный файл"
    )

    parser.add_argument(
        "--export",
        help="Экспортировать анимацию в указанный файл"
    )

    parser.add_argument(
        "--preset",
        help="Загрузить пресет скелета"
    )

    args = parser.parse_args()

    if args.no_gui:
        print("Режим командной строки пока не реализован")
        return 0

    # Запускаем GUI приложение
    app = MocapProApplication()
    return app.run()


if __name__ == "__main__":
    # Точка входа в приложение
    exit_code = run_cli()
    sys.exit(exit_code)