"""
Тест только OpenCV - без PyQt
"""

import cv2
import numpy as np

def test_cameras_opencv():
    """Тестирование камер только через OpenCV"""
    print("🎥 Тест камер через OpenCV...")

    for camera_id in range(4):
        print(f"\n🔍 Проверяю камеру {camera_id}...")

        # Пробуем разные backends
        backends = [
            cv2.CAP_DSHOW,
            cv2.CAP_MSMF,
            cv2.CAP_ANY
        ]

        backend_names = {
            cv2.CAP_DSHOW: "DSHOW",
            cv2.CAP_MSMF: "MSMF",
            cv2.CAP_ANY: "ANY"
        }

        for backend in backends:
            cap = cv2.VideoCapture(camera_id, backend)

            if cap.isOpened():
                print(f"  ✅ Открыта с {backend_names.get(backend, backend)}")

                # Пробуем получить кадр
                ret, frame = cap.read()
                if ret:
                    print(f"    📹 Кадр: {frame.shape}")

                    # Показываем кадр
                    cv2.putText(frame, f"Camera {camera_id}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Backend: {backend_names.get(backend, backend)}",
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow(f'Camera {camera_id}', frame)
                    cv2.waitKey(1000)  # Показываем 1 секунду
                    cv2.destroyAllWindows()
                else:
                    print(f"    ❌ Не удалось получить кадр")

                cap.release()
                break  # Переходим к следующей камере
            else:
                cap.release()
                print(f"  ❌ Не открывается с {backend_names.get(backend, backend)}")

    print("\n✅ Тест завершен")

if __name__ == "__main__":
    test_cameras_opencv()