
from pathlib import Path
import cv2

# Папка проекта
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Пути к двум видео
video_paths = [
    PROJECT_DIR / "videos" / "input.mp4",
    PROJECT_DIR / "videos" / "input2.mp4",
]

for video_path in video_paths:
    print(f"\nПроверяем: {video_path.name}")

    if not video_path.exists():
        print("ОШИБКА: файл не найден!")
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ОШИБКА: не удалось открыть видео!")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    print(f"Размер кадра: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Количество кадров: {frame_count}")
    print(f"Длительность: {duration:.2f} секунд")

    cap.release()
