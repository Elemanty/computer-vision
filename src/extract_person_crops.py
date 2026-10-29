
from pathlib import Path
import cv2
from ultralytics import YOLO

# Пути к проекту и модели
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "yolo11n.pt"

VIDEOS = {
    "video1": PROJECT_DIR / "videos" / "input.mp4",
    "video2": PROJECT_DIR / "videos" / "input2.mp4",
}

# Настройки
CONFIDENCE_THRESHOLD = 0.4
SAMPLE_EVERY_N_FRAMES = 10
MAX_CROPS_PER_PERSON = 20
MIN_PERSON_WIDTH = 40
MIN_PERSON_HEIGHT = 80
PADDING_RATIO = 0.08


def extract_crops(video_name, video_path):
    if not video_path.exists():
        print(f"Файл не найден: {video_path}")
        return

    output_dir = PROJECT_DIR / "outputs" / "person_crops" / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Отдельный экземпляр модели для каждого видео:
    # у каждого ролика будет собственная нумерация треков.
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return

    frame_number = 0
    saved_per_track = {}
    total_saved = 0

    print(f"\nОбрабатываем {video_name}: {video_path.name}")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Детекцию выполняем на каждом кадре,
            # а сохраняем кропы только через заданный интервал.
            results = model.track(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                device="cpu",
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )

            result = results[0]

            if (
                frame_number % SAMPLE_EVERY_N_FRAMES == 0
                and result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = result.boxes.xyxy.cpu().tolist()
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()

                frame_height, frame_width = frame.shape[:2]

                for box, track_id, class_id in zip(
                    boxes, track_ids, class_ids
                ):
                    # В стандартной модели COCO класс 0 — человек.
                    if class_id != 0:
                        continue

                    if saved_per_track.get(track_id, 0) >= MAX_CROPS_PER_PERSON:
                        continue

                    x1, y1, x2, y2 = map(int, box)

                    # Добавляем небольшой запас вокруг человека.
                    box_width = x2 - x1
                    box_height = y2 - y1
                    pad_x = int(box_width * PADDING_RATIO)
                    pad_y = int(box_height * PADDING_RATIO)

                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(frame_width, x2 + pad_x)
                    y2 = min(frame_height, y2 + pad_y)

                    if (
                        x2 - x1 < MIN_PERSON_WIDTH
                        or y2 - y1 < MIN_PERSON_HEIGHT
                    ):
                        continue

                    person_crop = frame[y1:y2, x1:x2]

                    if person_crop.size == 0:
                        continue

                    track_dir = output_dir / f"track_{track_id:03d}"
                    track_dir.mkdir(parents=True, exist_ok=True)

                    crop_path = track_dir / f"frame_{frame_number:06d}.jpg"

                    if cv2.imwrite(str(crop_path), person_crop):
                        saved_per_track[track_id] = (
                            saved_per_track.get(track_id, 0) + 1
                        )
                        total_saved += 1

            frame_number += 1

    finally:
        cap.release()

    print(f"Обработано кадров: {frame_number}")
    print(f"Найдено треков с сохранёнными кропами: {len(saved_per_track)}")
    print(f"Сохранено изображений людей: {total_saved}")
    print(f"Папка с результатами: {output_dir}")


def main():
    for video_name, video_path in VIDEOS.items():
        extract_crops(video_name, video_path)


if __name__ == "__main__":
    main()
