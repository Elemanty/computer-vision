
from pathlib import Path

import cv2
from ultralytics import YOLO
import csv
from datetime import datetime


# Папка проекта: находим её относительно текущего файла
PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_DIR / "videos" / "input.mp4"
OUTPUT_DIR = PROJECT_DIR / "outputs"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_PATH = OUTPUT_DIR / f"detected_{RUN_ID}.mp4"
CSV_PATH = OUTPUT_DIR / f"tracks_{RUN_ID}.csv"

MODEL_PATH = PROJECT_DIR / "yolo11n.pt"

# Порог уверенности: показываем обнаружения с confidence >= 0.4
CONFIDENCE_THRESHOLD = 0.4


def main():
    # Проверяем наличие видео
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Видео не найдено: {VIDEO_PATH}\n"
            "Положи видео в папку videos и назови input.mp4."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Загружаем нейросеть
    model = YOLO(str(MODEL_PATH))

    # Открываем видео
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {VIDEO_PATH}")

    # Получаем характеристики исходного видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0

    # Настраиваем запись обработанного видео
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Не удалось создать выходной видеофайл.")

    frame_number = 0
    track_rows = []
    print("Начинаем обработку видео...")

    try:
        while True:
            # Читаем следующий кадр
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Запускаем YOLO на текущем кадре
            results = model.track(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                device="cpu",
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )

            result = results[0]

            # Рисуем рамки и ID
            annotated_frame = result.plot()

            # Получаем данные об обнаруженных объектах
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().tolist()
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()

                for box, track_id, class_id, confidence in zip(
                    boxes, track_ids, class_ids, confidences
                ):
                    x1, y1, x2, y2 = box

                    track_rows.append({
                        "frame": frame_number,
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": model.names[class_id],
                        "confidence": round(confidence, 4),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    })

            # Сохраняем кадр в видео
            writer.write(annotated_frame)

            if frame_number % 30 == 0:
                print(f"Обработано кадров: {frame_number}")

    finally:
        cap.release()
        writer.release()

    if track_rows:
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as file:
            fieldnames = list(track_rows[0].keys())
            csv_writer = csv.DictWriter(file, fieldnames=fieldnames)

            csv_writer.writeheader()
            csv_writer.writerows(track_rows)

        print(f"Данные треков сохранены: {CSV_PATH}")
    else:
        print("Треки с назначенными ID не обнаружены.")

    

    print(f"Готово! Обработано кадров: {frame_number}")
    print(f"Результат сохранён: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
