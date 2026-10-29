
from pathlib import Path

import numpy as np
from torchreid.reid.utils import FeatureExtractor


PROJECT_DIR = Path(__file__).resolve().parent.parent
CROPS_DIR = PROJECT_DIR / "outputs" / "person_crops"
REID_MODEL_PATH = PROJECT_DIR / "models" / "osnet_reid.pth"


def normalize(vector):
    """Нормализует вектор признаков."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def extract_track_embeddings(extractor):
    """Получает один вектор признаков для каждого трека."""
    track_embeddings = {}

    for video_name in ("video1", "video2"):
        video_dir = CROPS_DIR / video_name

        if not video_dir.exists():
            print(f"Папка не найдена: {video_dir}")
            continue

        track_dirs = sorted(
            path for path in video_dir.iterdir() if path.is_dir()
        )

        for track_dir in track_dirs:
            image_paths = sorted(track_dir.glob("*.jpg"))

            if not image_paths:
                continue

            print(
                f"  {video_name} / {track_dir.name}: "
                f"{len(image_paths)} изображений"
            )

            features = extractor([str(path) for path in image_paths])
            features = np.asarray(features, dtype=np.float32)

            # Нормализуем признаки каждого кадра.
            features = np.stack(
                [normalize(feature) for feature in features]
            )

            # Усредняем признаки кадров одного трека.
            mean_embedding = features.mean(axis=0)
            mean_embedding = normalize(mean_embedding)

            track_embeddings[(video_name, track_dir.name)] = mean_embedding

    return track_embeddings


def compare_videos(track_embeddings):
    """Сравнивает все треки первого видео со вторым."""
    video1_tracks = {
        key: value
        for key, value in track_embeddings.items()
        if key[0] == "video1"
    }

    video2_tracks = {
        key: value
        for key, value in track_embeddings.items()
        if key[0] == "video2"
    }

    if not video1_tracks or not video2_tracks:
        print("Не удалось найти треки для сравнения.")
        return None

    comparisons = []

    for key1, embedding1 in video1_tracks.items():
        for key2, embedding2 in video2_tracks.items():
            # Скалярное произведение нормализованных векторов
            # равно cosine similarity.
            similarity = float(np.dot(embedding1, embedding2))
            comparisons.append((similarity, key1, key2))

    comparisons.sort(key=lambda item: item[0], reverse=True)

    print("\nСходство между треками:")
    for similarity, key1, key2 in comparisons:
        print(
            f"  {key1[1]} ↔ {key2[1]}: "
            f"{similarity:.4f}"
        )

    best_similarity, best_track1, best_track2 = comparisons[0]

    print("\nНаиболее похожая пара:")
    print(f"  Первое видео: {best_track1[1]}")
    print(f"  Второе видео: {best_track2[1]}")
    print(f"  Cosine similarity: {best_similarity:.4f}")

    return best_similarity, best_track1, best_track2


def run_experiment(title, model_path):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    extractor = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path=model_path,
        device="cpu",
    )

    embeddings = extract_track_embeddings(extractor)
    return compare_videos(embeddings)


def main():
    if not REID_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Не найдены Re-ID-веса: {REID_MODEL_PATH}"
        )

    print("ЭКСПЕРИМЕНТ 1: базовые ImageNet-веса")
    imagenet_result = run_experiment(
        "OSNet с ImageNet-весами",
        model_path="",
    )

    print("\nЭКСПЕРИМЕНТ 2: специализированные Re-ID-веса")
    reid_result = run_experiment(
        "OSNet с Re-ID-весами",
        model_path=str(REID_MODEL_PATH),
    )

    print(f"\n{'=' * 60}")
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    if imagenet_result and reid_result:
        imagenet_score = imagenet_result[0]
        reid_score = reid_result[0]

        print(f"ImageNet similarity: {imagenet_score:.4f}")
        print(f"Re-ID similarity:    {reid_score:.4f}")
        print(f"Разница:             {reid_score - imagenet_score:+.4f}")

        if reid_score > imagenet_score:
            print("В этом эксперименте Re-ID-веса дали большее сходство.")
        elif reid_score < imagenet_score:
            print("В этом эксперименте ImageNet-веса дали большее сходство.")
        else:
            print("Обе модели дали одинаковое сходство.")

        print(
            "\nВажно: большее сходство в одной паре не доказывает, "
            "что модель в целом лучше распознаёт людей."
        )


if __name__ == "__main__":
    main()

