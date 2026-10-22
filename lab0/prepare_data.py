# prepare_data.py — подготовка данных для лабораторной работы №0.
#
# Назначение: зафиксировать разбиение датасета, чтобы все эксперименты
# использовали одни и те же обучающие, валидационные и тестовые примеры.
#
# Скрипт загружает весь load_digits из scikit-learn: 1797 изображений,
# по 64 признака, с метками цифр 0–9. Создаёт разбиение с split_seed=0
# и проверяет размеры частей, индексы и наличие всех классов.
#
# Результат — два файла:
# 1. data/split.json — индексы изображений:
#    train — 1077, val — 360, test — 360.
# 2. data/manifest.json — паспорт данных: источник, версия библиотеки,
#    правила подготовки, размеры частей и SHA-256 файла split.json.
#
# Сами изображения не сохраняются: при обучении они загружаются
# из scikit-learn и выбираются по сохранённым индексам.
# Нормировку признаков (деление на 16), проверку хеша перед обучением
# и само обучение реализуем в harness.py.


import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# Пути считаются относительно расположения скрипта.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLIT_PATH = DATA_DIR / "split.json"
MANIFEST_PATH = DATA_DIR / "manifest.json"

SPLIT_SEED = 0
SPLIT_SIZES = {"train": 1077, "val": 360, "test": 360}
CLASSES = set(range(10))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, content):
    # UTF-8 и одинаковые переносы строк для повторяемого хеша.
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
        file.write("\n")


def validate_split(split, y):
    # Эту функцию позже можно использовать в harness.py и verify.py.
    if not isinstance(split, dict) or set(split) != set(SPLIT_SIZES):
        raise ValueError("Разбиение должно содержать train, val и test")

    all_indices = []

    for name, expected_size in SPLIT_SIZES.items():
        indices = split[name]

        if not isinstance(indices, list) or len(indices) != expected_size:
            raise ValueError(
                f"{name}: ожидается список из {expected_size} индексов"
            )

        if any(type(index) is not int for index in indices):
            raise ValueError(f"{name}: индексы должны быть целыми числами")

        all_indices.extend(indices)

    # Вместе эти условия проверяют:
    # покрытие всех записей, границы, повторы и пересечения.
    if (
        len(all_indices) != len(y)
        or set(all_indices) != set(range(len(y)))
    ):
        raise ValueError(
            "Каждый индекс датасета должен встречаться ровно один раз"
        )

    # Проверяем классы после проверки допустимости индексов.
    for name, indices in split.items():
        if set(y[indices].tolist()) != CLASSES:
            raise ValueError(f"{name}: представлены не все классы 0–9")


def main():
    # Загружаем весь встроенный датасет без отбора записей.
    X, y = load_digits(return_X_y=True)

    if X.shape != (1797, 64) or y.shape != (1797,):
        raise ValueError("Ожидаются 1797 изображений по 64 признака")

    if not np.isfinite(X).all() or X.min() < 0 or X.max() > 16:
        raise ValueError("Признаки должны быть конечными числами от 0 до 16")

    if (
        not np.issubdtype(y.dtype, np.integer)
        or set(y.tolist()) != CLASSES
    ):
        raise ValueError("Метки должны быть целыми числами от 0 до 9")

    # Разбиение строго по приложению методички.
    indices = np.arange(len(y))

    rest, test = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=SPLIT_SEED
    )
    train, val = train_test_split(
        rest, test_size=0.25, stratify=y[rest], random_state=SPLIT_SEED
    )

    split = {
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
    }
    validate_split(split, y)

    # Сохраняем записи других датасетов в общем манифесте.
    manifest = (
        read_json(MANIFEST_PATH)
        if MANIFEST_PATH.exists()
        else {"datasets": {}}
    )

    if not isinstance(manifest, dict):
        raise ValueError("Манифест должен быть JSON-объектом")

    datasets = manifest.setdefault("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("Поле datasets должно быть словарём")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Существующее разбиение не заменяем другим автоматически.
    if SPLIT_PATH.exists():
        if read_json(SPLIT_PATH) != split:
            raise ValueError(
                "Существующий split.json отличается. Файл не перезаписан."
            )
    else:
        save_json(SPLIT_PATH, split)

    # Хешируем именно сохранённый файл, а не объект в памяти.
    split_hash = hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()

    # Повторный запуск не должен скрывать изменение зарегистрированных данных.
    previous = datasets.get("digits")
    if previous is not None:
        if not isinstance(previous, dict):
            raise ValueError("Запись digits должна быть словарём")
        if previous.get("split_sha256") != split_hash:
            raise ValueError("Хеш split.json отличается от манифеста")
        if previous.get("source_version") != sklearn.__version__:
            raise ValueError("Версия scikit-learn отличается от манифеста")

    datasets["digits"] = {
        "source": "sklearn.datasets.load_digits",
        "source_url": (
            "https://scikit-learn.org/stable/modules/"
            "generated/sklearn.datasets.load_digits.html"
        ),
        "source_version": sklearn.__version__,
        "original_records": len(y),
        "selected_records": len(y),
        "excluded_records": 0,
        "exclusion_reason": None,
        "selection_rule": "Используются все записи исходного набора",
        "n_features": X.shape[1],
        "classes": sorted(CLASSES),
        "split_seed": SPLIT_SEED,
        "split_rule": {
            "stratified": True,
            "test_size": 0.2,
            "val_size_of_remainder": 0.25,
            "random_state_both_calls": SPLIT_SEED,
        },
        "split_sizes": {
            name: len(indices) for name, indices in split.items()
        },
        "split_file": "data/split.json",
        "split_sha256": split_hash,
        "preprocessing": "Деление признаков на 16 при обучении",
    }

    save_json(MANIFEST_PATH, manifest)

    print("Данные подготовлены.")
    for name, indices in split.items():
        print(f"{name}: {len(indices)} изображений")
    print(f"SHA-256 split.json: {split_hash}")
    print("Сохранены data/split.json и data/manifest.json")


# Импорт функции validate_split не запускает подготовку повторно.
if __name__ == "__main__":
    main()