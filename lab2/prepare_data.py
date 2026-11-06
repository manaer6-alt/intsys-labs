# Подготовка данных ЛР2: скачиваем и проверяем официальный SciFact.
# Сохраняем исходные файлы без изменений и фиксируем разбиение утверждений:
# dev — 90 для ЛР2, holdout — 210, отложенных до ЛР3.
# Источники, контрольные суммы и списки ID добавляем в data/manifest.json.
# Здесь нет обучения, поиска или вызовов Qwen. Нужен только стандартный Python.

import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "scifact"
MANIFEST = ROOT / "data" / "manifest.json"

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
ARCHIVE_SHA256 = (
    "11c621288d41ac144d29b13b0f8503b3820"
    "b7d6e8b1f6ff24dff335c196d76be"
)

LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")
FILES = ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    # Используем UTF-8 и LF, чтобы байты не зависели от Windows/Linux.
    return (
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def save_fixed(path, data):
    # Совпадающий файл оставляем; отличающийся автоматически не заменяем.
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Файл отличается и не перезаписан: {path}")
    else:
        path.write_bytes(data)


def claim_label(claim):
    if not claim["evidence"]:
        return "NOINFO"

    labels = {
        r["label"]
        for rationales in claim["evidence"].values()
        for r in rationales
    }

    if not labels or not labels <= {"SUPPORT", "CONTRADICT"}:
        raise ValueError(
            f"Некорректная разметка утверждения {claim['id']}"
        )

    # Конфликтующие метки — причина исключения утверждения.
    return next(iter(labels)) if len(labels) == 1 else None


def main():
    # Читаем общий манифест, чтобы сохранить сведения предыдущих работ.
    manifest = (
        json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
        if MANIFEST.exists() else {"datasets": {}}
    )

    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("datasets"), dict)
    ):
        raise ValueError("Манифест должен содержать объект datasets")

    previous = manifest["datasets"].get("scifact", {})
    if not isinstance(previous, dict):
        raise ValueError("Запись scifact должна быть объектом")

    DATA.mkdir(parents=True, exist_ok=True)
    archive_path = DATA / "data.tar.gz"

    # При повторном запуске используем уже скачанный архив.
    if archive_path.exists():
        archive_bytes = archive_path.read_bytes()
    else:
        print("Скачиваем официальный архив SciFact...")
        with urlopen(URL, timeout=60) as response:
            archive_bytes = response.read()

    if sha256(archive_bytes) != ARCHIVE_SHA256:
        raise ValueError("SHA-256 архива не совпал с методичкой")

    save_fixed(archive_path, archive_bytes)

    # Извлекаем только нужные файлы, сохраняя их исходные байты.
    raw, rows = {}, {}

    with tarfile.open(archive_path, "r:gz") as archive:
        for name in FILES:
            members = [
                m for m in archive.getmembers()
                if m.isfile() and Path(m.name).name == name
            ]

            if len(members) != 1:
                raise ValueError(f"Ожидается ровно один файл {name}")

            with archive.extractfile(members[0]) as stream:
                raw[name] = stream.read()

            rows[name] = [
                json.loads(line)
                for line in raw[name].decode("utf-8").splitlines()
                if line.strip()
            ]

            key = "doc_id" if name == "corpus.jsonl" else "id"
            ids = [r[key] for r in rows[name]]

            if (
                not ids
                or any(type(i) is not int for i in ids)
                or len(ids) != len(set(ids))
            ):
                raise ValueError(
                    f"Пустые данные или некорректные ID: {name}"
                )

    corpus = {str(r["doc_id"]): r for r in rows["corpus.jsonl"]}
    claims = rows["claims_dev.jsonl"]

    if len(corpus) != 5183 or len(claims) != 300:
        raise ValueError(
            "Ожидаются 5183 аннотации и 300 утверждений claims_dev"
        )

    # Эталон здесь нужен для разбиения и проверки данных.
    # В дальнейшем поиск и модель не должны получать эту разметку.
    labels, excluded = {}, []

    for claim in claims:
        label = claim_label(claim)

        if label is None:
            excluded.append({
                "id": claim["id"],
                "reason": "SUPPORT и CONTRADICT одновременно",
            })
        else:
            labels[claim["id"]] = label

        # Проверяем существование документов и номеров предложений.
        for doc_id, rationales in claim["evidence"].items():
            if doc_id not in corpus:
                raise ValueError(
                    f"Неизвестный документ в evidence: {doc_id}"
                )

            for rationale in rationales:
                if not rationale["sentences"] or any(
                    type(i) is not int
                    or not 0 <= i < len(corpus[doc_id]["abstract"])
                    for i in rationale["sentences"]
                ):
                    raise ValueError(
                        f"Некорректные номера предложений: {claim['id']}"
                    )

    # Правило методички: сортировка по хешу, первые 30 каждого класса в dev.
    split = {"dev": [], "holdout": []}

    for label in LABELS:
        ids = sorted(
            (i for i in labels if labels[i] == label),
            key=lambda i: sha256(f"scifact:{i}".encode("utf-8")),
        )

        if len(ids) < 30:
            raise ValueError(
                f"Недостаточно утверждений класса {label}"
            )

        split["dev"].extend(ids[:30])
        split["holdout"].extend(ids[30:])

    combined = split["dev"] + split["holdout"]

    if (
        len(combined) != len(set(combined))
        or set(combined) != set(labels)
    ):
        raise ValueError(
            "Разбиение содержит повторы, пересечения или пропуски"
        )

    class_counts = {
        name: dict(Counter(labels[i] for i in ids))
        for name, ids in split.items()
    }

    split_path = DATA / "split.json"
    split_bytes = json_bytes(split)

    entry = {
        "source": "allenai/scifact, official release/latest",
        "source_url": URL,
        "source_version": ARCHIVE_SHA256,
        "archive_file": "data/scifact/data.tar.gz",
        "archive_sha256": sha256(archive_path.read_bytes()),
        "files": {
            name: {
                "path": f"data/scifact/{name}",
                "records": len(rows[name]),
                "sha256": sha256(raw[name]),
            }
            for name in FILES
        },
        "original_records": len(claims),
        "selected_records": len(labels),
        "excluded_records": len(excluded),
        "exclusions": excluded,
        "selection_rule": (
            "Все claims_dev, кроме утверждений с конфликтующими "
            "метками; весь корпус"
        ),
        "split_rule": (
            "В каждом классе сортировка по SHA-256 UTF-8 строки "
            "scifact:<id>; первые 30 в dev"
        ),
        "splits": split,
        "split_file": "data/scifact/split.json",
        "split_sha256": sha256(split_bytes),
        "split_sizes": {
            name: len(ids) for name, ids in split.items()
        },
        "class_counts": class_counts,
        "usage": {
            "claims_train": "Только проверки и прогрев",
            "dev": "Рабочая часть ЛР2–ЛР3",
            "holdout": "Отложено до ЛР3",
        },
    }

    # Повторный запуск не должен скрывать смену зарегистрированных данных.
    for key in ("archive_sha256", "split_sha256"):
        if key in previous and previous[key] != entry[key]:
            raise ValueError(
                f"Зарегистрированное значение {key} изменилось"
            )

    for name in FILES:
        save_fixed(DATA / name, raw[name])

    save_fixed(split_path, split_bytes)

    # В манифест записываем хеши действительно сохранённых файлов.
    for name in FILES:
        entry["files"][name]["sha256"] = sha256(
            (DATA / name).read_bytes()
        )

    entry["split_sha256"] = sha256(split_path.read_bytes())
    manifest["datasets"]["scifact"] = {**previous, **entry}

    # Сначала записываем временный файл, затем заменяем манифест.
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_bytes(json_bytes(manifest))
    temporary.replace(MANIFEST)

    print("Готово. Аннотаций:", len(corpus))
    print(
        "Утверждений для проверок и прогрева:",
        len(rows["claims_train.jsonl"]),
    )
    print("Исходные классы:", dict(Counter(labels.values())))
    print("Исключено утверждений:", len(excluded))

    for name in split:
        print(
            f"{name}: {len(split[name])}, "
            f"классы: {class_counts[name]}"
        )

    print("SHA-256 split.json:", entry["split_sha256"])
    print("Манифест обновлён:", MANIFEST)


if __name__ == "__main__":
    main()