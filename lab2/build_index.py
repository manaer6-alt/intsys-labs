# Векторный индекс ЛР2: кодируем все 5183 аннотации готовой моделью BGE.
# Результаты: embeddings.npy, doc_ids.json и passport.json.
# Паспорт с параметрами, версиями и хешами добавляем в общий манифест.
# Разметку утверждений не читаем. Обучения и обращений к Qwen здесь нет.
# --check выполняет короткую проверку без сохранения индекса.

import argparse
import json
import platform
import random
from importlib.metadata import version

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from prepare_data import ROOT, MANIFEST, sha256, json_bytes, save_fixed


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_vectors(vectors, count, dimension):
    if vectors.shape != (count, dimension) or not np.isfinite(vectors).all():
        raise ValueError("Некорректная форма матрицы или нечисловые значения")

    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("Векторы должны иметь единичную длину")


def register(manifest, passport):
    dataset = manifest["datasets"]["scifact"]

    if dataset.get("index") is not None and dataset["index"] != passport:
        raise ValueError("В манифесте уже зарегистрирован другой индекс")

    dataset["index"] = passport

    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_bytes(json_bytes(manifest))
    temporary.replace(MANIFEST)


def main():
    parser = argparse.ArgumentParser(description="Построение индекса SciFact")
    parser.add_argument("--config", default="configs/lab2_index.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config = read_json(ROOT / args.config)

    for key in ("dimension", "max_length", "batch_size", "num_threads"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} должен быть положительным целым числом")

    if config["normalize_embeddings"] is not True:
        raise ValueError("В этой работе используем нормализованные векторы")

    if len(config["revision"]) != 40:
        raise ValueError("Нужна полная ревизия кодировщика")

    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("Некорректный seed")

    manifest = read_json(MANIFEST)
    source = manifest["datasets"]["scifact"]["files"]["corpus.jsonl"]
    corpus_bytes = (ROOT / source["path"]).read_bytes()

    if sha256(corpus_bytes) != source["sha256"]:
        raise ValueError("Хеш корпуса отличается от манифеста")

    versions = {
        name: version(name)
        for name in (
            "numpy", "torch", "sentence-transformers",
            "transformers", "huggingface-hub",
        )
    }
    versions["python"] = platform.python_version()

    out = ROOT / config["output_dir"]
    passport_path = out / "passport.json"

    # Готовый индекс повторно не вычисляем: проверяем параметры и файлы.
    if passport_path.exists() and not args.check:
        passport = read_json(passport_path)

        if (
            passport["config"] != config
            or passport["versions"] != versions
            or passport["corpus_sha256"] != source["sha256"]
        ):
            raise ValueError(
                "Параметры, версии или корпус отличаются от индекса"
            )

        for item in passport["files"].values():
            if sha256((ROOT / item["path"]).read_bytes()) != item["sha256"]:
                raise ValueError(
                    f"Повреждён файл индекса: {item['path']}"
                )

        register(manifest, passport)
        print("Готовый индекс проверен:", out)
        return

    if not args.check and out.exists() and any(out.iterdir()):
        raise ValueError("Папка индекса непустая, но паспорт отсутствует")

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["num_threads"])
    torch.use_deterministic_algorithms(True)

    # Загружаем готовый кодировщик на CPU.
    model = SentenceTransformer(
        config["model"],
        revision=config["revision"],
        device="cpu",
    )
    model.float().eval()

    if config["max_length"] > model.tokenizer.model_max_length:
        raise ValueError(
            "Предел длины превышает возможности кодировщика"
        )

    model.max_seq_length = config["max_length"]

    def encode(texts, progress=False):
        return model.encode(
            texts,
            batch_size=config["batch_size"],
            prompt="",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        ).astype(np.float32, copy=False)

    # Сначала проверяем работу на нескольких коротких текстах.
    probe = "Scientific evidence can support or contradict a claim."
    test = encode([
        config["document_prefix"] + probe,
        config["document_prefix"] + probe,
        config["query_prefix"] + probe,
    ])

    check_vectors(test, 3, config["dimension"])

    same_text = float(test[0] @ test[1])
    if not np.isclose(same_text, 1, atol=1e-5):
        raise ValueError("Самопроверка одинаковых текстов не пройдена")

    query_document = float(test[0] @ test[2])

    print(f"Одинаковый текст без изменения префикса: {same_text:.6f}")
    print(f"Тот же текст как запрос и документ: {query_document:.6f}")

    if args.check:
        print("Короткая проверка завершена; индекс не записывался.")
        return

    documents = [
        json.loads(line)
        for line in corpus_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    doc_ids = [str(d["doc_id"]) for d in documents]

    if len(doc_ids) != 5183 or len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Ожидаются 5183 уникальных документа")

    # Формат текста документа задан методичкой.
    texts = [
        config["document_prefix"]
        + d["title"] + "\n" + " ".join(d["abstract"])
        for d in documents
    ]

    # Измеряем длину ДО обрезки, включая специальные токены BGE.
    lengths = model.tokenizer(
        texts,
        truncation=False,
        padding=False,
        return_length=True,
        verbose=False,
    )["length"]

    truncated = sum(
        length > config["max_length"] for length in lengths
    )

    # Каждая строка результата соответствует одному документу.
    vectors = encode(texts, progress=True)
    check_vectors(vectors, len(doc_ids), config["dimension"])

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vectors, allow_pickle=False)
    save_fixed(out / "doc_ids.json", json_bytes(doc_ids))

    passport = {
        "config": config,
        "versions": versions,
        "device": "cpu",
        "dtype": "float32",
        "corpus_sha256": source["sha256"],
        "documents": len(doc_ids),
        "shape": list(vectors.shape),
        "normalization": "L2",
        "document_text": (
            "document_prefix + title + newline + join(abstract, space)"
        ),
        "truncated_documents": truncated,
        "truncated_fraction": truncated / len(doc_ids),
        "token_lengths_include_special_tokens": True,
        "same_text_similarity": same_text,
        "query_document_similarity": query_document,
        "files": {
            name: {
                "path": (out / name).relative_to(ROOT).as_posix(),
                "sha256": sha256((out / name).read_bytes()),
            }
            for name in ("embeddings.npy", "doc_ids.json")
        },
    }

    save_fixed(passport_path, json_bytes(passport))
    register(manifest, passport)

    print("Форма матрицы:", vectors.shape)
    print(f"Размер матрицы: {vectors.nbytes / 1024**2:.2f} МиБ")
    print(
        f"Обрезано документов: {truncated}/{len(doc_ids)} "
        f"({truncated / len(doc_ids):.2%})"
    )
    print(
        "SHA-256 матрицы:",
        passport["files"]["embeddings.npy"]["sha256"],
    )
    print("Индекс сохранён:", out)


if __name__ == "__main__":
    main()