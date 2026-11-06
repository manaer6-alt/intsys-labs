# Поисковый компонент ЛР2: утверждение -> BGE-вектор -> пять документов.
# Используем готовую матрицу: повторно кодировать 5183 аннотации не нужно.
# Возвращаем ID документов, косинусную близость и время поиска.
# Метку SUPPORT/CONTRADICT/NOINFO позже будет определять Qwen.
# При запуске файла проверяем поиск на claims_train.
# Результаты основных экспериментов этот файл самостоятельно не записывает.

import argparse
import json
import platform
import random
import time
from importlib.metadata import version

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from prepare_data import ROOT, MANIFEST, sha256
from build_index import read_json, check_vectors


def top_documents(matrix, doc_ids, query_vector, k):
    # Для нормализованных векторов скалярное произведение
    # равно косинусной близости.
    scores = matrix @ query_vector

    # Сортируем от большей близости к меньшей.
    # При равных оценках сохраняем исходный порядок строк.
    order = np.argsort(-scores, kind="stable")[:k]

    return [
        {"doc_id": doc_ids[i], "score": float(scores[i])}
        for i in order
    ]


class DenseRetriever:
    def __init__(self, config_path="configs/lab2_index.json"):
        self.config = read_json(ROOT / config_path)
        self.dataset = read_json(MANIFEST)["datasets"]["scifact"]

        self.passport = read_json(
            ROOT / self.config["output_dir"] / "passport.json"
        )

        if self.passport != self.dataset["index"]:
            raise ValueError("Паспорт индекса отличается от манифеста")

        if self.config != self.passport["config"]:
            raise ValueError(
                "Конфигурация отличается от построенного индекса"
            )

        # Запросы кодируем в том же окружении, что и документы.
        for name, expected in self.passport["versions"].items():
            actual = (
                platform.python_version()
                if name == "python" else version(name)
            )
            if actual != expected:
                raise ValueError(
                    f"Версия {name}: ожидается {expected}, "
                    f"получено {actual}"
                )

        # Проверяем хеши матрицы и соответствия строк документам.
        for item in self.passport["files"].values():
            if sha256((ROOT / item["path"]).read_bytes()) != item["sha256"]:
                raise ValueError(
                    f"Повреждён файл индекса: {item['path']}"
                )

        files = self.passport["files"]

        self.matrix = np.load(
            ROOT / files["embeddings.npy"]["path"],
            allow_pickle=False,
        )
        self.doc_ids = read_json(
            ROOT / files["doc_ids.json"]["path"]
        )

        check_vectors(
            self.matrix,
            self.passport["documents"],
            self.config["dimension"],
        )

        if (
            len(self.doc_ids) != len(self.matrix)
            or len(set(self.doc_ids)) != len(self.doc_ids)
        ):
            raise ValueError(
                "Некорректное соответствие строк и документов"
            )

        # Полные документы понадобятся для заголовков и сборки контекста.
        corpus_info = self.dataset["files"]["corpus.jsonl"]
        corpus_bytes = (ROOT / corpus_info["path"]).read_bytes()

        if (
            sha256(corpus_bytes) != corpus_info["sha256"]
            or corpus_info["sha256"] != self.passport["corpus_sha256"]
        ):
            raise ValueError(
                "Корпус отличается от использованного при индексации"
            )

        documents = [
            json.loads(line)
            for line in corpus_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
        self.documents = {
            str(d["doc_id"]): d for d in documents
        }

        if set(self.doc_ids) != set(self.documents):
            raise ValueError("Документы корпуса не совпадают с индексом")

        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        torch.set_num_threads(self.config["num_threads"])
        torch.use_deterministic_algorithms(True)

        # Загружаем ту же ревизию BGE, которая создала индекс.
        self.model = SentenceTransformer(
            self.config["model"],
            revision=self.config["revision"],
            device="cpu",
        )
        self.model.float().eval()
        self.model.max_seq_length = self.config["max_length"]

    def search(self, claim, k=5):
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Нужен непустой текст утверждения")

        if type(k) is not int or not 1 <= k <= len(self.doc_ids):
            raise ValueError("Некорректное число документов k")

        # Загрузка файлов и модели в измеряемый интервал не входят.
        start = time.perf_counter()

        vector = self.model.encode(
            [self.config["query_prefix"] + claim],
            prompt="",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)

        check_vectors(vector, 1, self.config["dimension"])
        encoded = time.perf_counter()

        found = top_documents(
            self.matrix, self.doc_ids, vector[0], k
        )
        finished = time.perf_counter()

        return {
            "found": found,
            # Включает кодирование и проверку вектора.
            "query_encoding_s": encoded - start,
            "ranking_s": finished - encoded,
            "search_s": finished - start,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Проверка поиска SciFact на claims_train"
    )
    parser.add_argument(
        "--config", default="configs/lab2_index.json"
    )
    args = parser.parse_args()

    retriever = DenseRetriever(args.config)

    info = retriever.dataset["files"]["claims_train.jsonl"]
    raw = (ROOT / info["path"]).read_bytes()

    if sha256(raw) != info["sha256"]:
        raise ValueError("Хеш claims_train отличается от манифеста")

    # Для проверки оставляем только ID и текст первых двух утверждений.
    # Метки, evidence и cited_doc_ids в поиск не передаём.
    samples = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            samples.append({
                "id": row["id"],
                "claim": row["claim"],
            })
            if len(samples) == 2:
                break

    if len(samples) != 2:
        raise ValueError("Недостаточно утверждений для проверки")

    print("Прогрев на первом утверждении claims_train...")
    retriever.search(samples[0]["claim"])

    claim = samples[1]
    result = retriever.search(claim["claim"])

    print(f"\nУтверждение {claim['id']}: {claim['claim']}")

    for rank, item in enumerate(result["found"], 1):
        title = retriever.documents[item["doc_id"]]["title"]
        print(
            f"{rank}. doc_id={item['doc_id']} | "
            f"score={item['score']:.6f} | {title}"
        )

    print(
        f"Кодирование и проверка запроса: "
        f"{result['query_encoding_s'] * 1000:.2f} мс"
    )
    print(
        f"Ранжирование документов: "
        f"{result['ranking_s'] * 1000:.2f} мс"
    )
    print(f"Весь поиск: {result['search_s'] * 1000:.2f} мс")
    print(
        "Проверка завершена. "
        "Основные 90 утверждений не запускались."
    )


if __name__ == "__main__":
    main()