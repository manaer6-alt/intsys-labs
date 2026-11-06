# Проверка ответов ЛР2: разбираем JSON и проверяем поля и ссылки.
# Схему RESPONSE_FORMAT позже передадим серверу через response_format.
# parse_answer возвращает разобранный ответ, статус и причину ошибки.
# Правильность метки и обоснования здесь не оцениваем: эталон не читаем.
# Прямой запуск проверяет искусственные примеры без вызова Qwen.

import json
from pathlib import Path


SCHEMA_PATH = Path(__file__).with_name("answer_schema.json")

RESPONSE_FORMAT = json.loads(
    SCHEMA_PATH.read_text(encoding="utf-8-sig")
)

LABELS = (
    RESPONSE_FORMAT["json_schema"]["schema"]
    ["properties"]["label"]["enum"]
)


def validate_answer(answer, mode, passed):
    if (
        not isinstance(answer, dict)
        or set(answer) != {"label", "doc_id", "sentences"}
    ):
        raise ValueError(
            "Нужен объект ровно с полями label, doc_id, sentences"
        )

    label = answer["label"]
    doc_id = answer["doc_id"]
    sentences = answer["sentences"]

    if not isinstance(label, str) or label not in LABELS:
        raise ValueError("Неизвестная метка")

    if doc_id is not None and not isinstance(doc_id, str):
        raise ValueError("doc_id должен быть строкой или null")

    if (
        not isinstance(sentences, list)
        or any(type(i) is not int or i < 0 for i in sentences)
    ):
        raise ValueError(
            "sentences должен быть списком "
            "неотрицательных целых чисел"
        )

    # В LLM-only документов нет.
    # При NOINFO свидетельство не указывается.
    if mode == "llm_only" or label == "NOINFO":
        if doc_id is not None or sentences:
            raise ValueError(
                "Для LLM-only и NOINFO нужны "
                "doc_id=null и sentences=[]"
            )
    else:
        # Разрешены только документы и предложения из готового промпта.
        available = {
            item["doc_id"]: set(item["sentences"])
            for item in passed
        }

        if doc_id not in available:
            raise ValueError("Ссылка на непереданный документ")

        if (
            not sentences
            or not set(sentences) <= available[doc_id]
        ):
            raise ValueError(
                "Нужна непустая ссылка только "
                "на переданные предложения"
            )

    return answer


def parse_answer(raw, mode, passed=()):
    if mode not in ("rag", "oracle", "llm_only"):
        raise ValueError("Неизвестный режим проверки")

    try:
        answer = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as error:
        return {
            "parsed": None,
            "parse_status": "invalid_json",
            "parse_error": str(error),
        }

    try:
        validate_answer(answer, mode, passed)
    except ValueError as error:
        # Сохраняем объект даже при ошибке его полей или ссылок.
        return {
            "parsed": answer,
            "parse_status": "invalid_answer",
            "parse_error": str(error),
        }

    return {
        "parsed": answer,
        "parse_status": "ok",
        "parse_error": None,
    }


def main():
    # Искусственные примеры: это не ответы языковой модели.
    passed = [{"doc_id": "doc-1", "sentences": [0, 2]}]

    valid = {
        "label": "SUPPORT",
        "doc_id": "doc-1",
        "sentences": [2],
    }

    cases = [
        ("Корректный RAG",
         valid, "rag", "ok"),

        ("Корректный oracle",
         valid, "oracle", "ok"),

        ("Корректный NOINFO",
         {"label": "NOINFO", "doc_id": None, "sentences": []},
         "rag", "ok"),

        ("Корректный LLM-only",
         {**valid, "doc_id": None, "sentences": []},
         "llm_only", "ok"),

        ("Неизвестная метка",
         {**valid, "label": "TRUE"},
         "rag", "invalid_answer"),

        ("Лишнее поле",
         {**valid, "confidence": 0.9},
         "rag", "invalid_answer"),

        ("Непереданный документ",
         {**valid, "doc_id": "doc-2"},
         "rag", "invalid_answer"),

        ("Непереданное предложение",
         {**valid, "sentences": [1]},
         "rag", "invalid_answer"),

        ("Отрицательный номер",
         {**valid, "sentences": [-1]},
         "rag", "invalid_answer"),

        ("Булево вместо номера",
         {**valid, "sentences": [True]},
         "rag", "invalid_answer"),

        ("Пустая ссылка SUPPORT",
         {**valid, "sentences": []},
         "rag", "invalid_answer"),

        ("Ссылка при NOINFO",
         {**valid, "label": "NOINFO"},
         "rag", "invalid_answer"),
    ]

    for name, answer, mode, expected in cases:
        result = parse_answer(
            json.dumps(answer), mode, passed
        )

        if result["parse_status"] != expected:
            raise AssertionError(f"{name}: {result}")

        print("OK:", name)

    # Незаконченный ответ должен считаться ошибкой JSON.
    result = parse_answer('{"label":', "rag", passed)

    if result["parse_status"] != "invalid_json":
        raise AssertionError("Повреждённый JSON не отклонён")

    print("OK: Повреждённый JSON")
    print(
        f"Проверок пройдено: {len(cases) + 1}. "
        "Qwen не вызывался."
    )


if __name__ == "__main__":
    main()