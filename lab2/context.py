# Сборка промпта ЛР2: утверждение + документы в пределах бюджета Qwen.
# Считаем весь вход с шаблоном диалога, добавляем целые предложения.
# Сохраняем исходные номера предложений.
# Различаем найденные документы и реально переданные предложения.
# Веса Qwen не загружаем: используем только его токенизатор.
# Прямой запуск — отладочная проверка на claims_train без вызова сервера.

import argparse
import json
import uuid

from transformers import AutoTokenizer

from prepare_data import ROOT, sha256, json_bytes
from build_index import read_json
from retrieval import DenseRetriever


def format_document(document, sentence_ids):
    header = (
        f"Document {document['doc_id']}\n"
        f"Title: {document['title']}"
    )
    sentences = [
        f"[{i}] {document['abstract'][i]}"
        for i in sentence_ids
    ]
    return header + "\n" + "\n".join(sentences)


class PromptBuilder:
    def __init__(self, config_path="configs/lab2.json"):
        self.config = read_json(ROOT / config_path)

        if (
            type(self.config["input_budget"]) is not int
            or self.config["input_budget"] <= 0
        ):
            raise ValueError(
                "input_budget должен быть положительным целым числом"
            )

        path = ROOT / self.config["prompts_file"]
        self.prompts = {
            key: "\n".join(lines)
            for key, lines in read_json(path).items()
        }
        self.prompts_sha256 = sha256(path.read_bytes())

        # Скачиваются файлы токенизатора, без весов языковой модели.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"],
            revision=self.config["revision"],
        )

    def messages(self, claim, context="", mode="rag"):
        if mode not in ("rag", "oracle", "llm_only"):
            raise ValueError("Неизвестный режим")

        # Для oracle используем тот же промпт, что и для RAG.
        template = self.prompts[
            "rag" if mode == "oracle" else mode
        ]

        return [
            {
                "role": "system",
                "content": self.prompts["system"],
            },
            {
                "role": "user",
                "content": template.format(
                    claim=claim, context=context
                ),
            },
        ]

    def count_tokens(self, messages):
        # Учитываем служебные токены ролей и начало ответа assistant.
        return len(self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        ))

    def package(self, claim, context="", mode="rag"):
        messages = self.messages(claim, context, mode)
        count = self.count_tokens(messages)

        if count > self.config["input_budget"]:
            raise ValueError(
                f"Промпт содержит {count} токенов "
                "и превышает бюджет"
            )

        return {
            "mode": mode,
            "messages": messages,
            "prompt": self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ),
            "prompt_tokens": count,
            "prompts_sha256": self.prompts_sha256,
        }

    def pack(self, claim, found, documents, k):
        if type(k) is not int or not 1 <= k <= len(found):
            raise ValueError(
                "Недостаточно найденных документов для заданного k"
            )

        # Проверяем, помещаются ли инструкция и само утверждение.
        self.package(claim)

        blocks, passed = [], []
        stopped = False

        for item in found[:k]:
            doc_id = item["doc_id"]
            document = documents[doc_id]
            chosen = []

            for i in range(len(document["abstract"])):
                # Проверяем добавление следующего целого предложения.
                block = format_document(
                    document, chosen + [i]
                )
                candidate = "\n\n".join(blocks + [block])

                count = self.count_tokens(
                    self.messages(claim, candidate)
                )

                if count > self.config["input_budget"]:
                    stopped = True
                    break

                chosen.append(i)

            if chosen:
                blocks.append(
                    format_document(document, chosen)
                )
                passed.append({
                    "doc_id": doc_id,
                    "sentences": chosen,
                })

            # После переполнения дальше по выдаче не идём.
            if stopped:
                break

        return {
            **self.package(claim, "\n\n".join(blocks)),
            "k": k,
            "found": found[:k],
            "passed": passed,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Проверка сборки контекста ЛР2"
    )
    parser.add_argument(
        "--config", default="configs/lab2.json"
    )
    args = parser.parse_args()

    builder = PromptBuilder(args.config)
    retriever = DenseRetriever(
        builder.config["retrieval_config"]
    )

    info = retriever.dataset["files"]["claims_train.jsonl"]
    raw = (ROOT / info["path"]).read_bytes()

    if sha256(raw) != info["sha256"]:
        raise ValueError(
            "Хеш claims_train отличается от манифеста"
        )

    # Из проверочных примеров оставляем только ID и текст.
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
        raise ValueError(
            "Недостаточно утверждений для проверки"
        )

    # Первый пример — прогрев поиска.
    retriever.search(samples[0]["claim"])

    # Второй — поиск пяти документов и сборка контекста для k=3.
    claim = samples[1]
    search = retriever.search(claim["claim"], k=5)

    packed = builder.pack(
        claim["claim"],
        search["found"],
        retriever.documents,
        k=3,
    )

    # Каждая проверка получает новую папку.
    out = (
        ROOT / "runs" / "lab2_context_debug"
        / uuid.uuid4().hex
    )
    out.mkdir(parents=True)

    record = {
        "debug": True,
        "claim_id": claim["id"],
        "claim": claim["claim"],
        "config": builder.config,
        "search_s": search["search_s"],
        **packed,
    }

    (out / "context.json").write_bytes(
        json_bytes(record)
    )
    (out / "prompt.txt").write_bytes(
        packed["prompt"].encode("utf-8")
    )

    print("Утверждение:", claim["claim"])
    print(
        "Найдено для k=3:",
        [item["doc_id"] for item in packed["found"]],
    )
    print(
        "Передано:",
        json.dumps(packed["passed"], ensure_ascii=False),
    )
    print(
        f"Входных токенов: {packed['prompt_tokens']}"
        f"/{builder.config['input_budget']}"
    )
    print("Промпт:", out / "prompt.txt")
    print("Полная запись:", out / "context.json")
    print("Это отладочная проверка. Qwen не вызывался.")


if __name__ == "__main__":
    main()