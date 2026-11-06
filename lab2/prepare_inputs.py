# Подготовка входов ЛР2 на ноутбуке: поиск и промпты для будущих вызовов Qwen.
# Полный запуск: 90 LLM-only, по 90 RAG для k=1/3/5 и 60 oracle = 420 заданий.
# Поиск выполняем один раз на утверждение; все k используют начала одной выдачи.
# Разметку для oracle выбирает модуль evaluation; в обычные промпты она не входит.
# --debug использует три утверждения claims_train и допускает незакоммиченный код.
# Полный запуск требует чистого Git и закоммиченного прогноза forecasts/lab2.md.
# Результат — новая папка runs/lab2_prepare/<run_id> и ZIP для переноса в Kaggle.
# Генерацию и подсчёт Accuracy этот скрипт не выполняет.

import argparse
import json
import platform
import random
import shutil
import subprocess
import traceback
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version

import yaml

from prepare_data import ROOT, MANIFEST, sha256, json_bytes
from build_index import read_json
from context import PromptBuilder, format_document
from retrieval import DenseRetriever
from evaluation import load_gold, oracle_plan


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, encoding="utf-8"
    ).strip()


def checked_commit(debug):
    commit = git("rev-parse", "HEAD")
    if not debug:
        status = git("status", "--porcelain")
        if status:
            raise ValueError("Рабочее дерево Git не чистое:\n" + status)
        git("ls-files", "--error-unmatch", "forecasts/lab2.md")
        if not (ROOT / "forecasts/lab2.md").read_text(encoding="utf-8-sig").strip():
            raise ValueError("Файл прогноза пуст")
    return commit


def queries(dataset, source_name):
    # Сразу выделяем только ID и текст. evidence и cited_doc_ids не используем.
    info = dataset["files"][source_name]
    raw = (ROOT / info["path"]).read_bytes()
    if sha256(raw) != info["sha256"]:
        raise ValueError(f"Хеш {source_name} отличается от манифеста")
    result = {}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["id"] in result:
                raise ValueError("Повторяющийся ID утверждения")
            result[row["id"]] = row["claim"]
    if not result:
        raise ValueError("Набор утверждений пуст")
    return result


def append(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def task(claim_id, claim, packed, search=None):
    mode, k = packed["mode"], packed.get("k")
    return {
        "task_id": f"{mode}:{k if k is not None else 0}:{claim_id}",
        "claim_id": claim_id, "claim": claim,
        "found": [], "passed": [], "k": None,
        "search_s": search["search_s"] if search else 0.0,
        "search_source": "prepared_once" if search else "not_used",
        **packed,
    }


def series_name(record):
    return f"rag_k{record['k']}" if record["mode"] == "rag" else record["mode"]


def snapshot(out, config_path, config):
    sources = [
        config_path, config["retrieval_config"], "data/manifest.json",
        config["prompts_file"], "lab2/answer_schema.json", "lab2/requirements-local.txt",
    ]
    if (ROOT / "forecasts/lab2.md").exists():
        sources.append("forecasts/lab2.md")
    sources += [p.relative_to(ROOT).as_posix() for p in (ROOT / "lab2").glob("*.py")]
    hashes = {}
    for relative in sources:
        source = (ROOT / relative).resolve()
        relative = source.relative_to(ROOT.resolve()).as_posix()
        target = out / "provenance" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative] = sha256(target.read_bytes())
    return hashes


def make_archive(out, run_id):
    path = out / f"lab2_inputs_{run_id[:8]}.zip"
    # Список собираем до создания ZIP, чтобы архив не включил сам себя.
    members = [p for p in out.rglob("*") if p.is_file()]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            archive.write(member, member.relative_to(out).as_posix())
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Проверка ZIP не пройдена")
    checksum = sha256(path.read_bytes())
    path.with_suffix(".zip.sha256").write_text(
        f"{checksum}  {path.name}\n", encoding="utf-8"
    )
    return path, checksum


def main():
    parser = argparse.ArgumentParser(description="Подготовка входов экспериментов ЛР2")
    parser.add_argument("--config", default="configs/lab2.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    commit = checked_commit(args.debug)
    config = read_json(ROOT / args.config)
    if config["k_values"] != [1, 3, 5]:
        raise ValueError("Для основной серии нужны k_values=[1, 3, 5]")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("Некорректный seed")

    run_id = uuid.uuid4().hex
    out = ROOT / "runs" / "lab2_prepare" / run_id
    out.mkdir(parents=True)
    metadata = {
        "run_id": run_id, "lab": 2, "series": "prepare_inputs", "debug": args.debug,
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "python": platform.python_version(),
        "seed": config["seed"], "device": "cpu", "model": config["model"],
        "model_revision": config["revision"],
        "oracle_rule": "lexicographic doc_id, then tuple of original sentence numbers",
    }
    (out / "metadata.json").write_bytes(json_bytes(metadata))
    (out / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    try:
        metadata["source_sha256"] = snapshot(out, args.config, config)
        builder = PromptBuilder(args.config)
        retriever = DenseRetriever(config["retrieval_config"])
        metadata["versions"] = {**retriever.passport["versions"], "PyYAML": version("PyYAML")}
        metadata["index_matrix_sha256"] = retriever.passport["files"]["embeddings.npy"]["sha256"]
        training = queries(retriever.dataset, "claims_train.jsonl")
        source_name = "claims_train.jsonl" if args.debug else "claims_dev.jsonl"
        texts = training if args.debug else queries(retriever.dataset, source_name)
        if args.debug:
            ids = sorted(training)[:3]
        else:
            ids = read_json(ROOT / retriever.dataset["split_file"])["dev"]
            if len(ids) != 90 or len(set(ids)) != 90:
                raise ValueError("Ожидаются 90 уникальных ID dev")
        if not set(ids) <= set(texts):
            raise ValueError("Не найдены все утверждения выбранной части")
        if not args.debug and set(ids) & set(training):
            raise ValueError("Рабочие утверждения пересекаются с claims_train")

        # Порядок вызовов не кодирует классы: сначала числовая сортировка, затем shuffle.
        ids = sorted(ids)
        random.Random(config["seed"]).shuffle(ids)
        # Единственное использование эталона при подготовке промптов — oracle.
        oracle = oracle_plan(load_gold(source_name, ids))
        if not args.debug and len(oracle) != 60:
            raise ValueError("Ожидаются 60 oracle-утверждений")

        # Два входа для будущего прогрева сервера, оба из claims_train.
        warm_id = min(training)
        warm_claim = training[warm_id]
        retriever.search(warm_claim)
        warm_search = retriever.search(warm_claim, k=5)
        warmups = [
            task(warm_id, warm_claim, builder.package(warm_claim, mode="llm_only")),
            task(warm_id, warm_claim, builder.pack(warm_claim, warm_search["found"], retriever.documents, 3), warm_search),
        ]
        with (out / "warmups.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for record in warmups:
                append(stream, {"warmup": True, **record})

        counts, seen, planned = Counter(), set(), {}
        with (out / "tasks.jsonl").open("w", encoding="utf-8", newline="\n") as inputs, \
                (out / "searches.jsonl").open("w", encoding="utf-8", newline="\n") as searches:
            for position, claim_id in enumerate(ids, 1):
                claim = texts[claim_id]
                search = retriever.search(claim, k=5)
                append(searches, {"claim_id": claim_id, "claim": claim, **search})
                records = [task(claim_id, claim, builder.package(claim, mode="llm_only"))]
                for k in config["k_values"]:
                    packed = builder.pack(claim, search["found"], retriever.documents, k)
                    records.append(task(claim_id, claim, packed, search))
                if claim_id in oracle:
                    witness = oracle[claim_id]
                    block = format_document(retriever.documents[witness["doc_id"]], witness["sentences"])
                    packed = {**builder.package(claim, block, mode="oracle"), "passed": [witness]}
                    records.append(task(claim_id, claim, packed))

                for record in records:
                    if record["task_id"] in seen:
                        raise ValueError("Повтор task_id")
                    seen.add(record["task_id"])
                    name = series_name(record)
                    counts[name] += 1
                    planned.setdefault(name, []).append(claim_id)
                    append(inputs, record)
                print(f"{position}/{len(ids)}: подготовлено заданий {sum(counts.values())}", flush=True)

        expected = {"llm_only": len(ids), **{f"rag_k{k}": len(ids) for k in config["k_values"]}}
        if oracle:
            expected["oracle"] = len(oracle)
        if dict(counts) != expected:
            raise ValueError(f"Число заданий не совпало с планом: {counts}")
        if checked_commit(args.debug) != commit:
            raise ValueError("Коммит изменился во время подготовки")
        metadata.update({
            "status": "completed", "task_count": len(seen), "counts": dict(counts),
            "expected_ids_by_series": planned,
            "files_sha256": {name: sha256((out / name).read_bytes()) for name in (
                "tasks.jsonl", "searches.jsonl", "warmups.jsonl", "config.yaml"
            )},
        })
    except BaseException as error:
        metadata.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        (out / "metadata.json").write_bytes(json_bytes(metadata))

    archive, checksum = make_archive(out, run_id)
    print("Режим:", "DEBUG — только claims_train" if args.debug else "ОСНОВНАЯ ПОДГОТОВКА")
    print("Число заданий:", metadata["task_count"])
    print("По сериям:", json.dumps(metadata["counts"], ensure_ascii=False))
    print("Каталог:", out)
    print("Архив:", archive)
    print("SHA-256 архива:", checksum)
    print("Qwen не вызывался. Ответы и Accuracy ещё не получены.")


if __name__ == "__main__":
    main()
