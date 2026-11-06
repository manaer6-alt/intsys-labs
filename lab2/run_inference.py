# Запуски ЛР2 в Kaggle: читаем подготовленные входы и обращаемся к vLLM по HTTP.
# Каждый основной режим получает отдельный run_id и каталог с результатами.
# check: сверка токенизации всех входов и два пробных ответа на claims_train.
# Основные режимы: llm_only, rag_k3, oracle, rag_k1, rag_k5.
# Ответы сохраняются сразу. Повтор команды продолжает только отсутствующие записи.
# Уже записанные ответы, включая неудачные, повторно не запрашиваются.
# Эталон и Accuracy здесь не используются; оценку проведём локально после скачивания.

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml

from answers import RESPONSE_FORMAT, parse_answer

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("check", "llm_only", "rag_k3", "oracle", "rag_k1", "rag_k5")
EXPECTED = {"llm_only": 90, "rag_k1": 90, "rag_k3": 90, "rag_k5": 90, "oracle": 60}
DEFAULTS = {"top_k": -1, "repetition_penalty": 1.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "ignore_eos": False, "stream": False}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def series_name(task):
    if task["mode"] == "rag" and type(task["k"]) is int and task["k"] in (1, 3, 5):
        return f"rag_k{task['k']}"
    if task["mode"] in ("llm_only", "oracle") and task["k"] is None:
        return task["mode"]
    raise ValueError("Неизвестный режим подготовленного задания")


def git_commit():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    if git("status", "--porcelain"):
        raise ValueError("Рабочее дерево Git не чистое")
    return git("rev-parse", "HEAD")


def load_inputs(path):
    meta = read_json(path / "metadata.json")
    if meta["debug"] or meta["status"] != "completed" or meta["task_count"] != 420:
        raise ValueError("Нужен завершённый основной пакет из 420 заданий")
    for relative, expected in meta["files_sha256"].items():
        if sha((path / relative).read_bytes()) != expected:
            raise ValueError(f"Повреждён входной файл: {relative}")
    for relative, expected in meta["source_sha256"].items():
        archived = (path / "provenance" / relative).read_bytes()
        if sha(archived) != expected:
            raise ValueError(f"Повреждён снимок исходника: {relative}")
        current = (ROOT / relative).read_bytes()
        # Git может преобразовать CRLF в LF при переносе Windows -> Linux.
        if current.replace(b"\r\n", b"\n") != archived.replace(b"\r\n", b"\n"):
            raise ValueError(f"Исходник изменён после подготовки: {relative}")
    config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    if config != read_json(path / "provenance/configs/lab2.json"):
        raise ValueError("Снимки конфигурации не совпали")
    if config["temperature"] != 0 or type(config["max_tokens"]) is not int or not 1 <= config["max_tokens"] <= 128:
        raise ValueError("Нужны temperature=0 и не более 128 выходных токенов")
    if type(config["input_budget"]) is not int or not 1 <= config["input_budget"] <= 2048:
        raise ValueError("Входной бюджет должен быть не более 2048 токенов")
    schema = read_json(path / "provenance/lab2/answer_schema.json")
    if schema != RESPONSE_FORMAT:
        raise ValueError("Схема ответа отличается от подготовленной")
    tasks, warmups = read_lines(path / "tasks.jsonl"), read_lines(path / "warmups.jsonl")
    if len(tasks) != 420 or len({t["task_id"] for t in tasks}) != 420:
        raise ValueError("Нужны 420 уникальных заданий")
    if dict(Counter(map(series_name, tasks))) != EXPECTED:
        raise ValueError("Количество заданий по режимам отличается от плана")
    for stage in EXPECTED:
        ids = [t["claim_id"] for t in tasks if series_name(t) == stage]
        if ids != meta["expected_ids_by_series"][stage] or len(set(ids)) != len(ids):
            raise ValueError(f"Список утверждений не совпал: {stage}")
    if len(warmups) != 2 or {t["mode"] for t in warmups} != {"llm_only", "rag"}:
        raise ValueError("Не найдены два входа для прогрева")
    for task in tasks + warmups:
        if type(task["prompt_tokens"]) is not int or not 0 < task["prompt_tokens"] <= config["input_budget"]:
            raise ValueError(f"Нарушен входной бюджет: {task['task_id']}")
    return {"metadata": meta, "config": config, "tasks": tasks, "warmups": warmups,
            "fingerprint": sha((path / "metadata.json").read_bytes())}


def http_json(url, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as error:
        with error:
            body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error
    return json.loads(body)


def validate_server(server, config):
    profile = server["profile"]
    if profile["model"] != config["model"] or profile["model_revision"] != config["revision"]:
        raise ValueError("Модель сервера отличается от подготовленных входов")
    if profile["tokenizer_revision"] != config["revision"] or profile["dtype"] != "half":
        raise ValueError("Нужны та же ревизия токенизатора и формат FP16")
    if profile["tensor_parallel_size"] != 1 or profile["prefix_caching"]:
        raise ValueError("Нужны одна GPU и отключённое кеширование префикса")
    environment = profile["environment"]
    if environment["cuda_device_count"] != 1 or "T4" not in environment["gpu_name"]:
        raise ValueError("Процесс должен использовать одну T4")
    if platform.python_version() != environment["python"]:
        raise ValueError("Python клиента и сервера отличается")
    for name, expected in environment["versions"].items():
        if version(name) != expected:
            raise ValueError(f"Версия {name} отличается от паспорта сервера")


def call_model(task, config):
    payload = {"model": config["model"], "messages": task["messages"],
               "temperature": config["temperature"], "top_p": config["top_p"],
               "max_tokens": config["max_tokens"], "seed": config["seed"],
               "response_format": RESPONSE_FORMAT, **DEFAULTS}
    request = Request(config["base_url"].rstrip("/") + "/chat/completions",
                      data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    record = {**task, "started_at": now(), "status": "failed", "raw_response": None,
              "raw_http_body": None, "response": None, "usage": None, "error": None,
              "fatal_error": False, "request_parameters": {k: v for k, v in payload.items() if k != "messages"}}
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=180) as response:
            body = response.read().decode("utf-8")
            record["http_status"] = response.status
        record["generation_s"] = time.perf_counter() - start
        record["raw_http_body"] = body
        answer = json.loads(body)
        record["response"] = answer
        choice = answer["choices"][0]
        record["raw_response"] = choice["message"].get("content")
        record["refusal"] = choice["message"].get("refusal")
        record["finish_reason"] = choice["finish_reason"]
        record["usage"] = answer["usage"]
        if answer["model"] != config["model"]:
            raise ValueError("Сервер вернул другое имя модели")
        if record["usage"]["prompt_tokens"] != task["prompt_tokens"]:
            raise ValueError("usage.prompt_tokens не совпал с подготовленным входом")
        count = record["usage"]["completion_tokens"]
        if type(count) is not int or not 0 <= count <= config["max_tokens"]:
            raise ValueError("Нарушен предел выходных токенов")
        record["status"] = "completed"
    except BaseException as error:
        record["generation_s"] = time.perf_counter() - start
        if isinstance(error, HTTPError):
            record["http_status"] = error.code
            with error:
                record["raw_http_body"] = error.read().decode("utf-8", errors="replace")
        record["error"] = f"{type(error).__name__}: {error}"
        record["fatal_error"] = True
        record["interrupted"] = isinstance(error, (KeyboardInterrupt, SystemExit))
    record.update(parse_answer(record["raw_response"], task["mode"], task["passed"]))
    record["finished_at"] = now()
    return record


def validate_record_ids(records, planned):
    ids = [r["task_id"] for r in records]
    allowed = {t["task_id"] for t in planned}
    if len(ids) != len(set(ids)) or not set(ids) <= allowed:
        raise ValueError("Повторяющиеся или посторонние записи результатов")


def operational_metrics(records, planned):
    return {"planned": planned, "recorded": len(records),
            "call_failed": sum(r["status"] != "completed" for r in records),
            "parse_failed": sum(r["parse_status"] != "ok" for r in records),
            "generation_seconds_total": sum(r["generation_s"] for r in records)}


def stop_on_signal(*_):
    raise KeyboardInterrupt


def run_stage(stage, prepared, root, server, commit):
    config = prepared["config"]
    planned = prepared["warmups"] if stage == "check" else [t for t in prepared["tasks"] if series_name(t) == stage]
    signature = fingerprint({"stage": stage, "inputs": prepared["fingerprint"],
                             "execution_commit": commit, "profile": server["profile"],
                             "generation_defaults": DEFAULTS})
    selection_path = root / "selection.json"
    selection = read_json(selection_path) if selection_path.exists() else {}
    run_id = selection.get(stage)
    meta = read_json(root / "runs" / run_id / "metadata.json") if run_id else None
    # Проверку повторяем после перезапуска сервера; старый каталог сохраняется.
    if stage == "check" and meta and (meta["server_session_id"] != server["session_id"] or meta["status"] != "completed"):
        run_id, meta = None, None
    if meta and meta["signature"] != signature:
        raise ValueError("Код, входы или условия изменились: эту серию автоматически не продолжаем")
    marker_path = root / "approved_check.json"
    if stage != "check" and (meta is None or meta["status"] != "completed"):
        marker = read_json(marker_path)
        if (marker["server_session_id"] != server["session_id"] or marker["inputs_fingerprint"] != prepared["fingerprint"]
                or marker["execution_commit"] != commit):
            raise ValueError("Сначала выполни check для текущего сервера и кода")
    if run_id is None:
        run_id = uuid.uuid4().hex
        out = root / "runs" / run_id
        out.mkdir(parents=True)
        meta = {"run_id": run_id, "lab": 2, "series": stage, "status": "running",
                "started_at": now(), "execution_commit": commit,
                "preparation_commit": prepared["metadata"]["git_commit"],
                "preparation_run_id": prepared["metadata"]["run_id"],
                "inputs_fingerprint": prepared["fingerprint"], "signature": signature,
                "server_session_id": server["session_id"], "server": server,
                "planned": len(planned), "resumed_at": []}
        write_json(out / "metadata.json", meta)
        (out / "config.yaml").write_text(yaml.safe_dump({**config, "series": stage,
            "generation_defaults": DEFAULTS, "response_format": RESPONSE_FORMAT}, sort_keys=False), encoding="utf-8")
        selection[stage] = run_id
        write_json(selection_path, selection)
    else:
        out = root / "runs" / run_id

    results_path = out / "results.jsonl"
    records = read_lines(results_path)
    validate_record_ids(records, planned)
    if meta["status"] == "completed":
        if len(records) != len(planned) or sha(results_path.read_bytes()) != meta["results_sha256"]:
            raise ValueError("Завершённые результаты повреждены")
        if stage == "check":
            if sha((out / "token_checks.jsonl").read_bytes()) != meta["token_checks_sha256"]:
                raise ValueError("Файл сверки токенизации повреждён")
            write_json(marker_path, {"run_id": run_id, "server_session_id": server["session_id"],
                "inputs_fingerprint": prepared["fingerprint"], "execution_commit": commit})
        print(f"{stage}: уже завершено, run_id={run_id}; повторных запросов нет.", flush=True)
        return

    meta["status"] = "running"
    meta["resumed_at"].append(now())
    if server["session_id"] not in meta.setdefault("server_sessions", []):
        meta["server_sessions"].append(server["session_id"])
    write_json(out / "metadata.json", meta)

    try:
        models = http_json(config["base_url"].rstrip("/") + "/models", timeout=10)
        if config["model"] not in [item["id"] for item in models["data"]]:
            raise ValueError("Нужная модель недоступна на сервере")
        if stage == "check":
            endpoint = config["base_url"].rstrip("/").removesuffix("/v1") + "/tokenize"
            with (out / "token_checks.jsonl").open("w", encoding="utf-8") as stream:
                for i, task in enumerate(prepared["tasks"] + prepared["warmups"], 1):
                    value = http_json(endpoint, {"model": config["model"], "messages": task["messages"],
                                                "add_generation_prompt": True, "add_special_tokens": False})
                    match = value["count"] == task["prompt_tokens"]
                    append(stream, {"task_id": task["task_id"], "expected": task["prompt_tokens"],
                                    "actual": value["count"], "matches": match})
                    if not match or value["count"] > config["input_budget"]:
                        raise ValueError(f"Длина входа не совпала: {task['task_id']}")
                    if value["max_model_len"] != server["profile"]["max_model_len"]:
                        raise ValueError("Предел контекста сервера отличается от паспорта")
                    if i % 50 == 0:
                        print(f"Сверка токенизации: {i}/422", flush=True)
            meta["token_checks"] = 422
        else:
            # Oracle использует такой же документный шаблон, как RAG.
            warmup_mode = "llm_only" if stage == "llm_only" else "rag"
            warmup = next(t for t in prepared["warmups"] if t["mode"] == warmup_mode)
            with (out / "warmups.jsonl").open("a", encoding="utf-8") as stream:
                value = call_model(warmup, config)
                value["server_session_id"] = server["session_id"]
                append(stream, {"warmup": True, **value})
                if value["fatal_error"]:
                    raise RuntimeError(value["error"])

        done = {r["task_id"] for r in records}
        print(f"{stage}: уже записано {len(records)}/{len(planned)}, run_id={run_id}", flush=True)
        with results_path.open("a", encoding="utf-8") as stream:
            for task in planned:
                if task["task_id"] in done:
                    continue
                value = call_model(task, config)
                value["server_session_id"] = server["session_id"]
                append(stream, value)
                records.append(value)
                print(f"{stage}: {len(records)}/{len(planned)} | claim={task['claim_id']} | "
                      f"{value['status']} | {value['parse_status']} | {value['generation_s']:.2f} с", flush=True)
                if value["fatal_error"]:
                    raise RuntimeError(value["error"])
                if stage == "check":
                    print("Пробный ответ", task["mode"], ":", value["raw_response"], flush=True)
                if stage == "check" and value["parse_status"] == "invalid_json":
                    raise ValueError("Пробный ответ не разобран как JSON; основные серии не запускаем")
        if len(records) != len(planned):
            raise ValueError("Не получены все запланированные записи")
        if git_commit() != commit:
            raise ValueError("Код изменился во время выполнения")
        write_json(out / "metrics.json", operational_metrics(records, len(planned)))
        meta.update(status="completed", results_sha256=sha(results_path.read_bytes()))
        if stage == "check":
            meta["token_checks_sha256"] = sha((out / "token_checks.jsonl").read_bytes())
            write_json(marker_path, {"run_id": run_id, "server_session_id": server["session_id"],
                "inputs_fingerprint": prepared["fingerprint"], "execution_commit": commit})
    except BaseException as error:
        meta.update(status="failed", error=f"{type(error).__name__}: {error}")
        error_path = out / f"error_{uuid.uuid4().hex[:8]}.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        meta["finished_at"] = now()
        meta["recorded"] = len(records)
        write_json(out / "metadata.json", meta)
    print(f"ЗАВЕРШЕНО: {stage}; результаты: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Вызовы Qwen для ЛР2")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    prepared = load_inputs(args.inputs)
    commit = git_commit()
    server = read_json(args.root / "current_server.json")
    validate_server(server, prepared["config"])
    # Позволяет аккуратно записать текущую неудачную попытку при остановке процесса.
    signal.signal(signal.SIGTERM, stop_on_signal)
    run_stage(args.stage, prepared, args.root, server, commit)


if __name__ == "__main__":
    main()
