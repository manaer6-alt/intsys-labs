# ЛР3, опыт 2. Проверяем входы и запускаем Qwen для RRF и RRF + MiniLM.
# Используем тот же HTTP-клиент, схему ответа и проверки, что в ЛР2.
# Записываем каждый ответ сразу. Повтор команды продолжает незаписанные задания.
# Эталонная разметка остаётся на ноутбуке для последующей оценки.

import argparse
import sys
from collections import Counter
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lab2"))
from run_inference import (
    DEFAULTS, RESPONSE_FORMAT, append, call_model, fingerprint, git_commit,
    http_json, now, operational_metrics, read_json, read_lines, sha,
    validate_record_ids, validate_server, write_json,
)

SERIES = ("rrf", "rrf_ce")
FILES = ("tasks.jsonl", "warmups.jsonl", "generation_config.json", "answer_schema.json")
EXPECTED_META_SHA = "ff5b2a5d12eef732458829aee901a20383084b8f018c0c4d3448bd25658a24df"


def load_inputs(path):
    meta_path = path / "metadata.json"
    if sha(meta_path.read_bytes()) != EXPECTED_META_SHA:
        raise ValueError("Нужен пакет входов ЛР3 из подготовленного запуска 061da805")
    meta = read_json(meta_path)
    for name in FILES:
        if sha((path / name).read_bytes()) != meta["files_sha256"][name]:
            raise ValueError(f"Не совпал SHA-256 файла {name}")
    config = read_json(path / "generation_config.json")
    if config != read_json(REPO / "configs/lab2.json"):
        raise ValueError("Настройки генерации отличаются от ЛР2")
    if read_json(path / "answer_schema.json") != RESPONSE_FORMAT:
        raise ValueError("Схемы ответа ЛР2 и ЛР3 отличаются")
    tasks = read_lines(path / "tasks.jsonl")
    warmups = read_lines(path / "warmups.jsonl")
    if Counter(t["series"] for t in tasks) != {name: 90 for name in SERIES}:
        raise ValueError("Ожидалось по 90 заданий в каждой серии")
    ids = [[t["claim_id"] for t in tasks if t["series"] == name] for name in SERIES]
    if len(set(ids[0])) != 90 or ids[0] != ids[1]:
        raise ValueError("Нужны те же 90 утверждений в том же порядке")
    if any(t["mode"] != "rag" or t["k"] != 3 for t in tasks):
        raise ValueError("Основные запросы должны быть RAG с k=3")
    if len(warmups) != 2 or {t["mode"] for t in warmups} != {"llm_only", "rag"}:
        raise ValueError("Ожидались два проверочных запроса из claims_train")
    all_tasks = tasks + warmups
    if len({t["task_id"] for t in all_tasks}) != len(all_tasks):
        raise ValueError("ID заданий повторяются")
    if any(not 0 < t["prompt_tokens"] <= config["input_budget"] for t in all_tasks):
        raise ValueError("Нарушен входной бюджет")
    return config, tasks, warmups


def check_tokens(out, tasks, config, server):
    endpoint = config["base_url"].removesuffix("/v1") + "/tokenize"
    with (out / "token_checks.jsonl").open("w", encoding="utf-8") as stream:
        for i, task in enumerate(tasks, 1):
            reply = http_json(endpoint, {"model": config["model"], "messages": task["messages"],
                                         "add_generation_prompt": True, "add_special_tokens": False})
            append(stream, {"task_id": task["task_id"], "expected": task["prompt_tokens"],
                            "actual": reply["count"]})
            if (reply["count"] != task["prompt_tokens"]
                    or reply["max_model_len"] != server["profile"]["max_model_len"]):
                raise ValueError(f"Токенизация или предел контекста отличаются: {task['task_id']}")
            if i % 30 == 0 or i == len(tasks):
                print(f"Проверка длины входов: {i}/{len(tasks)}", flush=True)


def run(stage, root, server, commit, config, tasks, warmups):
    signature = fingerprint({"inputs": EXPECTED_META_SHA, "commit": commit,
                             "profile": server["profile"], "defaults": DEFAULTS})
    approval = {"signature": signature, "session": server["session_id"]}
    selection_path = root / "selection.json"
    selection = read_json(selection_path) if selection_path.exists() else {}
    if stage != "check" and read_json(root / "approved_check.json") != approval:
        raise ValueError("Выполни check для текущего сервера и коммита")
    run_id = selection.get(stage)
    if run_id:
        out = root / "runs" / run_id
        meta = read_json(out / "metadata.json")
        if stage == "check" and (meta.get("approval") != approval or meta["status"] != "completed"):
            run_id = None
        elif meta["signature"] != signature:
            raise ValueError("Условия изменились. Продолжать серию с этим набором результатов нельзя")
    if not run_id:
        run_id = uuid4().hex
        out = root / "runs" / run_id
        meta = {"run_id": run_id, "series": stage, "status": "running", "started_at": now(),
                "signature": signature, "approval": approval, "execution_commit": commit,
                "inputs_metadata_sha256": EXPECTED_META_SHA, "server_sessions": []}
        write_json(out / "metadata.json", meta)
        selection[stage] = run_id
        write_json(selection_path, selection)

    planned = warmups if stage == "check" else [t for t in tasks if t["series"] == stage]
    results = out / "results.jsonl"
    records = read_lines(results)
    validate_record_ids(records, planned)
    if meta["status"] == "completed":
        if len(records) != len(planned) or sha(results.read_bytes()) != meta["results_sha256"]:
            raise ValueError("Повреждены результаты завершённой серии")
        if stage == "check":
            if sha((out / "token_checks.jsonl").read_bytes()) != meta["token_checks_sha256"]:
                raise ValueError("Повреждён журнал проверки входов")
            write_json(root / "approved_check.json", approval)
        print(f"{stage}: уже завершено, {len(records)}/{len(planned)}, run_id={run_id}")
        return

    meta["status"] = "running"
    meta.pop("error", None)
    if server["session_id"] not in meta["server_sessions"]:
        meta["server_sessions"].append(server["session_id"])
    write_json(out / "metadata.json", meta)
    try:
        if stage == "check":
            check_tokens(out, tasks + warmups, config, server)
        done = {r["task_id"] for r in records}
        with results.open("a", encoding="utf-8") as stream:
            for task in planned:
                if task["task_id"] in done:
                    continue
                record = call_model(task, config)
                record["server_session_id"] = server["session_id"]
                if record.get("interrupted"):
                    # Прерванный вызов оставляем в журнале, а задание — в очереди.
                    with (out / "interrupted.jsonl").open("a", encoding="utf-8") as journal:
                        append(journal, record)
                    raise KeyboardInterrupt
                append(stream, record)
                records.append(record)
                print(f"{stage} {len(records)}/{len(planned)} | claim={task['claim_id']} | "
                      f"{record['status']} | {record['parse_status']} | {record['generation_s']:.2f} с", flush=True)
                if record["fatal_error"]:
                    raise RuntimeError(record["error"])
                if stage == "check" and record["parse_status"] != "ok":
                    raise ValueError("Проверочный ответ отклонён валидатором")
        write_json(out / "metrics.json", operational_metrics(records, len(planned)))
        meta.update(status="completed", results_sha256=sha(results.read_bytes()))
        if stage == "check":
            meta["token_checks_sha256"] = sha((out / "token_checks.jsonl").read_bytes())
    except BaseException as error:
        meta.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        meta.update(recorded=len(records), finished_at=now())
        write_json(out / "metadata.json", meta)
    if stage == "check":
        write_json(root / "approved_check.json", approval)
    print(f"ЗАВЕРШЕНО: {stage}, run_id={run_id}", flush=True)


def main():
    import fcntl
    import signal

    from run_inference import stop_on_signal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("check", *SERIES), required=True)
    args = parser.parse_args()
    config, tasks, warmups = load_inputs(args.inputs)
    server = read_json(args.root / "current_server.json")
    validate_server(server, config)
    if not server["active"]:
        raise ValueError("Сервер остановлен")
    signal.signal(signal.SIGTERM, stop_on_signal)
    with (args.root / "inference.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.stage, args.root, server, git_commit(), config, tasks, warmups)


if __name__ == "__main__":
    main()
