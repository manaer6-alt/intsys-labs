"""Подготовить фиксированное разбиение AgentDojo для ЛР4."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suite


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_VERSION = "v1.2.2"
PACKAGE_VERSION = "0.1.35"
SUITE_NAME = "workspace"


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main():
    installed = version("agentdojo")
    if installed != PACKAGE_VERSION:
        raise RuntimeError(f"AgentDojo: нужен {PACKAGE_VERSION}, установлен {installed}")

    suite = get_suite(BENCHMARK_VERSION, SUITE_NAME)
    task_ids = sorted(suite.user_tasks, key=lambda task_id: int(task_id.rsplit("_", 1)[1]))
    expected = [f"user_task_{n}" for n in range(40)]
    if task_ids != expected:
        raise RuntimeError("Набор задач отличается от методички")

    debug = [task_id for task_id in task_ids if int(task_id.rsplit("_", 1)[1]) % 4 == 0]
    final = [task_id for task_id in task_ids if task_id not in debug]
    assert len(debug) == 10 and len(final) == 30

    split = {"debug": debug, "final": final}
    split_bytes = json_bytes(split)
    manifest = {
        "source": "AgentDojo",
        "package_version": installed,
        "benchmark_version": BENCHMARK_VERSION,
        "suite": SUITE_NAME,
        "task_count": len(task_ids),
        "debug_count": len(debug),
        "final_count": len(final),
        "split_rule": "debug: numeric task ID divisible by 4; final: remaining tasks",
        "task_ids_sha256": hashlib.sha256(json_bytes(task_ids)).hexdigest(),
        "split_path": "data/agentdojo/split.json",
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
    }
    directory = ROOT / "data" / "agentdojo"
    directory.mkdir(parents=True, exist_ok=True)
    for path, content in (
        (directory / "split.json", split_bytes),
        (directory / "manifest.json", json_bytes(manifest)),
    ):
        if path.exists() and path.read_bytes() != content:
            raise RuntimeError(f"Данные уже зафиксированы и отличаются: {path}")
        path.write_bytes(content)

    print(f"AgentDojo {installed}, benchmark {BENCHMARK_VERSION}, suite {SUITE_NAME}")
    print(f"Всего: {len(task_ids)}, debug: {len(debug)}, final: {len(final)}")
    print("Debug:", ", ".join(debug))
    print("SHA-256 split:", manifest["split_sha256"])
    print("Файлы:", directory)


if __name__ == "__main__":
    main()
