# Проверка кеширования префикса: три одинаковых запроса по 8192 токена.
# Запускается в Colab из клона репозитория после коммита.
# Данные и результаты хранятся отдельно от исходного кода.

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from openai import OpenAI
from harness import measure


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Проверка кеширования префикса")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--server-command", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()

    # Определяем репозиторий по расположению программы.
    repo = Path(__file__).resolve().parent.parent

    def git(*arguments):
        return subprocess.check_output(
            ["git", *arguments], cwd=repo, text=True
        ).strip()

    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise RuntimeError("Перед экспериментом Git должен быть чистым")

    command = args.server_command.read_text(encoding="utf-8")
    if "--enable-prefix-caching" not in shlex.split(command):
        raise ValueError("В сохранённой команде не включён кеш префикса")

    # Берём один запрос из исходной серии и повторяем его без изменений.
    prompt_path = args.source / "prompts.jsonl"
    records = [
        json.loads(line)
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = next(
        r for r in records if r["target_prompt_tokens"] == 8192
    )

    # Параметры генерации сохраняем такими же, как в первой серии.
    generation = read_json(args.source / "generation.json")
    if generation["temperature"] != 0 or generation["max_tokens"] != 16:
        raise ValueError("Ожидаются температура 0 и ответ на 16 токенов")

    run_dir = args.output / uuid4().hex
    run_dir.mkdir(parents=True)

    metadata = {
        "experiment": "prefix_cache",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "source_run": str(args.source.resolve()),
        "request_id": record["request_id"],
        "base_url": args.base_url,
        "vllm_version": version("vllm"),
        "openai_version": version("openai"),
        "prompts_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "request.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copyfile(args.source / "generation.json", run_dir / "generation.json")
    shutil.copyfile(args.server_command, run_dir / "server_command.txt")

    client = OpenAI(
        base_url=args.base_url, api_key="none",
        timeout=300, max_retries=0,
    )

    ttfts = []

    for attempt in range(4):
        warmup = attempt == 0

        # Прогрев не должен заранее обработать длинный измерительный запрос.
        messages = (
            [{"role": "user", "content": "Warmup: explain what a token is."}]
            if warmup else record["messages"]
        )

        result = measure(client, messages, generation)
        result.update({
            "kind": "warmup" if warmup else "prefix_cache",
            "attempt": attempt,
        })

        if result["status"] == "completed":
            if result["output_tokens"] != 16:
                result.update(status="failed", error="Ответ не равен 16 токенам")
            elif not warmup and result["prompt_tokens"] != 8192:
                result.update(status="failed", error="Вход не равен 8192 токенам")

        # Сначала сохраняем результат, затем проверяем успешность.
        filename = "warmup.jsonl" if warmup else "results.jsonl"
        with (run_dir / filename).open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

        if result["status"] != "completed":
            raise RuntimeError(f"Результат сохранён, ошибка: {result['error']}")

        if warmup:
            print("Прогрев завершён", flush=True)
        else:
            ttfts.append(result["ttft_s"])
            print(
                f"Запрос {attempt}: TTFT = {result['ttft_s']:.3f} с",
                flush=True,
            )

    print(f"Ускорение второго: {ttfts[0] / ttfts[1]:.2f} раза")
    print(f"Ускорение третьего: {ttfts[0] / ttfts[2]:.2f} раза")
    print("Результаты:", run_dir)


if __name__ == "__main__":
    main()