# Запуск: python lab1/kaggle_run.py --stage check|exp1|cache|exp2|exp3|analyze --root PATH
# Условия берём из Git, результаты сохраняем за пределами репозитория.
import argparse
import asyncio
import csv
import hashlib
import json
import math
import shutil
import statistics as stats
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import httpx
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from kaggle_harness import measure, metrics

REPO = Path(__file__).resolve().parent.parent
C = json.loads((REPO / "configs/lab1_kaggle.json").read_text(encoding="utf-8"))
DATA = REPO / "lab1/data"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append(path, value):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def generation(n):
    return dict(model=C["model"], temperature=0, top_p=1, seed=C["seed"],
                presence_penalty=0, frequency_penalty=0, max_tokens=n,
                extra_body=dict(ignore_eos=True, top_k=-1, repetition_penalty=1.0))


class Prompts:
    def __init__(self):
        manifest = read(DATA / "manifest.json")
        assert sha(DATA / manifest["book_file"]) == manifest["book_sha256"]
        assert sha(DATA / "model_config.json") == manifest["model_config_sha256"]
        assert manifest["model_revision"] == C["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            C["model"], revision=C["revision"], local_files_only=True)
        text = (DATA / manifest["book_file"]).read_bytes().decode("utf-8-sig")
        start = text.index("\n", text.index("*** START OF")) + 1
        text = text[start:text.index("*** END OF", start)].strip()
        self.tokens = self.tokenizer.encode(text, add_special_tokens=False)

    def count(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def make(self, length, request_id, start=0):
        header = f"Request ID: {request_id}\nSummarize the following text in one sentence.\n\n"
        budget = length - self.count([dict(role="user", content=header)])
        for _ in range(30):
            if not 0 < budget <= len(self.tokens) - start:
                raise ValueError("Недостаточно текста для запроса")
            text = header + self.tokenizer.decode(self.tokens[start:start + budget])
            messages = [dict(role="user", content=text)]
            actual = self.count(messages)
            if actual == length:
                return dict(request_id=request_id, target_prompt_tokens=length, messages=messages)
            budget += length - actual
        raise ValueError(f"Не получилось ровно {length} токенов")


def validate(result, request, n):
    if result["status"] == "completed" and (
        result["prompt_tokens"] != request["target_prompt_tokens"] or result["output_tokens"] != n
    ):
        result.update(status="failed", error="Длина входа или выхода не совпала с условием")


async def one(client, request, n, folder, kind="measurement", **extra):
    result = await measure(client, request["messages"], generation(n))
    validate(result, request, n)
    result.update(request_id=request["request_id"], kind=kind,
                  target_prompt_tokens=request["target_prompt_tokens"], **extra)
    append(folder / ("warmup.jsonl" if kind == "warmup" else "results.jsonl"), result)
    return result


def ensure(result):
    if result["status"] != "completed":
        raise RuntimeError(f"Результат ошибки сохранён: {result.get('error')}")


async def monitor(base_url, path, stop, ready):
    # Снимки до, во время и после уровня нужны для наблюдения очереди и KV-кеша.
    url = base_url.removesuffix("/v1").rstrip("/") + "/metrics"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            item = {"time": time.perf_counter()}
            try:
                response = await client.get(url)
                response.raise_for_status()
                item["raw"] = response.text
            except Exception as error:
                item["error"] = repr(error)
            append(path, item)
            ready.set()
            if stop.is_set():
                break
            try:
                await asyncio.wait_for(stop.wait(), timeout=1)
            except asyncio.TimeoutError:
                pass


async def experiment(stage, root):
    if git("status", "--porcelain"):
        raise RuntimeError("Git должен быть чистым; результаты храни вне репозитория")
    server = read(root / "current_server.json")
    if server["prefix_caching"] != (stage == "cache"):
        raise ValueError("Неправильный режим кеширования сервера для этого этапа")
    if server["revision"] != C["revision"]:
        raise ValueError("Ревизия сервера не совпала с конфигурацией")
    folder = root / stage / uuid4().hex
    folder.mkdir(parents=True)
    meta = dict(stage=stage, run_id=folder.name, status="running", commit=git("rev-parse", "HEAD"),
                started_at=datetime.now(timezone.utc).isoformat(), config=C, server=server,
                versions={name: version(name) for name in ["vllm", "openai", "torch", "transformers", "numpy"]},
                python=sys.version, book_sha256=read(DATA / "manifest.json")["book_sha256"])
    write(folder / "metadata.json", meta)
    shutil.copyfile(server["log"], folder / "server_startup.log")
    print("Каталог:", folder, flush=True)
    try:
        prompts = Prompts()
        prefix = uuid4().hex
        async with AsyncOpenAI(base_url=C["base_url"], api_key="none", max_retries=0,
                               timeout=C["request_timeout_s"]) as client:
            warm = prompts.make(256, "warmup-" + prefix)
            if stage == "check":
                append(folder / "requests.jsonl", {**warm, "max_tokens": 64, "kind": "repeatability_check"})
                assert math.isclose(metrics(0, [0.4, 2.4], 101)["ttft_s"], 0.4)
                assert math.isclose(metrics(0, [0.4, 2.4], 101)["tpot_s"], 0.02)
                assert metrics(0, [0.4], 1)["tpot_s"] is None
                answers = []
                for i in range(2):
                    result = await one(client, warm, 64, folder, "check", attempt=i + 1)
                    ensure(result)
                    answers.append(result["text"])
                checks = dict(arithmetic_ok=True, output_64_ok=True, texts_equal=answers[0] == answers[1])
                for n in C["input_lengths"]:
                    req = prompts.make(n, f"check-{prefix}-{n}")
                    append(folder / "requests.jsonl", req)
                    ensure(await one(client, req, 1, folder, "length_check"))
                mc = read(DATA / "model_config.json")
                head_dim = mc.get("head_dim") or mc["hidden_size"] // mc["num_attention_heads"]
                checks["kv_bytes_per_token_fp16"] = 2 * mc["num_hidden_layers"] * mc["num_key_value_heads"] * head_dim * 2
                checks["kv_bytes_for_8_users_4000_tokens"] = checks["kv_bytes_per_token_fp16"] * 8 * 4000
                checks["server_kv_log_lines"] = [s for s in Path(server["log"]).read_text(errors="replace").splitlines()
                                                 if any(k in s.lower() for k in ["kv cache", "kv_cache", "concurrency"])]
                write(folder / "checks.json", checks)
                print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
            elif stage in ["exp1", "exp2", "cache"]:
                if stage == "exp1":
                    plan = [(n, 16) for n in C["input_lengths"] for _ in range(C["repeats"])]
                elif stage == "exp2":
                    plan = [(1024, n) for n in C["output_lengths"] for _ in range(C["repeats"])]
                else:
                    plan = [(8192, 16)] * 3
                requests = [prompts.make(n, f"{prefix}-{i}") for i, (n, _) in enumerate(plan)]
                if stage == "cache":
                    requests = [requests[0]] * 3  # Полностью одинаковые метка и текст.
                for req, (_, n) in zip(requests, plan):
                    append(folder / "requests.jsonl", {**req, "max_tokens": n})
                append(folder / "requests.jsonl", {**warm, "max_tokens": 16, "kind": "warmup"})
                ensure(await one(client, warm, 16, folder, "warmup"))
                for i, (req, (_, n)) in enumerate(zip(requests, plan), 1):
                    result = await one(client, req, n, folder, attempt=i)
                    ensure(result)
                    print(f"{stage} {i}/{len(plan)} | вход {result['prompt_tokens']} | выход {n} | "
                          f"TTFT {result['ttft_s']:.3f} с | TPOT {result['tpot_s']*1000:.2f} мс", flush=True)
            elif stage == "exp3":
                for c in C["concurrency"]:
                    total = max(16, 4 * c)
                    warm_load = prompts.make(1024, f"warmup-{prefix}-{c}")
                    append(folder / "requests.jsonl", {**warm_load, "max_tokens": 128, "concurrency": c, "kind": "warmup"})
                    ensure(await one(client, warm_load, 128,
                                     folder, "warmup", concurrency=c))
                    queue = asyncio.Queue()
                    for i in range(total):
                        # Разные участки книги: не только разные метки.
                        start = i * max(0, len(prompts.tokens) - 2048) // max(1, total - 1)
                        req = prompts.make(1024, f"{prefix}-{c}-{i}", start)
                        append(folder / "requests.jsonl", {**req, "max_tokens": 128, "concurrency": c})
                        queue.put_nowait(req)
                    results = []

                    async def worker():
                        while True:
                            try:
                                req = queue.get_nowait()
                            except asyncio.QueueEmpty:
                                return
                            results.append(await one(client, req, 128, folder, concurrency=c))

                    stop = asyncio.Event()
                    ready = asyncio.Event()
                    monitoring = asyncio.create_task(monitor(C["base_url"], folder / f"metrics_c{c}.jsonl", stop, ready))
                    await ready.wait()
                    try:
                        await asyncio.gather(*(worker() for _ in range(c)))
                    finally:
                        stop.set()
                        await monitoring
                    assert len(results) == total
                    for result in results:
                        ensure(result)
                    print(f"Нагрузка {c}: выполнено {total} запросов", flush=True)
        meta["status"] = "completed"
    except Exception:
        meta.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(folder / "metadata.json", meta)
    print("ЗАВЕРШЕНО:", folder, flush=True)


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import mlflow

    selected = {}
    for stage in ["check", "exp1", "cache", "exp2", "exp3"]:
        candidates = [p for p in (root / stage).glob("*/metadata.json") if read(p)["status"] == "completed"]
        if not candidates:
            raise ValueError(f"Нет завершённой серии {stage}")
        selected[stage] = max(candidates, key=lambda p: read(p)["finished_at"]).parent
    out = root / "analysis" / uuid4().hex
    out.mkdir(parents=True)
    write(out / "selection.json", {k: str(v) for k, v in selected.items()})
    metadata = [read(p / "metadata.json") for p in selected.values()]
    assert all(m["config"] == C for m in metadata), "Различаются конфигурации серий"
    assert len({m["commit"] for m in metadata}) == 1, "Различаются версии кода серий"
    assert all(m["versions"] == metadata[0]["versions"] for m in metadata), "Различаются версии библиотек"
    # Пересчитываем все метрики по исходным отметкам и проверяем сохранённые значения.
    datasets = {}
    for stage, folder in selected.items():
        rows = lines(folder / "results.jsonl")
        for row in rows:
            assert row["status"] == "completed"
            calculated = metrics(row["t0"], row["text_chunk_times"], row["output_tokens"])
            for key, value in calculated.items():
                assert (value is None and row[key] is None) or (value is not None and math.isclose(value, row[key], abs_tol=1e-8))
            row.update(calculated)
        datasets[stage] = rows
    assert len(datasets["exp1"]) == len(C["input_lengths"]) * C["repeats"]
    assert len(datasets["exp2"]) == len(C["output_lengths"]) * C["repeats"]
    assert len(datasets["cache"]) == 3
    assert len(datasets["exp3"]) == sum(max(16, 4*c) for c in C["concurrency"])

    table1 = []
    for n in C["input_lengths"]:
        group = [r for r in datasets["exp1"] if r["prompt_tokens"] == n]
        assert len(group) == C["repeats"] and all(r["output_tokens"] == 16 for r in group)
        values = [r["ttft_s"] for r in group]
        med = stats.median(values)
        table1.append(dict(source_run_id=selected["exp1"].name, input_tokens=n, count=len(group), ttft_median_s=med,
                           ttft_min_s=min(values), ttft_max_s=max(values), prefill_estimate_tokens_s=n/med))
    save_csv(out / "exp1.csv", table1)
    fig, ax = plt.subplots()
    ax.errorbar([r["input_tokens"] for r in table1], [r["ttft_median_s"] for r in table1],
                yerr=[[r["ttft_median_s"]-r["ttft_min_s"] for r in table1],
                      [r["ttft_max_s"]-r["ttft_median_s"] for r in table1]], fmt="o-", capsize=4)
    ax.set(xlabel="Длина запроса, токенов", ylabel="TTFT, с", title=f"Медиана и диапазон, {C['repeats']} запросов")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "exp1_ttft.png", dpi=200); plt.close(fig)

    table2 = []
    for n in C["output_lengths"]:
        group = [r for r in datasets["exp2"] if r["output_tokens"] == n]
        assert len(group) == C["repeats"] and all(r["prompt_tokens"] == 1024 for r in group)
        times = [r["e2e_s"] for r in group]
        table2.append(dict(source_run_id=selected["exp2"].name, output_tokens=n, count=len(group), e2e_median_s=stats.median(times),
                           e2e_min_s=min(times), e2e_max_s=max(times),
                           ttft_median_s=stats.median(r["ttft_s"] for r in group),
                           tpot_median_s=stats.median(r["tpot_s"] for r in group)))
    save_csv(out / "exp2.csv", table2)
    # y = intercept + slope*(N-1): intercept сопоставим с TTFT, slope — с TPOT.
    x = np.array([r["output_tokens"]-1 for r in datasets["exp2"]], dtype=float)
    y = np.array([r["e2e_s"] for r in datasets["exp2"]])
    slope, intercept = np.polyfit(x, y, 1)
    fit = dict(intercept_s=float(intercept), slope_s_per_token=float(slope),
               formula="e2e_s = intercept_s + slope_s_per_token * (N - 1)",
               rough_t4_4b_fp16_lower_bound_s=8e9/300e9,
               lower_bound_note="Грубая оценка из методички, не точное время нашей модели")
    write(out / "exp2_fit.json", fit)
    fig, ax = plt.subplots()
    ax.scatter(x+1, y, label="Отдельные запросы")
    ax.plot(x+1, intercept+slope*x, label="Аппроксимация: a + b*(N-1)")
    ax.set(xlabel="Длина ответа, токенов", ylabel="Полное время ответа, с")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out / "exp2_time.png", dpi=200); plt.close(fig)

    load = []
    diagnostics = {}
    for c in C["concurrency"]:
        group = [r for r in datasets["exp3"] if r["concurrency"] == c]
        assert len(group) == max(16, 4*c)
        assert all(r["prompt_tokens"] == 1024 and r["output_tokens"] == 128 for r in group)
        duration = max(r["text_chunk_times"][-1] for r in group) - min(r["t0"] for r in group)
        ttft = [r["ttft_s"] for r in group]
        tpot = [r["tpot_s"] for r in group]
        good = sum(r["ttft_s"] <= C["ttft_limit_s"] and r["tpot_s"] <= C["tpot_limit_s"] for r in group)
        p90 = float(np.percentile(ttft, 90))
        row = dict(source_run_id=selected["exp3"].name, concurrency=c, requests=len(group), failed_requests=0, window_s=duration,
                   output_tokens_s=sum(r["output_tokens"] for r in group)/duration,
                   requests_s=len(group)/duration, ttft_median_s=stats.median(ttft), ttft_p90_s=p90,
                   tpot_median_s=stats.median(tpot), good_requests=good, goodput_requests_s=good/duration,
                   meets_limits=p90 <= C["ttft_limit_s"] and stats.median(tpot) <= C["tpot_limit_s"])
        load.append(row)
        # Не принимаем отсутствие метрики за нулевую очередь.
        found = {"vllm:num_requests_waiting": [], "vllm:kv_cache_usage_perc": []}
        for snap in lines(selected["exp3"] / f"metrics_c{c}.jsonl"):
            for line in snap.get("raw", "").splitlines():
                if line.startswith("#"):
                    continue
                for name in found:
                    if line.startswith(name + "{") or line.startswith(name + " "):
                        found[name].append(float(line.split()[-1]))
        diagnostics[str(c)] = {name: {"samples": len(values), "max_observed": max(values) if values else None}
                                for name, values in found.items()}
    save_csv(out / "load.csv", load)
    write(out / "server_metrics_summary.json", diagnostics)
    fig, ax = plt.subplots()
    ax.plot([r["output_tokens_s"] for r in load], [r["ttft_p90_s"] for r in load], "o-")
    for r in load:
        ax.annotate(str(r["concurrency"]), (r["output_tokens_s"], r["ttft_p90_s"]))
    ax.axhline(C["ttft_limit_s"], ls="--", color="red", label="Предел TTFT")
    ax.set(xlabel="Выходных токенов/с, все клиенты", ylabel="p90 TTFT, с", title="Подписи точек: число клиентов")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out / "exp3_load.png", dpi=200); plt.close(fig)
    cache = datasets["cache"]
    assert len(set(r["request_id"] for r in cache)) == 1
    summary = dict(selected_runs={k: v.name for k,v in selected.items()},
                   cache_ttft_s=[r["ttft_s"] for r in cache],
                   cache_speedup_second=cache[0]["ttft_s"]/cache[1]["ttft_s"],
                   cache_speedup_third=cache[0]["ttft_s"]/cache[2]["ttft_s"],
                   max_concurrency_meeting_limits=max([r["concurrency"] for r in load if r["meets_limits"]], default=0),
                   note="0 означает: ни один проверенный уровень не выполнил оба условия", fit=fit)
    write(out / "summary.json", summary)
    write(root / "latest_analysis.json", {"analysis_id": out.name})

    # Регистрация сохранённых опытов и графиков: новых запросов к модели нет.
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name("lr1_kaggle") is None:
        mlflow.create_experiment("lr1_kaggle", artifact_location=(root / "mlartifacts").as_uri())
    mlflow.set_experiment("lr1_kaggle")
    index_path = root / "mlflow_imports.json"
    index = read(index_path) if index_path.exists() else {}
    for stage, folder in selected.items():
        if folder.name in index:
            continue
        with mlflow.start_run(run_name=f"{stage}-{folder.name[:8]}") as run:
            meta = read(folder / "metadata.json")
            mlflow.set_tags({"stage": stage, "source_run_id": folder.name, "commit": meta["commit"],
                             "platform": "Kaggle", "logging_mode": "imported_from_saved_files"})
            mlflow.log_params({"model": C["model"], "revision": C["revision"], "seed": C["seed"],
                               "prefix_caching": meta["server"]["prefix_caching"], "gpu_count_used": 1})
            for i, row in enumerate(datasets[stage], 1):
                mlflow.log_metrics({k: row[k] for k in ["ttft_s", "tpot_s", "e2e_s"] if row[k] is not None}, step=i)
            mlflow.log_artifacts(str(folder), artifact_path="run")
            index[folder.name] = run.info.run_id
        write(index_path, index)
    with mlflow.start_run(run_name="analysis-" + out.name[:8]):
        mlflow.set_tag("logging_mode", "analysis_only")
        mlflow.log_artifacts(str(out), artifact_path="analysis")
        mlflow.log_artifact(str(REPO / "forecasts/lab1_kaggle.md"), artifact_path="provenance")
    mlflow.search_runs(max_results=10000).to_csv(root / "mlflow_runs.csv", index=False, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("ТАБЛИЦЫ, ГРАФИКИ И MLFLOW:", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["check", "exp1", "cache", "exp2", "exp3", "analyze"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "analyze":
        analyze(root)
    else:
        asyncio.run(experiment(args.stage, root))


if __name__ == "__main__":
    main()
