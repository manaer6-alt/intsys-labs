# Локальный анализ ЛР2 по ZIP из Kaggle. Повторные вызовы Qwen не выполняются.
# Проверяем хеши и соответствие входов, заново разбираем ответы и считаем метрики.
# Ошибки ссылок остаются неуспехами по заранее зафиксированному протоколу.
# Диагностика только метки явно отделена от основной метрики с валидацией.
# Результат: таблицы CSV, три рисунка, четыре случая ошибок и summary.json.
# --budget-check 1024 пересобирает контексты для защиты без поиска и генерации.
# --mlflow регистрирует сохранённые запуски и результаты анализа в локальном MLflow.

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from prepare_data import ROOT, sha256, json_bytes, claim_label
from answers import parse_answer
from evaluation import contains_rationale, score_record, summarize

STAGES = ("llm_only", "rag_k1", "rag_k3", "rag_k5", "oracle")
EXPECTED = {"llm_only": 90, "rag_k1": 90, "rag_k3": 90, "rag_k5": 90, "oracle": 60}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def save_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def raw_label(record):
    try:
        answer = json.loads(record["raw_response"])
    except (TypeError, json.JSONDecodeError):
        return None
    return answer.get("label") if isinstance(answer, dict) else None


def load_archive(archive_path, source):
    digest = sha256(archive_path.read_bytes())
    sidecar = archive_path.with_suffix(".zip.sha256")
    require(sidecar.exists(), "Рядом с ZIP нужен файл .zip.sha256")
    require(sidecar.read_text(encoding="utf-8-sig").split()[0].lower() == digest, "SHA-256 архива не совпал")
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, "ZIP повреждён")
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            destination = (source / name).resolve()
            require(destination.is_relative_to(source.resolve()), "Недопустимый путь внутри ZIP")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
    inputs = source / "inputs"
    preparation = read_json(inputs / "metadata.json")
    require(not preparation["debug"] and preparation["status"] == "completed", "Нужны основные подготовленные входы")
    for name, expected in preparation["files_sha256"].items():
        require(sha256((inputs / name).read_bytes()) == expected, f"Повреждены входы: {name}")
    for name, expected in preparation["source_sha256"].items():
        require(sha256((inputs / "provenance" / name).read_bytes()) == expected, f"Повреждён исходник: {name}")
    # Правила оценки должны совпадать с зафиксированными до эксперимента.
    for name in ("evaluation.py", "answers.py", "prepare_data.py"):
        current = (ROOT / "lab2" / name).read_bytes().replace(b"\r\n", b"\n")
        old = (inputs / "provenance/lab2" / name).read_bytes().replace(b"\r\n", b"\n")
        require(current == old, f"Изменились зафиксированные правила оценки: {name}")

    dataset = read_json(inputs / "provenance/data/manifest.json")["datasets"]["scifact"]
    raw = (ROOT / dataset["files"]["claims_dev.jsonl"]["path"]).read_bytes()
    require(sha256(raw) == dataset["files"]["claims_dev.jsonl"]["sha256"], "Изменился эталон SciFact")
    ids = set(dataset["splits"]["dev"])
    gold = {c["id"]: c for c in map(json.loads, raw.decode("utf-8").splitlines()) if c["id"] in ids}
    require(len(gold) == 90 and Counter(claim_label(c) for c in gold.values()) ==
            {"SUPPORT": 30, "CONTRADICT": 30, "NOINFO": 30}, "Неверный состав dev")
    tasks = {t["task_id"]: t for t in rows(inputs / "tasks.jsonl")}
    require(len(tasks) == 420, "Нужны 420 уникальных входов")
    selection = read_json(source / "selection.json")
    runs, metadata = {}, {}
    profile = read_json(source / "current_server.json")["profile"]
    commits = set()
    for stage in ("check", *STAGES):
        run_id = selection[stage]
        folder = source / "runs" / run_id
        meta = read_json(folder / "metadata.json")
        records = rows(folder / "results.jsonl")
        require(meta["status"] == "completed", f"Серия не завершена: {stage}")
        require(sha256((folder / "results.jsonl").read_bytes()) == meta["results_sha256"], f"Повреждены ответы: {stage}")
        require(meta["inputs_fingerprint"] == sha256((inputs / "metadata.json").read_bytes()), "Смешаны входные пакеты")
        require(meta["server"]["profile"] == profile, "Смешаны профили сервера")
        require(meta["preparation_commit"] == preparation["git_commit"], "Изменился коммит подготовки")
        commits.add(meta["execution_commit"])
        metadata[stage] = meta
        if stage == "check":
            checks = rows(folder / "token_checks.jsonl")
            require(len(checks) == 422 and all(c["matches"] and c["actual"] == c["expected"] for c in checks), "Сверка токенизации не пройдена")
            require(sha256((folder / "token_checks.jsonl").read_bytes()) == meta["token_checks_sha256"], "Изменился журнал токенизации")
            require(len(records) == 2, "Неверное число проверочных ответов")
            continue
        expected_ids = set(gold) if stage != "oracle" else {i for i, c in gold.items() if c["evidence"]}
        require(len(records) == EXPECTED[stage] and {r["claim_id"] for r in records} == expected_ids, f"Нарушен знаменатель: {stage}")
        require(len({r["task_id"] for r in records}) == len(records), "Повтор результатов")
        for record in records:
            task = tasks[record["task_id"]]
            require(all(record.get(k) == v for k, v in task.items()), "Ответ связан с изменённым входом")
            parsed = parse_answer(record["raw_response"], record["mode"], record["passed"])
            require(parsed["parse_status"] == record["parse_status"] and parsed["parsed"] == record["parsed"], "Разбор ответа не повторился")
            require(record["prompt_tokens"] <= 2048, "Превышен входной бюджет")
            if record["status"] == "completed":
                body = json.loads(record["raw_http_body"])
                require(body["choices"][0]["message"].get("content") == record["raw_response"], "Сырой ответ изменён")
                require(body["usage"] == record["usage"], "Изменены usage")
                require(record["usage"]["prompt_tokens"] == record["prompt_tokens"] and
                        record["usage"]["completion_tokens"] <= 128, "Не совпали серверные длины")
            parameters = record["request_parameters"]
            require(parameters["temperature"] == 0 and parameters["max_tokens"] <= 128 and not parameters["ignore_eos"], "Изменены условия генерации")
            require(math.isfinite(record["generation_s"]) and record["generation_s"] >= 0, "Некорректное время")
        runs[stage] = {r["claim_id"]: r for r in records}
    require(len(commits) == 1, "Серии выполнены на разных версиях кода")
    return digest, preparation, dataset, gold, runs, metadata, profile


def evaluate(gold, runs):
    scores, summaries = {}, {}
    for stage, records in runs.items():
        scored = [score_record(gold[i], record) for i, record in records.items()]
        expected = list(gold) if stage != "oracle" else [i for i in gold if gold[i]["evidence"]]
        summary = summarize(scored, expected)
        summary.update(correct=sum(s["label_correct"] for s in scored),
            raw_label_correct=sum(r["status"] == "completed" and raw_label(r) == claim_label(gold[i]) for i, r in records.items()),
            parse_failed=sum(r["parse_status"] != "ok" for r in records.values()),
            call_failed=sum(r["status"] != "completed" for r in records.values()),
            generation_seconds_total=sum(r["generation_s"] for r in records.values()),
            generation_median_s=statistics.median(r["generation_s"] for r in records.values()),
            mean_prompt_tokens=statistics.mean(r["prompt_tokens"] for r in records.values()),
            mean_passed_docs=statistics.mean(len(r["passed"]) for r in records.values()))
        summary["label_only_accuracy_diagnostic"] = summary["raw_label_correct"] / summary["n"]
        summaries[stage], scores[stage] = summary, {s["claim_id"]: s for s in scored}
    return scores, summaries


def paired_comparisons(scores):
    aggregate, details = [], []
    for before, after in (("llm_only", "rag_k3"), ("rag_k3", "rag_k1"), ("rag_k3", "rag_k5")):
        require(set(scores[before]) == set(scores[after]), "Попарно сравниваются разные утверждения")
        counts = Counter()
        for claim_id in sorted(scores[before]):
            a, b = scores[before][claim_id]["label_correct"], scores[after][claim_id]["label_correct"]
            kind = "both_correct" if a and b else "fixed" if b else "spoiled" if a else "both_wrong"
            counts[kind] += 1
            details.append({"before": before, "after": after, "claim_id": claim_id, "before_correct": a, "after_correct": b, "transition": kind})
        n = len(scores[before])
        aggregate.append({"before": before, "after": after, "n": n, **{key: counts[key] for key in ("fixed", "spoiled", "both_correct", "both_wrong")},
                          "delta_pp": 100 * (counts["fixed"] - counts["spoiled"]) / n})
    return aggregate, details


def oracle_diagnostics(gold, records):
    table = []
    for claim_id, record in records.items():
        answer = record["parsed"] if isinstance(record["parsed"], dict) else {}
        supplied = {i for doc in record["passed"] for i in doc["sentences"]}
        returned = answer.get("sentences", [])
        table.append({"claim_id": claim_id, "gold_label": claim_label(gold[claim_id]),
            "raw_label": raw_label(record), "label_only_correct": raw_label(record) == claim_label(gold[claim_id]),
            "parse_status": record["parse_status"], "parse_error": record["parse_error"],
            "passed": json.dumps(record["passed"]), "returned_doc_id": answer.get("doc_id"),
            "returned_sentences": json.dumps(returned),
            "exact_minus_one": bool(supplied) and set(returned) == {i - 1 for i in supplied}})
    invalid = [r for r in table if r["parse_status"] != "ok"]
    return table, {"n": len(table), "invalid_answers": len(invalid),
                   "invalid_with_correct_label": sum(r["label_only_correct"] for r in invalid),
                   "invalid_exact_minus_one": sum(r["exact_minus_one"] for r in invalid),
                   "raw_label_correct": sum(r["label_only_correct"] for r in table)}


def error_cases(gold, runs, scores, searches, documents, out):
    positive = [i for i, c in gold.items() if c["evidence"]]
    categories = {}
    for stage in ("rag_k1", "rag_k3", "rag_k5"):
        categories[stage] = {
            "not_found": sorted(i for i in positive if not scores[stage][i]["hit"]),
            "found_not_passed": sorted(i for i in positive if scores[stage][i]["hit"] and not scores[stage][i]["evidence_present"]),
            "passed_wrong": sorted(i for i in positive if scores[stage][i]["evidence_present"] and not scores[stage][i]["label_correct"]),
            "noinfo_wrong": sorted(i for i, c in gold.items() if not c["evidence"] and not scores[stage][i]["label_correct"]),
        }
    base = categories["rag_k3"]
    selected = []
    for kind in ("not_found", "found_not_passed", "passed_wrong", "noinfo_wrong"):
        candidates = base[kind]
        if candidates:
            # Для первого случая при наличии выбираем также неверную метку.
            errors = [i for i in candidates if not scores["rag_k3"][i]["label_correct"]]
            selected.append((kind, "rag_k3", (errors or candidates)[0]))
    if not base["found_not_passed"]:
        candidates = sorted(i for i, r in runs["oracle"].items() if r["parse_status"] != "ok" and raw_label(r) == claim_label(gold[i]))
        if candidates:
            selected.append(("oracle_numbering", "oracle", candidates[0]))
    cases = []
    text = ["# Разбор случаев ЛР2", "", "Выбор детерминированный: минимальный ID, удовлетворяющий категории.",
            "Категория found_not_passed может отсутствовать; случаи не создаются искусственно.", ""]
    titles = {"not_found": "Эталонный документ не найден", "found_not_passed": "Обоснование найдено, но не передано",
              "passed_wrong": "Обоснование передано, метка неверна", "noinfo_wrong": "Ошибка на NOINFO",
              "oracle_numbering": "Oracle: неверный номер при переданном эталоне"}
    for kind, stage, claim_id in selected[:4]:
        record = runs[stage][claim_id]
        passed_text = [{**p, "title": documents[p["doc_id"]]["title"],
                        "text": [{"number": i, "text": documents[p["doc_id"]]["abstract"][i]} for i in p["sentences"]]} for p in record["passed"]]
        case = {"category": kind, "stage": stage, "claim_id": claim_id, "claim": gold[claim_id]["claim"],
                "gold_label": claim_label(gold[claim_id]), "gold_evidence": gold[claim_id]["evidence"],
                "top5": searches[claim_id]["found"] if stage.startswith("rag") else [],
                "passed": passed_text, "raw_response": record["raw_response"], "parse_error": record["parse_error"],
                "score": scores[stage][claim_id]}
        cases.append(case)
        text += [f"## {titles[kind]} — ID {claim_id}", "", f"Утверждение: {case['claim']}",
                 f"Эталон: {case['gold_label']}; режим: {stage}.", "", "Найдено (top-5):", "```json", json.dumps(case["top5"], indent=2), "```", "Передано:"]
        for p in passed_text:
            text += [f"Документ {p['doc_id']}: {p['title']}"] + [f"[{s['number']}] {s['text']}" for s in p["text"]]
        text += ["", "Эталонные обоснования:", "```json", json.dumps(case["gold_evidence"], indent=2), "```",
                 "Сырой ответ:", "```json", record["raw_response"] or "null", "```",
                 f"Проверка: {record['parse_status']}; ошибка: {record['parse_error']}", ""]
    text += ["## Число случаев по категориям", "```json", json.dumps(categories, indent=2), "```"]
    (out / "case_studies.md").write_text("\n".join(text), encoding="utf-8")
    save_json(out / "case_studies.json", cases)
    return categories


def budget_check(source, gold, runs, searches, documents, budget, out):
    # Используем только кеш токенизатора Qwen, без загрузки весов и без BGE-поиска.
    os.environ["HF_HUB_OFFLINE"] = "1"
    for name in ("context.py", "prompts.json", "configs/lab2.json"):
        relative = name if name.startswith("configs/") else "lab2/" + name
        current = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        archived = (source / "inputs/provenance" / relative).read_bytes().replace(b"\r\n", b"\n")
        require(current == archived, f"Изменились условия сборки контекста: {relative}")
    from transformers import AutoTokenizer
    config = read_json(source / "inputs/provenance/configs/lab2.json")
    prompts = {key: "\n".join(lines) for key, lines in read_json(source / "inputs/provenance/lab2/prompts.json").items()}
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    original_budget = config["input_budget"]
    require(budget < original_budget, "Проверочный бюджет должен быть меньше исходного")
    # Повторяем упаковку context.py без импорта BGE и Sentence Transformers.
    # Точное совпадение всех исходных промптов проверяется перед пересчётом.
    def pack(claim, found, limit):
        def render(blocks):
            messages = [{"role": "system", "content": prompts["system"]},
                        {"role": "user", "content": prompts["rag"].format(claim=claim, context="\n\n".join(blocks))}]
            count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False))
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return {"prompt": text, "prompt_tokens": count}
        blocks, passed = [], []
        require(render(blocks)["prompt_tokens"] <= limit, "Инструкция не помещается в бюджет")
        for item in found[:3]:
            doc_id, chosen, stopped = item["doc_id"], [], False
            document = documents[doc_id]
            header = f"Document {document['doc_id']}\nTitle: {document['title']}"
            for index, sentence in enumerate(document["abstract"]):
                block = header + "\n" + "\n".join(f"[{i}] {document['abstract'][i]}" for i in chosen + [index])
                if render(blocks + [block])["prompt_tokens"] > limit:
                    stopped = True
                    break
                chosen.append(index)
            if chosen:
                blocks.append(header + "\n" + "\n".join(f"[{i}] {document['abstract'][i]}" for i in chosen))
                passed.append({"doc_id": doc_id, "sentences": chosen})
            if stopped:
                break
        return {**render(blocks), "passed": passed}
    records = []
    for position, claim_id in enumerate(sorted(gold), 1):
        claim = gold[claim_id]["claim"]
        original = pack(claim, searches[claim_id]["found"], original_budget)
        require(original["prompt"] == runs["rag_k3"][claim_id]["prompt"] and
                original["prompt_tokens"] == runs["rag_k3"][claim_id]["prompt_tokens"], "Исходный промпт не воспроизвёлся")
        reduced = pack(claim, searches[claim_id]["found"], budget)
        records.append({"claim_id": claim_id, "old_tokens": original["prompt_tokens"], "new_tokens": reduced["prompt_tokens"],
                        "old_evidence": contains_rationale(gold[claim_id], original["passed"]),
                        "new_evidence": contains_rationale(gold[claim_id], reduced["passed"]),
                        "new_passed": json.dumps(reduced["passed"])})
        if position % 30 == 0:
            print(f"Пересчёт бюджета {budget}: {position}/90", flush=True)
    eligible = [r for r in records if r["old_evidence"] is not None]
    require(all(not r["new_evidence"] or r["old_evidence"] for r in eligible), "Уменьшение бюджета неожиданно добавило обоснование")
    save_csv(out / "tables" / f"budget_{budget}.csv", records)
    return {"k": 3, "n": len(eligible), "old_budget": original_budget, "new_budget": budget,
            "old_evidence_count": sum(r["old_evidence"] for r in eligible),
            "new_evidence_count": sum(r["new_evidence"] for r in eligible), "new_model_calls": 0}


def plots(summaries, out):
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(out.parent / ".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    folder = out / "figures"
    folder.mkdir()
    names = ["llm_only", "rag_k1", "rag_k3", "rag_k5"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    values = [100 * summaries[s]["accuracy"] for s in names]
    bars = ax.bar(["LLM-only", "RAG k=1", "RAG k=3", "RAG k=5"], values, color=["#777777", "#4C78A8", "#4C78A8", "#4C78A8"])
    ax.set(ylim=(0, 100), ylabel="Доля верных ответов, %", title="Основная метрика с валидацией: 90 утверждений")
    for bar, stage, value in zip(bars, names, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 2, f"{summaries[stage]['correct']}/90\n{value:.1f}%", ha="center", fontsize=10)
    fig.tight_layout(); fig.savefig(folder / "quality.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for key, label, style in (("hit_at_k", "Hit@k", "o-"), ("recall_at_k", "Recall@k", "s-"),
                               ("mrr_at_k", "MRR@k", "D-"), ("evidence_coverage", "Обоснование передано", "x--")):
        ax.plot([1, 3, 5], [summaries[f"rag_k{k}"][key] for k in (1, 3, 5)], style, label=label)
    ax.set(xticks=[1, 3, 5], ylim=(0, 1), xlabel="Число найденных документов k", ylabel="Значение", title="Поиск и передача обоснования: 60 утверждений")
    ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(folder / "retrieval.png", dpi=200); plt.close(fig)
    oracle = summaries["oracle"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    values = [oracle["accuracy"], oracle["label_only_accuracy_diagnostic"], oracle["reference_accuracy"]]
    bars = ax.bar(["Основная метрика\nс валидацией", "Только метка\n(диагностика)", "Метка и полная\nэталонная ссылка"], [100*v for v in values], color=["#4C78A8", "#F58518", "#777777"])
    bars[1].set_hatch("//")
    ax.set(ylim=(0, 100), ylabel="Доля, %", title="Oracle: разные показатели на одних и тех же 60 утверждениях")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value*100+2, f"{round(value*60)}/60", ha="center")
    fig.tight_layout(); fig.savefig(folder / "oracle_diagnostics.png", dpi=200); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Локальный анализ результатов ЛР2")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/lab2_analysis")
    parser.add_argument("--budget-check", type=int)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-root", type=Path, default=ROOT / "lab2")
    parser.add_argument("--budget-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.budget_worker:
        # Этот процесс использует токенизатор и завершается до запуска MLflow.
        out = args.budget_worker
        source = out / "source"
        dataset = read_json(source / "inputs/provenance/data/manifest.json")["datasets"]["scifact"]
        gold = {c["id"]: c for c in rows(ROOT / dataset["files"]["claims_dev.jsonl"]["path"]) if c["id"] in dataset["splits"]["dev"]}
        documents = {str(d["doc_id"]): d for d in rows(ROOT / dataset["files"]["corpus.jsonl"]["path"])}
        searches = {r["claim_id"]: r for r in rows(source / "inputs/searches.jsonl")}
        run_id = read_json(source / "selection.json")["rag_k3"]
        runs = {"rag_k3": {r["claim_id"]: r for r in rows(source / "runs" / run_id / "results.jsonl")}}
        result = budget_check(source, gold, runs, searches, documents, args.budget_check, out)
        save_json(out / "budget_summary.json", result)
        return
    analysis_id = uuid.uuid4().hex
    out = args.output_root.resolve() / analysis_id
    out.mkdir(parents=True)
    source = out / "source"
    digest, preparation, dataset, gold, runs, metadata, profile = load_archive(args.archive.resolve(), source)
    scores, summaries = evaluate(gold, runs)
    pairs, pair_details = paired_comparisons(scores)
    positive = sorted(i for i, c in gold.items() if c["evidence"])
    tables = out / "tables"
    metrics = []
    for stage in STAGES:
        s = summaries[stage]
        metrics.append({"series": stage, "run_id": metadata[stage]["run_id"], "n": s["n"], "correct": s["correct"], "accuracy": s["accuracy"],
            **{f"recall_{label}": s["recall_by_class"][label] for label in ("SUPPORT", "CONTRADICT", "NOINFO")},
            "reference_accuracy": s["reference_accuracy"], "reference_n": s["reference_accuracy_n"],
            "parse_failed": s["parse_failed"], "call_failed": s["call_failed"], "label_only_accuracy_diagnostic": s["label_only_accuracy_diagnostic"],
            "mean_prompt_tokens": s["mean_prompt_tokens"], "mean_passed_docs": s["mean_passed_docs"], "generation_seconds_total": s["generation_seconds_total"]})
    save_csv(tables / "metrics.csv", metrics)
    retrieval = [{"k": k, "n": 60, **{key: summaries[f"rag_k{k}"][key] for key in (
        "hit_at_k", "recall_at_k", "mrr_at_k", "evidence_coverage", "mean_prompt_tokens", "mean_passed_docs")}} for k in (1, 3, 5)]
    save_csv(tables / "retrieval.csv", retrieval)
    common = [{"series": stage, "n": 60, "correct": sum(scores[stage][i]["label_correct"] for i in positive),
               "accuracy": sum(scores[stage][i]["label_correct"] for i in positive)/60,
               "label_only_accuracy_diagnostic": sum(runs[stage][i]["status"] == "completed" and raw_label(runs[stage][i]) == claim_label(gold[i]) for i in positive)/60}
              for stage in ("llm_only", "rag_k3", "oracle")]
    save_csv(tables / "common60.csv", common)
    save_csv(tables / "paired.csv", pairs)
    save_csv(tables / "pair_details.csv", pair_details)
    prediction_rows = [{"series": stage, **score, "raw_label_diagnostic": raw_label(runs[stage][i]),
                        "prompt_tokens": runs[stage][i]["prompt_tokens"], "generation_s": runs[stage][i]["generation_s"]}
                       for stage in STAGES for i, score in scores[stage].items()]
    save_csv(tables / "predictions.csv", prediction_rows)
    diagnostics, diagnostic_summary = oracle_diagnostics(gold, runs["oracle"])
    save_csv(tables / "oracle_diagnostics.csv", diagnostics)
    corpus_raw = (ROOT / dataset["files"]["corpus.jsonl"]["path"]).read_bytes()
    require(sha256(corpus_raw) == dataset["files"]["corpus.jsonl"]["sha256"], "Изменился корпус")
    documents = {str(d["doc_id"]): d for d in map(json.loads, corpus_raw.decode("utf-8").splitlines())}
    searches = {r["claim_id"]: r for r in rows(source / "inputs/searches.jsonl")}
    categories = error_cases(gold, runs, scores, searches, documents, out)
    summary = {"analysis_id": analysis_id, "archive_sha256": digest,
               "preparation_commit": preparation["git_commit"], "execution_commit": metadata["llm_only"]["execution_commit"],
               "series": summaries, "common60": common, "paired": pairs,
               "oracle_diagnostics": diagnostic_summary, "error_categories": categories,
               "accuracy_policy": "completed call and parse_status=ok; other planned answers count as incorrect",
               "diagnostic_policy": "label-only metric uses all planned records and does not replace primary accuracy"}
    if args.budget_check:
        subprocess.run([sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--archive", str(args.archive.resolve()),
                        "--budget-worker", str(out), "--budget-check", str(args.budget_check)], check=True)
        summary["budget_check"] = read_json(out / "budget_summary.json")
    plots(summaries, out)
    shutil.copyfile(source / "inputs/provenance/forecasts/lab2.md", out / "forecast_original.md")
    provenance = out / "provenance"
    provenance.mkdir()
    shutil.copyfile(Path(__file__), provenance / "analyze.py")
    importer = Path(__file__).with_name("import_mlflow_local.py")
    if importer.exists():
        shutil.copyfile(importer, provenance / importer.name)
    save_json(out / "summary.json", summary)
    save_json(out / "analysis_metadata.json", {"analysis_id": analysis_id, "status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_archive": str(args.archive.resolve()), "source_archive_sha256": digest,
        "analysis_code_sha256": sha256(Path(__file__).read_bytes()), "server_profile": profile,
        "local_versions": {name: version(name) for name in ("numpy", "matplotlib", "PyYAML", "transformers", "mlflow")},
        "new_model_calls": 0, "mlflow_root": str(args.mlflow_root.resolve())})
    save_json(args.output_root.resolve() / "latest.json", {"analysis_id": analysis_id, "path": str(out)})
    if args.mlflow:
        from import_mlflow_local import register
        register(out, args.mlflow_root.resolve())
    print("СЕРИЯ | ВЕРНО | ACCURACY | ОТКЛОНЕНО ВАЛИДАТОРОМ")
    for stage in STAGES:
        s = summaries[stage]
        print(f"{stage}: {s['correct']}/{s['n']} | {s['accuracy']:.2%} | {s['parse_failed']}")
    print("Oracle, только метка — диагностика:", f"{diagnostic_summary['raw_label_correct']}/60")
    print("Результаты анализа:", out)
    print("Сводка:", out / "summary.json")
    print("Новых вызовов Qwen: 0")


if __name__ == "__main__":
    main()
