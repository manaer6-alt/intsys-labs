"""Эксперимент 2: оценка cross-encoder на dev и подготовка 180 запросов."""

import argparse
import io
import json
import math
import random
import statistics
import sys
import time
import uuid
import zipfile
from importlib.metadata import version
from pathlib import Path

from search_dev import ROOT, git, write_csv

sys.path.insert(0, str(ROOT / "lab2"))
from build_index import read_json
from evaluation import contains_rationale, load_gold, retrieval_metrics
from prepare_data import MANIFEST, json_bytes, sha256


def checked_file(folder, name, meta):
    raw = (folder / name).read_bytes()
    if sha256(raw) != meta[name + "_sha256"]:
        raise ValueError(f"Хеш {name} отличается от метаданных")
    return raw


def load_stage1(folder, gold, split_sha):
    meta = read_json(folder / "metadata.json")
    if meta["split_sha256"] != split_sha or meta["dense_top5_matched"] != len(gold):
        raise ValueError("Результат этапа 1 не соответствует dev")
    rows = [json.loads(line) for line in checked_file(folder, "searches.jsonl", meta).splitlines()]
    by_id = {row["claim_id"]: row for row in rows}
    if len(rows) != len(gold) or set(by_id) != set(gold):
        raise ValueError("Нужна ровно одна выдача на каждый ID dev")
    for claim_id, row in by_id.items():
        ids = [item["doc_id"] for item in row["rrf"]]
        if row["claim"] != gold[claim_id]["claim"] or len(ids) != 20 or len(set(ids)) != 20:
            raise ValueError(f"Некорректные кандидаты RRF, ID {claim_id}")
    return meta, by_id


def rerank(candidates, scores):
    if len(scores) != len(candidates) or not all(math.isfinite(score) for score in scores):
        raise ValueError("Некорректные оценки cross-encoder")
    paired = sorted(zip(candidates, scores), key=lambda pair: (-pair[1], int(pair[0]["doc_id"])))
    return [{"doc_id": row["doc_id"], "score": score, "rank": rank,
             "rrf_rank": row["rank"], "rrf_score": row["score"]}
            for rank, (row, score) in enumerate(paired, 1)]


class CrossEncoder:
    def __init__(self, config):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.config = config
        torch.set_num_threads(config["threads"])
        torch.use_deterministic_algorithms(True)
        snapshot = Path(snapshot_download(
            config["model"], revision=config["revision"],
            allow_patterns=["config.json", "model.safetensors", "tokenizer*", "vocab.txt", "special_tokens_map.json"],
        ))
        self.revision = snapshot.name
        if len(self.revision) != 40 or not self.revision.startswith(config["revision"]):
            raise ValueError("Не удалось подтвердить ревизию cross-encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot)
        self.model = AutoModelForSequenceClassification.from_pretrained(snapshot, use_safetensors=True)
        self.model.to(config["device"]).eval()

    def score(self, claim, texts):
        scores = []
        step = self.config["batch_size"]
        with self.torch.inference_mode():
            for start in range(0, len(texts), step):
                docs = texts[start:start + step]
                batch = self.tokenizer([claim] * len(docs), docs, truncation="only_second",
                                       max_length=self.config["max_length"], padding=True, return_tensors="pt")
                logits = self.model(**batch.to(self.config["device"])).logits
                if tuple(logits.shape) != (len(docs), 1):
                    raise ValueError("Cross-encoder должен вернуть одну оценку на пару")
                scores.extend(logits[:, 0].tolist())
        return scores


def read_lab2(archive_path, expected_sha):
    raw = archive_path.read_bytes()
    if sha256(raw) != expected_sha:
        raise ValueError("Нужен тот же архив ЛР2, что использовал этап 1")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = ["configs/lab2.json", "lab2/prompts.json", "lab2/answer_schema.json", "lab2/context.py"]
        for name in names:
            old = archive.read("inputs/provenance/" + name).replace(b"\r\n", b"\n")
            if old != (ROOT / name).read_bytes().replace(b"\r\n", b"\n"):
                raise ValueError(f"Изменились условия ЛР2: {name}")
        inputs = [json.loads(line) for line in archive.read("inputs/tasks.jsonl").splitlines()]
        source_id = json.loads(archive.read("selection.json"))["rag_k3"]
        warmups = archive.read("inputs/warmups.jsonl")
    return {row["claim_id"]: row for row in inputs if row["mode"] == "rag" and row["k"] == 3}, source_id, warmups


def compare(gold, traces):
    rows = []
    for method in ("rrf", "rrf_ce"):
        positives = [x for x in traces if gold[x["claim_id"]]["evidence"]]
        metrics = {"method": method, "n": len(positives)}
        for depth in (3, 20):
            values = [retrieval_metrics(gold[x["claim_id"]], x[method], depth) for x in positives]
            for field, label in (("hit", "hit"), ("recall", "recall"), ("rr", "mrr")):
                metrics[f"{label}@{depth}"] = statistics.mean(row[field] for row in values)
        metrics["evidence_coverage"] = statistics.mean(
            contains_rationale(gold[x["claim_id"]], x["passed"][method]) for x in positives
        )
        rows.append(metrics)
    if rows[0]["recall@20"] != rows[1]["recall@20"]:
        raise ValueError("Переранжирование не должно менять Recall@20")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-run", type=Path, required=True)
    parser.add_argument("--lab2-archive", type=Path, default=ROOT / "lab2/lab2_final_2fb415a1.zip")
    args = parser.parse_args()
    git("ls-files", "--error-unmatch", "lab3/rerank_dev.py", "configs/lab3_rerank.json", "forecasts/lab3.md")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Сначала закоммитьте изменения отслеживаемых файлов")
    commit = git("rev-parse", "HEAD")
    config_path = ROOT / "configs/lab3_rerank.json"
    config = read_json(config_path)
    dataset = read_json(MANIFEST)["datasets"]["scifact"]
    gold = load_gold()
    meta, by_id = load_stage1(args.search_run, gold, dataset["split_sha256"])
    baseline, baseline_id, warmups = read_lab2(args.lab2_archive, meta["lab2_archive_sha256"])
    if set(baseline) != set(gold):
        raise ValueError("Задания Dense из ЛР2 не соответствуют dev")
    from context import PromptBuilder

    source = dataset["files"]["corpus.jsonl"]
    corpus = (ROOT / source["path"]).read_bytes()
    if sha256(corpus) != source["sha256"]:
        raise ValueError("Хеш корпуса отличается от манифеста")
    documents = {str(d["doc_id"]): d for d in (json.loads(line) for line in corpus.splitlines() if line.strip())}
    texts = {key: d["title"] + "\n" + " ".join(d["abstract"]) for key, d in documents.items()}
    builder = PromptBuilder()
    ce = CrossEncoder(config)
    warm_claim = next(json.loads(line)["claim"] for line in warmups.splitlines())
    ce.score(warm_claim, list(texts.values())[:20])

    traces, tasks = [], []
    ids = sorted(gold)
    random.Random(builder.config["seed"]).shuffle(ids)
    for index, claim_id in enumerate(ids, 1):
        row = by_id[claim_id]
        claim, candidates = row["claim"], row["rrf"]
        start = time.perf_counter()
        reordered = rerank(candidates, ce.score(claim, [texts[d["doc_id"]] for d in candidates]))
        ce_s = time.perf_counter() - start
        if {d["doc_id"] for d in reordered} != {d["doc_id"] for d in candidates}:
            raise ValueError("Изменился состав кандидатов")
        dense = builder.pack(claim, row["dense"], documents, 3)
        if any(dense[key] != baseline[claim_id][key] for key in ("messages", "prompt", "prompt_tokens", "passed")):
            raise ValueError(f"Запрос Dense не воспроизвёлся, ID {claim_id}")
        trace = {**row, "rrf_ce": reordered, "passed": {"dense": dense["passed"]}, "ce_s": ce_s}
        for method in ("rrf", "rrf_ce"):
            packed = builder.pack(claim, trace[method], documents, 3)
            trace["passed"][method] = packed["passed"]
            search_s = row["timing_s"]["rrf_total_sequential"] + (ce_s if method == "rrf_ce" else 0)
            tasks.append({"task_id": f"{method}:3:{claim_id}", "claim_id": claim_id, "claim": claim,
                          "series": method, "search_s": search_s, **packed})
        traces.append(trace)
        if index % 10 == 0:
            print(f"Переранжирование и подготовка: {index}/90", flush=True)

    table = compare(gold, traces)
    timings = []
    # RRF измерен на первом этапе, CE измерен отдельно. Сумма — оценка последовательного пути.
    for stage, values in (("ce", [x["ce_s"] for x in traces]),
                          ("rrf_ce_sequential_estimate", [x["ce_s"] + x["timing_s"]["rrf_total_sequential"] for x in traces])):
        timings.append({"stage": stage, "n": len(values), "median_ms": statistics.median(values)*1000,
                        "min_ms": min(values)*1000, "max_ms": max(values)*1000})
    out = ROOT / "runs/lab3_rerank_dev" / uuid.uuid4().hex
    out.mkdir(parents=True)
    for name, rows in (("tasks.jsonl", tasks), ("traces.jsonl", traces)):
        (out / name).write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))
    (out / "warmups.jsonl").write_bytes(warmups)
    (out / "generation_config.json").write_bytes(json_bytes(builder.config))
    (out / "answer_schema.json").write_bytes((ROOT / "lab2/answer_schema.json").read_bytes())
    write_csv(out / "retrieval.csv", table)
    write_csv(out / "timings.csv", timings)
    metadata = {"run_id": out.name, "commit": commit, "config": config, "model_revision": ce.revision,
                "config_sha256": sha256(config_path.read_bytes()), "lab2_archive_sha256": meta["lab2_archive_sha256"],
                "split_sha256": meta["split_sha256"], "source_search_run": args.search_run.name,
                "source_searches_sha256": meta["searches.jsonl_sha256"], "dense_source_run_id": baseline_id,
                "dense_prompts_matched": len(traces), "pairs": sum(len(x["rrf"]) for x in traces),
                "tasks": len(tasks), "qwen_calls": 0,
                "versions": {name: version(name) for name in ("torch", "transformers", "huggingface-hub")},
                "files_sha256": {p.name: sha256(p.read_bytes()) for p in out.iterdir()}}
    (out / "metadata.json").write_bytes(json_bytes(metadata))
    path = out / "lab3_dev_inputs.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("tasks.jsonl", "warmups.jsonl", "generation_config.json", "answer_schema.json", "metadata.json"):
            archive.write(out / name, name)
    path.with_suffix(".zip.sha256").write_text(sha256(path.read_bytes()) + "  " + path.name + "\n", encoding="ascii")
    for row in table:
        print(json.dumps(row, ensure_ascii=False))
    print(f"Пар CE: {metadata['pairs']}; совпало промптов Dense: {metadata['dense_prompts_matched']}/90; заданий: {len(tasks)}")
    print(f"Результаты: {out}\nАрхив подготовленных запросов: {path}")


if __name__ == "__main__":
    main()
