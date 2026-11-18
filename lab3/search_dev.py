"""ЛР3, эксперимент 1: поиск на dev и сравнение с выдачей ЛР2."""

import argparse
import csv
import hashlib
import io
import json
import statistics
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path

from search_methods import BM25, rrf

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("dense", "bm25", "rrf")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def read_baseline(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.read_text(encoding="utf-8-sig").split()[0].lower() != digest:
        raise ValueError("Хеш архива ЛР2 не совпал")
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("inputs/searches.jsonl")
        metadata = json.loads(archive.read("inputs/metadata.json"))
        if hashlib.sha256(raw).hexdigest() != metadata["files_sha256"]["searches.jsonl"]:
            raise ValueError("Хеш выдачи ЛР2 не совпал")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    result = {row["claim_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Повторяющиеся ID в выдаче ЛР2")
    return result, digest


def dense_search(retriever, claim, depth):
    import numpy as np

    from build_index import check_vectors

    # Те же параметры кодирования, что в ЛР2. При равенстве сортируем по числовому ID.
    start = time.perf_counter()
    vector = retriever.model.encode(
        [retriever.config["query_prefix"] + claim], prompt="", normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float32, copy=False)
    check_vectors(vector, 1, retriever.config["dimension"])
    encoded = time.perf_counter()
    scores = retriever.matrix @ vector[0]
    order = np.lexsort((retriever.numeric_ids, -scores))[:depth]
    found = [{"doc_id": retriever.doc_ids[i], "score": float(scores[i]), "rank": rank}
             for rank, i in enumerate(order, 1)]
    end = time.perf_counter()
    return found, {"dense_encoding": encoded - start, "dense_ranking": end - encoded, "dense": end - start}


def write_csv(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8-sig", newline="")


def evaluate(records, gold, metric):
    positives = [r for r in records if gold[r["claim_id"]]["evidence"]]
    if len(positives) != 60:
        raise ValueError("Ожидалось 60 утверждений с эталонными документами")
    tables, groups = [], []
    for method in METHODS:
        table = {"method": method, "n": len(positives)}
        for depth in (3, 20):
            scores = [metric(gold[r["claim_id"]], r[method], depth) for r in positives]
            for key, name in (("hit", "hit"), ("recall", "recall"), ("rr", "mrr")):
                table[f"{name}@{depth}"] = statistics.mean(s[key] for s in scores)
        tables.append(table)
    for depth in (3, 20):
        for name, flags in (("both", (True, True)), ("dense_only", (True, False)),
                            ("bm25_only", (False, True)), ("neither", (False, False))):
            subset = []
            for r in positives:
                relevant = gold[r["claim_id"]]["evidence"]
                hits = tuple(bool(set(relevant) & {d["doc_id"] for d in r[m][:depth]}) for m in ("dense", "bm25"))
                if hits == flags:
                    subset.append(r)
            groups.append({"depth": depth, "group": name, "n": len(subset),
                           "rrf_hit_at_3": sum(metric(gold[r["claim_id"]], r["rrf"], 3)["hit"] for r in subset),
                           "claim_ids": " ".join(str(r["claim_id"]) for r in subset)})
    return tables, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab2-archive", type=Path, default=ROOT / "lab2/lab2_final_2fb415a1.zip")
    args = parser.parse_args()
    required = ["configs/lab3_search.json", "forecasts/lab3.md", "lab3/search_dev.py", "lab3/search_methods.py"]
    git("ls-files", "--error-unmatch", *required)
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Сначала закоммитьте изменения отслеживаемых файлов")
    commit = git("rev-parse", "HEAD")
    config_bytes = (ROOT / required[0]).read_bytes()
    config = json.loads(config_bytes)
    baseline, baseline_sha = read_baseline(args.lab2_archive)

    # Импортируем ЛР2 без изменения её файлов.
    sys.path.insert(0, str(ROOT / "lab2"))
    from evaluation import load_gold, retrieval_metrics
    from prepare_data import MANIFEST, sha256
    from retrieval import DenseRetriever

    gold = load_gold()
    if set(gold) != set(baseline):
        raise ValueError("ID dev отличаются от ЛР2")
    dataset = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))["datasets"]["scifact"]
    depth = config["depth"]
    dense = DenseRetriever(config["dense_config"])
    dense.numeric_ids = [int(doc_id) for doc_id in dense.doc_ids]
    texts = {doc_id: doc["title"] + "\n" + " ".join(doc["abstract"]) for doc_id, doc in dense.documents.items()}
    lexical = BM25(texts, **config["bm25"])
    warmup = dataset["files"]["claims_train.jsonl"]
    warmup_raw = (ROOT / warmup["path"]).read_bytes()
    if sha256(warmup_raw) != warmup["sha256"]:
        raise ValueError("Хеш набора прогрева не совпал")
    train_claim = json.loads(warmup_raw.splitlines()[0])["claim"]
    warmup_dense, _ = dense_search(dense, train_claim, depth)
    rrf(warmup_dense, lexical.search(train_claim, depth), depth, config["rrf_constant"])

    records = []
    for claim_id, row in sorted(gold.items()):
        claim = row["claim"]
        d, dense_timing = dense_search(dense, claim, depth)
        if [x["doc_id"] for x in d[:5]] != [x["doc_id"] for x in baseline[claim_id]["found"]]:
            raise ValueError(f"Первая пятёрка Dense отличается от ЛР2, ID {claim_id}")
        start = time.perf_counter()
        b = lexical.search(claim, depth)
        bm25_s = time.perf_counter() - start
        start = time.perf_counter()
        fusion = rrf(d, b, depth, config["rrf_constant"])
        rrf_s = time.perf_counter() - start
        records.append({"claim_id": claim_id, "claim": claim, "dense": d, "bm25": b, "rrf": fusion,
                        "timing_s": {**dense_timing, "bm25": bm25_s, "rrf_merge": rrf_s,
                                     "rrf_total_sequential": dense_timing["dense"] + bm25_s + rrf_s}})

    table, groups = evaluate(records, gold, retrieval_metrics)
    timings = []
    for stage in records[0]["timing_s"]:
        values = [r["timing_s"][stage] * 1000 for r in records]
        timings.append({"stage": stage, "n": len(values), "median_ms": statistics.median(values),
                        "min_ms": min(values), "max_ms": max(values)})
    out = ROOT / "runs" / "lab3_search_dev" / uuid.uuid4().hex
    out.mkdir(parents=True)
    (out / "searches.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    write_csv(out / "retrieval.csv", table)
    write_csv(out / "complementarity.csv", groups)
    write_csv(out / "timings.csv", timings)
    metadata = {"commit": commit, "config_sha256": sha256(config_bytes), "split_sha256": dataset["split_sha256"],
                "forecast_sha256": sha256((ROOT / required[1]).read_bytes()),
                "lab2_archive_sha256": baseline_sha, "dense_top5_matched": len(records),
                "query_count": len(records), "qwen_calls": 0,
                **{name + "_sha256": sha256((out / name).read_bytes()) for name in ("searches.jsonl", "retrieval.csv", "complementarity.csv", "timings.csv")}}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Dense: первые пять совпали для {len(records)}/{len(records)} утверждений")
    for row in table:
        print(row["method"], f"Hit@3={row['hit@3']:.4f} Recall@3={row['recall@3']:.4f} MRR@3={row['mrr@3']:.4f} Recall@20={row['recall@20']:.4f}")
    print(f"Результаты: {out}")


if __name__ == "__main__":
    main()
