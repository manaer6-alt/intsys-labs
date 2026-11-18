"""ЛР3, опыт 3. Готовим запросы Dense и RRF + MiniLM на holdout."""

import argparse
import json
import random
import subprocess
import time
import zipfile
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from rerank_dev import CrossEncoder, read_lab2, rerank
from search_dev import ROOT, dense_search, git
from search_methods import BM25, rrf

from context import PromptBuilder
from prepare_data import MANIFEST, json_bytes, sha256
from retrieval import DenseRetriever

SELECTION_PATH = "lab3/selection.json"
METHODS = ("dense", "rrf_ce")
CONFIGS = ("configs/lab3_search.json", "configs/lab3_rerank.json", "configs/lab2.json")
INPUTS = ("tasks.jsonl", "warmups.jsonl", "generation_config.json", "answer_schema.json",
          "selection.json", "metadata.json")


def read_verified(relative_path, expected_sha):
    raw = (ROOT / relative_path).read_bytes()
    if sha256(raw) != expected_sha:
        raise ValueError(f"Не совпадает SHA-256: {relative_path}")
    return raw


def load_selection():
    # Выбор должен быть сохранён в Git до чтения holdout.
    frozen_commit = git("log", "-1", "--format=%H", "--", SELECTION_PATH)
    if not frozen_commit:
        raise ValueError("Сначала закоммитьте выбор метода в lab3/selection.json")
    committed = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{frozen_commit}:{SELECTION_PATH}"]
    )
    raw = (ROOT / SELECTION_PATH).read_bytes()
    if raw.replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
        raise ValueError("Рабочая версия selection.json отличается от коммита")
    choice = json.loads(raw)
    if (choice["selection_split"], choice["baseline"], choice["selected_method"], choice["k"]) != (
        "dev", *METHODS, 3
    ):
        raise ValueError("Этот запуск подготовлен для зафиксированного сравнения Dense и RRF + MiniLM")
    return frozen_commit, raw, choice


def load_claims(dataset, split_sha, smoke_claim_id=None):
    if dataset["split_sha256"] != split_sha:
        raise ValueError("Разбиение отличается от зафиксированного")
    split = json.loads(read_verified(dataset["split_file"], split_sha))
    if split != dataset["splits"] or len(split["holdout"]) != 210 or len(set(split["holdout"])) != 210:
        raise ValueError("Ожидаются исходные 210 ID holdout")
    if smoke_claim_id is None:
        ids, source = set(split["holdout"]), dataset["files"]["claims_dev.jsonl"]
    else:
        ids, source = {smoke_claim_id}, dataset["files"]["claims_train.jsonl"]

    # На этапе сборки используем только ID и текст. Эталон понадобится при оценке.
    claims = {}
    for line in read_verified(source["path"], source["sha256"]).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        claim_id = row["id"]
        if claim_id in ids:
            if claim_id in claims:
                raise ValueError(f"Повтор ID утверждения: {claim_id}")
            claims[claim_id] = row["claim"]
    if set(claims) != ids:
        raise ValueError("Не все запланированные утверждения найдены")
    return claims


def build_rows(claims, dense, ce, builder, search, k, warm_claim):
    depth = search["depth"]
    dense.numeric_ids = [int(doc_id) for doc_id in dense.doc_ids]
    texts = {key: doc["title"] + "\n" + " ".join(doc["abstract"])
             for key, doc in dense.documents.items()}
    lexical = BM25(texts, **search["bm25"])

    def lookup(claim):
        dense_rank, timings = dense_search(dense, claim, depth)
        start = time.perf_counter()
        bm25_rank = lexical.search(claim, depth)
        lexical_s = time.perf_counter() - start
        start = time.perf_counter()
        fused = rrf(dense_rank, bm25_rank, depth, search["rrf_constant"])
        merge_s = time.perf_counter() - start
        if len(fused) != depth:
            raise ValueError("RRF не вернул ожидаемые 20 кандидатов")
        start = time.perf_counter()
        reordered = rerank(fused, ce.score(claim, [texts[doc["doc_id"]] for doc in fused]))
        rerank_s = time.perf_counter() - start
        return {"dense": dense_rank, "bm25": bm25_rank, "rrf": fused, "rrf_ce": reordered,
                "timing_s": {**timings, "bm25": lexical_s, "rrf_merge": merge_s, "ce": rerank_s,
                             "rrf_ce": timings["dense"] + lexical_s + merge_s + rerank_s}}

    lookup(warm_claim)
    ids = sorted(claims)
    random.Random(builder.config["seed"]).shuffle(ids)
    tasks, traces = [], []
    for index, claim_id in enumerate(ids, 1):
        claim = claims[claim_id]
        trace = {"claim_id": claim_id, "claim": claim, **lookup(claim), "passed": {}}
        for method in METHODS:
            packed = builder.pack(claim, trace[method], dense.documents, k)
            trace["passed"][method] = packed["passed"]
            tasks.append({"task_id": f"{method}:{k}:{claim_id}", "claim_id": claim_id,
                          "claim": claim, "series": method, "search_s": trace["timing_s"][method],
                          **packed})
        traces.append(trace)
        if index % 10 == 0 or index == len(ids):
            print(f"Подготовлено утверждений: {index}/{len(ids)}", flush=True)
    return tasks, traces


def save_bundle(out, tasks, traces, warmups, builder, selection_raw, metadata):
    out.mkdir(parents=True)
    for name, rows in (("tasks.jsonl", tasks), ("traces.jsonl", traces)):
        (out / name).write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                         for row in rows).encode("utf-8"))
    (out / "warmups.jsonl").write_bytes(warmups)
    (out / "generation_config.json").write_bytes(json_bytes(builder.config))
    (out / "answer_schema.json").write_bytes((ROOT / "lab2/answer_schema.json").read_bytes())
    (out / "selection.json").write_bytes(selection_raw)
    metadata["files_sha256"] = {path.name: sha256(path.read_bytes()) for path in out.iterdir()}
    (out / "metadata.json").write_bytes(json_bytes(metadata))

    archive_path = out / "lab3_holdout_inputs.zip"
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in INPUTS:
            archive.write(out / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Повреждён архив подготовленных запросов")
    digest = sha256(archive_path.read_bytes())
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{digest}  {archive_path.name}\n", encoding="ascii"
    )
    return archive_path, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-prepared", type=Path, required=True)
    parser.add_argument("--lab2-archive", type=Path, default=ROOT / "lab2/lab2_final_2fb415a1.zip")
    parser.add_argument("--smoke", action="store_true", help="Два пробных промпта из claims_train")
    args = parser.parse_args()

    frozen_commit, selection_raw, selection = load_selection()
    if not args.smoke:
        git("ls-files", "--error-unmatch", "lab3/prepare_holdout.py")
        if git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError("Закоммитьте изменения отслеживаемых файлов перед основным запуском")
    commit = git("rev-parse", "HEAD")
    configs = {name: json.loads(read_verified(name, selection["config_sha256"][name]))
               for name in CONFIGS}
    search, ce_config, generation = (configs[name] for name in CONFIGS)
    dataset = json.loads(MANIFEST.read_bytes().decode("utf-8-sig"))["datasets"]["scifact"]

    dev = json.loads((args.dev_prepared / "metadata.json").read_bytes())
    if (dev["run_id"] != selection["source_runs"]["rerank_dev"]["run_id"]
            or dev["split_sha256"] != selection["split_sha256"]):
        raise ValueError("Укажите папку подготовки dev, использованную при выборе метода")
    _, _, warmups = read_lab2(args.lab2_archive, dev["lab2_archive_sha256"])
    warm_task = json.loads(warmups.splitlines()[0])

    # Сверяем условия сборки с архивом ЛР2 и локальными входами dev.
    builder = PromptBuilder()
    config_bytes = json_bytes(builder.config)
    if builder.config != generation or sha256(config_bytes) != dev["files_sha256"]["generation_config.json"]:
        raise ValueError("Условия генерации отличаются от dev")
    schema_raw = (ROOT / "lab2/answer_schema.json").read_bytes()
    if sha256(schema_raw) != dev["files_sha256"]["answer_schema.json"]:
        raise ValueError("Схема ответа отличается от dev")

    claims = load_claims(dataset, selection["split_sha256"],
                         warm_task["claim_id"] if args.smoke else None)
    dense = DenseRetriever(search["dense_config"])
    ce = CrossEncoder(ce_config)
    if ce.revision != dev["model_revision"]:
        raise ValueError("Ревизия MiniLM отличается от dev")
    tasks, traces = build_rows(claims, dense, ce, builder, search, selection["k"], warm_task["claim"])
    if git("rev-parse", "HEAD") != commit:
        raise ValueError("Коммит изменился во время подготовки")
    subset = "train_smoke" if args.smoke else "holdout"
    out = ROOT / "runs" / f"lab3_prepare_{subset}" / uuid4().hex
    meta = {"run_id": out.name, "commit": commit, "selection_commit": frozen_commit,
            "selection_sha256": sha256(selection_raw), "split": subset,
            "split_sha256": selection["split_sha256"], "config_sha256": selection["config_sha256"],
            "baseline": METHODS[0], "selected_method": METHODS[1], "k": selection["k"],
            "model_revision_ce": ce.revision, "source_dev_run": dev["run_id"],
            "lab2_archive_sha256": dev["lab2_archive_sha256"],
            "claims": len(claims), "tasks": len(tasks), "qwen_calls": 0,
            "versions": {name: version(name) for name in ("torch", "transformers", "huggingface-hub")}}
    if args.smoke:
        print(f"Короткая проверка пройдена: {len(claims)} train-утверждение, {len(tasks)} промпта.")
        print("Архив не создавался. Вызовов Qwen: 0.")
        return
    archive, digest = save_bundle(out, tasks, traces, warmups, builder, selection_raw, meta)
    print(f"Утверждений holdout: {len(claims)}; запросов: {len(tasks)}; вызовов Qwen: 0.")
    print("Каталог подготовки:", out)
    print("Архив:", archive)
    print("SHA-256:", digest)


if __name__ == "__main__":
    main()
