"""ЛР3: сравнение Dense и RRF + MiniLM на отложенной выборке SciFact."""

import argparse
import json
import statistics
import zipfile
from pathlib import Path
from uuid import uuid4

from analyze_dev import (
    MANIFEST, ROOT, checked, load_run, log_mlflow, pair_details, read_rows,
    save_csv, score_series, sha256,
)
from evaluation import retrieval_metrics
from prepare_data import claim_label

METHODS = ("dense", "rrf_ce")


def load_holdout(split_sha):
    dataset = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))["datasets"]["scifact"]
    split_path = ROOT / dataset["split_file"]
    split = json.loads(checked(split_path.read_bytes(), split_sha, split_path.name))
    ids = set(split["holdout"])
    if split_sha != dataset["split_sha256"] or split != dataset["splits"] or len(ids) != 210:
        raise ValueError("Разбиение holdout отличается от исходного")
    source = dataset["files"]["claims_dev.jsonl"]
    raw = checked((ROOT / source["path"]).read_bytes(), source["sha256"], "claims_dev.jsonl")
    rows = [r for r in read_rows(raw) if r["id"] in ids]
    if len(rows) != len(ids) or {r["id"] for r in rows} != ids or any(claim_label(r) is None for r in rows):
        raise ValueError("Неполная или противоречивая разметка holdout")
    return {r["id"]: r for r in rows}


def evaluate(archive_path, prepared):
    raw_archive = archive_path.read_bytes()
    archive_sha = sha256(raw_archive)
    sidecar = archive_path.with_suffix(".zip.sha256")
    if sidecar.is_file():
        checked(raw_archive, sidecar.read_text(encoding="utf-8-sig").split()[0].lower(), archive_path.name)
    meta_raw = (prepared / "metadata.json").read_bytes()
    prep = json.loads(meta_raw)
    if (prep["split"], prep["baseline"], prep["selected_method"], prep["k"], prep["tasks"]) != (
        "holdout", *METHODS, 3, 420
    ):
        raise ValueError("Нужна основная подготовка holdout")
    gold = load_holdout(prep["split_sha256"])
    traces = read_rows(checked((prepared / "traces.jsonl").read_bytes(),
                               prep["files_sha256"]["traces.jsonl"], "traces.jsonl"))
    by_id = {row["claim_id"]: row for row in traces}
    if len(traces) != len(gold) or set(by_id) != set(gold):
        raise ValueError("Трассировка не соответствует ID holdout")
    scored, stats, runs = {}, {}, {}
    retrieval, timing = [], []
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or archive.read("inputs/metadata.json") != meta_raw:
            raise ValueError("Повреждён архив или отличается запуск подготовки")
        inputs = {name: checked(archive.read(f"inputs/{name}"), prep["files_sha256"][name], name)
                  for name in ("tasks.jsonl", "warmups.jsonl", "generation_config.json",
                               "answer_schema.json", "selection.json")}
        selection = json.loads(inputs["selection.json"])
        if (selection["baseline"], selection["selected_method"]) != METHODS:
            raise ValueError("Архив не соответствует выбору, зафиксированному до holdout")
        config = json.loads(inputs["generation_config.json"])
        tasks = read_rows(inputs["tasks.jsonl"])
        warmups = read_rows(inputs["warmups.jsonl"])
        expected_tokens = {r["task_id"]: r["prompt_tokens"] for r in tasks + warmups}
        check, probes = load_run(archive, "check")
        token_checks = read_rows(checked(
            archive.read(f"runs/{check['run_id']}/token_checks.jsonl"),
            check["token_checks_sha256"], "token_checks.jsonl",
        ))
        if (len(tasks) != 420 or len(expected_tokens) != 422 or len(token_checks) != 422
                or {r["task_id"]: r["actual"] for r in token_checks} != expected_tokens
                or any(r["actual"] != r["expected"] for r in token_checks)
                or len(probes) != 2 or any(r["status"] != "completed" or r["parse_status"] != "ok" for r in probes)):
            raise ValueError("Проверка токенизации или прогрев не прошли")
        server = json.loads(archive.read("current_server.json"))
        profile = server["profile"]
        if (profile["model"], profile["model_revision"]) != (config["model"], config["revision"]):
            raise ValueError("Модель сервера отличается от конфигурации")

        for method in METHODS:
            meta, records = load_run(archive, method)
            if (meta["inputs_metadata_sha256"] != sha256(meta_raw) or meta["approval"] != check["approval"]
                    or meta["execution_commit"] != check["execution_commit"]):
                raise ValueError(f"Условия запуска и проверки различаются: {method}")
            plan = [t for t in tasks if t["series"] == method]
            for task in plan:
                trace = by_id[task["claim_id"]]
                if (task["found"] != trace[method][:3] or task["passed"] != trace["passed"][method]
                        or task["search_s"] != trace["timing_s"][method]):
                    raise ValueError(f"Промпт не соответствует трассировке: {task['task_id']}")
            scored[method], stats[method] = score_series(gold, records, plan, method)
            runs[method] = meta["run_id"]
            positive = [(gold[row["claim_id"]], row[method]) for row in traces
                        if gold[row["claim_id"]]["evidence"]]
            for depth in (3, 20):
                measures = [retrieval_metrics(claim, found, depth) for claim, found in positive]
                retrieval.append({"method": method, "k": depth, "n": len(measures),
                                  "hit": statistics.mean(row["hit"] for row in measures),
                                  "recall": statistics.mean(row["recall"] for row in measures),
                                  "mrr": statistics.mean(row["rr"] for row in measures)})
            stats[method]["search_median_s"] = statistics.median(t["search_s"] for t in plan)
            stats[method]["search_plus_generation_median_s"] = statistics.median(
                r["search_s"] + r["generation_s"] for r in records)
            timing.append({"method": method, **{key: value for key, value in stats[method].items()
                                                if key.endswith("_s")}})
    pairs = pair_details(gold, scored, *METHODS)
    fixed = sum(r["accuracy_delta"] == 1 for r in pairs)
    spoiled = sum(r["accuracy_delta"] == -1 for r in pairs)
    summary = {"archive_sha256": archive_sha, "sidecar_verified": sidecar.is_file(),
               "preparation_run": prep["run_id"], "split_sha256": prep["split_sha256"],
               "selection_commit": prep["selection_commit"], "execution_commit": check["execution_commit"],
               "model": profile["model"], "model_revision": profile["model_revision"], "gpu": profile["gpu"],
               "source_runs": runs, "series": stats, "retrieval": retrieval,
               "paired": [{"from_method": METHODS[0], "to_method": METHODS[1],
                           "fixed": fixed, "spoiled": spoiled, "net": fixed - spoiled,
                           "accuracy_delta_pp": 100 * (fixed - spoiled) / len(gold),
                           "coverage_added": sum(r["coverage_before"] is False and r["coverage_after"] is True for r in pairs),
                           "coverage_lost": sum(r["coverage_before"] is True and r["coverage_after"] is False for r in pairs)}]}
    details = [row for method in METHODS for row in scored[method].values()]
    return summary, details, pairs, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    summary, details, pairs, timing = evaluate(args.archive, args.prepared)
    out = args.out or ROOT / "runs" / "lab3_analysis_holdout" / uuid4().hex
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_csv(out / "predictions.csv", details)
    save_csv(out / "paired.csv", summary["paired"])
    save_csv(out / "pair_details.csv", pairs)
    save_csv(out / "retrieval.csv", summary["retrieval"])
    save_csv(out / "timings.csv", timing)
    save_csv(out / "series.csv", [
        {"method": name, **{k: v for k, v in stats.items() if isinstance(v, (int, float))},
         **{f"recall_{k}": v for k, v in stats["recall_by_class"].items()}}
        for name, stats in summary["series"].items()
    ])
    for name, stats in summary["series"].items():
        print(f"{name}: {stats['correct']}/{stats['n']} | Accuracy {stats['accuracy']:.2%} | "
              f"ссылки {stats['references_correct']}/{stats['reference_accuracy_n']} | "
              f"обоснование {stats['evidence_present']}/{stats['evidence_coverage_n']}")
    print("Результаты:", out.resolve())
    if args.mlflow:
        print("MLflow lr3_scifact:", log_mlflow(out, summary, stage="analysis_holdout"))


if __name__ == "__main__":
    main()
