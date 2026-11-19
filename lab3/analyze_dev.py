"""ЛР3: локальная оценка ответов dev и регистрация итогов в MLflow."""

import argparse
import csv
import json
import statistics
import sys
import zipfile
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab2"))
from evaluation import load_gold, score_record, summarize
from prepare_data import MANIFEST, sha256

METHODS = ("dense", "rrf", "rrf_ce")
PAIRS = (("dense", "rrf"), ("dense", "rrf_ce"), ("rrf", "rrf_ce"))
TASK_FIELDS = ("task_id", "claim_id", "claim", "mode", "k", "prompt", "messages",
               "prompt_tokens", "found", "passed")


def read_rows(raw):
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def checked(raw, expected, name):
    if sha256(raw) != expected:
        raise ValueError(f"Не совпал SHA-256: {name}")
    return raw


def load_run(archive, series):
    run_id = json.loads(archive.read("selection.json"))[series]
    prefix = f"runs/{run_id}/"
    meta = json.loads(archive.read(prefix + "metadata.json"))
    if meta["run_id"] != run_id or meta["series"] != series or meta["status"] != "completed":
        raise ValueError(f"Серия {series} не завершена")
    raw = checked(archive.read(prefix + "results.jsonl"), meta["results_sha256"], series)
    return meta, read_rows(raw)


def score_series(gold, records, planned, method):
    plan = {t["claim_id"]: t for t in planned}
    if len(plan) != len(planned) or set(plan) != set(gold):
        raise ValueError(f"План должен содержать ровно 90 утверждений dev: {method}")
    detail = {}
    for row in records:
        claim_id = row["claim_id"]
        task = plan[claim_id]
        if claim_id in detail or any(row[k] != task[k] for k in TASK_FIELDS):
            raise ValueError(f"Повтор или несоответствие входа: {method}, ID {claim_id}")
        if row["claim"] != gold[claim_id]["claim"]:
            raise ValueError(f"Текст не соответствует SciFact: ID {claim_id}")
        scored = score_record(gold[claim_id], row)
        if scored["parse_status"] != row["parse_status"]:
            raise ValueError(f"Валидаторы расходятся: {method}, ID {claim_id}")
        detail[claim_id] = {"method": method, **scored, "claim": row["claim"],
                            "status": row["status"], "parse_error": row.get("parse_error"),
                            "prompt_tokens": row["prompt_tokens"],
                            "generation_s": row["generation_s"]}
    stats = summarize(detail.values(), gold)
    for name, key in (("correct", "label_correct"), ("references_correct", "reference_correct"),
                      ("evidence_present", "evidence_present")):
        stats[name] = sum(row[key] is True for row in detail.values())
    stats["parse_failed"] = sum(row["parse_status"] != "ok" for row in detail.values())
    stats["call_failed"] = sum(row["status"] != "completed" for row in detail.values())
    stats["generation_median_s"] = statistics.median(row["generation_s"] for row in detail.values())
    stats["generation_total_s"] = sum(row["generation_s"] for row in detail.values())
    return detail, stats


def pair_details(gold, scored, first, second):
    rows = []
    for claim_id in sorted(gold):
        before, after = scored[first][claim_id], scored[second][claim_id]
        delta = int(after["label_correct"]) - int(before["label_correct"])
        coverage_before, coverage_after = before["evidence_present"], after["evidence_present"]
        rows.append({"from_method": first, "to_method": second, "claim_id": claim_id,
                     "gold_label": before["gold_label"], "before_label": before["predicted_label"],
                     "after_label": after["predicted_label"], "accuracy_delta": delta,
                     "coverage_before": coverage_before, "coverage_after": coverage_after})
    return rows


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    gold = load_gold()
    input_meta = (args.prepared / "metadata.json").read_bytes()
    prep = json.loads(input_meta)
    manifest = json.loads(MANIFEST.read_bytes().decode("utf-8-sig"))["datasets"]["scifact"]
    if prep["split_sha256"] != manifest["split_sha256"] or prep["dense_prompts_matched"] != len(gold):
        raise ValueError("Подготовленные входы не соответствуют текущему dev или Dense")
    sidecar = args.archive.with_suffix(".zip.sha256")
    archive_bytes = args.archive.read_bytes()
    archive_sha = sha256(archive_bytes)
    if sidecar.is_file():
        checked(archive_bytes, sidecar.read_text(encoding="utf-8-sig").split()[0].lower(),
                args.archive.name)
    checked(args.lab2_archive.read_bytes(), prep["lab2_archive_sha256"], args.lab2_archive.name)
    runs, details, summaries = {}, {}, {}
    with zipfile.ZipFile(args.archive) as cloud, zipfile.ZipFile(args.lab2_archive) as lab2:
        if cloud.testzip() is not None:
            raise ValueError("Повреждён архив Kaggle")
        if cloud.read("inputs/metadata.json") != input_meta:
            raise ValueError("Метаданные Kaggle и локального запуска различаются")
        for name in ("tasks.jsonl", "warmups.jsonl", "generation_config.json", "answer_schema.json"):
            checked(cloud.read("inputs/" + name), prep["files_sha256"][name], name)
        tasks = read_rows(cloud.read("inputs/tasks.jsonl"))
        check_meta, check_rows = load_run(cloud, "check")
        warmups = read_rows(cloud.read("inputs/warmups.jsonl"))
        checks = read_rows(checked(
            cloud.read(f"runs/{check_meta['run_id']}/token_checks.jsonl"),
            check_meta["token_checks_sha256"], "token_checks.jsonl",
        ))
        lengths = {task["task_id"]: task["prompt_tokens"] for task in tasks + warmups}
        if (len(check_rows) != len(warmups) or any(row["parse_status"] != "ok" for row in check_rows)
                or len(checks) != len(lengths)
                or {row["task_id"]: row["actual"] for row in checks} != lengths):
            raise ValueError("Проверка токенизации или пробных ответов не пройдена")

        dense_meta, dense = load_run(lab2, "rag_k3")
        if dense_meta["run_id"] != prep["dense_source_run_id"]:
            raise ValueError("Несовпадение источника Dense")
        plan_dense = [t for t in read_rows(lab2.read("inputs/tasks.jsonl"))
                      if t["mode"] == "rag" and t["k"] == 3]
        profile = json.loads(cloud.read("current_server.json"))["profile"]
        old_profile = dense_meta["server"]["profile"]
        if {k: v for k, v in profile.items() if k != "gpu"} != {
                k: v for k, v in old_profile.items() if k != "gpu"}:
            raise ValueError("Программные настройки или тип ускорителя отличаются от базы ЛР2")
        details["dense"], summaries["dense"] = score_series(gold, dense, plan_dense, "dense")
        runs["dense"] = dense_meta["run_id"]
        for method in METHODS[1:]:
            meta, rows = load_run(cloud, method)
            if meta["approval"] != check_meta["approval"] or meta["execution_commit"] != check_meta["execution_commit"]:
                raise ValueError(f"Проверка и серия {method} выполнены в разных условиях")
            planned = [t for t in tasks if t["series"] == method]
            details[method], summaries[method] = score_series(gold, rows, planned, method)
            runs[method] = meta["run_id"]

    comparisons = []
    all_pairs = []
    for first, second in PAIRS:
        rows = pair_details(gold, details, first, second)
        all_pairs.extend(rows)
        improved = sum(row["accuracy_delta"] == 1 for row in rows)
        worsened = sum(row["accuracy_delta"] == -1 for row in rows)
        coverage_added = sum(row["coverage_before"] is False and row["coverage_after"] is True for row in rows)
        coverage_lost = sum(row["coverage_before"] is True and row["coverage_after"] is False for row in rows)
        comparisons.append({"from_method": first, "to_method": second, "fixed": improved,
                            "spoiled": worsened, "net": improved - worsened,
                            "accuracy_delta_pp": 100 * (improved - worsened) / len(rows),
                            "coverage_added": coverage_added, "coverage_lost": coverage_lost})

    result = {"archive_sha256": archive_sha, "sidecar_verified": sidecar.is_file(),
              "preparation_run": prep["run_id"], "split_sha256": prep["split_sha256"],
              "execution_commit": check_meta["execution_commit"], "source_runs": runs,
              "model": profile["model"], "model_revision": profile["model_revision"],
              "gpu": profile["gpu"], "dense_gpu": old_profile["gpu"],
              "series": summaries, "paired": comparisons}
    return result, [row for method in METHODS for row in details[method].values()], all_pairs


def log_mlflow(out, result, *, stage="analysis_dev"):
    import mlflow

    store = ROOT / "lab3"
    mlflow.set_tracking_uri("sqlite:///" + (store / "mlflow_local.db").as_posix())
    experiment = mlflow.get_experiment_by_name("lr3_scifact")
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(
        "lr3_scifact", artifact_location=(store / "mlartifacts").as_uri(),
    )
    digest = result["archive_sha256"]
    previous = mlflow.search_runs(
        experiment_ids=[experiment_id], output_format="list",
        filter_string=f"tags.archive_sha256 = '{digest}' AND tags.stage = '{stage}' AND attributes.status = 'FINISHED'",
    )
    if previous:
        return previous[0].info.run_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{stage}-{digest[:8]}",
                          tags={"archive_sha256": digest, "stage": stage}) as run:
        mlflow.log_params({key: result[key] for key in
                           ("preparation_run", "split_sha256", "execution_commit", "model", "model_revision")})
        mlflow.log_params({f"{name}_source_run": run_id for name, run_id in result["source_runs"].items()})
        if "selection_commit" in result:
            mlflow.log_param("selection_commit", result["selection_commit"])
        for method, stats in result["series"].items():
            values = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
            values.update({f"recall_{label}": value for label, value in stats["recall_by_class"].items()})
            mlflow.log_metrics({f"{method}.{key}": value for key, value in values.items()})
        for pair in result["paired"]:
            name = f"{pair['from_method']}_to_{pair['to_method']}"
            mlflow.log_metrics({f"{name}.{key}": value for key, value in pair.items()
                                if isinstance(value, (int, float))})
        for row in result.get("retrieval", ()):
            mlflow.log_metrics({f"{row['method']}.{key}_at_{row['k']}": row[key]
                                for key in ("hit", "recall", "mrr")})
        artifact_dir = "dev_analysis" if stage == "analysis_dev" else "holdout_analysis"
        mlflow.log_artifacts(str(out), artifact_path=artifact_dir)
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--lab2-archive", type=Path, default=ROOT / "lab2/lab2_final_2fb415a1.zip")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    result, predictions, pairs = analyze(args)
    out = args.out or ROOT / "runs" / "lab3_analysis_dev" / uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_csv(out / "predictions.csv", predictions)
    save_csv(out / "paired.csv", result["paired"])
    save_csv(out / "pair_details.csv", pairs)
    summary = [
        {"method": name, **{k: v for k, v in stats.items() if not isinstance(v, dict)},
         **{f"recall_{k}": v for k, v in stats["recall_by_class"].items()}}
        for name, stats in result["series"].items()
    ]
    save_csv(out / "series.csv", summary)
    for name, stats in result["series"].items():
        print(f"{name}: {stats['correct']}/{stats['n']} | Accuracy {stats['accuracy']:.2%} | "
              f"ссылки {stats['references_correct']}/{stats['reference_accuracy_n']} | "
              f"обоснование передано {stats['evidence_present']}/{stats['evidence_coverage_n']}")
    print("Результаты:", out.resolve())
    if args.mlflow:
        print("MLflow lr3_scifact:", log_mlflow(out, result))


if __name__ == "__main__":
    main()
