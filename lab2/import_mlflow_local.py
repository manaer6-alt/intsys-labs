# Перенос уже сохранённых результатов ЛР2 в локальный MLflow.
# Здесь не вызывается Qwen и не меняются исходные ответы из Kaggle.
# Для каждой облачной серии создаётся запись с исходным run_id в тегах.
# Повторный импорт той же серии из того же архива не создаёт её копию.
# Отдельный запуск analysis хранит общие таблицы, графики и разбор ошибок.
# База: lab2/mlflow_local.db; файлы MLflow: lab2/mlartifacts/.

import argparse
import csv
import json
import math
from pathlib import Path

from prepare_data import ROOT, json_bytes


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def register(analysis_path, store_path):
    import mlflow

    analysis_path, store_path = Path(analysis_path).resolve(), Path(store_path).resolve()
    meta = read(analysis_path / "analysis_metadata.json")
    summary = read(analysis_path / "summary.json")
    source = analysis_path / "source"
    selection = read(source / "selection.json")
    store_path.mkdir(parents=True, exist_ok=True)
    artifact_root = store_path / "mlartifacts"
    artifact_root.mkdir(exist_ok=True)
    # Один и тот же абсолютный путь применяется для импорта и просмотра.
    tracking_uri = "sqlite:///" + (store_path / "mlflow_local.db").as_posix()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("lr2_scifact")
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(
        "lr2_scifact", artifact_location=artifact_root.as_uri())
    tags = {"lab": "2", "source_archive_sha256": summary["archive_sha256"],
            "execution_commit": summary["execution_commit"], "preparation_commit": summary["preparation_commit"]}
    profile = meta["server_profile"]
    mappings = []

    for stage in ("check", "llm_only", "rag_k1", "rag_k3", "rag_k5", "oracle"):
        source_id = selection[stage]
        existing = mlflow.search_runs(experiment_ids=[experiment_id], output_format="list", max_results=1,
            filter_string=f"tags.source_run_id = '{source_id}' AND tags.source_archive_sha256 = '{summary['archive_sha256']}'")
        if existing:
            run_id = existing[0].info.run_id
            print(f"MLflow: {stage} уже импортирован — {run_id}", flush=True)
        else:
            folder = source / "runs" / source_id
            operational = read(folder / "metrics.json")
            with mlflow.start_run(experiment_id=experiment_id, run_name=f"{stage}-{source_id[:8]}",
                                  tags={**tags, "series": stage, "source_run_id": source_id}) as run:
                run_id = run.info.run_id
                mlflow.log_params({"source_run_id": source_id, "model": profile["model"],
                                   "revision": profile["model_revision"], "temperature": 0, "max_tokens": 128,
                                   "input_budget": 2048, "new_model_calls_during_import": 0})
                # Служебные метрики не подменяют пересчитанное качество.
                metrics = {f"execution_{key}": value for key, value in operational.items()
                           if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)}
                if stage in summary["series"]:
                    scored = summary["series"][stage]
                    metrics.update({key: value for key, value in scored.items()
                                    if isinstance(value, (int, float)) and math.isfinite(value)})
                    metrics.update({f"recall_{label}": value for label, value in scored["recall_by_class"].items() if value is not None})
                    mlflow.set_tag("accuracy_policy", summary["accuracy_policy"])
                    mlflow.set_tag("label_only_is_diagnostic", "true")
                mlflow.log_metrics(metrics)
                mlflow.log_artifacts(str(folder), artifact_path="source_run")
                print(f"MLflow: импортирован {stage} — {run_id}", flush=True)
        mappings.append({"series": stage, "source_run_id": source_id, "mlflow_run_id": run_id})

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"analysis-{summary['analysis_id'][:8]}",
                          tags={**tags, "series": "analysis", "analysis_id": summary["analysis_id"]}) as run:
        mlflow.log_params({"analysis_id": summary["analysis_id"], "new_model_calls": 0,
                           "analysis_code_sha256": meta["analysis_code_sha256"]})
        for pair in summary["paired"]:
            prefix = pair["before"] + "_to_" + pair["after"]
            mlflow.log_metrics({f"{prefix}_{key}": pair[key] for key in ("fixed", "spoiled", "delta_pp")})
        if "budget_check" in summary:
            mlflow.log_metrics({f"budget_{key}": value for key, value in summary["budget_check"].items()})
        for name in ("tables", "figures", "provenance"):
            mlflow.log_artifacts(str(analysis_path / name), artifact_path=name)
        mlflow.log_artifacts(str(source / "inputs/provenance"), artifact_path="preparation_provenance")
        for name in ("summary.json", "analysis_metadata.json", "case_studies.md", "case_studies.json", "forecast_original.md"):
            mlflow.log_artifact(str(analysis_path / name))
        archive = Path(meta["source_archive"])
        if archive.is_file():
            mlflow.log_artifact(str(archive), artifact_path="original_archive")
        mappings.append({"series": "analysis", "source_run_id": summary["analysis_id"], "mlflow_run_id": run.info.run_id})

    result = {"tracking_uri": tracking_uri, "experiment": "lr2_scifact", "runs": mappings}
    (analysis_path / "mlflow_import.json").write_bytes(json_bytes(result))
    with (analysis_path / "tables/mlflow_runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mappings[0]))
        writer.writeheader()
        writer.writerows(mappings)
    print("MLflow: эксперимент lr2_scifact; база", store_path / "mlflow_local.db", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Импорт готового анализа ЛР2 в локальный MLflow")
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=ROOT / "lab2")
    args = parser.parse_args()
    register(args.analysis, args.store)


if __name__ == "__main__":
    main()
