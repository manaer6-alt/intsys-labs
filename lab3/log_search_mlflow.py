"""Регистрирует сохранённые запуски поиска ЛР3 в локальном MLflow."""

import argparse
import csv
import json
from pathlib import Path

from search_dev import ROOT


def log_run(folder, experiment_id, stage):
    import mlflow

    metadata = json.loads((folder / "metadata.json").read_bytes())
    source_id = folder.name
    previous = mlflow.search_runs(
        experiment_ids=[experiment_id], output_format="list",
        filter_string=f"tags.source_run_id = '{source_id}' AND tags.stage = '{stage}' AND attributes.status = 'FINISHED'",
    )
    if previous:
        return previous[0].info.run_id
    metrics = {}
    for filename, key in (("retrieval.csv", "method"), ("timings.csv", "stage")):
        with (folder / filename).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                prefix = row.pop(key)
                metrics.update({f"{prefix}.{name.replace('@', '_at_')}": float(value) for name, value in row.items()})
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{stage}-{source_id[:8]}",
                          tags={"stage": stage, "source_run_id": source_id}) as run:
        mlflow.log_params({k: metadata[k] for k in ("commit", "config_sha256", "split_sha256", "qwen_calls")})
        if "model_revision" in metadata:
            mlflow.log_params({"ce_model": metadata["config"]["model"], "ce_revision": metadata["model_revision"],
                               "ce_device": metadata["config"]["device"], "ce_pairs": metadata["pairs"]})
        mlflow.log_metrics(metrics)
        for filename in ("retrieval.csv", "timings.csv", "complementarity.csv", "metadata.json", "traces.jsonl", "searches.jsonl"):
            if (folder / filename).is_file():
                mlflow.log_artifact(str(folder / filename))
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-run", type=Path, required=True)
    parser.add_argument("--rerank-run", type=Path)
    args = parser.parse_args()
    import mlflow

    store = ROOT / "lab3"
    mlflow.set_tracking_uri("sqlite:///" + (store / "mlflow_local.db").as_posix())
    experiment = mlflow.get_experiment_by_name("lr3_scifact")
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(
        "lr3_scifact", artifact_location=(store / "mlartifacts").as_uri(),
    )
    for folder, stage in ((args.search_run, "search_dev"), (args.rerank_run, "rerank_dev")):
        if folder:
            print(f"MLflow lr3_scifact / {stage}-{folder.name[:8]}: {log_run(folder, experiment_id, stage)}")


if __name__ == "__main__":
    main()
