# Импорт скачанных результатов Kaggle в MLflow на ноутбуке.
# GPU, vLLM и сервер Qwen для этого не нужны. Исходную базу Kaggle не изменяем.
import argparse
import json
from pathlib import Path
import mlflow


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    analysis = root / "analysis" / read(root / "latest_analysis.json")["analysis_id"]
    summary = read(analysis / "summary.json")
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow_local.db').as_posix()}")
    if mlflow.get_experiment_by_name("lr1_kaggle_local") is None:
        mlflow.create_experiment("lr1_kaggle_local", artifact_location=(root / "local_mlartifacts").as_uri())
    mlflow.set_experiment("lr1_kaggle_local")
    index_path = root / "mlflow_local_imports.json"
    index = read(index_path) if index_path.exists() else {}
    sources = {stage: root / stage / run_id for stage, run_id in summary["selected_runs"].items()}
    sources["analysis"] = analysis
    for stage, folder in sources.items():
        if folder.name in index:
            continue
        with mlflow.start_run(run_name=f"{stage}-{folder.name[:8]}") as run:
            mlflow.set_tags({"stage": stage, "source_run_id": folder.name,
                             "platform": "Kaggle", "logging_mode": "imported_on_local_machine"})
            if stage != "analysis":
                meta = read(folder / "metadata.json")
                config = meta["config"]
                mlflow.set_tag("commit", meta["commit"])
                mlflow.log_params({"model": config["model"], "revision": config["revision"],
                                   "seed": config["seed"], "prefix_caching": meta["server"]["prefix_caching"]})
                for i, line in enumerate((folder / "results.jsonl").read_text(encoding="utf-8").splitlines(), 1):
                    row = json.loads(line)
                    if row["status"] == "completed":
                        mlflow.log_metrics({k: row[k] for k in ["ttft_s", "tpot_s", "e2e_s"] if row.get(k) is not None}, step=i)
            mlflow.log_artifacts(str(folder), artifact_path="analysis" if stage == "analysis" else "run")
            index[folder.name] = run.info.run_id
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    mlflow.search_runs(max_results=10000).to_csv(root / "mlflow_local_runs.csv", index=False, encoding="utf-8")
    print("Импорт завершён. База:", root / "mlflow_local.db")


if __name__ == "__main__":
    main()
