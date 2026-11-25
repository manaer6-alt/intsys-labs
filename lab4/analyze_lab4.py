"""Recalculate and optionally import completed, recorded Lab 4 series."""

import argparse
import csv
import hashlib
import io
import json
import statistics
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODES = ("react", "plan_then_execute")
METRICS = ("model_calls", "tool_calls", "input_tokens", "output_tokens", "seconds")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document(path, result):
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_series(path, expected_n):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():
            raise ValueError(f"Damaged archive: {path}")
        meta = json.loads(archive.read("metadata.json"))
        ids = [f"user_task_{n}" for n in meta["task_numbers"]]
        raw = archive.read("episodes.csv").decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(raw)))
        expected = {(task, mode) for task in ids for mode in MODES}
        if len(ids) != expected_n or len(rows) != 2 * expected_n:
            raise ValueError(f"Unexpected episode count: {path}")
        if {(r["task_id"], r["mode"]) for r in rows} != expected:
            raise ValueError(f"Unexpected episode IDs: {path}")
        results = []
        for row in rows:
            name = f"{row['task_id']}_{row['mode']}.json"
            trace = json.loads(archive.read(name))
            if row["utility"] not in ("True", "False") or (row["utility"] == "True") != trace["utility"]:
                raise ValueError(f"Utility mismatch: {name}")
            if not all(float(row[key]) == float(trace[key]) for key in METRICS):
                raise ValueError(f"Metric mismatch: {name}")
            model_steps = trace["model_trace"]
            tools = trace["tool_trace"]
            if len(tools) != trace["tool_calls"]:
                raise ValueError(f"Tool count mismatch: {name}")
            if sum(not item.get("preflight_only", False) for item in model_steps) != trace["model_calls"]:
                raise ValueError(f"Model count mismatch: {name}")
            for saved, actual in (
                (trace["input_tokens"], sum(t.get("usage", {}).get("prompt_tokens", 0) for t in model_steps)),
                (trace["output_tokens"], sum(t.get("usage", {}).get("completion_tokens", 0) for t in model_steps)),
            ):
                if saved != actual:
                    raise ValueError(f"Token count mismatch: {name}")
            result = {key: trace[key] for key in ("task_id", "mode", "utility", *METRICS, "stop_reason")}
            result["tool_errors"] = sum(bool(t.get("error")) for t in tools)
            result["plan_steps"] = None
            plans = [t for t in model_steps if t["phase"] == "plan" and t.get("message")]
            if plans:
                try:
                    result["plan_steps"] = len(json.loads(plans[0]["message"]["content"])["steps"])
                except (ValueError, KeyError, TypeError):
                    pass
            result["tool_sequence"] = " | ".join(t["name"] for t in tools)
            results.append(result)
        return meta, results


def stats(rows):
    values = {"episodes": len(rows), "success": sum(row["utility"] for row in rows)}
    if rows:
        values["success_rate"] = values["success"] / len(rows)
        for key in METRICS:
            data = [row[key] for row in rows]
            values[key] = {"median": statistics.median(data), "min": min(data), "max": max(data)}
    return values


def summarize(rows):
    output = {}
    for mode in MODES:
        group = [r for r in rows if r["mode"] == mode]
        output[mode] = stats(group)
        output[mode]["successful"] = stats([r for r in group if r["utility"]])
        output[mode]["failed"] = stats([r for r in group if not r["utility"]])
        output[mode]["episodes_with_tool_errors"] = sum(r["tool_errors"] > 0 for r in group)
        plans = [r["plan_steps"] for r in group if r["plan_steps"] is not None]
        if plans:
            output[mode]["plan_length"] = {"median": statistics.median(plans),
                                            "min": min(plans), "max": max(plans)}
    return output


def pair_rows(rows):
    by_id = {(r["task_id"], r["mode"]): r["utility"] for r in rows}
    result = []
    for task in sorted({r["task_id"] for r in rows}, key=lambda s: int(s.rsplit("_", 1)[1])):
        react, plan = (by_id[(task, mode)] for mode in MODES)
        group = "both" if react and plan else "react_only" if react else "plan_only" if plan else "neither"
        result.append({"task_id": task, "react": react, "plan_then_execute": plan, "group": group})
    return result


def import_mlflow(series_dir, stage, source_hash, summary):
    import mlflow

    store = ROOT / "lab4"
    mlflow.set_tracking_uri("sqlite:///" + (store / "mlflow_local.db").as_posix())
    existing = mlflow.get_experiment_by_name("lr4_agentdojo")
    experiment_id = existing.experiment_id if existing else mlflow.create_experiment(
        "lr4_agentdojo", artifact_location=(store / "mlartifacts").as_uri())
    tag = f"tags.archive_sha256 = '{source_hash}' AND tags.stage = '{stage}' AND attributes.status = 'FINISHED'"
    previous = mlflow.search_runs(experiment_ids=[experiment_id], output_format="list", filter_string=tag)
    if previous:
        return previous[0].info.run_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{stage}-{source_hash[:8]}",
                          tags={"stage": stage, "archive_sha256": source_hash}) as run:
        for mode, values in summary["methods"].items():
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{mode}.{key}", value)
                elif isinstance(value, dict) and "median" in value:
                    for subkey, number in value.items():
                        mlflow.log_metric(f"{mode}.{key}_{subkey}", number)
        mlflow.log_metrics({f"paired.{key}": value for key, value in summary["pairs"].items()})
        mlflow.log_artifacts(str(series_dir))
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("debug-v1", "debug-v2", "final"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    stages = (("debug_v1", args.debug_v1, 8), ("debug_v2", args.debug_v2, 8), ("holdout_subset", args.final, 12))
    args.out.mkdir(parents=True, exist_ok=False)
    combined = {}
    for stage, path, n in stages:
        metadata, rows = read_series(path, n)
        out = args.out / stage
        out.mkdir()
        pairs = pair_rows(rows)
        summary = {"source_sha256": digest(path), "task_count": n, "task_ids": metadata["task_numbers"],
                   "methods": summarize(rows),
                   "pairs": {g: sum(r["group"] == g for r in pairs)
                             for g in ("both", "react_only", "plan_only", "neither")}}
        table(out / "episodes.csv", rows)
        table(out / "paired.csv", pairs)
        document(out / "summary.json", summary)
        document(out / "metadata.json", metadata)
        combined[stage] = summary
        print(stage, {mode: f"{summary['methods'][mode]['success']}/{n}" for mode in MODES},
              summary["pairs"])
        if args.mlflow:
            print("MLflow:", import_mlflow(out, stage, summary["source_sha256"], summary))
    document(args.out / "summary.json", combined)
    print("Saved:", args.out.resolve())


if __name__ == "__main__":
    main()
