"""Validate optional Lab 4 results and import recorded metrics into MLflow."""

import argparse
import csv
import hashlib
import io
import json
import statistics
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("baseline", "truncate_1024", "prefix_cache")
MODES = ("react", "plan_then_execute")
TASKS = (0, 4, 8, 12, 20, 28, 32, 36)


def metrics(rows):
    return {
        "episodes": len(rows), "success": sum(r["utility"] == "True" for r in rows),
        "success_rate": sum(r["utility"] == "True" for r in rows) / len(rows),
        "input_tokens_total": sum(int(r["input_tokens"]) for r in rows),
        "output_tokens_total": sum(int(r["output_tokens"]) for r in rows),
        "input_tokens_median": statistics.median(int(r["input_tokens"]) for r in rows),
        "output_tokens_median": statistics.median(int(r["output_tokens"]) for r in rows),
        "seconds_total": sum(float(r["seconds"]) for r in rows),
        "seconds_median": statistics.median(float(r["seconds"]) for r in rows),
        "truncated_calls": sum(int(r["truncated_calls"]) for r in rows),
        "chars_saved": sum(int(r["result_chars"]) - int(r["sent_chars"]) for r in rows),
    }


def counter(archive, path, name):
    data = archive.read(path).decode("utf-8")
    values = [float(line.rsplit(" ", 1)[1]) for line in data.splitlines() if line.startswith(name + "{")]
    if not values:
        return None
    return int(sum(values))


def analyze(path):
    summary, tables, sources, pair_tables = {}, {}, {}, {}
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():
            raise ValueError("Archive damaged")
        server_off = json.loads(archive.read("server_off.json"))
        server_on = json.loads(archive.read("server_on.json"))
        flags = ("--no-enable-prefix-caching", "--enable-prefix-caching")
        if (server_off["gpu"] != server_on["gpu"] or
                server_off["revision"] != server_on["revision"] or
                flags[0] not in server_off["command"] or flags[1] not in server_on["command"] or
                [x for x in server_off["command"] if x not in flags] != [x for x in server_on["command"] if x not in flags]):
            raise ValueError("Server conditions do not match")
        expected = {(f"user_task_{n}", mode) for n in TASKS for mode in MODES}
        first = None
        for c in CONDITIONS:
            metadata = json.loads(archive.read(f"{c}/metadata.json"))
            if metadata["condition"] != c:
                raise ValueError("Different condition label")
            if first is None:
                first = metadata
            for key in ("frozen_sha256", "config", "prompts", "task_numbers"):
                if metadata[key] != first[key]:
                    raise ValueError(f"Changed frozen setting: {key}")
            raw = archive.read(f"{c}/episodes.csv")
            rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
            if len(rows) != 16 or {(r["task_id"], r["mode"]) for r in rows} != expected:
                raise ValueError(f"Invalid episode IDs: {c}")
            for row in rows:
                trace = json.loads(archive.read(f"{c}/{row['task_id']}_{row['mode']}.json"))
                if (row["utility"] == "True") != trace["utility"]:
                    raise ValueError("Utility mismatch")
                if trace["truncated_calls"] != sum(t["truncated"] for t in trace["tool_trace"]):
                    raise ValueError("Truncation count mismatch")
                if not all(float(trace[k]) == float(row[k]) for k in (
                        "input_tokens", "output_tokens", "seconds", "truncated_calls", "result_chars", "sent_chars")):
                    raise ValueError("Trace and table disagree")
            cache = {}
            for metric in ("queries", "hits"):
                name = f"vllm:prefix_cache_{metric}_total"
                before = counter(archive, f"{c}/cache_before.txt", name)
                after = counter(archive, f"{c}/cache_after.txt", name)
                cache[metric + "_tokens"] = after - before if before is not None and after is not None else None
            cache["hit_rate"] = cache["hits_tokens"] / cache["queries_tokens"] if cache["queries_tokens"] else None
            summary[c] = {"overall": metrics(rows), "by_mode": {mode: metrics([r for r in rows if r["mode"] == mode])
                                                               for mode in MODES}, "cache": cache}
            tables[c] = raw
            sources[c] = metadata
            pair_tables[c] = {(r["task_id"], r["mode"]): r for r in rows}
        paired = []
        for c in CONDITIONS[1:]:
            for task, mode in sorted(expected):
                a, b = (pair_tables[x][(task, mode)] for x in ("baseline", c))
                paired.append({"condition": c, "task_id": task, "mode": mode, "before": a["utility"],
                               "after": b["utility"], "delta_input_tokens": int(b["input_tokens"]) - int(a["input_tokens"]),
                               "delta_seconds": float(b["seconds"]) - float(a["seconds"])})
    return summary, tables, sources, paired


def register(out, condition, source_hash, result):
    import mlflow
    store = ROOT / "lab4"
    mlflow.set_tracking_uri("sqlite:///" + (store / "mlflow_local.db").as_posix())
    current = mlflow.get_experiment_by_name("lr4_agentdojo")
    experiment = current.experiment_id if current else mlflow.create_experiment(
        "lr4_agentdojo", artifact_location=(store / "mlartifacts").as_uri())
    stage = "optional_" + condition
    previous = mlflow.search_runs(experiment_ids=[experiment], output_format="list",
        filter_string=f"tags.stage = '{stage}' AND tags.archive_sha256 = '{source_hash}' AND attributes.status = 'FINISHED'")
    if previous:
        return previous[0].info.run_id
    with mlflow.start_run(experiment_id=experiment, run_name=f"{stage}-{source_hash[:8]}",
                          tags={"stage": stage, "archive_sha256": source_hash}) as run:
        for group, data in [("overall", result["overall"]), *result["by_mode"].items(), ("cache", result["cache"])]:
            mlflow.log_metrics({f"{group}.{key}": value for key, value in data.items() if value is not None})
        mlflow.log_artifacts(str(out))
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    result, tables, sources, paired = analyze(args.archive)
    source_hash = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary.json").write_text(json.dumps({"archive_sha256": source_hash, "conditions": result},
                                                     indent=2) + "\n", encoding="utf-8")
    with (args.out / "paired.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    for condition in CONDITIONS:
        folder = args.out / condition
        folder.mkdir()
        (folder / "episodes.csv").write_bytes(tables[condition])
        (folder / "summary.json").write_text(json.dumps(result[condition], indent=2) + "\n", encoding="utf-8")
        (folder / "metadata.json").write_text(json.dumps(sources[condition], indent=2) + "\n", encoding="utf-8")
        print(condition, result[condition]["overall"], result[condition]["cache"])
        if args.mlflow:
            print("MLflow:", register(folder, condition, source_hash, result[condition]))


if __name__ == "__main__":
    main()
