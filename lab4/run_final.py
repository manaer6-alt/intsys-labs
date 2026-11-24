"""Отложенная проверка двух режимов на заранее выбранных учебных задачах."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from agent_debug import CONFIG, HERE
from run_debug import main as run_series


METRICS = ("model_calls", "tool_calls", "input_tokens", "output_tokens", "seconds")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    selection = json.loads((HERE / "final_selection.json").read_text(encoding="utf-8"))
    for name, field in (
        ("config.json", "debug_config_sha256"),
        ("prompts.json", "prompts_sha256"),
        ("agent_debug.py", "controller_sha256"),
    ):
        if sha(HERE / name) != selection[field].lower():
            raise ValueError(f"Frozen file changed: {name}")
    split_path = HERE.parent / "data" / "agentdojo" / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    numbers = sorted(sum(selection["task_groups"].values(), []))
    if len(numbers) != len(set(numbers)) or len(numbers) != 12:
        raise ValueError("Expected exactly 12 distinct selected tasks")
    if not {f"user_task_{number}" for number in numbers}.issubset(split["final"]):
        raise ValueError("Final selection overlaps with debug or contains unknown IDs")
    if set(numbers) & set(sum(CONFIG["task_groups"].values(), [])):
        raise ValueError("Final selection overlaps with debug")
    for number, tools in selection["task_writes"].items():
        group = next(group for group, ids in selection["task_groups"].items() if int(number) in ids)
        if not set(tools) <= set(CONFIG["tools"][group]) & set(CONFIG["write_tools"]):
            raise ValueError("Unfrozen write tool")
    CONFIG["task_groups"] = selection["task_groups"]
    CONFIG["task_writes"] = selection["task_writes"]
    return selection, numbers


def stats(rows):
    result = {"n": len(rows), "success": sum(r["utility"] == "True" for r in rows)}
    if rows:
        result["success_rate"] = result["success"] / len(rows)
        for key in METRICS:
            values = [float(row[key]) for row in rows]
            result[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    return result


def analyze(out, selection, numbers):
    with (out / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    modes = ("react", "plan_then_execute")
    wanted = {(f"user_task_{number}", mode) for number in numbers for mode in modes}
    if len(rows) != len(wanted) or {(r["task_id"], r["mode"]) for r in rows} != wanted:
        raise ValueError("Incomplete or duplicate episodes")
    by_mode = {mode: {r["task_id"]: r for r in rows if r["mode"] == mode} for mode in modes}
    groups = {"both": 0, "react_only": 0, "plan_only": 0, "neither": 0}
    paired = []
    for number in numbers:
        task = f"user_task_{number}"
        react, plan = (by_mode[mode][task]["utility"] == "True" for mode in modes)
        group = "both" if react and plan else "react_only" if react else "plan_only" if plan else "neither"
        groups[group] += 1
        paired.append({"task_id": task, "react": react, "plan_then_execute": plan, "group": group})
    with (out / "paired.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    summary = {"selection": selection, "pairs": groups, "episodes": len(rows), "methods": {}}
    for mode in modes:
        selected = list(by_mode[mode].values())
        by_success = {status: stats([r for r in selected if r["utility"] == status]) for status in ("True", "False")}
        writes = {f"user_task_{number}" for number in selection["task_writes"]}
        summary["methods"][mode] = {
            "overall": stats(selected),
            "by_success": by_success,
            "read_only": stats([r for r in selected if r["task_id"] not in writes]),
            "changes_state": stats([r for r in selected if r["task_id"] in writes]),
            "stop_reasons": dict((reason, sum(r["stop_reason"] == reason for r in selected))
                                 for reason in sorted({r["stop_reason"] for r in selected})),
        }
    (out / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["pairs"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selection, numbers = prepare()
    run_series()
    analyze(args.out, selection, numbers)


if __name__ == "__main__":
    main()
