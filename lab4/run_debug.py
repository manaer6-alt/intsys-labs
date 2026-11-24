"""Run both modes on ordinary calendar and file tasks in the debug split."""

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suite
from agent_debug import Agent, CONFIG, PROMPTS, HERE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", type=int, choices=sum(CONFIG["task_groups"].values(), []))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    suite = get_suite("v1.2.2", "workspace")
    numbers = sorted(sum(CONFIG["task_groups"].values(), []))
    if args.task is not None:
        numbers = [args.task]
    metadata = {"benchmark": "v1.2.2", "suite": "workspace", "task_numbers": numbers,
                "config": CONFIG, "prompts": PROMPTS,
                "config_sha256": hashlib.sha256((HERE / "config.json").read_bytes()).hexdigest()}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = []
    for index, number in enumerate(numbers):
        task = suite.get_user_task_by_id(f"user_task_{number}")
        modes = ("react", "plan_then_execute") if index % 2 == 0 else ("plan_then_execute", "react")
        for mode in modes:
            agent = Agent(mode, number)
            start = time.perf_counter()
            utility, _ = suite.run_task_with_pipeline(agent, task, None, {})
            row = {"task_id": task.ID, "mode": mode, "utility": utility,
                   "model_calls": sum(not r.get("preflight_only", False) for r in agent.model_trace),
                   "tool_calls": len(agent.tool_trace),
                   "input_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in agent.model_trace),
                   "output_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in agent.model_trace),
                   "seconds": time.perf_counter() - start, "stop_reason": agent.reason}
            rows.append(row)
            trace = {**row, "answer": agent.answer, "model_trace": agent.model_trace,
                     "tool_trace": agent.tool_trace}
            (args.out / f"{task.ID}_{mode}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
            print(row, flush=True)
            with (args.out / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = {}
    for mode in ("react", "plan_then_execute"):
        subset = [r for r in rows if r["mode"] == mode]
        summary[mode] = {"success": sum(r["utility"] for r in subset), "n": len(subset)}
        for key in ("model_calls", "tool_calls", "input_tokens", "output_tokens", "seconds"):
            values = [r[key] for r in subset]
            summary[mode][key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
