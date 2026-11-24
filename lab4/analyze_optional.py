"""Compare the three recorded optional series without model calls."""

import argparse
import csv
import json
import statistics
from pathlib import Path


CONDITIONS = ("baseline", "truncate_1024", "prefix_cache")
MODES = ("react", "plan_then_execute")
EXPECTED_TASKS = [f"user_task_{i}" for i in (0, 4, 8, 12, 20, 28, 32, 36)]
EXPECTED = {(task, mode) for task in EXPECTED_TASKS for mode in MODES}


def read(folder, condition):
    directory = folder / condition
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    expected_config = {"baseline": (None, False), "truncate_1024": (1024, False),
                       "prefix_cache": (None, True)}[condition]
    if meta["condition"] != condition or (meta["observation_char_limit"], meta["prefix_caching"]) != expected_config:
        raise ValueError("Condition mismatch")
    with (directory / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 16 or {(r["task_id"], r["mode"]) for r in rows} != EXPECTED:
        raise ValueError(f"Incomplete condition: {condition}")
    return meta, rows


def metrics(rows):
    return {"n": len(rows), "success": sum(r["utility"] == "True" for r in rows),
            "input_tokens_median": statistics.median(int(r["input_tokens"]) for r in rows),
            "output_tokens_median": statistics.median(int(r["output_tokens"]) for r in rows),
            "seconds_median": statistics.median(float(r["seconds"]) for r in rows),
            "total_seconds": sum(float(r["seconds"]) for r in rows),
            "truncated_calls": sum(int(r["truncated_calls"]) for r in rows),
            "total_chars_saved": sum(int(r["result_chars"]) - int(r["sent_chars"]) for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    all_rows = {}
    metadata = {}
    for condition in CONDITIONS:
        metadata[condition], all_rows[condition] = read(args.root, condition)
    for condition in CONDITIONS[1:]:
        for key in ("frozen_sha256", "config", "prompts", "task_numbers"):
            if metadata[condition][key] != metadata["baseline"][key]:
                raise ValueError(f"Different conditions: {key}")
        if metadata[condition]["server"]["gpu"] != metadata["baseline"]["server"]["gpu"]:
            raise ValueError("Different GPUs")
        flags = {"--enable-prefix-caching", "--no-enable-prefix-caching"}
        if [v for v in metadata[condition]["server"]["command"] if v not in flags] != [
                v for v in metadata["baseline"]["server"]["command"] if v not in flags]:
            raise ValueError("Other server parameters changed")
    summary = {c: {m: metrics([r for r in all_rows[c] if r["mode"] == m]) for m in MODES}
               for c in CONDITIONS}
    paired = []
    by_condition = {c: {(r["task_id"], r["mode"]): r for r in all_rows[c]} for c in CONDITIONS}
    for condition in CONDITIONS[1:]:
        for mode in MODES:
            for task in EXPECTED_TASKS:
                before, after = (by_condition[c][(task, mode)] for c in ("baseline", condition))
                paired.append({"condition": condition, "task_id": task, "mode": mode,
                               "before_correct": before["utility"], "after_correct": after["utility"],
                               "delta_input_tokens": int(after["input_tokens"]) - int(before["input_tokens"]),
                               "delta_seconds": float(after["seconds"]) - float(before["seconds"]),
                               "truncated_calls": int(after["truncated_calls"])})
    (args.root / "optional_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.root / "optional_paired.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
