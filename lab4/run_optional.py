"""Optional comparisons on the existing simulated calendar/document debug tasks."""

import argparse
import csv
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suite
from agent_debug import Agent, CONFIG, HERE, PROMPTS


CONDITIONS = {
    "baseline": (None, False),
    "truncate_1024": (1024, False),
    "prefix_cache": (None, True),
}
FROZEN = {
    "config.json": "f7e42fa3a66c6ea4459a59c343a5f009881ef900649f730c97e75f4ab17473bf",
    "prompts.json": "ab16bb52b2d701538976734889a21fb0211786643195bb7f673b556935b396e6",
    "agent_debug.py": "a5f37cc404220229116aa3cbed6c0269f2afc0347b23eeede8245516c4fa58b5",
}
MODES = ("react", "plan_then_execute")
MARKER = "\n[TRUNCATED: tool result exceeds character limit]"


def cap_text(value, limit):
    if limit is None or len(value) <= limit:
        return value
    return value[:limit - len(MARKER)] + MARKER


class ObservationAgent(Agent):
    def __init__(self, mode, number, limit, **kwargs):
        super().__init__(mode, number, **kwargs)
        self.limit = limit

    def execute(self, raw, allowed_tool=None):
        super().execute(raw, allowed_tool)
        record = self.tool_trace[-1]
        original = self.chat[-1]["content"]
        sent = original if record["error"] else cap_text(original, self.limit)
        self.chat[-1]["content"] = sent
        record.update(result_chars=len(original), sent_chars=len(sent),
                      truncated=sent != original, model_observation=sent)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cache_metrics():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10) as resp:
            return "\n".join(line for line in resp.read().decode().splitlines()
                             if "prefix_cache" in line and not line.startswith("#"))
    except OSError:
        return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server-config", type=Path, required=True)
    args = parser.parse_args()
    for file, expected in FROZEN.items():
        if sha(HERE / file) != expected:
            raise ValueError(f"Frozen v2 file changed: {file}")
    limit, cache = CONDITIONS[args.condition]
    server = json.loads(args.server_config.read_text(encoding="utf-8"))
    flag = "--enable-prefix-caching" if cache else "--no-enable-prefix-caching"
    if flag not in server["command"] or server["model"] != CONFIG["model"] or server["revision"] != CONFIG["revision"]:
        raise ValueError("The recorded server configuration does not match the condition")
    numbers = sorted(sum(CONFIG["task_groups"].values(), []))
    if numbers != [0, 4, 8, 12, 20, 28, 32, 36]:
        raise ValueError("Expected the frozen eight ordinary debug tasks")
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "metadata.json", {
        "condition": args.condition, "observation_char_limit": limit, "prefix_caching": cache,
        "task_numbers": numbers, "frozen_sha256": FROZEN, "config": CONFIG, "prompts": PROMPTS,
        "server": server,
    })
    (args.out / "cache_before.txt").write_text(cache_metrics(), encoding="utf-8")
    suite = get_suite("v1.2.2", "workspace")
    rows = []
    for index, number in enumerate(numbers):
        task = suite.get_user_task_by_id(f"user_task_{number}")
        modes = MODES if index % 2 == 0 else MODES[::-1]
        for mode in modes:
            agent = ObservationAgent(mode, number, limit)
            start = time.perf_counter()
            utility, _ = suite.run_task_with_pipeline(agent, task, None, {})
            usage = [step.get("usage", {}) for step in agent.model_trace]
            row = {
                "task_id": task.ID, "mode": mode, "utility": utility,
                "model_calls": sum(not step.get("preflight_only", False) for step in agent.model_trace),
                "tool_calls": len(agent.tool_trace),
                "input_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
                "output_tokens": sum(u.get("completion_tokens", 0) for u in usage),
                "seconds": time.perf_counter() - start,
                "tool_errors": sum(bool(step["error"]) for step in agent.tool_trace),
                "truncated_calls": sum(step["truncated"] for step in agent.tool_trace),
                "result_chars": sum(step["result_chars"] for step in agent.tool_trace),
                "sent_chars": sum(step["sent_chars"] for step in agent.tool_trace),
                "stop_reason": agent.reason,
            }
            save(args.out / f"{task.ID}_{mode}.json", {
                **row, "answer": agent.answer, "model_trace": agent.model_trace,
                "tool_trace": agent.tool_trace})
            rows.append(row)
            with (args.out / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(args.condition, row, flush=True)
    (args.out / "cache_after.txt").write_text(cache_metrics(), encoding="utf-8")
    print(f"Completed {len(rows)} episodes: {args.out}", flush=True)


if __name__ == "__main__":
    main()
