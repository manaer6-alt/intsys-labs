"""Сравнение ReAct и Plan-then-execute на учебном сценарии календаря."""

import json

import yaml
from agentdojo.task_suite.load_suites import get_suite

from tool_executor import ToolExecutor


TOOL = "search_calendar_events"
QUESTION = "Who else is invited to the Networking event on May 26th?"


class ScriptedModel:
    """Предопределённые ответы вместо Qwen для проверки двух режимов."""

    def __init__(self):
        self.calls = 0

    def complete(self, history, phase):
        self.calls += 1
        if phase == "plan":
            assert not any(row["role"] == "tool" for row in history)
            return {"steps": [{"tool": TOOL, "purpose": "Find event participants"}]}
        if phase == "arguments" or not any(row["role"] == "tool" for row in history):
            return {"tool": TOOL, "arguments": {"query": "Networking event", "date": "2024-05-26"}}
        event = yaml.safe_load(history[-1]["content"])[0]
        return {"answer": "Invited participants: " + ", ".join(event["participants"])}


def run(mode):
    suite = get_suite("v1.2.2", "workspace")
    env = suite.load_and_inject_default_environment({})
    executor = ToolExecutor(env)
    model = ScriptedModel()
    history = [{"role": "user", "content": QUESTION}]

    def calendar_call(response):
        # Сценарий ограничен одним читающим инструментом учебного календаря.
        if response["tool"] != TOOL:
            raise ValueError("Unexpected calendar tool")
        history.append({"role": "assistant", "content": response})
        record = executor.execute(TOOL, json.dumps(response["arguments"]), allowed_tool=TOOL)
        history.append({"role": "tool", "content": record["output"]})
        assert not record["error"]

    if mode == "react":
        response = model.complete(history, "action")
        if "tool" in response:
            calendar_call(response)
        response = model.complete(history, "action")
    elif mode == "plan_then_execute":
        plan = model.complete(history, "plan")
        assert plan["steps"] == [{"tool": TOOL, "purpose": "Find event participants"}]
        history.append({"role": "assistant", "content": plan})
        calendar_call(model.complete(history, "arguments"))
        response = model.complete(history, "final")
    else:
        raise ValueError(mode)

    history.append({"role": "assistant", "content": response["answer"]})
    assert model.calls <= 15 and len(executor.trace) <= 15
    return response["answer"], model.calls, len(executor.trace)


def main():
    react = run("react")
    plan = run("plan_then_execute")
    assert react[0] == plan[0]
    assert react[1:] == (2, 1)
    assert plan[1:] == (3, 1)
    for mode, result in (("react", react), ("plan_then_execute", plan)):
        print(f"{mode}: model_calls={result[1]}, tool_calls={result[2]}")
    print("OK: оба режима получили одинаковый ответ.")


if __name__ == "__main__":
    main()
