"""Один сценарный тест связи обвязки с оценкой AgentDojo."""

import json

import yaml
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import FunctionCall
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import text_content_block_from_string as text

from tool_executor import ToolExecutor


class CalendarExample(BasePipelineElement):
    """Фиксированный учебный сценарий для user_task_0."""

    name = "calendar-interface-check"

    def __init__(self, correct=True):
        self.correct = correct
        self.query_count = 0

    def query(self, query, runtime, env, messages=(), extra_args=None):
        self.query_count += 1
        executor = ToolExecutor(env)

        log = [
            {"role": "system", "content": [text("Use the simulated calendar.")]},
            {"role": "user", "content": [text(query)]},
        ]
        call = FunctionCall(
            function="search_calendar_events",
            args={"query": "Networking event", "date": "2024-05-26"},
            id="calendar-check-1",
        )
        log.append({"role": "assistant", "content": None, "tool_calls": [call]})

        record = executor.execute(call.function, json.dumps(call.args))
        log.append({
            "role": "tool",
            "tool_call": call,
            "tool_call_id": call.id,
            "content": [text(record["output"])],
            "error": record["error"],
        })

        if record["error"] is None and self.correct:
            events = yaml.safe_load(record["output"])
            participants = events[0]["participants"]
            answer = "Invited participants: " + ", ".join(participants)
        else:
            answer = "Unable to provide the participant list."

        log.append({"role": "assistant", "content": [text(answer)], "tool_calls": None})
        return query, runtime, env, log, extra_args or {}


def main():
    suite = get_suite("v1.2.2", "workspace")
    task = suite.get_user_task_by_id("user_task_0")
    for correct in (True, False):
        pipeline = CalendarExample(correct)
        utility, _ = suite.run_task_with_pipeline(pipeline, task, None, {})
        print(f"task={task.ID}, correct={correct}, utility={utility}, query_count={pipeline.query_count}")
        assert utility == correct
        assert pipeline.query_count == 1
    print("OK: AgentDojo distinguishes correct and incorrect answers.")


if __name__ == "__main__":
    main()
