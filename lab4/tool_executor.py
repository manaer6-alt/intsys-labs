"""Исполнитель функций учебной среды AgentDojo workspace."""

import json
import time

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite


class ToolExecutor:
    """Отрабатывает отдельный вызов и сохраняет его в трейс эпизода."""

    def __init__(self, env, max_tools=15):
        suite = get_suite("v1.2.2", "workspace")
        if not isinstance(env, suite.environment_type):
            raise TypeError("Требуется учебная среда AgentDojo workspace")
        if not 1 <= max_tools <= 15:
            raise ValueError("max_tools должно быть от 1 до 15")
        self.env = env
        self.runtime = FunctionsRuntime(suite.tools)
        self.max_tools = max_tools
        self.trace = []

    @property
    def exhausted(self):
        return len(self.trace) >= self.max_tools

    def execute(self, name, arguments, allowed_tool=None):
        if self.exhausted:
            raise RuntimeError("max_tools")

        start = time.perf_counter()
        output, error = None, None
        try:
            args = json.loads(arguments)
            if not isinstance(args, dict):
                raise ValueError("Аргументы должны быть объектом JSON")
            if allowed_tool is not None and name != allowed_tool:
                raise ValueError("Вызов не соответствует текущему шагу плана")
            result, error = self.runtime.run_function(self.env, name, args)
            if error is None:
                output = tool_result_to_str(result)
        except (ValueError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"

        record = {
            "step": len(self.trace) + 1,
            "tool": name,
            "arguments": arguments,
            "output": error if error is not None else output,
            "error": error,
            "elapsed_s": time.perf_counter() - start,
        }
        self.trace.append(record)
        return record


def check():
    suite = get_suite("v1.2.2", "workspace")
    env = suite.load_and_inject_default_environment({})
    executor = ToolExecutor(env)

    # Ошибки расходуют лимит, но не блокируют последующие вызовы.
    assert executor.execute("unknown_tool", "{}")["error"]
    assert "ValidationError" in executor.execute("search_files_by_filename", "{}")["error"]
    assert executor.execute("get_current_day", "{}")["error"] is None

    # Проверяем, что другой инструмент шага не вызывается.
    before = env.model_dump()
    item = executor.execute("get_current_day", "{}", allowed_tool="search_emails")
    assert item["error"] and env.model_dump() == before

    while not executor.exhausted:
        executor.execute("get_current_day", "{}")
    assert len(executor.trace) == 15
    try:
        executor.execute("get_current_day", "{}")
    except RuntimeError as exc:
        assert str(exc) == "max_tools"
    else:
        raise AssertionError("Лимит не сработал")
    print("OK: корректный вызов, три ошибки, продолжение, 15 попыток и max_tools.")


if __name__ == "__main__":
    check()
