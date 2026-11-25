"""Проверить каталог инструментов AgentDojo и три учебных вызова."""

import argparse
import json
from pathlib import Path

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    suite = get_suite("v1.2.2", "workspace")
    runtime = FunctionsRuntime(suite.tools)
    env = suite.load_and_inject_default_environment({})

    tools = [
        {
            "type": "function",
            "function": {
                "name": function.name,
                "description": function.description,
                "parameters": function.parameters.model_json_schema(),
            },
        }
        for function in runtime.functions.values()
    ]
    assert len(tools) == 24

    for tool in tools:
        function = tool["function"]
        required = function["parameters"].get("required", [])
        print(function["name"], "required:", required)

    # Обычный запрос только к учебному календарю.
    result, error = runtime.run_function(env, "get_current_day", {})
    assert error is None
    print("\nValid call:", result)

    # Две проверки ошибок интерфейса, без изменения состояния.
    _, error = runtime.run_function(env, "search_files_by_filename", {})
    assert error and "ValidationError" in error
    print("Missing argument:", error)

    _, error = runtime.run_function(env, "unknown_tool", {})
    assert error and "ToolNotFoundError" in error
    print("Unknown tool:", error)

    if not args.check_only:
        output = Path(__file__).resolve().parents[1] / "configs" / "lab4_tools.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(tools, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("Saved:", output)


if __name__ == "__main__":
    main()
