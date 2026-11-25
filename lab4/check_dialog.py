"""Проверка формата диалога с инструментом без запуска Qwen."""

import json

from agentdojo.task_suite.load_suites import get_suite
from tool_executor import ToolExecutor


def main():
    suite = get_suite("v1.2.2", "workspace")
    env = suite.load_and_inject_default_environment({})
    executor = ToolExecutor(env)

    messages = [
        {"role": "system", "content": "Answer the user's question using the simulated calendar."},
        {"role": "user", "content": "What is today's date?"},
    ]

    # Имитируем ответ модели с одним вызовом инструмента.
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_current_day", "arguments": "{}"},
    }
    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})

    result = executor.execute(call["function"]["name"], call["function"]["arguments"])
    assert result["error"] is None
    messages.append({
        "role": "tool",
        "tool_call_id": call["id"],
        "content": result["output"],
    })

    # Имитируем финальный текст модели по наблюдению.
    messages.append({"role": "assistant", "content": f"Today's date is {result['output']}."})
    assert [message["role"] for message in messages] == [
        "system", "user", "assistant", "tool", "assistant"
    ]
    assert messages[3]["tool_call_id"] == messages[2]["tool_calls"][0]["id"]
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print("OK: история из пяти сообщений, результат инструмента и текстовый ответ.")


if __name__ == "__main__":
    main()
