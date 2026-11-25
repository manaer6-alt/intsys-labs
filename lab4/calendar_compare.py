"""Сравнение двух режимов на одной учебной задаче календаря user_task_0."""

import json
import time
import urllib.request
from pathlib import Path

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionCall
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import text_content_block_from_string as text


MODEL = "Qwen/Qwen3-4B-Instruct-2507"
URL = "http://127.0.0.1:8000/v1/chat/completions"
TOOL = "search_calendar_events"
SYSTEM = "Answer the user's question using the simulated calendar. Respond concisely."


def complete(messages, **options):
    payload = {
        "model": MODEL, "messages": messages, "temperature": 0,
        "top_p": 1, "seed": 0, "max_tokens": 1024, **options,
    }
    request = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


class CalendarAgent(BasePipelineElement):
    """Проверочный агент одного сценария с единственным читающим инструментом."""

    name = "local-calendar-compare"

    def __init__(self, mode):
        self.mode = mode
        self.trace = []
        self.tool_calls = 0
        self.stop_reason = "completed"

    def query(self, query, runtime, env, messages=(), extra_args=None):
        function = runtime.functions[TOOL]
        tools = [{"type": "function", "function": {
            "name": function.name, "description": function.description,
            "parameters": function.parameters.model_json_schema(),
        }}]
        chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
        log = [{"role": "system", "content": [text(SYSTEM)]},
               {"role": "user", "content": [text(query)]}]

        def ask(phase, **options):
            start = time.perf_counter()
            response = complete(chat, **options)
            message = response["choices"][0]["message"]
            self.trace.append({"phase": phase, "messages": json.loads(json.dumps(chat)),
                               "message": message, "usage": response.get("usage", {}),
                               "seconds": time.perf_counter() - start})
            return message

        try:
            if self.mode == "plan_then_execute":
                chat.append({"role": "user", "content":
                    "Plan a single calendar-search step. Return JSON with tool and purpose."})
                plan_message = ask("plan", response_format={"type": "json_schema", "json_schema": {
                    "name": "calendar_plan", "strict": True, "schema": {
                        "type": "object",
                        "properties": {"tool": {"type": "string", "enum": [TOOL]},
                                       "purpose": {"type": "string"}},
                        "required": ["tool", "purpose"], "additionalProperties": False,
                    }}})
                plan = json.loads(plan_message["content"])
                if plan["tool"] != TOOL:
                    raise ValueError("plan_invalid")
                chat.append({"role": "assistant", "content": plan_message["content"]})
                chat.append({"role": "user", "content": "Choose arguments for the planned calendar search."})

            message = ask("action", tools=tools, tool_choice="auto")
            calls = message.get("tool_calls") or []
            if len(calls) != 1 or calls[0]["function"]["name"] != TOOL:
                raise ValueError("Expected the single calendar search")
            raw = calls[0]
            args = json.loads(raw["function"]["arguments"])
            if not isinstance(args, dict) or not isinstance(args.get("query"), str):
                raise ValueError("Invalid calendar arguments")
            call = FunctionCall(function=TOOL, args=args, id=raw["id"])
            self.tool_calls += 1
            result, error = runtime.run_function(env, TOOL, args)
            observation = error if error else tool_result_to_str(result)
            self.trace[-1]["tool_result"] = observation
            self.trace[-1]["tool_error"] = error

            chat.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
            chat.append({"role": "tool", "tool_call_id": call.id, "content": observation})
            log.append({"role": "assistant", "content": None, "tool_calls": [call]})
            log.append({"role": "tool", "tool_call": call, "tool_call_id": call.id,
                        "content": [text(observation)], "error": error})
            final = ask("final")
            answer = final.get("content") or "Unable to answer."
            if error:
                self.stop_reason = "tool_error"
        except (ValueError, KeyError, TypeError) as exc:
            self.stop_reason = str(exc)
            answer = "Unable to complete this calendar check."

        log.append({"role": "assistant", "content": [text(answer)], "tool_calls": None})
        return query, runtime, env, log, extra_args or {}


def main():
    suite = get_suite("v1.2.2", "workspace")
    task = suite.get_user_task_by_id("user_task_0")
    results = []
    for mode in ("react", "plan_then_execute"):
        agent = CalendarAgent(mode)
        start = time.perf_counter()
        utility, _ = suite.run_task_with_pipeline(agent, task, None, {})
        result = {
            "task_id": task.ID, "mode": mode, "utility": utility,
            "model_calls": len(agent.trace), "tool_calls": agent.tool_calls,
            "input_tokens": sum(t["usage"].get("prompt_tokens", 0) for t in agent.trace),
            "output_tokens": sum(t["usage"].get("completion_tokens", 0) for t in agent.trace),
            "seconds": time.perf_counter() - start, "stop_reason": agent.stop_reason,
            "trace": agent.trace,
        }
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))
    output = Path("/kaggle/working/lab4_smoke/calendar_compare.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", output)


if __name__ == "__main__":
    main()
