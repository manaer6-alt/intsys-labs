"""Two control modes for ordinary calendar and document tasks in the local fixture."""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionCall
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import text_content_block_from_string as text

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
PROMPTS = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
if CONFIG["base_url"] != "http://127.0.0.1:8000":
    raise ValueError("Local simulated workspace only")


class LimitReached(Exception):
    pass


def post_json(path, data):
    req = urllib.request.Request(
        CONFIG["base_url"] + path, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as response:
        return json.load(response)


def request_model(messages, tools=None, schema=None):
    body = {"model": CONFIG["model"], "messages": messages, "temperature": CONFIG["temperature"],
            "top_p": 1, "seed": CONFIG["seed"], "max_tokens": CONFIG["max_output_tokens"]}
    if tools:
        body.update(tools=tools, tool_choice="auto")
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "plan", "strict": True, "schema": schema}}
    token_body = {"model": CONFIG["model"], "messages": messages, "add_generation_prompt": True}
    if tools:
        token_body["tools"] = tools
    count = post_json("/tokenize", token_body)["count"]
    if count + CONFIG["max_output_tokens"] > CONFIG["context_limit"]:
        raise LimitReached("context_limit")
    return post_json("/v1/chat/completions", body)


def schema_for(names):
    return {"type": "object", "additionalProperties": False, "required": ["steps"],
            "properties": {"steps": {"type": "array", "minItems": 1,
                "maxItems": CONFIG["max_plan_steps"], "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["tool", "purpose"], "properties": {
                        "tool": {"type": "string", "enum": names},
                        "purpose": {"type": "string"}}}}}}


class Agent(BasePipelineElement):
    name = "local-workspace-debug"

    def __init__(self, mode, task_number, client=request_model):
        self.mode, self.task_number, self.client = mode, task_number, client
        group = next(name for name, ids in CONFIG["task_groups"].items() if task_number in ids)
        self.allowed = set(CONFIG["tools"][group])
        self.model_trace = []
        self.tool_trace = []
        self.reason = "completed"

    def ask(self, phase, tools=None, schema=None):
        if len(self.model_trace) >= CONFIG["max_model_calls"]:
            raise LimitReached("max_calls")
        record = {"phase": phase, "messages": json.loads(json.dumps(self.chat)),
                  "tools": [t["function"]["name"] for t in tools or []]}
        self.model_trace.append(record)
        started = time.perf_counter()
        try:
            response = self.client(self.chat, tools=tools, schema=schema)
            message = response["choices"][0]["message"]
            record.update(message=message, usage=response.get("usage", {}))
            return message
        except Exception as exc:
            record["error"] = str(exc)
            record["preflight_only"] = isinstance(exc, LimitReached) and str(exc) == "context_limit"
            raise
        finally:
            record["seconds"] = time.perf_counter() - started

    def execute(self, raw, allowed_tool=None):
        if len(self.tool_trace) >= CONFIG["max_tool_calls"]:
            raise LimitReached("max_tools")
        start = time.perf_counter()
        name = raw.get("function", {}).get("name", "")
        arguments = raw.get("function", {}).get("arguments", "")
        args, error, output = {}, None, None
        try:
            args = json.loads(arguments)
            if not isinstance(args, dict):
                raise ValueError("Arguments must be an object")
            if name not in self.allowed or (allowed_tool and name != allowed_tool):
                raise ValueError("Tool is outside the current workspace step")
            result, error = self.runtime.run_function(self.env, name, args)
            output = error if error else tool_result_to_str(result)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            output = error
        call = FunctionCall(function=name or "invalid", args=args if isinstance(args, dict) else {}, id=raw.get("id"))
        self.tool_trace.append({"name": name, "arguments": arguments, "error": error,
                                "output": output, "seconds": time.perf_counter() - start})
        self.chat.append({"role": "tool", "tool_call_id": call.id, "content": output or ""})
        self.log.append({"role": "assistant", "content": None, "tool_calls": [call]})
        self.log.append({"role": "tool", "tool_call": call, "tool_call_id": call.id,
                         "content": [text(output or "")], "error": error})

    def apply_calls(self, message, allowed_tool=None):
        calls = message.get("tool_calls") or []
        if allowed_tool and len(calls) != 1:
            self.chat.append({"role": "assistant", "content": message.get("content") or "Step failed."})
            self.chat.append({"role": "user", "content": "Step failed: expected exactly one tool call."})
            return
        self.chat.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
        for raw in calls:
            self.execute(raw, allowed_tool)

    def react(self, tools):
        while True:
            message = self.ask("react", tools=tools)
            if not message.get("tool_calls"):
                return message.get("content") or "Unable to answer."
            self.apply_calls(message)
            if len(self.tool_trace) >= CONFIG["max_tool_calls"]:
                raise LimitReached("max_tools")

    def plan(self, tools):
        self.chat.append({"role": "user", "content": PROMPTS["plan"] + " Available tools: " +
                          json.dumps(tools, ensure_ascii=False)})
        message = self.ask("plan", schema=schema_for([t["function"]["name"] for t in tools]))
        try:
            plan = json.loads(message.get("content") or "")
            steps = plan["steps"]
            if not 1 <= len(steps) <= CONFIG["max_plan_steps"]:
                raise ValueError
            if any(s["tool"] not in self.allowed or not isinstance(s["purpose"], str) for s in steps):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise LimitReached("plan_invalid")
        self.chat.append({"role": "assistant", "content": json.dumps(plan)})
        by_name = {t["function"]["name"]: t for t in tools}
        for index, step in enumerate(steps, 1):
            if len(self.tool_trace) >= CONFIG["max_tool_calls"]:
                raise LimitReached("max_tools")
            self.chat.append({"role": "user", "content": PROMPTS["arguments"].format(step=index)})
            message = self.ask("arguments", tools=[by_name[step["tool"]]])
            self.apply_calls(message, allowed_tool=step["tool"])
        self.chat.append({"role": "user", "content": PROMPTS["final"]})
        message = self.ask("final")
        return message.get("content") or "Unable to answer."

    def query(self, query, runtime, env, messages=(), extra_args=None):
        suite = get_suite("v1.2.2", "workspace")
        task = suite.get_user_task_by_id(f"user_task_{self.task_number}")
        if type(env) is not suite.environment_type or query != task.PROMPT:
            raise ValueError("Expected the specified simulated workspace task")
        self.runtime, self.env = runtime, env
        system = PROMPTS["system"].format(date=env.calendar.current_day.isoformat())
        self.chat = [{"role": "system", "content": system}, {"role": "user", "content": query}]
        self.log = [{"role": item["role"], "content": [text(item["content"])]} for item in self.chat]
        tools = [{"type": "function", "function": {"name": f.name, "description": f.description,
                  "parameters": f.parameters.model_json_schema()}}
                 for name, f in runtime.functions.items() if name in self.allowed]
        try:
            answer = self.react(tools) if self.mode == "react" else self.plan(tools)
        except LimitReached as exc:
            self.reason = str(exc)
            answer = "The task could not be completed within the configured limits."
        except urllib.error.HTTPError as exc:
            self.reason = f"http_{exc.code}"
            answer = "The model request failed."
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"
            answer = "The task could not be completed."
        self.log.append({"role": "assistant", "content": [text(answer)], "tool_calls": None})
        self.answer = answer
        return query, runtime, env, self.log, extra_args or {}
