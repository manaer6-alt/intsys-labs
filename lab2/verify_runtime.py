# Проверка клиента ЛР2 на локальном искусственном HTTP-сервере.
# Не использует Qwen и GPU, не вычисляет научную точность по фиктивным ответам.
# Все тестовые результаты создаются во временной папке и удаляются по окончании.
# --inputs: каталог уже подготовленного основного пакета из 420 заданий.

import argparse
import contextlib
import copy
import io
import json
import shutil
import tempfile
import threading
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import run_inference as runtime

PREPARED = None
INPUTS = None


class RuntimeChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lab2_mock_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = copy.deepcopy(PREPARED)
        self.calls = Counter()
        self.fail_key = None
        self.did_fail = False
        self.bad_tokens = False
        self.parameters = []
        self.counts = {}
        for task in self.prepared["tasks"] + self.prepared["warmups"]:
            key = runtime.fingerprint(task["messages"])
            if key in self.counts:
                self.assertEqual(self.counts[key], task["prompt_tokens"])
            self.counts[key] = task["prompt_tokens"]
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_json(self, value, status=200):
                body = json.dumps(value).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send_json({"data": [{"id": outer.prepared["config"]["model"]}]})

            def do_POST(self):
                value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                key = runtime.fingerprint(value["messages"])
                count = outer.counts[key]
                if self.path == "/tokenize":
                    self.send_json({"count": count + int(outer.bad_tokens), "max_model_len": 32768, "tokens": []})
                    return
                outer.calls[key] += 1
                outer.parameters.append(value)
                if key == outer.fail_key and not outer.did_fail:
                    outer.did_fail = True
                    self.send_json({"error": "Искусственный сбой тестового сервера"}, 503)
                    return
                self.send_json({"model": value["model"], "choices": [{"finish_reason": "stop", "message": {
                    "content": '{"label":"NOINFO","doc_id":null,"sentences":[]}'}}],
                    "usage": {"prompt_tokens": count, "completion_tokens": 12, "total_tokens": count + 12}})

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.prepared["config"]["base_url"] = f"http://127.0.0.1:{self.http.server_port}/v1"
        self.server = {"session_id": "mock-session", "profile": {"max_model_len": 32768, "mock": True}}

    def close_server(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=5)

    def stage(self, name):
        with patch.object(runtime, "git_commit", return_value="mock-commit"), contextlib.redirect_stdout(io.StringIO()):
            runtime.run_stage(name, self.prepared, self.root, self.server, "mock-commit")

    def test_input_integrity(self):
        self.assertEqual(len(self.prepared["tasks"]), 420)
        self.assertEqual(len({t["task_id"] for t in self.prepared["tasks"]}), 420)
        self.assertTrue(all(t["prompt_tokens"] <= 2048 for t in self.prepared["tasks"]))

    def test_corrupted_input_rejected(self):
        target = self.root / "inputs"
        shutil.copytree(INPUTS, target)
        with (target / "tasks.jsonl").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaises(ValueError):
            runtime.load_inputs(target)

    def test_token_mismatch_blocks_check(self):
        self.bad_tokens = True
        with self.assertRaises(ValueError):
            self.stage("check")
        self.assertFalse((self.root / "approved_check.json").exists())
        self.assertEqual(sum(self.calls.values()), 0)

    def test_all_modes_and_no_repeated_completed_calls(self):
        self.stage("check")
        for name in runtime.EXPECTED:
            self.stage(name)
        selection = runtime.read_json(self.root / "selection.json")
        total = 0
        for name, expected in runtime.EXPECTED.items():
            folder = self.root / "runs" / selection[name]
            rows = runtime.read_lines(folder / "results.jsonl")
            self.assertEqual(len(rows), expected)
            self.assertEqual(len({r["task_id"] for r in rows}), expected)
            self.assertTrue(all(r["status"] == "completed" and r["parse_status"] == "ok" for r in rows))
            total += len(rows)
        self.assertEqual(total, 420)
        for payload in self.parameters:
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["max_tokens"], 128)
            self.assertFalse(payload["ignore_eos"])
            self.assertEqual(payload["response_format"], runtime.RESPONSE_FORMAT)
        before = sum(self.calls.values())
        for name in runtime.EXPECTED:
            self.stage(name)
        self.assertEqual(sum(self.calls.values()), before)

    def test_resume_keeps_recorded_failure(self):
        self.stage("check")
        tasks = [t for t in self.prepared["tasks"] if runtime.series_name(t) == "llm_only"]
        frequencies = Counter(runtime.fingerprint(t["messages"]) for t in tasks)
        warmup_keys = {runtime.fingerprint(t["messages"]) for t in self.prepared["warmups"]}
        self.fail_key = next(key for key, count in frequencies.items() if count == 1 and key not in warmup_keys)
        with self.assertRaises(RuntimeError):
            self.stage("llm_only")
        self.stage("llm_only")
        run_id = runtime.read_json(self.root / "selection.json")["llm_only"]
        folder = self.root / "runs" / run_id
        rows = runtime.read_lines(folder / "results.jsonl")
        self.assertEqual(len(rows), 90)
        self.assertEqual(sum(r["status"] == "failed" for r in rows), 1)
        self.assertEqual(self.calls[self.fail_key], 1)
        self.assertEqual(runtime.read_json(folder / "metrics.json")["call_failed"], 1)

    def test_changed_conditions_rejected(self):
        self.stage("check")
        self.stage("rag_k1")
        self.server["profile"]["mock"] = "changed"
        with self.assertRaises(ValueError):
            self.stage("rag_k1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверки клиента без GPU и Qwen")
    parser.add_argument("--inputs", type=Path, required=True)
    args, unittest_args = parser.parse_known_args()
    INPUTS = args.inputs.resolve()
    PREPARED = runtime.load_inputs(INPUTS)
    unittest.main(argv=[__file__, *unittest_args], verbosity=2)
