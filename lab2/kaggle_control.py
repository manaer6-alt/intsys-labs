# Управление ЛР2 из блокнота Kaggle. В ядре блокнота используются стандартные библиотеки.
# Сервер и клиент работают отдельными процессами /tmp/lab2-env.
# start_server — одна T4, FP16, фиксированная ревизия Qwen и настройки ЛР1.
# stage — запуск проверки или одной серии; checkpoint — ZIP с сохранёнными артефактами.
# Этот файл не выполняет оценку по эталону: её проведём после скачивания результатов.

import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_birth(pid):
    try:
        # Время старта процесса защищает от случайного повторного использования PID.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class KaggleRunner:
    def __init__(self, inputs, root, python, environment):
        self.inputs, self.root = Path(inputs), Path(root)
        self.python, self.env = str(python), dict(environment)
        self.config = read_json(self.inputs / "provenance/configs/lab2.json")
        self.process = None

    def ready(self):
        try:
            with urlopen(self.config["base_url"].rstrip("/") + "/models", timeout=3) as response:
                models = json.load(response)
            return self.config["model"] in [item["id"] for item in models["data"]]
        except (OSError, ValueError, KeyError):
            return False

    def start_server(self):
        environment = read_json(self.root / "environment.json")
        profile = {
            "model": self.config["model"], "model_revision": self.config["revision"],
            "tokenizer_revision": self.config["revision"], "dtype": "half",
            "max_model_len": 32768, "max_num_seqs": 32, "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.9, "tensor_parallel_size": 1,
            "prefix_caching": False, "attention_backend": "TRITON_ATTN",
            "seed": self.config["seed"], "environment": environment,
            "gpu": subprocess.check_output(["nvidia-smi", "--id=0",
                "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv"], text=True),
        }
        current_path = self.root / "current_server.json"
        if self.ready():
            current = read_json(current_path)
            if (current["profile"] != profile or process_birth(current["pid"]) != current["process_start_ticks"]):
                raise RuntimeError("На порту 8000 работает сервер с другим паспортом")
            print("Используем ранее запущенный сервер ЛР2.", flush=True)
            return

        session_id = uuid4().hex
        session = self.root / "server_sessions" / session_id
        session.mkdir(parents=True)
        command = [str(Path(self.python).with_name("vllm")), "serve", self.config["model"],
            "--revision", self.config["revision"], "--tokenizer-revision", self.config["revision"],
            "--dtype", "half", "--max-model-len", "32768", "--max-num-seqs", "32",
            "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.9",
            "--tensor-parallel-size", "1", "--seed", str(self.config["seed"]),
            "--attention-backend", "TRITON_ATTN", "--generation-config", "vllm",
            "--no-enable-prefix-caching", "--host", "127.0.0.1", "--port", "8000"]
        (session / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        log = session / "server.log"
        with log.open("w", encoding="utf-8") as stream:
            self.process = subprocess.Popen(command, cwd=REPO, env=self.env,
                                            stdout=stream, stderr=subprocess.STDOUT)
        passport = {"session_id": session_id, "profile": profile, "command": command,
                    "pid": self.process.pid, "process_start_ticks": process_birth(self.process.pid),
                    "log": str(log), "active": True}
        write_json(current_path, passport)
        write_json(session / "passport.json", passport)
        print("Запуск Qwen. Журнал:", log, flush=True)
        deadline, next_message = time.monotonic() + 1200, 0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                print(log.read_text(encoding="utf-8", errors="replace")[-16000:])
                self.checkpoint("startup_failed")
                raise RuntimeError("Сервер завершился; следующие этапы не запускай")
            if self.ready():
                print("СЕРВЕР ГОТОВ: одна T4, Qwen FP16, prefix cache отключён.", flush=True)
                for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                    if "kv cache" in line.lower() or "model loading took" in line.lower():
                        print(line, flush=True)
                return
            if time.monotonic() >= next_message:
                print("Ожидаем загрузку модели и подготовку сервера...", flush=True)
                next_message = time.monotonic() + 15
            time.sleep(2)
        self.stop_server()
        self.checkpoint("startup_timeout")
        raise TimeoutError("Сервер не запустился за 20 минут; сохрани журнал")

    def stop_server(self):
        path = self.root / "current_server.json"
        if not path.exists():
            return
        passport = read_json(path)
        pid = passport["pid"]
        if passport["process_start_ticks"] is not None and process_birth(pid) == passport["process_start_ticks"]:
            os.kill(pid, signal.SIGTERM)
            if self.process is not None and self.process.pid == pid:
                try:
                    self.process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            else:
                deadline = time.monotonic() + 45
                while process_birth(pid) == passport["process_start_ticks"] and time.monotonic() < deadline:
                    time.sleep(1)
                if process_birth(pid) == passport["process_start_ticks"]:
                    os.kill(pid, signal.SIGKILL)
        passport["active"] = False
        write_json(path, passport)
        print("Процесс модели остановлен. После скачивания архива заверши GPU-сессию Kaggle.", flush=True)

    def stage(self, name):
        if name not in ("check", "llm_only", "rag_k3", "oracle", "rag_k1", "rag_k5"):
            raise ValueError("Неизвестный этап")
        console = self.root / "console"
        console.mkdir(parents=True, exist_ok=True)
        log = console / f"{name}_{uuid4().hex[:8]}.log"
        command = [self.python, "-u", str(REPO / "lab2/run_inference.py"),
                   "--inputs", str(self.inputs), "--root", str(self.root), "--stage", name]
        process = subprocess.Popen(command, cwd=REPO, env=self.env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            with log.open("w", encoding="utf-8") as stream:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
            code = process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)
            self.checkpoint("interrupted")
            raise
        if code:
            self.checkpoint("failed_" + name)
            raise RuntimeError(f"Этап {name} завершился с ошибкой. Журнал: {log}")

    def checkpoint(self, label):
        provenance = self.root / "provenance"
        for name in ("run_inference.py", "kaggle_control.py", "requirements-gpu.txt", "verify_runtime.py",
                     "README_KAGGLE.md", "lab2_kaggle.ipynb"):
            source = REPO / "lab2" / name
            target = provenance / "lab2" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError("Исходники изменились между контрольными архивами")
            if not target.exists():
                shutil.copyfile(source, target)
        selected = read_json(self.root / "selection.json") if (self.root / "selection.json").exists() else {}
        summary = {}
        for stage, run_id in selected.items():
            folder = self.root / "runs" / run_id
            meta = read_json(folder / "metadata.json")
            metrics = read_json(folder / "metrics.json") if (folder / "metrics.json").exists() else {}
            summary[stage] = {"run_id": run_id, "status": meta["status"],
                              "recorded": meta.get("recorded", 0), **metrics}
        write_json(self.root / "execution_summary.json", summary)
        destination = self.root.parent / f"lab2_{label}_{uuid4().hex[:8]}.zip"
        with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in self.root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(self.root).as_posix())
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP повреждён")
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        destination.with_suffix(".zip.sha256").write_text(digest + "  " + destination.name + "\n", encoding="ascii")
        print("ИТОГИ ВЫПОЛНЕНИЯ:", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print("АРХИВ:", destination, flush=True)
        print("SHA-256:", digest, flush=True)
        return destination
