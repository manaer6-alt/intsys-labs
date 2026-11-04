# Управление сервером, запуск скриптов и резервные копии из ячеек Kaggle.
# Вычислительные библиотеки импортируются только отдельными процессами окружения.
import hashlib
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parent.parent
ART = Path("/kaggle/working/lab1_artifacts")
PYTHON = "/tmp/lab1-env/bin/python"
C = json.loads((REPO / "configs/lab1_kaggle.json").read_text(encoding="utf-8"))
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "HF_HOME": "/tmp/lab1-hf", "PYTHONUNBUFFERED": "1"}
ENV.update(OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
server_process = None


def run(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, env=ENV, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    result.check_returncode()
    return result


def stop_server():
    global server_process
    if server_process is not None and server_process.poll() is None:
        server_process.terminate()
        server_process.wait(timeout=120)
    print("Процесс сервера остановлен. GPU-сессия Kaggle ещё активна.", flush=True)


def start_server(prefix=False):
    global server_process
    stop_server()
    ART.mkdir(parents=True, exist_ok=True)
    session = ART / "server_sessions" / uuid4().hex
    session.mkdir(parents=True)
    command = ["/tmp/lab1-env/bin/vllm", "serve", C["model"],
               "--revision", C["revision"], "--tokenizer-revision", C["revision"],
               "--dtype", "half", "--max-model-len", str(C["context"]),
               "--max-num-seqs", str(C["max_num_seqs"]),
               "--max-num-batched-tokens", str(C["max_num_batched_tokens"]),
               "--gpu-memory-utilization", str(C["gpu_memory_utilization"]),
               "--tensor-parallel-size", "1", "--seed", str(C["seed"]),
               "--attention-backend", "TRITON_ATTN", "--generation-config", "vllm",
               "--enable-prefix-caching" if prefix else "--no-enable-prefix-caching",
               "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
               "--host", "127.0.0.1", "--port", "8000"]
    (session / "command.txt").write_text(shlex.join(command), encoding="utf-8")
    log = session / "server.log"
    with log.open("w", encoding="utf-8") as file:
        server_process = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=file, stderr=subprocess.STDOUT)
    print("Запускается сервер; журнал:", log, flush=True)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if server_process.poll() is not None:
            print(log.read_text(encoding="utf-8", errors="replace")[-16000:])
            raise RuntimeError("Сервер завершился. Не запускай следующие блоки; сохрани журнал.")
        try:
            with urllib.request.urlopen(C["base_url"] + "/models", timeout=3) as response:
                models = json.load(response)
            assert C["model"] in [m["id"] for m in models["data"]]
            break
        except urllib.error.URLError:
            print("Ожидание запуска...", flush=True)
            time.sleep(10)
    else:
        stop_server()
        raise TimeoutError("Сервер не запустился за 30 минут; проверь журнал")
    gpu_info = run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"]).stdout
    info = {"prefix_caching": prefix, "model": C["model"], "revision": C["revision"],
            "log": str(log), "command": command, "cuda_visible_devices": "0", "gpu_count_used": 1,
            "gpu_inventory": gpu_info, "context": C["context"], "attention_backend": "TRITON_ATTN"}
    info.update(cpu_logical_count=os.cpu_count(), omp_num_threads=ENV["OMP_NUM_THREADS"],
                cpuinfo=Path("/proc/cpuinfo").read_text(errors="replace"),
                meminfo=Path("/proc/meminfo").read_text(errors="replace"))
    (ART / "current_server.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (session / "passport.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("СЕРВЕР ГОТОВ; кеширование префикса:", prefix, flush=True)
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if any(word in line.lower() for word in ["kv cache", "concurrency", "available kv"]):
            print(line)


def stage(name):
    ART.mkdir(parents=True, exist_ok=True)
    console = ART / "console"
    console.mkdir(exist_ok=True)
    log = console / f"{name}_{uuid4().hex}.log"
    command = [PYTHON, str(REPO / "lab1/kaggle_run.py"), "--stage", name, "--root", str(ART)]
    process = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        with log.open("w", encoding="utf-8") as file:
            for line in process.stdout:
                print(line, end="", flush=True)
                file.write(line)
                file.flush()
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait(timeout=60)
        raise
    if code:
        raise RuntimeError(f"Этап {name} завершился с ошибкой. Журнал: {log}")


def checkpoint(label):
    ART.mkdir(parents=True, exist_ok=True)
    provenance = ART / "provenance"
    for relative in ["configs/lab1_kaggle.json", "forecasts/lab1_kaggle.md", "endpoint.yaml",
                     "lab1/kaggle_run.py", "lab1/kaggle_harness.py", "lab1/kaggle_helpers.py",
                     "lab1/prepare_kaggle_assets.py", "lab1/import_mlflow_local.py"]:
        target = provenance / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    shutil.copytree(REPO / "lab1/data", provenance / "lab1/data", dirs_exist_ok=True)
    archive = Path(f"/kaggle/working/lab1_{label}_{uuid4().hex[:8]}.zip")
    # SQLite копируем через backup, не копируем файлы WAL/SHM.
    with tempfile.TemporaryDirectory() as temp, zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for path in ART.rglob("*"):
            if not path.is_file() or path.name.endswith((".db-wal", ".db-shm")):
                continue
            source = path
            if path.suffix == ".db":
                source = Path(temp) / (uuid4().hex + ".db")
                with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as src:
                    with sqlite3.connect(source) as dst:
                        src.backup(dst)
                        assert dst.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            z.write(source, path.relative_to(ART))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(digest + "  " + archive.name + "\n", encoding="ascii")
    print("АРХИВ:", archive)
    print("SHA-256:", digest)
    print("Скачай его через панель файлов Kaggle перед завершением сессии.")
    return archive
