# harness.py — общий механизм выполнения экспериментов ЛР0.
#
# Что делает:
# 1. Проверяет конфигурацию, состояние Git и подготовленное разбиение.
# 2. Задаёт случайное зерно и создаёт нейросеть.
# 3. Обучает модель на train и оценивает её на val.
# 4. По отдельному указанию оценивает модель на test.
# 5. Сохраняет условия запуска, историю, предсказания и метрики.
# 6. Записывает результаты в MLflow и поддерживает поиск Optuna.
#
# Результат каждого запуска — отдельный каталог runs/<run_id>/:
# config.yaml, metadata.json, history.csv, predictions.csv, metrics.json.
# При ошибке её текст и traceback сохраняются в metadata.json.
#
# Сам файл обучение не запускает.
# Функцию run_experiment() будут вызывать train.py и search.py.

import csv
import hashlib
import math
import platform
import random
import subprocess
import time
import traceback
import uuid
from datetime import datetime, timezone

import mlflow
import numpy as np
import optuna
import sklearn
import torch
import yaml
from sklearn.datasets import load_digits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prepare_data import (
    ROOT, SPLIT_PATH, MANIFEST_PATH,
    read_json, save_json, validate_split,
)


HISTORY_FIELDS = [
    "epoch", "train_loss", "val_loss", "val_acc", "epoch_seconds"
]


def validate_config(config):
    # Проверяем состав конфигурации, чтобы опечатки не игнорировались.
    sections = {
        "model": {"n_layers", "hidden", "dropout"},
        "train": {"lr", "weight_decay", "batch_size", "epochs"},
        "runtime": {"device", "num_threads"},
    }
    expected = set(sections) | {"experiment", "split_seed", "seed"}

    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("Неверный состав полей конфигурации")

    for section, fields in sections.items():
        values = config[section]
        if not isinstance(values, dict) or set(values) != fields:
            raise ValueError(f"Неверные поля в разделе {section}")

    for name in ("seed", "split_seed"):
        value = config[name]
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError(f"{name}: ожидается целое число от 0 до 2**32-1")

    if config["experiment"] != "lr0" or config["split_seed"] != 0:
        raise ValueError("Для ЛР0 нужны experiment: lr0 и split_seed: 0")

    model, train, runtime = (
        config["model"], config["train"], config["runtime"]
    )

    for section, names in (
        (model, ("n_layers", "hidden")),
        (train, ("batch_size", "epochs")),
        (runtime, ("num_threads",)),
    ):
        for name in names:
            if type(section[name]) is not int or section[name] <= 0:
                raise ValueError(f"{name}: ожидается положительное целое")

    numbers = {
        "lr": train["lr"],
        "weight_decay": train["weight_decay"],
        "dropout": model["dropout"],
    }
    for name, value in numbers.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name}: ожидается конечное число")

    if train["lr"] <= 0 or train["weight_decay"] < 0:
        raise ValueError("Нужно lr > 0 и weight_decay >= 0")
    if not 0 <= model["dropout"] <= 0.5:
        raise ValueError("dropout должен находиться в диапазоне [0, 0.5]")
    if runtime["device"] != "cpu" or runtime["num_threads"] != 1:
        raise ValueError("В ЛР0 используем device: cpu и num_threads: 1")


def load_config(path):
    config = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def git_info(debug=False):
    # Git проверяется относительно корня проекта.
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()

    if changes and not debug:
        raise ValueError(
            "Есть незакоммиченные изменения. "
            "Сделай коммит или используй отладочный режим."
        )
    return commit, bool(changes)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    split = read_json(SPLIT_PATH)
    manifest = read_json(MANIFEST_PATH)["datasets"]["digits"]

    if file_sha256(SPLIT_PATH) != manifest["split_sha256"]:
        raise ValueError("Хеш split.json отличается от манифеста")
    if sklearn.__version__ != manifest["source_version"]:
        raise ValueError("Версия scikit-learn отличается от манифеста")
    if manifest["split_seed"] != 0:
        raise ValueError("В манифесте должен быть split_seed: 0")

    X, y = load_digits(return_X_y=True)
    validate_split(split, y)

    # Используем известную шкалу пикселей, без подбора по данным.
    X = torch.tensor(X / 16, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y, split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def build_model(config):
    layers = []
    inputs = 64

    for _ in range(config["n_layers"]):
        layers.extend([
            nn.Linear(inputs, config["hidden"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
        ])
        inputs = config["hidden"]

    # Возвращаем логиты: softmax перед CrossEntropyLoss не нужен.
    return nn.Sequential(*layers, nn.Linear(inputs, 10))


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Нужны непустые списки меток одинаковой длины")
    return float(np.mean(y_true == y_pred))


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(X_batch), y_batch)
        if not math.isfinite(loss.item()):
            raise FloatingPointError("При обучении получен NaN или inf")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)

    # Учитываем, что последний батч может быть меньше остальных.
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, X, y, criterion):
    model.eval()
    logits = model(X)
    loss = criterion(logits, y).item()
    if not math.isfinite(loss):
        raise FloatingPointError("При оценке получен NaN или inf")

    predictions = logits.argmax(dim=1).tolist()
    return loss, accuracy(y.tolist(), predictions), predictions


def recalculate_metrics(run_dir):
    # Итоговые метрики читаем из реально сохранённых файлов.
    with (run_dir / "history.csv").open(encoding="utf-8", newline="") as file:
        history = list(csv.DictReader(file))
    with (run_dir / "predictions.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        predictions = list(csv.DictReader(file))

    metrics = {
        "epochs_completed": len(history),
        "val_loss": float(history[-1]["val_loss"]),
        "train_seconds": sum(float(row["epoch_seconds"]) for row in history),
    }

    for name in sorted({row["split"] for row in predictions}):
        rows = [row for row in predictions if row["split"] == name]
        metrics[f"{name}_acc"] = accuracy(
            [int(row["y_true"]) for row in rows],
            [int(row["y_pred"]) for row in rows],
        )

    if metrics["val_acc"] != float(history[-1]["val_acc"]):
        raise ValueError("Точность из predictions.csv не совпала с history.csv")
    return metrics


def setup_tracking():
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name("lr0") is None:
        mlflow.create_experiment(
            "lr0", artifact_location=(ROOT / "mlartifacts").as_uri()
        )
    mlflow.set_experiment("lr0")


def run_experiment(
    config, series="manual", eval_test=False, debug=False, trial=None
):
    # Проверяем итоговую конфигурацию после переопределений из CLI.
    validate_config(config)
    if not isinstance(series, str) or not series.strip():
        raise ValueError("Нужно непустое название серии")
    if not debug and config["train"]["epochs"] != 30:
        raise ValueError("Итоговый запуск должен содержать 30 эпох")
    if eval_test and (debug or trial is not None):
        raise ValueError("Test нельзя использовать при отладке или поиске")

    commit, dirty = git_info(debug)
    X, y, split = load_data()
    seed = config["seed"]

    # Новый идентификатор защищает предыдущие результаты от перезаписи.
    run_id = uuid.uuid4().hex
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    metadata = {
        "run_id": run_id, "lab": 0, "series": series,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "commit": commit, "dirty": dirty, "debug": debug,
        "seed": seed, "device": "cpu", "num_threads": 1,
        "model": {"type": "MLP", **config["model"]},
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "optuna": optuna.__version__,
            "mlflow": mlflow.__version__, "pyyaml": yaml.__version__,
        },
        "split_sha256": file_sha256(SPLIT_PATH),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "timing": "Обучение и оценка val; без записи файлов и MLflow",
    }
    if trial is not None:
        metadata["trial_number"] = trial.number
    save_json(run_dir / "metadata.json", metadata)

    tracking_run = None
    state = "FAIL"

    try:
        setup_tracking()
        tracking_run = mlflow.start_run(
            run_name=f"{series}-seed{seed}",
            nested=mlflow.active_run() is not None,
        )
        metadata["mlflow_run_id"] = tracking_run.info.run_id

        # Разворачиваем вложенные параметры для таблицы MLflow.
        params = {}
        for name, value in config.items():
            if isinstance(value, dict):
                params.update({f"{name}.{key}": v for key, v in value.items()})
            else:
                params[name] = value
        mlflow.log_params(params)
        mlflow.set_tags({
            "commit": commit, "series": series, "seed": seed,
            "debug": str(debug).lower(),
        })
        if trial is not None:
            mlflow.set_tag("trial_number", trial.number)
            trial.set_user_attr("run_id", run_id)

        set_seed(seed)
        model = build_model(config["model"])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"],
        )
        loader = DataLoader(
            TensorDataset(X[split["train"]], y[split["train"]]),
            batch_size=config["train"]["batch_size"],
            shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        X_val, y_val = X[split["val"]], y[split["val"]]

        with (run_dir / "history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            file.flush()

            for epoch in range(1, config["train"]["epochs"] + 1):
                started = time.perf_counter()
                train_loss = train_epoch(model, loader, optimizer, criterion)
                val_loss, val_acc, _ = evaluate(model, X_val, y_val, criterion)
                seconds = time.perf_counter() - started

                row = dict(zip(HISTORY_FIELDS, [
                    epoch, train_loss, val_loss, val_acc, seconds
                ]))
                writer.writerow(row)
                file.flush()
                mlflow.log_metrics(
                    {key: value for key, value in row.items() if key != "epoch"},
                    step=epoch,
                )

                if trial is not None:
                    # Шаги Optuna считаем с нуля.
                    trial.report(val_acc, step=epoch - 1)
                    if trial.should_prune():
                        state = "PRUNED"
                        break
            else:
                state = "COMPLETE"

        # После обучения сохраняем предсказания текущей модели.
        with (run_dir / "predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["split", "index", "y_true", "y_pred"])
            for name in (["val", "test"] if eval_test else ["val"]):
                indices = split[name]
                _, _, predictions = evaluate(
                    model, X[indices], y[indices], criterion
                )
                for index, prediction in zip(indices, predictions):
                    writer.writerow([name, index, int(y[index]), prediction])
                    file.flush()

        metrics = recalculate_metrics(run_dir)
        save_json(run_dir / "metrics.json", metrics)
        mlflow.log_metrics(metrics, step=metrics["epochs_completed"])
        metadata["status"] = "completed"

    except BaseException:
        state = "FAIL"
        metadata["status"] = "failed"
        metadata["error"] = traceback.format_exc()
        raise

    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        if trial is not None:
            metadata["trial_state"] = state
        save_json(run_dir / "metadata.json", metadata)

        if tracking_run is not None:
            try:
                mlflow.set_tag("state", state)
                mlflow.log_artifacts(str(run_dir), artifact_path="run")
            except BaseException:
                metadata["status"] = "failed"
                metadata["error"] = (
                    metadata.get("error", "") + traceback.format_exc()
                )
                if trial is not None:
                    metadata["trial_state"] = "FAIL"
                save_json(run_dir / "metadata.json", metadata)
                raise
            finally:
                mlflow.end_run(
                    status="FINISHED"
                    if metadata["status"] == "completed" else "FAILED"
                )

    # Сначала сохраняем результаты остановленной попытки, затем сообщаем Optuna.
    if state == "PRUNED":
        raise optuna.TrialPruned(f"Досрочно остановлен запуск {run_id}")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "mlflow_run_id": tracking_run.info.run_id,
    }