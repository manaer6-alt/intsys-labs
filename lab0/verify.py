# verify.py — проверки корректности реализации ЛР0.
#
# Обычный запуск:
# проверяет точность, конфигурацию, данные, seed, пересчёт метрик и Git.
# Обучение при этом не запускается.
#
# Запуск с --runs:
# дополнительно выполняет три обучения с seed 0, 0 и 1
# через train.py в отдельных процессах.
# Требует чистого Git и сохраняет результаты в runs/.
#
# Искусственные данные и времена в тестах — только контрольные примеры.
# Они не являются результатами экспериментов и не включаются в отчёт.

import argparse
import csv
import random
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import harness


def learning_history(run_dir):
    # Сравниваем результаты обучения, исключая реальное время эпох.
    with (run_dir / "history.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        return [
            {key: row[key] for key in (
                "epoch", "train_loss", "val_loss", "val_acc"
            )}
            for row in csv.DictReader(file)
        ]


class BasicChecks(unittest.TestCase):
    def test_accuracy(self):
        # Три правильных ответа из четырёх.
        self.assertEqual(
            harness.accuracy([1, 2, 3, 4], [1, 2, 3, 3]), 0.75
        )

    def test_config(self):
        original = harness.load_config("configs/base.yaml")

        # Недопустимые значения должны отклоняться до обучения.
        for section, name, value in (
            ("train", "lr", -1),
            ("train", "lr", "0.001"),
            ("train", "batch_size", 0),
            ("train", "epochs", True),
            ("model", "dropout", 1.0),
        ):
            with self.subTest(parameter=name, value=value):
                config = deepcopy(original)
                config[section][name] = value
                with self.assertRaises(ValueError):
                    harness.validate_config(config)

    def test_data(self):
        X, y, split = harness.load_data()
        self.assertEqual(tuple(X.shape), (1797, 64))
        self.assertEqual(tuple(y.shape), (1797,))
        self.assertTrue(X.isfinite().all().item())
        self.assertGreaterEqual(X.min().item(), 0)
        self.assertLessEqual(X.max().item(), 1)

        # Создаём пересечение частей в копии разбиения.
        broken = deepcopy(split)
        broken["train"][0] = broken["test"][0]
        with self.assertRaises(ValueError):
            harness.validate_split(broken, y.numpy())

        # Даже изменение байтов без изменения JSON должно менять хеш.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.json"
            path.write_bytes(harness.SPLIT_PATH.read_bytes() + b"\n")
            with patch.object(harness, "SPLIT_PATH", path):
                with self.assertRaises(ValueError):
                    harness.load_data()

    def test_seed(self):
        def values():
            return random.random(), np.random.random(), torch.rand(3).tolist()

        harness.set_seed(0)
        first = values()
        harness.set_seed(0)
        self.assertEqual(first, values())

        harness.set_seed(1)
        self.assertNotEqual(first, values())
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_metrics(self):
        # Отдельные временные CSV с заранее известными ответами.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "history.csv").write_text(
                "epoch,train_loss,val_loss,val_acc,epoch_seconds\n"
                "1,1.0,1.0,0.75,0.1\n"
                "2,0.9,0.9,0.75,0.2\n",
                encoding="utf-8",
            )
            with (path / "predictions.csv").open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.writer(file)
                writer.writerow(["split", "index", "y_true", "y_pred"])

                # Ошибка на каждом четвёртом примере: 270/360 = 0.75.
                for index in range(360):
                    label = index % 10
                    prediction = label if index % 4 else (label + 1) % 10
                    writer.writerow(["val", index, label, prediction])

            metrics = harness.recalculate_metrics(path)
            self.assertEqual(metrics["val_acc"], 0.75)
            self.assertEqual(metrics["val_loss"], 0.9)
            self.assertEqual(metrics["epochs_completed"], 2)
            self.assertAlmostEqual(metrics["train_seconds"], 0.3)

    def test_git(self):
        # Проверяем Git в отдельном временном репозитории.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()

            def git(*args):
                subprocess.run(
                    ["git", "-C", str(root), *args],
                    check=True, stdout=subprocess.DEVNULL,
                )

            git("init", "-q")
            git(
                "-c", "user.name=Verify",
                "-c", "user.email=verify@example.invalid",
                "-c", "commit.gpgsign=false",
                "-c", f"core.hooksPath={root / 'empty-hooks'}",
                "commit", "--allow-empty", "-qm", "verify",
            )

            with patch.object(harness, "ROOT", root):
                self.assertFalse(harness.git_info()[1])

                (root / "changed.txt").write_text("test", encoding="utf-8")
                with self.assertRaises(ValueError):
                    harness.git_info()

                # Отладочный режим допускает незакоммиченные изменения.
                self.assertTrue(harness.git_info(debug=True)[1])


class TrainingChecks(unittest.TestCase):
    enabled = False

    def test_repeatability(self):
        if not self.enabled:
            self.skipTest("Полные запуски включаются аргументом --runs")

        paths = []

        for seed in (0, 0, 1):
            before = set((harness.ROOT / "runs").glob("*/metadata.json"))

            # Каждый запуск обучения выполняется в новом процессе.
            subprocess.run(
                [
                    sys.executable, str(harness.ROOT / "lab0" / "train.py"),
                    "--seed", str(seed), "--series", "verify",
                ],
                cwd=harness.ROOT, check=True,
                stdout=subprocess.DEVNULL,
            )

            created = (
                set((harness.ROOT / "runs").glob("*/metadata.json")) - before
            )
            self.assertEqual(len(created), 1)
            run_dir = created.pop().parent

            meta = harness.read_json(run_dir / "metadata.json")
            config = harness.load_config(run_dir / "config.yaml")
            metrics = harness.recalculate_metrics(run_dir)

            self.assertEqual(meta["status"], "completed")
            self.assertFalse(meta["debug"])
            self.assertFalse(meta["dirty"])
            self.assertEqual(meta["seed"], seed)
            self.assertEqual(config["seed"], seed)
            self.assertEqual(metrics["epochs_completed"], 30)
            self.assertEqual(
                metrics, harness.read_json(run_dir / "metrics.json")
            )
            paths.append(run_dir)

        first, repeated, different = paths

        self.assertEqual(
            learning_history(first), learning_history(repeated)
        )
        self.assertEqual(
            (first / "predictions.csv").read_bytes(),
            (repeated / "predictions.csv").read_bytes(),
        )

        # Итоговая точность может совпадать при разных ошибках.
        # Поэтому сравниваем историю обучения и сами предсказания.
        self.assertNotEqual(
            learning_history(first), learning_history(different)
        )
        self.assertNotEqual(
            (first / "predictions.csv").read_bytes(),
            (different / "predictions.csv").read_bytes(),
        )


def main():
    parser = argparse.ArgumentParser(description="Проверки реализации ЛР0")

    # Дополнительно выполнить три полных обучения после коммита.
    parser.add_argument("--runs", action="store_true")
    args = parser.parse_args()

    if args.runs:
        harness.git_info()

    TrainingChecks.enabled = args.runs
    unittest.main(argv=[sys.argv[0]], verbosity=2)


if __name__ == "__main__":
    main()