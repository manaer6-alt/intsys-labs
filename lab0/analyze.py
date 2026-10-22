# analyze.py — анализ сохранённых результатов ЛР0.
#
# Модели не обучает: читает файлы запусков и базу Optuna.
# Пересчитывает метрики, формирует таблицы, рисунки и сводки JSON.
#
# Этапы:
# base   — пять базовых запусков, разброс и сравнение ошибок;
# search — 40 попыток Optuna, затраты, графики и configs/best.yaml;
# final  — парное сравнение конфигураций на seed 101–105.
#
# Результаты анализа сохраняются в новые каталоги runs/analysis/.
# Исходные результаты запусков не перезаписываются.
# best.yaml создаётся на этапе search до итоговой оценки на test.

import argparse
import uuid
from copy import deepcopy

import matplotlib
import numpy as np
import optuna
import pandas as pd
import yaml

# Сохраняем рисунки без открытия отдельных окон.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness import (
    ROOT, SPLIT_PATH, file_sha256, load_config,
    read_json, save_json, recalculate_metrics,
)


def stats(values):
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def save_table(data, out, name):
    pd.DataFrame(data).to_csv(
        out / "tables" / f"{name}.csv", index=False, encoding="utf-8"
    )


def save_figure(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / "figures" / f"{name}.png", dpi=180)
    plt.close(fig)


def load_series(name, expected_count):
    # Отладочные запуски не являются результатами эксперимента.
    records = []
    split = read_json(SPLIT_PATH)
    split_hash = file_sha256(SPLIT_PATH)

    for path in sorted((ROOT / "runs").glob("*/metadata.json")):
        meta = read_json(path)
        if meta["series"] != name or meta["debug"]:
            continue

        if meta["status"] != "completed" or meta["dirty"]:
            raise ValueError(
                f"Проблемный запуск {meta['run_id']}: {meta['status']}"
            )

        run_dir = path.parent
        metrics = recalculate_metrics(run_dir)

        if (
            metrics != read_json(run_dir / "metrics.json")
            or metrics["epochs_completed"] != 30
        ):
            raise ValueError(f"Не прошла проверка метрик: {meta['run_id']}")

        if meta["split_sha256"] != split_hash:
            raise ValueError("Запуск использовал другое разбиение")

        predictions = pd.read_csv(run_dir / "predictions.csv")

        # Проверяем полноту предсказаний каждой сохранённой части.
        for part, frame in predictions.groupby("split"):
            if (
                len(frame) != len(split[part])
                or set(frame["index"]) != set(split[part])
            ):
                raise ValueError(f"Неполные предсказания: {meta['run_id']}")

        records.append({
            "meta": meta,
            "config": load_config(run_dir / "config.yaml"),
            "metrics": metrics,
            "pred": predictions,
        })

    if len(records) != expected_count:
        raise ValueError(
            f"Серия {name}: нужно {expected_count} запусков, "
            f"найдено {len(records)}"
        )

    # Внутри серии условия должны совпадать, кроме seed и параметров модели.
    for key in ("commit", "versions", "split_sha256", "manifest_sha256"):
        if any(r["meta"][key] != records[0]["meta"][key] for r in records):
            raise ValueError(f"В серии различается поле {key}")

    return records


def select_runs(records, reference, seeds):
    # Сравниваем конфигурации, временно убрав различие seed.
    selected = [
        r for r in records
        if dict(r["config"], seed=0) == dict(reference, seed=0)
    ]
    if sorted(r["config"]["seed"] for r in selected) != list(seeds):
        raise ValueError("Нужен ровно один запуск конфигурации на каждый seed")
    return {r["config"]["seed"]: r for r in selected}


def error_counts(first, second, part):
    # Сопоставляем ответы по исходному индексу изображения.
    frames = [
        r["pred"].loc[r["pred"]["split"] == part]
        .set_index("index").sort_index()
        for r in (first, second)
    ]
    a, b = frames

    if not a.index.equals(b.index) or not a["y_true"].equals(b["y_true"]):
        raise ValueError("Нельзя сравнивать разные изображения или метки")

    correct_a = a["y_true"] == a["y_pred"]
    correct_b = b["y_true"] == b["y_pred"]

    fixed = int((~correct_a & correct_b).sum())
    spoiled = int((correct_a & ~correct_b).sum())
    return fixed, spoiled


def baseline_runs():
    records = load_series("base_seeds", 5)
    if any(set(r["pred"]["split"]) != {"val"} for r in records):
        raise ValueError("В базовой серии обнаружена оценка на test")
    return select_runs(records, load_config("configs/base.yaml"), range(1, 6))


def analyze_base(out):
    runs = baseline_runs()
    rows = [
        {
            "seed": seed,
            "run_id": run["meta"]["run_id"],
            "val_acc": run["metrics"]["val_acc"],
            "train_seconds": run["metrics"]["train_seconds"],
        }
        for seed, run in sorted(runs.items())
    ]
    save_table(rows, out, "base_runs")

    fixed, spoiled = error_counts(runs[1], runs[2], "val")
    save_json(out / "summary.json", {
        "val_acc": stats([row["val_acc"] for row in rows]),
        "train_seconds": stats([row["train_seconds"] for row in rows]),
        "seed_2_fixed_vs_seed_1": fixed,
        "seed_2_spoiled_vs_seed_1": spoiled,
    })


def load_search():
    study = optuna.load_study(
        study_name="lr0_search",
        storage=f"sqlite:///{(ROOT / 'optuna.db').as_posix()}",
    )
    if len(study.trials) != 40 or any(
        not trial.state.is_finished() for trial in study.trials
    ):
        raise ValueError("Для итогового анализа нужны 40 законченных попыток")
    return study


def best_config(study):
    config = deepcopy(study.user_attrs["conditions"]["config"])
    if config != load_config("configs/base.yaml"):
        raise ValueError("Базовая конфигурация отличается от условий поиска")

    for name, value in study.best_params.items():
        section = "model" if name in config["model"] else "train"
        config[section][name] = value
    return config


def analyze_search(out):
    study = load_search()
    costs = []

    for trial in study.trials:
        state = trial.state.name
        run_id = trial.user_attrs.get("run_id")
        path = ROOT / "runs" / run_id / "history.csv" if run_id else None
        history = (
            pd.read_csv(path) if path and path.exists()
            else pd.DataFrame(columns=["epoch_seconds"])
        )
        epochs = len(history)
        seconds = float(history["epoch_seconds"].sum())

        # COMPLETE и PRUNED должны иметь сохранённые предсказания.
        if state != "FAIL":
            if path is None:
                raise ValueError(f"Нет файлов попытки {trial.number}")
            metrics = recalculate_metrics(path.parent)
            if metrics != read_json(path.parent / "metrics.json"):
                raise ValueError(f"Не совпали метрики trial {trial.number}")
            if not np.isclose(
                metrics["val_acc"], trial.value, rtol=0, atol=1e-12
            ):
                raise ValueError(f"Не совпала точность trial {trial.number}")
            if state == "COMPLETE" and epochs != 30:
                raise ValueError("COMPLETE должен содержать 30 эпох")

        saved = 30 - epochs if state == "PRUNED" else 0
        costs.append({
            "number": trial.number,
            "epochs_completed": epochs,
            "train_seconds": seconds,
            "pruner_saved_epochs": saved,
            # Оценка по средней длительности эпох именно этой попытки.
            "pruner_saved_seconds_estimate": seconds / epochs * saved
            if epochs else 0.0,
        })

    # В таблице остаются все попытки, включая FAIL и PRUNED.
    trials = study.trials_dataframe().merge(
        pd.DataFrame(costs), on="number", validate="one_to_one"
    )
    trials["value"] = pd.to_numeric(trials["value"], errors="coerce")
    save_table(trials, out, "trials")

    complete = trials["state"] == "COMPLETE"
    save_json(out / "summary.json", {
        "trial_states": {
            state: int((trials["state"] == state).sum())
            for state in ("COMPLETE", "PRUNED", "FAIL")
        },
        "planned_epochs": 1200,
        "completed_epochs": int(trials["epochs_completed"].sum()),
        "pruner_saved_epochs": int(trials["pruner_saved_epochs"].sum()),
        "recorded_seconds": float(trials["train_seconds"].sum()),
        "pruner_saved_seconds_estimate": float(
            trials["pruner_saved_seconds_estimate"].sum()
        ),
        "time_estimate_method": (
            "Невыполненные эпохи PRUNED умножены на среднее время "
            "сохранённых эпох соответствующей попытки"
        ),
        "best_trial": study.best_trial.number if complete.any() else None,
        "best_val_acc": study.best_value if complete.any() else None,
    })

    # График результатов. Для PRUNED показана последняя выполненная эпоха.
    fig, ax = plt.subplots(figsize=(7, 4))
    for state, marker in (("COMPLETE", "o"), ("PRUNED", "x")):
        part = trials[trials["state"] == state]
        ax.scatter(part["number"], part["value"], marker=marker, label=state)
    ax.step(
        trials["number"],
        trials["value"].where(complete).cummax().ffill(),
        where="post", label="Лучший COMPLETE",
    )
    ax.set(xlabel="Номер попытки", ylabel="Точность на val")
    ax.legend()
    save_figure(fig, out, "search_history")

    # fANOVA — функциональный дисперсионный анализ.
    # Фиксируем seed оценки важности для повторяемости.
    importance = {}
    if trials.loc[complete, "value"].nunique() > 1:
        importance = optuna.importance.get_param_importances(
            study,
            evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0),
        )
    save_table(
        pd.DataFrame(importance.items(), columns=["parameter", "importance"]),
        out, "importance",
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    if importance:
        ax.barh(list(importance), list(importance.values()))
        ax.invert_yaxis()
        ax.set_xlabel("Относительная важность")
    else:
        ax.text(
            0.5, 0.5, "Недостаточно различающихся\nрезультатов COMPLETE",
            ha="center", va="center", transform=ax.transAxes,
        )
    save_figure(fig, out, "parameter_importance")

    if not complete.any():
        print("Нет COMPLETE: лучшую конфигурацию выбрать невозможно.")
        return

    # Существующую отличающуюся конфигурацию автоматически не заменяем.
    config = best_config(study)
    path = ROOT / "configs" / "best.yaml"
    if path.exists():
        if load_config(path) != config:
            raise ValueError("Существующий best.yaml отличается от результата")
    else:
        path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def analyze_final(out):
    study = load_search()
    best = load_config("configs/best.yaml")
    if best != best_config(study):
        raise ValueError("best.yaml не соответствует лучшей попытке поиска")

    records = load_series("final", 10)
    if any(set(r["pred"]["split"]) != {"val", "test"} for r in records):
        raise ValueError("Итоговые запуски должны выполняться с --eval-test")

    base = select_runs(
        records, load_config("configs/base.yaml"), range(101, 106)
    )
    selected = select_runs(records, best, range(101, 106))
    rows = []

    for seed in range(101, 106):
        a, b = base[seed], selected[seed]
        fixed, spoiled = error_counts(a, b, "test")
        # Проверяем сопоставимость изображений также на val.
        error_counts(a, b, "val")
        am, bm = a["metrics"], b["metrics"]

        rows.append({
            "seed": seed,
            "base_run_id": a["meta"]["run_id"],
            "best_run_id": b["meta"]["run_id"],
            "base_val_acc": am["val_acc"],
            "best_val_acc": bm["val_acc"],
            "base_test_acc": am["test_acc"],
            "best_test_acc": bm["test_acc"],
            # Разности выражены в процентных пунктах.
            "delta_val_pp": 100 * (bm["val_acc"] - am["val_acc"]),
            "delta_test_pp": 100 * (fixed - spoiled) / 360,
            "fixed_test": fixed,
            "spoiled_test": spoiled,
        })

    result = pd.DataFrame(rows)
    save_table(result, out, "final_pairs")

    baseline = baseline_runs()
    scores = [run["metrics"]["val_acc"] for run in baseline.values()]

    # Сравниваем целые количества ответов, избегая погрешности дробей.
    spread_count = round((max(scores) - min(scores)) * 360)
    delta_count = result["fixed_test"] - result["spoiled_test"]

    if (delta_count > spread_count).all():
        verdict = "Все пять улучшений превышают разброс базовой серии."
    elif (delta_count > 0).any() and (delta_count < 0).any():
        verdict = "Знаки разностей различаются: устойчивое улучшение не показано."
    else:
        verdict = "Критерий устойчивого улучшения из методички не выполнен."

    save_json(out / "summary.json", {
        "delta_val_pp": stats(result["delta_val_pp"]),
        "delta_test_pp": stats(result["delta_test_pp"]),
        "baseline_spread_pp": 100 * spread_count / 360,
        "fixed_test_total": int(result["fixed_test"].sum()),
        "spoiled_test_total": int(result["spoiled_test"].sum()),
        "test_evaluations": 1800,
        "unique_test_images": 360,
        "search_best_val_acc": study.best_value,
        "new_seeds_best_val_mean": float(result["best_val_acc"].mean()),
        "new_seeds_best_test_mean": float(result["best_test_acc"].mean()),
        "verdict": verdict,
        "limitation": (
            "Использован описательный критерий методички, "
            "а не проверка статистической значимости"
        ),
    })


def main():
    parser = argparse.ArgumentParser(description="Анализ результатов ЛР0")

    # Выполняем только выбранный этап анализа.
    parser.add_argument(
        "--stage", choices=("base", "search", "final"), required=True
    )
    args = parser.parse_args()

    out = ROOT / "runs" / "analysis" / args.stage / uuid.uuid4().hex
    (out / "tables").mkdir(parents=True)
    (out / "figures").mkdir()

    {
        "base": analyze_base,
        "search": analyze_search,
        "final": analyze_final,
    }[args.stage](out)

    print("Результаты анализа:", out)


if __name__ == "__main__":
    main()