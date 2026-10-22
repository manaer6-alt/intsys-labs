# search.py — подбор гиперпараметров через Optuna.
#
# Читает configs/base.yaml и configs/search_space.yaml.
# Каждую предложенную конфигурацию обучает через harness.py.
# Сравнивает точность на val после последней эпохи; test не использует.
#
# Результаты: история в optuna.db, файлы обучений в runs/,
# родительская запись поиска и вложенные попытки в MLflow.
#
# --until — общее число попыток в базе, включая COMPLETE, PRUNED и FAIL.

import argparse
from copy import deepcopy

import mlflow
import optuna
import yaml

from harness import (
    ROOT, SPLIT_PATH, MANIFEST_PATH,
    file_sha256, git_info, load_config,
    run_experiment, setup_tracking,
)


def main():
    parser = argparse.ArgumentParser(description="Поиск параметров ЛР0")

    # Сколько попыток должно быть в базе после завершения программы.
    parser.add_argument("--until", type=int, default=40)
    args = parser.parse_args()

    if not 1 <= args.until <= 40:
        parser.error("--until должен быть от 1 до 40")

    base = load_config("configs/base.yaml")
    space = yaml.safe_load(
        (ROOT / "configs/search_space.yaml").read_text(encoding="utf-8")
    )

    if base["seed"] != 0 or base["train"]["epochs"] != 30:
        parser.error("Для поиска нужны seed: 0 и epochs: 30")

    expected = set(base["model"]) | (set(base["train"]) - {"epochs"})
    if not isinstance(space, dict) or set(space) != expected:
        parser.error("Неверные параметры в search_space.yaml")

    commit, _ = git_info()
    study = optuna.create_study(
        study_name="lr0_search",
        storage=f"sqlite:///{(ROOT / 'optuna.db').as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5,
        ),
    )

    # Добавляем только недостающие попытки, учитывая все состояния.
    remaining = args.until - len(study.trials)
    if remaining <= 0:
        print("Целевое число попыток уже достигнуто.")
        return

    # Защищаем продолжение поиска от изменения условий.
    conditions = {
        "commit": commit,
        "config": base,
        "space": space,
        "split_sha256": file_sha256(SPLIT_PATH),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
    }
    if study.trials and study.user_attrs.get("conditions") != conditions:
        raise ValueError("Условия поиска изменились — продолжение невозможно.")
    study.set_user_attr("conditions", conditions)

    def objective(trial):
        config = deepcopy(base)

        # Определяем раздел конфигурации и способ выбора параметра.
        for name, bounds in space.items():
            section = "model" if name in config["model"] else "train"

            if name == "batch_size":
                value = trial.suggest_categorical(name, bounds)
            else:
                integer = name in ("n_layers", "hidden")
                suggest = trial.suggest_int if integer else trial.suggest_float
                value = suggest(name, **bounds)

            config[section][name] = value

        result = run_experiment(config, series="search", trial=trial)
        return result["metrics"]["val_acc"]

    setup_tracking()

    # При продолжении используем ту же родительскую запись MLflow.
    with mlflow.start_run(
        run_id=study.user_attrs.get("mlflow_parent_run_id"),
        run_name="lr0_search",
    ) as parent:
        study.set_user_attr("mlflow_parent_run_id", parent.info.run_id)
        mlflow.set_tags({"series": "search", "commit": commit})
        mlflow.log_dict(conditions, "search_conditions.json")

        # Последовательный поиск; NaN/inf дают FAIL и не останавливают серию.
        study.optimize(
            objective, n_trials=remaining, n_jobs=1,
            catch=(FloatingPointError,),
        )

    print("Всего попыток в базе:", len(study.trials))


if __name__ == "__main__":
    main()