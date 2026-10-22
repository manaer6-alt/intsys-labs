# train.py — запуск одного эксперимента из командной строки.
# Загружает YAML, применяет переопределения и вызывает harness.py.
# Обучение, проверки и сохранение результатов выполняет harness.
# Один вызов команды — один запуск в отдельном каталоге runs/.

import argparse

from harness import load_config, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Запуск эксперимента ЛР0")

    # Путь к конфигурации относительно корня проекта.
    parser.add_argument("--config", default="configs/base.yaml")

    # Переопределяет seed из YAML; без аргумента остаётся значение из файла.
    parser.add_argument("--seed", type=int)

    # Название серии для metadata.json и MLflow.
    parser.add_argument("--series", default="manual")

    # Добавляет оценку на test — для итогового эксперимента.
    parser.add_argument("--eval-test", action="store_true")

    # Разрешает незакоммиченные изменения и сокращённое обучение.
    parser.add_argument("--debug", action="store_true")

    # Переопределяет число эпох только в отладочном режиме.
    parser.add_argument("--epochs", type=int)

    # Читаем аргументы, переданные при запуске команды.
    args = parser.parse_args()

    if args.epochs is not None and not args.debug:
        parser.error("--epochs разрешён только вместе с --debug")

    config = load_config(args.config)

    # Проверяем наличие аргумента: seed=0 тоже допустим.
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs

    # Harness проверит итоговую конфигурацию и выполнит эксперимент.
    result = run_experiment(
        config,
        series=args.series,
        eval_test=args.eval_test,
        debug=args.debug,
    )

    print("Результаты:", result["run_dir"])
    print("Метрики:", result["metrics"])


# При импорте файла обучение автоматически не запускается.
if __name__ == "__main__":
    main()