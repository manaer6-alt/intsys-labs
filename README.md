# Интеллектуальные системы

Студент: **Грязнов Александр Евгеньевич**.  
Группа: **8ВМ63**.  
Университет: **Томский политехнический университет**.  
Преподаватель: **Григорьев Дмитрий Сергеевич, старший преподаватель**.

Задания и правила курса: https://github.com/dm780009/intsys

## Лабораторная работа 0

Тема: «Воспроизводимый эксперимент». Проверяется, превышает ли улучшение после подбора гиперпараметров случайный разброс точности при смене seed.

Работа выполнена: подготовлены данные, каркас обучения, проверки, базовая серия, поиск Optuna из 40 попыток, итоговое сравнение на пяти новых seed и отчёт.

## Среда и установка

Фактическая среда экспериментов: Windows, Python 3.14.0, PyTorch 2.14.1+cpu, NumPy 2.5.4, scikit-learn 1.9.1, Optuna 5.0.0, MLflow 3.17.2. Обучение выполняется на CPU в одном потоке. Все зависимости зафиксированы в `requirements.txt`.

Команды ниже записаны для PowerShell. Все команды проекта выполняются из корня `intsys-labs`; активация окружения не требуется.

Для нового окружения:

```powershell
py -3.14 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Для итогового обучения рабочее дерево Git должно быть чистым. Каталоги запусков, виртуальное окружение и базы исключены через `.gitignore`.

## Данные и файлы

Используются все 1797 изображений цифр из `sklearn.datasets.load_digits`. Изображение 8 × 8 даёт 64 признака; метки — 0–9. Признаки делятся на 16. Исходный датасет отдельно скачивать не требуется.

Стратифицированное разбиение при `split_seed=0`: train — 1077, val — 360, test — 360. Исходные индексы находятся в `data/split.json`, источник, версия и SHA-256 — в `data/manifest.json`. При обучении проверяются эти же индексы и хеш.

| Файл или каталог | Назначение |
|---|---|
| `lab0/prepare_data.py` | Подготовка и проверка разбиения, манифест |
| `lab0/harness.py` | Общее обучение, метрики, файлы запусков, MLflow |
| `lab0/train.py` | Один запуск из командной строки |
| `lab0/search.py` | Продолжаемый поиск Optuna |
| `lab0/analyze.py` | Пересчёт результатов, таблицы и рисунки |
| `lab0/verify.py` | Проверки расчётов, конфигурации, Git и повторяемости |
| `configs/` | Базовая и выбранная конфигурации, пространство поиска |
| `forecasts/lab0.md` | Прогноз, зафиксированный до основных экспериментов |
| `lab0/reports/` | Отчёт Word/PDF, оформление и подготовка к защите |
| `lab0/figures/`, `lab0/tables/` | Рисунки и таблицы отчёта, CSV журнала MLflow |

## Проверка и повторный анализ сохранённых результатов

Быстрые проверки не выполняют обучение:

```powershell
& ".\.venv\Scripts\python.exe" lab0\verify.py
```

Дополнительная проверка `verify.py --runs` запускает три обучения в серии `verify`. Она уже пройдена при выполнении работы; для просмотра результатов повторять её не требуется.

Для пересчёта таблиц из артефактов распакуйте предоставленный архив в отдельный чистый клон проекта: в корне должны появиться `runs/`, `optuna.db`, `mlflow.db` и `mlartifacts/`. Установите зафиксированное окружение и выполните:

```powershell
& ".\.venv\Scripts\python.exe" lab0\analyze.py --stage base
& ".\.venv\Scripts\python.exe" lab0\analyze.py --stage search
& ".\.venv\Scripts\python.exe" lab0\analyze.py --stage final
```

Новые результаты анализа появятся в `runs/analysis/<stage>/<id>/`. Обучение эти команды не выполняют. Поиск анализируется по 40 завершённым по состоянию попыткам, включая PRUNED и FAIL. Итоговое сравнение использует ровно 10 плановых запусков серии `final`.

В базе MLflow исходные адреса артефактов могут содержать абсолютный путь компьютера автора. Файлы также переданы в `mlartifacts/`; журнал доступен без интерфейса MLflow в `lab0/tables/mlflow_runs.csv`. Для пересчёта таблиц используются `runs/` и `optuna.db`.

## Полное повторение экспериментов в отдельном клоне

Этот раздел предназначен для нового воспроизведения. Выполняйте его в отдельном клоне без восстановленных баз и каталогов запусков. Повторное выполнение серий в исходной рабочей папке создаст дубликаты.

Базовая серия и поиск выполнены на коммите `c293baa0f19bcc584e82f9e43c973fd6fcc6c227`. Состояние до появления `best.yaml` удобно использовать как начало воспроизведения:

```powershell
Set-Location ..
git clone git@github.com:manaer6-alt/intsys-labs.git intsys-labs-repro
Set-Location .\intsys-labs-repro
git switch --detach c293baa0f19bcc584e82f9e43c973fd6fcc6c227
```

Установите окружение по разделу выше. Подготовьте данные и выполните проверки:

```powershell
& ".\.venv\Scripts\python.exe" lab0\prepare_data.py
& ".\.venv\Scripts\python.exe" lab0\verify.py --runs
```

`verify.py --runs` включает два запуска базовой модели с seed 0 и один с seed 1 в отдельных процессах. Сравниваются предсказания и показатели обучения; фактически измеренное время эпох может различаться.

Базовая серия: пять запусков, только val.

```powershell
foreach ($labSeed in 1..5) {
    & ".\.venv\Scripts\python.exe" lab0\train.py --config configs\base.yaml --seed $labSeed --series base_seeds
    if ($LASTEXITCODE -ne 0) { throw "Base run failed: seed=$labSeed" }
}
& ".\.venv\Scripts\python.exe" lab0\analyze.py --stage base
```

Поиск: сначала 10 попыток, затем продолжение до 40. `--until` задаёт общее число попыток в SQLite, включая все состояния.

```powershell
& ".\.venv\Scripts\python.exe" lab0\search.py --until 10
& ".\.venv\Scripts\python.exe" lab0\search.py --until 40
& ".\.venv\Scripts\python.exe" lab0\analyze.py --stage search
```

Созданный `configs/best.yaml` следует зафиксировать до оценки test:

```powershell
git add configs/best.yaml
git commit -m "Freeze selected lab0 configuration before test evaluation"
```

Итоговая серия: две конфигурации с seed 101–105, всего 10 запусков. Test используется только на этом этапе.

```powershell
foreach ($labSeed in 101..105) {
    foreach ($labConfig in "configs/base.yaml", "configs/best.yaml") {
        & ".\.venv\Scripts\python.exe" lab0\train.py --config $labConfig --seed $labSeed --series final --eval-test
        if ($LASTEXITCODE -ne 0) { throw "Final run failed: seed=$labSeed, config=$labConfig" }
    }
}
& ".\.venv\Scripts\python.exe" lab0\analyze.py --stage final
```

Исходные итоговые запуски выполнены на коммите `cefebfd65bafb722f6dd1ebea5c975c4f87487ba`. При воспроизведении новый коммит с выбранной конфигурацией получит свой хеш. Повторение последовательности поиска требует тех же версий и такого же разделения на вызовы до 10 и до 40: создание sampler заново с тем же seed не восстанавливает состояние его генератора.

## Отчёт и проверка результатов

- [Отчёт PDF](<lab0/reports/8ВМ63 Грязнов ЛР 0.pdf>)
- [Отчёт Word](<lab0/reports/8ВМ63 Грязнов ЛР 0.docx>)
- [Подготовка к защите](lab0/reports/Подготовка_к_защите_ЛР0.md)
- [Первоначальный прогноз](forecasts/lab0.md)
- [Базовая серия](lab0/tables/base_runs.csv)
- [Все попытки поиска](lab0/tables/trials.csv)
- [Пять итоговых пар](lab0/tables/final_pairs.csv)
- [Журнал MLflow](lab0/tables/mlflow_runs.csv)

Готовые рисунки, таблицы и отчёт дополнительно зарегистрированы в MLflow записью `2bb10baf079a46358f18e75eed572088`, серия `report_analysis`; обучение в этой записи не выполнялось.

В отчёте приведены только фактически полученные результаты. 1800 тестовых сравнений — это 360 уникальных изображений при пяти seed. Экономия времени прунера оценочная; устойчивость улучшения определена описательным критерием методички.

Случайно начатый повтор после 10 плановых итоговых запусков прерван и сохранён в `runs/excluded/0786bca436d84925b79f374a987f6be2` вместе с причиной исключения. Его результаты не входят в итоговую таблицу.

Архив с `runs/`, `optuna.db`, `mlflow.db` и `mlartifacts/` передаётся отдельно. Исходные данные `load_digits` и виртуальное окружение в архив не входят.
